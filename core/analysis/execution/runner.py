"""
Pipeline Execution Engine for Analysis Workflows.

Provides the core pipeline executor with:
- DAG-based task execution with dependency resolution
- Parallel execution of independent tasks
- Error handling with configurable retry policies
- Progress tracking and callbacks
- Integration with checkpoint/recovery system
- Execution context with resource management

This module implements the runtime for assembled analysis pipelines.
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Optional numpy import - only used if array handling is needed
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Type variable for task results
T = TypeVar("T")


class TaskStatus(Enum):
    """Status of a pipeline task."""

    PENDING = "pending"  # Not yet started
    QUEUED = "queued"  # Waiting for dependencies
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Cancelled by user/system
    SKIPPED = "skipped"  # Skipped due to dependency failure


class RetryPolicy(Enum):
    """Retry policies for failed tasks."""

    NONE = "none"  # No retries
    IMMEDIATE = "immediate"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Retry with linear backoff


class ExecutionMode(Enum):
    """Execution modes for the pipeline."""

    SEQUENTIAL = "sequential"  # Execute tasks one at a time
    PARALLEL = "parallel"  # Execute independent tasks in parallel
    DISTRIBUTED = "distributed"  # Use distributed backend (Dask/Ray)


@dataclass
class RetryConfig:
    """
    Configuration for task retry behavior.

    Attributes:
        policy: Retry policy to use
        max_retries: Maximum number of retry attempts
        base_delay_seconds: Base delay between retries
        max_delay_seconds: Maximum delay between retries
        retry_on_exceptions: Exception types to retry on (None = all)
    """

    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    retry_on_exceptions: Optional[Set[type]] = None

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        if self.policy == RetryPolicy.NONE:
            return 0.0
        elif self.policy == RetryPolicy.IMMEDIATE:
            return 0.0
        elif self.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.base_delay_seconds * attempt
        else:  # EXPONENTIAL_BACKOFF
            delay = self.base_delay_seconds * (2 ** (attempt - 1))

        return min(delay, self.max_delay_seconds)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if task should be retried for given exception."""
        if self.policy == RetryPolicy.NONE:
            return False
        if attempt >= self.max_retries:
            return False
        if self.retry_on_exceptions is None:
            return True
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)


@dataclass
class ExecutionConfig:
    """
    Configuration for pipeline execution.

    Attributes:
        mode: Execution mode (sequential, parallel, distributed)
        max_workers: Maximum parallel workers
        timeout_seconds: Overall execution timeout (None = no timeout)
        task_timeout_seconds: Per-task timeout (None = no timeout)
        retry_config: Configuration for retry behavior
        checkpoint_enabled: Whether to save checkpoints
        checkpoint_interval: Checkpoint after N completed tasks
        fail_fast: Stop execution on first failure
        continue_on_skip: Continue if dependencies are skipped
    """

    mode: ExecutionMode = ExecutionMode.PARALLEL
    max_workers: int = 4
    timeout_seconds: Optional[float] = None
    task_timeout_seconds: Optional[float] = None
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 5
    fail_fast: bool = False
    continue_on_skip: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be >= 1, got {self.checkpoint_interval}"
            )


@dataclass
class TaskResult:
    """
    Result from executing a pipeline task.

    Attributes:
        task_id: Task identifier
        status: Final task status
        result: Task output (if successful)
        error: Error message (if failed)
        exception: Exception object (if failed)
        start_time: When task started
        end_time: When task completed
        retry_count: Number of retry attempts made
        metadata: Additional metadata
    """

    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    exception: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task execution duration."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """Whether task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result_type": type(self.result).__name__ if self.result is not None else None,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionContext:
    """
    Context passed to task executors.

    Provides access to execution environment, input data,
    and utilities for the task being executed.

    Attributes:
        task_id: Current task identifier
        execution_id: Unique execution run identifier
        input_data: Input data for the task
        parameters: Task-specific parameters
        shared_state: Shared state accessible by all tasks
        working_dir: Working directory for intermediate files
        resources: Available compute resources
        checkpoint_manager: For saving/loading checkpoints
        progress_callback: Callback for progress updates
    """

    task_id: str
    execution_id: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    working_dir: Optional[Path] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    checkpoint_manager: Optional[Any] = None  # Will be CheckpointManager
    progress_callback: Optional[Callable[[float, str], None]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get input data by key."""
        return self.input_data.get(key, default)

    def set_shared(self, key: str, value: Any) -> None:
        """Set shared state value (thread-safe)."""
        with self._lock:
            self.shared_state[key] = value

    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get shared state value (thread-safe)."""
        with self._lock:
            return self.shared_state.get(key, default)

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report task progress (0.0 to 1.0)."""
        if self.progress_callback is not None:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)


class TaskExecutor(ABC):
    """
    Abstract base class for task executors.

    Implement this class to create custom task types
    that can be executed within pipelines.
    """

    @abstractmethod
    def execute(self, context: ExecutionContext) -> Any:
        """
        Execute the task.

        Args:
            context: Execution context with inputs and utilities

        Returns:
            Task result data

        Raises:
            Exception: If task execution fails
        """
        pass

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique task identifier."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """List of task IDs this task depends on."""
        return []

    @property
    def outputs(self) -> List[str]:
        """List of output keys this task produces."""
        return []


@dataclass
class PipelineTask:
    """
    Represents a task within a pipeline.

    Attributes:
        task_id: Unique task identifier
        executor: Task executor instance or callable
        dependencies: Task IDs that must complete before this task
        parameters: Task-specific parameters
        timeout_seconds: Task-specific timeout override
        retry_config: Task-specific retry configuration override
        priority: Execution priority (higher = earlier)
        metadata: Additional task metadata
    """

    task_id: str
    executor: Union[TaskExecutor, Callable[["ExecutionContext"], Any]]
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    retry_config: Optional[RetryConfig] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, context: ExecutionContext) -> Any:
        """Execute the task with given context."""
        if isinstance(self.executor, TaskExecutor):
            return self.executor.execute(context)
        else:
            return self.executor(context)


@dataclass
class ExecutionProgress:
    """
    Progress information for pipeline execution.

    Attributes:
        total_tasks: Total number of tasks
        completed_tasks: Number of completed tasks
        failed_tasks: Number of failed tasks
        running_tasks: Number of currently running tasks
        pending_tasks: Number of pending tasks
        current_phase: Current execution phase
        start_time: Execution start time
        elapsed_seconds: Elapsed time
        estimated_remaining_seconds: Estimated time remaining
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    current_phase: str = ""
    start_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    @property
    def progress_fraction(self) -> float:
        """Calculate progress as fraction (0.0 to 1.0)."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks

    @property
    def is_complete(self) -> bool:
        """Whether execution is complete."""
        return (self.completed_tasks + self.failed_tasks) >= self.total_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "running_tasks": self.running_tasks,
            "pending_tasks": self.pending_tasks,
            "current_phase": self.current_phase,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "progress_percent": self.progress_fraction * 100,
        }


@dataclass
class ExecutionResult:
    """
    Complete result of pipeline execution.

    Attributes:
        execution_id: Unique execution identifier
        status: Overall execution status
        task_results: Results from each task
        start_time: When execution started
        end_time: When execution completed
        total_duration_seconds: Total execution time
        checkpoints_saved: Number of checkpoints created
        metadata: Additional execution metadata
    """

    execution_id: str
    status: TaskStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    checkpoints_saved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether execution completed successfully."""
        return self.status == TaskStatus.COMPLETED

    @property
    def failed_tasks(self) -> List[TaskResult]:
        """Get list of failed task results."""
        return [r for r in self.task_results.values() if r.status == TaskStatus.FAILED]

    @property
    def successful_tasks(self) -> List[TaskResult]:
        """Get list of successful task results."""
        return [r for r in self.task_results.values() if r.status == TaskStatus.COMPLETED]

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for specific task."""
        return self.task_results.get(task_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "task_results": {k: v.to_dict() for k, v in self.task_results.items()},
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "checkpoints_saved": self.checkpoints_saved,
            "successful_tasks": len(self.successful_tasks),
            "failed_tasks_count": len(self.failed_tasks),
            "metadata": self.metadata,
        }


class Pipeline:
    """
    Represents an analysis pipeline as a DAG of tasks.

    Provides methods for:
    - Adding and validating tasks
    - Resolving execution order
    - Detecting cycles
    - Visualizing the pipeline structure
    """

    def __init__(self, pipeline_id: str, name: str = "", description: str = ""):
        """
        Initialize a pipeline.

        Args:
            pipeline_id: Unique pipeline identifier
            name: Human-readable name
            description: Pipeline description
        """
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.tasks: Dict[str, PipelineTask] = {}
        self._execution_order: Optional[List[str]] = None
        self._lock = threading.Lock()

    def add_task(self, task: PipelineTask) -> "Pipeline":
        """
        Add a task to the pipeline.

        Args:
            task: Task to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If task ID already exists
        """
        with self._lock:
            if task.task_id in self.tasks:
                raise ValueError(f"Task {task.task_id} already exists in pipeline")
            self.tasks[task.task_id] = task
            self._execution_order = None  # Invalidate cached order
        return self

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the pipeline.

        Args:
            task_id: ID of task to remove

        Returns:
            True if task was removed, False if not found
        """
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._execution_order = None
                return True
            return False

    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the pipeline structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        # Check for empty pipeline
        if not self.tasks:
            errors.append("Pipeline has no tasks")
            return False, errors

        # Check for missing dependencies
        all_task_ids = set(self.tasks.keys())
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in all_task_ids:
                    errors.append(
                        f"Task {task_id} has missing dependency: {dep_id}"
                    )

        # Check for cycles
        has_cycle, cycle_path = self._detect_cycle()
        if has_cycle:
            errors.append(f"Pipeline contains cycle: {' -> '.join(cycle_path)}")

        return len(errors) == 0, errors

    def _detect_cycle(self) -> Tuple[bool, List[str]]:
        """
        Detect cycles in the pipeline graph.

        Returns:
            Tuple of (has_cycle, cycle_path)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors: Dict[str, int] = {task_id: WHITE for task_id in self.tasks}
        path: List[str] = []

        def visit(task_id: str) -> bool:
            if colors[task_id] == GRAY:
                # Found cycle
                cycle_start = path.index(task_id)
                return True
            if colors[task_id] == BLACK:
                return False

            colors[task_id] = GRAY
            path.append(task_id)

            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in colors and visit(dep_id):
                    return True

            path.pop()
            colors[task_id] = BLACK
            return False

        for task_id in self.tasks:
            if colors[task_id] == WHITE:
                if visit(task_id):
                    return True, path

        return False, []

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        Returns:
            List of task IDs in execution order

        Raises:
            ValueError: If pipeline is invalid
        """
        with self._lock:
            if self._execution_order is not None:
                return self._execution_order.copy()

            is_valid, errors = self.validate()
            if not is_valid:
                raise ValueError(f"Invalid pipeline: {'; '.join(errors)}")

            # Kahn's algorithm for topological sort
            in_degree: Dict[str, int] = {task_id: 0 for task_id in self.tasks}

            for task in self.tasks.values():
                for dep_id in task.dependencies:
                    if dep_id in in_degree:
                        in_degree[task.task_id] += 1

            # Queue tasks with no dependencies, sorted by priority
            queue: List[str] = sorted(
                [tid for tid, deg in in_degree.items() if deg == 0],
                key=lambda tid: -self.tasks[tid].priority,
            )
            order: List[str] = []

            while queue:
                # Take highest priority task
                task_id = queue.pop(0)
                order.append(task_id)

                # Reduce in-degree of dependent tasks
                for other_id, other_task in self.tasks.items():
                    if task_id in other_task.dependencies:
                        in_degree[other_id] -= 1
                        if in_degree[other_id] == 0:
                            # Insert in priority order
                            priority = other_task.priority
                            insert_idx = len(queue)
                            for i, qid in enumerate(queue):
                                if self.tasks[qid].priority < priority:
                                    insert_idx = i
                                    break
                            queue.insert(insert_idx, other_id)

            self._execution_order = order
            return order.copy()

    def get_independent_tasks(self, completed: Set[str]) -> List[str]:
        """
        Get tasks that can run in parallel given completed tasks.

        Args:
            completed: Set of completed task IDs

        Returns:
            List of task IDs ready to execute
        """
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in completed:
                continue
            if all(dep in completed for dep in task.dependencies):
                ready.append(task_id)
        return sorted(ready, key=lambda tid: -self.tasks[tid].priority)

    def __len__(self) -> int:
        """Number of tasks in pipeline."""
        return len(self.tasks)


class PipelineRunner:
    """
    Executes analysis pipelines with configurable execution modes.

    Features:
    - Sequential and parallel execution
    - Automatic dependency resolution
    - Retry handling with backoff
    - Progress tracking and callbacks
    - Checkpoint integration
    - Timeout handling
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        checkpoint_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ):
        """
        Initialize the pipeline runner.

        Args:
            config: Execution configuration
            checkpoint_manager: Manager for saving/loading checkpoints
            progress_callback: Callback for progress updates
        """
        self.config = config or ExecutionConfig()
        self.checkpoint_manager = checkpoint_manager
        self.progress_callback = progress_callback
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def execute(
        self,
        pipeline: Pipeline,
        initial_inputs: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: bool = False,
    ) -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            initial_inputs: Initial input data for tasks
            resume_from_checkpoint: Whether to resume from last checkpoint

        Returns:
            ExecutionResult with all task results
        """
        execution_id = self._generate_execution_id(pipeline)
        logger.info(f"Starting pipeline execution: {execution_id}")

        # Validate pipeline
        is_valid, errors = pipeline.validate()
        if not is_valid:
            return ExecutionResult(
                execution_id=execution_id,
                status=TaskStatus.FAILED,
                metadata={"validation_errors": errors},
            )

        # Initialize execution state
        start_time = datetime.now(timezone.utc)
        task_results: Dict[str, TaskResult] = {}
        shared_state: Dict[str, Any] = {}
        completed: Set[str] = set()
        checkpoints_saved = 0

        # Resume from checkpoint if requested
        if resume_from_checkpoint and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_latest(execution_id)
            if checkpoint:
                task_results = checkpoint.get("task_results", {})
                shared_state = checkpoint.get("shared_state", {})
                completed = set(checkpoint.get("completed", []))
                logger.info(f"Resumed from checkpoint with {len(completed)} completed tasks")

        # Get execution order
        try:
            execution_order = pipeline.get_execution_order()
        except ValueError as e:
            return ExecutionResult(
                execution_id=execution_id,
                status=TaskStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                metadata={"error": str(e)},
            )

        # Initialize progress
        progress = ExecutionProgress(
            total_tasks=len(pipeline),
            completed_tasks=len(completed),
            pending_tasks=len(pipeline) - len(completed),
            start_time=start_time,
        )
        self._report_progress(progress)

        # Execute based on mode
        if self.config.mode == ExecutionMode.SEQUENTIAL:
            task_results, completed = self._execute_sequential(
                pipeline,
                execution_id,
                initial_inputs or {},
                shared_state,
                task_results,
                completed,
                progress,
            )
        else:
            task_results, completed = self._execute_parallel(
                pipeline,
                execution_id,
                initial_inputs or {},
                shared_state,
                task_results,
                completed,
                progress,
            )

        # Determine final status
        end_time = datetime.now(timezone.utc)
        if self._cancelled.is_set():
            final_status = TaskStatus.CANCELLED
        elif all(
            task_results.get(tid, TaskResult(tid, TaskStatus.PENDING)).success
            for tid in pipeline.tasks
        ):
            final_status = TaskStatus.COMPLETED
        else:
            final_status = TaskStatus.FAILED

        result = ExecutionResult(
            execution_id=execution_id,
            status=final_status,
            task_results=task_results,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=(end_time - start_time).total_seconds(),
            checkpoints_saved=checkpoints_saved,
        )

        logger.info(
            f"Pipeline execution completed: {execution_id} - {final_status.value}"
        )
        return result

    def _execute_sequential(
        self,
        pipeline: Pipeline,
        execution_id: str,
        initial_inputs: Dict[str, Any],
        shared_state: Dict[str, Any],
        task_results: Dict[str, TaskResult],
        completed: Set[str],
        progress: ExecutionProgress,
    ) -> Tuple[Dict[str, TaskResult], Set[str]]:
        """Execute tasks sequentially."""
        execution_order = pipeline.get_execution_order()

        for task_id in execution_order:
            if self._cancelled.is_set():
                break
            if task_id in completed:
                continue

            task = pipeline.get_task(task_id)
            if task is None:
                continue

            # Check dependencies
            deps_ok, skip_reason = self._check_dependencies(
                task, task_results, completed
            )
            if not deps_ok:
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.SKIPPED,
                    error=skip_reason,
                    metadata={"skip_reason": skip_reason},
                )
                task_results[task_id] = result
                if self.config.fail_fast and not self.config.continue_on_skip:
                    break
                continue

            # Prepare input data from dependencies
            input_data = self._prepare_inputs(
                task, task_results, initial_inputs
            )

            # Execute task
            result = self._execute_task(
                task,
                execution_id,
                input_data,
                shared_state,
            )
            task_results[task_id] = result

            if result.success:
                completed.add(task_id)
                progress.completed_tasks += 1
            else:
                progress.failed_tasks += 1
                if self.config.fail_fast:
                    break

            progress.pending_tasks -= 1
            self._report_progress(progress)

            # Checkpoint if needed
            if (
                self.config.checkpoint_enabled
                and self.checkpoint_manager
                and len(completed) % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(
                    execution_id, task_results, shared_state, completed
                )

        return task_results, completed

    def _execute_parallel(
        self,
        pipeline: Pipeline,
        execution_id: str,
        initial_inputs: Dict[str, Any],
        shared_state: Dict[str, Any],
        task_results: Dict[str, TaskResult],
        completed: Set[str],
        progress: ExecutionProgress,
    ) -> Tuple[Dict[str, TaskResult], Set[str]]:
        """Execute tasks in parallel where possible."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            running: Dict[str, concurrent.futures.Future] = {}

            while True:
                if self._cancelled.is_set():
                    # Cancel all running futures
                    for future in running.values():
                        future.cancel()
                    break

                # Get tasks ready to execute
                ready = pipeline.get_independent_tasks(completed)
                ready = [tid for tid in ready if tid not in running]

                # Submit ready tasks
                for task_id in ready:
                    task = pipeline.get_task(task_id)
                    if task is None:
                        continue

                    # Check dependencies
                    deps_ok, skip_reason = self._check_dependencies(
                        task, task_results, completed
                    )
                    if not deps_ok:
                        result = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.SKIPPED,
                            error=skip_reason,
                        )
                        task_results[task_id] = result
                        completed.add(task_id)
                        progress.completed_tasks += 1
                        progress.pending_tasks -= 1
                        continue

                    # Prepare inputs and submit
                    input_data = self._prepare_inputs(
                        task, task_results, initial_inputs
                    )
                    future = executor.submit(
                        self._execute_task,
                        task,
                        execution_id,
                        input_data,
                        shared_state,
                    )
                    running[task_id] = future
                    progress.running_tasks += 1

                # Wait for any task to complete
                if running:
                    done, _ = concurrent.futures.wait(
                        running.values(),
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    # Process completed tasks
                    completed_ids = []
                    for task_id, future in running.items():
                        if future.done():
                            try:
                                result = future.result()
                            except Exception as e:
                                result = TaskResult(
                                    task_id=task_id,
                                    status=TaskStatus.FAILED,
                                    error=str(e),
                                    exception=e,
                                )

                            task_results[task_id] = result
                            completed.add(task_id)
                            completed_ids.append(task_id)

                            if result.success:
                                progress.completed_tasks += 1
                            else:
                                progress.failed_tasks += 1
                                if self.config.fail_fast:
                                    self._cancelled.set()

                            progress.running_tasks -= 1
                            progress.pending_tasks -= 1

                    for task_id in completed_ids:
                        del running[task_id]

                    self._report_progress(progress)

                # Check if we're done
                if not running and not ready:
                    break

                # Small sleep to prevent busy waiting
                time.sleep(0.01)

        return task_results, completed

    def _execute_task(
        self,
        task: PipelineTask,
        execution_id: str,
        input_data: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> TaskResult:
        """Execute a single task with retry handling."""
        task_id = task.task_id
        retry_config = task.retry_config or self.config.retry_config
        timeout = task.timeout_seconds or self.config.task_timeout_seconds

        start_time = datetime.now(timezone.utc)
        attempt = 0
        last_error: Optional[Exception] = None

        while True:
            attempt += 1
            logger.debug(f"Executing task {task_id} (attempt {attempt})")

            try:
                # Create execution context
                context = ExecutionContext(
                    task_id=task_id,
                    execution_id=execution_id,
                    input_data=input_data,
                    parameters=task.parameters,
                    shared_state=shared_state,
                    checkpoint_manager=self.checkpoint_manager,
                )

                # Execute with optional timeout
                if timeout:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exec_pool:
                        future = exec_pool.submit(task.execute, context)
                        result = future.result(timeout=timeout)
                else:
                    result = task.execute(context)

                end_time = datetime.now(timezone.utc)
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    retry_count=attempt - 1,
                )

            except concurrent.futures.TimeoutError:
                last_error = TimeoutError(
                    f"Task {task_id} timed out after {timeout} seconds"
                )
                logger.warning(f"Task {task_id} timed out")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Task {task_id} failed (attempt {attempt}): {e}"
                )

                # Check if we should retry
                if retry_config.should_retry(e, attempt):
                    delay = retry_config.get_delay(attempt)
                    logger.info(
                        f"Retrying task {task_id} in {delay:.1f}s (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue

            # No more retries - return failure
            end_time = datetime.now(timezone.utc)
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(last_error) if last_error else "Unknown error",
                exception=last_error,
                start_time=start_time,
                end_time=end_time,
                retry_count=attempt - 1,
            )

    def _check_dependencies(
        self,
        task: PipelineTask,
        task_results: Dict[str, TaskResult],
        completed: Set[str],
    ) -> Tuple[bool, Optional[str]]:
        """Check if all dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in completed:
                return False, f"Dependency {dep_id} not completed"

            dep_result = task_results.get(dep_id)
            if dep_result and dep_result.status == TaskStatus.FAILED:
                return False, f"Dependency {dep_id} failed"
            if (
                dep_result
                and dep_result.status == TaskStatus.SKIPPED
                and not self.config.continue_on_skip
            ):
                return False, f"Dependency {dep_id} was skipped"

        return True, None

    def _prepare_inputs(
        self,
        task: PipelineTask,
        task_results: Dict[str, TaskResult],
        initial_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare input data from dependencies and initial inputs."""
        inputs = initial_inputs.copy()

        # Add results from dependencies
        for dep_id in task.dependencies:
            dep_result = task_results.get(dep_id)
            if dep_result and dep_result.result is not None:
                inputs[dep_id] = dep_result.result

        return inputs

    def _report_progress(self, progress: ExecutionProgress) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            progress.elapsed_seconds = (
                datetime.now(timezone.utc) - progress.start_time
            ).total_seconds() if progress.start_time else 0.0

            # Estimate remaining time
            if progress.completed_tasks > 0:
                avg_time = progress.elapsed_seconds / progress.completed_tasks
                progress.estimated_remaining_seconds = avg_time * progress.pending_tasks

            self.progress_callback(progress)

    def _save_checkpoint(
        self,
        execution_id: str,
        task_results: Dict[str, TaskResult],
        shared_state: Dict[str, Any],
        completed: Set[str],
    ) -> None:
        """Save execution checkpoint."""
        if self.checkpoint_manager:
            checkpoint_data = {
                "execution_id": execution_id,
                "task_results": {k: v.to_dict() for k, v in task_results.items()},
                "shared_state": shared_state,
                "completed": list(completed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.checkpoint_manager.save(execution_id, checkpoint_data)
            logger.debug(f"Saved checkpoint for execution {execution_id}")

    def _generate_execution_id(self, pipeline: Pipeline) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = f"{pipeline.pipeline_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancelled.set()
        logger.info("Pipeline execution cancelled")


# Convenience function for simple execution
def run_pipeline(
    pipeline: Pipeline,
    inputs: Optional[Dict[str, Any]] = None,
    config: Optional[ExecutionConfig] = None,
) -> ExecutionResult:
    """
    Execute a pipeline with default configuration.

    Args:
        pipeline: Pipeline to execute
        inputs: Initial input data
        config: Optional execution configuration

    Returns:
        ExecutionResult with all task results
    """
    runner = PipelineRunner(config=config)
    return runner.execute(pipeline, initial_inputs=inputs)
