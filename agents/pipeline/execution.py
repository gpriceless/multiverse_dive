"""
Execution Management Module for Pipeline Agent.

Provides execution engine wrapping core/analysis/execution/, step-by-step
execution with progress tracking, parallel step execution where possible,
resource management, and integration with distributed execution backends.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    NodeType,
    NodeStatus,
)
from core.analysis.assembly.optimizer import (
    PipelineOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    ExecutionPlan,
    ExecutionGroup,
)
from core.analysis.execution.runner import (
    ExecutionConfig,
    ExecutionMode,
    ExecutionResult,
    ExecutionProgress,
    ExecutionContext,
    Pipeline as RunnerPipeline,
    PipelineTask,
    PipelineRunner,
    TaskResult,
    TaskStatus,
    RetryConfig,
    RetryPolicy,
)
from core.analysis.execution.distributed import (
    DistributedConfig,
    DistributedBackend,
    DistributedPipelineRunner,
    get_executor,
    ClusterInfo,
)
from core.analysis.execution.checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetadata,
    CheckpointType,
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    get_global_registry,
)

logger = logging.getLogger(__name__)


class ResourceLimitPolicy(Enum):
    """Policy for handling resource limits."""
    STRICT = "strict"           # Fail if limits exceeded
    WARN = "warn"               # Warn but continue
    ADAPTIVE = "adaptive"       # Reduce parallelism to fit


@dataclass
class ResourceLimits:
    """
    Resource limits for execution.

    Attributes:
        max_memory_mb: Maximum memory usage
        max_concurrent_steps: Maximum parallel steps
        max_runtime_minutes: Maximum total runtime
        step_timeout_seconds: Timeout per step
        gpu_memory_mb: GPU memory limit
    """
    max_memory_mb: Optional[int] = None
    max_concurrent_steps: int = 4
    max_runtime_minutes: Optional[float] = None
    step_timeout_seconds: Optional[float] = None
    gpu_memory_mb: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_concurrent_steps": self.max_concurrent_steps,
            "max_runtime_minutes": self.max_runtime_minutes,
            "step_timeout_seconds": self.step_timeout_seconds,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


@dataclass
class ExecutionEngineConfig:
    """
    Configuration for the execution engine.

    Attributes:
        mode: Execution mode (sequential, parallel, distributed)
        resource_limits: Resource constraints
        limit_policy: Policy for resource limits
        checkpoint_enabled: Enable checkpointing
        checkpoint_interval: Steps between checkpoints
        distributed_config: Configuration for distributed execution
        retry_config: Configuration for task retries
        fail_fast: Stop on first failure
        continue_on_skip: Continue if step skipped
    """
    mode: ExecutionMode = ExecutionMode.PARALLEL
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    limit_policy: ResourceLimitPolicy = ResourceLimitPolicy.WARN
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 5
    distributed_config: Optional[DistributedConfig] = None
    retry_config: Optional[RetryConfig] = None
    fail_fast: bool = False
    continue_on_skip: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mode": self.mode.value,
            "resource_limits": self.resource_limits.to_dict(),
            "limit_policy": self.limit_policy.value,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_interval": self.checkpoint_interval,
            "fail_fast": self.fail_fast,
            "continue_on_skip": self.continue_on_skip,
        }


@dataclass
class StepExecutionResult:
    """
    Result from executing a single step.

    Attributes:
        step_id: Step identifier
        status: Execution status
        output: Step output data
        duration_seconds: Execution duration
        memory_used_mb: Peak memory usage
        retry_count: Number of retries
        error: Error message if failed
        metrics: Execution metrics
    """
    step_id: str
    status: TaskStatus
    output: Optional[Any] = None
    duration_seconds: float = 0.0
    memory_used_mb: float = 0.0
    retry_count: int = 0
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether step completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "memory_used_mb": self.memory_used_mb,
            "retry_count": self.retry_count,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class ExecutionState:
    """
    Current state of pipeline execution.

    Attributes:
        execution_id: Unique execution identifier
        pipeline_id: Pipeline being executed
        status: Current execution status
        current_group: Current execution group index
        completed_steps: Set of completed step IDs
        failed_steps: Set of failed step IDs
        running_steps: Set of currently running step IDs
        step_results: Results from completed steps
        start_time: Execution start time
        checkpoints_created: Number of checkpoints
    """
    execution_id: str
    pipeline_id: str
    status: TaskStatus = TaskStatus.PENDING
    current_group: int = 0
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    running_steps: Set[str] = field(default_factory=set)
    step_results: Dict[str, StepExecutionResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    checkpoints_created: int = 0

    @property
    def total_steps(self) -> int:
        """Total number of steps processed."""
        return len(self.completed_steps) + len(self.failed_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "current_group": self.current_group,
            "completed_steps": list(self.completed_steps),
            "failed_steps": list(self.failed_steps),
            "running_steps": list(self.running_steps),
            "total_steps": self.total_steps,
            "checkpoints_created": self.checkpoints_created,
        }


class ExecutionEngine:
    """
    Execution engine for running analysis pipelines.

    Wraps core/analysis/execution/ components and adds:
    - Step-by-step execution with progress tracking
    - Parallel step execution within resource limits
    - Resource monitoring and management
    - Integration with distributed backends (Dask/Ray)
    - Checkpoint management for fault tolerance

    Usage:
        engine = ExecutionEngine(config)
        result = await engine.execute(
            pipeline=graph,
            execution_plan=plan,
            inputs=data,
        )
    """

    def __init__(
        self,
        config: Optional[ExecutionEngineConfig] = None,
        registry: Optional[AlgorithmRegistry] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ):
        """
        Initialize the execution engine.

        Args:
            config: Engine configuration
            registry: Algorithm registry
            checkpoint_manager: Checkpoint manager for fault tolerance
            progress_callback: Callback for progress updates
        """
        self.config = config or ExecutionEngineConfig()
        self.registry = registry or get_global_registry()
        self.checkpoint_manager = checkpoint_manager
        self.progress_callback = progress_callback

        # Initialize optimizer for execution planning
        self.optimizer = PipelineOptimizer(registry=self.registry)

        # State
        self._state: Optional[ExecutionState] = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

        # Thread pool for parallel execution
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

        logger.info(f"ExecutionEngine initialized with mode {self.config.mode.value}")

    async def execute(
        self,
        pipeline: PipelineGraph,
        execution_plan: Optional[ExecutionPlan] = None,
        inputs: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline graph to execute
            execution_plan: Pre-computed execution plan (computed if None)
            inputs: Input data for the pipeline
            resume_from: Checkpoint ID to resume from

        Returns:
            ExecutionResult with all step results
        """
        import uuid

        execution_id = str(uuid.uuid4())[:16]
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting execution {execution_id} for pipeline {pipeline.id}")

        # Initialize state
        self._state = ExecutionState(
            execution_id=execution_id,
            pipeline_id=pipeline.id,
            status=TaskStatus.RUNNING,
            start_time=start_time,
        )

        # Compute execution plan if not provided
        if execution_plan is None:
            optimization_config = OptimizationConfig(
                strategy=self._get_optimization_strategy(),
                max_memory_mb=self.config.resource_limits.max_memory_mb,
                max_parallel_nodes=self.config.resource_limits.max_concurrent_steps,
                enable_checkpoints=self.config.checkpoint_enabled,
                checkpoint_interval_nodes=self.config.checkpoint_interval,
            )
            optimization_result = self.optimizer.optimize(pipeline, optimization_config)
            execution_plan = optimization_result.execution_plan

        # Resume from checkpoint if specified
        if resume_from and self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load(resume_from)
            if checkpoint_data:
                self._restore_state(checkpoint_data)
                logger.info(f"Resumed from checkpoint {resume_from}")

        # Report initial progress
        self._report_progress(ExecutionProgress(
            total_tasks=len([n for n in pipeline.nodes if n.node_type == NodeType.PROCESSOR]),
            completed_tasks=len(self._state.completed_steps),
            pending_tasks=len(execution_plan.execution_order) - len(self._state.completed_steps),
            start_time=start_time,
            current_phase="executing",
        ))

        # Execute based on mode
        try:
            if self.config.mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(pipeline, execution_plan, inputs or {})
            elif self.config.mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(pipeline, execution_plan, inputs or {})
            elif self.config.mode == ExecutionMode.DISTRIBUTED:
                await self._execute_distributed(pipeline, execution_plan, inputs or {})

            # Determine final status
            if self._cancelled.is_set():
                self._state.status = TaskStatus.CANCELLED
            elif self._state.failed_steps:
                self._state.status = TaskStatus.FAILED
            else:
                self._state.status = TaskStatus.COMPLETED

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self._state.status = TaskStatus.FAILED

        # Build result
        end_time = datetime.now(timezone.utc)
        task_results = {
            step_id: TaskResult(
                task_id=step_id,
                status=result.status,
                result=result.output,
                error=result.error,
            )
            for step_id, result in self._state.step_results.items()
        }

        result = ExecutionResult(
            execution_id=execution_id,
            status=self._state.status,
            task_results=task_results,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=(end_time - start_time).total_seconds(),
            checkpoints_saved=self._state.checkpoints_created,
            metadata={
                "pipeline_id": pipeline.id,
                "completed_steps": list(self._state.completed_steps),
                "failed_steps": list(self._state.failed_steps),
            },
        )

        logger.info(
            f"Execution {execution_id} completed: {self._state.status.value} "
            f"({len(self._state.completed_steps)} completed, {len(self._state.failed_steps)} failed)"
        )

        return result

    async def _execute_sequential(
        self,
        pipeline: PipelineGraph,
        plan: ExecutionPlan,
        inputs: Dict[str, Any],
    ) -> None:
        """Execute steps sequentially."""
        for step_id in plan.execution_order:
            if self._cancelled.is_set():
                break

            if step_id in self._state.completed_steps:
                continue

            node = pipeline.get_node(step_id)
            if node is None or node.node_type != NodeType.PROCESSOR:
                continue

            # Execute step
            result = await self._execute_step(node, inputs, pipeline)
            self._state.step_results[step_id] = result

            if result.success:
                self._state.completed_steps.add(step_id)
                # Add output to inputs for next steps
                if result.output is not None:
                    inputs[step_id] = result.output
            else:
                self._state.failed_steps.add(step_id)
                if self.config.fail_fast:
                    break

            # Checkpoint if needed
            if self.config.checkpoint_enabled:
                if len(self._state.completed_steps) % self.config.checkpoint_interval == 0:
                    await self._create_checkpoint(inputs)

            # Report progress
            self._report_progress(ExecutionProgress(
                total_tasks=len(plan.execution_order),
                completed_tasks=len(self._state.completed_steps),
                failed_tasks=len(self._state.failed_steps),
                pending_tasks=len(plan.execution_order) - len(self._state.completed_steps) - len(self._state.failed_steps),
                current_phase="executing",
            ))

    async def _execute_parallel(
        self,
        pipeline: PipelineGraph,
        plan: ExecutionPlan,
        inputs: Dict[str, Any],
    ) -> None:
        """Execute steps in parallel where possible."""
        max_workers = min(
            self.config.resource_limits.max_concurrent_steps,
            len(plan.execution_order),
        )

        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            self._executor = executor

            for group in plan.groups:
                if self._cancelled.is_set():
                    break

                self._state.current_group += 1

                # Execute checkpoint before group if marked
                if group.checkpoint_before and self.config.checkpoint_enabled:
                    await self._create_checkpoint(inputs)

                # Get steps that can run in parallel
                runnable_steps = [
                    step_id for step_id in group.node_ids
                    if step_id not in self._state.completed_steps
                    and step_id not in self._state.failed_steps
                ]

                if not runnable_steps:
                    continue

                # Check resource limits
                if self.config.limit_policy == ResourceLimitPolicy.ADAPTIVE:
                    runnable_steps = self._apply_resource_limits(
                        runnable_steps, pipeline
                    )

                # Submit all steps in group
                futures = {}
                for step_id in runnable_steps:
                    node = pipeline.get_node(step_id)
                    if node is None or node.node_type != NodeType.PROCESSOR:
                        continue

                    self._state.running_steps.add(step_id)
                    future = executor.submit(
                        self._execute_step_sync,
                        node, inputs, pipeline
                    )
                    futures[future] = step_id

                # Wait for all in group to complete
                for future in concurrent.futures.as_completed(futures.keys()):
                    step_id = futures[future]
                    self._state.running_steps.discard(step_id)

                    try:
                        result = future.result()
                        self._state.step_results[step_id] = result

                        if result.success:
                            self._state.completed_steps.add(step_id)
                            if result.output is not None:
                                inputs[step_id] = result.output
                        else:
                            self._state.failed_steps.add(step_id)
                            if self.config.fail_fast:
                                self._cancelled.set()

                    except Exception as e:
                        logger.error(f"Step {step_id} raised exception: {e}")
                        self._state.failed_steps.add(step_id)
                        self._state.step_results[step_id] = StepExecutionResult(
                            step_id=step_id,
                            status=TaskStatus.FAILED,
                            error=str(e),
                        )

                # Report progress after group
                self._report_progress(ExecutionProgress(
                    total_tasks=len(plan.execution_order),
                    completed_tasks=len(self._state.completed_steps),
                    failed_tasks=len(self._state.failed_steps),
                    running_tasks=len(self._state.running_steps),
                    pending_tasks=len(plan.execution_order) - len(self._state.completed_steps) - len(self._state.failed_steps),
                    current_phase=f"group_{self._state.current_group}",
                ))

            self._executor = None

    async def _execute_distributed(
        self,
        pipeline: PipelineGraph,
        plan: ExecutionPlan,
        inputs: Dict[str, Any],
    ) -> None:
        """Execute using distributed backend (Dask/Ray)."""
        distributed_config = self.config.distributed_config or DistributedConfig()

        # Build runner pipeline from graph
        runner_pipeline = self._build_runner_pipeline(pipeline, plan)

        # Use distributed runner
        runner = DistributedPipelineRunner(
            distributed_config=distributed_config,
            progress_callback=self.progress_callback,
        )

        result = runner.execute(runner_pipeline, initial_inputs=inputs)

        # Transfer results to our state
        for task_id, task_result in result.task_results.items():
            self._state.step_results[task_id] = StepExecutionResult(
                step_id=task_id,
                status=task_result.status,
                output=task_result.result,
                duration_seconds=task_result.duration_seconds or 0.0,
                error=task_result.error,
            )

            if task_result.success:
                self._state.completed_steps.add(task_id)
            else:
                self._state.failed_steps.add(task_id)

    async def _execute_step(
        self,
        node: PipelineNode,
        inputs: Dict[str, Any],
        pipeline: PipelineGraph,
    ) -> StepExecutionResult:
        """Execute a single step asynchronously."""
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_step_sync,
            node, inputs, pipeline
        )

    def _execute_step_sync(
        self,
        node: PipelineNode,
        inputs: Dict[str, Any],
        pipeline: PipelineGraph,
    ) -> StepExecutionResult:
        """Execute a single step synchronously."""
        start_time = datetime.now(timezone.utc)
        step_id = node.id

        logger.info(f"Executing step {step_id} ({node.processor})")

        try:
            # Prepare step inputs from incoming edges
            step_inputs = self._gather_step_inputs(node, inputs, pipeline)

            # Get algorithm
            algorithm = self.registry.get(node.processor) if node.processor else None

            # Execute (stub implementation - real would call algorithm)
            output = self._run_algorithm(algorithm, step_inputs, node.parameters)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            return StepExecutionResult(
                step_id=step_id,
                status=TaskStatus.COMPLETED,
                output=output,
                duration_seconds=duration,
                metrics={
                    "input_count": len(step_inputs),
                    "parameter_count": len(node.parameters),
                },
            )

        except Exception as e:
            logger.error(f"Step {step_id} failed: {e}")
            end_time = datetime.now(timezone.utc)

            return StepExecutionResult(
                step_id=step_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_seconds=(end_time - start_time).total_seconds(),
            )

    def _gather_step_inputs(
        self,
        node: PipelineNode,
        inputs: Dict[str, Any],
        pipeline: PipelineGraph,
    ) -> Dict[str, Any]:
        """Gather inputs for a step from pipeline inputs and previous results."""
        step_inputs = {}

        for edge in pipeline.get_incoming_edges(node.id):
            source_node = pipeline.get_node(edge.source_node)
            if source_node is None:
                continue

            if source_node.node_type == NodeType.INPUT:
                # Get from pipeline inputs
                if source_node.name in inputs:
                    step_inputs[edge.target_port] = inputs[source_node.name]
            else:
                # Get from previous step output
                if edge.source_node in inputs:
                    step_inputs[edge.target_port] = inputs[edge.source_node]

        return step_inputs

    def _run_algorithm(
        self,
        algorithm: Optional[AlgorithmMetadata],
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run an algorithm with inputs and parameters.

        This is a stub - real implementation would:
        1. Load algorithm module
        2. Instantiate algorithm class
        3. Execute with inputs
        4. Return output data
        """
        time.sleep(0.01)  # Simulate processing

        return {
            "algorithm_id": algorithm.id if algorithm else "unknown",
            "processed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _apply_resource_limits(
        self,
        step_ids: List[str],
        pipeline: PipelineGraph,
    ) -> List[str]:
        """Apply resource limits to reduce parallel execution."""
        if not self.config.resource_limits.max_memory_mb:
            return step_ids

        # Estimate memory for each step
        memory_estimates = {}
        for step_id in step_ids:
            node = pipeline.get_node(step_id)
            if node and node.processor:
                algorithm = self.registry.get(node.processor)
                if algorithm and algorithm.resources.memory_mb:
                    memory_estimates[step_id] = algorithm.resources.memory_mb
                else:
                    memory_estimates[step_id] = 512  # Default
            else:
                memory_estimates[step_id] = 512

        # Select steps that fit within limit
        available_memory = self.config.resource_limits.max_memory_mb
        selected = []

        for step_id in step_ids:
            if memory_estimates[step_id] <= available_memory:
                selected.append(step_id)
                available_memory -= memory_estimates[step_id]

        return selected

    def _build_runner_pipeline(
        self,
        graph: PipelineGraph,
        plan: ExecutionPlan,
    ) -> RunnerPipeline:
        """Build a runner Pipeline from graph and plan."""
        runner_pipeline = RunnerPipeline(
            pipeline_id=graph.id,
            name=graph.name,
        )

        for step_id in plan.execution_order:
            node = graph.get_node(step_id)
            if node is None or node.node_type != NodeType.PROCESSOR:
                continue

            # Get dependencies from incoming edges
            dependencies = [
                edge.source_node
                for edge in graph.get_incoming_edges(step_id)
                if graph.get_node(edge.source_node).node_type == NodeType.PROCESSOR
            ]

            task = PipelineTask(
                task_id=step_id,
                executor=lambda ctx: {"step_id": ctx.task_id, "processed": True},
                dependencies=dependencies,
                parameters=node.parameters,
            )
            runner_pipeline.add_task(task)

        return runner_pipeline

    async def _create_checkpoint(
        self,
        inputs: Dict[str, Any],
    ) -> Optional[CheckpointMetadata]:
        """Create execution checkpoint."""
        if not self.checkpoint_manager or not self._state:
            return None

        checkpoint_data = {
            "state": self._state.to_dict(),
            "step_results": {
                k: v.to_dict() for k, v in self._state.step_results.items()
            },
            "completed": list(self._state.completed_steps),
        }

        metadata = self.checkpoint_manager.save(
            execution_id=self._state.execution_id,
            data=checkpoint_data,
            checkpoint_type=CheckpointType.AUTO,
        )

        self._state.checkpoints_created += 1
        logger.debug(f"Created checkpoint {metadata.checkpoint_id}")

        return metadata

    def _restore_state(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore state from checkpoint data."""
        if "state" in checkpoint_data:
            state_dict = checkpoint_data["state"]
            self._state.completed_steps = set(state_dict.get("completed_steps", []))
            self._state.failed_steps = set(state_dict.get("failed_steps", []))

        if "step_results" in checkpoint_data:
            for step_id, result_dict in checkpoint_data["step_results"].items():
                self._state.step_results[step_id] = StepExecutionResult(
                    step_id=step_id,
                    status=TaskStatus(result_dict.get("status", "completed")),
                    output=result_dict.get("output"),
                    duration_seconds=result_dict.get("duration_seconds", 0.0),
                    error=result_dict.get("error"),
                )

    def _get_optimization_strategy(self) -> OptimizationStrategy:
        """Get optimization strategy based on config."""
        if self.config.mode == ExecutionMode.SEQUENTIAL:
            return OptimizationStrategy.MINIMIZE_MEMORY
        elif self.config.mode == ExecutionMode.DISTRIBUTED:
            return OptimizationStrategy.MAXIMIZE_PARALLELISM
        else:
            return OptimizationStrategy.BALANCED

    def _report_progress(self, progress: ExecutionProgress) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            if self._state and self._state.start_time:
                progress.start_time = self._state.start_time
                progress.elapsed_seconds = (
                    datetime.now(timezone.utc) - self._state.start_time
                ).total_seconds()

            self.progress_callback(progress)

    def cancel(self) -> None:
        """Cancel current execution."""
        self._cancelled.set()
        logger.info("Execution cancelled")

    def get_state(self) -> Optional[ExecutionState]:
        """Get current execution state."""
        return self._state

    def get_cluster_info(self) -> Optional[ClusterInfo]:
        """Get distributed cluster info if available."""
        if self.config.mode != ExecutionMode.DISTRIBUTED:
            return None

        distributed_config = self.config.distributed_config or DistributedConfig()
        executor = get_executor(distributed_config)

        try:
            return executor.get_cluster_info()
        except Exception:
            return None
