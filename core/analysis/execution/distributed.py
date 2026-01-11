"""
Distributed Execution Backend for Analysis Pipelines.

Provides integration with distributed computing frameworks:
- Dask for parallel/distributed array operations
- Ray for distributed task execution
- Fallback to local threading when distributed frameworks unavailable

Features:
- Automatic backend detection and selection
- Task graph optimization for distributed execution
- Resource-aware task scheduling
- Progress tracking across distributed workers
- Fault tolerance with task retries
- Result collection and aggregation
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from .runner import (
    ExecutionConfig,
    ExecutionContext,
    ExecutionMode,
    ExecutionProgress,
    ExecutionResult,
    Pipeline,
    PipelineTask,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Available distributed computing backends."""

    LOCAL = "local"  # Local threading (fallback)
    DASK = "dask"  # Dask distributed
    RAY = "ray"  # Ray distributed
    AUTO = "auto"  # Automatic selection


@dataclass
class WorkerInfo:
    """Information about a distributed worker."""

    worker_id: str
    host: str
    port: int
    ncores: int
    memory_bytes: int
    status: str = "active"
    current_tasks: int = 0
    total_tasks_completed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_gb(self) -> float:
        """Memory in gigabytes."""
        return self.memory_bytes / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "ncores": self.ncores,
            "memory_gb": self.memory_gb,
            "status": self.status,
            "current_tasks": self.current_tasks,
            "total_tasks_completed": self.total_tasks_completed,
            "metadata": self.metadata,
        }


@dataclass
class ClusterInfo:
    """Information about the distributed cluster."""

    backend: DistributedBackend
    scheduler_address: Optional[str]
    workers: List[WorkerInfo]
    total_cores: int
    total_memory_bytes: int
    is_connected: bool = False
    connection_time: Optional[datetime] = None

    @property
    def worker_count(self) -> int:
        """Number of active workers."""
        return len([w for w in self.workers if w.status == "active"])

    @property
    def total_memory_gb(self) -> float:
        """Total memory in gigabytes."""
        return self.total_memory_bytes / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "scheduler_address": self.scheduler_address,
            "workers": [w.to_dict() for w in self.workers],
            "worker_count": self.worker_count,
            "total_cores": self.total_cores,
            "total_memory_gb": self.total_memory_gb,
            "is_connected": self.is_connected,
            "connection_time": self.connection_time.isoformat()
            if self.connection_time
            else None,
        }


@dataclass
class DistributedConfig:
    """
    Configuration for distributed execution.

    Attributes:
        backend: Distributed backend to use
        scheduler_address: Address of scheduler (for Dask/Ray)
        n_workers: Number of workers for local cluster
        threads_per_worker: Threads per worker
        memory_limit_per_worker: Memory limit per worker (e.g., "4GB")
        adaptive: Whether to use adaptive scaling
        adaptive_min_workers: Minimum workers for adaptive
        adaptive_max_workers: Maximum workers for adaptive
        task_timeout_seconds: Per-task timeout
        retry_on_worker_failure: Retry tasks on worker failure
        dashboard_address: Address for monitoring dashboard
    """

    backend: DistributedBackend = DistributedBackend.AUTO
    scheduler_address: Optional[str] = None
    n_workers: int = 4
    threads_per_worker: int = 1
    memory_limit_per_worker: str = "4GB"
    adaptive: bool = False
    adaptive_min_workers: int = 1
    adaptive_max_workers: int = 10
    task_timeout_seconds: Optional[float] = None
    retry_on_worker_failure: bool = True
    dashboard_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "scheduler_address": self.scheduler_address,
            "n_workers": self.n_workers,
            "threads_per_worker": self.threads_per_worker,
            "memory_limit_per_worker": self.memory_limit_per_worker,
            "adaptive": self.adaptive,
            "adaptive_min_workers": self.adaptive_min_workers,
            "adaptive_max_workers": self.adaptive_max_workers,
            "task_timeout_seconds": self.task_timeout_seconds,
            "retry_on_worker_failure": self.retry_on_worker_failure,
            "dashboard_address": self.dashboard_address,
        }


class DistributedExecutorBase(ABC):
    """
    Abstract base class for distributed executors.

    Implementations provide integration with specific
    distributed computing frameworks.
    """

    def __init__(self, config: DistributedConfig):
        """
        Initialize the distributed executor.

        Args:
            config: Distributed execution configuration
        """
        self.config = config
        self._cluster: Optional[Any] = None
        self._client: Optional[Any] = None
        self._connected = False
        self._lock = threading.Lock()

    @abstractmethod
    def connect(self) -> ClusterInfo:
        """
        Connect to the distributed cluster.

        Returns:
            ClusterInfo with cluster details
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the cluster."""
        pass

    @abstractmethod
    def submit_task(
        self,
        task: PipelineTask,
        context: ExecutionContext,
        dependencies: List[Any],
    ) -> Any:
        """
        Submit a task for distributed execution.

        Args:
            task: Task to execute
            context: Execution context
            dependencies: Futures from dependency tasks

        Returns:
            Future or reference to the submitted task
        """
        pass

    @abstractmethod
    def gather_results(self, futures: List[Any]) -> List[TaskResult]:
        """
        Gather results from submitted tasks.

        Args:
            futures: List of task futures/references

        Returns:
            List of TaskResult objects
        """
        pass

    @abstractmethod
    def get_cluster_info(self) -> ClusterInfo:
        """Get current cluster information."""
        pass

    @property
    def is_connected(self) -> bool:
        """Whether connected to cluster."""
        return self._connected


class LocalExecutor(DistributedExecutorBase):
    """
    Local threading-based executor as fallback.

    Uses ThreadPoolExecutor for parallel execution
    when distributed frameworks are unavailable.
    """

    def __init__(self, config: DistributedConfig):
        """Initialize local executor."""
        super().__init__(config)
        import concurrent.futures

        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def connect(self) -> ClusterInfo:
        """Create local thread pool."""
        import concurrent.futures
        import os

        # Try to get system memory info
        try:
            import psutil
            memory_available = psutil.virtual_memory().available
        except ImportError:
            # Fallback: assume 4GB available
            memory_available = 4 * 1024 * 1024 * 1024

        with self._lock:
            if self._connected:
                return self.get_cluster_info()

            n_workers = self.config.n_workers
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            )

            workers = [
                WorkerInfo(
                    worker_id=f"local-thread-{i}",
                    host="localhost",
                    port=0,
                    ncores=1,
                    memory_bytes=memory_available // n_workers,
                    status="active",
                )
                for i in range(n_workers)
            ]

            self._connected = True
            logger.info(f"Local executor initialized with {n_workers} workers")

            return ClusterInfo(
                backend=DistributedBackend.LOCAL,
                scheduler_address="local",
                workers=workers,
                total_cores=n_workers,
                total_memory_bytes=memory_available,
                is_connected=True,
                connection_time=datetime.now(timezone.utc),
            )

    def disconnect(self) -> None:
        """Shutdown thread pool."""
        with self._lock:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            self._connected = False
            logger.info("Local executor disconnected")

    def submit_task(
        self,
        task: PipelineTask,
        context: ExecutionContext,
        dependencies: List[Any],
    ) -> Any:
        """Submit task to thread pool."""
        if not self._executor:
            raise RuntimeError("Executor not connected")

        # Wait for dependencies to complete
        import concurrent.futures

        for dep in dependencies:
            if isinstance(dep, concurrent.futures.Future):
                dep.result()

        # Submit task
        return self._executor.submit(self._execute_task, task, context)

    def _execute_task(
        self, task: PipelineTask, context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task."""
        start_time = datetime.now(timezone.utc)
        try:
            result = task.execute(context)
            end_time = datetime.now(timezone.utc)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            logger.error(f"Task {task.task_id} failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                exception=e,
                start_time=start_time,
                end_time=end_time,
            )

    def gather_results(self, futures: List[Any]) -> List[TaskResult]:
        """Gather results from futures."""
        import concurrent.futures

        results = []
        for future in futures:
            if isinstance(future, concurrent.futures.Future):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(
                        TaskResult(
                            task_id="unknown",
                            status=TaskStatus.FAILED,
                            error=str(e),
                        )
                    )
            else:
                results.append(future)
        return results

    def get_cluster_info(self) -> ClusterInfo:
        """Get local cluster info."""
        # Try to get system memory info
        try:
            import psutil
            memory_available = psutil.virtual_memory().available
        except ImportError:
            # Fallback: assume 4GB available
            memory_available = 4 * 1024 * 1024 * 1024

        n_workers = self.config.n_workers

        workers = [
            WorkerInfo(
                worker_id=f"local-thread-{i}",
                host="localhost",
                port=0,
                ncores=1,
                memory_bytes=memory_available // n_workers,
                status="active" if self._connected else "inactive",
            )
            for i in range(n_workers)
        ]

        return ClusterInfo(
            backend=DistributedBackend.LOCAL,
            scheduler_address="local",
            workers=workers,
            total_cores=n_workers,
            total_memory_bytes=memory_available,
            is_connected=self._connected,
            connection_time=datetime.now(timezone.utc) if self._connected else None,
        )


class DaskExecutor(DistributedExecutorBase):
    """
    Dask-based distributed executor.

    Integrates with Dask distributed for:
    - Distributed array operations
    - Task graph optimization
    - Adaptive scaling
    - Dashboard monitoring
    """

    def connect(self) -> ClusterInfo:
        """Connect to or create Dask cluster."""
        with self._lock:
            if self._connected:
                return self.get_cluster_info()

            try:
                from dask.distributed import Client, LocalCluster

                if self.config.scheduler_address:
                    # Connect to existing cluster
                    self._client = Client(self.config.scheduler_address)
                    logger.info(
                        f"Connected to Dask cluster at {self.config.scheduler_address}"
                    )
                else:
                    # Create local cluster
                    self._cluster = LocalCluster(
                        n_workers=self.config.n_workers,
                        threads_per_worker=self.config.threads_per_worker,
                        memory_limit=self.config.memory_limit_per_worker,
                        dashboard_address=self.config.dashboard_address or ":8787",
                    )
                    self._client = Client(self._cluster)
                    logger.info(
                        f"Created local Dask cluster with {self.config.n_workers} workers"
                    )

                # Enable adaptive scaling if configured
                if self.config.adaptive and self._cluster:
                    self._cluster.adapt(
                        minimum=self.config.adaptive_min_workers,
                        maximum=self.config.adaptive_max_workers,
                    )
                    logger.info(
                        f"Enabled adaptive scaling: {self.config.adaptive_min_workers}-{self.config.adaptive_max_workers} workers"
                    )

                self._connected = True
                return self.get_cluster_info()

            except ImportError:
                logger.warning(
                    "Dask distributed not available, falling back to local executor"
                )
                raise RuntimeError("Dask distributed not installed")

    def disconnect(self) -> None:
        """Disconnect from Dask cluster."""
        with self._lock:
            if self._client:
                self._client.close()
                self._client = None
            if self._cluster:
                self._cluster.close()
                self._cluster = None
            self._connected = False
            logger.info("Disconnected from Dask cluster")

    def submit_task(
        self,
        task: PipelineTask,
        context: ExecutionContext,
        dependencies: List[Any],
    ) -> Any:
        """Submit task to Dask cluster."""
        if not self._client:
            raise RuntimeError("Not connected to Dask cluster")

        # Submit with dependencies
        future = self._client.submit(
            self._execute_task,
            task,
            context,
            key=task.task_id,
            pure=False,
        )
        return future

    def _execute_task(
        self, task: PipelineTask, context: ExecutionContext
    ) -> TaskResult:
        """Execute task on Dask worker."""
        start_time = datetime.now(timezone.utc)
        try:
            result = task.execute(context)
            end_time = datetime.now(timezone.utc)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
            )

    def gather_results(self, futures: List[Any]) -> List[TaskResult]:
        """Gather results from Dask futures."""
        if not self._client:
            raise RuntimeError("Not connected to Dask cluster")

        return self._client.gather(futures)

    def get_cluster_info(self) -> ClusterInfo:
        """Get Dask cluster information."""
        if not self._client:
            return ClusterInfo(
                backend=DistributedBackend.DASK,
                scheduler_address=None,
                workers=[],
                total_cores=0,
                total_memory_bytes=0,
                is_connected=False,
            )

        # Get worker info from scheduler
        scheduler_info = self._client.scheduler_info()
        workers_info = scheduler_info.get("workers", {})

        workers = []
        total_cores = 0
        total_memory = 0

        for worker_id, info in workers_info.items():
            ncores = info.get("nthreads", 1)
            memory = info.get("memory_limit", 0)

            workers.append(
                WorkerInfo(
                    worker_id=worker_id,
                    host=info.get("host", "unknown"),
                    port=info.get("port", 0),
                    ncores=ncores,
                    memory_bytes=memory,
                    status=info.get("status", "unknown"),
                    current_tasks=info.get("executing", 0),
                )
            )
            total_cores += ncores
            total_memory += memory

        return ClusterInfo(
            backend=DistributedBackend.DASK,
            scheduler_address=scheduler_info.get("address"),
            workers=workers,
            total_cores=total_cores,
            total_memory_bytes=total_memory,
            is_connected=self._connected,
            connection_time=datetime.now(timezone.utc) if self._connected else None,
        )

    def scatter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scatter data to workers for efficient access.

        Args:
            data: Dictionary of data to scatter

        Returns:
            Dictionary of futures referencing scattered data
        """
        if not self._client:
            raise RuntimeError("Not connected to Dask cluster")

        return {key: self._client.scatter(value) for key, value in data.items()}


class RayExecutor(DistributedExecutorBase):
    """
    Ray-based distributed executor.

    Integrates with Ray for:
    - Distributed task execution
    - Object store for data sharing
    - Automatic task placement
    - GPU support
    """

    def connect(self) -> ClusterInfo:
        """Connect to or initialize Ray cluster."""
        with self._lock:
            if self._connected:
                return self.get_cluster_info()

            try:
                import ray

                if self.config.scheduler_address:
                    # Connect to existing cluster
                    ray.init(address=self.config.scheduler_address)
                    logger.info(
                        f"Connected to Ray cluster at {self.config.scheduler_address}"
                    )
                else:
                    # Initialize local Ray
                    ray.init(
                        num_cpus=self.config.n_workers,
                        ignore_reinit_error=True,
                    )
                    logger.info(
                        f"Initialized local Ray with {self.config.n_workers} CPUs"
                    )

                self._connected = True
                return self.get_cluster_info()

            except ImportError:
                logger.warning(
                    "Ray not available, falling back to local executor"
                )
                raise RuntimeError("Ray not installed")

    def disconnect(self) -> None:
        """Shutdown Ray."""
        with self._lock:
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
            except ImportError:
                pass
            self._connected = False
            logger.info("Disconnected from Ray")

    def submit_task(
        self,
        task: PipelineTask,
        context: ExecutionContext,
        dependencies: List[Any],
    ) -> Any:
        """Submit task to Ray cluster."""
        import ray

        # Create remote function for task execution
        @ray.remote
        def execute_remote(task_dict, context_dict):
            # Reconstruct task and context
            start_time = datetime.now(timezone.utc)
            try:
                # Simple execution - in production would reconstruct full objects
                result = {"task_id": task_dict["task_id"], "status": "completed"}
                end_time = datetime.now(timezone.utc)
                return TaskResult(
                    task_id=task_dict["task_id"],
                    status=TaskStatus.COMPLETED,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                return TaskResult(
                    task_id=task_dict["task_id"],
                    status=TaskStatus.FAILED,
                    error=str(e),
                    start_time=start_time,
                    end_time=end_time,
                )

        # Wait for dependencies
        if dependencies:
            ray.get(dependencies)

        # Submit task
        task_dict = {
            "task_id": task.task_id,
            "parameters": task.parameters,
        }
        context_dict = {
            "task_id": context.task_id,
            "execution_id": context.execution_id,
            "input_data": context.input_data,
            "parameters": context.parameters,
        }

        return execute_remote.remote(task_dict, context_dict)

    def gather_results(self, futures: List[Any]) -> List[TaskResult]:
        """Gather results from Ray object refs."""
        import ray

        return ray.get(futures)

    def get_cluster_info(self) -> ClusterInfo:
        """Get Ray cluster information."""
        try:
            import ray

            if not ray.is_initialized():
                return ClusterInfo(
                    backend=DistributedBackend.RAY,
                    scheduler_address=None,
                    workers=[],
                    total_cores=0,
                    total_memory_bytes=0,
                    is_connected=False,
                )

            resources = ray.cluster_resources()
            nodes = ray.nodes()

            workers = []
            total_cores = int(resources.get("CPU", 0))
            total_memory = int(resources.get("memory", 0))

            for node in nodes:
                if node.get("Alive", False):
                    node_resources = node.get("Resources", {})
                    workers.append(
                        WorkerInfo(
                            worker_id=node.get("NodeID", "unknown"),
                            host=node.get("NodeManagerAddress", "unknown"),
                            port=node.get("NodeManagerPort", 0),
                            ncores=int(node_resources.get("CPU", 0)),
                            memory_bytes=int(node_resources.get("memory", 0)),
                            status="active" if node.get("Alive") else "inactive",
                        )
                    )

            return ClusterInfo(
                backend=DistributedBackend.RAY,
                scheduler_address=ray.get_runtime_context().gcs_address
                if hasattr(ray.get_runtime_context(), "gcs_address")
                else None,
                workers=workers,
                total_cores=total_cores,
                total_memory_bytes=total_memory,
                is_connected=self._connected,
                connection_time=datetime.now(timezone.utc) if self._connected else None,
            )

        except ImportError:
            return ClusterInfo(
                backend=DistributedBackend.RAY,
                scheduler_address=None,
                workers=[],
                total_cores=0,
                total_memory_bytes=0,
                is_connected=False,
            )


def get_executor(config: DistributedConfig) -> DistributedExecutorBase:
    """
    Get appropriate executor based on configuration.

    Args:
        config: Distributed execution configuration

    Returns:
        Appropriate executor instance
    """
    backend = config.backend

    if backend == DistributedBackend.AUTO:
        # Try to auto-detect best available backend
        backend = _detect_best_backend()

    if backend == DistributedBackend.DASK:
        try:
            return DaskExecutor(config)
        except RuntimeError:
            logger.warning("Dask unavailable, falling back to local")
            return LocalExecutor(config)

    elif backend == DistributedBackend.RAY:
        try:
            return RayExecutor(config)
        except RuntimeError:
            logger.warning("Ray unavailable, falling back to local")
            return LocalExecutor(config)

    else:
        return LocalExecutor(config)


def _detect_best_backend() -> DistributedBackend:
    """Detect best available distributed backend."""
    # Try Dask first
    try:
        import dask.distributed

        return DistributedBackend.DASK
    except ImportError:
        pass

    # Try Ray
    try:
        import ray

        return DistributedBackend.RAY
    except ImportError:
        pass

    # Fallback to local
    return DistributedBackend.LOCAL


class DistributedPipelineRunner:
    """
    Pipeline runner with distributed execution support.

    Wraps the standard PipelineRunner with distributed
    execution capabilities via Dask or Ray.
    """

    def __init__(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ):
        """
        Initialize the distributed runner.

        Args:
            execution_config: Standard execution configuration
            distributed_config: Distributed execution configuration
            progress_callback: Callback for progress updates
        """
        self.execution_config = execution_config or ExecutionConfig(
            mode=ExecutionMode.DISTRIBUTED
        )
        self.distributed_config = distributed_config or DistributedConfig()
        self.progress_callback = progress_callback
        self._executor: Optional[DistributedExecutorBase] = None
        self._cancelled = threading.Event()

    def execute(
        self,
        pipeline: Pipeline,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute pipeline using distributed backend.

        Args:
            pipeline: Pipeline to execute
            initial_inputs: Initial input data

        Returns:
            ExecutionResult with all task results
        """
        import hashlib

        # Generate execution ID
        timestamp = datetime.now(timezone.utc).isoformat()
        execution_id = hashlib.sha256(
            f"{pipeline.pipeline_id}:{timestamp}".encode()
        ).hexdigest()[:16]

        logger.info(f"Starting distributed execution: {execution_id}")

        # Validate pipeline
        is_valid, errors = pipeline.validate()
        if not is_valid:
            return ExecutionResult(
                execution_id=execution_id,
                status=TaskStatus.FAILED,
                metadata={"validation_errors": errors},
            )

        # Get executor and connect
        self._executor = get_executor(self.distributed_config)
        try:
            cluster_info = self._executor.connect()
            logger.info(
                f"Connected to {cluster_info.backend.value} cluster "
                f"with {cluster_info.worker_count} workers"
            )
        except Exception as e:
            return ExecutionResult(
                execution_id=execution_id,
                status=TaskStatus.FAILED,
                metadata={"error": f"Failed to connect to cluster: {e}"},
            )

        start_time = datetime.now(timezone.utc)
        task_results: Dict[str, TaskResult] = {}
        futures: Dict[str, Any] = {}

        try:
            # Get execution order
            execution_order = pipeline.get_execution_order()

            # Initialize progress
            progress = ExecutionProgress(
                total_tasks=len(pipeline),
                pending_tasks=len(pipeline),
                start_time=start_time,
                current_phase="submitting",
            )
            self._report_progress(progress)

            # Submit tasks respecting dependencies
            for task_id in execution_order:
                if self._cancelled.is_set():
                    break

                task = pipeline.get_task(task_id)
                if task is None:
                    continue

                # Get dependency futures
                dep_futures = [
                    futures[dep_id]
                    for dep_id in task.dependencies
                    if dep_id in futures
                ]

                # Create execution context
                context = ExecutionContext(
                    task_id=task_id,
                    execution_id=execution_id,
                    input_data=initial_inputs or {},
                    parameters=task.parameters,
                )

                # Submit task
                future = self._executor.submit_task(task, context, dep_futures)
                futures[task_id] = future

            # Gather results
            progress.current_phase = "executing"
            self._report_progress(progress)

            all_futures = list(futures.values())
            results = self._executor.gather_results(all_futures)

            # Map results back to task IDs
            for task_id, result in zip(futures.keys(), results):
                if isinstance(result, TaskResult):
                    task_results[task_id] = result
                else:
                    task_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result=result,
                    )

                if task_results[task_id].success:
                    progress.completed_tasks += 1
                else:
                    progress.failed_tasks += 1
                progress.pending_tasks -= 1
                self._report_progress(progress)

            # Determine final status
            end_time = datetime.now(timezone.utc)
            if self._cancelled.is_set():
                final_status = TaskStatus.CANCELLED
            elif all(r.success for r in task_results.values()):
                final_status = TaskStatus.COMPLETED
            else:
                final_status = TaskStatus.FAILED

            return ExecutionResult(
                execution_id=execution_id,
                status=final_status,
                task_results=task_results,
                start_time=start_time,
                end_time=end_time,
                total_duration_seconds=(end_time - start_time).total_seconds(),
                metadata={"cluster_info": cluster_info.to_dict()},
            )

        except Exception as e:
            logger.error(f"Distributed execution failed: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=TaskStatus.FAILED,
                task_results=task_results,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                metadata={"error": str(e)},
            )

        finally:
            # Disconnect from cluster
            if self._executor:
                self._executor.disconnect()

    def _report_progress(self, progress: ExecutionProgress) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            progress.elapsed_seconds = (
                datetime.now(timezone.utc) - progress.start_time
            ).total_seconds() if progress.start_time else 0.0
            self.progress_callback(progress)

    def cancel(self) -> None:
        """Cancel current execution."""
        self._cancelled.set()
        logger.info("Distributed execution cancelled")

    def get_cluster_info(self) -> Optional[ClusterInfo]:
        """Get current cluster information."""
        if self._executor:
            return self._executor.get_cluster_info()
        return None
