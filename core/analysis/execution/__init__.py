"""
Execution Engine for Analysis Pipelines.

This module provides the runtime infrastructure for executing
analysis pipelines, including:

- **Pipeline**: DAG-based task composition and validation
- **PipelineRunner**: Local sequential/parallel execution
- **DistributedPipelineRunner**: Distributed execution via Dask/Ray
- **CheckpointManager**: State persistence and recovery

Basic Usage:

    from core.analysis.execution import (
        Pipeline,
        PipelineTask,
        PipelineRunner,
        ExecutionConfig,
    )

    # Create pipeline
    pipeline = Pipeline("my_pipeline", name="My Analysis Pipeline")
    pipeline.add_task(PipelineTask(
        task_id="load_data",
        executor=lambda ctx: load_data(ctx.get_input("source")),
    ))
    pipeline.add_task(PipelineTask(
        task_id="process",
        executor=lambda ctx: process(ctx.get_input("load_data")),
        dependencies=["load_data"],
    ))

    # Execute
    runner = PipelineRunner()
    result = runner.execute(pipeline, initial_inputs={"source": "data.tif"})

Distributed Execution:

    from core.analysis.execution import (
        DistributedPipelineRunner,
        DistributedConfig,
        DistributedBackend,
    )

    config = DistributedConfig(
        backend=DistributedBackend.DASK,
        n_workers=8,
    )
    runner = DistributedPipelineRunner(distributed_config=config)
    result = runner.execute(pipeline)

Checkpointing:

    from core.analysis.execution import (
        CheckpointManager,
        CheckpointConfig,
    )

    checkpoint_mgr = CheckpointManager(CheckpointConfig(
        base_path=Path("./checkpoints"),
        max_checkpoints=5,
    ))
    runner = PipelineRunner(checkpoint_manager=checkpoint_mgr)
"""

# Runner module exports
from .runner import (
    # Core enums
    TaskStatus,
    RetryPolicy,
    ExecutionMode,
    # Configuration
    RetryConfig,
    ExecutionConfig,
    # Task components
    TaskResult,
    ExecutionContext,
    TaskExecutor,
    PipelineTask,
    # Progress and results
    ExecutionProgress,
    ExecutionResult,
    # Pipeline
    Pipeline,
    # Runner
    PipelineRunner,
    run_pipeline,
)

# Distributed module exports
from .distributed import (
    # Backends
    DistributedBackend,
    # Worker info
    WorkerInfo,
    ClusterInfo,
    # Configuration
    DistributedConfig,
    # Executors
    DistributedExecutorBase,
    LocalExecutor,
    DaskExecutor,
    RayExecutor,
    get_executor,
    # Runner
    DistributedPipelineRunner,
)

# Checkpoint module exports
from .checkpoint import (
    # Enums
    CheckpointStatus,
    CheckpointType,
    # Data classes
    CheckpointMetadata,
    CheckpointData,
    CheckpointConfig,
    # Storage
    CheckpointStorageBackend,
    LocalStorageBackend,
    # Manager
    CheckpointManager,
    AutoCheckpointer,
)

__all__ = [
    # Runner - Enums
    "TaskStatus",
    "RetryPolicy",
    "ExecutionMode",
    # Runner - Config
    "RetryConfig",
    "ExecutionConfig",
    # Runner - Tasks
    "TaskResult",
    "ExecutionContext",
    "TaskExecutor",
    "PipelineTask",
    # Runner - Progress
    "ExecutionProgress",
    "ExecutionResult",
    # Runner - Pipeline
    "Pipeline",
    "PipelineRunner",
    "run_pipeline",
    # Distributed - Backends
    "DistributedBackend",
    "WorkerInfo",
    "ClusterInfo",
    "DistributedConfig",
    # Distributed - Executors
    "DistributedExecutorBase",
    "LocalExecutor",
    "DaskExecutor",
    "RayExecutor",
    "get_executor",
    "DistributedPipelineRunner",
    # Checkpoint - Enums
    "CheckpointStatus",
    "CheckpointType",
    # Checkpoint - Data
    "CheckpointMetadata",
    "CheckpointData",
    "CheckpointConfig",
    # Checkpoint - Storage
    "CheckpointStorageBackend",
    "LocalStorageBackend",
    # Checkpoint - Manager
    "CheckpointManager",
    "AutoCheckpointer",
]
