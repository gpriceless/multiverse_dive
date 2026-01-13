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

# Tiled runner module exports (Group L, Track 3)
from .tiled_runner import (
    # Enums
    StitchMethod,
    AggregationMethod,
    # Data classes
    TileContext,
    TileResult,
    ProcessingProgress,
    TiledProcessingResult,
    # Runner
    TiledAlgorithmRunner,
    # Stitcher
    ResultStitcher,
    # Utilities
    check_algorithm_tiled_support,
    run_algorithm_tiled,
    estimate_tiles_for_memory,
)

# Dask-based tiled processing (Stream B: Distributed Processing)
from .dask_tiled import (
    # Configuration
    DaskProcessingConfig,
    # Data classes
    TileInfo,
    TileResult as DaskTileResult,
    ProcessingProgress as DaskProcessingProgress,
    DaskProcessingResult,
    # Processor
    DaskTileProcessor,
    # Utilities
    process_with_dask,
    estimate_processing_time,
    get_optimal_config,
)

# Execution Router (Stream B: Distributed Processing)
from .router import (
    # Enums
    ExecutionProfile,
    ResourceLevel,
    # Data classes
    ResourceEstimate,
    SystemResources,
    RoutingConfig,
    RoutingResult,
    # Components
    ResourceEstimator,
    BackendSelector,
    ExecutionRouter,
    # Utilities
    auto_route,
    get_recommended_profile,
    create_router_for_profile,
)

# Algorithm Dask Adapters (Stream B: Distributed Processing)
from .dask_adapters import (
    # Data classes
    TileContext as AdapterTileContext,
    TileResult as AdapterTileResult,
    # Adapters
    DaskAlgorithmAdapter,
    TiledAlgorithmMixin,
    AlgorithmWrapper,
    FloodAlgorithmAdapter,
    WildfireAlgorithmAdapter,
    StormAlgorithmAdapter,
    # Factory functions
    wrap_algorithm_for_dask,
    create_tiled_algorithm,
    adapt_all_algorithms,
    # Validation
    check_algorithm_compatibility,
    validate_adapter,
)

# Sedona Backend (Group C: Distributed Processing - Cloud)
from .sedona_backend import (
    # Enums
    SedonaDeployMode,
    PartitionStrategy,
    CheckpointMode,
    RasterFormat,
    # Configuration
    SedonaConfig,
    # Data classes
    ClusterStatus,
    TilePartition,
    SedonaProcessingResult,
    # Backend
    SedonaBackend,
    SedonaTileProcessor,
    RasterSerializer,
    # Utilities
    create_sedona_backend,
    process_with_sedona,
    is_sedona_available,
    get_sedona_info,
)

# Sedona Adapters (Group C: Distributed Processing - Cloud)
from .sedona_adapters import (
    # Configuration
    AdapterConfig,
    TileData as SedonaTileData,
    TileResult as SedonaTileResult,
    # Adapters
    SedonaAlgorithmAdapter,
    FloodSedonaAdapter,
    WildfireSedonaAdapter,
    StormSedonaAdapter,
    AlgorithmSerializer,
    ResultCollector,
    # Factory functions
    wrap_algorithm_for_sedona,
    adapt_algorithms_for_sedona,
    check_sedona_compatibility,
    validate_sedona_adapter,
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
    # Tiled Runner - Enums
    "StitchMethod",
    "AggregationMethod",
    # Tiled Runner - Data
    "TileContext",
    "TileResult",
    "ProcessingProgress",
    "TiledProcessingResult",
    # Tiled Runner - Classes
    "TiledAlgorithmRunner",
    "ResultStitcher",
    # Tiled Runner - Utilities
    "check_algorithm_tiled_support",
    "run_algorithm_tiled",
    "estimate_tiles_for_memory",
    # Dask Tiled Processing - Config
    "DaskProcessingConfig",
    # Dask Tiled Processing - Data
    "TileInfo",
    "DaskTileResult",
    "DaskProcessingProgress",
    "DaskProcessingResult",
    # Dask Tiled Processing - Processor
    "DaskTileProcessor",
    # Dask Tiled Processing - Utilities
    "process_with_dask",
    "estimate_processing_time",
    "get_optimal_config",
    # Execution Router - Enums
    "ExecutionProfile",
    "ResourceLevel",
    # Execution Router - Data
    "ResourceEstimate",
    "SystemResources",
    "RoutingConfig",
    "RoutingResult",
    # Execution Router - Components
    "ResourceEstimator",
    "BackendSelector",
    "ExecutionRouter",
    # Execution Router - Utilities
    "auto_route",
    "get_recommended_profile",
    "create_router_for_profile",
    # Dask Adapters - Data
    "AdapterTileContext",
    "AdapterTileResult",
    # Dask Adapters - Classes
    "DaskAlgorithmAdapter",
    "TiledAlgorithmMixin",
    "AlgorithmWrapper",
    "FloodAlgorithmAdapter",
    "WildfireAlgorithmAdapter",
    "StormAlgorithmAdapter",
    # Dask Adapters - Factory
    "wrap_algorithm_for_dask",
    "create_tiled_algorithm",
    "adapt_all_algorithms",
    # Dask Adapters - Validation
    "check_algorithm_compatibility",
    "validate_adapter",
    # Sedona Backend - Config
    "SedonaConfig",
    "SedonaDeployMode",
    "PartitionStrategy",
    "CheckpointMode",
    "RasterFormat",
    # Sedona Backend - Data
    "ClusterStatus",
    "TilePartition",
    "SedonaProcessingResult",
    # Sedona Backend - Classes
    "SedonaBackend",
    "SedonaTileProcessor",
    "RasterSerializer",
    # Sedona Backend - Utilities
    "create_sedona_backend",
    "process_with_sedona",
    "is_sedona_available",
    "get_sedona_info",
    # Sedona Adapters - Config
    "AdapterConfig",
    "SedonaTileData",
    "SedonaTileResult",
    # Sedona Adapters - Classes
    "SedonaAlgorithmAdapter",
    "FloodSedonaAdapter",
    "WildfireSedonaAdapter",
    "StormSedonaAdapter",
    "AlgorithmSerializer",
    "ResultCollector",
    # Sedona Adapters - Factory
    "wrap_algorithm_for_sedona",
    "adapt_algorithms_for_sedona",
    "check_sedona_compatibility",
    "validate_sedona_adapter",
]
