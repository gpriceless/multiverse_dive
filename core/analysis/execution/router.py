"""
Execution Router for Intelligent Backend Selection.

Routes analysis tasks to the appropriate execution backend based on:
- Data size and complexity
- Available compute resources
- Target performance requirements

Backends:
- Serial: Simple in-memory processing for small datasets (<100 tiles)
- Dask Local: Parallel tile processing for medium datasets (100-1000 tiles)
- Dask Distributed: Cluster processing for large datasets (1000+ tiles)
- Sedona (Future): Apache Sedona on Spark for continental scale

Key Components:
- ExecutionRouter: Main routing logic
- ResourceEstimator: Estimate compute requirements
- BackendSelector: Select optimal backend

Example Usage:
    from core.analysis.execution.router import (
        ExecutionRouter,
        ExecutionProfile,
    )

    # Create router
    router = ExecutionRouter()

    # Route automatically based on data size
    result = router.execute(
        data=input_array,
        algorithm=flood_detector,
        profile=ExecutionProfile.AUTO,
    )

    # Or specify backend explicitly
    result = router.execute(
        data=input_array,
        algorithm=flood_detector,
        profile=ExecutionProfile.DASK_LOCAL,
    )
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Import execution backends
try:
    from .dask_tiled import (
        DaskTileProcessor,
        DaskProcessingConfig,
        DaskProcessingResult,
        ProcessingProgress,
        SchedulerType,
    )
    HAS_DASK_TILED = True
except ImportError:
    HAS_DASK_TILED = False
    DaskTileProcessor = None
    DaskProcessingConfig = None
    DaskProcessingResult = None

try:
    from .tiled_runner import (
        TiledAlgorithmRunner,
        TiledProcessingResult,
    )
    HAS_TILED_RUNNER = True
except ImportError:
    HAS_TILED_RUNNER = False
    TiledAlgorithmRunner = None
    TiledProcessingResult = None

try:
    from .sedona_backend import (
        SedonaBackend,
        SedonaConfig,
        SedonaTileProcessor,
        SedonaProcessingResult,
        is_sedona_available,
    )
    HAS_SEDONA = True
except ImportError:
    HAS_SEDONA = False
    SedonaBackend = None
    SedonaConfig = None
    SedonaTileProcessor = None
    SedonaProcessingResult = None
    is_sedona_available = lambda: False


# =============================================================================
# Enumerations
# =============================================================================


class ExecutionProfile(Enum):
    """Execution profile presets."""

    AUTO = "auto"  # Automatic selection based on data size
    SERIAL = "serial"  # Single-threaded, no tiling
    TILED_SERIAL = "tiled_serial"  # Tiled but single-threaded
    DASK_LOCAL = "dask_local"  # Dask with local cluster
    DASK_DISTRIBUTED = "dask_distributed"  # Dask with remote cluster
    SEDONA = "sedona"  # Apache Sedona on Spark local
    SEDONA_CLUSTER = "sedona_cluster"  # Apache Sedona on Spark cluster

    # Convenience aliases
    LAPTOP = "laptop"  # Optimized for laptop resources
    WORKSTATION = "workstation"  # Optimized for workstation
    CLOUD = "cloud"  # Optimized for cloud execution
    CONTINENTAL = "continental"  # Continental scale (10,000+ tiles)


class ResourceLevel(Enum):
    """Available resource levels."""

    MINIMAL = "minimal"  # < 2GB RAM, 2 cores
    LAPTOP = "laptop"  # 4-8GB RAM, 4 cores
    WORKSTATION = "workstation"  # 16-32GB RAM, 8+ cores
    CLOUD = "cloud"  # Distributed cluster


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ResourceEstimate:
    """
    Estimated resource requirements for processing.

    Attributes:
        memory_gb: Estimated memory requirement in GB
        n_tiles: Number of tiles to process
        estimated_time_serial_seconds: Estimated serial processing time
        estimated_time_parallel_seconds: Estimated parallel processing time
        recommended_backend: Recommended execution backend
        recommended_workers: Recommended number of workers
    """

    memory_gb: float
    n_tiles: int
    estimated_time_serial_seconds: float
    estimated_time_parallel_seconds: float
    recommended_backend: ExecutionProfile
    recommended_workers: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_gb": self.memory_gb,
            "n_tiles": self.n_tiles,
            "estimated_time_serial_seconds": self.estimated_time_serial_seconds,
            "estimated_time_parallel_seconds": self.estimated_time_parallel_seconds,
            "recommended_backend": self.recommended_backend.value,
            "recommended_workers": self.recommended_workers,
        }


@dataclass
class SystemResources:
    """
    Detected system resources.

    Attributes:
        total_memory_gb: Total system memory in GB
        available_memory_gb: Available memory in GB
        cpu_count: Number of CPU cores
        has_gpu: Whether GPU is available
        cluster_address: Address of distributed cluster (if any)
    """

    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    has_gpu: bool = False
    cluster_address: Optional[str] = None
    spark_master: Optional[str] = None
    has_spark: bool = False
    has_sedona: bool = False

    @classmethod
    def detect(cls) -> "SystemResources":
        """Detect system resources."""
        cpu_count = os.cpu_count() or 4

        # Try to get memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_memory_gb = mem.total / (1024**3)
            available_memory_gb = mem.available / (1024**3)
        except ImportError:
            # Fallback estimates
            total_memory_gb = 4.0
            available_memory_gb = 2.0

        # Check for GPU
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        # Check for Spark/Sedona
        has_spark = False
        has_sedona = False
        try:
            from pyspark.sql import SparkSession
            has_spark = True
            try:
                from sedona.register import SedonaRegistrator
                has_sedona = True
            except ImportError:
                pass
        except ImportError:
            pass

        # Also check via our module
        if HAS_SEDONA and is_sedona_available():
            has_sedona = True
            has_spark = True

        return cls(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            has_spark=has_spark,
            has_sedona=has_sedona,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "cpu_count": self.cpu_count,
            "has_gpu": self.has_gpu,
            "cluster_address": self.cluster_address,
            "spark_master": self.spark_master,
            "has_spark": self.has_spark,
            "has_sedona": self.has_sedona,
        }


@dataclass
class RoutingConfig:
    """
    Configuration for execution routing.

    Attributes:
        tile_size: Default tile size in pixels
        overlap: Overlap in pixels
        tile_threshold_serial: Below this, use serial processing
        tile_threshold_dask_local: Below this, use Dask local
        tile_threshold_sedona: Above this, consider Sedona for continental scale
        memory_safety_factor: Safety margin for memory estimates
        target_time_minutes: Target processing time
        prefer_distributed: Prefer distributed even for smaller jobs
        prefer_sedona: Prefer Sedona when available for large datasets
        sedona_master: Spark master URL for Sedona
    """

    tile_size: Tuple[int, int] = (512, 512)
    overlap: int = 32
    tile_threshold_serial: int = 50
    tile_threshold_dask_local: int = 1000
    tile_threshold_sedona: int = 10000  # Continental scale
    memory_safety_factor: float = 2.0
    target_time_minutes: float = 10.0
    prefer_distributed: bool = False
    prefer_sedona: bool = False
    sedona_master: Optional[str] = None

    # Time per tile estimates (seconds)
    time_per_tile_serial: float = 0.5
    time_per_tile_parallel: float = 0.1
    time_per_tile_sedona: float = 0.05  # Sedona is faster at scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tile_threshold_serial": self.tile_threshold_serial,
            "tile_threshold_dask_local": self.tile_threshold_dask_local,
            "tile_threshold_sedona": self.tile_threshold_sedona,
            "memory_safety_factor": self.memory_safety_factor,
            "target_time_minutes": self.target_time_minutes,
            "prefer_distributed": self.prefer_distributed,
            "prefer_sedona": self.prefer_sedona,
            "sedona_master": self.sedona_master,
        }


@dataclass
class RoutingResult:
    """
    Result from execution routing.

    Attributes:
        backend_used: Backend that was used
        result: Processing result
        resource_estimate: Resource estimate used
        actual_time_seconds: Actual processing time
        routing_decision: Explanation of routing decision
    """

    backend_used: ExecutionProfile
    result: Any
    resource_estimate: ResourceEstimate
    actual_time_seconds: float
    routing_decision: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_used": self.backend_used.value,
            "resource_estimate": self.resource_estimate.to_dict(),
            "actual_time_seconds": self.actual_time_seconds,
            "routing_decision": self.routing_decision,
        }


# =============================================================================
# ResourceEstimator
# =============================================================================


class ResourceEstimator:
    """
    Estimate resource requirements for processing.

    Analyzes input data and algorithm requirements to estimate:
    - Memory usage
    - Processing time
    - Optimal parallelization
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        """
        Initialize estimator.

        Args:
            config: Routing configuration
        """
        self.config = config or RoutingConfig()

    def estimate(
        self,
        data_shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        algorithm: Optional[Any] = None,
    ) -> ResourceEstimate:
        """
        Estimate resources for processing.

        Args:
            data_shape: Shape of input data
            dtype: Data type
            algorithm: Algorithm to use (for complexity hints)

        Returns:
            ResourceEstimate with recommendations
        """
        # Calculate data dimensions
        if len(data_shape) == 2:
            n_bands = 1
            height, width = data_shape
        else:
            n_bands = data_shape[0]
            height = data_shape[1]
            width = data_shape[2]

        # Calculate number of tiles
        tile_h, tile_w = self.config.tile_size
        n_rows = int(np.ceil(height / tile_h))
        n_cols = int(np.ceil(width / tile_w))
        n_tiles = n_rows * n_cols

        # Estimate memory per tile
        bytes_per_element = np.dtype(dtype).itemsize
        tile_memory_bytes = (
            n_bands * tile_h * tile_w * bytes_per_element *
            self.config.memory_safety_factor
        )
        tile_memory_gb = tile_memory_bytes / (1024**3)

        # Estimate total memory requirement
        # Need at least one tile per worker plus output buffer
        n_workers = min(n_tiles, os.cpu_count() or 4)
        memory_gb = (
            tile_memory_gb * (n_workers + 1) +  # Active tiles
            (n_bands * height * width * bytes_per_element / (1024**3))  # Output buffer
        )

        # Estimate processing time
        # Check algorithm complexity if available
        complexity_factor = 1.0
        if algorithm is not None:
            if hasattr(algorithm, "METADATA"):
                meta = algorithm.METADATA
                compute_reqs = meta.get("requirements", {}).get("compute", {})
                if compute_reqs.get("gpu"):
                    complexity_factor = 0.1  # GPU is much faster
                elif compute_reqs.get("memory_gb", 4) > 8:
                    complexity_factor = 2.0  # Memory-intensive

        time_serial = n_tiles * self.config.time_per_tile_serial * complexity_factor
        time_parallel = (
            (n_tiles / n_workers) *
            self.config.time_per_tile_parallel *
            complexity_factor
        )

        # Determine recommended backend
        if n_tiles < self.config.tile_threshold_serial:
            recommended_backend = ExecutionProfile.TILED_SERIAL
        elif n_tiles < self.config.tile_threshold_dask_local:
            recommended_backend = ExecutionProfile.DASK_LOCAL
        else:
            recommended_backend = ExecutionProfile.DASK_DISTRIBUTED

        # Override if distributed is preferred
        if self.config.prefer_distributed and n_tiles > 10:
            recommended_backend = ExecutionProfile.DASK_LOCAL

        return ResourceEstimate(
            memory_gb=memory_gb,
            n_tiles=n_tiles,
            estimated_time_serial_seconds=time_serial,
            estimated_time_parallel_seconds=time_parallel,
            recommended_backend=recommended_backend,
            recommended_workers=n_workers,
        )


# =============================================================================
# BackendSelector
# =============================================================================


class BackendSelector:
    """
    Select optimal backend based on resources and requirements.

    Considers:
    - Available system resources
    - Data size and complexity
    - Target performance requirements
    - Available backends
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        system_resources: Optional[SystemResources] = None,
    ):
        """
        Initialize selector.

        Args:
            config: Routing configuration
            system_resources: System resources (auto-detected if None)
        """
        self.config = config or RoutingConfig()
        self.resources = system_resources or SystemResources.detect()

    def select(
        self,
        estimate: ResourceEstimate,
        profile: ExecutionProfile = ExecutionProfile.AUTO,
    ) -> Tuple[ExecutionProfile, str]:
        """
        Select execution backend.

        Args:
            estimate: Resource estimate
            profile: Requested profile (AUTO = automatic selection)

        Returns:
            Tuple of (selected profile, decision explanation)
        """
        # If explicit profile requested (not AUTO or aliases), validate and return
        if profile not in [ExecutionProfile.AUTO, ExecutionProfile.LAPTOP,
                          ExecutionProfile.WORKSTATION, ExecutionProfile.CLOUD,
                          ExecutionProfile.CONTINENTAL]:
            return self._validate_profile(profile, estimate)

        # Map convenience aliases
        if profile == ExecutionProfile.LAPTOP:
            return ExecutionProfile.DASK_LOCAL, "Using Dask local for laptop profile"
        elif profile == ExecutionProfile.WORKSTATION:
            return ExecutionProfile.DASK_LOCAL, "Using Dask local for workstation profile"
        elif profile == ExecutionProfile.CLOUD:
            if self.resources.spark_master or self.resources.has_spark:
                return ExecutionProfile.SEDONA_CLUSTER, "Using Sedona on Spark cluster"
            if self.resources.cluster_address:
                return ExecutionProfile.DASK_DISTRIBUTED, "Using distributed Dask cluster"
            return ExecutionProfile.DASK_LOCAL, "Using Dask local (no cluster available)"
        elif profile == ExecutionProfile.CONTINENTAL:
            # Continental scale - prefer Sedona
            if self.resources.has_spark or HAS_SEDONA:
                if self.resources.spark_master:
                    return ExecutionProfile.SEDONA_CLUSTER, "Using Sedona cluster for continental scale"
                return ExecutionProfile.SEDONA, "Using Sedona local for continental scale"
            return ExecutionProfile.DASK_LOCAL, "Using Dask local for continental scale (Sedona not available)"

        # Automatic selection
        return self._auto_select(estimate)

    def _auto_select(
        self,
        estimate: ResourceEstimate,
    ) -> Tuple[ExecutionProfile, str]:
        """Automatically select best backend."""
        n_tiles = estimate.n_tiles
        memory_req = estimate.memory_gb
        available_mem = self.resources.available_memory_gb

        # Check memory constraint
        if memory_req > available_mem * 0.8:
            # Need to be more conservative
            return (
                ExecutionProfile.TILED_SERIAL,
                f"Memory constrained ({memory_req:.1f}GB required, "
                f"{available_mem:.1f}GB available) - using tiled serial"
            )

        # Select based on tile count
        if n_tiles < self.config.tile_threshold_serial:
            if n_tiles < 10:
                return (
                    ExecutionProfile.SERIAL,
                    f"Small dataset ({n_tiles} tiles) - using serial processing"
                )
            return (
                ExecutionProfile.TILED_SERIAL,
                f"Small-medium dataset ({n_tiles} tiles) - using tiled serial"
            )

        if n_tiles < self.config.tile_threshold_dask_local:
            return (
                ExecutionProfile.DASK_LOCAL,
                f"Medium dataset ({n_tiles} tiles) - using Dask local cluster"
            )

        # Check for continental scale (Sedona preferred)
        if n_tiles >= self.config.tile_threshold_sedona:
            if self.config.prefer_sedona or n_tiles >= self.config.tile_threshold_sedona * 2:
                # Use Sedona for very large datasets
                if self.resources.spark_master:
                    return (
                        ExecutionProfile.SEDONA_CLUSTER,
                        f"Continental scale ({n_tiles} tiles) - using Sedona cluster"
                    )
                if self.resources.has_spark or HAS_SEDONA:
                    return (
                        ExecutionProfile.SEDONA,
                        f"Continental scale ({n_tiles} tiles) - using Sedona local"
                    )

        # Large dataset
        if self.resources.cluster_address:
            return (
                ExecutionProfile.DASK_DISTRIBUTED,
                f"Large dataset ({n_tiles} tiles) - using distributed cluster"
            )

        return (
            ExecutionProfile.DASK_LOCAL,
            f"Large dataset ({n_tiles} tiles) - using Dask local (no cluster)"
        )

    def _validate_profile(
        self,
        profile: ExecutionProfile,
        estimate: ResourceEstimate,
    ) -> Tuple[ExecutionProfile, str]:
        """Validate requested profile is appropriate."""
        # Check if distributed is requested but no cluster
        if profile == ExecutionProfile.DASK_DISTRIBUTED:
            if not self.resources.cluster_address:
                return (
                    ExecutionProfile.DASK_LOCAL,
                    "Distributed requested but no cluster - falling back to local"
                )

        # Check if Sedona requested
        if profile == ExecutionProfile.SEDONA:
            if not HAS_SEDONA and not self.resources.has_spark:
                return (
                    ExecutionProfile.DASK_LOCAL,
                    "Sedona requested but Spark not available - falling back to Dask local"
                )
            return profile, "Using Sedona local for processing"

        # Check if Sedona cluster requested
        if profile == ExecutionProfile.SEDONA_CLUSTER:
            if not self.resources.spark_master:
                if HAS_SEDONA or self.resources.has_spark:
                    return (
                        ExecutionProfile.SEDONA,
                        "Sedona cluster requested but no Spark master - using Sedona local"
                    )
                return (
                    ExecutionProfile.DASK_LOCAL,
                    "Sedona cluster requested but Spark not available - falling back to Dask local"
                )
            return profile, f"Using Sedona cluster at {self.resources.spark_master}"

        return profile, f"Using requested profile: {profile.value}"


# =============================================================================
# ExecutionRouter
# =============================================================================


class ExecutionRouter:
    """
    Routes execution to appropriate backend.

    Main entry point for intelligent backend selection and execution.

    Example:
        router = ExecutionRouter()

        # Automatic routing
        result = router.execute(
            data=sar_image,
            algorithm=ThresholdSARAlgorithm(),
        )

        # With specific profile
        result = router.execute(
            data=sar_image,
            algorithm=ThresholdSARAlgorithm(),
            profile=ExecutionProfile.DASK_LOCAL,
        )

        # Get resource estimate without executing
        estimate = router.estimate_resources(data_shape)
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ):
        """
        Initialize router.

        Args:
            config: Routing configuration
            progress_callback: Callback for progress updates
        """
        self.config = config or RoutingConfig()
        self.progress_callback = progress_callback
        self.estimator = ResourceEstimator(self.config)
        self.resources = SystemResources.detect()
        self.selector = BackendSelector(self.config, self.resources)

        logger.info(
            f"ExecutionRouter initialized: "
            f"{self.resources.cpu_count} cores, "
            f"{self.resources.available_memory_gb:.1f}GB available"
        )

    def execute(
        self,
        data: np.ndarray,
        algorithm: Any,
        profile: ExecutionProfile = ExecutionProfile.AUTO,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> RoutingResult:
        """
        Execute processing with automatic backend selection.

        Args:
            data: Input data array
            algorithm: Algorithm instance
            profile: Execution profile (AUTO for automatic selection)
            bounds: Geographic bounds
            resolution: Pixel resolution
            output_path: Optional output path

        Returns:
            RoutingResult with processing result and metadata
        """
        start_time = time.time()

        # Estimate resources
        estimate = self.estimator.estimate(
            data_shape=data.shape,
            dtype=data.dtype,
            algorithm=algorithm,
        )

        # Select backend
        selected_profile, decision = self.selector.select(estimate, profile)

        logger.info(
            f"Routing decision: {decision} "
            f"({estimate.n_tiles} tiles, {estimate.memory_gb:.1f}GB memory)"
        )

        # Execute with selected backend
        result = self._execute_with_backend(
            data=data,
            algorithm=algorithm,
            backend=selected_profile,
            bounds=bounds,
            resolution=resolution,
            output_path=output_path,
        )

        actual_time = time.time() - start_time

        return RoutingResult(
            backend_used=selected_profile,
            result=result,
            resource_estimate=estimate,
            actual_time_seconds=actual_time,
            routing_decision=decision,
        )

    def estimate_resources(
        self,
        data_shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        algorithm: Optional[Any] = None,
    ) -> ResourceEstimate:
        """
        Estimate resources without executing.

        Args:
            data_shape: Shape of input data
            dtype: Data type
            algorithm: Algorithm (for complexity hints)

        Returns:
            ResourceEstimate
        """
        return self.estimator.estimate(data_shape, dtype, algorithm)

    def recommend_profile(
        self,
        data_shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        algorithm: Optional[Any] = None,
    ) -> Tuple[ExecutionProfile, str]:
        """
        Recommend execution profile without executing.

        Args:
            data_shape: Shape of input data
            dtype: Data type
            algorithm: Algorithm (for complexity hints)

        Returns:
            Tuple of (recommended profile, explanation)
        """
        estimate = self.estimate_resources(data_shape, dtype, algorithm)
        return self.selector.select(estimate, ExecutionProfile.AUTO)

    def _execute_with_backend(
        self,
        data: np.ndarray,
        algorithm: Any,
        backend: ExecutionProfile,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
        output_path: Optional[Union[str, Path]],
    ) -> Any:
        """Execute with the specified backend."""

        if backend == ExecutionProfile.SERIAL:
            return self._execute_serial(data, algorithm)

        elif backend == ExecutionProfile.TILED_SERIAL:
            return self._execute_tiled_serial(
                data, algorithm, bounds, resolution
            )

        elif backend in [ExecutionProfile.DASK_LOCAL, ExecutionProfile.DASK_DISTRIBUTED]:
            return self._execute_dask(
                data, algorithm, backend, bounds, resolution, output_path
            )

        elif backend in [ExecutionProfile.SEDONA, ExecutionProfile.SEDONA_CLUSTER]:
            return self._execute_sedona(
                data, algorithm, backend, bounds, resolution, output_path
            )

        else:
            # Fallback to tiled serial
            logger.warning(f"Unknown backend {backend}, using tiled serial")
            return self._execute_tiled_serial(
                data, algorithm, bounds, resolution
            )

    def _execute_serial(
        self,
        data: np.ndarray,
        algorithm: Any,
    ) -> Any:
        """Execute without tiling."""
        logger.info("Executing serial (no tiling)")

        if hasattr(algorithm, "execute"):
            return algorithm.execute(data.squeeze())
        elif hasattr(algorithm, "run"):
            return algorithm.run(data=data)
        elif callable(algorithm):
            return algorithm(data)
        else:
            raise ValueError(
                f"Algorithm {type(algorithm)} has no recognized processing method"
            )

    def _execute_tiled_serial(
        self,
        data: np.ndarray,
        algorithm: Any,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
    ) -> Any:
        """Execute with tiling but serial processing."""
        logger.info("Executing tiled serial")

        if HAS_TILED_RUNNER:
            runner = TiledAlgorithmRunner(
                algorithm=algorithm,
                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
                parallel=False,
            )
            return runner.process(
                data=data,
                bounds=bounds,
                resolution=resolution,
            )
        else:
            # Fallback to serial
            return self._execute_serial(data, algorithm)

    def _execute_dask(
        self,
        data: np.ndarray,
        algorithm: Any,
        backend: ExecutionProfile,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
        output_path: Optional[Union[str, Path]],
    ) -> Any:
        """Execute with Dask backend."""
        logger.info(f"Executing with {backend.value}")

        if not HAS_DASK_TILED:
            logger.warning("Dask tiled not available, falling back to serial")
            return self._execute_tiled_serial(
                data, algorithm, bounds, resolution
            )

        # Configure Dask
        if backend == ExecutionProfile.DASK_DISTRIBUTED:
            if self.resources.cluster_address:
                config = DaskProcessingConfig.for_cluster(
                    self.resources.cluster_address,
                    n_workers=self.resources.cpu_count,
                )
            else:
                config = DaskProcessingConfig.for_workstation(
                    self.resources.available_memory_gb
                )
        else:
            # Local Dask
            if self.resources.available_memory_gb > 8:
                config = DaskProcessingConfig.for_workstation(
                    self.resources.available_memory_gb
                )
            else:
                config = DaskProcessingConfig.for_laptop(
                    self.resources.available_memory_gb
                )

        # Override tile size from routing config
        config.tile_size = self.config.tile_size
        config.overlap = self.config.overlap

        # Create processor
        processor = DaskTileProcessor(
            config=config,
            progress_callback=self.progress_callback,
        )

        return processor.process(
            data=data,
            algorithm=algorithm,
            bounds=bounds,
            resolution=resolution,
            output_path=output_path,
        )

    def _execute_sedona(
        self,
        data: np.ndarray,
        algorithm: Any,
        backend: ExecutionProfile,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
        output_path: Optional[Union[str, Path]],
    ) -> Any:
        """Execute with Sedona/Spark backend."""
        logger.info(f"Executing with {backend.value}")

        if not HAS_SEDONA:
            logger.warning("Sedona not available, falling back to Dask")
            return self._execute_dask(
                data, algorithm, ExecutionProfile.DASK_LOCAL,
                bounds, resolution, output_path
            )

        # Configure Sedona
        if backend == ExecutionProfile.SEDONA_CLUSTER:
            master = self.config.sedona_master or self.resources.spark_master
            if master:
                sedona_config = SedonaConfig.for_cluster(
                    master=master,
                    num_executors=max(4, self.resources.cpu_count),
                )
            else:
                # Fallback to local Spark
                sedona_config = SedonaConfig.for_local_testing()
        else:
            # Local Sedona
            sedona_config = SedonaConfig(
                master=f"local[{self.resources.cpu_count}]",
                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
            )

        # Override tile size from routing config
        sedona_config.tile_size = self.config.tile_size
        sedona_config.overlap = self.config.overlap

        # Create backend and processor
        sedona_backend = SedonaBackend(sedona_config)

        try:
            sedona_backend.connect()

            processor = SedonaTileProcessor(
                backend=sedona_backend,
                progress_callback=self._sedona_progress_adapter,
            )

            result = processor.process(
                data=data,
                algorithm=algorithm,
                bounds=bounds,
                resolution=resolution,
                output_path=str(output_path) if output_path else None,
            )

            # Return mosaic from result (SedonaProcessingResult)
            # The result contains mosaic data in the output file if output_path was given
            if hasattr(result, 'success') and result.success:
                return result
            return result

        finally:
            sedona_backend.disconnect()

    def _sedona_progress_adapter(self, progress_dict: Dict) -> None:
        """Adapt Sedona progress to standard callback."""
        if self.progress_callback is not None:
            # Convert to ProcessingProgress if needed
            try:
                from .dask_tiled import ProcessingProgress
                progress = ProcessingProgress(
                    total_tiles=progress_dict.get("total_tiles", 0),
                    completed_tiles=progress_dict.get("completed_tiles", 0),
                    failed_tiles=progress_dict.get("failed_tiles", 0),
                )
                self.progress_callback(progress)
            except Exception:
                # Progress callback failed, continue silently
                pass

    def get_system_info(self) -> Dict[str, Any]:
        """Get system resource information."""
        return self.resources.to_dict()

    def get_config(self) -> Dict[str, Any]:
        """Get routing configuration."""
        return self.config.to_dict()


# =============================================================================
# Convenience Functions
# =============================================================================


def auto_route(
    data: np.ndarray,
    algorithm: Any,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution: Optional[Tuple[float, float]] = None,
    progress_callback: Optional[Callable] = None,
) -> Any:
    """
    Convenience function for automatic routing.

    Args:
        data: Input data array
        algorithm: Algorithm instance
        bounds: Geographic bounds
        resolution: Pixel resolution
        progress_callback: Progress callback

    Returns:
        Processing result
    """
    router = ExecutionRouter(progress_callback=progress_callback)
    routing_result = router.execute(
        data=data,
        algorithm=algorithm,
        bounds=bounds,
        resolution=resolution,
    )
    return routing_result.result


def get_recommended_profile(
    data_shape: Tuple[int, ...],
    available_memory_gb: Optional[float] = None,
) -> ExecutionProfile:
    """
    Get recommended execution profile.

    Args:
        data_shape: Shape of input data
        available_memory_gb: Available memory (auto-detected if None)

    Returns:
        Recommended ExecutionProfile
    """
    router = ExecutionRouter()
    profile, _ = router.recommend_profile(data_shape)
    return profile


def create_router_for_profile(
    profile: ExecutionProfile,
) -> ExecutionRouter:
    """
    Create router optimized for a specific profile.

    Args:
        profile: Target execution profile

    Returns:
        Configured ExecutionRouter
    """
    if profile == ExecutionProfile.LAPTOP:
        config = RoutingConfig(
            tile_size=(512, 512),
            overlap=32,
            tile_threshold_serial=20,
            tile_threshold_dask_local=200,
        )
    elif profile == ExecutionProfile.WORKSTATION:
        config = RoutingConfig(
            tile_size=(1024, 1024),
            overlap=64,
            tile_threshold_serial=10,
            tile_threshold_dask_local=500,
        )
    elif profile == ExecutionProfile.CLOUD:
        config = RoutingConfig(
            tile_size=(1024, 1024),
            overlap=64,
            tile_threshold_serial=10,
            tile_threshold_dask_local=100,
            prefer_distributed=True,
        )
    else:
        config = RoutingConfig()

    return ExecutionRouter(config=config)
