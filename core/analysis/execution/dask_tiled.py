"""
Dask-Based Tiled Processing for Parallel Raster Analysis.

Enables parallel tile processing using Dask for:
- Local parallelization on laptops/workstations (4-8x speedup)
- Distributed processing on Dask clusters
- Memory-efficient streaming of large datasets

Key Components:
- DaskTileProcessor: Main class for parallel tile processing
- TileTask: Represents a single tile processing task
- DaskProgressTracker: Progress tracking across distributed workers

Performance Target:
- 100km flood analysis in <10 minutes on laptop (vs 30+ min serial)
- 80%+ CPU utilization on multi-core systems

Example Usage:
    from core.analysis.execution.dask_tiled import (
        DaskTileProcessor,
        DaskProcessingConfig,
    )

    # Create processor
    processor = DaskTileProcessor(
        config=DaskProcessingConfig(
            n_workers=4,
            tile_size=(512, 512),
            memory_limit_per_worker="2GB",
        )
    )

    # Process data with algorithm
    result = processor.process(
        data=input_array,
        algorithm=flood_detector,
        bounds=aoi_bounds,
    )

    # Or use virtual raster index
    result = processor.process_vrt(
        vrt_index=virtual_index,
        algorithm=flood_detector,
        output_path="flood_result.tif",
    )
"""

import gc
import hashlib
import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import dask
    import dask.array as da
    from dask import delayed
    from dask.distributed import Client, LocalCluster, progress, as_completed
    from dask.diagnostics import ProgressBar
    HAS_DASK = True
except ImportError:
    dask = None
    da = None
    delayed = None
    Client = None
    LocalCluster = None
    HAS_DASK = False
    logger.warning("Dask not available - parallel processing disabled")

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    Window = None
    HAS_RASTERIO = False


# =============================================================================
# Enumerations
# =============================================================================


class ProcessingBackend(Enum):
    """Backend for tile processing."""

    SERIAL = "serial"  # Single-threaded processing
    THREADING = "threading"  # Thread pool
    DASK_LOCAL = "dask_local"  # Dask with local cluster
    DASK_DISTRIBUTED = "dask_distributed"  # Dask with remote cluster


class SchedulerType(Enum):
    """Dask scheduler type."""

    SYNCHRONOUS = "synchronous"  # Serial execution (debugging)
    THREADS = "threads"  # ThreadPoolExecutor
    PROCESSES = "processes"  # ProcessPoolExecutor
    DISTRIBUTED = "distributed"  # Distributed cluster


class TileStatus(Enum):
    """Status of a tile task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BlendMode(Enum):
    """Modes for blending overlapping tile results."""

    REPLACE = "replace"  # Later tiles replace earlier
    FEATHER = "feather"  # Linear feather blend
    AVERAGE = "average"  # Average overlapping regions
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DaskProcessingConfig:
    """
    Configuration for Dask-based tile processing.

    Attributes:
        n_workers: Number of worker processes/threads
        threads_per_worker: Threads per worker (for distributed)
        memory_limit_per_worker: Memory limit per worker (e.g., "2GB")
        tile_size: Tile size in pixels (height, width)
        overlap: Overlap in pixels for edge handling
        blend_mode: Method for blending overlapping results
        scheduler: Dask scheduler type
        scheduler_address: Address of distributed scheduler (optional)
        dashboard_port: Port for Dask dashboard (0 = disabled)
        retry_failed_tiles: Retry failed tiles
        max_retries: Maximum retries per tile
        progress_interval_seconds: Progress reporting interval
        preserve_dtype: Preserve input data type in output
        chunk_size_mb: Target chunk size in MB for memory management
    """

    n_workers: int = 4
    threads_per_worker: int = 1
    memory_limit_per_worker: str = "2GB"
    tile_size: Tuple[int, int] = (512, 512)
    overlap: int = 32
    blend_mode: BlendMode = BlendMode.FEATHER
    scheduler: SchedulerType = SchedulerType.THREADS
    scheduler_address: Optional[str] = None
    dashboard_port: int = 0
    retry_failed_tiles: bool = True
    max_retries: int = 2
    progress_interval_seconds: float = 1.0
    preserve_dtype: bool = True
    chunk_size_mb: float = 64.0

    def __post_init__(self):
        """Validate configuration."""
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.tile_size[0] < 64 or self.tile_size[1] < 64:
            raise ValueError(f"tile_size must be >= 64, got {self.tile_size}")

    @classmethod
    def for_laptop(cls, memory_gb: float = 4.0) -> "DaskProcessingConfig":
        """Create config optimized for laptop with limited memory."""
        n_workers = max(1, os.cpu_count() or 4)
        memory_per_worker = f"{max(1, memory_gb / n_workers):.1f}GB"

        return cls(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit_per_worker=memory_per_worker,
            tile_size=(512, 512),
            overlap=32,
            scheduler=SchedulerType.THREADS,
            dashboard_port=0,
        )

    @classmethod
    def for_workstation(cls, memory_gb: float = 16.0) -> "DaskProcessingConfig":
        """Create config optimized for workstation."""
        n_workers = max(1, os.cpu_count() or 8)
        memory_per_worker = f"{max(2, memory_gb / n_workers):.1f}GB"

        return cls(
            n_workers=n_workers,
            threads_per_worker=2,
            memory_limit_per_worker=memory_per_worker,
            tile_size=(1024, 1024),
            overlap=64,
            scheduler=SchedulerType.THREADS,
            dashboard_port=8787,
        )

    @classmethod
    def for_cluster(
        cls,
        scheduler_address: str,
        n_workers: int = 10,
    ) -> "DaskProcessingConfig":
        """Create config for distributed cluster."""
        return cls(
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit_per_worker="4GB",
            tile_size=(1024, 1024),
            overlap=64,
            scheduler=SchedulerType.DISTRIBUTED,
            scheduler_address=scheduler_address,
            dashboard_port=8787,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_workers": self.n_workers,
            "threads_per_worker": self.threads_per_worker,
            "memory_limit_per_worker": self.memory_limit_per_worker,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "blend_mode": self.blend_mode.value,
            "scheduler": self.scheduler.value,
            "scheduler_address": self.scheduler_address,
            "dashboard_port": self.dashboard_port,
            "retry_failed_tiles": self.retry_failed_tiles,
            "max_retries": self.max_retries,
        }


@dataclass
class TileInfo:
    """
    Information about a tile to process.

    Attributes:
        col: Column index in grid
        row: Row index in grid
        pixel_bounds: Pixel bounds (col_start, row_start, col_end, row_end)
        geo_bounds: Geographic bounds (minx, miny, maxx, maxy) if available
        overlap_pixel_bounds: Bounds including overlap
        status: Processing status
        retry_count: Number of retries attempted
    """

    col: int
    row: int
    pixel_bounds: Tuple[int, int, int, int]
    geo_bounds: Optional[Tuple[float, float, float, float]] = None
    overlap_pixel_bounds: Optional[Tuple[int, int, int, int]] = None
    status: TileStatus = TileStatus.PENDING
    retry_count: int = 0

    @property
    def index(self) -> Tuple[int, int]:
        """Get tile index."""
        return (self.col, self.row)

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.pixel_bounds[2] - self.pixel_bounds[0]

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.pixel_bounds[3] - self.pixel_bounds[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "col": self.col,
            "row": self.row,
            "pixel_bounds": self.pixel_bounds,
            "geo_bounds": self.geo_bounds,
            "status": self.status.value,
            "retry_count": self.retry_count,
        }


@dataclass
class TileResult:
    """
    Result from processing a single tile.

    Attributes:
        tile_info: Information about the tile
        data: Result data array
        confidence: Optional confidence array
        statistics: Per-tile statistics
        processing_time_seconds: Time to process
        error: Error message if failed
    """

    tile_info: TileInfo
    data: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether processing succeeded."""
        return self.data is not None and self.error is None


@dataclass
class ProcessingProgress:
    """
    Progress information for distributed processing.

    Attributes:
        total_tiles: Total number of tiles
        completed_tiles: Successfully completed tiles
        failed_tiles: Failed tiles
        pending_tiles: Pending tiles
        current_tiles: Currently processing tiles
        elapsed_seconds: Total elapsed time
        estimated_remaining_seconds: Estimated time remaining
        tiles_per_second: Processing throughput
    """

    total_tiles: int = 0
    completed_tiles: int = 0
    failed_tiles: int = 0
    pending_tiles: int = 0
    current_tiles: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    tiles_per_second: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Progress as percentage."""
        if self.total_tiles == 0:
            return 0.0
        return (self.completed_tiles + self.failed_tiles) / self.total_tiles * 100

    @property
    def is_complete(self) -> bool:
        """Whether all tiles are processed."""
        return (self.completed_tiles + self.failed_tiles) >= self.total_tiles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tiles": self.total_tiles,
            "completed_tiles": self.completed_tiles,
            "failed_tiles": self.failed_tiles,
            "pending_tiles": self.pending_tiles,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "tiles_per_second": self.tiles_per_second,
        }


@dataclass
class DaskProcessingResult:
    """
    Complete result from Dask-based processing.

    Attributes:
        mosaic: Stitched result mosaic array
        confidence_mosaic: Optional confidence mosaic
        tile_results: Individual tile results
        statistics: Aggregated statistics
        progress: Final progress state
        processing_time_seconds: Total processing time
        metadata: Additional metadata
    """

    mosaic: np.ndarray
    confidence_mosaic: Optional[np.ndarray] = None
    tile_results: Dict[Tuple[int, int], TileResult] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    progress: Optional[ProcessingProgress] = None
    processing_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def aggregate_statistics(self) -> Dict[str, float]:
        """Aggregate statistics across all tiles."""
        if not self.tile_results:
            return {}

        # Collect all keys
        all_keys: Set[str] = set()
        for result in self.tile_results.values():
            all_keys.update(result.statistics.keys())

        aggregated = {}
        for key in all_keys:
            values = [
                r.statistics.get(key)
                for r in self.tile_results.values()
                if r.statistics.get(key) is not None
            ]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_sum"] = np.sum(values)

        return aggregated


# =============================================================================
# Protocol for Tiled Algorithms
# =============================================================================


class TiledAlgorithmProtocol(Protocol):
    """Protocol for algorithms that support tiled processing."""

    def process_tile(
        self,
        data: np.ndarray,
        tile_info: TileInfo,
    ) -> np.ndarray:
        """Process a single tile."""
        ...


# =============================================================================
# DaskTileProcessor
# =============================================================================


class DaskTileProcessor:
    """
    Parallel tile processor using Dask.

    Processes raster data in tiles using Dask for parallelization.
    Supports local threading, local processes, and distributed clusters.

    Example:
        # Create processor
        processor = DaskTileProcessor(
            config=DaskProcessingConfig.for_laptop()
        )

        # Process with algorithm
        result = processor.process(
            data=sar_image,
            algorithm=ThresholdSARAlgorithm(),
        )

        # Access results
        flood_mask = result.mosaic
        print(f"Processing time: {result.processing_time_seconds:.1f}s")
    """

    def __init__(
        self,
        config: Optional[DaskProcessingConfig] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ):
        """
        Initialize DaskTileProcessor.

        Args:
            config: Processing configuration
            progress_callback: Callback for progress updates
        """
        self.config = config or DaskProcessingConfig()
        self.progress_callback = progress_callback

        self._client: Optional[Any] = None
        self._cluster: Optional[Any] = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

        if not HAS_DASK:
            logger.warning(
                "Dask not available - falling back to serial processing"
            )

    def process(
        self,
        data: np.ndarray,
        algorithm: Any,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> DaskProcessingResult:
        """
        Process data using parallel tile processing.

        Args:
            data: Input data array (height, width) or (bands, height, width)
            algorithm: Algorithm instance with process_tile or run method
            bounds: Geographic bounds (optional)
            resolution: Pixel resolution (optional)
            output_path: Optional output path for result

        Returns:
            DaskProcessingResult with mosaic and statistics
        """
        self._cancelled.clear()
        start_time = time.time()

        # Normalize input shape
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        n_bands, height, width = data.shape

        # Generate tile grid
        tiles = self._generate_tiles(height, width, bounds, resolution)

        # Initialize progress
        progress = ProcessingProgress(
            total_tiles=len(tiles),
            pending_tiles=len(tiles),
        )
        self._report_progress(progress)

        # Process tiles
        if HAS_DASK and self.config.scheduler != SchedulerType.SYNCHRONOUS:
            tile_results = self._process_with_dask(data, algorithm, tiles, progress)
        else:
            tile_results = self._process_serial(data, algorithm, tiles, progress)

        # Stitch results
        mosaic, confidence_mosaic = self._stitch_results(
            tile_results, (height, width)
        )

        # Calculate total time
        processing_time = time.time() - start_time

        # Build result
        result = DaskProcessingResult(
            mosaic=mosaic.squeeze(),
            confidence_mosaic=confidence_mosaic.squeeze() if confidence_mosaic is not None else None,
            tile_results=tile_results,
            progress=progress,
            processing_time_seconds=processing_time,
            metadata={
                "config": self.config.to_dict(),
                "n_tiles": len(tiles),
                "input_shape": (n_bands, height, width),
            },
        )
        result.statistics = result.aggregate_statistics()

        # Save if output path provided
        if output_path and HAS_RASTERIO:
            self._save_result(result, output_path, bounds, resolution)

        logger.info(
            f"Processing complete: {len(tiles)} tiles in {processing_time:.1f}s "
            f"({len(tiles)/processing_time:.1f} tiles/sec)"
        )

        return result

    def process_vrt(
        self,
        vrt_path: Union[str, Path],
        algorithm: Any,
        output_path: Union[str, Path],
        bands: Optional[List[int]] = None,
    ) -> DaskProcessingResult:
        """
        Process a Virtual Raster Index.

        Args:
            vrt_path: Path to VRT file
            algorithm: Algorithm instance
            output_path: Output path for result
            bands: Band indices to process (1-based)

        Returns:
            DaskProcessingResult
        """
        from core.data.ingestion.virtual_index import VirtualRasterIndex

        # Open VRT
        index = VirtualRasterIndex(vrt_path)

        # Get as Dask array
        dask_array = index.as_dask_array(
            chunks=(index.n_bands, *self.config.tile_size)
        )

        if dask_array is None:
            # Fallback to loading entire array
            logger.warning("Could not create Dask array, loading full data")
            data = index.read_region(
                bounds=index.bounds,
                shape=index.shape,
                bands=bands,
            )
            return self.process(
                data=data,
                algorithm=algorithm,
                bounds=index.bounds,
                resolution=index.resolution,
                output_path=output_path,
            )

        # Process Dask array
        return self._process_dask_array(
            dask_array,
            algorithm,
            index.bounds,
            index.resolution,
            output_path,
        )

    def _generate_tiles(
        self,
        height: int,
        width: int,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
    ) -> List[TileInfo]:
        """Generate tile grid."""
        tile_h, tile_w = self.config.tile_size
        overlap = self.config.overlap

        tiles = []
        n_rows = int(np.ceil(height / tile_h))
        n_cols = int(np.ceil(width / tile_w))

        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate pixel bounds (without overlap)
                col_start = col * tile_w
                row_start = row * tile_h
                col_end = min(col_start + tile_w, width)
                row_end = min(row_start + tile_h, height)

                # Calculate overlap bounds
                overlap_col_start = max(0, col_start - overlap)
                overlap_row_start = max(0, row_start - overlap)
                overlap_col_end = min(width, col_end + overlap)
                overlap_row_end = min(height, row_end + overlap)

                # Calculate geographic bounds if available
                geo_bounds = None
                if bounds and resolution:
                    minx = bounds[0] + col_start * resolution[0]
                    maxy = bounds[3] - row_start * resolution[1]
                    maxx = bounds[0] + col_end * resolution[0]
                    miny = bounds[3] - row_end * resolution[1]
                    geo_bounds = (minx, miny, maxx, maxy)

                tiles.append(TileInfo(
                    col=col,
                    row=row,
                    pixel_bounds=(col_start, row_start, col_end, row_end),
                    geo_bounds=geo_bounds,
                    overlap_pixel_bounds=(
                        overlap_col_start, overlap_row_start,
                        overlap_col_end, overlap_row_end
                    ),
                ))

        return tiles

    def _process_with_dask(
        self,
        data: np.ndarray,
        algorithm: Any,
        tiles: List[TileInfo],
        progress: ProcessingProgress,
    ) -> Dict[Tuple[int, int], TileResult]:
        """Process tiles using Dask."""
        results: Dict[Tuple[int, int], TileResult] = {}
        start_time = time.time()

        # Create or connect to cluster
        self._setup_cluster()

        try:
            # Create delayed tasks
            delayed_tasks = []
            for tile in tiles:
                task = delayed(self._process_single_tile)(
                    data, algorithm, tile
                )
                delayed_tasks.append((tile.index, task))

            # Configure scheduler
            if self.config.scheduler == SchedulerType.DISTRIBUTED:
                # Use distributed client
                if self._client is not None:
                    futures = self._client.compute(
                        [task for _, task in delayed_tasks]
                    )

                    # Collect results as they complete
                    for i, future in enumerate(as_completed(futures)):
                        if self._cancelled.is_set():
                            break

                        try:
                            result = future.result()
                            idx = delayed_tasks[i][0]
                            results[idx] = result

                            if result.success:
                                progress.completed_tiles += 1
                            else:
                                progress.failed_tiles += 1

                        except Exception as e:
                            idx = delayed_tasks[i][0]
                            tile = tiles[i]
                            results[idx] = TileResult(
                                tile_info=tile,
                                error=str(e),
                            )
                            progress.failed_tiles += 1

                        progress.pending_tiles -= 1
                        progress.elapsed_seconds = time.time() - start_time
                        self._report_progress(progress)

            else:
                # Use local scheduler
                scheduler_map = {
                    SchedulerType.SYNCHRONOUS: "synchronous",
                    SchedulerType.THREADS: "threads",
                    SchedulerType.PROCESSES: "processes",
                }
                scheduler = scheduler_map.get(
                    self.config.scheduler, "threads"
                )

                # Compute all at once with progress bar
                computed_results = dask.compute(
                    *[task for _, task in delayed_tasks],
                    scheduler=scheduler,
                    num_workers=self.config.n_workers,
                )

                # Map results to tile indices
                for i, result in enumerate(computed_results):
                    idx = delayed_tasks[i][0]
                    results[idx] = result

                    if result.success:
                        progress.completed_tiles += 1
                    else:
                        progress.failed_tiles += 1

                progress.pending_tiles = 0
                progress.elapsed_seconds = time.time() - start_time
                self._report_progress(progress)

        finally:
            self._cleanup_cluster()

        return results

    def _process_serial(
        self,
        data: np.ndarray,
        algorithm: Any,
        tiles: List[TileInfo],
        progress: ProcessingProgress,
    ) -> Dict[Tuple[int, int], TileResult]:
        """Process tiles serially (fallback)."""
        results: Dict[Tuple[int, int], TileResult] = {}
        start_time = time.time()

        for tile in tiles:
            if self._cancelled.is_set():
                break

            result = self._process_single_tile(data, algorithm, tile)
            results[tile.index] = result

            if result.success:
                progress.completed_tiles += 1
            else:
                progress.failed_tiles += 1

            progress.pending_tiles -= 1
            progress.elapsed_seconds = time.time() - start_time

            if progress.completed_tiles > 0:
                progress.tiles_per_second = (
                    progress.completed_tiles / progress.elapsed_seconds
                )
                remaining = progress.pending_tiles
                if progress.tiles_per_second > 0:
                    progress.estimated_remaining_seconds = (
                        remaining / progress.tiles_per_second
                    )

            self._report_progress(progress)

        return results

    def _process_single_tile(
        self,
        data: np.ndarray,
        algorithm: Any,
        tile: TileInfo,
    ) -> TileResult:
        """Process a single tile."""
        start_time = time.time()

        try:
            # Extract tile data (with overlap)
            if tile.overlap_pixel_bounds:
                bounds = tile.overlap_pixel_bounds
            else:
                bounds = tile.pixel_bounds

            tile_data = data[
                :,
                bounds[1]:bounds[3],
                bounds[0]:bounds[2]
            ].copy()

            # Process tile
            if hasattr(algorithm, "process_tile"):
                # Native tiled algorithm
                result_data = algorithm.process_tile(tile_data, tile)
            elif hasattr(algorithm, "execute"):
                # Algorithm with execute method
                result = algorithm.execute(tile_data.squeeze())
                if hasattr(result, "flood_extent"):
                    result_data = result.flood_extent
                elif hasattr(result, "data"):
                    result_data = result.data
                else:
                    result_data = result
            elif hasattr(algorithm, "run"):
                # Algorithm with run method
                result = algorithm.run(data=tile_data)
                if hasattr(result, "flood_extent"):
                    result_data = result.flood_extent
                elif hasattr(result, "data"):
                    result_data = result.data
                else:
                    result_data = result
            elif callable(algorithm):
                # Callable
                result_data = algorithm(tile_data)
            else:
                raise ValueError(
                    f"Algorithm {type(algorithm)} has no recognized processing method"
                )

            # Ensure result is numpy array
            result_data = np.asarray(result_data)

            # Get statistics if available
            statistics = {}
            if hasattr(algorithm, "last_statistics"):
                statistics = algorithm.last_statistics

            processing_time = time.time() - start_time

            return TileResult(
                tile_info=tile,
                data=result_data,
                statistics=statistics,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing tile {tile.index}: {e}")
            return TileResult(
                tile_info=tile,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    def _stitch_results(
        self,
        tile_results: Dict[Tuple[int, int], TileResult],
        output_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Stitch tile results into mosaic."""
        height, width = output_shape
        overlap = self.config.overlap

        # Determine output dtype and shape
        ref_result = next(
            (r for r in tile_results.values() if r.success),
            None
        )
        if ref_result is None or ref_result.data is None:
            return np.zeros((1, height, width), dtype=np.float32), None

        ref_data = ref_result.data
        if ref_data.ndim == 2:
            n_bands = 1
        else:
            n_bands = ref_data.shape[0]
        dtype = ref_data.dtype

        # Initialize output arrays
        if self.config.blend_mode in [BlendMode.AVERAGE, BlendMode.FEATHER]:
            output = np.zeros((n_bands, height, width), dtype=np.float64)
            weights = np.zeros((height, width), dtype=np.float64)
        else:
            output = np.full((n_bands, height, width), np.nan, dtype=np.float64)
            weights = None

        # Place each tile
        for (col, row), result in tile_results.items():
            if not result.success or result.data is None:
                continue

            tile_data = result.data
            if tile_data.ndim == 2:
                tile_data = tile_data[np.newaxis, :, :]

            tile_info = result.tile_info
            pb = tile_info.pixel_bounds

            # Calculate data region to use (trim overlap)
            data_row_start = overlap if row > 0 else 0
            data_col_start = overlap if col > 0 else 0

            # Get grid dimensions
            max_col = max(idx[0] for idx in tile_results.keys())
            max_row = max(idx[1] for idx in tile_results.keys())

            data_row_end = tile_data.shape[1] - (overlap if row < max_row else 0)
            data_col_end = tile_data.shape[2] - (overlap if col < max_col else 0)

            # Ensure bounds don't go negative
            data_row_end = max(data_row_start + 1, data_row_end)
            data_col_end = max(data_col_start + 1, data_col_end)

            # Extract core region
            core_data = tile_data[
                :,
                data_row_start:data_row_end,
                data_col_start:data_col_end
            ]

            # Calculate output position
            out_row_start = pb[1]
            out_col_start = pb[0]
            out_row_end = min(out_row_start + core_data.shape[1], height)
            out_col_end = min(out_col_start + core_data.shape[2], width)

            # Trim data to fit
            data_height = out_row_end - out_row_start
            data_width = out_col_end - out_col_start
            core_data = core_data[:, :data_height, :data_width]

            # Place in output
            if self.config.blend_mode == BlendMode.REPLACE:
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = core_data

            elif self.config.blend_mode in [BlendMode.AVERAGE, BlendMode.FEATHER]:
                tile_weight = np.ones((data_height, data_width), dtype=np.float64)
                for b in range(n_bands):
                    output[b, out_row_start:out_row_end, out_col_start:out_col_end] += (
                        core_data[b] * tile_weight
                    )
                weights[out_row_start:out_row_end, out_col_start:out_col_end] += tile_weight

            elif self.config.blend_mode == BlendMode.MAX:
                current = output[:, out_row_start:out_row_end, out_col_start:out_col_end]
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = np.fmax(
                    current, core_data
                )

            elif self.config.blend_mode == BlendMode.MIN:
                current = output[:, out_row_start:out_row_end, out_col_start:out_col_end]
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = np.fmin(
                    current, core_data
                )

        # Normalize by weights
        if self.config.blend_mode in [BlendMode.AVERAGE, BlendMode.FEATHER]:
            mask = weights > 0
            for b in range(n_bands):
                output[b, mask] /= weights[mask]

        # Convert to original dtype
        if self.config.preserve_dtype:
            output = output.astype(dtype)

        return output, None

    def _process_dask_array(
        self,
        dask_array: Any,
        algorithm: Any,
        bounds: Tuple[float, float, float, float],
        resolution: Tuple[float, float],
        output_path: Union[str, Path],
    ) -> DaskProcessingResult:
        """Process a Dask array directly."""
        start_time = time.time()

        # Setup cluster
        self._setup_cluster()

        try:
            # Define map function
            def process_chunk(block, block_info=None):
                if block_info is None:
                    return block

                # Create tile info from block info
                chunk_loc = block_info[0]["chunk-location"]
                tile_info = TileInfo(
                    col=chunk_loc[2],
                    row=chunk_loc[1],
                    pixel_bounds=(0, 0, block.shape[2], block.shape[1]),
                )

                # Process
                result = self._process_single_tile(
                    block, algorithm, tile_info
                )

                return result.data if result.success else block

            # Map function over blocks
            result_array = dask_array.map_blocks(
                process_chunk,
                dtype=np.float32,
            )

            # Compute and save
            with ProgressBar():
                computed = result_array.compute()

            processing_time = time.time() - start_time

            return DaskProcessingResult(
                mosaic=computed.squeeze(),
                processing_time_seconds=processing_time,
                metadata={
                    "bounds": bounds,
                    "resolution": resolution,
                },
            )

        finally:
            self._cleanup_cluster()

    def _setup_cluster(self) -> None:
        """Setup Dask cluster if needed."""
        if not HAS_DASK:
            return

        if self.config.scheduler == SchedulerType.DISTRIBUTED:
            if self.config.scheduler_address:
                # Connect to existing cluster
                self._client = Client(self.config.scheduler_address)
            else:
                # Create local cluster
                self._cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit_per_worker,
                    dashboard_address=f":{self.config.dashboard_port}"
                    if self.config.dashboard_port > 0 else None,
                )
                self._client = Client(self._cluster)

            logger.info(
                f"Connected to Dask cluster: {self._client.dashboard_link}"
                if self._client and hasattr(self._client, 'dashboard_link')
                else "Connected to Dask cluster"
            )

    def _cleanup_cluster(self) -> None:
        """Cleanup Dask cluster."""
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None

    def _report_progress(self, progress: ProcessingProgress) -> None:
        """Report progress via callback."""
        if self.progress_callback is not None:
            self.progress_callback(progress)

    def _save_result(
        self,
        result: DaskProcessingResult,
        output_path: Union[str, Path],
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
    ) -> None:
        """Save result to file."""
        if not HAS_RASTERIO:
            logger.warning("rasterio not available for saving")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = result.mosaic
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        n_bands, height, width = data.shape

        # Calculate transform
        if bounds and resolution:
            transform = from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3],
                width, height
            )
        else:
            transform = rasterio.transform.from_origin(0, height, 1, 1)

        # Write file
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_bands,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data)

        logger.info(f"Saved result to: {output_path}")

    def cancel(self) -> None:
        """Cancel processing."""
        self._cancelled.set()
        logger.info("Processing cancelled")


# =============================================================================
# Convenience Functions
# =============================================================================


def process_with_dask(
    data: np.ndarray,
    algorithm: Any,
    config: Optional[DaskProcessingConfig] = None,
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
) -> DaskProcessingResult:
    """
    Convenience function for Dask-based processing.

    Args:
        data: Input data array
        algorithm: Algorithm instance
        config: Processing configuration
        progress_callback: Progress callback

    Returns:
        DaskProcessingResult
    """
    processor = DaskTileProcessor(
        config=config or DaskProcessingConfig.for_laptop(),
        progress_callback=progress_callback,
    )
    return processor.process(data, algorithm)


def estimate_processing_time(
    data_shape: Tuple[int, ...],
    config: DaskProcessingConfig,
    time_per_tile_seconds: float = 0.5,
) -> float:
    """
    Estimate processing time.

    Args:
        data_shape: Shape of input data
        config: Processing configuration
        time_per_tile_seconds: Average time per tile

    Returns:
        Estimated time in seconds
    """
    if len(data_shape) == 2:
        height, width = data_shape
    else:
        height, width = data_shape[1], data_shape[2]

    tile_h, tile_w = config.tile_size
    n_tiles = (
        int(np.ceil(height / tile_h)) *
        int(np.ceil(width / tile_w))
    )

    # Account for parallelism
    effective_workers = config.n_workers * config.threads_per_worker
    parallel_batches = int(np.ceil(n_tiles / effective_workers))

    return parallel_batches * time_per_tile_seconds


def get_optimal_config(
    data_shape: Tuple[int, ...],
    available_memory_gb: float = 4.0,
    target_time_minutes: float = 10.0,
) -> DaskProcessingConfig:
    """
    Get optimal configuration for given constraints.

    Args:
        data_shape: Shape of input data
        available_memory_gb: Available memory in GB
        target_time_minutes: Target processing time in minutes

    Returns:
        Optimized DaskProcessingConfig
    """
    # Estimate data size
    if len(data_shape) == 2:
        n_pixels = data_shape[0] * data_shape[1]
    else:
        n_pixels = data_shape[1] * data_shape[2] * data_shape[0]

    data_size_mb = n_pixels * 4 / (1024 * 1024)  # Assume float32

    # Determine tile size based on memory
    if available_memory_gb < 2:
        tile_size = (256, 256)
    elif available_memory_gb < 4:
        tile_size = (512, 512)
    elif available_memory_gb < 8:
        tile_size = (768, 768)
    else:
        tile_size = (1024, 1024)

    # Determine workers based on target time
    n_workers = max(1, os.cpu_count() or 4)

    return DaskProcessingConfig(
        n_workers=n_workers,
        tile_size=tile_size,
        memory_limit_per_worker=f"{available_memory_gb / n_workers:.1f}GB",
    )
