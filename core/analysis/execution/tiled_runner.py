"""
Tiled Algorithm Runner for Memory-Efficient Processing.

Enables algorithms to process data tile-by-tile for low-memory execution.
Wraps any algorithm for tile processing, handles overlap/blending,
and stitches results into final mosaics.

Key Components:
- TileContext: Context information for tile processing
- TiledAlgorithmRunner: Execute algorithms tile-by-tile
- ResultStitcher: Stitch tile results into mosaic output

Example Usage:
    from core.analysis.execution.tiled_runner import (
        TileContext,
        TiledAlgorithmRunner,
        ResultStitcher,
    )

    # Create runner
    runner = TiledAlgorithmRunner(
        algorithm=ThresholdSARAlgorithm(),
        tile_size=(512, 512),
        overlap=32,
    )

    # Process data
    result = runner.process(input_data, bounds, resolution)

    # Access results
    mosaic = result.mosaic
    statistics = result.aggregate_statistics()
"""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
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
    Type,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Import from tiling module
try:
    from core.execution.tiling import (
        BlendMode,
        OverlapHandler,
        PixelBounds,
        TileBounds,
        TileGrid,
        TileIndex,
        TileInfo,
        TileManager,
        TileScheme,
        TileStatus,
    )
except ImportError:
    # Fallback for when module is not yet available
    logger.warning("Could not import from core.execution.tiling")


# =============================================================================
# Type Variables and Protocols
# =============================================================================


T = TypeVar("T")


class TiledAlgorithmProtocol(Protocol):
    """Protocol for algorithms that support tiled processing."""

    def process_tile(self, data: np.ndarray, context: "TileContext") -> np.ndarray:
        """Process a single tile."""
        ...

    @property
    def supports_tiled(self) -> bool:
        """Whether algorithm supports tiled processing."""
        ...


class AlgorithmProtocol(Protocol):
    """Protocol for standard algorithms."""

    def run(self, **kwargs) -> Any:
        """Run the algorithm."""
        ...


# =============================================================================
# Enumerations
# =============================================================================


class StitchMethod(Enum):
    """Methods for stitching tiles together."""

    REPLACE = "replace"  # Later tiles replace earlier
    FEATHER = "feather"  # Linear feather blend
    AVERAGE = "average"  # Average overlapping regions
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    WEIGHTED = "weighted"  # Custom weight-based blending


class AggregationMethod(Enum):
    """Methods for aggregating per-tile statistics."""

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TileContext:
    """
    Context information for tile processing.

    Provides tile metadata and utilities during algorithm execution
    on a specific tile.

    Attributes:
        tile_index: Index of current tile (col, row)
        tile_bounds: Geographic bounds of tile
        pixel_bounds: Pixel bounds in source image
        overlap_bounds: Geographic bounds including overlap
        overlap_pixel_bounds: Pixel bounds including overlap
        neighbors: References to neighboring tiles
        grid_shape: Total grid shape (rows, cols)
        progress_callback: Optional callback for progress updates
        metadata: Additional metadata
    """

    tile_index: "TileIndex"
    tile_bounds: "TileBounds"
    pixel_bounds: "PixelBounds"
    overlap_bounds: Optional["TileBounds"] = None
    overlap_pixel_bounds: Optional["PixelBounds"] = None
    neighbors: Dict[str, Optional["TileIndex"]] = field(default_factory=dict)
    grid_shape: Tuple[int, int] = (1, 1)
    progress_callback: Optional[Callable[[float, str], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_overlap(self) -> bool:
        """Whether tile has overlap region."""
        return self.overlap_bounds is not None

    @property
    def is_edge_tile(self) -> bool:
        """Whether tile is on edge of grid."""
        return (
            self.tile_index.col == 0
            or self.tile_index.row == 0
            or self.tile_index.col == self.grid_shape[1] - 1
            or self.tile_index.row == self.grid_shape[0] - 1
        )

    @property
    def is_corner_tile(self) -> bool:
        """Whether tile is in corner of grid."""
        return (
            (self.tile_index.col == 0 or self.tile_index.col == self.grid_shape[1] - 1)
            and (self.tile_index.row == 0 or self.tile_index.row == self.grid_shape[0] - 1)
        )

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report processing progress (0.0 to 1.0)."""
        if self.progress_callback is not None:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)

    def get_neighbor(self, direction: str) -> Optional["TileIndex"]:
        """
        Get neighbor tile index in given direction.

        Args:
            direction: One of "left", "right", "top", "bottom",
                      "top_left", "top_right", "bottom_left", "bottom_right"

        Returns:
            TileIndex of neighbor or None if at edge
        """
        return self.neighbors.get(direction)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tile_index": self.tile_index.to_dict(),
            "tile_bounds": self.tile_bounds.to_dict(),
            "pixel_bounds": self.pixel_bounds.to_dict(),
            "overlap_bounds": self.overlap_bounds.to_dict() if self.overlap_bounds else None,
            "grid_shape": list(self.grid_shape),
            "is_edge_tile": self.is_edge_tile,
            "is_corner_tile": self.is_corner_tile,
            "metadata": self.metadata,
        }


@dataclass
class TileResult:
    """
    Result from processing a single tile.

    Attributes:
        tile_index: Index of processed tile
        data: Result data array
        confidence: Optional confidence array
        statistics: Per-tile statistics
        processing_time_seconds: Time taken to process
        metadata: Additional metadata
    """

    tile_index: "TileIndex"
    data: np.ndarray
    confidence: Optional[np.ndarray] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingProgress:
    """
    Progress information for tiled processing.

    Attributes:
        total_tiles: Total number of tiles
        completed_tiles: Number of completed tiles
        failed_tiles: Number of failed tiles
        current_tile: Currently processing tile
        elapsed_seconds: Elapsed time
        estimated_remaining_seconds: Estimated time remaining
    """

    total_tiles: int = 0
    completed_tiles: int = 0
    failed_tiles: int = 0
    current_tile: Optional["TileIndex"] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    @property
    def progress_percent(self) -> float:
        """Progress as percentage."""
        if self.total_tiles == 0:
            return 0.0
        return (self.completed_tiles / self.total_tiles) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tiles": self.total_tiles,
            "completed_tiles": self.completed_tiles,
            "failed_tiles": self.failed_tiles,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
        }


@dataclass
class TiledProcessingResult:
    """
    Complete result from tiled processing.

    Attributes:
        mosaic: Stitched result mosaic
        confidence_mosaic: Optional stitched confidence
        tile_results: Per-tile results
        statistics: Aggregated statistics
        processing_time_seconds: Total processing time
        metadata: Additional metadata
    """

    mosaic: np.ndarray
    confidence_mosaic: Optional[np.ndarray] = None
    tile_results: Dict["TileIndex", TileResult] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def aggregate_statistics(
        self,
        method: AggregationMethod = AggregationMethod.MEAN,
    ) -> Dict[str, float]:
        """
        Aggregate statistics across all tiles.

        Args:
            method: Aggregation method

        Returns:
            Aggregated statistics dictionary
        """
        if not self.tile_results:
            return {}

        # Collect all statistics keys
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
            if not values:
                continue

            if method == AggregationMethod.SUM:
                aggregated[key] = sum(values)
            elif method == AggregationMethod.MEAN:
                aggregated[key] = np.mean(values)
            elif method == AggregationMethod.MIN:
                aggregated[key] = min(values)
            elif method == AggregationMethod.MAX:
                aggregated[key] = max(values)
            elif method == AggregationMethod.MEDIAN:
                aggregated[key] = np.median(values)

        return aggregated


# =============================================================================
# TiledAlgorithmRunner
# =============================================================================


class TiledAlgorithmRunner:
    """
    Run algorithms tile-by-tile for memory-efficient processing.

    Wraps any algorithm (with or without native tiled support) and
    handles tile iteration, overlap management, and result aggregation.

    Example:
        # With native tiled algorithm
        runner = TiledAlgorithmRunner(
            algorithm=TiledFloodDetector(),
            tile_size=(512, 512),
            overlap=32,
        )
        result = runner.process(sar_data, bounds, resolution)

        # With standard algorithm (automatic wrapping)
        runner = TiledAlgorithmRunner.wrap_algorithm(
            algorithm=StandardFloodDetector(),
            tile_size=(512, 512),
        )
        result = runner.process(sar_data, bounds, resolution)
    """

    def __init__(
        self,
        algorithm: Any,
        tile_size: Union[int, Tuple[int, int]] = (512, 512),
        overlap: int = 32,
        blend_mode: BlendMode = BlendMode.FEATHER,
        parallel: bool = False,
        max_workers: int = 4,
        workdir: Optional[Path] = None,
    ):
        """
        Initialize tiled runner.

        Args:
            algorithm: Algorithm instance (with or without tiled support)
            tile_size: Tile size in pixels
            overlap: Overlap pixels on each side
            blend_mode: Method for blending overlapping regions
            parallel: Enable parallel tile processing
            max_workers: Maximum parallel workers
            workdir: Working directory for intermediate files
        """
        self.algorithm = algorithm
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.parallel = parallel
        self.max_workers = max_workers
        self.workdir = Path(workdir) if workdir else None

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        self.tile_size = tile_size

        # Check if algorithm supports tiled processing
        self.supports_native_tiled = hasattr(algorithm, "process_tile") and callable(
            getattr(algorithm, "process_tile")
        )

        # Create overlap handler
        self._overlap_handler = OverlapHandler(
            overlap=overlap,
            blend_mode=blend_mode,
        )

        # Processing state
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

        # Image validation (lazy-loaded)
        self._validator = None
        self._validation_enabled = True

    @property
    def validator(self):
        """Lazy-load image validator."""
        if self._validator is None and self._validation_enabled:
            try:
                from core.data.ingestion.validation import ImageValidator
                self._validator = ImageValidator()
            except ImportError:
                logger.warning("Image validation module not available for tiled runner")
                self._validation_enabled = False
        return self._validator

    def validate_input(
        self,
        raster_path: Union[str, Path],
        data_source_spec: Optional[Dict[str, Any]] = None,
    ):
        """
        Validate input raster before tiled processing.

        Args:
            raster_path: Path to raster file
            data_source_spec: Optional data source specification

        Returns:
            ImageValidationResult or None if validation disabled

        Raises:
            ImageValidationError: If validation fails and rejection is configured
        """
        if not self._validation_enabled or self.validator is None:
            return None

        from core.data.ingestion.validation import ImageValidationError

        validation_result = self.validator.validate(
            raster_path=raster_path,
            data_source_spec=data_source_spec,
        )

        if not validation_result.is_valid:
            raise ImageValidationError(
                f"Image validation failed: {validation_result.errors}",
                {"validation_result": validation_result.to_dict()},
            )

        return validation_result

    @classmethod
    def wrap_algorithm(
        cls,
        algorithm: Any,
        tile_size: Union[int, Tuple[int, int]] = (512, 512),
        overlap: int = 0,
        **kwargs,
    ) -> "TiledAlgorithmRunner":
        """
        Wrap a standard algorithm for tiled processing.

        Args:
            algorithm: Algorithm instance
            tile_size: Tile size
            overlap: Overlap size
            **kwargs: Additional runner arguments

        Returns:
            TiledAlgorithmRunner instance
        """
        return cls(
            algorithm=algorithm,
            tile_size=tile_size,
            overlap=overlap,
            **kwargs,
        )

    def process(
        self,
        data: np.ndarray,
        bounds: Optional[Union[Tuple[float, float, float, float], "TileBounds"]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        resume: bool = False,
    ) -> TiledProcessingResult:
        """
        Process data tile-by-tile.

        Args:
            data: Input data array (height, width) or (bands, height, width)
            bounds: Geographic bounds (optional)
            resolution: Pixel resolution (optional)
            progress_callback: Progress callback
            resume: Resume from previous partial processing

        Returns:
            TiledProcessingResult with mosaic and statistics
        """
        self._cancelled.clear()
        start_time = time.time()

        # Normalize input
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        n_bands, height, width = data.shape

        # Create tile grid
        if bounds is None:
            bounds = TileBounds(0, 0, float(width), float(height))
        elif not isinstance(bounds, TileBounds):
            bounds = TileBounds(*bounds)

        if resolution is None:
            resolution = (bounds.width / width, bounds.height / height)

        scheme = TileScheme(
            tile_size=self.tile_size,
            overlap=self.overlap,
        )
        grid = TileGrid(
            bounds=bounds,
            resolution=resolution,
            scheme=scheme,
            image_shape=(height, width),
        )

        # Initialize manager
        manager = None
        if self.workdir and resume:
            manager = TileManager(grid, workdir=self.workdir)

        # Initialize progress
        progress = ProcessingProgress(total_tiles=grid.total_tiles)

        # Process tiles
        tile_results: Dict[TileIndex, TileResult] = {}

        if self.parallel and self.max_workers > 1:
            tile_results = self._process_parallel(
                data, grid, manager, progress, progress_callback
            )
        else:
            tile_results = self._process_sequential(
                data, grid, manager, progress, progress_callback
            )

        # Stitch results
        stitcher = ResultStitcher(
            grid=grid,
            blend_mode=self.blend_mode,
        )

        mosaic, confidence_mosaic = stitcher.stitch(tile_results)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Aggregate statistics
        result = TiledProcessingResult(
            mosaic=mosaic.squeeze(),
            confidence_mosaic=confidence_mosaic.squeeze() if confidence_mosaic is not None else None,
            tile_results=tile_results,
            processing_time_seconds=processing_time,
            metadata={
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "n_tiles": grid.total_tiles,
                "grid_shape": grid.shape,
            },
        )
        result.statistics = result.aggregate_statistics()

        return result

    def _process_sequential(
        self,
        data: np.ndarray,
        grid: "TileGrid",
        manager: Optional["TileManager"],
        progress: ProcessingProgress,
        progress_callback: Optional[Callable[[ProcessingProgress], None]],
    ) -> Dict["TileIndex", TileResult]:
        """Process tiles sequentially."""
        tile_results = {}
        start_time = time.time()

        for tile_info in grid:
            if self._cancelled.is_set():
                break

            # Skip completed tiles when resuming
            if manager and manager.get_status(tile_info.index) == TileStatus.COMPLETED:
                continue

            progress.current_tile = tile_info.index

            try:
                # Extract tile data
                tile_data = self._extract_tile_data(data, tile_info)

                # Create context
                context = self._create_context(tile_info, grid)

                # Process tile
                tile_start = time.time()
                result_data = self._process_single_tile(tile_data, context)
                tile_time = time.time() - tile_start

                # Create result
                tile_result = TileResult(
                    tile_index=tile_info.index,
                    data=result_data,
                    processing_time_seconds=tile_time,
                )

                # Extract statistics if algorithm provides them
                if hasattr(self.algorithm, "last_statistics"):
                    tile_result.statistics = self.algorithm.last_statistics

                tile_results[tile_info.index] = tile_result

                if manager:
                    manager.mark_completed(tile_info.index, tile_result)

                progress.completed_tiles += 1

            except Exception as e:
                logger.error(f"Error processing tile {tile_info.index}: {e}")
                progress.failed_tiles += 1
                if manager:
                    manager.mark_failed(tile_info.index, str(e))

            # Update progress
            progress.elapsed_seconds = time.time() - start_time
            if progress.completed_tiles > 0:
                avg_time = progress.elapsed_seconds / progress.completed_tiles
                remaining = progress.total_tiles - progress.completed_tiles - progress.failed_tiles
                progress.estimated_remaining_seconds = avg_time * remaining

            if progress_callback:
                progress_callback(progress)

        return tile_results

    def _process_parallel(
        self,
        data: np.ndarray,
        grid: "TileGrid",
        manager: Optional["TileManager"],
        progress: ProcessingProgress,
        progress_callback: Optional[Callable[[ProcessingProgress], None]],
    ) -> Dict["TileIndex", TileResult]:
        """Process tiles in parallel."""
        import concurrent.futures

        tile_results = {}
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for tile_info in grid:
                if self._cancelled.is_set():
                    break

                # Skip completed tiles
                if manager and manager.get_status(tile_info.index) == TileStatus.COMPLETED:
                    continue

                # Submit task
                future = executor.submit(
                    self._process_tile_worker,
                    data,
                    tile_info,
                    grid,
                )
                futures[future] = tile_info.index

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                tile_index = futures[future]

                try:
                    tile_result = future.result()
                    tile_results[tile_index] = tile_result
                    progress.completed_tiles += 1

                    if manager:
                        manager.mark_completed(tile_index, tile_result)

                except Exception as e:
                    logger.error(f"Error processing tile {tile_index}: {e}")
                    progress.failed_tiles += 1
                    if manager:
                        manager.mark_failed(tile_index, str(e))

                # Update progress
                progress.elapsed_seconds = time.time() - start_time
                if progress_callback:
                    progress_callback(progress)

        return tile_results

    def _process_tile_worker(
        self,
        data: np.ndarray,
        tile_info: "TileInfo",
        grid: "TileGrid",
    ) -> TileResult:
        """Worker function for parallel processing."""
        tile_data = self._extract_tile_data(data, tile_info)
        context = self._create_context(tile_info, grid)

        tile_start = time.time()
        result_data = self._process_single_tile(tile_data, context)
        tile_time = time.time() - tile_start

        return TileResult(
            tile_index=tile_info.index,
            data=result_data,
            processing_time_seconds=tile_time,
        )

    def _extract_tile_data(
        self,
        data: np.ndarray,
        tile_info: "TileInfo",
    ) -> np.ndarray:
        """Extract tile data from full array."""
        if tile_info.overlap_pixel_bounds:
            pb = tile_info.overlap_pixel_bounds
        else:
            pb = tile_info.pixel_bounds

        return data[:, pb.row_start : pb.row_end, pb.col_start : pb.col_end].copy()

    def _create_context(
        self,
        tile_info: "TileInfo",
        grid: "TileGrid",
    ) -> TileContext:
        """Create tile context."""
        # Get neighbors
        neighbors = {}
        for direction, (dc, dr) in [
            ("left", (-1, 0)),
            ("right", (1, 0)),
            ("top", (0, -1)),
            ("bottom", (0, 1)),
            ("top_left", (-1, -1)),
            ("top_right", (1, -1)),
            ("bottom_left", (-1, 1)),
            ("bottom_right", (1, 1)),
        ]:
            new_col = tile_info.index.col + dc
            new_row = tile_info.index.row + dr
            if 0 <= new_col < grid.n_cols and 0 <= new_row < grid.n_rows:
                neighbors[direction] = TileIndex(col=new_col, row=new_row)
            else:
                neighbors[direction] = None

        return TileContext(
            tile_index=tile_info.index,
            tile_bounds=tile_info.geo_bounds,
            pixel_bounds=tile_info.pixel_bounds,
            overlap_bounds=tile_info.overlap_geo_bounds,
            overlap_pixel_bounds=tile_info.overlap_pixel_bounds,
            neighbors=neighbors,
            grid_shape=grid.shape,
        )

    def _process_single_tile(
        self,
        tile_data: np.ndarray,
        context: TileContext,
    ) -> np.ndarray:
        """Process a single tile."""
        if self.supports_native_tiled:
            # Use native tiled processing
            return self.algorithm.process_tile(tile_data, context)
        else:
            # Wrap standard algorithm
            return self._wrap_standard_algorithm(tile_data, context)

    def _wrap_standard_algorithm(
        self,
        tile_data: np.ndarray,
        context: TileContext,
    ) -> np.ndarray:
        """Wrap a standard algorithm for tile processing."""
        # Try different common algorithm interfaces
        if hasattr(self.algorithm, "run"):
            result = self.algorithm.run(data=tile_data)
            if hasattr(result, "flood_extent"):
                return result.flood_extent
            elif hasattr(result, "data"):
                return result.data
            elif isinstance(result, np.ndarray):
                return result
            else:
                return tile_data

        elif hasattr(self.algorithm, "process"):
            return self.algorithm.process(tile_data)

        elif hasattr(self.algorithm, "__call__"):
            return self.algorithm(tile_data)

        else:
            raise ValueError(
                f"Algorithm {type(self.algorithm)} does not have a recognized processing method"
            )

    def cancel(self) -> None:
        """Cancel processing."""
        self._cancelled.set()


# =============================================================================
# ResultStitcher
# =============================================================================


class ResultStitcher:
    """
    Stitch tile results into mosaic output.

    Handles overlap blending, edge handling, and incremental
    output writing for COG-friendly results.

    Example:
        stitcher = ResultStitcher(
            grid=tile_grid,
            blend_mode=BlendMode.FEATHER,
        )

        # Stitch all results
        mosaic = stitcher.stitch(tile_results)

        # Incremental stitching (COG-friendly)
        stitcher.begin_stitch(output_path)
        for tile_result in tile_results:
            stitcher.add_tile(tile_result)
        stitcher.finish_stitch()
    """

    def __init__(
        self,
        grid: "TileGrid",
        blend_mode: BlendMode = BlendMode.FEATHER,
        output_dtype: np.dtype = np.float32,
        fill_value: float = np.nan,
    ):
        """
        Initialize stitcher.

        Args:
            grid: Tile grid used for processing
            blend_mode: Method for blending overlapping regions
            output_dtype: Output data type
            fill_value: Fill value for missing tiles
        """
        self.grid = grid
        self.blend_mode = blend_mode
        self.output_dtype = output_dtype
        self.fill_value = fill_value

        self._overlap_handler = OverlapHandler(
            overlap=grid.scheme.overlap,
            blend_mode=blend_mode,
        )

        # Incremental stitching state
        self._output_array: Optional[np.ndarray] = None
        self._weight_array: Optional[np.ndarray] = None
        self._output_path: Optional[Path] = None

    def stitch(
        self,
        tile_results: Dict["TileIndex", TileResult],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Stitch all tile results into mosaic.

        Args:
            tile_results: Dictionary of tile index -> result

        Returns:
            Tuple of (mosaic array, confidence array or None)
        """
        # Determine output shape
        height = self.grid.image_height
        width = self.grid.image_width

        # Get reference result for bands/dtype
        ref_result = next(iter(tile_results.values())) if tile_results else None
        if ref_result is None:
            return np.full((height, width), self.fill_value, dtype=self.output_dtype), None

        ref_data = ref_result.data
        if ref_data.ndim == 2:
            n_bands = 1
        else:
            n_bands = ref_data.shape[0]

        # Initialize output arrays
        # For weighted blending modes, initialize with zeros (not NaN) since we'll add to it
        if self.blend_mode in [BlendMode.AVERAGE, BlendMode.FEATHER]:
            output = np.zeros((n_bands, height, width), dtype=self.output_dtype)
        else:
            output = np.full((n_bands, height, width), self.fill_value, dtype=self.output_dtype)
        weights = np.zeros((height, width), dtype=np.float32)

        # Check for confidence
        has_confidence = ref_result.confidence is not None
        confidence_output = None
        if has_confidence:
            confidence_output = np.full((height, width), self.fill_value, dtype=np.float32)

        # Create blend weight masks
        tile_weight = self._create_weight_mask(self.grid.scheme.tile_size)

        # Place each tile
        for tile_index, tile_result in tile_results.items():
            tile_info = self.grid.get_tile(tile_index)
            tile_data = tile_result.data

            # Convert to numpy array and ensure proper shape
            tile_data = np.asarray(tile_data)
            if tile_data.ndim == 2:
                tile_data = tile_data[np.newaxis, :, :]

            # Get core region (without overlap)
            pb = tile_info.pixel_bounds
            overlap = self.grid.scheme.overlap

            # Calculate data region to use (trim overlap on non-edge tiles)
            data_row_start = overlap if tile_index.row > 0 else 0
            data_col_start = overlap if tile_index.col > 0 else 0
            data_row_end = tile_data.shape[1] - overlap if tile_index.row < self.grid.n_rows - 1 else tile_data.shape[1]
            data_col_end = tile_data.shape[2] - overlap if tile_index.col < self.grid.n_cols - 1 else tile_data.shape[2]

            # Extract core region using explicit slicing
            core_data = tile_data[
                :,
                int(data_row_start):int(data_row_end),
                int(data_col_start):int(data_col_end)
            ].copy()

            # Calculate output position
            out_row_start = pb.row_start
            out_col_start = pb.col_start
            out_row_end = min(out_row_start + core_data.shape[1], height)
            out_col_end = min(out_col_start + core_data.shape[2], width)

            # Trim data to fit
            data_height = out_row_end - out_row_start
            data_width = out_col_end - out_col_start
            core_data = core_data[:, :int(data_height), :int(data_width)]

            # Place in output
            if self.blend_mode == BlendMode.AVERAGE or self.blend_mode == BlendMode.FEATHER:
                # Weighted blending
                # Create weight matrix matching the data size
                tile_w = np.ones((int(data_height), int(data_width)), dtype=np.float32)

                for b in range(n_bands):
                    output[b, out_row_start:out_row_end, out_col_start:out_col_end] += (
                        core_data[b] * tile_w
                    )
                weights[out_row_start:out_row_end, out_col_start:out_col_end] += tile_w

            elif self.blend_mode == BlendMode.MAX:
                current = output[:, out_row_start:out_row_end, out_col_start:out_col_end]
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = np.maximum(
                    current, core_data
                )
                weights[out_row_start:out_row_end, out_col_start:out_col_end] = 1

            elif self.blend_mode == BlendMode.MIN:
                current = output[:, out_row_start:out_row_end, out_col_start:out_col_end]
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = np.minimum(
                    current, core_data
                )
                weights[out_row_start:out_row_end, out_col_start:out_col_end] = 1

            else:
                # Simple replace
                output[:, out_row_start:out_row_end, out_col_start:out_col_end] = core_data
                weights[out_row_start:out_row_end, out_col_start:out_col_end] = 1

            # Handle confidence
            if has_confidence and tile_result.confidence is not None:
                conf = np.asarray(tile_result.confidence)
                if conf.ndim == 3:
                    conf = conf[0]  # Take first band
                conf = conf[int(data_row_start):int(data_row_end), int(data_col_start):int(data_col_end)]
                conf = conf[:int(data_height), :int(data_width)]
                confidence_output[out_row_start:out_row_end, out_col_start:out_col_end] = conf

        # Normalize by weights
        if self.blend_mode in [BlendMode.AVERAGE, BlendMode.FEATHER]:
            mask = weights > 0
            for b in range(n_bands):
                output[b, mask] /= weights[mask]

        # Squeeze single band output
        if n_bands == 1:
            output = output.squeeze(axis=0)

        return output, confidence_output

    def _create_weight_mask(
        self,
        tile_size: Tuple[int, int],
    ) -> np.ndarray:
        """Create weight mask for blending."""
        height, width = tile_size
        overlap = self.grid.scheme.overlap

        mask = np.ones((height, width), dtype=np.float32)

        if self.blend_mode == BlendMode.NONE:
            return mask

        if overlap > 0:
            # Create linear ramps at edges
            ramp_h = np.linspace(0, 1, overlap)
            ramp_v = np.linspace(0, 1, overlap)

            # Apply to edges
            mask[:overlap, :] *= ramp_v[:, np.newaxis]
            mask[-overlap:, :] *= ramp_v[::-1][:, np.newaxis]
            mask[:, :overlap] *= ramp_h[np.newaxis, :]
            mask[:, -overlap:] *= ramp_h[::-1][np.newaxis, :]

        if self.blend_mode == BlendMode.FEATHER:
            # Apply cosine smoothing
            mask = (1 - np.cos(mask * np.pi)) / 2

        return mask

    def begin_stitch(
        self,
        output_path: Union[str, Path],
        crs: Optional[str] = None,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Begin incremental stitching.

        Args:
            output_path: Path for output file
            crs: Coordinate reference system
            transform: Affine transform
        """
        self._output_path = Path(output_path)
        self._output_array = np.full(
            (self.grid.image_height, self.grid.image_width),
            self.fill_value,
            dtype=self.output_dtype,
        )
        self._weight_array = np.zeros(
            (self.grid.image_height, self.grid.image_width),
            dtype=np.float32,
        )

    def add_tile(self, tile_result: TileResult) -> None:
        """Add a tile to the incremental stitch."""
        if self._output_array is None:
            raise RuntimeError("Call begin_stitch() first")

        tile_info = self.grid.get_tile(tile_result.tile_index)
        pb = tile_info.pixel_bounds

        data = tile_result.data
        if data.ndim == 3:
            data = data[0]  # Take first band

        # Place in output
        self._output_array[pb.row_start : pb.row_end, pb.col_start : pb.col_end] = data
        self._weight_array[pb.row_start : pb.row_end, pb.col_start : pb.col_end] = 1

    def finish_stitch(self) -> np.ndarray:
        """Finish incremental stitching and return result."""
        if self._output_array is None:
            raise RuntimeError("Call begin_stitch() first")

        result = self._output_array.copy()
        self._output_array = None
        self._weight_array = None

        return result


# =============================================================================
# Utility Functions
# =============================================================================


def check_algorithm_tiled_support(algorithm: Any) -> Dict[str, Any]:
    """
    Check if an algorithm supports tiled processing.

    Args:
        algorithm: Algorithm instance

    Returns:
        Dictionary with support information
    """
    info = {
        "has_process_tile": hasattr(algorithm, "process_tile") and callable(getattr(algorithm, "process_tile")),
        "has_supports_tiled_flag": hasattr(algorithm, "supports_tiled"),
        "supports_tiled_value": getattr(algorithm, "supports_tiled", False),
        "has_run": hasattr(algorithm, "run"),
        "has_process": hasattr(algorithm, "process"),
        "is_callable": callable(algorithm),
        "algorithm_type": type(algorithm).__name__,
    }

    info["can_be_wrapped"] = info["has_run"] or info["has_process"] or info["is_callable"]
    info["native_tiled_support"] = info["has_process_tile"]
    info["supports_tiled_execution"] = info["native_tiled_support"] or info["can_be_wrapped"]

    return info


def run_algorithm_tiled(
    algorithm: Any,
    data: np.ndarray,
    tile_size: int = 512,
    overlap: int = 32,
    **kwargs,
) -> TiledProcessingResult:
    """
    Convenience function to run algorithm with tiled processing.

    Args:
        algorithm: Algorithm instance
        data: Input data array
        tile_size: Tile size
        overlap: Overlap pixels
        **kwargs: Additional arguments for process()

    Returns:
        TiledProcessingResult
    """
    runner = TiledAlgorithmRunner(
        algorithm=algorithm,
        tile_size=tile_size,
        overlap=overlap,
    )
    return runner.process(data, **kwargs)


def estimate_tiles_for_memory(
    data_shape: Tuple[int, ...],
    available_memory_mb: int,
    dtype: np.dtype = np.float32,
    overhead_factor: float = 3.0,
) -> Tuple[int, int]:
    """
    Estimate tile size for available memory.

    Args:
        data_shape: Shape of input data
        available_memory_mb: Available memory in MB
        dtype: Data type
        overhead_factor: Safety factor for processing overhead

    Returns:
        Suggested tile size (width, height)
    """
    if len(data_shape) == 2:
        height, width = data_shape
        n_bands = 1
    else:
        n_bands, height, width = data_shape[:3]

    bytes_per_pixel = np.dtype(dtype).itemsize * n_bands
    available_bytes = available_memory_mb * 1024 * 1024 / overhead_factor

    max_pixels = available_bytes / bytes_per_pixel
    max_side = int(np.sqrt(max_pixels))

    # Round to power of 2
    power = int(np.log2(max_side))
    tile_size = 2 ** power

    # Clamp to reasonable range
    tile_size = max(128, min(tile_size, 2048, max(height, width)))

    return (tile_size, tile_size)
