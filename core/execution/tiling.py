"""
Tiled Processing Infrastructure for Low-Memory Execution.

Enables the platform to run on low-memory devices by processing data tile-by-tile.
This module provides infrastructure for tile management, overlap handling,
and resumable tile processing.

Key Components:
- TileScheme: Define tile sizes and overlap configurations
- TileGrid: Generate tile grids covering an AOI
- TileManager: Track tile processing state and enable resume
- OverlapHandler: Handle edge effects through overlap blending

Example Usage:
    from core.execution.tiling import (
        TileScheme,
        TileGrid,
        TileManager,
        OverlapHandler,
    )

    # Create tile scheme
    scheme = TileScheme(tile_size=(512, 512), overlap=32)

    # Generate grid for AOI
    grid = TileGrid(
        bounds=(-122.5, 37.5, -122.0, 38.0),
        resolution=(10.0, 10.0),
        scheme=scheme,
    )

    # Process tiles with manager
    manager = TileManager(grid, workdir=Path("./tiles"))
    for tile in manager.pending_tiles():
        result = process_tile(tile)
        manager.mark_completed(tile.index, result)

    # Merge results
    final = manager.merge_results()
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class TileSizePreset(Enum):
    """Standard tile size presets."""

    TINY = (128, 128)  # For very constrained memory
    SMALL = (256, 256)  # Web map standard
    MEDIUM = (512, 512)  # Good balance
    LARGE = (1024, 1024)  # For high-memory systems
    XLARGE = (2048, 2048)  # For cloud/HPC


class BlendMode(Enum):
    """Overlap blending modes."""

    NONE = "none"  # No blending (cut at boundary)
    FEATHER = "feather"  # Linear feather blend
    AVERAGE = "average"  # Average overlapping values
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    PRIORITY_LEFT = "priority_left"  # Left/top tile takes priority
    PRIORITY_RIGHT = "priority_right"  # Right/bottom tile takes priority


class TileStatus(Enum):
    """Status of tile processing."""

    PENDING = "pending"  # Not yet processed
    IN_PROGRESS = "in_progress"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed
    SKIPPED = "skipped"  # Skipped (e.g., no data)


class CoordinateSystem(Enum):
    """Coordinate system for tile references."""

    PIXEL = "pixel"  # Pixel coordinates (row, col)
    GEOGRAPHIC = "geographic"  # Geographic coordinates (lat, lon)
    PROJECTED = "projected"  # Projected coordinates (x, y in CRS units)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TileIndex:
    """
    Index of a tile within a grid.

    Attributes:
        col: Column index (x direction)
        row: Row index (y direction)
        level: Optional level for multi-resolution tiles
    """

    col: int
    row: int
    level: int = 0

    def __hash__(self) -> int:
        return hash((self.col, self.row, self.level))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileIndex):
            return False
        return self.col == other.col and self.row == other.row and self.level == other.level

    def __str__(self) -> str:
        return f"({self.col}, {self.row}, L{self.level})"

    def to_tuple(self) -> Tuple[int, int, int]:
        """Return as tuple (col, row, level)."""
        return (self.col, self.row, self.level)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {"col": self.col, "row": self.row, "level": self.level}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "TileIndex":
        """Create from dictionary."""
        return cls(col=data["col"], row=data["row"], level=data.get("level", 0))


@dataclass
class TileBounds:
    """
    Bounds of a tile in geographic or projected coordinates.

    Attributes:
        minx: Minimum x coordinate (west)
        miny: Minimum y coordinate (south)
        maxx: Maximum x coordinate (east)
        maxy: Maximum y coordinate (north)
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        """Width of bounds."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Height of bounds."""
        return self.maxy - self.miny

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of bounds (x, y)."""
        return (
            (self.minx + self.maxx) / 2.0,
            (self.miny + self.maxy) / 2.0,
        )

    @property
    def area(self) -> float:
        """Area of bounds."""
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounds as tuple (minx, miny, maxx, maxy)."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other: "TileBounds") -> bool:
        """Check if bounds intersect another bounds."""
        return not (
            self.maxx < other.minx
            or self.minx > other.maxx
            or self.maxy < other.miny
            or self.miny > other.maxy
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if bounds contain a point."""
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def intersection(self, other: "TileBounds") -> Optional["TileBounds"]:
        """Get intersection with another bounds, or None if no intersection."""
        if not self.intersects(other):
            return None
        return TileBounds(
            minx=max(self.minx, other.minx),
            miny=max(self.miny, other.miny),
            maxx=min(self.maxx, other.maxx),
            maxy=min(self.maxy, other.maxy),
        )

    def union(self, other: "TileBounds") -> "TileBounds":
        """Get union (bounding box) with another bounds."""
        return TileBounds(
            minx=min(self.minx, other.minx),
            miny=min(self.miny, other.miny),
            maxx=max(self.maxx, other.maxx),
            maxy=max(self.maxy, other.maxy),
        )

    def buffer(self, amount: float) -> "TileBounds":
        """Expand bounds by given amount in all directions."""
        return TileBounds(
            minx=self.minx - amount,
            miny=self.miny - amount,
            maxx=self.maxx + amount,
            maxy=self.maxy + amount,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "minx": self.minx,
            "miny": self.miny,
            "maxx": self.maxx,
            "maxy": self.maxy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TileBounds":
        """Create from dictionary."""
        return cls(
            minx=data["minx"],
            miny=data["miny"],
            maxx=data["maxx"],
            maxy=data["maxy"],
        )


@dataclass
class PixelBounds:
    """
    Bounds in pixel coordinates.

    Attributes:
        col_start: Starting column (inclusive)
        row_start: Starting row (inclusive)
        col_end: Ending column (exclusive)
        row_end: Ending row (exclusive)
    """

    col_start: int
    row_start: int
    col_end: int
    row_end: int

    @property
    def width(self) -> int:
        """Width in pixels."""
        return self.col_end - self.col_start

    @property
    def height(self) -> int:
        """Height in pixels."""
        return self.row_end - self.row_start

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape as (height, width)."""
        return (self.height, self.width)

    def to_slice(self) -> Tuple[slice, slice]:
        """Convert to numpy slices (row_slice, col_slice)."""
        return (slice(self.row_start, self.row_end), slice(self.col_start, self.col_end))

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "col_start": self.col_start,
            "row_start": self.row_start,
            "col_end": self.col_end,
            "row_end": self.row_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "PixelBounds":
        """Create from dictionary."""
        return cls(
            col_start=data["col_start"],
            row_start=data["row_start"],
            col_end=data["col_end"],
            row_end=data["row_end"],
        )


@dataclass
class TileInfo:
    """
    Complete information about a tile.

    Attributes:
        index: Tile index in the grid
        geo_bounds: Geographic/projected bounds
        pixel_bounds: Pixel bounds in the source image
        overlap_geo_bounds: Geographic bounds including overlap
        overlap_pixel_bounds: Pixel bounds including overlap
        status: Current processing status
        metadata: Additional metadata
    """

    index: TileIndex
    geo_bounds: TileBounds
    pixel_bounds: PixelBounds
    overlap_geo_bounds: Optional[TileBounds] = None
    overlap_pixel_bounds: Optional[PixelBounds] = None
    status: TileStatus = TileStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_overlap(self) -> bool:
        """Check if tile has overlap region."""
        return self.overlap_geo_bounds is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index.to_dict(),
            "geo_bounds": self.geo_bounds.to_dict(),
            "pixel_bounds": self.pixel_bounds.to_dict(),
            "overlap_geo_bounds": self.overlap_geo_bounds.to_dict() if self.overlap_geo_bounds else None,
            "overlap_pixel_bounds": self.overlap_pixel_bounds.to_dict() if self.overlap_pixel_bounds else None,
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TileInfo":
        """Create from dictionary."""
        return cls(
            index=TileIndex.from_dict(data["index"]),
            geo_bounds=TileBounds.from_dict(data["geo_bounds"]),
            pixel_bounds=PixelBounds.from_dict(data["pixel_bounds"]),
            overlap_geo_bounds=TileBounds.from_dict(data["overlap_geo_bounds"]) if data.get("overlap_geo_bounds") else None,
            overlap_pixel_bounds=PixelBounds.from_dict(data["overlap_pixel_bounds"]) if data.get("overlap_pixel_bounds") else None,
            status=TileStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# TileScheme
# =============================================================================


@dataclass
class TileScheme:
    """
    Defines tile sizes and overlap configuration.

    Attributes:
        tile_size: Size of tiles in pixels (width, height)
        overlap: Overlap in pixels on each side
        crs: Coordinate reference system (EPSG code)
        origin: Origin position for grid
    """

    tile_size: Tuple[int, int] = (512, 512)
    overlap: int = 0
    crs: str = "EPSG:4326"
    origin: str = "top_left"

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.tile_size, int):
            self.tile_size = (self.tile_size, self.tile_size)

        if self.tile_size[0] < 16 or self.tile_size[1] < 16:
            raise ValueError(f"Tile size must be at least 16x16, got {self.tile_size}")

        if self.overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {self.overlap}")

        if self.overlap >= min(self.tile_size) // 2:
            raise ValueError(
                f"Overlap ({self.overlap}) must be less than half the tile size ({min(self.tile_size) // 2})"
            )

    @classmethod
    def from_preset(
        cls,
        preset: TileSizePreset,
        overlap: int = 0,
        crs: str = "EPSG:4326",
    ) -> "TileScheme":
        """Create scheme from preset."""
        return cls(tile_size=preset.value, overlap=overlap, crs=crs)

    @property
    def tile_width(self) -> int:
        """Tile width in pixels."""
        return self.tile_size[0]

    @property
    def tile_height(self) -> int:
        """Tile height in pixels."""
        return self.tile_size[1]

    @property
    def effective_tile_width(self) -> int:
        """Effective tile width (accounting for overlap)."""
        return self.tile_size[0] - 2 * self.overlap

    @property
    def effective_tile_height(self) -> int:
        """Effective tile height (accounting for overlap)."""
        return self.tile_size[1] - 2 * self.overlap

    @property
    def total_overlap_width(self) -> int:
        """Total overlap width (both sides)."""
        return 2 * self.overlap

    @property
    def total_overlap_height(self) -> int:
        """Total overlap height (both sides)."""
        return 2 * self.overlap

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tile_size": list(self.tile_size),
            "overlap": self.overlap,
            "crs": self.crs,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TileScheme":
        """Create from dictionary."""
        tile_size = data.get("tile_size", (512, 512))
        if isinstance(tile_size, list):
            tile_size = tuple(tile_size)
        return cls(
            tile_size=tile_size,
            overlap=data.get("overlap", 0),
            crs=data.get("crs", "EPSG:4326"),
            origin=data.get("origin", "top_left"),
        )


# =============================================================================
# TileGrid
# =============================================================================


class TileGrid:
    """
    Generate tile grid covering an Area of Interest.

    Manages the partitioning of a geographic area into tiles,
    supporting iteration, indexing, and neighbor lookups.

    Example:
        grid = TileGrid(
            bounds=(-122.5, 37.5, -122.0, 38.0),
            resolution=(0.001, 0.001),
            scheme=TileScheme(tile_size=(512, 512), overlap=32),
        )

        # Get all tiles
        for tile in grid:
            process_tile(tile)

        # Get tile by index
        tile = grid.get_tile(TileIndex(col=5, row=10))

        # Get neighboring tiles
        neighbors = grid.get_neighbors(tile.index)
    """

    def __init__(
        self,
        bounds: Union[Tuple[float, float, float, float], TileBounds],
        resolution: Tuple[float, float],
        scheme: Optional[TileScheme] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize tile grid.

        Args:
            bounds: Geographic bounds (minx, miny, maxx, maxy) or TileBounds
            resolution: Pixel resolution (x_res, y_res) in CRS units per pixel
            scheme: Tile scheme configuration
            image_shape: Optional source image shape (height, width) for pixel calculations
        """
        if isinstance(bounds, TileBounds):
            self.bounds = bounds
        else:
            self.bounds = TileBounds(*bounds)

        self.resolution = resolution
        self.scheme = scheme or TileScheme()

        # Calculate image dimensions from bounds and resolution if not provided
        if image_shape is None:
            self.image_width = int(np.ceil(self.bounds.width / abs(self.resolution[0])))
            self.image_height = int(np.ceil(self.bounds.height / abs(self.resolution[1])))
        else:
            self.image_height, self.image_width = image_shape

        # Calculate grid dimensions
        self._calculate_grid()

    def _calculate_grid(self) -> None:
        """Calculate grid dimensions and tile bounds."""
        scheme = self.scheme

        # Calculate number of tiles needed
        effective_width = scheme.effective_tile_width
        effective_height = scheme.effective_tile_height

        # Number of columns and rows
        self.n_cols = max(1, int(np.ceil(self.image_width / effective_width)))
        self.n_rows = max(1, int(np.ceil(self.image_height / effective_height)))
        self.n_tiles = self.n_cols * self.n_rows

        # Calculate geo tile size
        self.geo_tile_width = scheme.tile_width * abs(self.resolution[0])
        self.geo_tile_height = scheme.tile_height * abs(self.resolution[1])
        self.geo_effective_width = effective_width * abs(self.resolution[0])
        self.geo_effective_height = effective_height * abs(self.resolution[1])
        self.geo_overlap_x = scheme.overlap * abs(self.resolution[0])
        self.geo_overlap_y = scheme.overlap * abs(self.resolution[1])

        logger.debug(
            f"Grid initialized: {self.n_cols}x{self.n_rows} = {self.n_tiles} tiles"
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (rows, cols)."""
        return (self.n_rows, self.n_cols)

    @property
    def total_tiles(self) -> int:
        """Total number of tiles."""
        return self.n_tiles

    def get_tile(self, index: TileIndex) -> TileInfo:
        """
        Get tile information by index.

        Args:
            index: Tile index (col, row)

        Returns:
            TileInfo for the requested tile

        Raises:
            IndexError: If index is out of bounds
        """
        if not self.is_valid_index(index):
            raise IndexError(f"Tile index {index} out of bounds (0-{self.n_cols - 1}, 0-{self.n_rows - 1})")

        # Calculate pixel bounds (core region without overlap)
        effective_width = self.scheme.effective_tile_width
        effective_height = self.scheme.effective_tile_height

        col_start = index.col * effective_width
        row_start = index.row * effective_height
        col_end = min(col_start + self.scheme.tile_width, self.image_width)
        row_end = min(row_start + self.scheme.tile_height, self.image_height)

        pixel_bounds = PixelBounds(
            col_start=col_start,
            row_start=row_start,
            col_end=col_end,
            row_end=row_end,
        )

        # Calculate geographic bounds
        geo_bounds = self._pixel_to_geo_bounds(pixel_bounds)

        # Calculate overlap bounds
        overlap_pixel_bounds = None
        overlap_geo_bounds = None
        if self.scheme.overlap > 0:
            overlap_col_start = max(0, col_start - self.scheme.overlap)
            overlap_row_start = max(0, row_start - self.scheme.overlap)
            overlap_col_end = min(col_end + self.scheme.overlap, self.image_width)
            overlap_row_end = min(row_end + self.scheme.overlap, self.image_height)

            overlap_pixel_bounds = PixelBounds(
                col_start=overlap_col_start,
                row_start=overlap_row_start,
                col_end=overlap_col_end,
                row_end=overlap_row_end,
            )
            overlap_geo_bounds = self._pixel_to_geo_bounds(overlap_pixel_bounds)

        return TileInfo(
            index=index,
            geo_bounds=geo_bounds,
            pixel_bounds=pixel_bounds,
            overlap_geo_bounds=overlap_geo_bounds,
            overlap_pixel_bounds=overlap_pixel_bounds,
        )

    def _pixel_to_geo_bounds(self, pixel_bounds: PixelBounds) -> TileBounds:
        """Convert pixel bounds to geographic bounds."""
        if self.scheme.origin == "top_left":
            minx = self.bounds.minx + pixel_bounds.col_start * abs(self.resolution[0])
            maxx = self.bounds.minx + pixel_bounds.col_end * abs(self.resolution[0])
            maxy = self.bounds.maxy - pixel_bounds.row_start * abs(self.resolution[1])
            miny = self.bounds.maxy - pixel_bounds.row_end * abs(self.resolution[1])
        else:
            # bottom_left origin
            minx = self.bounds.minx + pixel_bounds.col_start * abs(self.resolution[0])
            maxx = self.bounds.minx + pixel_bounds.col_end * abs(self.resolution[0])
            miny = self.bounds.miny + pixel_bounds.row_start * abs(self.resolution[1])
            maxy = self.bounds.miny + pixel_bounds.row_end * abs(self.resolution[1])

        return TileBounds(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def get_tile_at_point(self, x: float, y: float) -> Optional[TileIndex]:
        """
        Get tile index containing a geographic point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            TileIndex or None if point is outside grid
        """
        if not self.bounds.contains(x, y):
            return None

        effective_width = self.scheme.effective_tile_width
        effective_height = self.scheme.effective_tile_height

        if self.scheme.origin == "top_left":
            col = int((x - self.bounds.minx) / (effective_width * abs(self.resolution[0])))
            row = int((self.bounds.maxy - y) / (effective_height * abs(self.resolution[1])))
        else:
            col = int((x - self.bounds.minx) / (effective_width * abs(self.resolution[0])))
            row = int((y - self.bounds.miny) / (effective_height * abs(self.resolution[1])))

        col = max(0, min(col, self.n_cols - 1))
        row = max(0, min(row, self.n_rows - 1))

        return TileIndex(col=col, row=row)

    def get_tile_at_pixel(self, col: int, row: int) -> Optional[TileIndex]:
        """
        Get tile index containing a pixel position.

        Args:
            col: Pixel column
            row: Pixel row

        Returns:
            TileIndex or None if pixel is outside image
        """
        if col < 0 or col >= self.image_width or row < 0 or row >= self.image_height:
            return None

        effective_width = self.scheme.effective_tile_width
        effective_height = self.scheme.effective_tile_height

        tile_col = col // effective_width
        tile_row = row // effective_height

        return TileIndex(
            col=min(tile_col, self.n_cols - 1),
            row=min(tile_row, self.n_rows - 1),
        )

    def is_valid_index(self, index: TileIndex) -> bool:
        """Check if tile index is valid."""
        return 0 <= index.col < self.n_cols and 0 <= index.row < self.n_rows

    def get_neighbors(
        self,
        index: TileIndex,
        include_diagonal: bool = True,
    ) -> List[TileIndex]:
        """
        Get neighboring tile indices.

        Args:
            index: Center tile index
            include_diagonal: Include diagonal neighbors

        Returns:
            List of neighboring tile indices
        """
        neighbors = []

        # Direct neighbors (4-connected)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if include_diagonal:
            # Add diagonal neighbors (8-connected)
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dc, dr in offsets:
            new_col = index.col + dc
            new_row = index.row + dr

            if 0 <= new_col < self.n_cols and 0 <= new_row < self.n_rows:
                neighbors.append(TileIndex(col=new_col, row=new_row))

        return neighbors

    def tiles_in_bounds(
        self,
        query_bounds: Union[Tuple[float, float, float, float], TileBounds],
    ) -> Generator[TileInfo, None, None]:
        """
        Get tiles that intersect given bounds.

        Args:
            query_bounds: Geographic bounds to query

        Yields:
            TileInfo for each intersecting tile
        """
        if not isinstance(query_bounds, TileBounds):
            query_bounds = TileBounds(*query_bounds)

        # Find potential tile range
        min_idx = self.get_tile_at_point(query_bounds.minx, query_bounds.maxy)
        max_idx = self.get_tile_at_point(query_bounds.maxx, query_bounds.miny)

        if min_idx is None or max_idx is None:
            return

        # Expand range to be safe
        start_col = max(0, min_idx.col - 1)
        start_row = max(0, min_idx.row - 1)
        end_col = min(self.n_cols, max_idx.col + 2)
        end_row = min(self.n_rows, max_idx.row + 2)

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                tile = self.get_tile(TileIndex(col=col, row=row))
                if tile.geo_bounds.intersects(query_bounds):
                    yield tile

    def __iter__(self) -> Iterator[TileInfo]:
        """Iterate over all tiles in row-major order."""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield self.get_tile(TileIndex(col=col, row=row))

    def __len__(self) -> int:
        """Number of tiles."""
        return self.n_tiles

    def __getitem__(self, key: Union[TileIndex, Tuple[int, int]]) -> TileInfo:
        """Get tile by index."""
        if isinstance(key, tuple):
            key = TileIndex(col=key[0], row=key[1])
        return self.get_tile(key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert grid configuration to dictionary."""
        return {
            "bounds": self.bounds.to_dict(),
            "resolution": list(self.resolution),
            "scheme": self.scheme.to_dict(),
            "n_cols": self.n_cols,
            "n_rows": self.n_rows,
            "n_tiles": self.n_tiles,
            "image_shape": (self.image_height, self.image_width),
        }


# =============================================================================
# TileManager
# =============================================================================


class TileManager:
    """
    Manage tile processing state for resumable processing.

    Tracks completed/pending tiles, enables resume from partial completion,
    and provides utilities for merging tile results.

    Example:
        manager = TileManager(grid, workdir=Path("./processing"))

        # Get pending tiles
        for tile in manager.pending_tiles():
            result = process(tile)
            manager.mark_completed(tile.index, result)

        # Resume after interruption
        manager = TileManager.from_state(Path("./processing/state.json"))
        for tile in manager.pending_tiles():
            ...

        # Get summary
        print(manager.get_progress())
    """

    def __init__(
        self,
        grid: TileGrid,
        workdir: Optional[Path] = None,
        use_db: bool = True,
    ):
        """
        Initialize tile manager.

        Args:
            grid: Tile grid to manage
            workdir: Working directory for state persistence
            use_db: Use SQLite database for state (vs JSON file)
        """
        self.grid = grid
        self.workdir = Path(workdir) if workdir else Path.cwd() / ".tiles"
        self.use_db = use_db

        # Thread safety
        self._lock = threading.Lock()

        # State tracking
        self._status: Dict[TileIndex, TileStatus] = {}
        self._results: Dict[TileIndex, Any] = {}
        self._errors: Dict[TileIndex, str] = {}
        self._timestamps: Dict[TileIndex, datetime] = {}

        # Create workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Initialize or load state
        if use_db:
            self._init_db()
        else:
            self._state_file = self.workdir / "tile_state.json"
            if self._state_file.exists():
                self._load_state_json()

    def _init_db(self) -> None:
        """Initialize SQLite database for state tracking."""
        self._db_path = self.workdir / "tile_state.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)

        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS tile_status (
                    col INTEGER,
                    row_ INTEGER,
                    level INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    result_path TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (col, row_, level)
                )
            """)

            # Load existing state
            cursor = self._conn.execute("SELECT col, row_, level, status, error FROM tile_status")
            for row in cursor:
                index = TileIndex(col=row[0], row=row[1], level=row[2])
                self._status[index] = TileStatus(row[3])
                if row[4]:
                    self._errors[index] = row[4]

    def _save_state_db(self, index: TileIndex) -> None:
        """Save tile state to database."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tile_status (col, row_, level, status, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    index.col,
                    index.row,
                    index.level,
                    self._status.get(index, TileStatus.PENDING).value,
                    self._errors.get(index),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _load_state_json(self) -> None:
        """Load state from JSON file."""
        with open(self._state_file, "r") as f:
            data = json.load(f)

        for item in data.get("tiles", []):
            index = TileIndex.from_dict(item["index"])
            self._status[index] = TileStatus(item["status"])
            if item.get("error"):
                self._errors[index] = item["error"]

    def _save_state_json(self) -> None:
        """Save state to JSON file."""
        tiles = []
        for index, status in self._status.items():
            tiles.append({
                "index": index.to_dict(),
                "status": status.value,
                "error": self._errors.get(index),
            })

        data = {
            "grid": self.grid.to_dict(),
            "tiles": tiles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self._state_file, "w") as f:
            json.dump(data, f, indent=2)

    def mark_in_progress(self, index: TileIndex) -> None:
        """Mark tile as in progress."""
        with self._lock:
            self._status[index] = TileStatus.IN_PROGRESS
            self._timestamps[index] = datetime.now(timezone.utc)

            if self.use_db:
                self._save_state_db(index)
            else:
                self._save_state_json()

    def mark_completed(
        self,
        index: TileIndex,
        result: Any = None,
        result_path: Optional[Path] = None,
    ) -> None:
        """
        Mark tile as completed.

        Args:
            index: Tile index
            result: Optional result data
            result_path: Optional path to result file
        """
        with self._lock:
            self._status[index] = TileStatus.COMPLETED
            if result is not None:
                self._results[index] = result
            self._timestamps[index] = datetime.now(timezone.utc)

            if self.use_db:
                self._save_state_db(index)
            else:
                self._save_state_json()

    def mark_failed(self, index: TileIndex, error: str) -> None:
        """Mark tile as failed."""
        with self._lock:
            self._status[index] = TileStatus.FAILED
            self._errors[index] = error
            self._timestamps[index] = datetime.now(timezone.utc)

            if self.use_db:
                self._save_state_db(index)
            else:
                self._save_state_json()

    def mark_skipped(self, index: TileIndex, reason: str = "") -> None:
        """Mark tile as skipped."""
        with self._lock:
            self._status[index] = TileStatus.SKIPPED
            if reason:
                self._errors[index] = reason
            self._timestamps[index] = datetime.now(timezone.utc)

            if self.use_db:
                self._save_state_db(index)
            else:
                self._save_state_json()

    def get_status(self, index: TileIndex) -> TileStatus:
        """Get status of a tile."""
        return self._status.get(index, TileStatus.PENDING)

    def get_result(self, index: TileIndex) -> Optional[Any]:
        """Get result for a tile."""
        return self._results.get(index)

    def get_error(self, index: TileIndex) -> Optional[str]:
        """Get error message for a tile."""
        return self._errors.get(index)

    def pending_tiles(self) -> Generator[TileInfo, None, None]:
        """
        Iterate over pending tiles.

        Yields:
            TileInfo for each pending tile
        """
        for tile in self.grid:
            if self.get_status(tile.index) == TileStatus.PENDING:
                yield tile

    def completed_tiles(self) -> Generator[TileInfo, None, None]:
        """
        Iterate over completed tiles.

        Yields:
            TileInfo for each completed tile
        """
        for tile in self.grid:
            if self.get_status(tile.index) == TileStatus.COMPLETED:
                yield tile

    def failed_tiles(self) -> Generator[TileInfo, None, None]:
        """
        Iterate over failed tiles.

        Yields:
            TileInfo for each failed tile
        """
        for tile in self.grid:
            if self.get_status(tile.index) == TileStatus.FAILED:
                yield tile

    def reset_failed(self) -> int:
        """
        Reset failed tiles to pending.

        Returns:
            Number of tiles reset
        """
        count = 0
        with self._lock:
            for index in list(self._status.keys()):
                if self._status[index] == TileStatus.FAILED:
                    self._status[index] = TileStatus.PENDING
                    self._errors.pop(index, None)
                    count += 1

            if self.use_db:
                with self._conn:
                    self._conn.execute(
                        "UPDATE tile_status SET status = ? WHERE status = ?",
                        (TileStatus.PENDING.value, TileStatus.FAILED.value),
                    )
            else:
                self._save_state_json()

        return count

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress summary.

        Returns:
            Dictionary with progress statistics
        """
        total = len(self.grid)
        completed = sum(1 for s in self._status.values() if s == TileStatus.COMPLETED)
        failed = sum(1 for s in self._status.values() if s == TileStatus.FAILED)
        skipped = sum(1 for s in self._status.values() if s == TileStatus.SKIPPED)
        in_progress = sum(1 for s in self._status.values() if s == TileStatus.IN_PROGRESS)
        pending = total - completed - failed - skipped - in_progress

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "in_progress": in_progress,
            "pending": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
        }

    def merge_results(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        dtype: np.dtype = np.float32,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Merge tile results into a single array.

        Args:
            output_shape: Shape of output array (height, width)
            dtype: Output data type
            fill_value: Fill value for missing tiles

        Returns:
            Merged array
        """
        if output_shape is None:
            output_shape = (self.grid.image_height, self.grid.image_width)

        output = np.full(output_shape, fill_value, dtype=dtype)

        for tile in self.completed_tiles():
            result = self.get_result(tile.index)
            if result is None:
                continue

            # Get result data
            if hasattr(result, "data"):
                data = result.data
            elif isinstance(result, np.ndarray):
                data = result
            else:
                continue

            # Handle overlap - use core region only
            if tile.overlap_pixel_bounds is not None:
                overlap = self.grid.scheme.overlap
                core_start_row = overlap if tile.index.row > 0 else 0
                core_start_col = overlap if tile.index.col > 0 else 0
                core_end_row = data.shape[0] - overlap if tile.index.row < self.grid.n_rows - 1 else data.shape[0]
                core_end_col = data.shape[1] - overlap if tile.index.col < self.grid.n_cols - 1 else data.shape[1]
                data = data[core_start_row:core_end_row, core_start_col:core_end_col]

            # Copy to output
            pb = tile.pixel_bounds
            row_slice, col_slice = pb.to_slice()
            output[row_slice, col_slice] = data

        return output

    def close(self) -> None:
        """Close resources."""
        if self.use_db and hasattr(self, "_conn"):
            self._conn.close()

    def __enter__(self) -> "TileManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# OverlapHandler
# =============================================================================


class OverlapHandler:
    """
    Handle overlap regions for seamless tile blending.

    Provides methods for blending overlapping tile edges to avoid
    visible seams in the merged output.

    Example:
        handler = OverlapHandler(
            overlap=32,
            blend_mode=BlendMode.FEATHER,
        )

        # Blend two adjacent tiles
        blended = handler.blend_tiles(left_tile, right_tile, direction="horizontal")

        # Create blend mask
        mask = handler.create_blend_mask((512, 512), direction="horizontal")
    """

    def __init__(
        self,
        overlap: int,
        blend_mode: BlendMode = BlendMode.FEATHER,
    ):
        """
        Initialize overlap handler.

        Args:
            overlap: Overlap size in pixels
            blend_mode: Blending mode to use
        """
        self.overlap = overlap
        self.blend_mode = blend_mode

    def create_blend_mask(
        self,
        shape: Tuple[int, int],
        direction: str = "horizontal",
    ) -> np.ndarray:
        """
        Create a blend mask for tile transitions.

        Args:
            shape: Shape of the mask (height, width)
            direction: "horizontal" or "vertical"

        Returns:
            Blend mask array with values 0-1
        """
        mask = np.ones(shape, dtype=np.float32)

        if self.blend_mode == BlendMode.NONE:
            return mask

        if direction == "horizontal":
            # Create linear ramp in x direction
            if shape[1] > 0:
                ramp = np.linspace(0, 1, self.overlap)
                mask[:, :self.overlap] = ramp[np.newaxis, :]
                mask[:, -self.overlap:] = ramp[::-1][np.newaxis, :]
        else:
            # Create linear ramp in y direction
            if shape[0] > 0:
                ramp = np.linspace(0, 1, self.overlap)
                mask[:self.overlap, :] = ramp[:, np.newaxis]
                mask[-self.overlap:, :] = ramp[::-1][:, np.newaxis]

        if self.blend_mode == BlendMode.FEATHER:
            # Apply smooth feathering (cosine)
            mask = (1 - np.cos(mask * np.pi)) / 2

        return mask

    def blend_tiles(
        self,
        tile1: np.ndarray,
        tile2: np.ndarray,
        direction: str = "horizontal",
    ) -> np.ndarray:
        """
        Blend two overlapping tiles.

        Args:
            tile1: First tile (left or top)
            tile2: Second tile (right or bottom)
            direction: "horizontal" or "vertical"

        Returns:
            Blended overlap region
        """
        if self.blend_mode == BlendMode.NONE:
            return tile2

        if self.blend_mode == BlendMode.PRIORITY_LEFT:
            return tile1

        if self.blend_mode == BlendMode.PRIORITY_RIGHT:
            return tile2

        if self.blend_mode == BlendMode.MAX:
            return np.maximum(tile1, tile2)

        if self.blend_mode == BlendMode.MIN:
            return np.minimum(tile1, tile2)

        if self.blend_mode == BlendMode.AVERAGE:
            return (tile1 + tile2) / 2

        # Feather blend
        mask = self.create_blend_mask(tile1.shape, direction)
        return tile1 * (1 - mask) + tile2 * mask

    def extract_overlap_region(
        self,
        tile: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Extract overlap region from a tile.

        Args:
            tile: Tile data array
            position: "left", "right", "top", or "bottom"

        Returns:
            Overlap region
        """
        if position == "left":
            return tile[:, :self.overlap]
        elif position == "right":
            return tile[:, -self.overlap:]
        elif position == "top":
            return tile[:self.overlap, :]
        elif position == "bottom":
            return tile[-self.overlap:, :]
        else:
            raise ValueError(f"Invalid position: {position}")

    def apply_overlap(
        self,
        tile: np.ndarray,
        position: str,
        neighbor_overlap: np.ndarray,
    ) -> np.ndarray:
        """
        Apply blending with neighbor's overlap region.

        Args:
            tile: Current tile data
            position: Position of overlap ("left", "right", "top", "bottom")
            neighbor_overlap: Overlap region from neighboring tile

        Returns:
            Tile with blended overlap
        """
        result = tile.copy()

        if position == "left":
            blended = self.blend_tiles(neighbor_overlap, tile[:, :self.overlap], "horizontal")
            result[:, :self.overlap] = blended
        elif position == "right":
            blended = self.blend_tiles(tile[:, -self.overlap:], neighbor_overlap, "horizontal")
            result[:, -self.overlap:] = blended
        elif position == "top":
            blended = self.blend_tiles(neighbor_overlap, tile[:self.overlap, :], "vertical")
            result[:self.overlap, :] = blended
        elif position == "bottom":
            blended = self.blend_tiles(tile[-self.overlap:, :], neighbor_overlap, "vertical")
            result[-self.overlap:, :] = blended

        return result


# =============================================================================
# Utility Functions
# =============================================================================


def create_tile_grid(
    bounds: Tuple[float, float, float, float],
    resolution: Tuple[float, float],
    tile_size: Union[int, Tuple[int, int]] = 512,
    overlap: int = 0,
    crs: str = "EPSG:4326",
) -> TileGrid:
    """
    Convenience function to create a tile grid.

    Args:
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        resolution: Pixel resolution (x_res, y_res)
        tile_size: Tile size in pixels (or tuple for width, height)
        overlap: Overlap in pixels
        crs: Coordinate reference system

    Returns:
        TileGrid instance
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    scheme = TileScheme(tile_size=tile_size, overlap=overlap, crs=crs)
    return TileGrid(bounds=bounds, resolution=resolution, scheme=scheme)


def estimate_memory_per_tile(
    tile_size: Tuple[int, int],
    n_bands: int = 1,
    dtype: np.dtype = np.float32,
    overlap: int = 0,
) -> int:
    """
    Estimate memory usage per tile in bytes.

    Args:
        tile_size: Tile size (width, height)
        n_bands: Number of bands
        dtype: Data type
        overlap: Overlap pixels

    Returns:
        Estimated memory in bytes
    """
    effective_size = (tile_size[0] + 2 * overlap, tile_size[1] + 2 * overlap)
    bytes_per_element = np.dtype(dtype).itemsize
    return effective_size[0] * effective_size[1] * n_bands * bytes_per_element


def suggest_tile_size(
    image_shape: Tuple[int, int],
    available_memory_mb: int = 2048,
    n_bands: int = 1,
    dtype: np.dtype = np.float32,
    overhead_factor: float = 3.0,
) -> Tuple[int, int]:
    """
    Suggest optimal tile size for available memory.

    Args:
        image_shape: Image shape (height, width)
        available_memory_mb: Available memory in MB
        n_bands: Number of bands
        dtype: Data type
        overhead_factor: Safety factor for processing overhead

    Returns:
        Suggested tile size (width, height)
    """
    available_bytes = available_memory_mb * 1024 * 1024 / overhead_factor
    bytes_per_pixel = np.dtype(dtype).itemsize * n_bands

    max_pixels_per_tile = available_bytes / bytes_per_pixel
    max_side = int(np.sqrt(max_pixels_per_tile))

    # Round down to nearest power of 2
    power = int(np.log2(max_side))
    suggested_size = 2 ** power

    # Ensure reasonable bounds
    suggested_size = max(128, min(suggested_size, 4096))

    # Don't exceed image dimensions
    suggested_size = min(suggested_size, max(image_shape))

    return (suggested_size, suggested_size)


def compute_grid_hash(grid: TileGrid) -> str:
    """
    Compute a hash for a tile grid configuration.

    Args:
        grid: Tile grid

    Returns:
        Hash string (16 characters)
    """
    data = json.dumps(grid.to_dict(), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]
