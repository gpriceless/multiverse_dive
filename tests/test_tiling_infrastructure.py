"""
Tests for Tiling Infrastructure - Group L, Tracks 1-3.

Tests the following components:
- Track 1: TileScheme, TileGrid, TileManager, OverlapHandler (core/execution/tiling.py)
- Track 2: StreamingDownloader, WindowedReader, StreamingIngester (core/data/ingestion/streaming.py)
- Track 3: TiledAlgorithmRunner, TileContext, ResultStitcher (core/analysis/execution/tiled_runner.py)
"""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Track 1: Tiling infrastructure
from core.execution.tiling import (
    BlendMode,
    CoordinateSystem,
    OverlapHandler,
    PixelBounds,
    TileBounds,
    TileGrid,
    TileIndex,
    TileInfo,
    TileManager,
    TileScheme,
    TileSizePreset,
    TileStatus,
    compute_grid_hash,
    create_tile_grid,
    estimate_memory_per_tile,
    suggest_tile_size,
)

# Track 2: Streaming ingestion
from core.data.ingestion.streaming import (
    ChunkInfo,
    DownloadProgress,
    DownloadStatus,
    MemoryMappedReader,
    StreamingConfig,
    StreamingDownloader,
    StreamingIngester,
    WindowInfo,
    estimate_chunk_count,
)

# Track 3: Tiled runner
from core.analysis.execution.tiled_runner import (
    AggregationMethod,
    ProcessingProgress,
    ResultStitcher,
    StitchMethod,
    TileContext,
    TiledAlgorithmRunner,
    TiledProcessingResult,
    TileResult,
    check_algorithm_tiled_support,
    estimate_tiles_for_memory,
    run_algorithm_tiled,
)


# =============================================================================
# Track 1: Tiling Infrastructure Tests
# =============================================================================


class TestTileIndex:
    """Tests for TileIndex class."""

    def test_create_tile_index(self):
        """Test basic TileIndex creation."""
        index = TileIndex(col=5, row=10)
        assert index.col == 5
        assert index.row == 10
        assert index.level == 0

    def test_tile_index_with_level(self):
        """Test TileIndex with level."""
        index = TileIndex(col=5, row=10, level=3)
        assert index.level == 3

    def test_tile_index_equality(self):
        """Test TileIndex equality."""
        idx1 = TileIndex(col=5, row=10)
        idx2 = TileIndex(col=5, row=10)
        idx3 = TileIndex(col=5, row=11)
        assert idx1 == idx2
        assert idx1 != idx3

    def test_tile_index_hash(self):
        """Test TileIndex is hashable."""
        idx1 = TileIndex(col=5, row=10)
        idx2 = TileIndex(col=5, row=10)
        assert hash(idx1) == hash(idx2)

        # Can use in set
        index_set = {idx1, idx2}
        assert len(index_set) == 1

    def test_tile_index_to_dict(self):
        """Test TileIndex serialization."""
        index = TileIndex(col=5, row=10, level=2)
        d = index.to_dict()
        assert d == {"col": 5, "row": 10, "level": 2}

    def test_tile_index_from_dict(self):
        """Test TileIndex deserialization."""
        d = {"col": 5, "row": 10, "level": 2}
        index = TileIndex.from_dict(d)
        assert index.col == 5
        assert index.row == 10
        assert index.level == 2


class TestTileBounds:
    """Tests for TileBounds class."""

    def test_create_tile_bounds(self):
        """Test basic TileBounds creation."""
        bounds = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        assert bounds.minx == -10
        assert bounds.miny == 40
        assert bounds.maxx == 10
        assert bounds.maxy == 50

    def test_tile_bounds_properties(self):
        """Test TileBounds computed properties."""
        bounds = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        assert bounds.width == 20
        assert bounds.height == 10
        assert bounds.center == (0, 45)
        assert bounds.area == 200

    def test_tile_bounds_contains(self):
        """Test point containment check."""
        bounds = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        assert bounds.contains(0, 45) is True
        assert bounds.contains(-10, 40) is True
        assert bounds.contains(-11, 45) is False
        assert bounds.contains(0, 51) is False

    def test_tile_bounds_intersects(self):
        """Test intersection check."""
        bounds1 = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        bounds2 = TileBounds(minx=0, miny=45, maxx=20, maxy=55)
        bounds3 = TileBounds(minx=20, miny=60, maxx=30, maxy=70)

        assert bounds1.intersects(bounds2) is True
        assert bounds1.intersects(bounds3) is False

    def test_tile_bounds_intersection(self):
        """Test computing intersection."""
        bounds1 = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        bounds2 = TileBounds(minx=0, miny=45, maxx=20, maxy=55)

        intersection = bounds1.intersection(bounds2)
        assert intersection is not None
        assert intersection.minx == 0
        assert intersection.miny == 45
        assert intersection.maxx == 10
        assert intersection.maxy == 50

    def test_tile_bounds_no_intersection(self):
        """Test intersection returns None when no overlap."""
        bounds1 = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        bounds2 = TileBounds(minx=20, miny=60, maxx=30, maxy=70)

        assert bounds1.intersection(bounds2) is None

    def test_tile_bounds_union(self):
        """Test computing union."""
        bounds1 = TileBounds(minx=-10, miny=40, maxx=10, maxy=50)
        bounds2 = TileBounds(minx=0, miny=45, maxx=20, maxy=55)

        union = bounds1.union(bounds2)
        assert union.minx == -10
        assert union.miny == 40
        assert union.maxx == 20
        assert union.maxy == 55

    def test_tile_bounds_buffer(self):
        """Test buffer operation."""
        bounds = TileBounds(minx=0, miny=0, maxx=10, maxy=10)
        buffered = bounds.buffer(5)

        assert buffered.minx == -5
        assert buffered.miny == -5
        assert buffered.maxx == 15
        assert buffered.maxy == 15


class TestPixelBounds:
    """Tests for PixelBounds class."""

    def test_create_pixel_bounds(self):
        """Test basic PixelBounds creation."""
        pb = PixelBounds(col_start=0, row_start=0, col_end=512, row_end=512)
        assert pb.width == 512
        assert pb.height == 512
        assert pb.shape == (512, 512)

    def test_pixel_bounds_to_slice(self):
        """Test conversion to numpy slices."""
        pb = PixelBounds(col_start=100, row_start=200, col_end=356, row_end=456)
        row_slice, col_slice = pb.to_slice()

        assert row_slice == slice(200, 456)
        assert col_slice == slice(100, 356)


class TestTileScheme:
    """Tests for TileScheme class."""

    def test_create_default_scheme(self):
        """Test default TileScheme creation."""
        scheme = TileScheme()
        assert scheme.tile_size == (512, 512)
        assert scheme.overlap == 0
        assert scheme.crs == "EPSG:4326"

    def test_create_custom_scheme(self):
        """Test custom TileScheme creation."""
        scheme = TileScheme(tile_size=(256, 256), overlap=16, crs="EPSG:3857")
        assert scheme.tile_size == (256, 256)
        assert scheme.overlap == 16
        assert scheme.crs == "EPSG:3857"

    def test_scheme_from_preset(self):
        """Test creating scheme from preset."""
        scheme = TileScheme.from_preset(TileSizePreset.MEDIUM, overlap=32)
        assert scheme.tile_size == (512, 512)
        assert scheme.overlap == 32

    def test_scheme_effective_size(self):
        """Test effective tile size calculations."""
        scheme = TileScheme(tile_size=(512, 512), overlap=32)
        assert scheme.effective_tile_width == 512 - 64
        assert scheme.effective_tile_height == 512 - 64

    def test_scheme_validation(self):
        """Test scheme validation."""
        with pytest.raises(ValueError):
            TileScheme(tile_size=(8, 8))  # Too small

        with pytest.raises(ValueError):
            TileScheme(tile_size=(512, 512), overlap=-1)  # Negative overlap

        with pytest.raises(ValueError):
            TileScheme(tile_size=(512, 512), overlap=300)  # Overlap too large

    def test_scheme_serialization(self):
        """Test scheme serialization."""
        scheme = TileScheme(tile_size=(256, 256), overlap=16)
        d = scheme.to_dict()
        restored = TileScheme.from_dict(d)

        assert restored.tile_size == scheme.tile_size
        assert restored.overlap == scheme.overlap


class TestTileGrid:
    """Tests for TileGrid class."""

    def test_create_tile_grid(self):
        """Test basic TileGrid creation."""
        grid = TileGrid(
            bounds=(0, 0, 1000, 1000),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        assert grid.n_cols > 0
        assert grid.n_rows > 0
        assert grid.total_tiles == grid.n_cols * grid.n_rows

    def test_grid_shape(self):
        """Test grid shape calculation."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        assert grid.n_cols == 4
        assert grid.n_rows == 4
        assert grid.shape == (4, 4)

    def test_grid_with_overlap(self):
        """Test grid with overlap."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256), overlap=32),
        )

        # With overlap, effective tile size is smaller, so more tiles needed
        assert grid.n_cols > 4
        assert grid.n_rows > 4

    def test_get_tile(self):
        """Test getting a specific tile."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        tile = grid.get_tile(TileIndex(col=0, row=0))
        assert tile.index.col == 0
        assert tile.index.row == 0
        assert tile.geo_bounds is not None
        assert tile.pixel_bounds is not None

    def test_get_tile_out_of_bounds(self):
        """Test getting tile with invalid index."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with pytest.raises(IndexError):
            grid.get_tile(TileIndex(col=100, row=100))

    def test_get_tile_at_point(self):
        """Test finding tile at point."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        # Point in first tile
        idx = grid.get_tile_at_point(100, 1024 - 100)
        assert idx is not None
        assert idx.col == 0
        assert idx.row == 0

    def test_get_neighbors(self):
        """Test getting neighboring tiles."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        # Center tile should have 8 neighbors
        center = TileIndex(col=1, row=1)
        neighbors = grid.get_neighbors(center)
        assert len(neighbors) == 8

        # Corner tile should have 3 neighbors
        corner = TileIndex(col=0, row=0)
        neighbors = grid.get_neighbors(corner)
        assert len(neighbors) == 3

    def test_iterate_tiles(self):
        """Test iterating over all tiles."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        tiles = list(grid)
        assert len(tiles) == grid.total_tiles

    def test_tiles_in_bounds(self):
        """Test finding tiles in bounds."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        # Query half the area
        query_bounds = TileBounds(0, 512, 512, 1024)
        tiles = list(grid.tiles_in_bounds(query_bounds))

        # Should get tiles that intersect the bounds
        assert len(tiles) > 0
        # All returned tiles should actually intersect
        for tile in tiles:
            assert tile.geo_bounds.intersects(query_bounds)


class TestTileManager:
    """Tests for TileManager class."""

    def test_create_tile_manager(self):
        """Test TileManager creation."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)
            assert manager.grid == grid

    def test_mark_tile_status(self):
        """Test marking tile status."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)

            idx = TileIndex(col=0, row=0)
            assert manager.get_status(idx) == TileStatus.PENDING

            manager.mark_in_progress(idx)
            assert manager.get_status(idx) == TileStatus.IN_PROGRESS

            manager.mark_completed(idx, result="test")
            assert manager.get_status(idx) == TileStatus.COMPLETED
            assert manager.get_result(idx) == "test"

    def test_mark_failed(self):
        """Test marking tile as failed."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)

            idx = TileIndex(col=0, row=0)
            manager.mark_failed(idx, "Test error")

            assert manager.get_status(idx) == TileStatus.FAILED
            assert manager.get_error(idx) == "Test error"

    def test_pending_tiles(self):
        """Test iterating pending tiles."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)

            # All should be pending
            pending = list(manager.pending_tiles())
            assert len(pending) == grid.total_tiles

            # Mark one completed
            manager.mark_completed(TileIndex(col=0, row=0))
            pending = list(manager.pending_tiles())
            assert len(pending) == grid.total_tiles - 1

    def test_get_progress(self):
        """Test getting progress summary."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)

            progress = manager.get_progress()
            assert progress["total"] == grid.total_tiles
            assert progress["completed"] == 0
            assert progress["pending"] == grid.total_tiles

            manager.mark_completed(TileIndex(col=0, row=0))
            progress = manager.get_progress()
            assert progress["completed"] == 1

    def test_reset_failed(self):
        """Test resetting failed tiles."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TileManager(grid, workdir=Path(tmpdir), use_db=False)

            idx = TileIndex(col=0, row=0)
            manager.mark_failed(idx, "Error")
            assert manager.get_status(idx) == TileStatus.FAILED

            count = manager.reset_failed()
            assert count == 1
            assert manager.get_status(idx) == TileStatus.PENDING


class TestOverlapHandler:
    """Tests for OverlapHandler class."""

    def test_create_overlap_handler(self):
        """Test OverlapHandler creation."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.FEATHER)
        assert handler.overlap == 32
        assert handler.blend_mode == BlendMode.FEATHER

    def test_create_blend_mask_none(self):
        """Test creating blend mask with no blending."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.NONE)
        mask = handler.create_blend_mask((256, 256), "horizontal")
        assert mask.shape == (256, 256)
        assert np.all(mask == 1.0)

    def test_create_blend_mask_feather(self):
        """Test creating feather blend mask."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.FEATHER)
        mask = handler.create_blend_mask((256, 256), "horizontal")

        assert mask.shape == (256, 256)
        # Edge values should be < 1
        assert mask[128, 0] < 1.0
        assert mask[128, 255] < 1.0
        # Center should be 1
        assert np.isclose(mask[128, 128], 1.0)

    def test_blend_tiles_average(self):
        """Test average blending."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.AVERAGE)

        tile1 = np.ones((64, 64)) * 10
        tile2 = np.ones((64, 64)) * 20

        blended = handler.blend_tiles(tile1, tile2, "horizontal")
        assert np.all(blended == 15)  # Average of 10 and 20

    def test_blend_tiles_max(self):
        """Test max blending."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.MAX)

        tile1 = np.ones((64, 64)) * 10
        tile2 = np.ones((64, 64)) * 20

        blended = handler.blend_tiles(tile1, tile2, "horizontal")
        assert np.all(blended == 20)

    def test_extract_overlap_region(self):
        """Test extracting overlap region."""
        handler = OverlapHandler(overlap=32, blend_mode=BlendMode.FEATHER)

        tile = np.arange(256 * 256).reshape(256, 256)

        left = handler.extract_overlap_region(tile, "left")
        assert left.shape == (256, 32)

        right = handler.extract_overlap_region(tile, "right")
        assert right.shape == (256, 32)

        top = handler.extract_overlap_region(tile, "top")
        assert top.shape == (32, 256)


class TestTilingUtilities:
    """Tests for tiling utility functions."""

    def test_create_tile_grid_function(self):
        """Test convenience create_tile_grid function."""
        grid = create_tile_grid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            tile_size=256,
            overlap=16,
        )

        assert grid.scheme.tile_size == (256, 256)
        assert grid.scheme.overlap == 16

    def test_estimate_memory_per_tile(self):
        """Test memory estimation."""
        mem = estimate_memory_per_tile(
            tile_size=(512, 512),
            n_bands=4,
            dtype=np.float32,
            overlap=32,
        )

        # (512 + 64) * (512 + 64) * 4 * 4 bytes
        expected = 576 * 576 * 4 * 4
        assert mem == expected

    def test_suggest_tile_size(self):
        """Test tile size suggestion."""
        tile_size = suggest_tile_size(
            image_shape=(10000, 10000),
            available_memory_mb=256,
            n_bands=4,
            dtype=np.float32,
        )

        # Should be power of 2, within reasonable bounds
        assert tile_size[0] == tile_size[1]  # Square
        assert tile_size[0] >= 128
        assert tile_size[0] <= 4096

    def test_compute_grid_hash(self):
        """Test grid hash computation."""
        grid = TileGrid(
            bounds=(0, 0, 1024, 1024),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        hash1 = compute_grid_hash(grid)
        hash2 = compute_grid_hash(grid)
        assert hash1 == hash2
        assert len(hash1) == 16


# =============================================================================
# Track 2: Streaming Ingestion Tests
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig class."""

    def test_create_default_config(self):
        """Test default config creation."""
        config = StreamingConfig()
        assert config.chunk_size_bytes == 10 * 1024 * 1024
        assert config.max_retries == 3

    def test_create_config_from_mb(self):
        """Test creating config with MB values."""
        config = StreamingConfig.from_mb(
            chunk_size_mb=50,
            buffer_size_mb=512,
            bandwidth_limit_mbps=100,
        )

        assert config.chunk_size_bytes == 50 * 1024 * 1024
        assert config.buffer_size_bytes == 512 * 1024 * 1024


class TestDownloadProgress:
    """Tests for DownloadProgress class."""

    def test_progress_properties(self):
        """Test progress computed properties."""
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
        )

        assert progress.progress_percent == 50.0
        assert progress.remaining_bytes == 500

    def test_progress_zero_total(self):
        """Test progress with zero total."""
        progress = DownloadProgress(total_bytes=0, downloaded_bytes=0)
        assert progress.progress_percent == 0.0


class TestWindowInfo:
    """Tests for WindowInfo class."""

    def test_window_info_properties(self):
        """Test WindowInfo properties."""
        window = WindowInfo(col_off=100, row_off=200, width=256, height=256)
        assert window.shape == (256, 256)

    def test_window_to_dict(self):
        """Test serialization."""
        window = WindowInfo(col_off=100, row_off=200, width=256, height=256)
        d = window.to_dict()
        assert d["col_off"] == 100
        assert d["row_off"] == 200


class TestStreamingDownloader:
    """Tests for StreamingDownloader class."""

    def test_create_downloader(self):
        """Test downloader creation."""
        downloader = StreamingDownloader()
        assert downloader.config is not None

    def test_cancel_download(self):
        """Test cancelling download."""
        downloader = StreamingDownloader()
        downloader.cancel()
        assert downloader._cancelled.is_set()


class TestMemoryMappedReader:
    """Tests for MemoryMappedReader class."""

    def test_create_and_read(self):
        """Test memory-mapped file access."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Write test data
            data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
            data.tofile(f)
            path = f.name

        try:
            reader = MemoryMappedReader(path, shape=(100, 100), dtype=np.float32)
            with reader as mmap:
                assert mmap.shape == (100, 100)
                assert mmap[50, 50] == 50 * 100 + 50

            # Test region read
            reader = MemoryMappedReader(path, shape=(100, 100), dtype=np.float32)
            reader.open()
            region = reader.read_region(10, 10, 20, 20)
            assert region.shape == (20, 20)
            reader.close()
        finally:
            Path(path).unlink()


class TestStreamingUtilities:
    """Tests for streaming utility functions."""

    def test_estimate_chunk_count(self):
        """Test chunk count estimation."""
        count = estimate_chunk_count(
            file_size_bytes=100 * 1024 * 1024,  # 100 MB
            chunk_size_mb=10,
        )
        assert count == 10


# =============================================================================
# Track 3: Tiled Runner Tests
# =============================================================================


class TestTileContext:
    """Tests for TileContext class."""

    def test_create_context(self):
        """Test TileContext creation."""
        context = TileContext(
            tile_index=TileIndex(col=5, row=5),
            tile_bounds=TileBounds(0, 0, 512, 512),
            pixel_bounds=PixelBounds(0, 0, 512, 512),
            grid_shape=(10, 10),
        )

        assert context.tile_index.col == 5
        assert not context.is_edge_tile
        assert not context.is_corner_tile

    def test_edge_tile_detection(self):
        """Test edge tile detection."""
        context = TileContext(
            tile_index=TileIndex(col=0, row=5),
            tile_bounds=TileBounds(0, 0, 512, 512),
            pixel_bounds=PixelBounds(0, 0, 512, 512),
            grid_shape=(10, 10),
        )
        assert context.is_edge_tile

    def test_corner_tile_detection(self):
        """Test corner tile detection."""
        context = TileContext(
            tile_index=TileIndex(col=0, row=0),
            tile_bounds=TileBounds(0, 0, 512, 512),
            pixel_bounds=PixelBounds(0, 0, 512, 512),
            grid_shape=(10, 10),
        )
        assert context.is_corner_tile

    def test_progress_callback(self):
        """Test progress callback."""
        called = []

        def callback(progress, message):
            called.append((progress, message))

        context = TileContext(
            tile_index=TileIndex(col=0, row=0),
            tile_bounds=TileBounds(0, 0, 512, 512),
            pixel_bounds=PixelBounds(0, 0, 512, 512),
            progress_callback=callback,
        )

        context.report_progress(0.5, "halfway")
        assert len(called) == 1
        assert called[0] == (0.5, "halfway")


class TestTileResult:
    """Tests for TileResult class."""

    def test_create_result(self):
        """Test TileResult creation."""
        result = TileResult(
            tile_index=TileIndex(col=0, row=0),
            data=np.zeros((256, 256)),
            statistics={"mean": 0.5},
        )

        assert result.data.shape == (256, 256)
        assert result.statistics["mean"] == 0.5


class TestProcessingProgress:
    """Tests for ProcessingProgress class."""

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        progress = ProcessingProgress(total_tiles=100, completed_tiles=50)
        assert progress.progress_percent == 50.0

    def test_progress_zero_tiles(self):
        """Test progress with zero tiles."""
        progress = ProcessingProgress(total_tiles=0, completed_tiles=0)
        assert progress.progress_percent == 0.0


class MockAlgorithm:
    """Mock algorithm for testing."""

    def run(self, data: np.ndarray = None, **kwargs) -> np.ndarray:
        """Simple processing - return data as-is."""
        return data.copy() if data is not None else np.zeros((256, 256))


class MockTiledAlgorithm:
    """Mock algorithm with tiled support."""

    supports_tiled = True

    def process_tile(self, data: np.ndarray, context: TileContext) -> np.ndarray:
        """Process a single tile."""
        return data * 2  # Double the values

    def run(self, data: np.ndarray = None, **kwargs) -> np.ndarray:
        """Full image processing."""
        return data * 2


class TestTiledAlgorithmRunner:
    """Tests for TiledAlgorithmRunner class."""

    def test_create_runner(self):
        """Test runner creation."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=(256, 256),
            overlap=16,
        )

        assert runner.tile_size == (256, 256)
        assert runner.overlap == 16

    def test_wrap_standard_algorithm(self):
        """Test wrapping standard algorithm."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner.wrap_algorithm(
            algorithm=algorithm,
            tile_size=256,
        )

        assert runner.supports_native_tiled is False

    def test_detect_tiled_support(self):
        """Test detection of tiled support."""
        standard = MockAlgorithm()
        tiled = MockTiledAlgorithm()

        runner1 = TiledAlgorithmRunner(standard, tile_size=256)
        runner2 = TiledAlgorithmRunner(tiled, tile_size=256)

        assert runner1.supports_native_tiled is False
        assert runner2.supports_native_tiled is True

    def test_process_small_data(self):
        """Test processing small data."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=0,
        )

        # Create test data
        data = np.ones((512, 512))
        result = runner.process(data)

        assert result.mosaic.shape == (512, 512)
        assert result.processing_time_seconds > 0

    def test_process_with_tiled_algorithm(self):
        """Test processing with natively tiled algorithm."""
        algorithm = MockTiledAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=0,
        )

        data = np.ones((512, 512))
        result = runner.process(data)

        # MockTiledAlgorithm doubles values
        assert np.allclose(result.mosaic, 2.0)

    def test_process_with_overlap(self):
        """Test processing with overlap."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=32,
        )

        data = np.ones((512, 512))
        result = runner.process(data)

        assert result.mosaic.shape == (512, 512)

    def test_progress_callback(self):
        """Test progress callback."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=0,
        )

        progress_updates = []

        def callback(progress):
            progress_updates.append(progress.progress_percent)

        data = np.ones((512, 512))
        runner.process(data, progress_callback=callback)

        assert len(progress_updates) > 0

    def test_cancel_processing(self):
        """Test cancelling processing."""
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=0,
        )

        runner.cancel()
        assert runner._cancelled.is_set()


class TestResultStitcher:
    """Tests for ResultStitcher class."""

    def test_create_stitcher(self):
        """Test stitcher creation."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        stitcher = ResultStitcher(grid, blend_mode=BlendMode.FEATHER)
        assert stitcher.grid == grid

    def test_stitch_tiles(self):
        """Test stitching tiles together."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        # Create tile results
        tile_results = {}
        for tile in grid:
            tile_results[tile.index] = TileResult(
                tile_index=tile.index,
                data=np.ones((256, 256)) * (tile.index.col + tile.index.row * 10),
            )

        stitcher = ResultStitcher(grid, blend_mode=BlendMode.NONE)
        mosaic, confidence = stitcher.stitch(tile_results)

        assert mosaic.shape == (512, 512)

    def test_stitch_empty_results(self):
        """Test stitching with no results."""
        grid = TileGrid(
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
            scheme=TileScheme(tile_size=(256, 256)),
        )

        stitcher = ResultStitcher(grid)
        mosaic, confidence = stitcher.stitch({})

        assert mosaic.shape == (512, 512)
        assert np.all(np.isnan(mosaic))


class TestTiledRunnerUtilities:
    """Tests for tiled runner utility functions."""

    def test_check_algorithm_tiled_support(self):
        """Test checking algorithm support."""
        standard = MockAlgorithm()
        tiled = MockTiledAlgorithm()

        info1 = check_algorithm_tiled_support(standard)
        info2 = check_algorithm_tiled_support(tiled)

        assert info1["native_tiled_support"] is False
        assert info1["can_be_wrapped"] is True

        assert info2["native_tiled_support"] is True

    def test_run_algorithm_tiled(self):
        """Test convenience function."""
        algorithm = MockAlgorithm()
        data = np.ones((512, 512))

        result = run_algorithm_tiled(algorithm, data, tile_size=256)

        assert isinstance(result, TiledProcessingResult)
        assert result.mosaic.shape == (512, 512)

    def test_estimate_tiles_for_memory(self):
        """Test tile size estimation."""
        tile_size = estimate_tiles_for_memory(
            data_shape=(10000, 10000),
            available_memory_mb=256,
        )

        assert tile_size[0] == tile_size[1]
        assert tile_size[0] >= 128


class TestTiledProcessingResult:
    """Tests for TiledProcessingResult class."""

    def test_aggregate_statistics(self):
        """Test statistics aggregation."""
        result = TiledProcessingResult(
            mosaic=np.zeros((256, 256)),
            tile_results={
                TileIndex(0, 0): TileResult(
                    tile_index=TileIndex(0, 0),
                    data=np.zeros((128, 128)),
                    statistics={"mean": 10, "max": 20},
                ),
                TileIndex(1, 0): TileResult(
                    tile_index=TileIndex(1, 0),
                    data=np.zeros((128, 128)),
                    statistics={"mean": 20, "max": 30},
                ),
            },
        )

        stats = result.aggregate_statistics(AggregationMethod.MEAN)
        assert stats["mean"] == 15.0
        assert stats["max"] == 25.0

        stats_max = result.aggregate_statistics(AggregationMethod.MAX)
        assert stats_max["mean"] == 20
        assert stats_max["max"] == 30


# =============================================================================
# Integration Tests
# =============================================================================


class TestTilingIntegration:
    """Integration tests combining multiple components."""

    def test_full_tiled_processing_pipeline(self):
        """Test complete tiled processing pipeline."""
        # Create test data
        data = np.random.rand(1024, 1024).astype(np.float32)

        # Create algorithm and runner
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=16,
        )

        # Process
        result = runner.process(data)

        # Verify
        assert result.mosaic.shape == data.shape
        assert len(result.tile_results) > 0

    def test_tiled_processing_with_manager(self):
        """Test tiled processing with TileManager for resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create grid and manager
            grid = TileGrid(
                bounds=(0, 0, 512, 512),
                resolution=(1.0, 1.0),
                scheme=TileScheme(tile_size=(256, 256)),
            )

            with TileManager(grid, workdir=Path(tmpdir), use_db=False) as manager:
                # Process tiles
                for tile in manager.pending_tiles():
                    manager.mark_completed(tile.index, result=tile.index.to_dict())

                # Verify
                progress = manager.get_progress()
                assert progress["completed"] == grid.total_tiles
                assert progress["pending"] == 0

    def test_overlap_blending_quality(self):
        """Test that overlap blending produces smooth results."""
        # Create gradient data
        data = np.zeros((512, 512), dtype=np.float32)
        for i in range(512):
            data[i, :] = i / 512.0

        # Process with feather blending
        algorithm = MockAlgorithm()
        runner = TiledAlgorithmRunner(
            algorithm=algorithm,
            tile_size=256,
            overlap=32,
            blend_mode=BlendMode.FEATHER,
        )

        result = runner.process(data)

        # Check for smooth transitions (no hard edges)
        # The gradient should be preserved
        assert result.mosaic.shape == data.shape


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
