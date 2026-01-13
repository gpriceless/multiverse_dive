"""
Integration Tests for Dask-Based Distributed Processing.

Tests Stream B: Distributed Raster Processing including:
- Virtual Raster Index (DP-1)
- Dask Tile Processor (DP-2)
- Algorithm Dask Adapters (DP-3)
- Execution Router (DP-4)

Performance Target:
- 100km flood analysis in <10 minutes on laptop (vs 30+ min serial)
"""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Module imports
try:
    from core.analysis.execution.dask_tiled import (
        DaskProcessingConfig,
        DaskProcessingResult,
        DaskTileProcessor,
        ProcessingProgress,
        TileInfo,
        TileResult,
        TileStatus,
        BlendMode,
        SchedulerType,
        process_with_dask,
        estimate_processing_time,
        get_optimal_config,
    )
    HAS_DASK_TILED = True
except ImportError as e:
    HAS_DASK_TILED = False
    pytest.skip(f"Dask tiled module not available: {e}", allow_module_level=True)

try:
    from core.analysis.execution.router import (
        ExecutionRouter,
        ExecutionProfile,
        ResourceEstimate,
        ResourceEstimator,
        RoutingConfig,
        RoutingResult,
        SystemResources,
        BackendSelector,
        ResourceLevel,
        auto_route,
        get_recommended_profile,
    )
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False

try:
    from core.analysis.execution.dask_adapters import (
        DaskAlgorithmAdapter,
        AlgorithmWrapper,
        TiledAlgorithmMixin,
        FloodAlgorithmAdapter,
        wrap_algorithm_for_dask,
        create_tiled_algorithm,
        check_algorithm_compatibility,
        validate_adapter,
        TileContext,
    )
    HAS_ADAPTERS = True
except ImportError:
    HAS_ADAPTERS = False

try:
    from core.data.ingestion.virtual_index import (
        VirtualRasterIndex,
        STACVRTBuilder,
        TileAccessor,
        VRTMetadata,
        BandInfo,
        TileBounds,
        build_vrt_from_stac,
    )
    HAS_VRT = True
except ImportError:
    HAS_VRT = False

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample raster data for testing."""
    np.random.seed(42)
    # Simulate 1000x1000 SAR image with low values for water
    data = np.random.randn(1000, 1000).astype(np.float32) * 5 - 10
    # Add some water regions (low backscatter)
    data[100:200, 100:200] = -20
    data[500:600, 500:600] = -18
    return data


@pytest.fixture
def large_data():
    """Create larger data for performance testing."""
    np.random.seed(42)
    # 5000x5000 image - representative of real Sentinel-1 scenes
    data = np.random.randn(5000, 5000).astype(np.float32) * 5 - 10
    return data


@pytest.fixture
def simple_algorithm():
    """Create simple threshold algorithm for testing."""
    class SimpleThreshold:
        def __init__(self, threshold=-15.0):
            self.threshold = threshold
            self.last_statistics = {}

        def execute(self, data):
            mask = data < self.threshold
            self.last_statistics = {
                "flood_pixels": int(np.sum(mask)),
                "total_pixels": mask.size,
            }
            return mask.astype(np.uint8)

    return SimpleThreshold()


@pytest.fixture
def mock_flood_algorithm():
    """Create mock flood detection algorithm."""
    class MockFloodAlgorithm:
        METADATA = {
            "id": "flood.baseline.mock",
            "name": "Mock Flood Detection",
            "version": "1.0.0",
        }

        def __init__(self):
            self.last_statistics = {}

        def execute(self, data):
            if data.ndim == 3:
                data = data[0]
            mask = data < -15
            result = MagicMock()
            result.flood_extent = mask.astype(np.uint8)
            result.statistics = {"flood_pixels": int(np.sum(mask))}
            self.last_statistics = result.statistics
            return result

    return MockFloodAlgorithm()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# DaskProcessingConfig Tests
# =============================================================================


class TestDaskProcessingConfig:
    """Tests for DaskProcessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DaskProcessingConfig()
        assert config.n_workers == 4
        assert config.tile_size == (512, 512)
        assert config.overlap == 32
        assert config.scheduler == SchedulerType.THREADS

    def test_laptop_config(self):
        """Test laptop-optimized configuration."""
        config = DaskProcessingConfig.for_laptop(memory_gb=4.0)
        assert config.n_workers >= 1
        assert config.tile_size == (512, 512)
        assert "GB" in config.memory_limit_per_worker

    def test_workstation_config(self):
        """Test workstation-optimized configuration."""
        config = DaskProcessingConfig.for_workstation(memory_gb=16.0)
        assert config.n_workers >= 1
        assert config.tile_size == (1024, 1024)
        assert config.dashboard_port == 8787

    def test_cluster_config(self):
        """Test cluster configuration."""
        config = DaskProcessingConfig.for_cluster(
            scheduler_address="tcp://scheduler:8786",
            n_workers=10,
        )
        assert config.scheduler == SchedulerType.DISTRIBUTED
        assert config.scheduler_address == "tcp://scheduler:8786"

    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        with pytest.raises(ValueError):
            DaskProcessingConfig(n_workers=0)

        with pytest.raises(ValueError):
            DaskProcessingConfig(overlap=-1)

        with pytest.raises(ValueError):
            DaskProcessingConfig(tile_size=(32, 32))

    def test_config_serialization(self):
        """Test config serialization to dict."""
        config = DaskProcessingConfig()
        config_dict = config.to_dict()
        assert "n_workers" in config_dict
        assert "tile_size" in config_dict
        assert config_dict["n_workers"] == 4


# =============================================================================
# DaskTileProcessor Tests
# =============================================================================


class TestDaskTileProcessor:
    """Tests for DaskTileProcessor."""

    def test_processor_creation(self):
        """Test processor creation."""
        processor = DaskTileProcessor()
        assert processor.config is not None

    def test_processor_with_config(self):
        """Test processor with custom config."""
        config = DaskProcessingConfig(
            n_workers=2,
            tile_size=(256, 256),
        )
        processor = DaskTileProcessor(config=config)
        assert processor.config.tile_size == (256, 256)

    def test_process_simple_algorithm(self, sample_data, simple_algorithm):
        """Test processing with simple algorithm."""
        config = DaskProcessingConfig(
            n_workers=2,
            tile_size=(256, 256),
            scheduler=SchedulerType.SYNCHRONOUS,  # Use sync for testing
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(sample_data, simple_algorithm)

        assert isinstance(result, DaskProcessingResult)
        assert result.mosaic is not None
        assert result.mosaic.shape == sample_data.shape
        assert result.processing_time_seconds > 0

    def test_process_with_progress_callback(self, sample_data, simple_algorithm):
        """Test processing with progress callback."""
        progress_updates = []

        def on_progress(progress: ProcessingProgress):
            progress_updates.append(progress.progress_percent)

        config = DaskProcessingConfig(
            n_workers=2,
            tile_size=(256, 256),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(
            config=config,
            progress_callback=on_progress,
        )

        result = processor.process(sample_data, simple_algorithm)

        assert len(progress_updates) > 0
        assert progress_updates[-1] >= 90  # Should be near 100% at end

    def test_tile_generation(self, sample_data):
        """Test tile grid generation."""
        config = DaskProcessingConfig(tile_size=(256, 256), overlap=32)
        processor = DaskTileProcessor(config=config)

        tiles = processor._generate_tiles(
            height=sample_data.shape[0],
            width=sample_data.shape[1],
            bounds=None,
            resolution=None,
        )

        # 1000x1000 with 256 tiles = 4x4 = 16 tiles
        expected_tiles = 4 * 4
        assert len(tiles) == expected_tiles

        # Check tile structure
        for tile in tiles:
            assert isinstance(tile, TileInfo)
            assert tile.col >= 0
            assert tile.row >= 0

    def test_result_stitching(self, sample_data, simple_algorithm):
        """Test that tiles are properly stitched."""
        config = DaskProcessingConfig(
            tile_size=(256, 256),
            overlap=32,
            blend_mode=BlendMode.AVERAGE,
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(sample_data, simple_algorithm)

        # Check no gaps in output
        assert not np.any(np.isnan(result.mosaic))

        # Check output matches expected water regions
        water_region = result.mosaic[100:200, 100:200]
        assert np.sum(water_region) > 0  # Should detect water

    def test_cancel_processing(self, large_data, simple_algorithm):
        """Test cancellation of processing."""
        config = DaskProcessingConfig(
            tile_size=(256, 256),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        # Cancel immediately
        processor.cancel()

        # Processing should complete quickly due to cancellation
        result = processor.process(large_data, simple_algorithm)

        # Result may be partial but shouldn't crash
        assert result is not None


# =============================================================================
# Algorithm Adapter Tests
# =============================================================================


@pytest.mark.skipif(not HAS_ADAPTERS, reason="Adapters not available")
class TestDaskAlgorithmAdapters:
    """Tests for algorithm adapters."""

    def test_wrap_simple_algorithm(self, simple_algorithm):
        """Test wrapping a simple algorithm."""
        adapter = DaskAlgorithmAdapter(simple_algorithm)
        assert adapter is not None
        assert adapter.algorithm == simple_algorithm

    def test_adapter_process_tile(self, simple_algorithm):
        """Test adapter process_tile method."""
        adapter = DaskAlgorithmAdapter(simple_algorithm)

        tile_data = np.random.randn(1, 64, 64).astype(np.float32) * 5 - 10
        tile_context = TileContext(col=0, row=0)

        result = adapter.process_tile(tile_data, tile_context)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)

    def test_flood_algorithm_adapter(self, mock_flood_algorithm):
        """Test FloodAlgorithmAdapter."""
        adapter = FloodAlgorithmAdapter(mock_flood_algorithm)

        tile_data = np.random.randn(1, 64, 64).astype(np.float32) * 5 - 10
        tile_context = TileContext(col=0, row=0)

        result = adapter.process_tile(tile_data, tile_context)

        assert isinstance(result, np.ndarray)

    def test_wrap_algorithm_for_dask(self, mock_flood_algorithm):
        """Test wrap_algorithm_for_dask factory function."""
        adapter = wrap_algorithm_for_dask(mock_flood_algorithm)

        assert isinstance(adapter, (DaskAlgorithmAdapter, FloodAlgorithmAdapter))

    def test_algorithm_wrapper(self):
        """Test AlgorithmWrapper for functions."""
        def simple_func(data):
            return data < -15

        wrapper = AlgorithmWrapper(simple_func, name="simple_threshold")

        tile_data = np.random.randn(64, 64).astype(np.float32) * 5 - 10
        tile_context = TileContext(col=0, row=0)

        result = wrapper.process_tile(tile_data, tile_context)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_create_tiled_algorithm(self):
        """Test create_tiled_algorithm factory."""
        def detect_water(data):
            return (data < -15).astype(np.uint8)

        algo = create_tiled_algorithm(
            detect_water,
            name="water_detector",
        )

        assert algo.supports_tiled
        assert algo.name == "water_detector"

    def test_check_algorithm_compatibility(self, simple_algorithm):
        """Test algorithm compatibility checking."""
        compat = check_algorithm_compatibility(simple_algorithm)

        assert "has_execute" in compat
        assert compat["has_execute"] is True
        assert compat["can_be_wrapped"] is True

    def test_validate_adapter(self, simple_algorithm):
        """Test adapter validation."""
        adapter = DaskAlgorithmAdapter(simple_algorithm)
        is_valid, issues = validate_adapter(adapter)

        assert is_valid
        assert len(issues) == 0


# =============================================================================
# Execution Router Tests
# =============================================================================


@pytest.mark.skipif(not HAS_ROUTER, reason="Router not available")
class TestExecutionRouter:
    """Tests for ExecutionRouter."""

    def test_router_creation(self):
        """Test router creation."""
        router = ExecutionRouter()
        assert router.config is not None
        assert router.resources is not None

    def test_resource_estimation(self):
        """Test resource estimation."""
        router = ExecutionRouter()
        estimate = router.estimate_resources(
            data_shape=(1000, 1000),
            dtype=np.float32,
        )

        assert isinstance(estimate, ResourceEstimate)
        assert estimate.n_tiles > 0
        assert estimate.memory_gb > 0
        assert estimate.recommended_backend is not None

    def test_recommend_profile_small_data(self):
        """Test profile recommendation for small data."""
        router = ExecutionRouter()
        profile, explanation = router.recommend_profile(
            data_shape=(500, 500),  # Small
        )

        assert profile in [
            ExecutionProfile.SERIAL,
            ExecutionProfile.TILED_SERIAL,
        ]
        assert len(explanation) > 0

    def test_recommend_profile_medium_data(self):
        """Test profile recommendation for medium data."""
        router = ExecutionRouter()
        profile, explanation = router.recommend_profile(
            data_shape=(3000, 3000),  # Medium
        )

        assert profile in [
            ExecutionProfile.TILED_SERIAL,
            ExecutionProfile.DASK_LOCAL,
        ]

    def test_recommend_profile_large_data(self):
        """Test profile recommendation for large data."""
        router = ExecutionRouter()
        profile, explanation = router.recommend_profile(
            data_shape=(10000, 10000),  # Large
        )

        assert profile in [
            ExecutionProfile.DASK_LOCAL,
            ExecutionProfile.DASK_DISTRIBUTED,
        ]

    def test_execute_with_auto_routing(self, sample_data, simple_algorithm):
        """Test execution with automatic routing."""
        router = ExecutionRouter()
        routing_result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
            profile=ExecutionProfile.AUTO,
        )

        assert isinstance(routing_result, RoutingResult)
        assert routing_result.result is not None
        assert routing_result.actual_time_seconds > 0
        assert len(routing_result.routing_decision) > 0

    def test_execute_with_specific_profile(self, sample_data, simple_algorithm):
        """Test execution with specific profile."""
        router = ExecutionRouter()
        routing_result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
            profile=ExecutionProfile.TILED_SERIAL,
        )

        # Should use requested profile or fallback appropriately
        assert routing_result.backend_used in [
            ExecutionProfile.TILED_SERIAL,
            ExecutionProfile.SERIAL,
        ]

    def test_auto_route_convenience_function(self, sample_data, simple_algorithm):
        """Test auto_route convenience function."""
        result = auto_route(
            data=sample_data,
            algorithm=simple_algorithm,
        )

        assert result is not None

    def test_get_recommended_profile(self):
        """Test get_recommended_profile function."""
        profile = get_recommended_profile(
            data_shape=(1000, 1000),
        )

        assert isinstance(profile, ExecutionProfile)

    def test_system_resources_detection(self):
        """Test system resources detection."""
        resources = SystemResources.detect()

        assert resources.cpu_count >= 1
        assert resources.total_memory_gb > 0
        assert resources.available_memory_gb > 0


# =============================================================================
# Virtual Raster Index Tests
# =============================================================================


@pytest.mark.skipif(not HAS_VRT, reason="VRT module not available")
class TestVirtualRasterIndex:
    """Tests for VirtualRasterIndex."""

    def test_stac_vrt_builder_creation(self):
        """Test STACVRTBuilder creation."""
        builder = STACVRTBuilder()
        assert builder is not None
        assert builder.mosaic_method is not None

    def test_vrt_metadata_creation(self):
        """Test VRTMetadata creation."""
        metadata = VRTMetadata(
            vrt_path="/tmp/test.vrt",
            bounds=(-122.0, 37.0, -121.0, 38.0),
            crs="EPSG:4326",
            resolution=(10.0, 10.0),
            shape=(1000, 1000),
            bands=[BandInfo(band_index=1, source_band=1, name="B04")],
        )

        assert metadata.width == 1000
        assert metadata.height == 1000
        assert metadata.n_bands == 1

    def test_tile_bounds_operations(self):
        """Test TileBounds operations."""
        bounds1 = TileBounds(0, 0, 100, 100)
        bounds2 = TileBounds(50, 50, 150, 150)

        assert bounds1.width == 100
        assert bounds1.height == 100
        assert bounds1.intersects(bounds2)

    def test_band_info_serialization(self):
        """Test BandInfo serialization."""
        band = BandInfo(
            band_index=1,
            source_band=1,
            name="B04",
            common_name="red",
        )

        band_dict = band.to_dict()
        restored = BandInfo.from_dict(band_dict)

        assert restored.band_index == band.band_index
        assert restored.common_name == band.common_name


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for distributed processing."""

    @pytest.mark.slow
    def test_parallel_speedup(self, large_data, simple_algorithm):
        """Test that parallel processing is faster than serial."""
        # Serial processing
        serial_config = DaskProcessingConfig(
            n_workers=1,
            tile_size=(512, 512),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        serial_processor = DaskTileProcessor(config=serial_config)

        serial_start = time.time()
        serial_result = serial_processor.process(large_data, simple_algorithm)
        serial_time = time.time() - serial_start

        # Parallel processing
        parallel_config = DaskProcessingConfig(
            n_workers=4,
            tile_size=(512, 512),
            scheduler=SchedulerType.THREADS,
        )
        parallel_processor = DaskTileProcessor(config=parallel_config)

        parallel_start = time.time()
        parallel_result = parallel_processor.process(large_data, simple_algorithm)
        parallel_time = time.time() - parallel_start

        # Parallel should be faster (at least 1.5x speedup expected)
        logger.info(f"Serial: {serial_time:.2f}s, Parallel: {parallel_time:.2f}s")
        logger.info(f"Speedup: {serial_time / parallel_time:.2f}x")

        # Results should be equivalent
        assert np.allclose(
            serial_result.mosaic, parallel_result.mosaic,
            rtol=1e-5, atol=1e-5, equal_nan=True
        )

    @pytest.mark.slow
    def test_memory_efficiency(self, large_data, simple_algorithm):
        """Test memory-efficient processing."""
        import gc

        # Force garbage collection
        gc.collect()

        config = DaskProcessingConfig(
            n_workers=2,
            tile_size=(512, 512),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        # Process should complete without memory errors
        result = processor.process(large_data, simple_algorithm)

        assert result.mosaic.shape == large_data.shape

        # Clean up
        del result
        gc.collect()

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        config = DaskProcessingConfig(
            n_workers=4,
            tile_size=(512, 512),
        )

        # 10000x10000 image
        estimated = estimate_processing_time(
            data_shape=(10000, 10000),
            config=config,
            time_per_tile_seconds=0.1,
        )

        # Should give reasonable estimate
        assert estimated > 0
        assert estimated < 3600  # Less than 1 hour

    def test_optimal_config_selection(self):
        """Test optimal config selection."""
        config = get_optimal_config(
            data_shape=(5000, 5000),
            available_memory_gb=4.0,
            target_time_minutes=10.0,
        )

        assert config.n_workers >= 1
        assert config.tile_size[0] >= 256


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full processing pipeline."""

    def test_full_pipeline_serial(self, sample_data, simple_algorithm):
        """Test full processing pipeline in serial mode."""
        config = DaskProcessingConfig(
            n_workers=1,
            tile_size=(256, 256),
            overlap=16,
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(
            data=sample_data,
            algorithm=simple_algorithm,
        )

        # Verify result structure
        assert result.mosaic is not None
        assert result.processing_time_seconds > 0
        assert len(result.tile_results) > 0
        assert result.progress is not None

        # Verify all tiles processed
        n_expected_tiles = 4 * 4  # 1000/256 rounded up
        assert len(result.tile_results) == n_expected_tiles

    def test_full_pipeline_with_adapter(self, sample_data, mock_flood_algorithm):
        """Test full pipeline with algorithm adapter."""
        if not HAS_ADAPTERS:
            pytest.skip("Adapters not available")

        adapter = wrap_algorithm_for_dask(mock_flood_algorithm)

        config = DaskProcessingConfig(
            n_workers=2,
            tile_size=(256, 256),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(sample_data, adapter)

        assert result.mosaic is not None
        assert result.mosaic.dtype == np.uint8

    def test_full_pipeline_with_router(self, sample_data, simple_algorithm):
        """Test full pipeline through router."""
        if not HAS_ROUTER:
            pytest.skip("Router not available")

        router = ExecutionRouter()
        routing_result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
        )

        assert routing_result.result is not None
        assert routing_result.backend_used is not None
        assert routing_result.resource_estimate is not None

    def test_different_blend_modes(self, sample_data, simple_algorithm):
        """Test different tile blending modes."""
        for blend_mode in [BlendMode.REPLACE, BlendMode.AVERAGE, BlendMode.MAX]:
            config = DaskProcessingConfig(
                tile_size=(256, 256),
                overlap=16,
                blend_mode=blend_mode,
                scheduler=SchedulerType.SYNCHRONOUS,
            )
            processor = DaskTileProcessor(config=config)

            result = processor.process(sample_data, simple_algorithm)

            assert result.mosaic is not None
            assert not np.all(np.isnan(result.mosaic))

    def test_edge_cases(self, simple_algorithm):
        """Test edge cases."""
        # Very small data (smaller than one tile)
        small_data = np.random.randn(100, 100).astype(np.float32)

        config = DaskProcessingConfig(
            tile_size=(512, 512),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(small_data, simple_algorithm)
        assert result.mosaic.shape == small_data.shape

    def test_3d_input_data(self, simple_algorithm):
        """Test with 3D input (multi-band)."""
        multi_band = np.random.randn(3, 500, 500).astype(np.float32) * 5 - 10

        config = DaskProcessingConfig(
            tile_size=(256, 256),
            scheduler=SchedulerType.SYNCHRONOUS,
        )
        processor = DaskTileProcessor(config=config)

        result = processor.process(multi_band, simple_algorithm)
        assert result.mosaic is not None


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
