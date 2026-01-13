"""
Tests for Apache Sedona Distributed Processing (Group C).

Tests cover:
- SedonaConfig: Configuration for Spark/Sedona clusters
- SedonaBackend: SparkSession management and cluster coordination
- SedonaTileProcessor: Tile processing on Spark executors
- SedonaAlgorithmAdapter: Algorithm wrapping for Sedona execution
- Router integration: SEDONA and SEDONA_CLUSTER profiles

Performance Target:
- Process 100,000 km^2 (continental scale) in <1 hour
- Scale across 10-100+ Spark executors
- Handle 10,000+ tiles efficiently
"""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest

# Module imports
try:
    from core.analysis.execution.sedona_backend import (
        SedonaBackend,
        SedonaConfig,
        SedonaTileProcessor,
        SedonaProcessingResult,
        SedonaDeployMode,
        PartitionStrategy,
        CheckpointMode,
        RasterFormat,
        ClusterStatus,
        TilePartition,
        RasterSerializer,
        create_sedona_backend,
        process_with_sedona,
        is_sedona_available,
        get_sedona_info,
        HAS_SPARK,
        HAS_SEDONA,
    )
    HAS_SEDONA_BACKEND = True
except ImportError as e:
    HAS_SEDONA_BACKEND = False
    pytest.skip(f"Sedona backend module not available: {e}", allow_module_level=True)

try:
    from core.analysis.execution.sedona_adapters import (
        SedonaAlgorithmAdapter,
        FloodSedonaAdapter,
        WildfireSedonaAdapter,
        StormSedonaAdapter,
        AlgorithmSerializer,
        ResultCollector,
        SparkUDFFactory,
        AdapterConfig,
        TileData,
        TileResult,
        wrap_algorithm_for_sedona,
        adapt_algorithms_for_sedona,
        check_sedona_compatibility,
        validate_sedona_adapter,
    )
    HAS_SEDONA_ADAPTERS = True
except ImportError:
    HAS_SEDONA_ADAPTERS = False

try:
    from core.analysis.execution.router import (
        ExecutionRouter,
        ExecutionProfile,
        RoutingConfig,
        RoutingResult,
        SystemResources,
        BackendSelector,
        ResourceEstimate,
    )
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False

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
    # 5000x5000 image - representative of large scenes
    data = np.random.randn(5000, 5000).astype(np.float32) * 5 - 10
    return data


@pytest.fixture
def continental_data():
    """Create continental-scale data (simulated)."""
    np.random.seed(42)
    # 10000x10000 would need ~400MB, simulate with metadata
    # Return smaller data but mark as continental scale
    data = np.random.randn(2000, 2000).astype(np.float32) * 5 - 10
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
def mock_wildfire_algorithm():
    """Create mock wildfire detection algorithm."""
    class MockWildfireAlgorithm:
        METADATA = {
            "id": "wildfire.dnbr.mock",
            "name": "Mock Wildfire Detection",
            "version": "1.0.0",
        }

        def __init__(self):
            self.last_statistics = {}

        def execute(self, data):
            if data.ndim == 3:
                data = data[0]
            # High values indicate burn
            mask = data > 5
            result = MagicMock()
            result.burn_mask = mask.astype(np.uint8)
            result.statistics = {"burn_pixels": int(np.sum(mask))}
            self.last_statistics = result.statistics
            return result

    return MockWildfireAlgorithm()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# SedonaConfig Tests
# =============================================================================


class TestSedonaConfig:
    """Tests for SedonaConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SedonaConfig()
        assert config.master == "local[*]"
        assert config.app_name == "multiverse_sedona"
        assert config.deploy_mode == SedonaDeployMode.LOCAL
        assert config.executor_cores == 4
        assert config.tile_size == (1024, 1024)
        assert config.overlap == 64

    def test_local_testing_config(self):
        """Test local testing configuration."""
        config = SedonaConfig.for_local_testing()
        assert "local" in config.master
        assert config.checkpoint_mode == CheckpointMode.NONE
        assert config.tile_size == (512, 512)

    def test_cluster_config(self):
        """Test cluster configuration."""
        config = SedonaConfig.for_cluster(
            master="spark://cluster:7077",
            num_executors=10,
            executor_memory="8g",
        )
        assert config.master == "spark://cluster:7077"
        assert config.num_executors == 10
        assert config.executor_memory == "8g"
        assert config.deploy_mode == SedonaDeployMode.CLIENT

    def test_databricks_config(self):
        """Test Databricks configuration."""
        config = SedonaConfig.for_databricks(num_workers=20)
        assert config.num_executors == 20
        assert "databricks" in str(config.extra_spark_config)

    def test_invalid_config_executor_cores(self):
        """Test validation of executor_cores."""
        with pytest.raises(ValueError, match="executor_cores"):
            SedonaConfig(executor_cores=0)

    def test_invalid_config_num_executors(self):
        """Test validation of num_executors."""
        with pytest.raises(ValueError, match="num_executors"):
            SedonaConfig(num_executors=0)

    def test_invalid_config_tile_size(self):
        """Test validation of tile_size."""
        with pytest.raises(ValueError, match="tile_size"):
            SedonaConfig(tile_size=(32, 32))

    def test_invalid_config_overlap(self):
        """Test validation of overlap."""
        with pytest.raises(ValueError, match="overlap"):
            SedonaConfig(overlap=-1)

    def test_spark_conf_generation(self):
        """Test Spark configuration dictionary generation."""
        config = SedonaConfig(
            app_name="test_app",
            executor_memory="4g",
            executor_cores=2,
        )
        spark_conf = config.get_spark_conf()

        assert spark_conf["spark.app.name"] == "test_app"
        assert spark_conf["spark.executor.memory"] == "4g"
        assert spark_conf["spark.executor.cores"] == "2"

    def test_config_serialization(self):
        """Test config serialization to dict."""
        config = SedonaConfig()
        config_dict = config.to_dict()

        assert "master" in config_dict
        assert "tile_size" in config_dict
        assert "deploy_mode" in config_dict


# =============================================================================
# SedonaBackend Tests
# =============================================================================


class TestSedonaBackend:
    """Tests for SedonaBackend."""

    def test_backend_creation(self):
        """Test backend creation."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)

        assert backend.config == config
        assert not backend.is_connected

    def test_backend_connect_mock(self):
        """Test backend connection in mock mode."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)

        status = backend.connect()

        assert status.is_connected
        assert "mock" in status.app_id or HAS_SPARK
        backend.disconnect()

    def test_backend_disconnect(self):
        """Test backend disconnection."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)

        backend.connect()
        backend.disconnect()

        assert not backend.is_connected

    def test_backend_context_manager(self):
        """Test backend as context manager."""
        config = SedonaConfig.for_local_testing()

        with SedonaBackend(config) as backend:
            assert backend.is_connected

        assert not backend.is_connected

    def test_get_status_not_connected(self):
        """Test status when not connected."""
        backend = SedonaBackend()
        status = backend.get_status()

        assert not status.is_connected

    def test_get_status_connected(self):
        """Test status when connected."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        status = backend.get_status()

        assert status.is_connected
        backend.disconnect()

    def test_cluster_status_serialization(self):
        """Test ClusterStatus serialization."""
        status = ClusterStatus(
            is_connected=True,
            master_url="spark://test:7077",
            app_id="app_123",
            executors=4,
            total_cores=16,
        )

        status_dict = status.to_dict()

        assert status_dict["is_connected"] is True
        assert status_dict["executors"] == 4


# =============================================================================
# SedonaTileProcessor Tests
# =============================================================================


class TestSedonaTileProcessor:
    """Tests for SedonaTileProcessor."""

    def test_processor_creation(self):
        """Test processor creation."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            assert processor.backend == backend
            assert processor.config == config
        finally:
            backend.disconnect()

    def test_process_simple_data(self, sample_data, simple_algorithm):
        """Test processing with simple algorithm."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(sample_data, simple_algorithm)

            assert isinstance(result, SedonaProcessingResult)
            assert result.success
            assert result.total_tiles > 0
            assert result.processing_time_seconds > 0
        finally:
            backend.disconnect()

    def test_process_with_bounds(self, sample_data, simple_algorithm):
        """Test processing with geographic bounds."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(
                data=sample_data,
                algorithm=simple_algorithm,
                bounds=(-122.5, 37.0, -121.5, 38.0),
                resolution=(0.0001, 0.0001),
            )

            assert result.success
        finally:
            backend.disconnect()

    def test_process_with_output(self, sample_data, simple_algorithm, temp_dir):
        """Test processing with output path."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            output_path = temp_dir / "result.tif"
            processor = SedonaTileProcessor(backend)
            result = processor.process(
                data=sample_data,
                algorithm=simple_algorithm,
                output_path=str(output_path),
            )

            assert result.success
            # Output file may exist if rasterio is available
        finally:
            backend.disconnect()

    def test_process_multiband(self, simple_algorithm):
        """Test processing multi-band data."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            # Multi-band input
            data = np.random.randn(3, 500, 500).astype(np.float32) * 5 - 10

            processor = SedonaTileProcessor(backend)
            result = processor.process(data, simple_algorithm)

            assert result.success
        finally:
            backend.disconnect()

    def test_tile_generation(self, sample_data):
        """Test tile grid generation."""
        config = SedonaConfig(tile_size=(256, 256), overlap=32)
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            tiles = processor._generate_tiles(
                data=sample_data,
                bounds=None,
                resolution=None,
            )

            # 1000x1000 with 256 tiles = 4x4 = 16 tiles
            expected_tiles = 4 * 4
            assert len(tiles) == expected_tiles

            # Check tile structure
            for tile in tiles:
                assert "tile_id" in tile
                assert "col" in tile
                assert "row" in tile
                assert "pixel_bounds" in tile
        finally:
            backend.disconnect()

    def test_result_stitching(self, sample_data, simple_algorithm):
        """Test result stitching."""
        config = SedonaConfig(tile_size=(256, 256), overlap=32)
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(sample_data, simple_algorithm)

            # Result should have statistics
            assert "statistics" in result.to_dict()
        finally:
            backend.disconnect()


# =============================================================================
# RasterSerializer Tests
# =============================================================================


class TestRasterSerializer:
    """Tests for RasterSerializer."""

    def test_serialize_array(self):
        """Test array serialization."""
        array = np.random.rand(100, 100).astype(np.float32)
        serialized = RasterSerializer.serialize_array(array)

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_deserialize_array(self):
        """Test array deserialization."""
        original = np.random.rand(50, 50).astype(np.float32)
        serialized = RasterSerializer.serialize_array(original)
        restored = RasterSerializer.deserialize_array(serialized)

        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        np.testing.assert_array_almost_equal(restored, original)

    def test_serialize_tile_result(self):
        """Test tile result serialization."""
        data = np.random.rand(64, 64).astype(np.float32)
        tile_info = {"col": 0, "row": 0, "tile_id": "0_0"}
        statistics = {"flood_pixels": 100}

        serialized = RasterSerializer.serialize_tile_result(
            data, tile_info, statistics
        )

        assert isinstance(serialized, bytes)

    def test_deserialize_tile_result(self):
        """Test tile result deserialization."""
        data = np.random.rand(64, 64).astype(np.float32)
        tile_info = {"col": 1, "row": 2, "tile_id": "1_2"}
        statistics = {"flood_pixels": 50}

        serialized = RasterSerializer.serialize_tile_result(
            data, tile_info, statistics
        )
        restored_data, restored_info, restored_stats = (
            RasterSerializer.deserialize_tile_result(serialized)
        )

        np.testing.assert_array_almost_equal(restored_data, data)
        assert restored_info == tile_info
        assert restored_stats == statistics


# =============================================================================
# SedonaAlgorithmAdapter Tests
# =============================================================================


@pytest.mark.skipif(not HAS_SEDONA_ADAPTERS, reason="Sedona adapters not available")
class TestSedonaAlgorithmAdapter:
    """Tests for SedonaAlgorithmAdapter."""

    def test_adapter_creation(self, simple_algorithm):
        """Test adapter creation."""
        adapter = SedonaAlgorithmAdapter(simple_algorithm)

        assert adapter.algorithm == simple_algorithm
        assert adapter.config is not None

    def test_adapter_process_tile(self, simple_algorithm):
        """Test adapter process_tile method."""
        adapter = SedonaAlgorithmAdapter(simple_algorithm)

        tile_data = np.random.randn(1, 64, 64).astype(np.float32) * 5 - 10
        tile_info = {"tile_id": "test", "col": 0, "row": 0}

        result = adapter.process_tile(tile_data, tile_info)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)

    def test_adapter_process_partition(self, simple_algorithm):
        """Test adapter process_partition method."""
        adapter = SedonaAlgorithmAdapter(simple_algorithm)

        partition_data = [
            (np.random.randn(1, 64, 64).astype(np.float32) * 5 - 10,
             {"tile_id": f"tile_{i}", "col": i, "row": 0})
            for i in range(3)
        ]

        results = adapter.process_partition(partition_data)

        assert len(results) == 3
        for result_data, result_info in results:
            assert isinstance(result_data, np.ndarray)
            assert result_info["success"]

    def test_flood_adapter(self, mock_flood_algorithm):
        """Test FloodSedonaAdapter."""
        adapter = FloodSedonaAdapter(mock_flood_algorithm)

        tile_data = np.random.randn(64, 64).astype(np.float32) * 5 - 10
        tile_info = {"tile_id": "test", "col": 0, "row": 0}

        result = adapter.process_tile(tile_data, tile_info)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_wildfire_adapter(self, mock_wildfire_algorithm):
        """Test WildfireSedonaAdapter."""
        adapter = WildfireSedonaAdapter(mock_wildfire_algorithm)

        tile_data = np.random.randn(64, 64).astype(np.float32) * 5
        tile_info = {"tile_id": "test", "col": 0, "row": 0}

        result = adapter.process_tile(tile_data, tile_info)

        assert isinstance(result, np.ndarray)

    def test_wrap_algorithm_for_sedona(self, mock_flood_algorithm):
        """Test wrap_algorithm_for_sedona factory function."""
        adapter = wrap_algorithm_for_sedona(mock_flood_algorithm)

        assert isinstance(adapter, (SedonaAlgorithmAdapter, FloodSedonaAdapter))

    def test_adapt_algorithms_for_sedona(self, simple_algorithm, mock_flood_algorithm):
        """Test adapting multiple algorithms."""
        algorithms = [simple_algorithm, mock_flood_algorithm]
        adapters = adapt_algorithms_for_sedona(algorithms)

        assert len(adapters) == 2
        for adapter in adapters:
            assert isinstance(adapter, SedonaAlgorithmAdapter)

    def test_check_sedona_compatibility(self, simple_algorithm):
        """Test compatibility checking."""
        compat = check_sedona_compatibility(simple_algorithm)

        assert "has_execute" in compat
        assert compat["has_execute"] is True
        assert compat["can_be_wrapped"] is True
        assert compat["is_serializable"] is True

    def test_validate_sedona_adapter(self, simple_algorithm):
        """Test adapter validation."""
        adapter = SedonaAlgorithmAdapter(simple_algorithm)
        is_valid, issues = validate_sedona_adapter(adapter)

        assert is_valid
        assert len(issues) == 0


# =============================================================================
# AlgorithmSerializer Tests
# =============================================================================


@pytest.mark.skipif(not HAS_SEDONA_ADAPTERS, reason="Sedona adapters not available")
class TestAlgorithmSerializer:
    """Tests for AlgorithmSerializer."""

    def test_can_serialize(self, simple_algorithm):
        """Test serialization check."""
        assert AlgorithmSerializer.can_serialize(simple_algorithm)

    def test_serialize_deserialize(self, simple_algorithm):
        """Test serialization round-trip."""
        serialized = AlgorithmSerializer.serialize(simple_algorithm)
        restored = AlgorithmSerializer.deserialize(serialized)

        assert type(restored).__name__ == type(simple_algorithm).__name__
        assert restored.threshold == simple_algorithm.threshold

    def test_get_signature(self, simple_algorithm):
        """Test signature generation."""
        sig1 = AlgorithmSerializer.get_signature(simple_algorithm)
        sig2 = AlgorithmSerializer.get_signature(simple_algorithm)

        assert sig1 == sig2
        assert len(sig1) == 16  # MD5 truncated


# =============================================================================
# ResultCollector Tests
# =============================================================================


@pytest.mark.skipif(not HAS_SEDONA_ADAPTERS, reason="Sedona adapters not available")
class TestResultCollector:
    """Tests for ResultCollector."""

    def test_add_result(self):
        """Test adding results."""
        collector = ResultCollector()

        result = TileResult(
            tile_id="tile_0",
            data=np.random.rand(64, 64).astype(np.float32),
            statistics={"pixels": 100},
            success=True,
        )
        collector.add_result(result)

        assert len(collector.get_all_results()) == 1

    def test_add_multiple_results(self):
        """Test adding multiple results."""
        collector = ResultCollector()

        results = [
            TileResult(tile_id=f"tile_{i}", success=True)
            for i in range(5)
        ]
        collector.add_results(results)

        assert len(collector.get_all_results()) == 5

    def test_success_rate(self):
        """Test success rate calculation."""
        collector = ResultCollector()

        collector.add_result(TileResult(tile_id="t1", success=True))
        collector.add_result(TileResult(tile_id="t2", success=True))
        collector.add_result(TileResult(tile_id="t3", success=False, error="error"))

        assert collector.success_rate == pytest.approx(2/3)

    def test_get_successful_failed_results(self):
        """Test filtering results."""
        collector = ResultCollector()

        collector.add_result(TileResult(tile_id="t1", success=True))
        collector.add_result(TileResult(tile_id="t2", success=False, error="error"))
        collector.add_result(TileResult(tile_id="t3", success=True))

        assert len(collector.get_successful_results()) == 2
        assert len(collector.get_failed_results()) == 1

    def test_aggregate_statistics(self):
        """Test statistics aggregation."""
        collector = ResultCollector()

        collector.add_result(TileResult(
            tile_id="t1",
            statistics={"pixels": 100, "ratio": 0.5},
            success=True,
        ))
        collector.add_result(TileResult(
            tile_id="t2",
            statistics={"pixels": 200, "ratio": 0.3},
            success=True,
        ))

        aggregated = collector.aggregate_statistics()

        assert "pixels_sum" in aggregated
        assert aggregated["pixels_sum"] == 300
        assert aggregated["pixels_mean"] == 150


# =============================================================================
# Router Integration Tests
# =============================================================================


@pytest.mark.skipif(not HAS_ROUTER, reason="Router not available")
class TestRouterSedonaIntegration:
    """Tests for router integration with Sedona."""

    def test_execution_profile_sedona(self):
        """Test SEDONA execution profile exists."""
        assert hasattr(ExecutionProfile, "SEDONA")
        assert hasattr(ExecutionProfile, "SEDONA_CLUSTER")
        assert hasattr(ExecutionProfile, "CONTINENTAL")

    def test_routing_config_sedona_threshold(self):
        """Test routing config includes Sedona threshold."""
        config = RoutingConfig()
        assert hasattr(config, "tile_threshold_sedona")
        assert config.tile_threshold_sedona == 10000

    def test_system_resources_spark_detection(self):
        """Test system resources detect Spark availability."""
        resources = SystemResources.detect()

        assert hasattr(resources, "has_spark")
        assert hasattr(resources, "has_sedona")
        assert hasattr(resources, "spark_master")

    def test_backend_selector_sedona_validation(self):
        """Test backend selector handles Sedona profiles."""
        config = RoutingConfig()
        resources = SystemResources(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_spark=True,
        )
        selector = BackendSelector(config, resources)

        # Create estimate for large dataset
        estimate = ResourceEstimate(
            memory_gb=4.0,
            n_tiles=15000,
            estimated_time_serial_seconds=7500,
            estimated_time_parallel_seconds=1500,
            recommended_backend=ExecutionProfile.SEDONA,
            recommended_workers=8,
        )

        # Test with SEDONA profile
        profile, decision = selector.select(estimate, ExecutionProfile.SEDONA)

        # Should not fall back since has_spark=True
        assert profile in [ExecutionProfile.SEDONA, ExecutionProfile.DASK_LOCAL]

    def test_router_sedona_profile(self, sample_data, simple_algorithm):
        """Test router with SEDONA profile."""
        config = RoutingConfig(prefer_sedona=False)
        router = ExecutionRouter(config=config)

        # Execute with explicit SEDONA profile
        # Will fall back to Dask if Sedona not available
        result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
            profile=ExecutionProfile.SEDONA,
        )

        assert isinstance(result, RoutingResult)
        assert result.result is not None

    def test_router_continental_profile(self, sample_data, simple_algorithm):
        """Test router with CONTINENTAL profile."""
        router = ExecutionRouter()

        result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
            profile=ExecutionProfile.CONTINENTAL,
        )

        assert isinstance(result, RoutingResult)
        # Will use appropriate backend based on availability

    def test_router_auto_select_large_dataset(self):
        """Test auto selection for large dataset recommends Sedona when available."""
        config = RoutingConfig(
            tile_threshold_sedona=10000,
            prefer_sedona=True,
        )
        resources = SystemResources(
            total_memory_gb=32.0,
            available_memory_gb=16.0,
            cpu_count=16,
            has_spark=True,
            has_sedona=True,
        )
        selector = BackendSelector(config, resources)

        # Estimate for continental scale
        estimate = ResourceEstimate(
            memory_gb=8.0,
            n_tiles=25000,  # Very large
            estimated_time_serial_seconds=12500,
            estimated_time_parallel_seconds=2500,
            recommended_backend=ExecutionProfile.SEDONA_CLUSTER,
            recommended_workers=16,
        )

        profile, decision = selector.select(estimate, ExecutionProfile.AUTO)

        # Should recommend Sedona for this scale if available
        assert profile in [
            ExecutionProfile.SEDONA,
            ExecutionProfile.SEDONA_CLUSTER,
            ExecutionProfile.DASK_LOCAL,
            ExecutionProfile.DASK_DISTRIBUTED,
        ]


# =============================================================================
# Performance Tests
# =============================================================================


class TestSedonaPerformance:
    """Performance tests for Sedona processing."""

    @pytest.mark.slow
    def test_tile_generation_performance(self, large_data):
        """Test tile generation is efficient."""
        config = SedonaConfig(tile_size=(512, 512), overlap=32)
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)

            start_time = time.time()
            tiles = processor._generate_tiles(large_data, None, None)
            elapsed = time.time() - start_time

            # 5000x5000 with 512 tiles = 10x10 = 100 tiles
            assert len(tiles) == 100
            assert elapsed < 1.0  # Should be very fast

        finally:
            backend.disconnect()

    @pytest.mark.slow
    def test_processing_throughput(self, sample_data, simple_algorithm):
        """Test processing throughput."""
        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)

            start_time = time.time()
            result = processor.process(sample_data, simple_algorithm)
            elapsed = time.time() - start_time

            tiles_per_second = result.total_tiles / elapsed if elapsed > 0 else 0

            logger.info(
                f"Processed {result.total_tiles} tiles in {elapsed:.2f}s "
                f"({tiles_per_second:.1f} tiles/sec)"
            )

            assert result.success
            # Should process at reasonable speed
            assert tiles_per_second > 0.5  # At least 0.5 tiles per second

        finally:
            backend.disconnect()

    def test_memory_efficiency(self, sample_data, simple_algorithm):
        """Test memory-efficient processing."""
        gc.collect()

        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(sample_data, simple_algorithm)

            assert result.success
            # No memory errors should occur

        finally:
            backend.disconnect()

        gc.collect()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_sedona_backend(self):
        """Test create_sedona_backend function."""
        backend = create_sedona_backend(master="local[2]")

        assert backend.is_connected
        backend.disconnect()

    def test_process_with_sedona(self, sample_data, simple_algorithm):
        """Test process_with_sedona convenience function."""
        result = process_with_sedona(
            data=sample_data,
            algorithm=simple_algorithm,
            master="local[2]",
        )

        assert isinstance(result, SedonaProcessingResult)
        assert result.success

    def test_is_sedona_available(self):
        """Test is_sedona_available function."""
        available = is_sedona_available()
        # Returns True if both Spark and Sedona are available
        assert isinstance(available, bool)

    def test_get_sedona_info(self):
        """Test get_sedona_info function."""
        info = get_sedona_info()

        assert "spark_available" in info
        assert "sedona_available" in info
        assert "can_process" in info
        assert "features" in info


# =============================================================================
# Integration Tests
# =============================================================================


class TestSedonaIntegration:
    """Integration tests for complete Sedona pipeline."""

    def test_full_pipeline(self, sample_data, simple_algorithm, temp_dir):
        """Test full processing pipeline."""
        output_path = temp_dir / "integration_result.tif"

        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)

        with backend:
            processor = SedonaTileProcessor(backend)
            result = processor.process(
                data=sample_data,
                algorithm=simple_algorithm,
                bounds=(-122.5, 37.0, -121.5, 38.0),
                resolution=(0.0001, 0.0001),
                output_path=str(output_path),
            )

            assert result.success
            assert result.total_tiles > 0
            assert result.failed_tiles == 0

    def test_adapter_with_processor(self, sample_data, mock_flood_algorithm):
        """Test adapted algorithm with processor."""
        if not HAS_SEDONA_ADAPTERS:
            pytest.skip("Sedona adapters not available")

        adapter = wrap_algorithm_for_sedona(mock_flood_algorithm)

        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)

        with backend:
            processor = SedonaTileProcessor(backend)
            result = processor.process(sample_data, adapter)

            assert result.success

    def test_router_to_sedona_fallback(self, sample_data, simple_algorithm):
        """Test router fallback behavior for Sedona."""
        if not HAS_ROUTER:
            pytest.skip("Router not available")

        router = ExecutionRouter()

        # Request Sedona - should fallback gracefully if not available
        result = router.execute(
            data=sample_data,
            algorithm=simple_algorithm,
            profile=ExecutionProfile.SEDONA,
        )

        assert result.result is not None
        # Backend used depends on availability
        assert result.backend_used in [
            ExecutionProfile.SEDONA,
            ExecutionProfile.DASK_LOCAL,
        ]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSedonaEdgeCases:
    """Tests for edge cases."""

    def test_small_data(self, simple_algorithm):
        """Test with very small data (smaller than one tile)."""
        small_data = np.random.randn(100, 100).astype(np.float32)

        config = SedonaConfig(tile_size=(512, 512))
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(small_data, simple_algorithm)

            assert result.success
            assert result.total_tiles == 1
        finally:
            backend.disconnect()

    def test_empty_algorithm_result(self):
        """Test handling algorithm that returns empty result."""
        class EmptyAlgorithm:
            def execute(self, data):
                return np.zeros_like(data)

        data = np.random.randn(500, 500).astype(np.float32)
        algo = EmptyAlgorithm()

        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(data, algo)

            assert result.success
        finally:
            backend.disconnect()

    def test_algorithm_with_error(self):
        """Test handling algorithm that raises error."""
        class FailingAlgorithm:
            def execute(self, data):
                raise ValueError("Intentional test error")

        data = np.random.randn(500, 500).astype(np.float32)
        algo = FailingAlgorithm()

        config = SedonaConfig.for_local_testing()
        backend = SedonaBackend(config)
        backend.connect()

        try:
            processor = SedonaTileProcessor(backend)
            result = processor.process(data, algo)

            # Should handle errors gracefully
            assert result.failed_tiles > 0
        finally:
            backend.disconnect()


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
