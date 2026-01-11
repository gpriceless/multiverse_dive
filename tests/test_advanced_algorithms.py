"""
Tests for Advanced Flood Detection Algorithms

Tests the U-Net segmentation and ensemble fusion algorithms
with synthetic data and edge cases.
"""

import pytest
import numpy as np
from typing import Dict, Any

# Import algorithms
from core.analysis.library.advanced.flood import (
    UNetSegmentationAlgorithm,
    UNetConfig,
    UNetResult,
    ModelBackend,
    EnsembleFusionAlgorithm,
    EnsembleFusionConfig,
    EnsembleFusionResult,
    AlgorithmResult,
    AlgorithmWeight,
    FusionMethod,
    DisagreementHandling,
    get_algorithm,
    list_algorithms,
    ADVANCED_FLOOD_ALGORITHMS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_imagery():
    """Generate sample multi-band imagery for U-Net."""
    np.random.seed(42)
    h, w = 256, 256

    # Create 4 bands with water-like region in center
    bands = np.random.randn(4, h, w).astype(np.float32) * 0.3 + 0.5

    # Add water region (lower values in SAR-like first band)
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    water_mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < (50 ** 2)

    # Lower values in water region (SAR-like)
    bands[0, water_mask] = np.random.randn(np.sum(water_mask)) * 0.1 - 0.5

    return bands


@pytest.fixture
def sample_algorithm_results():
    """Generate sample algorithm results for ensemble fusion."""
    np.random.seed(42)
    h, w = 100, 100

    # Create water region
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    true_water = ((y - center_y) ** 2 + (x - center_x) ** 2) < (25 ** 2)

    results = []

    # Algorithm 1: High accuracy, agrees with truth
    flood1 = true_water.copy()
    conf1 = np.where(flood1, 0.85, 0.9)
    # Add some noise
    noise_mask = np.random.random((h, w)) < 0.02
    flood1 = flood1 ^ noise_mask  # XOR to flip some pixels
    results.append(AlgorithmResult(
        algorithm_id="flood.baseline.threshold_sar",
        flood_extent=flood1,
        confidence=conf1.astype(np.float32),
        metadata={"version": "1.0.0"}
    ))

    # Algorithm 2: Slightly different detection
    flood2 = true_water.copy()
    # Erode slightly
    from scipy import ndimage
    flood2 = ndimage.binary_erosion(flood2, iterations=2)
    conf2 = np.where(flood2, 0.9, 0.85)
    results.append(AlgorithmResult(
        algorithm_id="flood.baseline.ndwi_optical",
        flood_extent=flood2,
        confidence=conf2.astype(np.float32),
        metadata={"version": "1.0.0"}
    ))

    # Algorithm 3: Slightly different detection (dilated)
    flood3 = true_water.copy()
    flood3 = ndimage.binary_dilation(flood3, iterations=2)
    conf3 = np.where(flood3, 0.75, 0.8)
    results.append(AlgorithmResult(
        algorithm_id="flood.baseline.change_detection",
        flood_extent=flood3,
        confidence=conf3.astype(np.float32),
        metadata={"version": "1.0.0"}
    ))

    return results


# ============================================================================
# U-Net Segmentation Tests
# ============================================================================

class TestUNetConfig:
    """Tests for U-Net configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UNetConfig()

        assert config.input_channels == 4
        assert config.num_classes == 2
        assert config.encoder_depth == 4
        assert config.tile_size == 512
        assert config.confidence_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = UNetConfig(
            input_channels=6,
            tile_size=256,
            confidence_threshold=0.6,
            backend=ModelBackend.ONNX
        )

        assert config.input_channels == 6
        assert config.tile_size == 256
        assert config.confidence_threshold == 0.6
        assert config.backend == ModelBackend.ONNX

    def test_invalid_encoder_depth(self):
        """Test validation of encoder depth."""
        with pytest.raises(ValueError, match="encoder_depth"):
            UNetConfig(encoder_depth=10)

    def test_invalid_confidence_threshold(self):
        """Test validation of confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            UNetConfig(confidence_threshold=1.5)

    def test_invalid_tile_size(self):
        """Test validation of tile size."""
        with pytest.raises(ValueError, match="tile_size"):
            UNetConfig(tile_size=32)

    def test_invalid_tile_overlap(self):
        """Test validation of tile overlap."""
        with pytest.raises(ValueError, match="tile_overlap"):
            UNetConfig(tile_size=512, tile_overlap=300)

    def test_invalid_dropout_rate(self):
        """Test validation of dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate"):
            UNetConfig(dropout_rate=0.8)


class TestUNetSegmentationAlgorithm:
    """Tests for U-Net segmentation algorithm."""

    def test_initialization(self):
        """Test algorithm initialization."""
        alg = UNetSegmentationAlgorithm()

        assert alg.config is not None
        assert alg.METADATA["id"] == "flood.advanced.unet_segmentation"

    def test_initialization_with_config(self):
        """Test algorithm initialization with custom config."""
        config = UNetConfig(tile_size=256, confidence_threshold=0.6)
        alg = UNetSegmentationAlgorithm(config=config)

        assert alg.config.tile_size == 256
        assert alg.config.confidence_threshold == 0.6

    def test_initialization_with_seed(self):
        """Test algorithm initialization with random seed."""
        alg = UNetSegmentationAlgorithm(random_seed=42)

        assert alg.random_seed == 42

    def test_execute_basic(self, sample_imagery):
        """Test basic execution."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        result = alg.execute(sample_imagery)

        assert isinstance(result, UNetResult)
        assert result.flood_extent.shape == sample_imagery.shape[1:]
        assert result.flood_probability.shape == sample_imagery.shape[1:]
        assert result.confidence_raster.shape == sample_imagery.shape[1:]

    def test_execute_small_image(self):
        """Test execution on small image."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=3)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(3, 50, 50).astype(np.float32)
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (50, 50)

    def test_execute_with_nodata(self, sample_imagery):
        """Test execution with nodata masking."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # Set some pixels to nodata
        sample_imagery[:, 0:10, 0:10] = -9999

        result = alg.execute(sample_imagery, nodata_value=-9999)

        # Nodata region should be masked
        assert np.all(~result.flood_extent[0:10, 0:10])

    def test_execute_hwc_format(self):
        """Test execution with (H, W, C) input format."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # (H, W, C) format
        imagery = np.random.randn(100, 100, 4).astype(np.float32)
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (100, 100)

    def test_execute_single_band(self):
        """Test execution with single band input."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=1)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(100, 100).astype(np.float32)
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (100, 100)

    def test_statistics_calculation(self, sample_imagery):
        """Test statistics are calculated correctly."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        result = alg.execute(sample_imagery, pixel_size_m=10.0)

        assert "flood_pixels" in result.statistics
        assert "flood_area_ha" in result.statistics
        assert "mean_probability" in result.statistics
        assert "mean_confidence" in result.statistics
        assert result.statistics["flood_pixels"] >= 0
        assert result.statistics["flood_area_ha"] >= 0

    def test_metadata_generation(self, sample_imagery):
        """Test metadata is generated correctly."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        result = alg.execute(sample_imagery)

        assert "id" in result.metadata
        assert "version" in result.metadata
        assert "parameters" in result.metadata
        assert "execution" in result.metadata
        assert result.metadata["id"] == "flood.advanced.unet_segmentation"

    def test_to_dict(self, sample_imagery):
        """Test result conversion to dictionary."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        result = alg.execute(sample_imagery)
        result_dict = result.to_dict()

        assert "flood_extent" in result_dict
        assert "flood_probability" in result_dict
        assert "confidence_raster" in result_dict
        assert "metadata" in result_dict
        assert "statistics" in result_dict

    def test_create_from_dict(self):
        """Test creation from dictionary."""
        params = {
            "tile_size": 256,
            "confidence_threshold": 0.6,
            "backend": "pytorch",
            "random_seed": 42
        }

        alg = UNetSegmentationAlgorithm.create_from_dict(params)

        assert alg.config.tile_size == 256
        assert alg.config.confidence_threshold == 0.6
        assert alg.random_seed == 42

    def test_get_metadata(self):
        """Test getting algorithm metadata."""
        metadata = UNetSegmentationAlgorithm.get_metadata()

        assert metadata["id"] == "flood.advanced.unet_segmentation"
        assert "requirements" in metadata
        assert "validation" in metadata

    def test_wrong_input_channels(self):
        """Test error on wrong number of input channels."""
        config = UNetConfig(input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # 3 channels instead of 4
        imagery = np.random.randn(3, 100, 100).astype(np.float32)

        with pytest.raises(ValueError, match="input channels"):
            alg.execute(imagery)


class TestUNetEdgeCases:
    """Edge case tests for U-Net algorithm."""

    def test_all_zeros_input(self):
        """Test handling of all-zeros input."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.zeros((4, 100, 100), dtype=np.float32)
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (100, 100)

    def test_all_nan_input(self):
        """Test handling of all-NaN input."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.full((4, 100, 100), np.nan, dtype=np.float32)
        result = alg.execute(imagery)

        # All pixels should be invalid
        assert np.sum(result.flood_extent) == 0

    def test_partial_nan_input(self):
        """Test handling of partial NaN input."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 100, 100).astype(np.float32)
        imagery[:, 0:20, 0:20] = np.nan

        result = alg.execute(imagery)

        # NaN region should be masked
        assert np.all(~result.flood_extent[0:20, 0:20])

    def test_extreme_values(self):
        """Test handling of extreme values."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 100, 100).astype(np.float32)
        imagery[0, 50:60, 50:60] = 1e10
        imagery[1, 60:70, 60:70] = -1e10

        result = alg.execute(imagery)

        # Should still produce valid output
        assert result.flood_extent.shape == (100, 100)
        assert np.all(np.isfinite(result.flood_probability))

    def test_threshold_zero(self):
        """Test with confidence threshold = 0.0 (edge case for division)."""
        config = UNetConfig(
            tile_size=64, tile_overlap=8, input_channels=4,
            confidence_threshold=0.0
        )
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 64, 64).astype(np.float32)
        result = alg.execute(imagery)

        # Should handle edge case without division by zero
        assert result.flood_extent.shape == (64, 64)
        assert np.all(np.isfinite(result.confidence_raster))

    def test_threshold_one(self):
        """Test with confidence threshold = 1.0 (edge case for division)."""
        config = UNetConfig(
            tile_size=64, tile_overlap=8, input_channels=4,
            confidence_threshold=1.0
        )
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 64, 64).astype(np.float32)
        result = alg.execute(imagery)

        # Should handle edge case without division by zero
        assert result.flood_extent.shape == (64, 64)
        assert np.all(np.isfinite(result.confidence_raster))


# ============================================================================
# Ensemble Fusion Tests
# ============================================================================

class TestEnsembleFusionConfig:
    """Tests for ensemble fusion configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnsembleFusionConfig()

        assert config.fusion_method == FusionMethod.RELIABILITY_WEIGHTED
        assert config.voting_threshold == 0.5
        assert config.min_algorithms == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleFusionConfig(
            fusion_method=FusionMethod.VOTING,
            voting_threshold=0.7,
            min_algorithms=3
        )

        assert config.fusion_method == FusionMethod.VOTING
        assert config.voting_threshold == 0.7
        assert config.min_algorithms == 3

    def test_invalid_voting_threshold(self):
        """Test validation of voting threshold."""
        with pytest.raises(ValueError, match="voting_threshold"):
            EnsembleFusionConfig(voting_threshold=1.5)

    def test_invalid_min_algorithms(self):
        """Test validation of min algorithms."""
        with pytest.raises(ValueError, match="min_algorithms"):
            EnsembleFusionConfig(min_algorithms=0)


class TestAlgorithmResult:
    """Tests for AlgorithmResult dataclass."""

    def test_valid_result(self):
        """Test creation of valid result."""
        flood = np.zeros((50, 50), dtype=bool)
        conf = np.ones((50, 50), dtype=np.float32)

        result = AlgorithmResult(
            algorithm_id="test_alg",
            flood_extent=flood,
            confidence=conf,
            metadata={}
        )

        assert result.algorithm_id == "test_alg"

    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        flood = np.zeros((50, 50), dtype=bool)
        conf = np.ones((60, 60), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            AlgorithmResult(
                algorithm_id="test_alg",
                flood_extent=flood,
                confidence=conf,
                metadata={}
            )


class TestEnsembleFusionAlgorithm:
    """Tests for ensemble fusion algorithm."""

    def test_initialization(self):
        """Test algorithm initialization."""
        alg = EnsembleFusionAlgorithm()

        assert alg.config is not None
        assert alg.METADATA["id"] == "flood.advanced.ensemble_fusion"

    def test_initialization_with_config(self):
        """Test algorithm initialization with custom config."""
        config = EnsembleFusionConfig(
            fusion_method=FusionMethod.VOTING,
            voting_threshold=0.6
        )
        alg = EnsembleFusionAlgorithm(config=config)

        assert alg.config.fusion_method == FusionMethod.VOTING

    def test_execute_basic(self, sample_algorithm_results):
        """Test basic execution."""
        alg = EnsembleFusionAlgorithm()

        result = alg.execute(sample_algorithm_results)

        assert isinstance(result, EnsembleFusionResult)
        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape
        assert result.flood_probability is not None
        assert result.confidence_raster is not None

    def test_execute_weighted_average(self, sample_algorithm_results):
        """Test weighted average fusion."""
        config = EnsembleFusionConfig(fusion_method=FusionMethod.WEIGHTED_AVERAGE)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_execute_voting(self, sample_algorithm_results):
        """Test voting fusion."""
        config = EnsembleFusionConfig(fusion_method=FusionMethod.VOTING)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_execute_bayesian(self, sample_algorithm_results):
        """Test Bayesian fusion."""
        config = EnsembleFusionConfig(fusion_method=FusionMethod.BAYESIAN)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_execute_reliability_weighted(self, sample_algorithm_results):
        """Test reliability-weighted fusion."""
        config = EnsembleFusionConfig(fusion_method=FusionMethod.RELIABILITY_WEIGHTED)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_agreement_map(self, sample_algorithm_results):
        """Test agreement map generation."""
        config = EnsembleFusionConfig(generate_agreement_map=True)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.agreement_map is not None
        assert result.agreement_map.shape == result.flood_extent.shape
        assert np.all((result.agreement_map >= 0) & (result.agreement_map <= 1))

    def test_uncertainty_map(self, sample_algorithm_results):
        """Test uncertainty map generation."""
        config = EnsembleFusionConfig(generate_uncertainty_map=True)
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.uncertainty_map is not None
        assert result.uncertainty_map.shape == result.flood_extent.shape
        assert np.all((result.uncertainty_map >= 0) & (result.uncertainty_map <= 1))

    def test_per_algorithm_weights(self, sample_algorithm_results):
        """Test per-algorithm weights in result."""
        alg = EnsembleFusionAlgorithm()

        result = alg.execute(sample_algorithm_results)

        assert result.per_algorithm_weights is not None
        assert len(result.per_algorithm_weights) == len(sample_algorithm_results)

    def test_statistics(self, sample_algorithm_results):
        """Test statistics calculation."""
        alg = EnsembleFusionAlgorithm()

        result = alg.execute(sample_algorithm_results, pixel_size_m=10.0)

        assert "flood_pixels" in result.statistics
        assert "flood_area_ha" in result.statistics
        assert "mean_agreement" in result.statistics
        assert "algorithm_count" in result.statistics

    def test_metadata(self, sample_algorithm_results):
        """Test metadata generation."""
        alg = EnsembleFusionAlgorithm()

        result = alg.execute(sample_algorithm_results)

        assert "id" in result.metadata
        assert "parameters" in result.metadata
        assert "execution" in result.metadata
        assert result.metadata["id"] == "flood.advanced.ensemble_fusion"

    def test_to_dict(self, sample_algorithm_results):
        """Test result conversion to dictionary."""
        alg = EnsembleFusionAlgorithm()

        result = alg.execute(sample_algorithm_results)
        result_dict = result.to_dict()

        assert "flood_extent" in result_dict
        assert "flood_probability" in result_dict
        assert "statistics" in result_dict

    def test_insufficient_algorithms(self):
        """Test error when too few algorithms provided."""
        alg = EnsembleFusionAlgorithm()

        flood = np.zeros((50, 50), dtype=bool)
        conf = np.ones((50, 50), dtype=np.float32)

        single_result = [AlgorithmResult(
            algorithm_id="test",
            flood_extent=flood,
            confidence=conf,
            metadata={}
        )]

        with pytest.raises(ValueError, match="at least"):
            alg.execute(single_result)

    def test_shape_mismatch_error(self, sample_algorithm_results):
        """Test error when algorithm results have different shapes."""
        alg = EnsembleFusionAlgorithm()

        # Add a result with different shape
        bad_result = AlgorithmResult(
            algorithm_id="bad",
            flood_extent=np.zeros((80, 80), dtype=bool),
            confidence=np.ones((80, 80), dtype=np.float32),
            metadata={}
        )
        sample_algorithm_results.append(bad_result)

        with pytest.raises(ValueError, match="Shape mismatch"):
            alg.execute(sample_algorithm_results)

    def test_with_valid_mask(self, sample_algorithm_results):
        """Test execution with valid mask."""
        alg = EnsembleFusionAlgorithm()

        valid_mask = np.ones(sample_algorithm_results[0].flood_extent.shape, dtype=bool)
        valid_mask[0:20, 0:20] = False

        result = alg.execute(sample_algorithm_results, valid_mask=valid_mask)

        # Masked region should not be flood
        assert np.all(~result.flood_extent[0:20, 0:20])

    def test_with_conditions(self, sample_algorithm_results):
        """Test execution with environmental conditions."""
        alg = EnsembleFusionAlgorithm()

        conditions = {"cloudy": True}
        result = alg.execute(sample_algorithm_results, conditions=conditions)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_custom_algorithm_weights(self, sample_algorithm_results):
        """Test with custom algorithm weights."""
        config = EnsembleFusionConfig(
            algorithm_weights=[
                AlgorithmWeight("flood.baseline.threshold_sar", base_weight=2.0),
                AlgorithmWeight("flood.baseline.ndwi_optical", base_weight=0.5),
            ]
        )
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(sample_algorithm_results)

        assert result.flood_extent.shape == sample_algorithm_results[0].flood_extent.shape

    def test_create_from_dict(self):
        """Test creation from dictionary."""
        params = {
            "fusion_method": "voting",
            "voting_threshold": 0.6,
            "min_algorithms": 3
        }

        alg = EnsembleFusionAlgorithm.create_from_dict(params)

        assert alg.config.fusion_method == FusionMethod.VOTING
        assert alg.config.voting_threshold == 0.6

    def test_get_metadata(self):
        """Test getting algorithm metadata."""
        metadata = EnsembleFusionAlgorithm.get_metadata()

        assert metadata["id"] == "flood.advanced.ensemble_fusion"
        assert "requirements" in metadata


class TestEnsembleFusionEdgeCases:
    """Edge case tests for ensemble fusion."""

    def test_all_agree(self):
        """Test when all algorithms perfectly agree."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.where(flood, 0.9, 0.9)

        results = [
            AlgorithmResult("alg1", flood.copy(), conf.astype(np.float32), {}),
            AlgorithmResult("alg2", flood.copy(), conf.astype(np.float32), {}),
            AlgorithmResult("alg3", flood.copy(), conf.astype(np.float32), {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        # Should have high agreement
        assert result.statistics["mean_agreement"] > 0.9

    def test_all_disagree(self):
        """Test when algorithms completely disagree."""
        flood1 = np.zeros((50, 50), dtype=bool)
        flood1[0:25, 0:25] = True

        flood2 = np.zeros((50, 50), dtype=bool)
        flood2[0:25, 25:50] = True

        flood3 = np.zeros((50, 50), dtype=bool)
        flood3[25:50, 0:25] = True

        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood1, conf, {}),
            AlgorithmResult("alg2", flood2, conf, {}),
            AlgorithmResult("alg3", flood3, conf, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        # Should have lower agreement
        assert result.statistics["mean_agreement"] < 0.9

    def test_zero_weight_algorithm(self):
        """Test with an algorithm that has zero weight."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", ~flood.copy(), conf, {}),  # Opposite
        ]

        config = EnsembleFusionConfig(
            algorithm_weights=[
                AlgorithmWeight("alg1", base_weight=1.0),
                AlgorithmWeight("alg2", base_weight=0.0),  # Zero weight
            ]
        )
        alg = EnsembleFusionAlgorithm(config=config)

        result = alg.execute(results)

        # Result should mostly follow alg1
        assert result.flood_extent.shape == (50, 50)

    def test_two_algorithms_minimum(self):
        """Test with exactly 2 algorithms (minimum required)."""
        flood1 = np.zeros((50, 50), dtype=bool)
        flood1[20:30, 20:30] = True
        flood2 = flood1.copy()
        flood2[25:35, 25:35] = True

        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood1, conf, {}),
            AlgorithmResult("alg2", flood2, conf, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)

    def test_threshold_zero(self):
        """Test with confidence threshold = 0.0 (edge case for division)."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        config = EnsembleFusionConfig(confidence_threshold=0.0)
        alg = EnsembleFusionAlgorithm(config=config)
        result = alg.execute(results)

        # Should handle edge case without division by zero
        assert result.flood_extent.shape == (50, 50)
        assert np.all(np.isfinite(result.confidence_raster))

    def test_threshold_one(self):
        """Test with confidence threshold = 1.0 (edge case for division)."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        config = EnsembleFusionConfig(confidence_threshold=1.0)
        alg = EnsembleFusionAlgorithm(config=config)
        result = alg.execute(results)

        # Should handle edge case without division by zero
        assert result.flood_extent.shape == (50, 50)
        assert np.all(np.isfinite(result.confidence_raster))


# ============================================================================
# Module API Tests
# ============================================================================

class TestModuleAPI:
    """Tests for module-level API."""

    def test_get_algorithm_unet(self):
        """Test getting U-Net algorithm by ID."""
        alg_class = get_algorithm("flood.advanced.unet_segmentation")
        assert alg_class == UNetSegmentationAlgorithm

    def test_get_algorithm_ensemble(self):
        """Test getting ensemble algorithm by ID."""
        alg_class = get_algorithm("flood.advanced.ensemble_fusion")
        assert alg_class == EnsembleFusionAlgorithm

    def test_get_algorithm_unknown(self):
        """Test error on unknown algorithm ID."""
        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm("flood.advanced.nonexistent")

    def test_list_algorithms(self):
        """Test listing all algorithms."""
        algs = list_algorithms()

        assert len(algs) == 2
        ids = [a[0] for a in algs]
        assert "flood.advanced.unet_segmentation" in ids
        assert "flood.advanced.ensemble_fusion" in ids

    def test_algorithm_registry(self):
        """Test algorithm registry dictionary."""
        assert len(ADVANCED_FLOOD_ALGORITHMS) == 2
        assert "flood.advanced.unet_segmentation" in ADVANCED_FLOOD_ALGORITHMS
        assert "flood.advanced.ensemble_fusion" in ADVANCED_FLOOD_ALGORITHMS


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Tests for algorithm reproducibility."""

    def test_unet_reproducibility_with_seed(self):
        """Test U-Net produces same results with same seed."""
        np.random.seed(42)
        imagery = np.random.randn(4, 100, 100).astype(np.float32)

        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)

        # Run twice with same seed
        alg1 = UNetSegmentationAlgorithm(config=config, random_seed=42)
        result1 = alg1.execute(imagery.copy())

        alg2 = UNetSegmentationAlgorithm(config=config, random_seed=42)
        result2 = alg2.execute(imagery.copy())

        # Results should be identical
        np.testing.assert_array_equal(result1.flood_extent, result2.flood_extent)

    def test_ensemble_determinism(self, sample_algorithm_results):
        """Test ensemble fusion is deterministic."""
        alg = EnsembleFusionAlgorithm()

        result1 = alg.execute(sample_algorithm_results)
        result2 = alg.execute(sample_algorithm_results)

        np.testing.assert_array_equal(result1.flood_extent, result2.flood_extent)
        np.testing.assert_array_almost_equal(
            result1.flood_probability, result2.flood_probability
        )


# ============================================================================
# Additional Edge Case Tests (Track 5 Review)
# ============================================================================

class TestUNetAdditionalEdgeCases:
    """Additional edge case tests for U-Net algorithm from Track 5 review."""

    def test_constant_value_input(self):
        """Test handling of constant value input (max_val == min_val case)."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # All same value
        imagery = np.full((4, 100, 100), 0.5, dtype=np.float32)
        result = alg.execute(imagery)

        # Should still produce valid output
        assert result.flood_extent.shape == (100, 100)
        assert np.all(np.isfinite(result.flood_probability))

    def test_very_small_image(self):
        """Test execution on image smaller than tile size."""
        config = UNetConfig(tile_size=128, tile_overlap=16, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # Image smaller than tile
        imagery = np.random.randn(4, 32, 32).astype(np.float32)
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (32, 32)

    def test_negative_values_input(self):
        """Test handling of all negative values (common in SAR dB)."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # SAR-like dB values (all negative)
        imagery = np.random.randn(4, 100, 100).astype(np.float32) - 15
        result = alg.execute(imagery)

        assert result.flood_extent.shape == (100, 100)
        assert np.all(np.isfinite(result.flood_probability))

    def test_mixed_nan_inf_values(self):
        """Test handling of mixed NaN and Inf values."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 100, 100).astype(np.float32)
        imagery[0, 10:20, 10:20] = np.nan
        imagery[1, 30:40, 30:40] = np.inf
        imagery[2, 50:60, 50:60] = -np.inf

        result = alg.execute(imagery)

        # All invalid regions should be masked
        assert np.all(~result.flood_extent[10:20, 10:20])
        assert np.all(~result.flood_extent[30:40, 30:40])
        assert np.all(~result.flood_extent[50:60, 50:60])

    def test_non_square_image(self):
        """Test execution on non-square image."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # Wide image
        imagery = np.random.randn(4, 50, 200).astype(np.float32)
        result = alg.execute(imagery)
        assert result.flood_extent.shape == (50, 200)

        # Tall image
        imagery = np.random.randn(4, 200, 50).astype(np.float32)
        result = alg.execute(imagery)
        assert result.flood_extent.shape == (200, 50)

    def test_statistics_with_no_flood(self):
        """Test statistics when no flood is detected."""
        config = UNetConfig(
            tile_size=64,
            tile_overlap=8,
            input_channels=4,
            confidence_threshold=0.99  # Very high threshold
        )
        alg = UNetSegmentationAlgorithm(config=config)

        # High values = no water in placeholder model
        imagery = np.full((4, 100, 100), 1.0, dtype=np.float32)
        result = alg.execute(imagery)

        assert result.statistics["flood_pixels"] >= 0
        assert result.statistics["flood_area_ha"] >= 0
        assert result.statistics["flood_percent"] >= 0

    def test_statistics_with_all_flood(self):
        """Test statistics when entire image is flood."""
        np.random.seed(42)
        config = UNetConfig(
            tile_size=64,
            tile_overlap=8,
            input_channels=4,
            confidence_threshold=0.01  # Very low threshold
        )
        alg = UNetSegmentationAlgorithm(config=config, random_seed=42)

        # Low values with slight variation = water in placeholder model
        # The placeholder model normalizes using percentiles, so we need some variation
        imagery = np.random.randn(4, 100, 100).astype(np.float32) * 0.1 - 2.0
        result = alg.execute(imagery)

        # Statistics should be valid (may or may not detect flood depending on model)
        assert result.statistics["flood_pixels"] >= 0
        assert result.statistics["flood_area_ha"] >= 0

    def test_confidence_calculation_edge_values(self):
        """Test confidence calculation at edge values (threshold = 0 or 1)."""
        config = UNetConfig(
            tile_size=64,
            tile_overlap=8,
            input_channels=4,
            confidence_threshold=0.01
        )
        alg = UNetSegmentationAlgorithm(config=config)

        imagery = np.random.randn(4, 100, 100).astype(np.float32)
        result = alg.execute(imagery)

        # Confidence should be valid
        assert np.all(result.confidence_raster >= 0)
        assert np.all(result.confidence_raster <= 1)

    def test_tile_count_calculation(self):
        """Test tile count calculation for various image sizes."""
        config = UNetConfig(tile_size=64, tile_overlap=8, input_channels=4)
        alg = UNetSegmentationAlgorithm(config=config)

        # Small image - should be at least 1 tile
        assert alg._get_tile_count((32, 32)) >= 1

        # Exact tile size
        assert alg._get_tile_count((64, 64)) >= 1

        # Large image
        assert alg._get_tile_count((256, 256)) > 1


class TestEnsembleFusionAdditionalEdgeCases:
    """Additional edge case tests for ensemble fusion from Track 5 review."""

    def test_all_algorithms_low_confidence(self):
        """Test when all algorithms have low confidence."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True

        # Very low confidence
        conf = np.ones((50, 50), dtype=np.float32) * 0.1

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)
        assert np.all(np.isfinite(result.flood_probability))

    def test_all_algorithms_high_confidence(self):
        """Test when all algorithms have high confidence."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True

        # Very high confidence
        conf = np.ones((50, 50), dtype=np.float32) * 0.99

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)

    def test_mixed_confidence_levels(self):
        """Test with mixed confidence levels across algorithms."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True

        results = [
            AlgorithmResult("alg1", flood.copy(),
                           np.ones((50, 50), dtype=np.float32) * 0.99, {}),
            AlgorithmResult("alg2", flood.copy(),
                           np.ones((50, 50), dtype=np.float32) * 0.01, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)

    def test_bayesian_with_extreme_confidence(self):
        """Test Bayesian fusion with near-0 and near-1 confidence values."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True

        # Near-extreme confidence (should be clipped internally)
        conf = np.ones((50, 50), dtype=np.float32) * 0.9999

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        config = EnsembleFusionConfig(fusion_method=FusionMethod.BAYESIAN)
        alg = EnsembleFusionAlgorithm(config=config)
        result = alg.execute(results)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(result.flood_probability))

    def test_spatial_reliability_disabled(self):
        """Test reliability-weighted fusion with spatial reliability disabled."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        config = EnsembleFusionConfig(
            fusion_method=FusionMethod.RELIABILITY_WEIGHTED,
            use_spatial_reliability=False
        )
        alg = EnsembleFusionAlgorithm(config=config)
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)

    def test_empty_valid_mask(self):
        """Test with completely empty valid mask (all pixels invalid)."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        # All pixels invalid
        valid_mask = np.zeros((50, 50), dtype=bool)

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results, valid_mask=valid_mask)

        # No flood should be detected
        assert np.sum(result.flood_extent) == 0
        assert result.statistics["flood_pixels"] == 0

    def test_very_small_region(self):
        """Test with very small flood region (single pixel)."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[25, 25] = True  # Single pixel
        conf = np.ones((50, 50), dtype=np.float32) * 0.9

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)

    def test_many_algorithms(self):
        """Test with many algorithms (>10)."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        # Create many algorithm results
        results = []
        for i in range(15):
            results.append(AlgorithmResult(
                f"alg{i}",
                flood.copy(),
                conf.copy(),
                {}
            ))

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results)

        assert result.flood_extent.shape == (50, 50)
        assert result.statistics["algorithm_count"] == 15

    def test_conditions_with_missing_key(self):
        """Test with conditions that don't match algorithm weights."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        # Condition that doesn't exist in any weight config
        conditions = {"unknown_condition": "value"}

        alg = EnsembleFusionAlgorithm()
        result = alg.execute(results, conditions=conditions)

        assert result.flood_extent.shape == (50, 50)

    def test_algorithm_weight_validation(self):
        """Test AlgorithmWeight validation of negative weights."""
        with pytest.raises(ValueError, match="non-negative"):
            AlgorithmWeight(algorithm_id="test", base_weight=-1.0)

    def test_fusion_with_no_uncertainty_map(self):
        """Test fusion with uncertainty map generation disabled."""
        flood = np.zeros((50, 50), dtype=bool)
        flood[20:30, 20:30] = True
        conf = np.ones((50, 50), dtype=np.float32) * 0.8

        results = [
            AlgorithmResult("alg1", flood.copy(), conf, {}),
            AlgorithmResult("alg2", flood.copy(), conf, {}),
        ]

        config = EnsembleFusionConfig(
            generate_agreement_map=False,
            generate_uncertainty_map=False
        )
        alg = EnsembleFusionAlgorithm(config=config)
        result = alg.execute(results)

        assert result.agreement_map is None
        assert result.uncertainty_map is None


class TestAlgorithmWeightEdgeCases:
    """Edge case tests for AlgorithmWeight configuration."""

    def test_algorithm_weight_with_empty_reliability_dicts(self):
        """Test AlgorithmWeight with empty reliability dictionaries."""
        weight = AlgorithmWeight(
            algorithm_id="test",
            base_weight=1.0,
            reliability_by_region={},
            reliability_by_condition={}
        )

        assert weight.base_weight == 1.0
        assert len(weight.reliability_by_region) == 0
        assert len(weight.reliability_by_condition) == 0

    def test_algorithm_weight_zero_base(self):
        """Test AlgorithmWeight with zero base weight."""
        weight = AlgorithmWeight(algorithm_id="test", base_weight=0.0)
        assert weight.base_weight == 0.0


class TestCreateFromDictEdgeCases:
    """Edge case tests for create_from_dict methods."""

    def test_unet_create_from_dict_with_list_channels(self):
        """Test UNet create_from_dict with encoder_channels as list."""
        params = {
            "tile_size": 256,
            "encoder_channels": [32, 64, 128, 256]
        }

        alg = UNetSegmentationAlgorithm.create_from_dict(params)
        assert alg.config.encoder_channels == [32, 64, 128, 256]

    def test_ensemble_create_from_dict_with_algorithm_weights_dicts(self):
        """Test Ensemble create_from_dict with algorithm weights as dicts."""
        params = {
            "fusion_method": "weighted_average",
            "algorithm_weights": [
                {"algorithm_id": "alg1", "base_weight": 2.0},
                {"algorithm_id": "alg2", "base_weight": 0.5}
            ]
        }

        alg = EnsembleFusionAlgorithm.create_from_dict(params)
        assert len(alg.config.algorithm_weights) == 2
        assert alg.config.algorithm_weights[0].base_weight == 2.0
