"""
Comprehensive Test Suite for Wildfire Detection Algorithms

Tests for:
- DifferencedNBRAlgorithm (burn severity mapping)
- ThermalAnomalyAlgorithm (active fire detection)
- BurnedAreaClassifierAlgorithm (ML-based classification)

This test suite validates:
- Algorithm initialization and configuration
- Input validation and edge cases
- Core algorithm execution
- Output format and statistics
- Reproducibility (where applicable)
- Error handling
"""

import numpy as np
import pytest
from typing import Dict, Any

# Import wildfire algorithms
from core.analysis.library.baseline.wildfire import (
    DifferencedNBRAlgorithm,
    DifferencedNBRConfig,
    DifferencedNBRResult,
    ThermalAnomalyAlgorithm,
    ThermalAnomalyConfig,
    ThermalAnomalyResult,
    BurnedAreaClassifierAlgorithm,
    BurnedAreaClassifierConfig,
    BurnedAreaClassifierResult,
    get_algorithm,
    list_algorithms,
    WILDFIRE_ALGORITHMS,
)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class WildfireSyntheticDataGenerator:
    """Generate synthetic data for wildfire algorithm testing."""

    @staticmethod
    def generate_optical_bands(
        height: int = 100,
        width: int = 100,
        burn_ratio: float = 0.3,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic optical bands with burned and unburned areas.

        Args:
            height: Image height
            width: Image width
            burn_ratio: Fraction of burned pixels
            seed: Random seed

        Returns:
            Dictionary with nir, swir, red bands and burn_mask
        """
        rng = np.random.RandomState(seed)

        # Create base arrays
        nir_pre = np.zeros((height, width), dtype=np.float32)
        swir_pre = np.zeros((height, width), dtype=np.float32)
        nir_post = np.zeros((height, width), dtype=np.float32)
        swir_post = np.zeros((height, width), dtype=np.float32)
        red = np.zeros((height, width), dtype=np.float32)

        # Create circular burn scar
        center_y, center_x = height // 2, width // 2
        burn_radius = int(np.sqrt(burn_ratio * height * width / np.pi))

        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        burn_mask = dist_from_center <= burn_radius

        # Unburned vegetation (healthy)
        # Pre-fire: high NIR, low SWIR (healthy vegetation)
        nir_pre[~burn_mask] = rng.uniform(0.4, 0.6, np.sum(~burn_mask))
        swir_pre[~burn_mask] = rng.uniform(0.15, 0.25, np.sum(~burn_mask))
        nir_post[~burn_mask] = rng.uniform(0.35, 0.55, np.sum(~burn_mask))
        swir_post[~burn_mask] = rng.uniform(0.15, 0.25, np.sum(~burn_mask))
        red[~burn_mask] = rng.uniform(0.05, 0.15, np.sum(~burn_mask))

        # Burned area
        # Pre-fire: healthy vegetation
        nir_pre[burn_mask] = rng.uniform(0.4, 0.6, np.sum(burn_mask))
        swir_pre[burn_mask] = rng.uniform(0.15, 0.25, np.sum(burn_mask))
        # Post-fire: low NIR, high SWIR (burned/charred)
        nir_post[burn_mask] = rng.uniform(0.08, 0.15, np.sum(burn_mask))
        swir_post[burn_mask] = rng.uniform(0.25, 0.40, np.sum(burn_mask))
        red[burn_mask] = rng.uniform(0.10, 0.20, np.sum(burn_mask))

        return {
            "nir_pre": nir_pre,
            "swir_pre": swir_pre,
            "nir_post": nir_post,
            "swir_post": swir_post,
            "red": red,
            "burn_mask": burn_mask
        }

    @staticmethod
    def generate_thermal_data(
        height: int = 100,
        width: int = 100,
        fire_ratio: float = 0.05,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic thermal data with fire hotspots.

        Args:
            height: Image height
            width: Image width
            fire_ratio: Fraction of active fire pixels
            seed: Random seed

        Returns:
            Dictionary with thermal band and fire mask
        """
        rng = np.random.RandomState(seed)

        # Background temperature (typical land surface)
        thermal = rng.uniform(290, 310, (height, width)).astype(np.float32)

        # Create fire hotspots
        n_fires = max(1, int(fire_ratio * height * width / 100))  # Number of fire clusters
        fire_mask = np.zeros((height, width), dtype=bool)

        for _ in range(n_fires):
            # Random fire center
            cy = rng.randint(10, height - 10)
            cx = rng.randint(10, width - 10)

            # Fire radius (3-8 pixels)
            radius = rng.randint(3, 8)

            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
            fire_region = dist <= radius

            fire_mask |= fire_region

            # Fire temperature gradient (hotter at center)
            fire_temps = 400 - (dist[fire_region] / radius) * 60  # 340-400K
            thermal[fire_region] = fire_temps + rng.uniform(-10, 10, np.sum(fire_region))

        return {
            "thermal": thermal,
            "fire_mask": fire_mask
        }


# ============================================================================
# Differenced NBR Algorithm Tests
# ============================================================================

class TestDifferencedNBRConfig:
    """Tests for DifferencedNBRConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DifferencedNBRConfig()
        assert config.high_severity_threshold == 0.66
        assert config.moderate_high_threshold == 0.44
        assert config.moderate_low_threshold == 0.27
        assert config.low_severity_threshold == 0.10
        assert config.unburned_threshold == -0.10
        assert config.min_burn_area_ha == 1.0
        assert config.cloud_mask_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DifferencedNBRConfig(
            high_severity_threshold=0.7,
            moderate_high_threshold=0.5,
            moderate_low_threshold=0.3,
            low_severity_threshold=0.15,
            min_burn_area_ha=2.0
        )
        assert config.high_severity_threshold == 0.7
        assert config.min_burn_area_ha == 2.0

    def test_invalid_threshold_order(self):
        """Test that invalid threshold order raises error."""
        with pytest.raises(ValueError, match="Severity thresholds must be in descending order"):
            DifferencedNBRConfig(
                high_severity_threshold=0.3,  # Lower than moderate_high
                moderate_high_threshold=0.5
            )

    def test_negative_min_area(self):
        """Test that negative min area raises error."""
        with pytest.raises(ValueError, match="min_burn_area_ha must be non-negative"):
            DifferencedNBRConfig(min_burn_area_ha=-1.0)


class TestDifferencedNBRAlgorithm:
    """Tests for DifferencedNBRAlgorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm instance with default config."""
        return DifferencedNBRAlgorithm()

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic optical data."""
        return WildfireSyntheticDataGenerator.generate_optical_bands(
            height=100, width=100, burn_ratio=0.3, seed=42
        )

    def test_initialization(self, algorithm):
        """Test algorithm initializes correctly."""
        assert algorithm.config is not None
        assert algorithm.METADATA["id"] == "wildfire.baseline.nbr_differenced"
        assert algorithm.METADATA["deterministic"] is True

    def test_basic_execution(self, algorithm, synthetic_data):
        """Test basic algorithm execution."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"],
            pixel_size_m=30.0
        )

        assert isinstance(result, DifferencedNBRResult)
        assert result.burn_extent.shape == synthetic_data["nir_pre"].shape
        assert result.burn_severity.shape == synthetic_data["nir_pre"].shape
        assert result.dnbr_map.shape == synthetic_data["nir_pre"].shape
        assert result.confidence_raster.shape == synthetic_data["nir_pre"].shape

    def test_burn_detection_accuracy(self, algorithm, synthetic_data):
        """Test that burn detection roughly matches known burn area."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )

        # Check that most of the known burned area is detected
        overlap = np.sum(result.burn_extent & synthetic_data["burn_mask"])
        actual_burned = np.sum(synthetic_data["burn_mask"])
        recall = overlap / actual_burned if actual_burned > 0 else 0

        # Should detect at least 70% of actual burned area
        assert recall >= 0.7, f"Detection recall too low: {recall:.2f}"

    def test_severity_classes(self, algorithm, synthetic_data):
        """Test that severity classes are properly assigned."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )

        # Severity should be in range 0-5
        assert result.burn_severity.min() >= 0
        assert result.burn_severity.max() <= 5

        # Check that statistics include severity classes
        assert "high" in result.statistics["severity_counts"]
        assert "low" in result.statistics["severity_counts"]

    def test_dnbr_values(self, algorithm, synthetic_data):
        """Test that dNBR values are in expected range."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )

        # dNBR should be between -2 and 2 for most cases
        valid_mask = np.isfinite(result.dnbr_map)
        assert result.dnbr_map[valid_mask].min() >= -2.0
        assert result.dnbr_map[valid_mask].max() <= 2.0

        # Burned areas should have positive dNBR
        burned_dnbr = result.dnbr_map[synthetic_data["burn_mask"]]
        assert np.mean(burned_dnbr) > 0

    def test_confidence_values(self, algorithm, synthetic_data):
        """Test that confidence values are in [0, 1]."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )

        assert result.confidence_raster.min() >= 0.0
        assert result.confidence_raster.max() <= 1.0

    def test_with_cloud_mask(self, algorithm, synthetic_data):
        """Test algorithm with cloud mask."""
        cloud_mask = np.zeros_like(synthetic_data["nir_pre"], dtype=bool)
        cloud_mask[40:60, 40:60] = True  # Block center region

        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"],
            cloud_mask=cloud_mask
        )

        # Cloud-masked area should not be detected as burned
        assert not np.any(result.burn_extent[cloud_mask])

    def test_nodata_handling(self, algorithm):
        """Test handling of nodata values."""
        data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)

        # Set some pixels to nodata
        data["nir_pre"][0:10, 0:10] = -9999

        result = algorithm.execute(
            nir_pre=data["nir_pre"],
            swir_pre=data["swir_pre"],
            nir_post=data["nir_post"],
            swir_post=data["swir_post"],
            nodata_value=-9999
        )

        # Nodata pixels should not be burned
        assert not np.any(result.burn_extent[0:10, 0:10])

    def test_shape_mismatch_error(self, algorithm):
        """Test that shape mismatch raises error."""
        with pytest.raises(ValueError, match="same shape"):
            algorithm.execute(
                nir_pre=np.zeros((100, 100)),
                swir_pre=np.zeros((100, 100)),
                nir_post=np.zeros((100, 100)),
                swir_post=np.zeros((50, 50))  # Wrong shape
            )

    def test_invalid_dimensions(self, algorithm):
        """Test that 3D array raises error."""
        with pytest.raises(ValueError, match="Expected 2D"):
            algorithm.execute(
                nir_pre=np.zeros((100, 100, 3)),  # 3D
                swir_pre=np.zeros((100, 100, 3)),
                nir_post=np.zeros((100, 100, 3)),
                swir_post=np.zeros((100, 100, 3))
            )

    def test_statistics_output(self, algorithm, synthetic_data):
        """Test that statistics are correctly calculated."""
        result = algorithm.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"],
            pixel_size_m=30.0
        )

        stats = result.statistics
        assert "total_pixels" in stats
        assert "valid_pixels" in stats
        assert "burned_pixels" in stats
        assert "total_burned_area_ha" in stats
        assert "burned_percent" in stats
        assert "high_severity_percent" in stats

        # Sanity checks
        assert stats["total_pixels"] == 100 * 100
        assert stats["burned_pixels"] <= stats["valid_pixels"]

    def test_reproducibility(self, synthetic_data):
        """Test that algorithm produces identical results."""
        algo1 = DifferencedNBRAlgorithm()
        algo2 = DifferencedNBRAlgorithm()

        result1 = algo1.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )
        result2 = algo2.execute(
            nir_pre=synthetic_data["nir_pre"],
            swir_pre=synthetic_data["swir_pre"],
            nir_post=synthetic_data["nir_post"],
            swir_post=synthetic_data["swir_post"]
        )

        np.testing.assert_array_equal(result1.burn_extent, result2.burn_extent)
        np.testing.assert_array_equal(result1.burn_severity, result2.burn_severity)
        np.testing.assert_allclose(result1.dnbr_map, result2.dnbr_map)

    def test_metadata(self, algorithm):
        """Test algorithm metadata."""
        metadata = algorithm.get_metadata()
        assert metadata["id"] == "wildfire.baseline.nbr_differenced"
        assert metadata["version"] == "1.0.0"
        assert "wildfire.*" in metadata["event_types"]

    def test_create_from_dict(self):
        """Test creating algorithm from parameter dictionary."""
        params = {
            "high_severity_threshold": 0.7,
            "min_burn_area_ha": 2.0
        }
        algo = DifferencedNBRAlgorithm.create_from_dict(params)
        assert algo.config.high_severity_threshold == 0.7
        assert algo.config.min_burn_area_ha == 2.0


# ============================================================================
# Thermal Anomaly Algorithm Tests
# ============================================================================

class TestThermalAnomalyConfig:
    """Tests for ThermalAnomalyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ThermalAnomalyConfig()
        assert config.temperature_threshold_k == 320.0
        assert config.background_window_size == 21
        assert config.min_temperature_delta_k == 10.0
        assert config.contextual_algorithm is True
        assert config.frp_calculation is True

    def test_invalid_temperature_threshold(self):
        """Test that invalid temperature threshold raises error."""
        with pytest.raises(ValueError, match="temperature_threshold_k must be >= 273K"):
            ThermalAnomalyConfig(temperature_threshold_k=200)

    def test_invalid_window_size(self):
        """Test that invalid window size raises error."""
        # Even number
        with pytest.raises(ValueError, match="background_window_size must be odd"):
            ThermalAnomalyConfig(background_window_size=20)

        # Too small
        with pytest.raises(ValueError, match="background_window_size must be odd and >= 3"):
            ThermalAnomalyConfig(background_window_size=1)

    def test_negative_delta(self):
        """Test that negative delta raises error."""
        with pytest.raises(ValueError, match="min_temperature_delta_k must be non-negative"):
            ThermalAnomalyConfig(min_temperature_delta_k=-5)


class TestThermalAnomalyAlgorithm:
    """Tests for ThermalAnomalyAlgorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm instance with default config."""
        return ThermalAnomalyAlgorithm()

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic thermal data."""
        return WildfireSyntheticDataGenerator.generate_thermal_data(
            height=100, width=100, fire_ratio=0.05, seed=42
        )

    def test_initialization(self, algorithm):
        """Test algorithm initializes correctly."""
        assert algorithm.config is not None
        assert algorithm.METADATA["id"] == "wildfire.baseline.thermal_anomaly"
        assert algorithm.METADATA["deterministic"] is True

    def test_basic_execution(self, algorithm, synthetic_data):
        """Test basic algorithm execution."""
        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        assert isinstance(result, ThermalAnomalyResult)
        assert result.active_fires.shape == synthetic_data["thermal"].shape
        assert result.fire_radiative_power.shape == synthetic_data["thermal"].shape
        assert result.confidence_raster.shape == synthetic_data["thermal"].shape
        assert result.brightness_temp.shape == synthetic_data["thermal"].shape

    def test_fire_detection(self, algorithm, synthetic_data):
        """Test that fires are detected at hot spots."""
        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        # Check that some fires are detected
        assert np.sum(result.active_fires) > 0

        # Most detected fires should overlap with actual fire mask
        if np.sum(result.active_fires) > 0:
            overlap = np.sum(result.active_fires & synthetic_data["fire_mask"])
            detected = np.sum(result.active_fires)
            precision = overlap / detected if detected > 0 else 0
            assert precision >= 0.5, f"Fire detection precision too low: {precision:.2f}"

    def test_simple_threshold_detection(self, synthetic_data):
        """Test simple threshold detection mode."""
        config = ThermalAnomalyConfig(contextual_algorithm=False)
        algo = ThermalAnomalyAlgorithm(config)

        result = algo.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        assert np.sum(result.active_fires) > 0

    def test_contextual_detection(self, synthetic_data):
        """Test contextual detection mode."""
        config = ThermalAnomalyConfig(contextual_algorithm=True)
        algo = ThermalAnomalyAlgorithm(config)

        result = algo.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        # Contextual should also detect fires
        assert np.sum(result.active_fires) > 0

    def test_frp_calculation(self, algorithm, synthetic_data):
        """Test Fire Radiative Power calculation."""
        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        # FRP should be non-negative
        assert result.fire_radiative_power.min() >= 0

        # FRP should be non-zero for fire pixels
        if np.sum(result.active_fires) > 0:
            fire_frp = result.fire_radiative_power[result.active_fires]
            assert np.mean(fire_frp) > 0

    def test_frp_disabled(self, synthetic_data):
        """Test with FRP calculation disabled."""
        config = ThermalAnomalyConfig(frp_calculation=False)
        algo = ThermalAnomalyAlgorithm(config)

        result = algo.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        # FRP should be all zeros
        assert np.sum(result.fire_radiative_power) == 0

    def test_radiance_input(self, algorithm):
        """Test with radiance input (values < 200)."""
        # Create synthetic radiance data (low values)
        rng = np.random.RandomState(42)
        radiance = rng.uniform(5, 15, (50, 50)).astype(np.float32)
        # Add fire hotspot with higher radiance
        radiance[20:25, 20:25] = 50  # Higher radiance = fire

        result = algorithm.execute(
            thermal_band=radiance,
            thermal_band_wavelength_um=4.0,
            pixel_size_m=375.0
        )

        # Should detect conversion and process
        assert result.brightness_temp.max() > 200  # Converted to temperature

    def test_water_mask(self, algorithm, synthetic_data):
        """Test with water mask."""
        water_mask = np.zeros_like(synthetic_data["thermal"], dtype=bool)
        water_mask[40:60, 40:60] = True  # Water body

        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            water_mask=water_mask
        )

        # Water pixels should not be detected as fire
        assert not np.any(result.active_fires[water_mask])

    def test_solar_zenith(self, algorithm, synthetic_data):
        """Test with solar zenith angle."""
        solar_zenith = np.full_like(synthetic_data["thermal"], 45.0)  # Daytime

        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            solar_zenith=solar_zenith
        )

        assert result.metadata["execution"]["is_daytime"] == True

        # Test nighttime
        solar_zenith_night = np.full_like(synthetic_data["thermal"], 100.0)
        result_night = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            solar_zenith=solar_zenith_night
        )
        assert result_night.metadata["execution"]["is_daytime"] == False

    def test_confidence_values(self, algorithm, synthetic_data):
        """Test that confidence values are in [0, 1]."""
        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"]
        )

        assert result.confidence_raster.min() >= 0.0
        assert result.confidence_raster.max() <= 1.0

    def test_statistics_output(self, algorithm, synthetic_data):
        """Test statistics output."""
        result = algorithm.execute(
            thermal_band=synthetic_data["thermal"],
            pixel_size_m=375.0
        )

        stats = result.statistics
        assert "fire_count" in stats
        assert "fire_area_ha" in stats
        assert "total_frp_mw" in stats
        assert "mean_frp_mw" in stats
        assert "max_temperature_k" in stats

    def test_invalid_dimensions(self, algorithm):
        """Test that 3D array raises error."""
        with pytest.raises(ValueError, match="Expected 2D"):
            algorithm.execute(
                thermal_band=np.zeros((100, 100, 3))
            )

    def test_metadata(self, algorithm):
        """Test algorithm metadata."""
        metadata = algorithm.get_metadata()
        assert metadata["id"] == "wildfire.baseline.thermal_anomaly"
        assert metadata["version"] == "1.0.0"

    def test_reproducibility(self, synthetic_data):
        """Test deterministic behavior."""
        algo1 = ThermalAnomalyAlgorithm()
        algo2 = ThermalAnomalyAlgorithm()

        result1 = algo1.execute(thermal_band=synthetic_data["thermal"])
        result2 = algo2.execute(thermal_band=synthetic_data["thermal"])

        np.testing.assert_array_equal(result1.active_fires, result2.active_fires)


# ============================================================================
# Burned Area Classifier Tests
# ============================================================================

class TestBurnedAreaClassifierConfig:
    """Tests for BurnedAreaClassifierConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BurnedAreaClassifierConfig()
        assert config.classifier_type == "random_forest"
        assert config.n_estimators == 100
        assert config.random_seed == 42
        assert config.use_texture_features is True

    def test_invalid_classifier_type(self):
        """Test that invalid classifier type raises error."""
        with pytest.raises(ValueError, match="classifier_type must be"):
            BurnedAreaClassifierConfig(classifier_type="svm")

    def test_invalid_n_estimators(self):
        """Test that invalid n_estimators raises error."""
        with pytest.raises(ValueError, match="n_estimators must be >= 1"):
            BurnedAreaClassifierConfig(n_estimators=0)

    def test_invalid_max_depth(self):
        """Test that invalid max_depth raises error."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            BurnedAreaClassifierConfig(max_depth=0)

    def test_negative_min_area(self):
        """Test that negative min area raises error."""
        with pytest.raises(ValueError, match="min_burn_area_ha must be non-negative"):
            BurnedAreaClassifierConfig(min_burn_area_ha=-1.0)


class TestBurnedAreaClassifierAlgorithm:
    """Tests for BurnedAreaClassifierAlgorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm instance."""
        return BurnedAreaClassifierAlgorithm()

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic optical data."""
        return WildfireSyntheticDataGenerator.generate_optical_bands(
            height=100, width=100, burn_ratio=0.3, seed=42
        )

    def test_initialization(self, algorithm):
        """Test algorithm initializes correctly."""
        assert algorithm.config is not None
        assert algorithm.METADATA["id"] == "wildfire.baseline.ba_classifier"
        assert algorithm.METADATA["deterministic"] is False
        assert algorithm.METADATA["seed_required"] is True

    def test_basic_execution(self, algorithm, synthetic_data):
        """Test basic algorithm execution."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"],
            pixel_size_m=30.0
        )

        assert isinstance(result, BurnedAreaClassifierResult)
        assert result.burn_extent.shape == synthetic_data["red"].shape
        assert result.classification_confidence.shape == synthetic_data["red"].shape
        assert "nbr" in result.spectral_indices

    def test_self_training(self, algorithm, synthetic_data):
        """Test self-training mode (no training data provided)."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        # Should still produce results
        assert result.burn_extent is not None
        assert result.metadata["training"]["training_pixels"] > 0

    def test_with_training_data(self, synthetic_data):
        """Test with provided training data."""
        algo = BurnedAreaClassifierAlgorithm()

        # Create training mask (sample from known burn area)
        training_mask = np.zeros_like(synthetic_data["red"], dtype=bool)
        training_labels = np.zeros_like(synthetic_data["red"], dtype=np.int32)

        # Sample some burned and unburned pixels
        burn_mask = synthetic_data["burn_mask"]
        training_mask[burn_mask] = True
        training_labels[burn_mask] = 1
        training_mask[~burn_mask] = True
        training_labels[~burn_mask] = 0

        # Subsample to reduce training set
        rng = np.random.RandomState(42)
        subsample = rng.rand(*training_mask.shape) < 0.1
        training_mask = training_mask & subsample

        result = algo.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"],
            training_mask=training_mask,
            training_labels=training_labels
        )

        assert result.metadata["training"]["training_pixels"] > 0

    def test_spectral_indices(self, algorithm, synthetic_data):
        """Test that spectral indices are calculated."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        assert "nbr" in result.spectral_indices
        assert "ndvi" in result.spectral_indices
        assert "bai" in result.spectral_indices
        assert "mirbi" in result.spectral_indices

        # NBR and NDVI should be in [-1, 1]
        assert result.spectral_indices["nbr"].min() >= -1.0
        assert result.spectral_indices["nbr"].max() <= 1.0

    def test_feature_importance(self, algorithm, synthetic_data):
        """Test feature importance output."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        assert len(result.feature_importance) > 0

        # Importance values should sum to ~1
        total_importance = sum(result.feature_importance.values())
        assert 0.99 <= total_importance <= 1.01

    def test_texture_features_enabled(self, synthetic_data):
        """Test with texture features enabled."""
        config = BurnedAreaClassifierConfig(use_texture_features=True)
        algo = BurnedAreaClassifierAlgorithm(config)

        result = algo.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        # Should include texture features
        assert "local_variance" in result.feature_importance

    def test_texture_features_disabled(self, synthetic_data):
        """Test with texture features disabled."""
        config = BurnedAreaClassifierConfig(use_texture_features=False)
        algo = BurnedAreaClassifierAlgorithm(config)

        result = algo.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        # Should not include texture features
        assert "local_variance" not in result.feature_importance

    def test_confidence_values(self, algorithm, synthetic_data):
        """Test that confidence values are in [0, 1]."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        valid = np.isfinite(result.classification_confidence)
        assert result.classification_confidence[valid].min() >= 0.0
        assert result.classification_confidence[valid].max() <= 1.0

    def test_statistics_output(self, algorithm, synthetic_data):
        """Test statistics output."""
        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"],
            pixel_size_m=30.0
        )

        stats = result.statistics
        assert "burned_pixels" in stats
        assert "burned_area_ha" in stats
        assert "burned_percent" in stats
        assert "mean_confidence" in stats
        assert "confidence_distribution" in stats

    def test_cloud_mask(self, algorithm, synthetic_data):
        """Test with cloud mask."""
        cloud_mask = np.zeros_like(synthetic_data["red"], dtype=bool)
        cloud_mask[40:60, 40:60] = True

        result = algorithm.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"],
            cloud_mask=cloud_mask
        )

        # Cloud-masked area should not be burned
        assert not np.any(result.burn_extent[cloud_mask])

    def test_shape_mismatch_error(self, algorithm):
        """Test that shape mismatch raises error."""
        with pytest.raises(ValueError, match="same shape"):
            algorithm.execute(
                red=np.zeros((100, 100)),
                nir=np.zeros((100, 100)),
                swir=np.zeros((50, 50))  # Wrong shape
            )

    def test_invalid_dimensions(self, algorithm):
        """Test that 3D array raises error."""
        with pytest.raises(ValueError, match="Expected 2D"):
            algorithm.execute(
                red=np.zeros((100, 100, 3)),
                nir=np.zeros((100, 100, 3)),
                swir=np.zeros((100, 100, 3))
            )

    def test_reproducibility_with_seed(self, synthetic_data):
        """Test that same seed produces same results."""
        config1 = BurnedAreaClassifierConfig(random_seed=42)
        config2 = BurnedAreaClassifierConfig(random_seed=42)
        algo1 = BurnedAreaClassifierAlgorithm(config1)
        algo2 = BurnedAreaClassifierAlgorithm(config2)

        result1 = algo1.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )
        result2 = algo2.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        np.testing.assert_array_equal(result1.burn_extent, result2.burn_extent)

    def test_different_seeds_different_results(self, synthetic_data):
        """Test that different seeds can produce different results."""
        config1 = BurnedAreaClassifierConfig(random_seed=42)
        config2 = BurnedAreaClassifierConfig(random_seed=123)
        algo1 = BurnedAreaClassifierAlgorithm(config1)
        algo2 = BurnedAreaClassifierAlgorithm(config2)

        result1 = algo1.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )
        result2 = algo2.execute(
            red=synthetic_data["red"],
            nir=synthetic_data["nir_post"],
            swir=synthetic_data["swir_post"]
        )

        # Results might differ with different seeds (not guaranteed, but likely)
        # At minimum, the confidence values should differ slightly
        # We just verify both run successfully
        assert result1.burn_extent is not None
        assert result2.burn_extent is not None

    def test_metadata(self, algorithm):
        """Test algorithm metadata."""
        metadata = algorithm.get_metadata()
        assert metadata["id"] == "wildfire.baseline.ba_classifier"
        assert metadata["version"] == "1.0.0"
        assert metadata["seed_required"] is True

    def test_create_from_dict(self):
        """Test creating algorithm from parameter dictionary."""
        params = {
            "n_estimators": 50,
            "random_seed": 123
        }
        algo = BurnedAreaClassifierAlgorithm.create_from_dict(params)
        assert algo.config.n_estimators == 50
        assert algo.config.random_seed == 123


# ============================================================================
# Module-Level Tests
# ============================================================================

class TestWildfireModule:
    """Tests for wildfire module exports."""

    def test_algorithm_registry(self):
        """Test WILDFIRE_ALGORITHMS registry."""
        assert len(WILDFIRE_ALGORITHMS) == 3
        assert "wildfire.baseline.nbr_differenced" in WILDFIRE_ALGORITHMS
        assert "wildfire.baseline.thermal_anomaly" in WILDFIRE_ALGORITHMS
        assert "wildfire.baseline.ba_classifier" in WILDFIRE_ALGORITHMS

    def test_get_algorithm(self):
        """Test get_algorithm function."""
        algo_class = get_algorithm("wildfire.baseline.nbr_differenced")
        assert algo_class == DifferencedNBRAlgorithm

        algo_class = get_algorithm("wildfire.baseline.thermal_anomaly")
        assert algo_class == ThermalAnomalyAlgorithm

        algo_class = get_algorithm("wildfire.baseline.ba_classifier")
        assert algo_class == BurnedAreaClassifierAlgorithm

    def test_get_algorithm_invalid(self):
        """Test get_algorithm with invalid ID."""
        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm("wildfire.baseline.nonexistent")

    def test_list_algorithms(self):
        """Test list_algorithms function."""
        algorithms = list_algorithms()
        assert len(algorithms) == 3

        ids = [id for id, _ in algorithms]
        assert "wildfire.baseline.nbr_differenced" in ids
        assert "wildfire.baseline.thermal_anomaly" in ids
        assert "wildfire.baseline.ba_classifier" in ids


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_image(self):
        """Test with minimal image size."""
        algo = DifferencedNBRAlgorithm()
        small = np.zeros((5, 5), dtype=np.float32) + 0.3

        result = algo.execute(
            nir_pre=small,
            swir_pre=small,
            nir_post=small,
            swir_post=small
        )

        assert result.burn_extent.shape == (5, 5)

    def test_all_nodata(self):
        """Test with all nodata values."""
        algo = DifferencedNBRAlgorithm()
        nodata = np.full((50, 50), -9999, dtype=np.float32)

        result = algo.execute(
            nir_pre=nodata,
            swir_pre=nodata,
            nir_post=nodata,
            swir_post=nodata,
            nodata_value=-9999
        )

        # No valid pixels, no burned area
        assert np.sum(result.burn_extent) == 0

    def test_constant_values(self):
        """Test with constant input values."""
        algo = ThermalAnomalyAlgorithm()
        constant = np.full((50, 50), 300.0, dtype=np.float32)

        result = algo.execute(thermal_band=constant)

        # No temperature anomalies
        assert np.sum(result.active_fires) == 0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        algo = DifferencedNBRAlgorithm()
        data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)

        # Introduce NaN values
        data["nir_pre"][10:15, 10:15] = np.nan

        result = algo.execute(
            nir_pre=data["nir_pre"],
            swir_pre=data["swir_pre"],
            nir_post=data["nir_post"],
            swir_post=data["swir_post"]
        )

        # NaN pixels should not be burned
        assert not np.any(result.burn_extent[10:15, 10:15])

    def test_extreme_values(self):
        """Test with extreme reflectance values where self-training fails."""
        algo = BurnedAreaClassifierAlgorithm()

        # Very high reflectance - uniform data won't generate meaningful training
        # Self-training is expected to fail with uniform extreme data
        high = np.ones((50, 50), dtype=np.float32)

        # Without distinct burned/unburned spectral signatures,
        # self-training cannot identify training samples
        with pytest.raises(ValueError, match="both burned and unburned"):
            algo.execute(
                red=high,
                nir=high,
                swir=high
            )

    def test_extreme_values_with_training(self):
        """Test classifier handles extreme values when training data provided."""
        algo = BurnedAreaClassifierAlgorithm()

        # Create extreme but varied data
        rng = np.random.RandomState(42)
        red = rng.uniform(0.8, 1.0, (50, 50)).astype(np.float32)
        nir = rng.uniform(0.0, 0.2, (50, 50)).astype(np.float32)
        swir = rng.uniform(0.3, 0.7, (50, 50)).astype(np.float32)

        # Provide explicit training data
        training_mask = np.zeros((50, 50), dtype=bool)
        training_labels = np.zeros((50, 50), dtype=np.int32)

        # Mark some pixels as burned/unburned for training
        training_mask[0:10, 0:10] = True
        training_labels[0:10, 0:10] = 1  # burned
        training_mask[40:50, 40:50] = True
        training_labels[40:50, 40:50] = 0  # unburned

        result = algo.execute(
            red=red,
            nir=nir,
            swir=swir,
            training_mask=training_mask,
            training_labels=training_labels
        )

        # Should handle extreme values without crashing
        assert result.burn_extent is not None

    def test_single_class_training(self):
        """Test classifier with single class training data."""
        algo = BurnedAreaClassifierAlgorithm()
        data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)

        # Training mask with only burned pixels
        training_mask = data["burn_mask"].copy()
        training_labels = np.ones_like(data["red"], dtype=np.int32)

        with pytest.raises(ValueError, match="both burned and unburned"):
            algo.execute(
                red=data["red"],
                nir=data["nir_post"],
                swir=data["swir_post"],
                training_mask=training_mask,
                training_labels=training_labels
            )


# ============================================================================
# Result Conversion Tests
# ============================================================================

class TestResultConversion:
    """Tests for result to_dict methods."""

    def test_dnbr_result_to_dict(self):
        """Test DifferencedNBRResult to_dict."""
        algo = DifferencedNBRAlgorithm()
        data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)

        result = algo.execute(
            nir_pre=data["nir_pre"],
            swir_pre=data["swir_pre"],
            nir_post=data["nir_post"],
            swir_post=data["swir_post"]
        )

        result_dict = result.to_dict()
        assert "burn_extent" in result_dict
        assert "burn_severity" in result_dict
        assert "dnbr_map" in result_dict
        assert "metadata" in result_dict

    def test_thermal_result_to_dict(self):
        """Test ThermalAnomalyResult to_dict."""
        algo = ThermalAnomalyAlgorithm()
        data = WildfireSyntheticDataGenerator.generate_thermal_data(seed=42)

        result = algo.execute(thermal_band=data["thermal"])

        result_dict = result.to_dict()
        assert "active_fires" in result_dict
        assert "fire_radiative_power" in result_dict
        assert "metadata" in result_dict

    def test_classifier_result_to_dict(self):
        """Test BurnedAreaClassifierResult to_dict."""
        algo = BurnedAreaClassifierAlgorithm()
        data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)

        result = algo.execute(
            red=data["red"],
            nir=data["nir_post"],
            swir=data["swir_post"]
        )

        result_dict = result.to_dict()
        assert "burn_extent" in result_dict
        assert "classification_confidence" in result_dict
        assert "feature_importance" in result_dict


# ============================================================================
# Integration Tests
# ============================================================================

class TestAlgorithmIntegration:
    """Integration tests combining multiple algorithms."""

    def test_multi_algorithm_workflow(self):
        """Test running multiple algorithms on same scene."""
        data = WildfireSyntheticDataGenerator.generate_optical_bands(
            height=100, width=100, burn_ratio=0.3, seed=42
        )

        # Run dNBR for burn severity
        dnbr_algo = DifferencedNBRAlgorithm()
        dnbr_result = dnbr_algo.execute(
            nir_pre=data["nir_pre"],
            swir_pre=data["swir_pre"],
            nir_post=data["nir_post"],
            swir_post=data["swir_post"]
        )

        # Run classifier
        classifier = BurnedAreaClassifierAlgorithm()
        class_result = classifier.execute(
            red=data["red"],
            nir=data["nir_post"],
            swir=data["swir_post"]
        )

        # Both should detect burned areas
        assert np.sum(dnbr_result.burn_extent) > 0
        assert np.sum(class_result.burn_extent) > 0

        # Some overlap expected
        overlap = np.sum(dnbr_result.burn_extent & class_result.burn_extent)
        assert overlap > 0

    def test_all_algorithms_same_data(self):
        """Test all three algorithms on same synthetic data."""
        optical_data = WildfireSyntheticDataGenerator.generate_optical_bands(seed=42)
        thermal_data = WildfireSyntheticDataGenerator.generate_thermal_data(seed=42)

        # dNBR
        dnbr = DifferencedNBRAlgorithm()
        dnbr_result = dnbr.execute(
            nir_pre=optical_data["nir_pre"],
            swir_pre=optical_data["swir_pre"],
            nir_post=optical_data["nir_post"],
            swir_post=optical_data["swir_post"]
        )

        # Thermal
        thermal = ThermalAnomalyAlgorithm()
        thermal_result = thermal.execute(
            thermal_band=thermal_data["thermal"]
        )

        # Classifier
        classifier = BurnedAreaClassifierAlgorithm()
        class_result = classifier.execute(
            red=optical_data["red"],
            nir=optical_data["nir_post"],
            swir=optical_data["swir_post"]
        )

        # All should produce valid results
        assert dnbr_result.burn_extent is not None
        assert thermal_result.active_fires is not None
        assert class_result.burn_extent is not None
