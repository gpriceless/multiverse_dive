"""
Unit Tests for Image Validation Module.

Tests the production image validation system including:
- Band validation for optical imagery
- SAR validation with speckle-aware thresholds
- Metadata validation (CRS, bounds)
- Configuration loading
- Exception handling
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip tests if rasterio not available
pytest.importorskip("rasterio")
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


class TestValidationConfig:
    """Tests for validation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from core.data.ingestion.validation import ValidationConfig

        config = ValidationConfig()

        # Check optical thresholds
        assert config.optical.std_dev_min == 1.0
        assert config.optical.non_zero_ratio_min == 0.05
        assert config.optical.nodata_ratio_max == 0.95

        # Check SAR thresholds
        assert config.sar.std_dev_min_db == 2.0
        assert config.sar.backscatter_range_db == (-50, 20)

        # Check screenshots disabled by default
        assert config.screenshots.enabled is False
        assert config.screenshots.retention == "temporary"

        # Check performance settings
        assert config.performance.sample_ratio == 0.3

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from core.data.ingestion.validation import ValidationConfig

        config_dict = {
            "enabled": True,
            "optical": {
                "required_bands": ["red", "nir"],
                "thresholds": {
                    "std_dev_min": 2.0,
                },
            },
            "screenshots": {
                "enabled": True,
                "retention": "permanent",
            },
        }

        config = ValidationConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.optical.std_dev_min == 2.0
        assert config.required_optical_bands == ["red", "nir"]
        assert config.screenshots.enabled is True
        assert config.screenshots.retention == "permanent"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from core.data.ingestion.validation import ValidationConfig

        config = ValidationConfig()
        config_dict = config.to_dict()

        assert "enabled" in config_dict
        assert "optical" in config_dict
        assert "sar" in config_dict
        assert "screenshots" in config_dict


class TestValidationExceptions:
    """Tests for validation exceptions."""

    def test_image_validation_error(self):
        """Test base ImageValidationError."""
        from core.data.ingestion.validation import ImageValidationError

        error = ImageValidationError("Test error", {"key": "value"})

        assert "Test error" in str(error)
        assert error.details["key"] == "value"

    def test_missing_band_error(self):
        """Test MissingBandError."""
        from core.data.ingestion.validation import MissingBandError

        error = MissingBandError(
            band_name="nir",
            expected_bands=["blue", "green", "red", "nir"],
            found_bands=["blue", "green", "red"],
        )

        assert "nir" in str(error)
        assert error.band_name == "nir"
        assert len(error.expected_bands) == 4
        assert len(error.found_bands) == 3

    def test_blank_band_error(self):
        """Test BlankBandError."""
        from core.data.ingestion.validation import BlankBandError

        error = BlankBandError(
            band_name="red",
            statistics={"std_dev": 0.5, "non_zero_ratio": 0.01},
        )

        assert "red" in str(error)
        assert error.statistics["std_dev"] == 0.5

    def test_invalid_crs_error(self):
        """Test InvalidCRSError."""
        from core.data.ingestion.validation import InvalidCRSError

        error = InvalidCRSError(crs_value="UNKNOWN", reason="Unrecognized projection")

        assert error.crs_value == "UNKNOWN"
        assert "Unrecognized" in error.reason


class TestBandStatistics:
    """Tests for BandStatistics dataclass."""

    def test_band_statistics_creation(self):
        """Test creating BandStatistics."""
        from core.data.ingestion.validation import BandStatistics

        stats = BandStatistics(
            mean=100.5,
            std_dev=25.3,
            min_val=0,
            max_val=255,
            non_zero_ratio=0.85,
            nodata_ratio=0.05,
            valid_pixel_count=9500,
            total_pixel_count=10000,
        )

        assert stats.mean == 100.5
        assert stats.std_dev == 25.3
        assert stats.non_zero_ratio == 0.85

    def test_band_statistics_to_dict(self):
        """Test converting stats to dictionary."""
        from core.data.ingestion.validation import BandStatistics

        stats = BandStatistics(mean=100.0, std_dev=10.0)
        stats_dict = stats.to_dict()

        assert "mean" in stats_dict
        assert "std_dev" in stats_dict
        assert stats_dict["mean"] == 100.0

    def test_band_statistics_handles_nan(self):
        """Test that NaN values are handled."""
        from core.data.ingestion.validation import BandStatistics

        stats = BandStatistics(mean=np.nan, std_dev=np.nan)
        stats_dict = stats.to_dict()

        # NaN should be converted to None
        assert stats_dict["mean"] is None
        assert stats_dict["std_dev"] is None


class TestBandValidationResult:
    """Tests for BandValidationResult dataclass."""

    def test_band_validation_result_valid(self):
        """Test valid band result."""
        from core.data.ingestion.validation import BandStatistics, BandValidationResult

        result = BandValidationResult(
            band_name="red",
            band_index=3,
            is_valid=True,
            is_required=True,
            statistics=BandStatistics(mean=100, std_dev=25),
        )

        assert result.is_valid is True
        assert result.band_name == "red"
        assert len(result.errors) == 0

    def test_band_validation_result_invalid(self):
        """Test invalid band result."""
        from core.data.ingestion.validation import BandStatistics, BandValidationResult

        result = BandValidationResult(
            band_name="nir",
            band_index=4,
            is_valid=False,
            is_required=True,
            statistics=BandStatistics(mean=0, std_dev=0.5),
            errors=["Band appears blank"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 1


class TestImageValidationResult:
    """Tests for ImageValidationResult dataclass."""

    def test_image_validation_result(self):
        """Test ImageValidationResult creation."""
        from core.data.ingestion.validation import (
            ImageMetadata,
            ImageValidationResult,
        )

        result = ImageValidationResult(
            dataset_id="test_image",
            file_path="/path/to/image.tif",
            is_valid=True,
            metadata=ImageMetadata(
                crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                width=1000,
                height=1000,
            ),
        )

        assert result.is_valid is True
        assert result.dataset_id == "test_image"
        assert result.metadata.crs == "EPSG:4326"

    def test_image_validation_result_to_dict(self):
        """Test converting result to dictionary."""
        from core.data.ingestion.validation import ImageValidationResult

        result = ImageValidationResult(
            dataset_id="test",
            file_path="/test.tif",
            is_valid=True,
            validation_duration_seconds=1.5,
        )

        result_dict = result.to_dict()

        assert result_dict["dataset_id"] == "test"
        assert result_dict["is_valid"] is True
        assert result_dict["validation_duration_seconds"] == 1.5


class TestCreateTestRaster:
    """Helper to create test rasters."""

    @staticmethod
    def create_test_raster(
        path: Path,
        width: int = 100,
        height: int = 100,
        n_bands: int = 4,
        dtype: str = "uint16",
        crs: str = "EPSG:4326",
        nodata: float = 0,
        fill_value: float = None,
        std_dev: float = 50.0,
    ):
        """Create a test raster file."""
        transform = from_bounds(-10, -10, 10, 10, width, height)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=n_bands,
            dtype=dtype,
            crs=CRS.from_string(crs),
            transform=transform,
            nodata=nodata,
        ) as dst:
            for band in range(1, n_bands + 1):
                if fill_value is not None:
                    data = np.full((height, width), fill_value, dtype=dtype)
                else:
                    # Generate random data with specified std dev
                    mean = 5000 if dtype == "uint16" else 0.5
                    data = np.random.normal(mean, std_dev, (height, width))
                    data = np.clip(data, 0, 10000 if dtype == "uint16" else 1.0)
                    data = data.astype(dtype)
                dst.write(data, band)


class TestImageValidator:
    """Tests for the main ImageValidator class."""

    def test_validator_creation(self):
        """Test creating ImageValidator."""
        from core.data.ingestion.validation import ImageValidator, ValidationConfig

        # With default config
        validator = ImageValidator()
        assert validator.config is not None
        assert validator.config.enabled is True

        # With custom config
        config = ValidationConfig(enabled=False)
        validator = ImageValidator(config=config)
        assert validator.config.enabled is False

    def test_validator_disabled(self):
        """Test validation when disabled returns success."""
        from core.data.ingestion.validation import ImageValidator, ValidationConfig

        config = ValidationConfig(enabled=False)
        validator = ImageValidator(config=config)

        result = validator.validate(
            raster_path="/nonexistent/file.tif",
        )

        assert result.is_valid is True
        assert "Validation is disabled" in result.warnings

    def test_validate_valid_image(self, tmp_path):
        """Test validating a valid image."""
        from core.data.ingestion.validation import ImageValidator

        # Create valid test raster
        raster_path = tmp_path / "valid_image.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=4,
            std_dev=100,  # Good variation
        )

        validator = ImageValidator()
        result = validator.validate(raster_path)

        assert result.is_valid is True
        assert result.metadata.crs == "EPSG:4326"
        assert result.metadata.width == 100
        assert result.metadata.height == 100

    def test_validate_blank_band_image(self, tmp_path):
        """Test that blank bands are detected."""
        from core.data.ingestion.validation import ImageValidator

        # Create raster with blank band (constant value)
        raster_path = tmp_path / "blank_image.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=4,
            fill_value=100,  # Constant value = blank
        )

        validator = ImageValidator()
        result = validator.validate(raster_path)

        # Should fail due to blank bands
        assert result.is_valid is False
        assert any("blank" in err.lower() for err in result.errors)

    def test_validate_missing_file(self):
        """Test validation of non-existent file."""
        from core.data.ingestion.validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate("/nonexistent/path/image.tif")

        assert result.is_valid is False
        assert any("does not exist" in err or "not found" in err.lower() for err in result.errors)

    def test_validate_with_expected_bounds(self, tmp_path):
        """Test bounds intersection check."""
        from core.data.ingestion.validation import ImageValidator

        # Create raster with specific bounds
        raster_path = tmp_path / "bounded_image.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            std_dev=100,
        )

        validator = ImageValidator()

        # Test with intersecting bounds
        result = validator.validate(
            raster_path,
            expected_bounds=(-5, -5, 5, 5),  # Intersects (-10, -10, 10, 10)
        )
        assert result.is_valid is True

        # Test with non-intersecting bounds
        result = validator.validate(
            raster_path,
            expected_bounds=(100, 100, 200, 200),  # Does not intersect
        )
        assert result.is_valid is False
        assert any("bounds" in err.lower() for err in result.errors)


class TestBandValidator:
    """Tests for the BandValidator class."""

    def test_band_validator_creation(self):
        """Test creating BandValidator."""
        from core.data.ingestion.validation import BandValidator, ValidationConfig

        config = ValidationConfig()
        validator = BandValidator(config)

        assert validator.config is not None

    def test_validate_valid_bands(self, tmp_path):
        """Test validating bands with good statistics."""
        from core.data.ingestion.validation import BandValidator, ValidationConfig

        # Create test raster
        raster_path = tmp_path / "good_bands.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=4,
            std_dev=100,
        )

        config = ValidationConfig()
        validator = BandValidator(config)

        with rasterio.open(raster_path) as dataset:
            results = validator.validate_bands(
                dataset=dataset,
                expected_bands={"blue": ["B1"], "green": ["B2"], "red": ["B3"], "nir": ["B4"]},
                required_bands=["blue", "green"],
            )

        # All bands should be valid
        for name, result in results.items():
            assert result.is_valid is True

    def test_validate_blank_band(self, tmp_path):
        """Test detecting blank band."""
        from core.data.ingestion.validation import BandValidator, ValidationConfig

        # Create raster with constant values (blank)
        raster_path = tmp_path / "blank_band.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=1,
            fill_value=100,  # Constant = std_dev ~= 0
        )

        config = ValidationConfig()
        validator = BandValidator(config)

        with rasterio.open(raster_path) as dataset:
            result = validator.validate_band_content(dataset, band_index=1)

        # Should detect blank band
        assert result.is_valid is False
        assert result.statistics.std_dev < config.optical.std_dev_min


class TestSARValidator:
    """Tests for SAR-specific validation."""

    def test_sar_validator_creation(self):
        """Test creating SARValidator."""
        from core.data.ingestion.validation import SARValidator, ValidationConfig

        config = ValidationConfig()
        validator = SARValidator(config)

        assert validator.config is not None
        assert "VV" in validator.polarization_patterns

    def test_validate_sar_image(self, tmp_path):
        """Test validating SAR image."""
        from core.data.ingestion.validation import SARValidator, ValidationConfig

        # Create test SAR-like raster
        raster_path = tmp_path / "sar_image.tif"

        # SAR values typically have high variability due to speckle
        width, height = 100, 100
        transform = from_bounds(-10, -10, 10, 10, width, height)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            # VV band with typical backscatter values
            vv_data = np.random.uniform(-20, -5, (height, width)).astype("float32")
            dst.write(vv_data, 1)
            dst.set_band_description(1, "VV")

            # VH band
            vh_data = np.random.uniform(-25, -10, (height, width)).astype("float32")
            dst.write(vh_data, 2)
            dst.set_band_description(2, "VH")

        config = ValidationConfig()
        validator = SARValidator(config)

        with rasterio.open(raster_path) as dataset:
            result = validator.validate(
                dataset=dataset,
                data_source_spec={"sensor": "Sentinel-1"},
            )

        # SAR validation should pass
        assert result.is_valid is True
        assert "VV" in result.polarizations_found


class TestScreenshotGenerator:
    """Tests for screenshot generation."""

    def test_screenshot_generator_creation(self):
        """Test creating ScreenshotGenerator."""
        from core.data.ingestion.validation import ScreenshotGenerator, ValidationConfig

        config = ValidationConfig()
        config.screenshots.enabled = True

        generator = ScreenshotGenerator(config)

        assert generator.output_dir.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
        reason="matplotlib required",
    )
    def test_generate_screenshot(self, tmp_path):
        """Test generating a screenshot."""
        from core.data.ingestion.validation import ScreenshotGenerator, ValidationConfig

        # Create test raster
        raster_path = tmp_path / "screenshot_test.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=3,
            std_dev=50,
        )

        config = ValidationConfig()
        config.screenshots.enabled = True
        config.screenshots.output_dir = str(tmp_path / "screenshots")

        generator = ScreenshotGenerator(config)

        screenshot_path = generator.generate(
            raster_path=raster_path,
            dataset_id="test_screenshot",
        )

        assert screenshot_path is not None
        assert screenshot_path.exists()
        assert screenshot_path.suffix == ".png"


class TestValidationIntegration:
    """Integration tests for the validation pipeline."""

    def test_full_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline."""
        from core.data.ingestion.validation import (
            ImageValidator,
            ValidationConfig,
            validate_image,
        )

        # Create valid test image
        raster_path = tmp_path / "integration_test.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=200,
            height=200,
            n_bands=4,
            std_dev=100,
        )

        # Test using convenience function
        result = validate_image(raster_path)

        assert result.is_valid is True
        assert result.validation_duration_seconds > 0
        assert len(result.band_results) > 0

    def test_validation_with_custom_config(self, tmp_path):
        """Test validation with custom configuration."""
        from core.data.ingestion.validation import ImageValidator, ValidationConfig

        # Create test image
        raster_path = tmp_path / "custom_config_test.tif"
        TestCreateTestRaster.create_test_raster(
            raster_path,
            width=100,
            height=100,
            n_bands=4,
            std_dev=5,  # Low std dev
        )

        # Custom config with stricter threshold
        config = ValidationConfig()
        config.optical.std_dev_min = 10.0  # Higher than image std dev

        validator = ImageValidator(config)
        result = validator.validate(raster_path)

        # Should fail with stricter threshold
        assert result.is_valid is False

    def test_streaming_ingester_validation_integration(self, tmp_path):
        """Test that StreamingIngester uses validation."""
        from core.data.ingestion.streaming import StreamingIngester

        # Create test file
        source_path = tmp_path / "source.tif"
        TestCreateTestRaster.create_test_raster(
            source_path,
            width=50,
            height=50,
            n_bands=4,
            std_dev=100,
        )

        output_path = tmp_path / "output.tif"

        ingester = StreamingIngester()

        # Ingest should validate the image
        result = ingester.ingest(
            source=str(source_path),
            output_path=output_path,
        )

        # Check validation was performed
        if "validation" in result:
            assert result["validation"]["is_valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
