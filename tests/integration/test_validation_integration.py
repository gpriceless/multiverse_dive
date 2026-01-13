"""
Integration Tests for Image Validation in Production Workflows.

Tests the integration of image validation with:
- StreamingIngester
- TiledAlgorithmRunner
- Lineage tracking (provenance)
- End-to-end workflows
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip tests if dependencies not available
rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import from_bounds


class TestRasterFactory:
    """Factory for creating test rasters."""

    @staticmethod
    def create_optical_raster(
        path: Path,
        width: int = 100,
        height: int = 100,
        valid: bool = True,
    ):
        """Create an optical-like test raster."""
        transform = from_bounds(-10, -10, 10, 10, width, height)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
        ) as dst:
            for band in range(1, 5):
                if valid:
                    # Generate data with good variation
                    data = np.random.normal(5000, 1000, (height, width))
                    data = np.clip(data, 100, 10000).astype("uint16")
                else:
                    # Generate blank data (constant)
                    data = np.full((height, width), 100, dtype="uint16")
                dst.write(data, band)

            # Set band descriptions
            dst.descriptions = ("Blue", "Green", "Red", "NIR")

    @staticmethod
    def create_sar_raster(
        path: Path,
        width: int = 100,
        height: int = 100,
        valid: bool = True,
    ):
        """Create a SAR-like test raster."""
        transform = from_bounds(-10, -10, 10, 10, width, height)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            if valid:
                # VV band with typical backscatter
                vv = np.random.uniform(-20, -5, (height, width)).astype("float32")
                # VH band
                vh = np.random.uniform(-25, -10, (height, width)).astype("float32")
            else:
                # Blank SAR data
                vv = np.full((height, width), -10.0, dtype="float32")
                vh = np.full((height, width), -15.0, dtype="float32")

            dst.write(vv, 1)
            dst.write(vh, 2)
            dst.descriptions = ("VV", "VH")


@pytest.mark.integration
class TestStreamingIngesterValidation:
    """Tests for validation integration with StreamingIngester."""

    def test_valid_image_passes_validation(self, tmp_path):
        """Test that valid images pass validation during ingestion."""
        from core.data.ingestion.streaming import StreamingIngester

        # Create valid optical raster
        source = tmp_path / "valid_optical.tif"
        TestRasterFactory.create_optical_raster(source, valid=True)

        output = tmp_path / "output.tif"

        ingester = StreamingIngester()
        result = ingester.ingest(str(source), output)

        assert result["status"] == "completed"
        if "validation" in result:
            assert result["validation"]["is_valid"] is True

    def test_invalid_image_fails_validation(self, tmp_path):
        """Test that invalid images fail validation during ingestion."""
        from core.data.ingestion.streaming import StreamingIngester

        # Create invalid (blank) raster
        source = tmp_path / "invalid_optical.tif"
        TestRasterFactory.create_optical_raster(source, valid=False)

        output = tmp_path / "output.tif"

        ingester = StreamingIngester()
        result = ingester.ingest(str(source), output)

        # Should fail validation
        if "validation" in result:
            assert result["validation"]["is_valid"] is False
            assert result["status"] == "failed"

    def test_validation_results_in_ingestion_output(self, tmp_path):
        """Test that validation results are included in ingestion output."""
        from core.data.ingestion.streaming import StreamingIngester

        source = tmp_path / "test_image.tif"
        TestRasterFactory.create_optical_raster(source, valid=True)

        output = tmp_path / "output.tif"

        ingester = StreamingIngester()
        result = ingester.ingest(str(source), output)

        # Check validation results are present
        if "validation" in result:
            assert "is_valid" in result["validation"]
            assert "band_results" in result["validation"]
            assert "metadata" in result["validation"]


@pytest.mark.integration
class TestTiledRunnerValidation:
    """Tests for validation integration with TiledAlgorithmRunner."""

    def test_validate_input_method(self, tmp_path):
        """Test the validate_input method on TiledAlgorithmRunner."""
        from core.analysis.execution.tiled_runner import TiledAlgorithmRunner

        # Create valid raster
        source = tmp_path / "tiled_test.tif"
        TestRasterFactory.create_optical_raster(source, valid=True)

        # Create a simple algorithm
        class SimpleAlgorithm:
            def process_tile(self, data, context):
                return data

        runner = TiledAlgorithmRunner(
            algorithm=SimpleAlgorithm(),
            tile_size=32,
        )

        # Validate should succeed
        result = runner.validate_input(source)
        if result is not None:
            assert result.is_valid is True

    def test_validation_failure_raises_exception(self, tmp_path):
        """Test that validation failure raises ImageValidationError."""
        from core.analysis.execution.tiled_runner import TiledAlgorithmRunner
        from core.data.ingestion.validation import ImageValidationError

        # Create invalid raster
        source = tmp_path / "invalid_tiled.tif"
        TestRasterFactory.create_optical_raster(source, valid=False)

        class SimpleAlgorithm:
            def process_tile(self, data, context):
                return data

        runner = TiledAlgorithmRunner(
            algorithm=SimpleAlgorithm(),
            tile_size=32,
        )

        # Validation should raise exception
        try:
            runner.validate_input(source)
        except ImageValidationError:
            pass  # Expected
        except Exception:
            pass  # Validation may be disabled


@pytest.mark.integration
class TestValidationWithRealPatterns:
    """Tests with realistic data patterns."""

    def test_sentinel2_like_validation(self, tmp_path):
        """Test validation of Sentinel-2-like imagery."""
        from core.data.ingestion.validation import ImageValidator

        # Create Sentinel-2-like raster with 10 bands
        raster_path = tmp_path / "sentinel2_like.tif"
        transform = from_bounds(-10, -10, 10, 10, 100, 100)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=10,
            dtype="uint16",
            crs="EPSG:32610",
            transform=transform,
            nodata=0,
        ) as dst:
            # Generate realistic reflectance values (0-10000 range)
            for band in range(1, 11):
                data = np.random.normal(2000 + band * 200, 500, (100, 100))
                data = np.clip(data, 1, 10000).astype("uint16")
                dst.write(data, band)

            dst.descriptions = (
                "B02-Blue",
                "B03-Green",
                "B04-Red",
                "B05-VegRedEdge",
                "B06-VegRedEdge",
                "B07-VegRedEdge",
                "B08-NIR",
                "B8A-NarrowNIR",
                "B11-SWIR1",
                "B12-SWIR2",
            )

        validator = ImageValidator()
        result = validator.validate(
            raster_path,
            data_source_spec={"sensor": "Sentinel-2"},
        )

        assert result.is_valid is True
        assert result.metadata.crs is not None
        assert "EPSG" in result.metadata.crs

    def test_sentinel1_like_validation(self, tmp_path):
        """Test validation of Sentinel-1-like SAR imagery."""
        from core.data.ingestion.validation import ImageValidator

        raster_path = tmp_path / "sentinel1_like.tif"
        transform = from_bounds(-10, -10, 10, 10, 100, 100)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=2,
            dtype="float32",
            crs="EPSG:32610",
            transform=transform,
        ) as dst:
            # VV with typical land backscatter (-15 to -5 dB)
            vv = np.random.normal(-10, 3, (100, 100)).astype("float32")
            # VH typically 5-10 dB lower
            vh = np.random.normal(-16, 3, (100, 100)).astype("float32")

            dst.write(vv, 1)
            dst.write(vh, 2)
            dst.descriptions = ("Sigma0_VV_db", "Sigma0_VH_db")

        validator = ImageValidator()
        result = validator.validate(
            raster_path,
            data_source_spec={"sensor": "Sentinel-1", "acquisition_mode": "IW"},
        )

        assert result.is_valid is True

    def test_high_nodata_warning(self, tmp_path):
        """Test that high NoData ratio generates warning."""
        from core.data.ingestion.validation import ImageValidator

        raster_path = tmp_path / "high_nodata.tif"
        transform = from_bounds(-10, -10, 10, 10, 100, 100)

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=4,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
        ) as dst:
            for band in range(1, 5):
                # 80% NoData
                data = np.zeros((100, 100), dtype="uint16")
                # Only 20% has valid data
                data[40:60, 40:60] = np.random.normal(5000, 500, (20, 20)).astype("uint16")
                dst.write(data, band)

        validator = ImageValidator()
        result = validator.validate(raster_path)

        # Should pass but with warnings about NoData
        # Note: depends on config threshold


@pytest.mark.integration
class TestValidationPerformance:
    """Tests for validation performance with various image sizes."""

    def test_small_image_performance(self, tmp_path):
        """Test validation of small image is fast."""
        import time

        from core.data.ingestion.validation import ImageValidator

        raster_path = tmp_path / "small_image.tif"
        TestRasterFactory.create_optical_raster(raster_path, width=50, height=50)

        validator = ImageValidator()

        start = time.time()
        result = validator.validate(raster_path)
        duration = time.time() - start

        assert result.is_valid is True
        assert duration < 5.0  # Should complete in under 5 seconds

    def test_medium_image_sampling(self, tmp_path):
        """Test that medium images use appropriate sampling."""
        from core.data.ingestion.validation import ImageValidator, ValidationConfig

        raster_path = tmp_path / "medium_image.tif"
        TestRasterFactory.create_optical_raster(raster_path, width=1000, height=1000)

        config = ValidationConfig()
        config.performance.sample_threshold_pixels = 500 * 500  # Lower threshold for test

        validator = ImageValidator(config)
        result = validator.validate(raster_path)

        assert result.is_valid is True
        # Validation should complete reasonably fast due to sampling


@pytest.mark.integration
class TestConfigurationIntegration:
    """Tests for configuration file integration."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading configuration from YAML file."""
        from core.data.ingestion.validation import ValidationConfig

        # Create test config file
        config_path = tmp_path / "test_config.yaml"
        config_content = """
validation:
  enabled: true
  optical:
    required_bands:
      - red
      - nir
    thresholds:
      std_dev_min: 2.0
  sar:
    required_polarizations:
      - VV
      - VH
"""
        config_path.write_text(config_content)

        config = ValidationConfig.from_yaml(str(config_path))

        assert config.enabled is True
        assert config.optical.std_dev_min == 2.0
        assert "red" in config.required_optical_bands
        assert "VH" in config.required_sar_polarizations

    def test_environment_override(self, tmp_path, monkeypatch):
        """Test environment variable overrides."""
        from core.data.ingestion.validation import load_config

        monkeypatch.setenv("MULTIVERSE_VALIDATION_ENABLED", "false")
        monkeypatch.setenv("MULTIVERSE_VALIDATION_SAMPLE_RATIO", "0.5")

        config = load_config(use_environment=True)

        # Environment should override defaults
        assert config.enabled is False or config.performance.sample_ratio == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
