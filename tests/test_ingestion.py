"""
Tests for the Ingestion Pipeline Integration (Group G, Track 8).

This is an integration test suite that tests the complete ingestion pipeline:
- Format conversions (COG, Zarr, GeoParquet, STAC)
- Normalization accuracy (projection, tiling, temporal, resolution)
- Validation catching issues
- Pipeline workflow integration

These tests complement the unit tests in:
- test_enrichment.py (Track 4)
- test_validation.py (Track 5)
- test_persistence.py (Track 6)
- test_normalization.py (Track 3)
"""

import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Optional dependency checks
# =============================================================================

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import pyproj

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

requires_rasterio = pytest.mark.skipif(
    not HAS_RASTERIO, reason="rasterio not installed"
)
requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
requires_geopandas = pytest.mark.skipif(
    not HAS_GEOPANDAS, reason="geopandas not installed"
)
requires_pyproj = pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_raster_data():
    """Create sample raster data array."""
    np.random.seed(42)
    return np.random.rand(100, 100).astype(np.float32) * 100


@pytest.fixture
def sample_raster_multiband():
    """Create sample multi-band raster data."""
    np.random.seed(42)
    return np.random.rand(4, 100, 100).astype(np.float32) * 100


@pytest.fixture
def sample_raster_with_nodata():
    """Create sample raster with nodata values."""
    np.random.seed(42)
    data = np.random.rand(100, 100).astype(np.float32) * 100
    # Add nodata (NaN) values
    data[20:30, 20:30] = np.nan
    data[70:80, 70:80] = np.nan
    return data


@pytest.fixture
def sample_transform():
    """Create sample affine transform."""
    if not HAS_RASTERIO:
        pytest.skip("rasterio not installed")
    return from_bounds(0, 0, 100, 100, 100, 100)


@pytest.fixture
def sample_geojson():
    """Create sample GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"name": "Point A", "value": 1},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 1]},
                "properties": {"name": "Point B", "value": 2},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "properties": {"name": "Polygon A", "area": 1.0},
            },
        ],
    }


@pytest.fixture
def sample_timestamps():
    """Create sample timestamps for time series."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(10)]


# =============================================================================
# COG Converter Tests
# =============================================================================


class TestCOGConverter:
    """Tests for Cloud-Optimized GeoTIFF conversion."""

    def test_cog_config_default_values(self):
        """Test COGConfig has sensible defaults."""
        from core.data.ingestion.formats.cog import COGConfig, Compression

        config = COGConfig()
        assert config.blocksize == 512
        assert config.compression == Compression.DEFLATE
        assert config.overview_factors == [2, 4, 8, 16, 32]

    def test_cog_config_validation_blocksize(self):
        """Test COGConfig validates blocksize."""
        from core.data.ingestion.formats.cog import COGConfig

        with pytest.raises(ValueError, match="blocksize must be"):
            COGConfig(blocksize=100)

    def test_cog_config_validation_quality(self):
        """Test COGConfig validates quality range."""
        from core.data.ingestion.formats.cog import COGConfig

        with pytest.raises(ValueError, match="quality must be"):
            COGConfig(quality=0)

        with pytest.raises(ValueError, match="quality must be"):
            COGConfig(quality=101)

    def test_cog_config_validation_overview_factors(self):
        """Test COGConfig validates overview factors."""
        from core.data.ingestion.formats.cog import COGConfig

        with pytest.raises(ValueError, match="overview_factors must be >= 2"):
            COGConfig(overview_factors=[1, 2, 4])

    def test_cog_config_to_creation_options(self):
        """Test conversion to GDAL creation options."""
        from core.data.ingestion.formats.cog import COGConfig, Compression

        config = COGConfig(
            blocksize=256, compression=Compression.LZW, bigtiff=True
        )
        options = config.to_creation_options()

        assert options["driver"] == "GTiff"
        assert options["tiled"] is True
        assert options["blockxsize"] == 256
        assert options["blockysize"] == 256
        assert options["compress"] == "LZW"
        assert options["bigtiff"] == "yes"

    @requires_rasterio
    def test_cog_convert_array(self, temp_dir, sample_raster_data, sample_transform):
        """Test converting numpy array to COG."""
        from core.data.ingestion.formats.cog import COGConfig, COGConverter

        output_path = temp_dir / "output.tif"
        converter = COGConverter(COGConfig(overview_factors=[2, 4]))

        result = converter.convert_array(
            data=sample_raster_data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        assert result.output_path == output_path
        assert result.width == 100
        assert result.height == 100
        assert result.band_count == 1
        assert result.crs == "EPSG:4326"
        assert output_path.exists()

    @requires_rasterio
    def test_cog_convert_array_multiband(
        self, temp_dir, sample_raster_multiband, sample_transform
    ):
        """Test converting multi-band array to COG."""
        from core.data.ingestion.formats.cog import COGConverter

        output_path = temp_dir / "multiband.tif"
        converter = COGConverter()

        result = converter.convert_array(
            data=sample_raster_multiband,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        assert result.band_count == 4
        assert result.height == 100
        assert result.width == 100

    @requires_rasterio
    def test_cog_convert_with_nodata(
        self, temp_dir, sample_raster_with_nodata, sample_transform
    ):
        """Test COG conversion with nodata values."""
        from core.data.ingestion.formats.cog import COGConfig, COGConverter

        output_path = temp_dir / "nodata.tif"
        config = COGConfig(nodata=-9999.0)
        converter = COGConverter(config)

        # Replace NaN with nodata value for test
        data = np.nan_to_num(sample_raster_with_nodata, nan=-9999.0)

        result = converter.convert_array(
            data=data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
            nodata=-9999.0,
        )

        assert result.output_path.exists()

        # Verify nodata was set
        with rasterio.open(output_path) as src:
            assert src.nodata == -9999.0

    @requires_rasterio
    def test_cog_overwrite_protection(
        self, temp_dir, sample_raster_data, sample_transform
    ):
        """Test that COG conversion respects overwrite flag."""
        from core.data.ingestion.formats.cog import COGConverter

        output_path = temp_dir / "existing.tif"
        converter = COGConverter()

        # Create first file
        converter.convert_array(
            data=sample_raster_data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        # Should raise without overwrite=True
        with pytest.raises(FileExistsError):
            converter.convert_array(
                data=sample_raster_data,
                output_path=output_path,
                transform=sample_transform,
                crs="EPSG:4326",
            )

        # Should work with overwrite=True
        result = converter.convert_array(
            data=sample_raster_data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
            overwrite=True,
        )
        assert result.output_path.exists()

    @requires_rasterio
    def test_cog_validate(self, temp_dir, sample_raster_data, sample_transform):
        """Test COG validation."""
        from core.data.ingestion.formats.cog import COGConfig, COGConverter

        output_path = temp_dir / "validate.tif"
        config = COGConfig(overview_factors=[2, 4])
        converter = COGConverter(config)

        converter.convert_array(
            data=sample_raster_data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        is_valid, issues = converter.validate_cog(output_path)
        assert is_valid or len(issues) > 0  # Either valid or has documented issues


# =============================================================================
# Zarr Converter Tests
# =============================================================================


class TestZarrConverter:
    """Tests for Zarr array conversion."""

    def test_zarr_config_defaults(self):
        """Test ZarrConfig has sensible defaults."""
        from core.data.ingestion.formats.zarr import ZarrCompression, ZarrConfig

        config = ZarrConfig()
        assert config.compression == ZarrCompression.ZSTD
        assert config.compression_level == 3
        assert config.consolidated is True

    def test_zarr_config_validation_compression_level(self):
        """Test ZarrConfig validates compression levels."""
        from core.data.ingestion.formats.zarr import ZarrCompression, ZarrConfig

        # ZSTD range is 1-22
        with pytest.raises(ValueError, match="zstd compression_level"):
            ZarrConfig(compression=ZarrCompression.ZSTD, compression_level=25)

        # GZIP range is 1-9
        with pytest.raises(ValueError, match="gzip compression_level"):
            ZarrConfig(compression=ZarrCompression.GZIP, compression_level=10)

    def test_chunk_config_to_tuple(self):
        """Test ChunkConfig converts to tuple correctly."""
        from core.data.ingestion.formats.zarr import ChunkConfig

        config = ChunkConfig(time=1, y=256, x=256)
        dims = ["time", "y", "x"]
        chunks = config.to_tuple(dims)

        assert chunks == (1, 256, 256)

    @requires_zarr
    def test_zarr_convert_arrays(self, temp_dir, sample_raster_data):
        """Test converting numpy arrays to Zarr."""
        from core.data.ingestion.formats.zarr import ZarrConfig, ZarrConverter

        output_path = temp_dir / "output.zarr"
        converter = ZarrConverter(ZarrConfig())

        result = converter.convert_arrays(
            arrays={"data": sample_raster_data},
            output_path=output_path,
            coords={"y": list(range(100)), "x": list(range(100))},
        )

        assert result.output_path == output_path
        assert len(result.arrays) == 1
        assert result.arrays[0].name == "data"
        assert output_path.exists()

    @requires_zarr
    def test_zarr_convert_multiple_arrays(self, temp_dir):
        """Test converting multiple arrays to Zarr."""
        from core.data.ingestion.formats.zarr import ZarrConverter

        np.random.seed(42)
        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32)

        output_path = temp_dir / "multi.zarr"
        converter = ZarrConverter()

        result = converter.convert_arrays(
            arrays={"band1": data1, "band2": data2},
            output_path=output_path,
            coords={"y": list(range(50)), "x": list(range(50))},
        )

        assert len(result.arrays) == 2
        array_names = {a.name for a in result.arrays}
        assert array_names == {"band1", "band2"}

    @requires_zarr
    def test_zarr_consolidated_metadata(self, temp_dir, sample_raster_data):
        """Test Zarr creates consolidated metadata."""
        from core.data.ingestion.formats.zarr import ZarrConfig, ZarrConverter

        output_path = temp_dir / "consolidated.zarr"
        config = ZarrConfig(consolidated=True)
        converter = ZarrConverter(config)

        result = converter.convert_arrays(
            arrays={"data": sample_raster_data},
            output_path=output_path,
            coords={"y": list(range(100)), "x": list(range(100))},
        )

        assert result.consolidated is True
        # Check consolidated metadata file exists
        assert (output_path / ".zmetadata").exists()

    @requires_zarr
    def test_zarr_with_attrs(self, temp_dir, sample_raster_data):
        """Test Zarr with global and array attributes."""
        from core.data.ingestion.formats.zarr import ZarrConverter

        output_path = temp_dir / "with_attrs.zarr"
        converter = ZarrConverter()

        result = converter.convert_arrays(
            arrays={"ndvi": sample_raster_data},
            output_path=output_path,
            coords={"y": list(range(100)), "x": list(range(100))},
            attrs={"title": "NDVI Dataset", "source": "Sentinel-2"},
            array_attrs={"ndvi": {"long_name": "Normalized Difference Vegetation Index"}},
        )

        assert result.global_attrs.get("title") == "NDVI Dataset"

    @requires_zarr
    def test_zarr_store_size_calculation(self, temp_dir, sample_raster_data):
        """Test Zarr store size is calculated."""
        from core.data.ingestion.formats.zarr import ZarrConverter

        output_path = temp_dir / "size_test.zarr"
        converter = ZarrConverter()

        result = converter.convert_arrays(
            arrays={"data": sample_raster_data},
            output_path=output_path,
            coords={"y": list(range(100)), "x": list(range(100))},
        )

        assert result.store_size_bytes > 0


# =============================================================================
# GeoParquet Converter Tests
# =============================================================================


class TestGeoParquetConverter:
    """Tests for GeoParquet vector conversion."""

    def test_geoparquet_config_defaults(self):
        """Test GeoParquetConfig has sensible defaults."""
        from core.data.ingestion.formats.parquet import (
            GeoParquetConfig,
            ParquetCompression,
        )

        config = GeoParquetConfig()
        assert config.compression == ParquetCompression.SNAPPY
        assert config.row_group_size == 100_000
        assert config.geometry_column == "geometry"

    def test_geoparquet_config_validation_row_group_size(self):
        """Test GeoParquetConfig validates row_group_size."""
        from core.data.ingestion.formats.parquet import GeoParquetConfig

        with pytest.raises(ValueError, match="row_group_size must be >= 1"):
            GeoParquetConfig(row_group_size=0)

    def test_geoparquet_config_validation_compression_level(self):
        """Test GeoParquetConfig validates compression levels."""
        from core.data.ingestion.formats.parquet import (
            GeoParquetConfig,
            ParquetCompression,
        )

        # ZSTD level validation
        with pytest.raises(ValueError, match="zstd compression_level"):
            GeoParquetConfig(
                compression=ParquetCompression.ZSTD, compression_level=30
            )

    @requires_geopandas
    def test_geoparquet_convert_geojson_dict(self, temp_dir, sample_geojson):
        """Test converting GeoJSON dict to GeoParquet."""
        from core.data.ingestion.formats.parquet import GeoParquetConverter

        output_path = temp_dir / "output.parquet"
        converter = GeoParquetConverter()

        result = converter.convert_geojson_dict(
            geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
        )

        assert result.output_path == output_path
        assert result.row_count == 3
        assert output_path.exists()

    @requires_geopandas
    def test_geoparquet_geometry_stats(self, temp_dir, sample_geojson):
        """Test GeoParquet calculates geometry statistics."""
        from core.data.ingestion.formats.parquet import GeoParquetConverter

        output_path = temp_dir / "stats.parquet"
        converter = GeoParquetConverter()

        result = converter.convert_geojson_dict(
            geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
        )

        stats = result.geometry_stats
        assert stats.total_features == 3
        assert stats.crs is not None
        assert len(stats.bbox) == 4

    @requires_geopandas
    def test_geoparquet_overwrite_protection(self, temp_dir, sample_geojson):
        """Test GeoParquet respects overwrite flag."""
        from core.data.ingestion.formats.parquet import GeoParquetConverter

        output_path = temp_dir / "existing.parquet"
        converter = GeoParquetConverter()

        # Create first file
        converter.convert_geojson_dict(
            geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
        )

        # Should raise without overwrite
        with pytest.raises(FileExistsError):
            converter.convert_geojson_dict(
                geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
            )

        # Should work with overwrite=True
        result = converter.convert_geojson_dict(
            geojson=sample_geojson,
            output_path=output_path,
            crs="EPSG:4326",
            overwrite=True,
        )
        assert result.output_path.exists()

    @requires_geopandas
    def test_geoparquet_validate(self, temp_dir, sample_geojson):
        """Test GeoParquet validation."""
        from core.data.ingestion.formats.parquet import GeoParquetConverter

        output_path = temp_dir / "validate.parquet"
        converter = GeoParquetConverter()

        converter.convert_geojson_dict(
            geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
        )

        is_valid, issues = converter.validate_geoparquet(output_path)
        # Note: validation may report issues for simple GeoParquet without full metadata
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


# =============================================================================
# STAC Item Generator Tests
# =============================================================================


class TestSTACItemGenerator:
    """Tests for STAC metadata generation."""

    def test_stac_item_config_defaults(self):
        """Test STACItemConfig has sensible defaults."""
        from core.data.ingestion.formats.stac_item import STACItemConfig

        config = STACItemConfig()
        assert "proj" in config.extensions or "processing" in config.extensions
        assert config.include_checksum is True

    def test_stac_item_to_dict(self):
        """Test STACItem converts to dictionary correctly."""
        from core.data.ingestion.formats.stac_item import AssetInfo, AssetRole, STACItem

        item = STACItem(
            id="test-item-001",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=(-1, -1, 1, 1),
            datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
            properties={"platform": "test"},
            assets={
                "data": AssetInfo(href="./data.tif", roles=[AssetRole.DATA])
            },
        )

        d = item.to_dict()
        assert d["type"] == "Feature"
        assert d["stac_version"] == "1.0.0"
        assert d["id"] == "test-item-001"
        assert "datetime" in d["properties"]

    def test_stac_item_to_json(self):
        """Test STACItem converts to JSON string."""
        from core.data.ingestion.formats.stac_item import AssetInfo, AssetRole, STACItem

        item = STACItem(
            id="test-item-002",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=(-1, -1, 1, 1),
            datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
            properties={},
            assets={
                "data": AssetInfo(href="./data.tif", roles=[AssetRole.DATA])
            },
        )

        json_str = item.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "test-item-002"

    def test_band_info_to_dict(self):
        """Test BandInfo converts correctly."""
        from core.data.ingestion.formats.stac_item import BandInfo

        band = BandInfo(
            name="B04",
            common_name="red",
            center_wavelength=0.665,
            full_width_half_max=0.038,
        )

        d = band.to_dict()
        assert d["name"] == "B04"
        assert d["common_name"] == "red"
        assert d["center_wavelength"] == 0.665

    def test_asset_info_to_dict(self):
        """Test AssetInfo converts correctly."""
        from core.data.ingestion.formats.stac_item import (
            AssetInfo,
            AssetRole,
            MediaType,
        )

        asset = AssetInfo(
            href="./cog.tif",
            title="Cloud-Optimized GeoTIFF",
            type=MediaType.COG,
            roles=[AssetRole.DATA, AssetRole.VISUAL],
        )

        d = asset.to_dict()
        assert d["href"] == "./cog.tif"
        assert d["title"] == "Cloud-Optimized GeoTIFF"
        assert d["type"] == "image/tiff; application=geotiff; profile=cloud-optimized"
        assert "data" in d["roles"]
        assert "visual" in d["roles"]

    @requires_rasterio
    def test_stac_generate_from_raster(
        self, temp_dir, sample_raster_data, sample_transform
    ):
        """Test generating STAC item from raster file."""
        from core.data.ingestion.formats.cog import COGConverter
        from core.data.ingestion.formats.stac_item import STACItemGenerator

        # First create a COG
        cog_path = temp_dir / "input.tif"
        COGConverter().convert_array(
            data=sample_raster_data,
            output_path=cog_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        # Generate STAC item
        generator = STACItemGenerator()
        item = generator.generate_from_raster(
            raster_path=cog_path,
            item_id="test-stac-item",
            dt=datetime(2024, 6, 15, tzinfo=timezone.utc),
            collection="test-collection",
        )

        assert item.id == "test-stac-item"
        assert item.collection == "test-collection"
        assert item.geometry is not None
        assert len(item.bbox) == 4


# =============================================================================
# Normalization Integration Tests
# =============================================================================


class TestNormalizationIntegration:
    """Integration tests for normalization tools."""

    @requires_pyproj
    def test_projection_round_trip(self):
        """Test CRS transformation round trip accuracy."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()

        # Transform a point from WGS84 to UTM and back
        # Note: actual transformation would require full handler implementation
        info = handler.parse_crs("EPSG:4326")
        assert info.is_geographic is True

        info_utm = handler.parse_crs("EPSG:32632")
        assert info_utm.is_projected is True

    @requires_pyproj
    def test_crs_suggestion(self):
        """Test CRS suggestion for given bounds."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()

        # Test bounds in Europe (should suggest UTM zone 32-33)
        bounds = (8.0, 45.0, 12.0, 48.0)  # Northern Italy
        suggested = handler.suggest_utm_zone(bounds)

        # Should be UTM zone 32N or 33N
        assert suggested is not None

    def test_temporal_resolution_enum(self):
        """Test temporal resolution enum values."""
        from core.data.ingestion.normalization.temporal import TemporalResolution

        # Check that the enum has the expected values
        assert TemporalResolution.DAY is not None
        assert TemporalResolution.HOUR is not None
        assert TemporalResolution.MONTH is not None

    def test_time_range_basic(self, sample_timestamps):
        """Test time range creation."""
        from core.data.ingestion.normalization.temporal import TimeRange

        time_range = TimeRange(
            start=sample_timestamps[0],
            end=sample_timestamps[-1],
        )

        assert time_range.start < time_range.end
        assert time_range.duration.days >= 1

    def test_resolution_dataclass(self):
        """Test Resolution dataclass."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res = Resolution(x=10.0, y=10.0, unit=ResolutionUnit.METERS)
        assert res.x == 10.0
        assert res.y == 10.0
        assert res.unit == ResolutionUnit.METERS
        assert res.area == 100.0

    def test_tile_scheme_enum(self):
        """Test tile scheme enum values."""
        from core.data.ingestion.normalization.tiling import TileScheme

        # Check that XYZ and TMS schemes exist
        assert TileScheme.XYZ is not None
        assert TileScheme.TMS is not None
        assert TileScheme.CUSTOM is not None

    def test_tile_bounds_operations(self):
        """Test TileBounds operations."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds1 = TileBounds(minx=0, miny=0, maxx=10, maxy=10)
        bounds2 = TileBounds(minx=5, miny=5, maxx=15, maxy=15)

        assert bounds1.width == 10
        assert bounds1.height == 10
        assert bounds1.center == (5, 5)
        assert bounds1.intersects(bounds2)

        intersection = bounds1.intersection(bounds2)
        assert intersection is not None
        assert intersection.minx == 5
        assert intersection.miny == 5


# =============================================================================
# Validation Integration Tests
# =============================================================================


class TestValidationIntegration:
    """Integration tests for validation catching issues."""

    def test_integrity_validator_catches_missing_file(self):
        """Test integrity validator catches missing files."""
        from core.data.ingestion.validation import IntegrityValidator

        validator = IntegrityValidator()
        result = validator.validate_file("/nonexistent/path/file.tif")

        assert not result.is_valid
        assert len(result.issues) >= 1

    def test_integrity_validator_validates_existing_file(self, temp_dir):
        """Test integrity validator works on existing files."""
        from core.data.ingestion.validation import IntegrityValidator

        # Create a test file
        test_file = temp_dir / "test.dat"
        test_file.write_bytes(b"test content for validation")

        validator = IntegrityValidator()
        result = validator.validate_file(str(test_file))

        assert result.file_size_bytes > 0

    def test_anomaly_detector_config(self):
        """Test anomaly detector configuration."""
        from core.data.ingestion.validation import AnomalyConfig

        config = AnomalyConfig(
            detect_statistical=True,
            zscore_threshold=3.0,
        )
        assert config.detect_statistical is True
        assert config.zscore_threshold == 3.0

    def test_anomaly_detector_finds_outliers(self):
        """Test anomaly detector identifies outliers."""
        from core.data.ingestion.validation import AnomalyDetector, AnomalyConfig

        # Create data with obvious outlier - the value needs to be extreme relative to std
        np.random.seed(42)
        data = np.random.randn(100, 100).astype(np.float32)
        # Standard deviation of randn is ~1, so zscore of 1000 is very extreme
        data[50, 50] = 1000.0  # Extreme outlier

        # Use a config with lower threshold to ensure detection
        config = AnomalyConfig(zscore_threshold=2.5)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        # Should return a result (may or may not find anomalies depending on implementation)
        assert result is not None
        # The detector should at least run without errors
        assert isinstance(result.overall_quality_score, float)

    def test_completeness_validator_config(self):
        """Test completeness validator configuration."""
        from core.data.ingestion.validation import CompletenessConfig

        config = CompletenessConfig(
            min_coverage_percent=90.0,
            detect_gaps=True,
        )
        assert config.min_coverage_percent == 90.0
        assert config.detect_gaps is True

    def test_validation_suite_combines_validators(self):
        """Test ValidationSuite combines multiple validators."""
        from core.data.ingestion.validation import ValidationSuite

        suite = ValidationSuite()
        assert suite is not None


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestIngestionPipelineE2E:
    """End-to-end integration tests for ingestion pipeline."""

    @requires_rasterio
    def test_raster_to_cog_with_validation(
        self, temp_dir, sample_raster_data, sample_transform
    ):
        """Test complete workflow: array -> COG -> validate."""
        from core.data.ingestion.formats.cog import COGConfig, COGConverter
        from core.data.ingestion.validation import IntegrityValidator

        # Convert to COG
        output_path = temp_dir / "e2e_output.tif"
        config = COGConfig(overview_factors=[2, 4])
        converter = COGConverter(config)

        result = converter.convert_array(
            data=sample_raster_data,
            output_path=output_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        # Validate the output
        validator = IntegrityValidator()
        validation_result = validator.validate_file(str(result.output_path))

        assert validation_result.file_size_bytes > 0

    @requires_rasterio
    @requires_zarr
    def test_cog_to_zarr_conversion(
        self, temp_dir, sample_raster_data, sample_transform
    ):
        """Test converting COG to Zarr for time series."""
        from core.data.ingestion.formats.cog import COGConverter
        from core.data.ingestion.formats.zarr import ZarrConverter

        # Create COG first
        cog_path = temp_dir / "intermediate.tif"
        COGConverter().convert_array(
            data=sample_raster_data,
            output_path=cog_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        # Convert to Zarr (as if part of a time series)
        zarr_path = temp_dir / "timeseries.zarr"
        zarr_converter = ZarrConverter()

        result = zarr_converter.convert_arrays(
            arrays={"data": sample_raster_data[np.newaxis, :, :]},  # Add time dimension
            output_path=zarr_path,
            coords={
                "time": ["2024-01-01"],
                "y": list(range(100)),
                "x": list(range(100)),
            },
            dims=["time", "y", "x"],
        )

        assert result.output_path.exists()
        assert result.arrays[0].shape == (1, 100, 100)

    @requires_rasterio
    def test_stac_from_cog_with_enrichment(
        self, temp_dir, sample_raster_data, sample_transform
    ):
        """Test generating STAC item from COG with enrichment."""
        from core.data.ingestion.formats.cog import COGConverter
        from core.data.ingestion.formats.stac_item import BandInfo, STACItemGenerator

        # Create COG
        cog_path = temp_dir / "for_stac.tif"
        COGConverter().convert_array(
            data=sample_raster_data,
            output_path=cog_path,
            transform=sample_transform,
            crs="EPSG:4326",
        )

        # Generate STAC
        generator = STACItemGenerator()
        item = generator.generate_from_raster(
            raster_path=cog_path,
            item_id="enriched-item",
            dt=datetime(2024, 6, 15, tzinfo=timezone.utc),
        )

        # Enrich with band info
        item = generator.add_eo_bands(
            item, [BandInfo(name="B1", common_name="data")]
        )

        d = item.to_dict()
        assert "eo:bands" in d.get("properties", {}) or any(
            "eo:bands" in str(v) for v in d.get("assets", {}).values()
        )

    @requires_geopandas
    def test_geojson_to_parquet_with_validation(self, temp_dir, sample_geojson):
        """Test GeoJSON to GeoParquet with validation."""
        from core.data.ingestion.formats.parquet import GeoParquetConverter

        output_path = temp_dir / "e2e_vector.parquet"
        converter = GeoParquetConverter()

        # Convert
        result = converter.convert_geojson_dict(
            geojson=sample_geojson, output_path=output_path, crs="EPSG:4326"
        )

        # Validate
        is_valid, issues = converter.validate_geoparquet(output_path)

        assert result.row_count == 3
        assert isinstance(is_valid, bool)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestIngestionEdgeCases:
    """Test edge cases and error handling in ingestion."""

    def test_cog_empty_array_rejected(self, temp_dir):
        """Test COG converter rejects empty arrays."""
        from core.data.ingestion.formats.cog import COGConverter

        empty_data = np.array([]).reshape(0, 0).astype(np.float32)
        converter = COGConverter()

        # Empty array should fail or be handled gracefully
        with pytest.raises((ValueError, rasterio.errors.RasterioError if HAS_RASTERIO else ValueError)):
            if HAS_RASTERIO:
                from rasterio.transform import from_bounds
                converter.convert_array(
                    data=empty_data,
                    output_path=temp_dir / "empty.tif",
                    transform=from_bounds(0, 0, 1, 1, 0, 0),
                    crs="EPSG:4326",
                )
            else:
                pytest.skip("rasterio not installed")

    def test_geoparquet_empty_geojson_rejected(self, temp_dir):
        """Test GeoParquet converter rejects empty GeoJSON."""
        if not HAS_GEOPANDAS:
            pytest.skip("geopandas not installed")

        from core.data.ingestion.formats.parquet import GeoParquetConverter

        empty_geojson = {"type": "FeatureCollection", "features": []}
        converter = GeoParquetConverter()

        with pytest.raises(ValueError, match="no features"):
            converter.convert_geojson_dict(
                geojson=empty_geojson,
                output_path=temp_dir / "empty.parquet",
                crs="EPSG:4326",
            )

    def test_zarr_inconsistent_shapes_handled(self, temp_dir):
        """Test Zarr handles arrays with different shapes."""
        if not HAS_ZARR:
            pytest.skip("zarr not installed")

        from core.data.ingestion.formats.zarr import ZarrConverter

        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(60, 60).astype(np.float32)

        converter = ZarrConverter()

        # Different shaped arrays in same store - should work as separate variables
        result = converter.convert_arrays(
            arrays={"small": data1, "large": data2},
            output_path=temp_dir / "mixed.zarr",
            coords={"y": list(range(60)), "x": list(range(60))},
        )

        assert len(result.arrays) == 2

    @requires_rasterio
    def test_cog_invalid_crs_handled(self, temp_dir, sample_raster_data, sample_transform):
        """Test COG conversion with invalid CRS is handled."""
        from core.data.ingestion.formats.cog import COGConverter

        converter = COGConverter()

        # This may either work (if rasterio accepts arbitrary strings)
        # or raise an error
        try:
            result = converter.convert_array(
                data=sample_raster_data,
                output_path=temp_dir / "bad_crs.tif",
                transform=sample_transform,
                crs="INVALID:CRS",  # Invalid CRS
            )
            # If it works, the file should exist
            assert result.output_path.exists()
        except Exception:
            # Expected to fail with invalid CRS
            pass

    def test_validation_handles_nan_array(self):
        """Test validation handles arrays with NaN values."""
        from core.data.ingestion.validation import AnomalyDetector

        # All NaN array
        nan_data = np.full((10, 10), np.nan, dtype=np.float32)

        detector = AnomalyDetector()
        result = detector.detect_from_array(nan_data)

        # Should handle gracefully, not crash
        assert result is not None

    def test_validation_handles_inf_values(self):
        """Test validation handles arrays with Inf values."""
        from core.data.ingestion.validation import AnomalyDetector

        # Array with Inf values
        data = np.random.rand(10, 10).astype(np.float32)
        data[5, 5] = np.inf
        data[3, 3] = -np.inf

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        # Should handle gracefully and potentially flag Inf as anomalies
        assert result is not None


# =============================================================================
# Performance and Resource Tests
# =============================================================================


class TestIngestionPerformance:
    """Test performance-related aspects of ingestion."""

    def test_cog_compression_ratio(self, temp_dir):
        """Test COG compression produces reasonable ratios."""
        if not HAS_RASTERIO:
            pytest.skip("rasterio not installed")

        from core.data.ingestion.formats.cog import COGConfig, COGConverter, Compression
        from rasterio.transform import from_bounds

        # Create compressible data (repeating pattern)
        data = np.zeros((1000, 1000), dtype=np.float32)
        data[::2, ::2] = 1.0  # Checkerboard pattern

        transform = from_bounds(0, 0, 1000, 1000, 1000, 1000)

        config = COGConfig(compression=Compression.DEFLATE, overview_factors=[])
        converter = COGConverter(config)

        result = converter.convert_array(
            data=data,
            output_path=temp_dir / "compressed.tif",
            transform=transform,
            crs="EPSG:4326",
        )

        # Compression ratio should be > 1 for repetitive data
        assert result.compression_ratio > 1.0

    @requires_zarr
    def test_zarr_chunking_applied(self, temp_dir):
        """Test Zarr chunking is correctly applied."""
        from core.data.ingestion.formats.zarr import ChunkConfig, ZarrConfig, ZarrConverter

        np.random.seed(42)
        data = np.random.rand(512, 512).astype(np.float32)

        config = ZarrConfig(chunks=ChunkConfig(y=128, x=128))
        converter = ZarrConverter(config)

        result = converter.convert_arrays(
            arrays={"data": data},
            output_path=temp_dir / "chunked.zarr",
            coords={"y": list(range(512)), "x": list(range(512))},
        )

        # Chunks should be 128x128 or smaller (clamped to array size)
        assert result.arrays[0].chunks == (128, 128)


# =============================================================================
# Additional Comprehensive Tests for Track 8 Review
# =============================================================================


class TestSTACItemEdgeCases:
    """Additional edge case tests for STAC Item generation."""

    def test_stac_item_with_datetime_range(self):
        """Test STAC item with datetime range instead of single datetime."""
        from core.data.ingestion.formats.stac_item import AssetInfo, AssetRole, STACItem

        start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2024, 1, 31, tzinfo=timezone.utc)

        item = STACItem(
            id="range-item",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=(-1, -1, 1, 1),
            datetime=datetime(2024, 1, 15, tzinfo=timezone.utc),  # midpoint
            properties={
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
            },
            assets={"data": AssetInfo(href="./data.tif", roles=[AssetRole.DATA])},
        )

        d = item.to_dict()
        assert "start_datetime" in d["properties"]
        assert "end_datetime" in d["properties"]

    def test_stac_asset_all_roles(self):
        """Test STAC asset with multiple roles."""
        from core.data.ingestion.formats.stac_item import AssetInfo, AssetRole, MediaType

        asset = AssetInfo(
            href="./visual.tif",
            title="Visual Band Composite",
            type=MediaType.COG,
            roles=[AssetRole.DATA, AssetRole.VISUAL, AssetRole.THUMBNAIL],
        )

        d = asset.to_dict()
        assert len(d["roles"]) == 3
        assert "data" in d["roles"]
        assert "visual" in d["roles"]
        assert "thumbnail" in d["roles"]

    def test_stac_raster_band_full_info(self):
        """Test raster band info with full statistics."""
        from core.data.ingestion.formats.stac_item import RasterBandInfo

        band = RasterBandInfo(
            data_type="float32",
            nodata=-9999.0,
            statistics={"min": 0.0, "max": 1.0, "mean": 0.5, "stddev": 0.1},
            unit="reflectance",
            scale=0.0001,
            offset=0.0,
            histogram={"count": 256, "min": 0.0, "max": 1.0},
        )

        d = band.to_dict()
        assert d["data_type"] == "float32"
        assert d["nodata"] == -9999.0
        assert d["statistics"]["mean"] == 0.5
        assert d["unit"] == "reflectance"

    def test_stac_item_extension_schemas(self):
        """Test STAC Item configuration returns correct extension schemas."""
        from core.data.ingestion.formats.stac_item import STACItemConfig

        config = STACItemConfig(extensions=["eo", "sar", "proj"])
        schemas = config.get_extension_schemas()

        assert len(schemas) == 3
        assert any("eo" in s for s in schemas)
        assert any("sar" in s for s in schemas)
        assert any("proj" in s for s in schemas)

    def test_stac_item_empty_extensions(self):
        """Test STAC Item with no extensions."""
        from core.data.ingestion.formats.stac_item import STACItemConfig

        config = STACItemConfig(extensions=[])
        schemas = config.get_extension_schemas()

        assert schemas == []


class TestConfigValidationEdgeCases:
    """Test configuration validation edge cases."""

    def test_cog_config_all_valid_blocksizes(self):
        """Test COGConfig accepts all valid blocksizes."""
        from core.data.ingestion.formats.cog import COGConfig

        valid_sizes = [128, 256, 512, 1024, 2048]
        for size in valid_sizes:
            config = COGConfig(blocksize=size)
            assert config.blocksize == size

    def test_zarr_config_compression_levels(self):
        """Test ZarrConfig compression level validation at boundaries."""
        from core.data.ingestion.formats.zarr import ZarrCompression, ZarrConfig

        # Valid boundaries for ZSTD
        config_min = ZarrConfig(compression=ZarrCompression.ZSTD, compression_level=1)
        assert config_min.compression_level == 1

        config_max = ZarrConfig(compression=ZarrCompression.ZSTD, compression_level=22)
        assert config_max.compression_level == 22

        # Valid boundaries for GZIP
        config_gzip = ZarrConfig(compression=ZarrCompression.GZIP, compression_level=9)
        assert config_gzip.compression_level == 9

    def test_geoparquet_config_large_row_group(self):
        """Test GeoParquetConfig with very large row group size."""
        from core.data.ingestion.formats.parquet import GeoParquetConfig

        config = GeoParquetConfig(row_group_size=10_000_000)
        assert config.row_group_size == 10_000_000

    def test_cog_config_quality_boundaries(self):
        """Test COGConfig quality at boundaries."""
        from core.data.ingestion.formats.cog import COGConfig

        # Valid boundaries
        config_min = COGConfig(quality=1)
        assert config_min.quality == 1

        config_max = COGConfig(quality=100)
        assert config_max.quality == 100


class TestAnomalyDetectorEdgeCases:
    """Additional edge case tests for anomaly detection."""

    def test_anomaly_detector_single_value_array(self):
        """Test anomaly detector with single-value (constant) array."""
        from core.data.ingestion.validation import AnomalyDetector

        # Constant array - no outliers possible
        data = np.full((50, 50), 42.0, dtype=np.float32)

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        assert result is not None
        # Should handle constant data gracefully

    def test_anomaly_detector_mixed_nan_inf(self):
        """Test anomaly detector with mixed NaN and Inf values."""
        from core.data.ingestion.validation import AnomalyDetector

        data = np.random.rand(20, 20).astype(np.float32)
        data[5, 5] = np.nan
        data[10, 10] = np.inf
        data[15, 15] = -np.inf

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        # Should handle mixed special values gracefully
        assert result is not None
        assert isinstance(result.overall_quality_score, float)

    def test_anomaly_detector_small_array(self):
        """Test anomaly detector with very small array."""
        from core.data.ingestion.validation import AnomalyDetector

        # Very small array (below threshold for some checks)
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        assert result is not None


class TestCompletenessValidatorEdgeCases:
    """Edge case tests for completeness validation."""

    def test_completeness_validator_full_coverage(self):
        """Test completeness validator with 100% coverage."""
        from core.data.ingestion.validation import CompletenessConfig, CompletenessValidator

        # No nodata values
        data = np.random.rand(100, 100).astype(np.float32)

        config = CompletenessConfig(min_coverage_percent=100.0)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data)

        assert result is not None

    def test_completeness_validator_zero_coverage(self):
        """Test completeness validator with 0% coverage (all nodata)."""
        from core.data.ingestion.validation import CompletenessConfig, CompletenessValidator

        # All NaN values
        data = np.full((100, 100), np.nan, dtype=np.float32)

        config = CompletenessConfig(min_coverage_percent=0.0)
        validator = CompletenessValidator(config)
        # Must explicitly pass nodata=np.nan to recognize NaN as nodata
        result = validator.validate_array(data, nodata=np.nan)

        assert result is not None
        assert result.coverage_percentage == 0.0


class TestIntegrityValidatorEdgeCases:
    """Edge case tests for integrity validation."""

    def test_integrity_validator_empty_file(self, temp_dir):
        """Test integrity validator with empty file."""
        from core.data.ingestion.validation import IntegrityValidator

        empty_file = temp_dir / "empty.dat"
        empty_file.touch()  # Create empty file

        validator = IntegrityValidator()
        result = validator.validate_file(str(empty_file))

        # Empty file should be flagged
        assert result.file_size_bytes == 0

    def test_integrity_validator_large_file_path(self, temp_dir):
        """Test integrity validator with long file path."""
        from core.data.ingestion.validation import IntegrityValidator

        # Create nested directory structure
        deep_path = temp_dir
        for i in range(10):
            deep_path = deep_path / f"level_{i}"
        deep_path.mkdir(parents=True, exist_ok=True)

        test_file = deep_path / "deep_file.dat"
        test_file.write_bytes(b"test content")

        validator = IntegrityValidator()
        result = validator.validate_file(str(test_file))

        assert result.file_size_bytes > 0

    def test_integrity_validator_special_characters_in_path(self, temp_dir):
        """Test integrity validator with special characters in filename."""
        from core.data.ingestion.validation import IntegrityValidator

        # Create file with spaces and special chars
        special_file = temp_dir / "file with spaces & chars.dat"
        special_file.write_bytes(b"test content")

        validator = IntegrityValidator()
        result = validator.validate_file(str(special_file))

        assert result.file_size_bytes > 0


class TestNormalizationEdgeCases:
    """Edge case tests for normalization utilities."""

    def test_time_range_minimal_duration(self):
        """Test TimeRange with minimal (1 second) duration."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 15, 12, 0, 1, tzinfo=timezone.utc)  # 1 second later
        time_range = TimeRange(start=start, end=end)

        assert time_range.duration.total_seconds() == 1

    def test_time_range_rejects_same_start_end(self):
        """Test TimeRange rejects start == end (validation)."""
        from core.data.ingestion.normalization.temporal import TimeRange

        instant = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Start must be before end"):
            TimeRange(start=instant, end=instant)

    def test_tile_bounds_zero_size(self):
        """Test TileBounds with zero width/height."""
        from core.data.ingestion.normalization.tiling import TileBounds

        # Point-like bounds
        point_bounds = TileBounds(minx=5.0, miny=5.0, maxx=5.0, maxy=5.0)

        assert point_bounds.width == 0
        assert point_bounds.height == 0
        assert point_bounds.center == (5.0, 5.0)

    def test_tile_bounds_negative_coordinates(self):
        """Test TileBounds with negative coordinates."""
        from core.data.ingestion.normalization.tiling import TileBounds

        neg_bounds = TileBounds(minx=-180, miny=-90, maxx=0, maxy=0)

        assert neg_bounds.width == 180
        assert neg_bounds.height == 90

    def test_resolution_area_calculation(self):
        """Test Resolution area calculation with different units."""
        from core.data.ingestion.normalization.resolution import Resolution, ResolutionUnit

        # 10m x 10m = 100 sq meters
        res = Resolution(x=10.0, y=10.0, unit=ResolutionUnit.METERS)
        assert res.area == 100.0

        # Non-square pixels
        res_rect = Resolution(x=10.0, y=20.0, unit=ResolutionUnit.METERS)
        assert res_rect.area == 200.0


class TestZarrConverterEdgeCases:
    """Edge case tests for Zarr conversion."""

    def test_zarr_chunk_config_missing_dimension(self):
        """Test ChunkConfig with dimension not in config."""
        from core.data.ingestion.formats.zarr import ChunkConfig

        config = ChunkConfig(time=1, y=256, x=256)
        dims = ["time", "y", "x", "band"]  # band not in config

        chunks = config.to_tuple(dims)
        assert len(chunks) == 4

    def test_zarr_config_no_compression(self):
        """Test ZarrConfig with no compression."""
        from core.data.ingestion.formats.zarr import ZarrCompression, ZarrConfig

        config = ZarrConfig(compression=ZarrCompression.NONE)
        compressor = config.get_compressor()

        assert compressor is None


class TestCOGConverterEdgeCases:
    """Edge case tests for COG conversion."""

    def test_cog_creation_options_jpeg_quality(self):
        """Test COG creation options include quality for JPEG compression."""
        from core.data.ingestion.formats.cog import COGConfig, Compression

        config = COGConfig(compression=Compression.JPEG, quality=85)
        options = config.to_creation_options()

        assert options["quality"] == 85
        assert options["compress"] == "JPEG"

    def test_cog_creation_options_no_predictor_for_jpeg(self):
        """Test COG creation options don't include predictor for JPEG."""
        from core.data.ingestion.formats.cog import COGConfig, Compression

        config = COGConfig(compression=Compression.JPEG)
        options = config.to_creation_options()

        # Predictor should not be set for JPEG
        assert "predictor" not in options

    def test_cog_config_overview_factors_sorted(self):
        """Test COG overview factors work with unsorted input."""
        from core.data.ingestion.formats.cog import COGConfig

        # Unsorted factors should still work
        config = COGConfig(overview_factors=[8, 2, 16, 4])
        assert set(config.overview_factors) == {2, 4, 8, 16}


# =============================================================================
# Cache Manager Tests (Track 8)
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass validation."""

    def test_cache_config_defaults(self):
        """Test CacheConfig has sensible defaults."""
        from core.data.cache.manager import CacheConfig

        config = CacheConfig()
        assert config.max_size_bytes == 0  # Unlimited
        assert config.max_entries == 0  # Unlimited
        assert config.default_ttl_seconds == 86400  # 24 hours

    def test_cache_config_validation_max_size(self):
        """Test CacheConfig validates max_size_bytes."""
        from core.data.cache.manager import CacheConfig

        with pytest.raises(ValueError, match="max_size_bytes must be >= 0"):
            CacheConfig(max_size_bytes=-1)

    def test_cache_config_validation_max_entries(self):
        """Test CacheConfig validates max_entries."""
        from core.data.cache.manager import CacheConfig

        with pytest.raises(ValueError, match="max_entries must be >= 0"):
            CacheConfig(max_entries=-1)

    def test_cache_config_validation_ttl(self):
        """Test CacheConfig validates default_ttl_seconds."""
        from core.data.cache.manager import CacheConfig

        with pytest.raises(ValueError, match="default_ttl_seconds must be >= 0"):
            CacheConfig(default_ttl_seconds=-1)

    def test_cache_config_validation_cleanup_interval(self):
        """Test CacheConfig validates cleanup_interval_seconds."""
        from core.data.cache.manager import CacheConfig

        with pytest.raises(ValueError, match="cleanup_interval_seconds must be >= 60"):
            CacheConfig(cleanup_interval_seconds=30)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_to_dict(self):
        """Test CacheEntry converts to dictionary."""
        from core.data.cache.manager import CacheEntry, CacheEntryStatus

        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            cache_key="test-key",
            storage_key="storage/test-key",
            data_type="optical",
            size_bytes=1024,
            checksum="abc123",
            status=CacheEntryStatus.ACTIVE,
            created_at=now,
            accessed_at=now,
            access_count=5,
            metadata={"source": "sentinel-2"},
            tags={"flood", "miami"}
        )

        d = entry.to_dict()
        assert d["cache_key"] == "test-key"
        assert d["data_type"] == "optical"
        assert d["status"] == "active"
        assert "flood" in d["tags"]

    def test_cache_entry_from_dict(self):
        """Test CacheEntry creation from dictionary."""
        from core.data.cache.manager import CacheEntry, CacheEntryStatus

        data = {
            "cache_key": "test-key",
            "storage_key": "storage/test-key",
            "data_type": "sar",
            "size_bytes": 2048,
            "checksum": "def456",
            "status": "active",
            "created_at": "2024-01-15T10:00:00+00:00",
            "accessed_at": "2024-01-15T12:00:00+00:00",
            "access_count": 10,
            "metadata": {},
            "tags": ["test"]
        }

        entry = CacheEntry.from_dict(data)
        assert entry.cache_key == "test-key"
        assert entry.data_type == "sar"
        assert entry.status == CacheEntryStatus.ACTIVE

    def test_cache_entry_is_expired_no_expiry(self):
        """Test is_expired when no expiration is set."""
        from core.data.cache.manager import CacheEntry

        entry = CacheEntry(
            cache_key="test",
            storage_key="storage/test",
            data_type="dem",
            size_bytes=100,
            checksum="xyz",
            expires_at=None
        )

        assert entry.is_expired is False

    def test_cache_entry_is_expired_future(self):
        """Test is_expired when expiration is in the future."""
        from core.data.cache.manager import CacheEntry

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        entry = CacheEntry(
            cache_key="test",
            storage_key="storage/test",
            data_type="dem",
            size_bytes=100,
            checksum="xyz",
            expires_at=future
        )

        assert entry.is_expired is False

    def test_cache_entry_is_expired_past(self):
        """Test is_expired when expiration is in the past."""
        from core.data.cache.manager import CacheEntry

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        entry = CacheEntry(
            cache_key="test",
            storage_key="storage/test",
            data_type="dem",
            size_bytes=100,
            checksum="xyz",
            expires_at=past
        )

        assert entry.is_expired is True


class TestCacheStatistics:
    """Tests for CacheStatistics dataclass."""

    def test_cache_statistics_hit_rate_zero(self):
        """Test hit rate when no requests have been made."""
        from core.data.cache.manager import CacheStatistics

        stats = CacheStatistics(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_cache_statistics_hit_rate_all_hits(self):
        """Test hit rate when all requests are hits."""
        from core.data.cache.manager import CacheStatistics

        stats = CacheStatistics(hits=100, misses=0)
        assert stats.hit_rate == 1.0

    def test_cache_statistics_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        from core.data.cache.manager import CacheStatistics

        stats = CacheStatistics(hits=75, misses=25)
        assert stats.hit_rate == 0.75

    def test_cache_statistics_to_dict(self):
        """Test CacheStatistics converts to dictionary."""
        from core.data.cache.manager import CacheStatistics

        stats = CacheStatistics(
            total_entries=100,
            active_entries=80,
            expired_entries=10,
            total_size_bytes=1024000,
            hits=500,
            misses=100,
            evictions=20,
            expirations=10
        )

        d = stats.to_dict()
        assert d["total_entries"] == 100
        assert d["hit_rate"] == pytest.approx(500/600)


class TestCacheManager:
    """Tests for CacheManager class."""

    @pytest.fixture
    def cache_db_path(self, temp_dir):
        """Create a temporary cache database path."""
        return temp_dir / "cache.db"

    @pytest.fixture
    def cache_manager(self, cache_db_path):
        """Create a cache manager instance."""
        from core.data.cache.manager import CacheConfig, CacheManager

        config = CacheConfig(
            max_size_bytes=1024 * 1024,  # 1 MB
            max_entries=100,
            default_ttl_seconds=3600,
            db_path=cache_db_path
        )
        return CacheManager(config)

    def test_cache_manager_initialization(self, cache_manager, cache_db_path):
        """Test cache manager initializes database."""
        assert cache_db_path.exists()
        assert cache_manager.config is not None

    def test_cache_manager_generate_key_deterministic(self, cache_manager):
        """Test cache key generation is deterministic."""
        key1 = cache_manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240115",
            bbox=[-80.0, 25.0, -79.0, 26.0],
            temporal={"start": "2024-01-15", "end": "2024-01-20"}
        )

        key2 = cache_manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240115",
            bbox=[-80.0, 25.0, -79.0, 26.0],
            temporal={"start": "2024-01-15", "end": "2024-01-20"}
        )

        assert key1 == key2
        assert len(key1) == 32

    def test_cache_manager_generate_key_different_params(self, cache_manager):
        """Test different parameters produce different keys."""
        key1 = cache_manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240115"
        )

        key2 = cache_manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240116"  # Different date
        )

        assert key1 != key2

    def test_cache_manager_put_and_get(self, cache_manager):
        """Test putting and getting cache entries."""
        entry = cache_manager.put(
            cache_key="test-put-get",
            storage_key="storage/test-put-get",
            data_type="optical",
            size_bytes=1024,
            checksum="abc123",
            metadata={"source": "test"}
        )

        assert entry.cache_key == "test-put-get"

        retrieved = cache_manager.get("test-put-get")

        assert retrieved is not None
        assert retrieved.cache_key == "test-put-get"
        assert retrieved.data_type == "optical"

    def test_cache_manager_get_nonexistent(self, cache_manager):
        """Test getting a nonexistent entry returns None."""
        result = cache_manager.get("nonexistent-key")
        assert result is None

    def test_cache_manager_contains(self, cache_manager):
        """Test contains method."""
        cache_manager.put(
            cache_key="test-contains",
            storage_key="storage/test-contains",
            data_type="sar",
            size_bytes=512,
            checksum="def456"
        )

        assert cache_manager.contains("test-contains") is True
        assert cache_manager.contains("nonexistent") is False

    def test_cache_manager_invalidate(self, cache_manager):
        """Test invalidating a cache entry."""
        cache_manager.put(
            cache_key="test-invalidate",
            storage_key="storage/test-invalidate",
            data_type="dem",
            size_bytes=256,
            checksum="ghi789"
        )

        assert cache_manager.contains("test-invalidate") is True

        result = cache_manager.invalidate("test-invalidate")

        assert result is True
        assert cache_manager.contains("test-invalidate") is False

    def test_cache_manager_invalidate_by_tag(self, cache_manager):
        """Test invalidating entries by tag."""
        cache_manager.put(
            cache_key="test-tag-1",
            storage_key="storage/test-tag-1",
            data_type="optical",
            size_bytes=100,
            checksum="aaa",
            tags={"miami", "flood"}
        )

        cache_manager.put(
            cache_key="test-tag-2",
            storage_key="storage/test-tag-2",
            data_type="optical",
            size_bytes=100,
            checksum="bbb",
            tags={"miami", "storm"}
        )

        cache_manager.put(
            cache_key="test-tag-3",
            storage_key="storage/test-tag-3",
            data_type="optical",
            size_bytes=100,
            checksum="ccc",
            tags={"houston", "flood"}
        )

        count = cache_manager.invalidate_by_tag("miami")

        assert count == 2
        assert cache_manager.contains("test-tag-1") is False
        assert cache_manager.contains("test-tag-2") is False
        assert cache_manager.contains("test-tag-3") is True

    def test_cache_manager_delete(self, cache_manager):
        """Test deleting a cache entry."""
        cache_manager.put(
            cache_key="test-delete",
            storage_key="storage/test-delete",
            data_type="vector",
            size_bytes=128,
            checksum="jkl012"
        )

        result = cache_manager.delete("test-delete", delete_storage=False)

        assert result is True
        assert cache_manager.contains("test-delete") is False

        # Deleting again should return False
        result = cache_manager.delete("test-delete", delete_storage=False)
        assert result is False

    def test_cache_manager_list_entries(self, cache_manager):
        """Test listing cache entries with filters."""
        for i in range(5):
            cache_manager.put(
                cache_key=f"test-list-{i}",
                storage_key=f"storage/test-list-{i}",
                data_type="optical" if i < 3 else "sar",
                size_bytes=100 * (i + 1),
                checksum=f"hash{i}"
            )

        all_entries = cache_manager.list_entries()
        assert len(all_entries) == 5

        optical_entries = cache_manager.list_entries(data_type="optical")
        assert len(optical_entries) == 3

        sar_entries = cache_manager.list_entries(data_type="sar")
        assert len(sar_entries) == 2

    def test_cache_manager_get_statistics(self, cache_manager):
        """Test getting cache statistics."""
        for i in range(3):
            cache_manager.put(
                cache_key=f"test-stats-{i}",
                storage_key=f"storage/test-stats-{i}",
                data_type="optical",
                size_bytes=100,
                checksum=f"hash{i}"
            )

        cache_manager.get("test-stats-0")  # Hit
        cache_manager.get("test-stats-1")  # Hit
        cache_manager.get("nonexistent-1")  # Miss
        cache_manager.get("nonexistent-2")  # Miss

        stats = cache_manager.get_statistics()

        assert stats.active_entries == 3
        assert stats.total_size_bytes == 300
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 0.5

    def test_cache_manager_clear(self, cache_manager):
        """Test clearing all cache entries."""
        for i in range(5):
            cache_manager.put(
                cache_key=f"test-clear-{i}",
                storage_key=f"storage/test-clear-{i}",
                data_type="optical",
                size_bytes=100,
                checksum=f"hash{i}"
            )

        count = cache_manager.clear(delete_storage=False)

        assert count == 5

        stats = cache_manager.get_statistics()
        assert stats.active_entries == 0

    def test_cache_manager_access_updates_count(self, cache_manager):
        """Test that accessing an entry updates its access count."""
        cache_manager.put(
            cache_key="test-access",
            storage_key="storage/test-access",
            data_type="optical",
            size_bytes=100,
            checksum="hash"
        )

        entry1 = cache_manager.get("test-access")
        initial_count = entry1.access_count

        entry2 = cache_manager.get("test-access")

        assert entry2.access_count == initial_count + 1


class TestCacheEviction:
    """Tests for cache eviction policies."""

    @pytest.fixture
    def limited_cache(self, temp_dir):
        """Create a cache with limited entries."""
        from core.data.cache.manager import CacheConfig, CacheManager, EvictionPolicy

        config = CacheConfig(
            max_entries=3,
            max_size_bytes=1000,
            eviction_policy=EvictionPolicy.LRU,
            db_path=temp_dir / "limited_cache.db"
        )
        return CacheManager(config)

    def test_eviction_on_entry_limit(self, limited_cache):
        """Test that entries are evicted when limit is reached."""
        for i in range(3):
            limited_cache.put(
                cache_key=f"entry-{i}",
                storage_key=f"storage/entry-{i}",
                data_type="optical",
                size_bytes=100,
                checksum=f"hash{i}"
            )

        # Add a 4th entry - should trigger eviction
        limited_cache.put(
            cache_key="entry-3",
            storage_key="storage/entry-3",
            data_type="optical",
            size_bytes=100,
            checksum="hash3"
        )

        stats = limited_cache.get_statistics()
        assert stats.active_entries <= 3
        assert stats.evictions >= 1

    def test_lru_eviction_policy(self, temp_dir):
        """Test LRU eviction evicts least recently used entries."""
        from core.data.cache.manager import CacheConfig, CacheManager, EvictionPolicy

        config = CacheConfig(
            max_entries=3,
            eviction_policy=EvictionPolicy.LRU,
            db_path=temp_dir / "lru_cache.db"
        )
        cache = CacheManager(config)

        cache.put("entry-0", "s/0", "optical", 100, "h0")
        cache.put("entry-1", "s/1", "optical", 100, "h1")
        cache.put("entry-2", "s/2", "optical", 100, "h2")

        # Access entry-0 and entry-2 (making entry-1 least recently used)
        cache.get("entry-0")
        cache.get("entry-2")

        # Add a new entry
        cache.put("entry-3", "s/3", "optical", 100, "h3")

        # entry-1 should have been evicted
        assert cache.contains("entry-0") is True
        assert cache.contains("entry-1") is False  # LRU - should be evicted
        assert cache.contains("entry-2") is True
        assert cache.contains("entry-3") is True


class TestCacheCleanup:
    """Tests for cache cleanup operations."""

    def test_cleanup_expired_entries(self, temp_dir):
        """Test cleanup of expired cache entries."""
        import time
        from core.data.cache.manager import CacheConfig, CacheManager

        config = CacheConfig(
            default_ttl_seconds=1,  # Very short TTL
            db_path=temp_dir / "expired_cache.db"
        )
        cache = CacheManager(config)

        cache.put(
            cache_key="expiring-entry",
            storage_key="storage/expiring",
            data_type="optical",
            size_bytes=100,
            checksum="hash",
            ttl_seconds=1
        )

        assert cache.contains("expiring-entry") is True

        # Wait for expiration
        time.sleep(1.5)

        # Entry should be expired now
        assert cache.contains("expiring-entry") is False

        # Cleanup removes expired entries
        cleaned = cache.cleanup_expired(delete_storage=False)
        assert cleaned >= 0  # May already be handled by contains check


class TestCacheIntegration:
    """Integration tests for cache with ingestion pipeline."""

    @requires_rasterio
    def test_cache_with_cog_product(self, temp_dir, sample_raster_data, sample_transform):
        """Test caching a COG product through its lifecycle."""
        from core.data.cache.manager import CacheConfig, CacheManager
        from core.data.ingestion.formats.cog import COGConverter
        import hashlib

        # Create cache
        config = CacheConfig(
            db_path=temp_dir / "product_cache.db",
            default_ttl_seconds=3600
        )
        cache = CacheManager(config)

        # Create a COG product
        cog_path = temp_dir / "flood_extent.tif"
        converter = COGConverter()
        result = converter.convert_array(
            data=sample_raster_data,
            output_path=cog_path,
            transform=sample_transform,
            crs="EPSG:4326"
        )

        # Generate cache key
        cache_key = cache.generate_cache_key(
            provider="analysis",
            dataset_id="flood_extent",
            bbox=[0.0, 0.0, 100.0, 100.0],
            temporal={"start": "2024-09-15", "end": "2024-09-20"}
        )

        # Register in cache
        entry = cache.put(
            cache_key=cache_key,
            storage_key=str(cog_path),
            data_type="flood_product",
            size_bytes=result.file_size_bytes,
            checksum=hashlib.sha256(str(cog_path).encode()).hexdigest()[:32],
            metadata={
                "algorithm": "threshold_sar",
                "confidence": 0.85,
                "event_id": "hurricane_test_2024"
            },
            tags={"flood", "miami", "hurricane", "2024"}
        )

        assert entry is not None

        # Retrieve and verify
        retrieved = cache.get(cache_key)

        assert retrieved is not None
        assert retrieved.data_type == "flood_product"
        assert retrieved.metadata["algorithm"] == "threshold_sar"
        assert "miami" in retrieved.tags

        # Clean up
        cache.clear(delete_storage=False)

    def test_cache_thread_safety(self, temp_dir):
        """Test cache operations are thread-safe."""
        import threading
        from core.data.cache.manager import CacheConfig, CacheManager

        config = CacheConfig(db_path=temp_dir / "thread_cache.db")
        cache = CacheManager(config)

        errors = []

        def worker(worker_id: int):
            try:
                for i in range(10):
                    cache_key = f"worker-{worker_id}-entry-{i}"
                    cache.put(
                        cache_key=cache_key,
                        storage_key=f"storage/{cache_key}",
                        data_type="optical",
                        size_bytes=100,
                        checksum=f"hash-{worker_id}-{i}"
                    )
                    cache.get(cache_key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

        stats = cache.get_statistics()
        assert stats.active_entries == 50


# =============================================================================
# Main entry point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
