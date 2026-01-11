"""
Format Converters for Ingestion Pipeline.

Provides converters for transforming geospatial data into cloud-native formats:
- COG: Cloud-Optimized GeoTIFF for raster data
- Zarr: Chunked arrays for multidimensional data
- GeoParquet: Columnar storage for vector data
- STAC: Metadata generation for discovery

Each converter follows a consistent pattern:
- Config dataclass for conversion options
- Result dataclass with conversion details
- Converter class with convert_* methods
- Convenience function for simple usage
"""

from core.data.ingestion.formats.cog import (
    COGConfig,
    COGConverter,
    COGResult,
    Compression,
    Predictor,
    ResamplingMethod,
    convert_to_cog,
)
from core.data.ingestion.formats.parquet import (
    GeoParquetConfig,
    GeoParquetConverter,
    GeoParquetResult,
    GeometryEncoding,
    GeometryStats,
    ParquetCompression,
    convert_to_geoparquet,
)
from core.data.ingestion.formats.stac_item import (
    AssetInfo,
    AssetRole,
    BandInfo,
    MediaType,
    RasterBandInfo,
    STACItem,
    STACItemConfig,
    STACItemGenerator,
    generate_stac_item,
)
from core.data.ingestion.formats.zarr import (
    ChunkConfig,
    ZarrArrayInfo,
    ZarrCompression,
    ZarrConfig,
    ZarrConverter,
    ZarrResult,
    ZarrStorageFormat,
    convert_to_zarr,
)

__all__ = [
    # COG
    "COGConfig",
    "COGConverter",
    "COGResult",
    "Compression",
    "Predictor",
    "ResamplingMethod",
    "convert_to_cog",
    # Zarr
    "ChunkConfig",
    "ZarrArrayInfo",
    "ZarrCompression",
    "ZarrConfig",
    "ZarrConverter",
    "ZarrResult",
    "ZarrStorageFormat",
    "convert_to_zarr",
    # GeoParquet
    "GeoParquetConfig",
    "GeoParquetConverter",
    "GeoParquetResult",
    "GeometryEncoding",
    "GeometryStats",
    "ParquetCompression",
    "convert_to_geoparquet",
    # STAC
    "AssetInfo",
    "AssetRole",
    "BandInfo",
    "MediaType",
    "RasterBandInfo",
    "STACItem",
    "STACItemConfig",
    "STACItemGenerator",
    "generate_stac_item",
]
