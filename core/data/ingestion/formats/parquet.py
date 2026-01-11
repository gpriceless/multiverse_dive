"""
GeoParquet Vector Converter.

Converts vector geospatial data to GeoParquet format for efficient
columnar storage and fast analytical queries.

GeoParquet Format Benefits:
- Columnar storage for fast attribute queries
- Efficient compression (Snappy, ZSTD)
- Partitioning support for large datasets
- Wide ecosystem support (DuckDB, GeoPandas, GDAL)
- Cloud-native with predicate pushdown
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ParquetCompression(Enum):
    """Supported compression methods for Parquet."""

    NONE = None
    SNAPPY = "snappy"
    GZIP = "gzip"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"


class GeometryEncoding(Enum):
    """Geometry encoding options for GeoParquet."""

    WKB = "WKB"  # Well-Known Binary (standard)
    GEOARROW = "geoarrow"  # Native Arrow geometry (experimental)


@dataclass
class GeoParquetConfig:
    """
    Configuration for GeoParquet conversion.

    Attributes:
        compression: Compression method
        compression_level: Compression level (for zstd: 1-22)
        row_group_size: Number of rows per row group
        geometry_column: Name of the geometry column
        geometry_encoding: Geometry encoding method
        primary_column: Primary geometry column for spatial index
        write_statistics: Write column statistics
        write_covering: Write spatial covering metadata
        schema_version: GeoParquet schema version
    """

    compression: ParquetCompression = ParquetCompression.SNAPPY
    compression_level: Optional[int] = None
    row_group_size: int = 100_000
    geometry_column: str = "geometry"
    geometry_encoding: GeometryEncoding = GeometryEncoding.WKB
    primary_column: Optional[str] = None
    write_statistics: bool = True
    write_covering: bool = True
    schema_version: str = "1.0.0"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.row_group_size < 1:
            raise ValueError(
                f"row_group_size must be >= 1, got {self.row_group_size}"
            )
        if self.compression_level is not None:
            if self.compression == ParquetCompression.ZSTD:
                if not 1 <= self.compression_level <= 22:
                    raise ValueError(
                        f"zstd compression_level must be in [1, 22], got {self.compression_level}"
                    )
            elif self.compression == ParquetCompression.GZIP:
                if not 1 <= self.compression_level <= 9:
                    raise ValueError(
                        f"gzip compression_level must be in [1, 9], got {self.compression_level}"
                    )


@dataclass
class GeometryStats:
    """Statistics about geometries in the dataset."""

    geometry_type: str
    geometry_types: List[str]
    bbox: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    crs: Optional[str]
    total_features: int
    valid_geometries: int
    invalid_geometries: int
    empty_geometries: int


@dataclass
class GeoParquetResult:
    """Result from GeoParquet conversion operation."""

    output_path: Path
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    geometry_stats: GeometryStats
    row_groups: int
    compression: Optional[str]
    geoparquet_version: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "output_path": str(self.output_path),
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "geometry_stats": {
                "geometry_type": self.geometry_stats.geometry_type,
                "geometry_types": self.geometry_stats.geometry_types,
                "bbox": self.geometry_stats.bbox,
                "crs": self.geometry_stats.crs,
                "total_features": self.geometry_stats.total_features,
            },
            "row_groups": self.row_groups,
            "compression": self.compression,
            "geoparquet_version": self.geoparquet_version,
            "metadata": self.metadata,
        }


class GeoParquetConverter:
    """
    GeoParquet converter for vector geospatial data.

    Converts vector data (GeoJSON, Shapefile, GeoDataFrame) to GeoParquet
    format with configurable compression, partitioning, and metadata.

    Example:
        converter = GeoParquetConverter(GeoParquetConfig(
            compression=ParquetCompression.ZSTD,
            row_group_size=100_000
        ))

        # From GeoDataFrame
        result = converter.convert_geodataframe(gdf, "output.parquet")

        # From GeoJSON file
        result = converter.convert_geojson("input.geojson", "output.parquet")

        # From Shapefile
        result = converter.convert_file("input.shp", "output.parquet")
    """

    def __init__(self, config: Optional[GeoParquetConfig] = None):
        """
        Initialize GeoParquet converter.

        Args:
            config: GeoParquet conversion configuration (uses defaults if None)
        """
        self.config = config or GeoParquetConfig()

    def convert_geodataframe(
        self,
        gdf: Any,
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> GeoParquetResult:
        """
        Convert a GeoDataFrame to GeoParquet format.

        Args:
            gdf: geopandas.GeoDataFrame to convert
            output_path: Path for output Parquet file
            overwrite: Whether to overwrite existing output

        Returns:
            GeoParquetResult with conversion details

        Raises:
            ImportError: If geopandas is not available
            FileExistsError: If output exists and overwrite=False
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for GeoDataFrame conversion")

        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting GeoDataFrame to GeoParquet: {output_path}")

        # Ensure geometry column name matches config
        if gdf.geometry.name != self.config.geometry_column:
            gdf = gdf.rename_geometry(self.config.geometry_column)

        # Calculate geometry statistics
        geom_stats = self._calculate_geometry_stats(gdf)

        # Build write options
        write_opts = {}
        if self.config.compression != ParquetCompression.NONE:
            write_opts["compression"] = self.config.compression.value
        if self.config.compression_level is not None:
            write_opts["compression_level"] = self.config.compression_level

        # Write to Parquet
        gdf.to_parquet(
            output_path,
            row_group_size=self.config.row_group_size,
            **write_opts,
        )

        # Get file stats
        file_size = output_path.stat().st_size

        # Count row groups
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(output_path)
            row_groups = pf.metadata.num_row_groups
        except Exception:
            row_groups = max(1, len(gdf) // self.config.row_group_size)

        logger.info(
            f"GeoParquet created: {output_path} "
            f"({geom_stats.total_features} features, {file_size / 1024 / 1024:.2f} MB)"
        )

        return GeoParquetResult(
            output_path=output_path,
            file_size_bytes=file_size,
            row_count=len(gdf),
            column_count=len(gdf.columns),
            columns=list(gdf.columns),
            geometry_stats=geom_stats,
            row_groups=row_groups,
            compression=self.config.compression.value if self.config.compression != ParquetCompression.NONE else None,
            geoparquet_version=self.config.schema_version,
            metadata={},
        )

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        layer: Optional[str] = None,
        overwrite: bool = False,
    ) -> GeoParquetResult:
        """
        Convert a vector file to GeoParquet format.

        Supports GeoJSON, Shapefile, GeoPackage, and other OGR-supported formats.

        Args:
            input_path: Path to input vector file
            output_path: Path for output Parquet file
            layer: Layer name (for multi-layer formats)
            overwrite: Whether to overwrite existing output

        Returns:
            GeoParquetResult with conversion details

        Raises:
            FileNotFoundError: If input file doesn't exist
            FileExistsError: If output exists and overwrite=False
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for file conversion")

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")

        logger.info(f"Converting {input_path} to GeoParquet")

        # Read input file
        read_opts = {}
        if layer:
            read_opts["layer"] = layer

        gdf = gpd.read_file(input_path, **read_opts)

        return self.convert_geodataframe(gdf, output_path, overwrite=True)

    def convert_geojson(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> GeoParquetResult:
        """
        Convert a GeoJSON file to GeoParquet format.

        Args:
            input_path: Path to input GeoJSON file
            output_path: Path for output Parquet file
            overwrite: Whether to overwrite existing output

        Returns:
            GeoParquetResult with conversion details
        """
        return self.convert_file(input_path, output_path, overwrite=overwrite)

    def convert_geojson_dict(
        self,
        geojson: Dict[str, Any],
        output_path: Union[str, Path],
        crs: str = "EPSG:4326",
        overwrite: bool = False,
    ) -> GeoParquetResult:
        """
        Convert a GeoJSON dictionary to GeoParquet format.

        Args:
            geojson: GeoJSON FeatureCollection dictionary
            output_path: Path for output Parquet file
            crs: Coordinate reference system
            overwrite: Whether to overwrite existing output

        Returns:
            GeoParquetResult with conversion details
        """
        try:
            import geopandas as gpd
            from shapely.geometry import shape
        except ImportError:
            raise ImportError("geopandas and shapely are required")

        output_path = Path(output_path)

        # Convert GeoJSON to GeoDataFrame
        features = geojson.get("features", [])
        if not features:
            raise ValueError("GeoJSON has no features")

        # Extract properties and geometries
        records = []
        geometries = []

        for feature in features:
            props = feature.get("properties", {}) or {}
            geom = feature.get("geometry")
            records.append(props)
            geometries.append(shape(geom) if geom else None)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)

        return self.convert_geodataframe(gdf, output_path, overwrite=overwrite)

    def convert_with_partitioning(
        self,
        gdf: Any,
        output_path: Union[str, Path],
        partition_columns: List[str],
        overwrite: bool = False,
    ) -> GeoParquetResult:
        """
        Convert a GeoDataFrame to partitioned GeoParquet format.

        Creates a directory structure with Parquet files partitioned by
        the specified columns.

        Args:
            gdf: geopandas.GeoDataFrame to convert
            output_path: Path for output directory
            partition_columns: Columns to partition by
            overwrite: Whether to overwrite existing output

        Returns:
            GeoParquetResult with conversion details
        """
        try:
            import geopandas as gpd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "geopandas and pyarrow are required for partitioned writing"
            )

        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output directory exists: {output_path}")

        logger.info(
            f"Converting GeoDataFrame to partitioned GeoParquet: {output_path}"
        )

        # Validate partition columns exist
        missing = set(partition_columns) - set(gdf.columns)
        if missing:
            raise ValueError(f"Partition columns not found: {missing}")

        # Ensure geometry column name matches config
        if gdf.geometry.name != self.config.geometry_column:
            gdf = gdf.rename_geometry(self.config.geometry_column)

        # Convert to PyArrow Table with GeoArrow extension
        from geopandas.io.arrow import _geopandas_to_arrow

        table = _geopandas_to_arrow(gdf)

        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=partition_columns,
            compression=self.config.compression.value if self.config.compression != ParquetCompression.NONE else None,
        )

        # Calculate geometry statistics
        geom_stats = self._calculate_geometry_stats(gdf)

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*.parquet")
        )

        return GeoParquetResult(
            output_path=output_path,
            file_size_bytes=total_size,
            row_count=len(gdf),
            column_count=len(gdf.columns),
            columns=list(gdf.columns),
            geometry_stats=geom_stats,
            row_groups=-1,  # Multiple files
            compression=self.config.compression.value if self.config.compression != ParquetCompression.NONE else None,
            geoparquet_version=self.config.schema_version,
            metadata={"partition_columns": partition_columns},
        )

    def validate_geoparquet(
        self, path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a file is a valid GeoParquet file.

        Args:
            path: Path to file to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        path = Path(path)
        issues = []

        if not path.exists():
            return False, [f"File not found: {path}"]

        try:
            import geopandas as gpd
            import pyarrow.parquet as pq
        except ImportError:
            return False, ["geopandas and pyarrow are required for validation"]

        # Check Parquet structure
        try:
            pf = pq.ParquetFile(path)
            schema = pf.schema_arrow
        except Exception as e:
            return False, [f"Invalid Parquet file: {e}"]

        # Check for geo metadata
        metadata = pf.metadata.metadata
        if metadata is None:
            issues.append("No file metadata present")
        elif b"geo" not in metadata:
            issues.append("No 'geo' metadata key (not GeoParquet)")
        else:
            try:
                geo_meta = json.loads(metadata[b"geo"])

                # Validate geo metadata structure
                if "version" not in geo_meta:
                    issues.append("Missing 'version' in geo metadata")

                if "columns" not in geo_meta:
                    issues.append("Missing 'columns' in geo metadata")
                else:
                    for col_name, col_meta in geo_meta["columns"].items():
                        if "encoding" not in col_meta:
                            issues.append(f"Column '{col_name}' missing 'encoding'")
                        if "geometry_types" not in col_meta:
                            issues.append(
                                f"Column '{col_name}' missing 'geometry_types'"
                            )

                if "primary_column" not in geo_meta:
                    issues.append("Missing 'primary_column' in geo metadata")

            except json.JSONDecodeError:
                issues.append("Invalid JSON in geo metadata")

        # Try to read the file
        try:
            gdf = gpd.read_parquet(path)
            if gdf.geometry.isna().all():
                issues.append("All geometries are null/empty")
        except Exception as e:
            issues.append(f"Failed to read as GeoDataFrame: {e}")

        return len(issues) == 0, issues

    def _calculate_geometry_stats(self, gdf: Any) -> GeometryStats:
        """Calculate statistics about geometries in a GeoDataFrame."""
        geom_col = gdf.geometry

        # Get geometry types
        geom_types = geom_col.geom_type.unique().tolist()
        primary_type = geom_types[0] if len(geom_types) == 1 else "Mixed"

        # Calculate bounds
        bounds = geom_col.total_bounds  # [minx, miny, maxx, maxy]

        # Count valid/invalid/empty geometries
        valid_count = geom_col.is_valid.sum()
        empty_count = geom_col.is_empty.sum()
        invalid_count = len(geom_col) - valid_count

        # Get CRS
        crs = str(gdf.crs) if gdf.crs else None

        return GeometryStats(
            geometry_type=primary_type,
            geometry_types=geom_types,
            bbox=(bounds[0], bounds[1], bounds[2], bounds[3]),
            crs=crs,
            total_features=len(gdf),
            valid_geometries=int(valid_count),
            invalid_geometries=int(invalid_count),
            empty_geometries=int(empty_count),
        )


def convert_to_geoparquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    compression: str = "snappy",
    row_group_size: int = 100_000,
    overwrite: bool = False,
) -> GeoParquetResult:
    """
    Convenience function to convert a vector file to GeoParquet format.

    Args:
        input_path: Path to input vector file
        output_path: Path for output Parquet file
        compression: Compression method
        row_group_size: Number of rows per row group
        overwrite: Whether to overwrite existing output

    Returns:
        GeoParquetResult with conversion details
    """
    config = GeoParquetConfig(
        compression=ParquetCompression(compression.lower()) if compression else ParquetCompression.NONE,
        row_group_size=row_group_size,
    )
    converter = GeoParquetConverter(config)
    return converter.convert_file(input_path, output_path, overwrite=overwrite)
