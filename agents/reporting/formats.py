"""
Format Handling for Reporting Agent.

Provides format conversion and optimization for output products:
- CRS transformation
- Resolution resampling
- Compression options
- Format validation
- Cloud-optimized outputs (COG, Cloud-native GeoJSON)

Integrates with core/data/ingestion/ for normalization operations.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ResamplingMethod(Enum):
    """Resampling methods for resolution changes."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBIC_SPLINE = "cubicspline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"


class CompressionMethod(Enum):
    """Compression methods for raster outputs."""

    NONE = "none"
    DEFLATE = "deflate"
    LZW = "lzw"
    ZSTD = "zstd"
    JPEG = "jpeg"
    WEBP = "webp"
    LERC = "lerc"
    LERC_DEFLATE = "lerc_deflate"
    LERC_ZSTD = "lerc_zstd"


class OutputCRS(Enum):
    """Common output coordinate reference systems."""

    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    UTM_AUTO = "utm_auto"  # Auto-detect UTM zone
    CUSTOM = "custom"


@dataclass
class FormatConfig:
    """
    Configuration for format conversion.

    Attributes:
        target_crs: Target coordinate reference system
        target_resolution_m: Target resolution in meters
        resampling: Resampling method
        compression: Compression method
        compression_level: Compression level (0-9 for deflate/lzw)
        predictor: Compression predictor (1=none, 2=horizontal, 3=floating-point)
        tiled: Use tiled output
        blocksize: Tile block size
        bigtiff: Enable BigTIFF
        cloud_optimized: Optimize for cloud access
        overview_factors: Overview pyramid factors
    """

    target_crs: Optional[str] = None
    target_resolution_m: Optional[float] = None
    resampling: ResamplingMethod = ResamplingMethod.BILINEAR
    compression: CompressionMethod = CompressionMethod.DEFLATE
    compression_level: int = 6
    predictor: int = 2
    tiled: bool = True
    blocksize: int = 512
    bigtiff: bool = False
    cloud_optimized: bool = True
    overview_factors: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_crs": self.target_crs,
            "target_resolution_m": self.target_resolution_m,
            "resampling": self.resampling.value,
            "compression": self.compression.value,
            "compression_level": self.compression_level,
            "predictor": self.predictor,
            "tiled": self.tiled,
            "blocksize": self.blocksize,
            "bigtiff": self.bigtiff,
            "cloud_optimized": self.cloud_optimized,
            "overview_factors": self.overview_factors,
        }


@dataclass
class ConversionResult:
    """Result of a format conversion operation."""

    success: bool
    input_crs: Optional[str]
    output_crs: Optional[str]
    input_resolution_m: Optional[float]
    output_resolution_m: Optional[float]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    input_bounds: Optional[Tuple[float, float, float, float]]
    output_bounds: Optional[Tuple[float, float, float, float]]
    compression_applied: str
    cloud_optimized: bool
    file_size_bytes: int
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "input_crs": self.input_crs,
            "output_crs": self.output_crs,
            "input_resolution_m": self.input_resolution_m,
            "output_resolution_m": self.output_resolution_m,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_bounds": self.input_bounds,
            "output_bounds": self.output_bounds,
            "compression_applied": self.compression_applied,
            "cloud_optimized": self.cloud_optimized,
            "file_size_bytes": self.file_size_bytes,
            "error_message": self.error_message,
        }


@dataclass
class ValidationResult:
    """Result of format validation."""

    valid: bool
    format: str
    issues: List[str]
    warnings: List[str]
    file_size_bytes: int
    crs: Optional[str]
    bounds: Optional[Tuple[float, float, float, float]]
    dimensions: Optional[Tuple[int, int]]
    band_count: Optional[int]
    dtype: Optional[str]
    is_cog: bool = False
    overview_levels: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "format": self.format,
            "issues": self.issues,
            "warnings": self.warnings,
            "file_size_bytes": self.file_size_bytes,
            "crs": self.crs,
            "bounds": self.bounds,
            "dimensions": self.dimensions,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "is_cog": self.is_cog,
            "overview_levels": self.overview_levels,
        }


class FormatConverter:
    """
    Format converter for output products.

    Handles CRS transformation, resolution resampling, compression,
    and cloud optimization of output products.

    Example:
        converter = FormatConverter()

        # Convert to web mercator with compression
        result = converter.convert_raster(
            input_path=Path("input.tif"),
            output_path=Path("output.tif"),
            config=FormatConfig(
                target_crs="EPSG:3857",
                target_resolution_m=30,
                compression=CompressionMethod.DEFLATE,
                cloud_optimized=True,
            ),
        )

        # Validate output
        validation = converter.validate_format(Path("output.tif"))
    """

    def __init__(self, default_config: Optional[FormatConfig] = None):
        """
        Initialize FormatConverter.

        Args:
            default_config: Default conversion configuration
        """
        self.default_config = default_config or FormatConfig()

    def convert_raster(
        self,
        input_path: Path,
        output_path: Path,
        config: Optional[FormatConfig] = None,
    ) -> ConversionResult:
        """
        Convert raster to specified format and CRS.

        Args:
            input_path: Input raster path
            output_path: Output raster path
            config: Conversion configuration

        Returns:
            ConversionResult with conversion details
        """
        config = config or self.default_config
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            return self._convert_with_rasterio(input_path, output_path, config)
        except ImportError:
            return self._convert_fallback(input_path, output_path, config)
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return ConversionResult(
                success=False,
                input_crs=None,
                output_crs=config.target_crs,
                input_resolution_m=None,
                output_resolution_m=config.target_resolution_m,
                input_size=(0, 0),
                output_size=(0, 0),
                input_bounds=None,
                output_bounds=None,
                compression_applied=config.compression.value,
                cloud_optimized=config.cloud_optimized,
                file_size_bytes=0,
                error_message=str(e),
            )

    def _convert_with_rasterio(
        self,
        input_path: Path,
        output_path: Path,
        config: FormatConfig,
    ) -> ConversionResult:
        """Convert using rasterio."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        with rasterio.open(input_path) as src:
            input_crs = str(src.crs) if src.crs else None
            input_bounds = tuple(src.bounds) if src.bounds else None
            input_size = (src.width, src.height)
            input_resolution = self._get_resolution(src)

            # Determine target CRS
            target_crs = config.target_crs
            if target_crs == "utm_auto":
                target_crs = self._get_utm_zone(input_bounds)
            elif not target_crs:
                target_crs = input_crs

            # Check if reprojection needed
            needs_reproject = (
                target_crs and input_crs and target_crs != input_crs
            ) or config.target_resolution_m

            if needs_reproject:
                # Calculate transform
                transform, width, height = calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=config.target_resolution_m,
                )

                # Build output profile
                profile = src.profile.copy()
                profile.update(
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                )
            else:
                profile = src.profile.copy()
                width, height = src.width, src.height
                transform = src.transform

            # Apply format options
            profile.update(self._get_format_options(config))

            # Write output
            resampling = getattr(Resampling, config.resampling.value.upper(), Resampling.BILINEAR)

            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    if needs_reproject:
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resampling,
                        )
                    else:
                        dst.write(src.read(band_idx), band_idx)

                # Build overviews for COG
                if config.cloud_optimized and config.overview_factors:
                    overview_resampling = Resampling.AVERAGE
                    dst.build_overviews(config.overview_factors, overview_resampling)
                    dst.update_tags(ns="rio_overview", resampling=overview_resampling.name)

        # Calculate output resolution and bounds
        with rasterio.open(output_path) as dst:
            output_bounds = tuple(dst.bounds)
            output_resolution = self._get_resolution(dst)

        return ConversionResult(
            success=True,
            input_crs=input_crs,
            output_crs=target_crs,
            input_resolution_m=input_resolution,
            output_resolution_m=output_resolution,
            input_size=input_size,
            output_size=(width, height),
            input_bounds=input_bounds,
            output_bounds=output_bounds,
            compression_applied=config.compression.value,
            cloud_optimized=config.cloud_optimized,
            file_size_bytes=output_path.stat().st_size,
        )

    def _convert_fallback(
        self,
        input_path: Path,
        output_path: Path,
        config: FormatConfig,
    ) -> ConversionResult:
        """Fallback conversion without rasterio."""
        import shutil

        # Simple copy with warning
        logger.warning("rasterio not available, copying without conversion")
        shutil.copy2(input_path, output_path)

        return ConversionResult(
            success=True,
            input_crs=None,
            output_crs=None,
            input_resolution_m=None,
            output_resolution_m=None,
            input_size=(0, 0),
            output_size=(0, 0),
            input_bounds=None,
            output_bounds=None,
            compression_applied="none",
            cloud_optimized=False,
            file_size_bytes=output_path.stat().st_size,
        )

    def _get_format_options(self, config: FormatConfig) -> Dict[str, Any]:
        """Get rasterio/GDAL format options."""
        options = {
            "driver": "GTiff",
            "tiled": config.tiled,
        }

        if config.tiled:
            options["blockxsize"] = config.blocksize
            options["blockysize"] = config.blocksize

        if config.compression != CompressionMethod.NONE:
            options["compress"] = config.compression.value.upper()

            if config.compression in (
                CompressionMethod.DEFLATE,
                CompressionMethod.LZW,
                CompressionMethod.ZSTD,
            ):
                options["predictor"] = config.predictor
                options["zlevel"] = config.compression_level

            if config.compression in (CompressionMethod.JPEG, CompressionMethod.WEBP):
                options["quality"] = 85

        if config.bigtiff:
            options["bigtiff"] = "yes"

        # COG layout
        if config.cloud_optimized:
            options["interleave"] = "band"

        return options

    def _get_resolution(self, dataset: Any) -> Optional[float]:
        """Get resolution from dataset."""
        try:
            transform = dataset.transform
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            return (res_x + res_y) / 2
        except Exception:
            return None

    def _get_utm_zone(
        self,
        bounds: Optional[Tuple[float, float, float, float]],
    ) -> str:
        """Get appropriate UTM zone for bounds."""
        if not bounds:
            return "EPSG:32632"  # Default

        # Calculate center
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

        # Calculate zone
        zone = int((center_lon + 180) / 6) + 1
        hemisphere = "N" if center_lat >= 0 else "S"

        if hemisphere == "N":
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        return f"EPSG:{epsg}"

    def convert_array(
        self,
        data: np.ndarray,
        source_crs: str,
        source_transform: Any,
        config: FormatConfig,
    ) -> Tuple[np.ndarray, str, Any]:
        """
        Convert array to different CRS/resolution.

        Args:
            data: Input array (2D or 3D)
            source_crs: Source CRS
            source_transform: Source affine transform
            config: Conversion configuration

        Returns:
            Tuple of (converted_data, target_crs, target_transform)
        """
        try:
            return self._convert_array_rasterio(
                data, source_crs, source_transform, config
            )
        except ImportError:
            return self._convert_array_fallback(
                data, source_crs, source_transform, config
            )

    def _convert_array_rasterio(
        self,
        data: np.ndarray,
        source_crs: str,
        source_transform: Any,
        config: FormatConfig,
    ) -> Tuple[np.ndarray, str, Any]:
        """Convert array using rasterio."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        # Handle shape
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        bands, src_height, src_width = data.shape

        # Determine target CRS
        target_crs = config.target_crs or source_crs
        if target_crs == "utm_auto":
            bounds = rasterio.transform.array_bounds(src_height, src_width, source_transform)
            target_crs = self._get_utm_zone(bounds)

        # Calculate transform
        bounds = rasterio.transform.array_bounds(src_height, src_width, source_transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            source_crs,
            target_crs,
            src_width,
            src_height,
            *bounds,
            resolution=config.target_resolution_m,
        )

        # Allocate output
        output = np.zeros((bands, dst_height, dst_width), dtype=data.dtype)

        # Reproject each band
        resampling = getattr(Resampling, config.resampling.value.upper(), Resampling.BILINEAR)

        for band_idx in range(bands):
            reproject(
                source=data[band_idx],
                destination=output[band_idx],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=resampling,
            )

        # Squeeze if single band
        if output.shape[0] == 1:
            output = output[0]

        return output, target_crs, dst_transform

    def _convert_array_fallback(
        self,
        data: np.ndarray,
        source_crs: str,
        source_transform: Any,
        config: FormatConfig,
    ) -> Tuple[np.ndarray, str, Any]:
        """Fallback array conversion without rasterio."""
        # Simple resampling without CRS conversion
        if config.target_resolution_m:
            # Calculate scale factor
            current_res = self._estimate_resolution(source_transform)
            if current_res and current_res > 0:
                scale = current_res / config.target_resolution_m
                data = self._resample_array(data, scale, config.resampling)

        return data, source_crs, source_transform

    def _estimate_resolution(self, transform: Any) -> Optional[float]:
        """Estimate resolution from transform."""
        try:
            if hasattr(transform, "a"):
                return abs(transform.a)
            elif isinstance(transform, tuple) and len(transform) >= 6:
                return abs(transform[0])
        except Exception:
            pass
        return None

    def _resample_array(
        self,
        data: np.ndarray,
        scale: float,
        method: ResamplingMethod,
    ) -> np.ndarray:
        """Resample array by scale factor."""
        from scipy import ndimage

        # Calculate zoom factors
        if data.ndim == 3:
            zoom = (1, scale, scale)
        else:
            zoom = (scale, scale)

        # Map resampling method to order
        order_map = {
            ResamplingMethod.NEAREST: 0,
            ResamplingMethod.BILINEAR: 1,
            ResamplingMethod.CUBIC: 3,
            ResamplingMethod.LANCZOS: 4,
        }
        order = order_map.get(method, 1)

        return ndimage.zoom(data, zoom, order=order)

    def resample(
        self,
        data: np.ndarray,
        target_size: Tuple[int, int],
        method: ResamplingMethod = ResamplingMethod.BILINEAR,
    ) -> np.ndarray:
        """
        Resample array to target size.

        Args:
            data: Input array
            target_size: Target (width, height)
            method: Resampling method

        Returns:
            Resampled array
        """
        target_width, target_height = target_size

        # Handle shape
        if data.ndim == 3:
            bands, src_height, src_width = data.shape
            zoom = (1, target_height / src_height, target_width / src_width)
        else:
            src_height, src_width = data.shape
            zoom = (target_height / src_height, target_width / src_width)

        # Map method to scipy order
        order_map = {
            ResamplingMethod.NEAREST: 0,
            ResamplingMethod.BILINEAR: 1,
            ResamplingMethod.CUBIC: 3,
            ResamplingMethod.LANCZOS: 4,
        }
        order = order_map.get(method, 1)

        from scipy import ndimage
        return ndimage.zoom(data, zoom, order=order)

    def validate_format(
        self,
        path: Path,
        expected_format: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate file format.

        Args:
            path: File path
            expected_format: Expected format (geotiff, geojson, etc.)

        Returns:
            ValidationResult with validation details
        """
        if not path.exists():
            return ValidationResult(
                valid=False,
                format="unknown",
                issues=["File does not exist"],
                warnings=[],
                file_size_bytes=0,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )

        # Detect format from extension
        ext = path.suffix.lower()
        if ext in (".tif", ".tiff"):
            return self._validate_geotiff(path)
        elif ext in (".geojson", ".json"):
            return self._validate_geojson(path)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._validate_image(path)
        else:
            return ValidationResult(
                valid=True,
                format="unknown",
                issues=[],
                warnings=[f"Unknown format: {ext}"],
                file_size_bytes=path.stat().st_size,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )

    def _validate_geotiff(self, path: Path) -> ValidationResult:
        """Validate GeoTIFF format."""
        issues = []
        warnings = []

        try:
            import rasterio

            with rasterio.open(path) as src:
                profile = src.profile
                crs = str(src.crs) if src.crs else None
                bounds = tuple(src.bounds) if src.bounds else None
                dimensions = (src.width, src.height)
                band_count = src.count
                dtype = str(src.dtypes[0])

                # Check for tiling (COG requirement)
                is_tiled = profile.get("tiled", False)
                if not is_tiled:
                    warnings.append("File is not internally tiled")

                # Check for overviews
                overview_levels = []
                if src.overviews(1):
                    overview_levels = list(src.overviews(1))
                else:
                    warnings.append("No overviews present")

                # Check interleave
                interleave = profile.get("interleave")
                if interleave != "band":
                    warnings.append(f"Interleave is '{interleave}', expected 'band' for COG")

                # Check CRS
                if not crs:
                    issues.append("No CRS defined")

                # Determine if valid COG
                is_cog = is_tiled and len(overview_levels) > 0 and interleave == "band"

            return ValidationResult(
                valid=len(issues) == 0,
                format="geotiff",
                issues=issues,
                warnings=warnings,
                file_size_bytes=path.stat().st_size,
                crs=crs,
                bounds=bounds,
                dimensions=dimensions,
                band_count=band_count,
                dtype=dtype,
                is_cog=is_cog,
                overview_levels=overview_levels,
            )

        except ImportError:
            return ValidationResult(
                valid=True,
                format="geotiff",
                issues=[],
                warnings=["rasterio not available for full validation"],
                file_size_bytes=path.stat().st_size,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                format="geotiff",
                issues=[f"Error reading file: {e}"],
                warnings=[],
                file_size_bytes=path.stat().st_size if path.exists() else 0,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )

    def _validate_geojson(self, path: Path) -> ValidationResult:
        """Validate GeoJSON format."""
        issues = []
        warnings = []

        try:
            with open(path) as f:
                data = json.load(f)

            # Check for valid structure
            geojson_type = data.get("type")
            if geojson_type not in ("Feature", "FeatureCollection", "GeometryCollection"):
                issues.append(f"Invalid GeoJSON type: {geojson_type}")

            # Get features
            if geojson_type == "FeatureCollection":
                features = data.get("features", [])
            elif geojson_type == "Feature":
                features = [data]
            else:
                features = []

            # Calculate bbox
            bounds = None
            if "bbox" in data:
                bounds = tuple(data["bbox"])

            # Check CRS
            crs = "EPSG:4326"  # Default
            if "crs" in data:
                crs_props = data["crs"].get("properties", {})
                crs = crs_props.get("name", crs)
                warnings.append("Non-standard CRS in GeoJSON (deprecated)")

            # Validate features
            for i, feature in enumerate(features[:10]):  # Sample
                if not feature.get("geometry"):
                    warnings.append(f"Feature {i} has no geometry")
                if not feature.get("properties"):
                    warnings.append(f"Feature {i} has no properties")

            return ValidationResult(
                valid=len(issues) == 0,
                format="geojson",
                issues=issues,
                warnings=warnings,
                file_size_bytes=path.stat().st_size,
                crs=crs,
                bounds=bounds,
                dimensions=(len(features), 0) if features else None,
                band_count=None,
                dtype=None,
            )

        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                format="geojson",
                issues=[f"Invalid JSON: {e}"],
                warnings=[],
                file_size_bytes=path.stat().st_size,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                format="geojson",
                issues=[f"Error reading file: {e}"],
                warnings=[],
                file_size_bytes=path.stat().st_size if path.exists() else 0,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )

    def _validate_image(self, path: Path) -> ValidationResult:
        """Validate image format."""
        try:
            from PIL import Image

            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode

            return ValidationResult(
                valid=True,
                format="image",
                issues=[],
                warnings=[],
                file_size_bytes=path.stat().st_size,
                crs=None,
                bounds=None,
                dimensions=(width, height),
                band_count=len(mode) if mode else None,
                dtype=mode,
            )
        except ImportError:
            return ValidationResult(
                valid=True,
                format="image",
                issues=[],
                warnings=["PIL not available for validation"],
                file_size_bytes=path.stat().st_size,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                format="image",
                issues=[f"Error reading image: {e}"],
                warnings=[],
                file_size_bytes=path.stat().st_size if path.exists() else 0,
                crs=None,
                bounds=None,
                dimensions=None,
                band_count=None,
                dtype=None,
            )

    def optimize_for_cloud(
        self,
        input_path: Path,
        output_path: Path,
        config: Optional[FormatConfig] = None,
    ) -> ConversionResult:
        """
        Optimize file for cloud access.

        Args:
            input_path: Input file path
            output_path: Output file path
            config: Conversion configuration

        Returns:
            ConversionResult
        """
        config = config or FormatConfig(cloud_optimized=True)
        config.cloud_optimized = True

        ext = input_path.suffix.lower()
        if ext in (".tif", ".tiff"):
            return self.convert_raster(input_path, output_path, config)
        elif ext in (".geojson", ".json"):
            return self._optimize_geojson(input_path, output_path)
        else:
            # Copy as-is
            import shutil
            shutil.copy2(input_path, output_path)
            return ConversionResult(
                success=True,
                input_crs=None,
                output_crs=None,
                input_resolution_m=None,
                output_resolution_m=None,
                input_size=(0, 0),
                output_size=(0, 0),
                input_bounds=None,
                output_bounds=None,
                compression_applied="none",
                cloud_optimized=False,
                file_size_bytes=output_path.stat().st_size,
            )

    def _optimize_geojson(
        self,
        input_path: Path,
        output_path: Path,
    ) -> ConversionResult:
        """Optimize GeoJSON for cloud access."""
        with open(input_path) as f:
            data = json.load(f)

        # Add/update bbox if not present
        if "bbox" not in data:
            features = data.get("features", [])
            if features:
                bbox = self._calculate_bbox(features)
                if bbox:
                    data["bbox"] = bbox

        # Write with minimal whitespace for smaller size
        with open(output_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        return ConversionResult(
            success=True,
            input_crs="EPSG:4326",
            output_crs="EPSG:4326",
            input_resolution_m=None,
            output_resolution_m=None,
            input_size=(0, 0),
            output_size=(0, 0),
            input_bounds=tuple(data.get("bbox", [])) if data.get("bbox") else None,
            output_bounds=tuple(data.get("bbox", [])) if data.get("bbox") else None,
            compression_applied="minified",
            cloud_optimized=True,
            file_size_bytes=output_path.stat().st_size,
        )

    def _calculate_bbox(
        self,
        features: List[Dict[str, Any]],
    ) -> Optional[List[float]]:
        """Calculate bounding box from features."""
        all_coords = []

        def extract_coords(obj):
            if isinstance(obj, (list, tuple)):
                if obj and isinstance(obj[0], (int, float)):
                    all_coords.append(obj[:2])
                else:
                    for item in obj:
                        extract_coords(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_coords(value)

        for feature in features:
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])
            extract_coords(coords)

        if not all_coords:
            return None

        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        return [min(xs), min(ys), max(xs), max(ys)]

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input/output formats."""
        return {
            "raster_input": [".tif", ".tiff", ".vrt", ".nc", ".zarr"],
            "raster_output": [".tif", ".tiff"],
            "vector_input": [".geojson", ".json", ".shp", ".gpkg"],
            "vector_output": [".geojson", ".json"],
            "image_output": [".png", ".jpg", ".jpeg"],
        }
