"""
Cloud-Optimized GeoTIFF (COG) Converter.

Converts raster data to Cloud-Optimized GeoTIFF format with configurable
compression, tiling, and overview generation.

COG Format Benefits:
- Efficient cloud/network access via HTTP range requests
- Internal tiling for partial reads
- Embedded overviews for fast zoom-out rendering
- Wide software compatibility
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Compression(Enum):
    """Supported compression methods for COG."""

    NONE = "none"
    DEFLATE = "deflate"
    LZW = "lzw"
    ZSTD = "zstd"
    JPEG = "jpeg"
    WEBP = "webp"
    LERC = "lerc"
    LERC_DEFLATE = "lerc_deflate"
    LERC_ZSTD = "lerc_zstd"


class Predictor(Enum):
    """Compression predictor options."""

    NONE = 1
    HORIZONTAL = 2
    FLOATINGPOINT = 3


class ResamplingMethod(Enum):
    """Resampling methods for overview generation."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBICSPLINE = "cubicspline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    RMS = "rms"
    MODE = "mode"


@dataclass
class COGConfig:
    """
    Configuration for COG conversion.

    Attributes:
        blocksize: Internal tile size (default 512)
        compression: Compression method
        predictor: Compression predictor for better ratios
        quality: Quality for lossy compression (1-100)
        overview_factors: List of overview reduction factors
        overview_resampling: Resampling method for overviews
        nodata: NoData value to set
        dtype: Output data type (auto if None)
        bigtiff: Enable BigTIFF for files > 4GB
        copy_metadata: Copy source metadata to output
        sparse: Enable sparse file support
    """

    blocksize: int = 512
    compression: Compression = Compression.DEFLATE
    predictor: Predictor = Predictor.HORIZONTAL
    quality: int = 75
    overview_factors: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    overview_resampling: ResamplingMethod = ResamplingMethod.AVERAGE
    nodata: Optional[float] = None
    dtype: Optional[str] = None
    bigtiff: bool = False
    copy_metadata: bool = True
    sparse: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.blocksize not in [128, 256, 512, 1024, 2048]:
            raise ValueError(
                f"blocksize must be 128, 256, 512, 1024, or 2048, got {self.blocksize}"
            )
        if not 1 <= self.quality <= 100:
            raise ValueError(f"quality must be in [1, 100], got {self.quality}")
        if self.overview_factors:
            for factor in self.overview_factors:
                if factor < 2:
                    raise ValueError(f"overview_factors must be >= 2, got {factor}")

    def to_creation_options(self) -> Dict[str, Any]:
        """Convert to GDAL/rasterio creation options."""
        options = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": self.blocksize,
            "blockysize": self.blocksize,
            "compress": self.compression.value.upper(),
            "interleave": "band",
        }

        # Add predictor for applicable compression types
        if self.compression in [
            Compression.DEFLATE,
            Compression.LZW,
            Compression.ZSTD,
        ]:
            options["predictor"] = self.predictor.value

        # Add quality for lossy compression
        if self.compression in [Compression.JPEG, Compression.WEBP]:
            options["quality"] = self.quality

        # BigTIFF support
        if self.bigtiff:
            options["bigtiff"] = "yes"

        return options


@dataclass
class COGResult:
    """Result from COG conversion operation."""

    output_path: Path
    file_size_bytes: int
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    bounds: Optional[Tuple[float, float, float, float]]
    overview_levels: List[int]
    compression_ratio: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "output_path": str(self.output_path),
            "file_size_bytes": self.file_size_bytes,
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": self.bounds,
            "overview_levels": self.overview_levels,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
        }


class COGConverter:
    """
    Cloud-Optimized GeoTIFF converter.

    Converts raster data to COG format with configurable compression,
    tiling, and overview generation. Supports both file-based and
    array-based inputs.

    Example:
        converter = COGConverter(COGConfig(compression=Compression.DEFLATE))
        result = converter.convert_file("input.tif", "output.tif")

        # Or from numpy array
        result = converter.convert_array(
            data=array,
            output_path="output.tif",
            transform=transform,
            crs="EPSG:4326"
        )
    """

    def __init__(self, config: Optional[COGConfig] = None):
        """
        Initialize COG converter.

        Args:
            config: COG conversion configuration (uses defaults if None)
        """
        self.config = config or COGConfig()

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> COGResult:
        """
        Convert a raster file to COG format.

        Args:
            input_path: Path to input raster file
            output_path: Path for output COG file
            overwrite: Whether to overwrite existing output

        Returns:
            COGResult with conversion details

        Raises:
            FileNotFoundError: If input file doesn't exist
            FileExistsError: If output exists and overwrite=False
            ValueError: If input is not a valid raster
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError:
            raise ImportError("rasterio is required for COG conversion")

        logger.info(f"Converting {input_path} to COG format")

        # Read source data
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            data = src.read()
            source_size = input_path.stat().st_size
            src_bounds = src.bounds
            src_crs = str(src.crs) if src.crs else None
            src_meta = dict(src.tags())

        # Update profile for COG
        profile.update(self.config.to_creation_options())

        if self.config.dtype:
            profile["dtype"] = self.config.dtype
            data = data.astype(self.config.dtype)

        if self.config.nodata is not None:
            profile["nodata"] = self.config.nodata

        # Write COG with internal overviews
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

            # Copy metadata if requested
            if self.config.copy_metadata:
                dst.update_tags(**src_meta)

            # Build overviews
            if self.config.overview_factors:
                resampling = getattr(
                    Resampling, self.config.overview_resampling.value.upper()
                )
                dst.build_overviews(self.config.overview_factors, resampling)
                dst.update_tags(ns="rio_overview", resampling=resampling.name)

        # Calculate compression ratio
        output_size = output_path.stat().st_size
        compression_ratio = source_size / output_size if output_size > 0 else 0.0

        logger.info(
            f"COG created: {output_path} "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )

        return COGResult(
            output_path=output_path,
            file_size_bytes=output_size,
            width=profile["width"],
            height=profile["height"],
            band_count=profile["count"],
            dtype=str(profile["dtype"]),
            crs=src_crs,
            bounds=(src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top)
            if src_bounds
            else None,
            overview_levels=self.config.overview_factors,
            compression_ratio=compression_ratio,
            metadata=src_meta,
        )

    def convert_array(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        transform: Any,
        crs: str,
        nodata: Optional[float] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ) -> COGResult:
        """
        Convert a numpy array to COG format.

        Args:
            data: Input array (bands, height, width) or (height, width)
            output_path: Path for output COG file
            transform: Affine transform for georeferencing
            crs: Coordinate reference system (e.g., "EPSG:4326")
            nodata: NoData value (overrides config)
            metadata: Additional metadata tags
            overwrite: Whether to overwrite existing output

        Returns:
            COGResult with conversion details

        Raises:
            ValueError: If array has invalid shape
            FileExistsError: If output exists and overwrite=False
        """
        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.transform import Affine
        except ImportError:
            raise ImportError("rasterio is required for COG conversion")

        # Handle 2D arrays (single band)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        band_count, height, width = data.shape
        dtype = data.dtype

        # Use provided nodata or config nodata
        nodata_val = nodata if nodata is not None else self.config.nodata

        # Build profile
        profile = {
            "width": width,
            "height": height,
            "count": band_count,
            "dtype": str(dtype) if self.config.dtype is None else self.config.dtype,
            "crs": crs,
            "transform": transform,
        }

        if nodata_val is not None:
            profile["nodata"] = nodata_val

        profile.update(self.config.to_creation_options())

        # Convert dtype if specified
        if self.config.dtype and self.config.dtype != str(dtype):
            data = data.astype(self.config.dtype)

        # Calculate input size for compression ratio
        input_size = data.nbytes

        logger.info(f"Creating COG from array: {data.shape}")

        # Write COG
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

            # Add metadata
            if metadata:
                dst.update_tags(**metadata)

            # Build overviews
            if self.config.overview_factors:
                resampling = getattr(
                    Resampling, self.config.overview_resampling.value.upper()
                )
                dst.build_overviews(self.config.overview_factors, resampling)
                dst.update_tags(ns="rio_overview", resampling=resampling.name)

        # Calculate bounds from transform
        bounds = rasterio.transform.array_bounds(height, width, transform)

        output_size = output_path.stat().st_size
        compression_ratio = input_size / output_size if output_size > 0 else 0.0

        logger.info(
            f"COG created: {output_path} "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )

        return COGResult(
            output_path=output_path,
            file_size_bytes=output_size,
            width=width,
            height=height,
            band_count=band_count,
            dtype=str(profile["dtype"]),
            crs=crs,
            bounds=bounds,
            overview_levels=self.config.overview_factors,
            compression_ratio=compression_ratio,
            metadata=metadata or {},
        )

    def validate_cog(self, path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """
        Validate that a file is a valid Cloud-Optimized GeoTIFF.

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
            import rasterio
        except ImportError:
            return False, ["rasterio is required for COG validation"]

        with rasterio.open(path) as src:
            # Check for tiling
            if not src.profile.get("tiled"):
                issues.append("File is not internally tiled")

            # Check block size
            block_shapes = src.block_shapes
            if block_shapes:
                for band_idx, (block_height, block_width) in enumerate(block_shapes):
                    if block_width != block_height:
                        issues.append(
                            f"Band {band_idx + 1}: Non-square tiles ({block_width}x{block_height})"
                        )

            # Check for overviews
            if not src.overviews(1):
                issues.append("No overviews present")

            # Check interleave (should be band for COG)
            interleave = src.profile.get("interleave")
            if interleave != "band":
                issues.append(f"Interleave is '{interleave}', expected 'band'")

            # Check if file can be accessed efficiently
            # COGs should have the IFD at the start
            # This is a simplified check - full validation would use rio-cogeo
            if src.profile.get("driver") != "GTiff":
                issues.append(f"Driver is '{src.profile.get('driver')}', expected 'GTiff'")

        return len(issues) == 0, issues


def convert_to_cog(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    compression: str = "deflate",
    blocksize: int = 512,
    overview_factors: Optional[List[int]] = None,
    overwrite: bool = False,
) -> COGResult:
    """
    Convenience function to convert a file to COG format.

    Args:
        input_path: Path to input raster file
        output_path: Path for output COG file
        compression: Compression method (deflate, lzw, zstd, etc.)
        blocksize: Internal tile size
        overview_factors: List of overview reduction factors
        overwrite: Whether to overwrite existing output

    Returns:
        COGResult with conversion details
    """
    config = COGConfig(
        compression=Compression(compression.lower()),
        blocksize=blocksize,
        overview_factors=overview_factors or [2, 4, 8, 16, 32],
    )
    converter = COGConverter(config)
    return converter.convert_file(input_path, output_path, overwrite=overwrite)
