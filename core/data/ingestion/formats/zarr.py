"""
Zarr Array Converter.

Converts multidimensional raster data to Zarr format for efficient
cloud-native storage and access. Zarr is ideal for time-series,
multi-sensor fusion, and large-scale analysis.

Zarr Format Benefits:
- Chunked storage for parallel I/O
- Hierarchical group structure
- Flexible compression per-array
- Excellent xarray integration
- Cloud-native (S3, GCS compatible)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ZarrCompression(Enum):
    """Supported compression methods for Zarr."""

    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"
    BLOSC_LZ4 = "blosc_lz4"
    BLOSC_ZSTD = "blosc_zstd"
    GZIP = "gzip"
    ZLIB = "zlib"


class ZarrStorageFormat(Enum):
    """Zarr storage format options."""

    DIRECTORY = "directory"  # Local filesystem directory
    ZIP = "zip"  # Single ZIP file
    MEMORY = "memory"  # In-memory (for testing)


@dataclass
class ChunkConfig:
    """
    Chunk configuration for Zarr arrays.

    Attributes:
        time: Chunk size for time dimension
        y: Chunk size for y (height) dimension
        x: Chunk size for x (width) dimension
        band: Chunk size for band dimension (if present)
        auto: Automatically determine chunk sizes
    """

    time: int = 1
    y: int = 512
    x: int = 512
    band: Optional[int] = None
    auto: bool = False

    def to_tuple(self, dims: List[str]) -> Tuple[int, ...]:
        """Convert to chunk tuple for given dimensions."""
        chunk_map = {"time": self.time, "y": self.y, "x": self.x, "band": self.band}
        chunks = []
        for dim in dims:
            if dim in chunk_map and chunk_map[dim] is not None:
                chunks.append(chunk_map[dim])
            else:
                chunks.append(self.y if dim in ["lat", "latitude"] else self.x)
        return tuple(chunks)


@dataclass
class ZarrConfig:
    """
    Configuration for Zarr conversion.

    Attributes:
        chunks: Chunk configuration
        compression: Compression method
        compression_level: Compression level (1-22 for zstd)
        storage_format: Output storage format
        fill_value: Fill value for missing data
        consolidated: Create consolidated metadata
        write_empty_chunks: Write chunks that are all fill_value
        dimension_separator: Separator for dimension hierarchies
    """

    chunks: ChunkConfig = field(default_factory=ChunkConfig)
    compression: ZarrCompression = ZarrCompression.ZSTD
    compression_level: int = 3
    storage_format: ZarrStorageFormat = ZarrStorageFormat.DIRECTORY
    fill_value: Optional[float] = None
    consolidated: bool = True
    write_empty_chunks: bool = False
    dimension_separator: str = "/"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.compression == ZarrCompression.ZSTD:
            if not 1 <= self.compression_level <= 22:
                raise ValueError(
                    f"zstd compression_level must be in [1, 22], got {self.compression_level}"
                )
        elif self.compression == ZarrCompression.GZIP:
            if not 1 <= self.compression_level <= 9:
                raise ValueError(
                    f"gzip compression_level must be in [1, 9], got {self.compression_level}"
                )

    def get_compressor(self) -> Optional[Any]:
        """Get numcodecs compressor instance."""
        if self.compression == ZarrCompression.NONE:
            return None

        try:
            import numcodecs
        except ImportError:
            logger.warning("numcodecs not available, using no compression")
            return None

        if self.compression in [ZarrCompression.BLOSC_LZ4, ZarrCompression.BLOSC_ZSTD]:
            cname = "lz4" if self.compression == ZarrCompression.BLOSC_LZ4 else "zstd"
            return numcodecs.Blosc(cname=cname, clevel=self.compression_level)
        elif self.compression == ZarrCompression.ZSTD:
            return numcodecs.Zstd(level=self.compression_level)
        elif self.compression == ZarrCompression.LZ4:
            return numcodecs.LZ4()
        elif self.compression == ZarrCompression.GZIP:
            return numcodecs.GZip(level=self.compression_level)
        elif self.compression == ZarrCompression.ZLIB:
            return numcodecs.Zlib(level=self.compression_level)

        return None


@dataclass
class ZarrArrayInfo:
    """Information about a Zarr array."""

    name: str
    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: str
    dimensions: List[str]
    compression: Optional[str]
    fill_value: Optional[float]
    attrs: Dict[str, Any]


@dataclass
class ZarrResult:
    """Result from Zarr conversion operation."""

    output_path: Path
    store_size_bytes: int
    arrays: List[ZarrArrayInfo]
    dimensions: Dict[str, int]
    coords: Dict[str, List[Any]]
    global_attrs: Dict[str, Any]
    consolidated: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "output_path": str(self.output_path),
            "store_size_bytes": self.store_size_bytes,
            "arrays": [
                {
                    "name": a.name,
                    "shape": a.shape,
                    "chunks": a.chunks,
                    "dtype": a.dtype,
                    "dimensions": a.dimensions,
                }
                for a in self.arrays
            ],
            "dimensions": self.dimensions,
            "coords": {k: len(v) for k, v in self.coords.items()},
            "global_attrs": self.global_attrs,
            "consolidated": self.consolidated,
        }


class ZarrConverter:
    """
    Zarr array converter for multidimensional geospatial data.

    Converts raster stacks to Zarr format with configurable chunking,
    compression, and metadata. Optimized for time-series analysis
    and cloud-native workflows.

    Example:
        converter = ZarrConverter(ZarrConfig(
            chunks=ChunkConfig(time=1, y=512, x=512),
            compression=ZarrCompression.ZSTD
        ))

        # From xarray Dataset
        result = converter.convert_xarray(ds, "output.zarr")

        # From numpy arrays with coordinates
        result = converter.convert_arrays(
            arrays={"ndvi": ndvi_stack, "cloud_mask": mask_stack},
            output_path="output.zarr",
            coords={"time": timestamps, "y": y_coords, "x": x_coords}
        )
    """

    def __init__(self, config: Optional[ZarrConfig] = None):
        """
        Initialize Zarr converter.

        Args:
            config: Zarr conversion configuration (uses defaults if None)
        """
        self.config = config or ZarrConfig()

    def convert_xarray(
        self,
        dataset: Any,
        output_path: Union[str, Path],
        mode: str = "w",
    ) -> ZarrResult:
        """
        Convert an xarray Dataset to Zarr format.

        Args:
            dataset: xarray.Dataset to convert
            output_path: Path for output Zarr store
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            ZarrResult with conversion details

        Raises:
            ImportError: If xarray is not available
            ValueError: If dataset is not valid
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for xarray Dataset conversion")

        output_path = Path(output_path)
        logger.info(f"Converting xarray Dataset to Zarr: {output_path}")

        # Determine chunks
        chunks = {}
        for dim in dataset.dims:
            if dim in ["time", "t"]:
                chunks[dim] = self.config.chunks.time
            elif dim in ["y", "lat", "latitude"]:
                chunks[dim] = self.config.chunks.y
            elif dim in ["x", "lon", "longitude"]:
                chunks[dim] = self.config.chunks.x
            elif dim == "band" and self.config.chunks.band:
                chunks[dim] = self.config.chunks.band
            else:
                # Keep original chunk size or default
                chunks[dim] = min(dataset.dims[dim], self.config.chunks.y)

        # Apply chunking
        dataset = dataset.chunk(chunks)

        # Get encoding for each variable
        encoding = {}
        compressor = self.config.get_compressor()
        for var in dataset.data_vars:
            encoding[var] = {
                "compressor": compressor,
            }
            if self.config.fill_value is not None:
                encoding[var]["_FillValue"] = self.config.fill_value

        # Write to Zarr
        dataset.to_zarr(
            output_path,
            mode=mode,
            encoding=encoding,
            consolidated=self.config.consolidated,
        )

        # Gather result info
        arrays = []
        for var_name, var in dataset.data_vars.items():
            arrays.append(
                ZarrArrayInfo(
                    name=var_name,
                    shape=var.shape,
                    chunks=tuple(chunks.get(d, var.shape[i]) for i, d in enumerate(var.dims)),
                    dtype=str(var.dtype),
                    dimensions=list(var.dims),
                    compression=self.config.compression.value,
                    fill_value=self.config.fill_value,
                    attrs=dict(var.attrs),
                )
            )

        # Calculate store size
        store_size = self._calculate_store_size(output_path)

        return ZarrResult(
            output_path=output_path,
            store_size_bytes=store_size,
            arrays=arrays,
            dimensions=dict(dataset.dims),
            coords={k: list(v.values) if hasattr(v, "values") else [] for k, v in dataset.coords.items()},
            global_attrs=dict(dataset.attrs),
            consolidated=self.config.consolidated,
        )

    def convert_arrays(
        self,
        arrays: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        coords: Dict[str, List[Any]],
        dims: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        array_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ZarrResult:
        """
        Convert numpy arrays to Zarr format.

        Args:
            arrays: Dictionary mapping variable names to numpy arrays
            output_path: Path for output Zarr store
            coords: Dictionary mapping dimension names to coordinate values
            dims: Dimension names (auto-detected if None)
            attrs: Global attributes
            array_attrs: Per-array attributes

        Returns:
            ZarrResult with conversion details

        Raises:
            ValueError: If arrays have inconsistent shapes
        """
        output_path = Path(output_path)
        logger.info(f"Converting arrays to Zarr: {output_path}")

        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required for array conversion")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open/create store based on format
        if self.config.storage_format == ZarrStorageFormat.ZIP:
            store = zarr.ZipStore(output_path, mode="w")
        else:
            store = zarr.DirectoryStore(output_path)

        # Create root group
        root = zarr.group(store, overwrite=True)

        # Set global attributes
        if attrs:
            root.attrs.update(attrs)

        # Get compressor
        compressor = self.config.get_compressor()

        # Auto-detect dimensions if not provided
        if dims is None:
            # Assume common dimension ordering
            sample_array = list(arrays.values())[0]
            ndim = sample_array.ndim
            if ndim == 2:
                dims = ["y", "x"]
            elif ndim == 3:
                dims = ["time", "y", "x"]
            elif ndim == 4:
                dims = ["time", "band", "y", "x"]
            else:
                dims = [f"dim_{i}" for i in range(ndim)]

        # Write coordinate arrays
        for coord_name, coord_values in coords.items():
            coord_array = np.array(coord_values)
            root.create_dataset(
                coord_name,
                data=coord_array,
                chunks=False,  # Coordinates are small, no chunking
                compressor=None,
            )
            # Mark as coordinate
            root[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]

        # Write data arrays
        array_infos = []
        for var_name, data in arrays.items():
            # Determine chunks for this array
            chunks = self.config.chunks.to_tuple(dims[: data.ndim])

            # Clamp chunks to array size
            chunks = tuple(min(c, s) for c, s in zip(chunks, data.shape))

            arr = root.create_dataset(
                var_name,
                data=data,
                chunks=chunks,
                compressor=compressor,
                fill_value=self.config.fill_value,
                write_empty_chunks=self.config.write_empty_chunks,
            )

            # Set dimension attributes (for xarray compatibility)
            arr.attrs["_ARRAY_DIMENSIONS"] = dims[: data.ndim]

            # Set per-array attributes
            if array_attrs and var_name in array_attrs:
                arr.attrs.update(array_attrs[var_name])

            array_infos.append(
                ZarrArrayInfo(
                    name=var_name,
                    shape=data.shape,
                    chunks=chunks,
                    dtype=str(data.dtype),
                    dimensions=dims[: data.ndim],
                    compression=self.config.compression.value if compressor else None,
                    fill_value=self.config.fill_value,
                    attrs=dict(arr.attrs),
                )
            )

        # Consolidate metadata if requested
        if self.config.consolidated:
            zarr.consolidate_metadata(store)

        # Close store
        if hasattr(store, "close"):
            store.close()

        # Calculate store size
        store_size = self._calculate_store_size(output_path)

        return ZarrResult(
            output_path=output_path,
            store_size_bytes=store_size,
            arrays=array_infos,
            dimensions={d: len(coords.get(d, [])) for d in dims},
            coords={k: list(v) for k, v in coords.items()},
            global_attrs=attrs or {},
            consolidated=self.config.consolidated,
        )

    def convert_raster_stack(
        self,
        raster_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        timestamps: List[datetime],
        variable_name: str = "data",
    ) -> ZarrResult:
        """
        Convert a stack of raster files to Zarr format.

        Args:
            raster_paths: List of paths to raster files
            output_path: Path for output Zarr store
            timestamps: Timestamp for each raster
            variable_name: Name for the data variable

        Returns:
            ZarrResult with conversion details

        Raises:
            ValueError: If rasters have inconsistent shapes
            FileNotFoundError: If any raster file doesn't exist
        """
        output_path = Path(output_path)

        if len(raster_paths) != len(timestamps):
            raise ValueError(
                f"Number of rasters ({len(raster_paths)}) must match "
                f"timestamps ({len(timestamps)})"
            )

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for raster stack conversion")

        logger.info(f"Converting {len(raster_paths)} rasters to Zarr stack")

        # Read first raster to get dimensions
        first_path = Path(raster_paths[0])
        with rasterio.open(first_path) as src:
            height, width = src.height, src.width
            band_count = src.count
            dtype = src.dtypes[0]
            crs = str(src.crs) if src.crs else None
            transform = src.transform

            # Generate coordinate arrays
            y_coords = [transform.c + transform.e * (i + 0.5) for i in range(height)]
            x_coords = [transform.f + transform.a * (i + 0.5) for i in range(width)]

        # Stack all rasters
        stack = np.empty(
            (len(raster_paths), band_count, height, width) if band_count > 1
            else (len(raster_paths), height, width),
            dtype=dtype,
        )

        for i, raster_path in enumerate(raster_paths):
            with rasterio.open(raster_path) as src:
                if src.height != height or src.width != width:
                    raise ValueError(
                        f"Raster {raster_path} has inconsistent shape: "
                        f"expected ({height}, {width}), got ({src.height}, {src.width})"
                    )
                data = src.read()
                if band_count > 1:
                    stack[i] = data
                else:
                    stack[i] = data[0]

        # Prepare coordinates and dimensions
        if band_count > 1:
            dims = ["time", "band", "y", "x"]
            coords = {
                "time": [t.isoformat() for t in timestamps],
                "band": list(range(1, band_count + 1)),
                "y": y_coords,
                "x": x_coords,
            }
        else:
            dims = ["time", "y", "x"]
            coords = {
                "time": [t.isoformat() for t in timestamps],
                "y": y_coords,
                "x": x_coords,
            }

        # Global attributes
        attrs = {
            "crs": crs,
            "transform": list(transform)[:6],
            "created": datetime.now(timezone.utc).isoformat(),
            "source_count": len(raster_paths),
        }

        return self.convert_arrays(
            arrays={variable_name: stack},
            output_path=output_path,
            coords=coords,
            dims=dims,
            attrs=attrs,
        )

    def append_timestep(
        self,
        zarr_path: Union[str, Path],
        data: np.ndarray,
        timestamp: datetime,
        variable_name: str = "data",
    ) -> None:
        """
        Append a new timestep to an existing Zarr store.

        Args:
            zarr_path: Path to existing Zarr store
            data: Data array for new timestep
            timestamp: Timestamp for new data
            variable_name: Name of the data variable
        """
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required for append operation")

        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store, mode="r+")

        # Get existing array
        arr = root[variable_name]

        # Append along time dimension
        arr.append(data[np.newaxis], axis=0)

        # Update time coordinate
        if "time" in root:
            time_arr = root["time"]
            new_times = np.append(time_arr[:], timestamp.isoformat())
            root["time"] = new_times

        logger.info(f"Appended timestep {timestamp} to {zarr_path}")

    def _calculate_store_size(self, path: Path) -> int:
        """Calculate total size of a Zarr store."""
        if not path.exists():
            return 0

        if path.is_file():
            return path.stat().st_size

        total_size = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size


def convert_to_zarr(
    arrays: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    coords: Dict[str, List[Any]],
    compression: str = "zstd",
    chunk_size: int = 512,
) -> ZarrResult:
    """
    Convenience function to convert arrays to Zarr format.

    Args:
        arrays: Dictionary mapping variable names to numpy arrays
        output_path: Path for output Zarr store
        coords: Dictionary mapping dimension names to coordinate values
        compression: Compression method
        chunk_size: Chunk size for spatial dimensions

    Returns:
        ZarrResult with conversion details
    """
    config = ZarrConfig(
        chunks=ChunkConfig(time=1, y=chunk_size, x=chunk_size),
        compression=ZarrCompression(compression.lower()),
    )
    converter = ZarrConverter(config)
    return converter.convert_arrays(arrays, output_path, coords)
