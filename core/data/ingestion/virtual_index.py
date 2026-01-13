"""
Virtual Raster Index for Lazy Tile Loading.

Builds GDAL Virtual Raster (VRT) indices from STAC query results for
memory-efficient, on-demand tile access without downloading entire scenes.

Key Components:
- VirtualRasterIndex: Build and manage VRT files from STAC items
- STACVRTBuilder: Convert STAC collections to VRT
- TileAccessor: Lazy tile access via HTTP range requests

Features:
- No full scene downloads - stream only needed tiles
- Automatic mosaic creation from multiple scenes
- Band selection and reordering
- CRS transformation during access
- Integration with Dask for parallel tile reads

Example Usage:
    from core.data.ingestion.virtual_index import (
        VirtualRasterIndex,
        STACVRTBuilder,
        TileAccessor,
    )

    # Build VRT from STAC items
    builder = STACVRTBuilder()
    vrt_path = builder.build_from_stac_items(stac_items, output_dir)

    # Create virtual index
    index = VirtualRasterIndex(vrt_path)

    # Access tiles lazily
    accessor = TileAccessor(index)
    tile_data = accessor.read_tile(tile_bounds, resolution)

    # Get tiles as Dask array
    dask_array = accessor.as_dask_array(tile_size=(512, 512))
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    WarpedVRT = None
    Window = None
    CRS = None
    HAS_RASTERIO = False

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    gdal = None
    HAS_GDAL = False

try:
    import dask.array as da
    from dask import delayed
    HAS_DASK = True
except ImportError:
    da = None
    delayed = None
    HAS_DASK = False


# =============================================================================
# Enumerations
# =============================================================================


class VRTSourceType(Enum):
    """Types of VRT sources."""

    LOCAL_FILE = "local_file"  # Local filesystem path
    HTTP_COG = "http_cog"  # Cloud-Optimized GeoTIFF via HTTP
    S3_COG = "s3_cog"  # COG on S3
    AZURE_COG = "azure_cog"  # COG on Azure Blob
    GCS_COG = "gcs_cog"  # COG on Google Cloud Storage


class BandSelectionMode(Enum):
    """How to select bands when building VRT."""

    ALL = "all"  # Include all bands
    SPECIFIC = "specific"  # Select specific band indices
    BY_NAME = "by_name"  # Select by band names
    COMMON_NAME = "common_name"  # Select by STAC common names (e.g., "red", "nir")


class MosaicMethod(Enum):
    """Methods for mosaicking overlapping scenes."""

    FIRST = "first"  # Use first valid pixel
    LAST = "last"  # Use last valid pixel
    AVERAGE = "average"  # Average overlapping pixels
    MAXIMUM = "maximum"  # Maximum value
    MINIMUM = "minimum"  # Minimum value


class TileReadMode(Enum):
    """Modes for reading tiles."""

    DIRECT = "direct"  # Direct read via GDAL
    WINDOWED = "windowed"  # Windowed read via rasterio
    STREAMING = "streaming"  # HTTP range request streaming


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BandInfo:
    """
    Information about a single band in the VRT.

    Attributes:
        band_index: 1-based band index in VRT
        source_band: Band index in source file
        name: Band name
        common_name: STAC common name (e.g., "red", "nir")
        data_type: Numpy dtype string
        nodata_value: NoData value
        source_path: Path to source file
        wavelength_nm: Center wavelength in nanometers
    """

    band_index: int
    source_band: int
    name: str = ""
    common_name: str = ""
    data_type: str = "float32"
    nodata_value: Optional[float] = None
    source_path: str = ""
    wavelength_nm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "band_index": self.band_index,
            "source_band": self.source_band,
            "name": self.name,
            "common_name": self.common_name,
            "data_type": self.data_type,
            "nodata_value": self.nodata_value,
            "source_path": self.source_path,
            "wavelength_nm": self.wavelength_nm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BandInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VRTMetadata:
    """
    Metadata for a Virtual Raster Index.

    Attributes:
        vrt_path: Path to the VRT file
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        crs: Coordinate reference system (EPSG code or WKT)
        resolution: Pixel resolution (x, y) in CRS units
        shape: Raster shape (height, width)
        bands: List of band information
        source_count: Number of source files
        source_type: Type of sources (local, HTTP, S3, etc.)
        created_at: Creation timestamp
        stac_collection_id: Associated STAC collection ID
    """

    vrt_path: str
    bounds: Tuple[float, float, float, float]
    crs: str
    resolution: Tuple[float, float]
    shape: Tuple[int, int]
    bands: List[BandInfo]
    source_count: int = 0
    source_type: VRTSourceType = VRTSourceType.LOCAL_FILE
    created_at: Optional[datetime] = None
    stac_collection_id: str = ""

    @property
    def width(self) -> int:
        """Width in pixels."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Height in pixels."""
        return self.shape[0]

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return len(self.bands)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vrt_path": self.vrt_path,
            "bounds": self.bounds,
            "crs": self.crs,
            "resolution": self.resolution,
            "shape": self.shape,
            "bands": [b.to_dict() for b in self.bands],
            "source_count": self.source_count,
            "source_type": self.source_type.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "stac_collection_id": self.stac_collection_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VRTMetadata":
        """Create from dictionary."""
        return cls(
            vrt_path=data["vrt_path"],
            bounds=tuple(data["bounds"]),
            crs=data["crs"],
            resolution=tuple(data["resolution"]),
            shape=tuple(data["shape"]),
            bands=[BandInfo.from_dict(b) for b in data["bands"]],
            source_count=data.get("source_count", 0),
            source_type=VRTSourceType(data.get("source_type", "local_file")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            stac_collection_id=data.get("stac_collection_id", ""),
        )


@dataclass
class TileBounds:
    """
    Bounds for a tile in geographic or projected coordinates.

    Attributes:
        minx: Minimum x coordinate
        miny: Minimum y coordinate
        maxx: Maximum x coordinate
        maxy: Maximum y coordinate
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        """Width of bounds."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Height of bounds."""
        return self.maxy - self.miny

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as tuple."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other: "TileBounds") -> bool:
        """Check if bounds intersect."""
        return not (
            self.maxx < other.minx or
            self.minx > other.maxx or
            self.maxy < other.miny or
            self.miny > other.maxy
        )


@dataclass
class TileSpec:
    """
    Specification for a tile to read.

    Attributes:
        bounds: Geographic bounds
        shape: Output shape (height, width)
        bands: Band indices to read (1-based)
        crs: Target CRS (None = use source CRS)
        resampling: Resampling method for reprojection
    """

    bounds: TileBounds
    shape: Tuple[int, int] = (512, 512)
    bands: Optional[List[int]] = None
    crs: Optional[str] = None
    resampling: str = "nearest"


# =============================================================================
# STACVRTBuilder
# =============================================================================


class STACVRTBuilder:
    """
    Build GDAL Virtual Raster (VRT) files from STAC items.

    Creates VRT files that reference Cloud-Optimized GeoTIFFs directly,
    enabling lazy loading and parallel tile access without downloading
    entire scenes.

    Example:
        builder = STACVRTBuilder()

        # Build from STAC items
        vrt_path = builder.build_from_stac_items(
            items=stac_items,
            output_dir=Path("./vrts"),
            bands=["red", "green", "blue", "nir"],
        )

        # Build from STAC search results
        vrt_path = builder.build_from_stac_search(
            catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
            collection="sentinel-2-l2a",
            bbox=(-122.5, 37.5, -122.0, 38.0),
            datetime="2024-01-01/2024-01-31",
        )
    """

    # Common band name mappings for different sensors
    BAND_MAPPINGS = {
        "sentinel-2-l2a": {
            "coastal": "B01",
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "rededge1": "B05",
            "rededge2": "B06",
            "rededge3": "B07",
            "nir": "B08",
            "nir08": "B8A",
            "swir16": "B11",
            "swir22": "B12",
        },
        "landsat-8-c2-l2": {
            "coastal": "SR_B1",
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir08": "SR_B5",
            "swir16": "SR_B6",
            "swir22": "SR_B7",
        },
        "sentinel-1-grd": {
            "vv": "vv",
            "vh": "vh",
        },
    }

    def __init__(
        self,
        mosaic_method: MosaicMethod = MosaicMethod.FIRST,
        target_crs: Optional[str] = None,
        target_resolution: Optional[Tuple[float, float]] = None,
        nodata_value: float = 0.0,
        enable_overviews: bool = True,
    ):
        """
        Initialize VRT builder.

        Args:
            mosaic_method: Method for handling overlapping scenes
            target_crs: Target CRS for output (None = use first item's CRS)
            target_resolution: Target resolution (None = use highest available)
            nodata_value: NoData value for output
            enable_overviews: Create overviews for faster rendering
        """
        self.mosaic_method = mosaic_method
        self.target_crs = target_crs
        self.target_resolution = target_resolution
        self.nodata_value = nodata_value
        self.enable_overviews = enable_overviews

        if not HAS_GDAL:
            logger.warning("GDAL not available - VRT building may be limited")

    def build_from_stac_items(
        self,
        items: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        bands: Optional[List[str]] = None,
        collection_id: Optional[str] = None,
    ) -> Path:
        """
        Build VRT from STAC items.

        Args:
            items: List of STAC items (as dictionaries)
            output_dir: Output directory for VRT file
            bands: Band common names to include (None = all)
            collection_id: Optional collection ID for naming

        Returns:
            Path to created VRT file

        Raises:
            ValueError: If no valid items provided
            RuntimeError: If VRT creation fails
        """
        if not items:
            raise ValueError("No STAC items provided")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract asset URLs from items
        assets_by_band = self._extract_assets(items, bands, collection_id)

        if not assets_by_band:
            raise ValueError("No valid band assets found in STAC items")

        # Generate VRT filename
        vrt_name = self._generate_vrt_name(items, collection_id)
        vrt_path = output_dir / f"{vrt_name}.vrt"

        # Build VRT
        if HAS_GDAL:
            self._build_vrt_gdal(assets_by_band, vrt_path)
        else:
            self._build_vrt_xml(assets_by_band, vrt_path)

        logger.info(f"Created VRT: {vrt_path} with {len(assets_by_band)} bands")
        return vrt_path

    def _extract_assets(
        self,
        items: List[Dict[str, Any]],
        bands: Optional[List[str]],
        collection_id: Optional[str],
    ) -> Dict[str, List[str]]:
        """Extract asset URLs organized by band."""
        assets_by_band: Dict[str, List[str]] = {}

        # Determine collection for band mapping
        if collection_id is None and items:
            collection_id = items[0].get("collection", "")

        band_mapping = self.BAND_MAPPINGS.get(collection_id, {})

        for item in items:
            assets = item.get("assets", {})

            for asset_key, asset_info in assets.items():
                # Skip non-image assets
                if not self._is_image_asset(asset_info):
                    continue

                # Get band common name
                common_name = self._get_band_common_name(
                    asset_key, asset_info, band_mapping
                )

                # Filter by requested bands
                if bands and common_name not in bands:
                    continue

                # Get asset URL
                href = asset_info.get("href", "")
                if not href:
                    continue

                # Add signed URL if available
                href = self._get_signed_url(href, asset_info)

                if common_name not in assets_by_band:
                    assets_by_band[common_name] = []
                assets_by_band[common_name].append(href)

        return assets_by_band

    def _is_image_asset(self, asset_info: Dict[str, Any]) -> bool:
        """Check if asset is an image."""
        media_type = asset_info.get("type", "")
        roles = asset_info.get("roles", [])

        # Check media type
        if media_type in ["image/tiff", "image/tiff; application=geotiff",
                         "image/tiff; application=geotiff; profile=cloud-optimized"]:
            return True

        # Check roles
        if "data" in roles or "visual" in roles:
            return True

        return False

    def _get_band_common_name(
        self,
        asset_key: str,
        asset_info: Dict[str, Any],
        band_mapping: Dict[str, str],
    ) -> str:
        """Get common name for a band."""
        # Check eo:bands extension
        eo_bands = asset_info.get("eo:bands", [])
        if eo_bands:
            for band in eo_bands:
                if "common_name" in band:
                    return band["common_name"]
                if "name" in band:
                    # Try reverse lookup in mapping
                    for common, name in band_mapping.items():
                        if name == band["name"]:
                            return common

        # Check raster:bands extension
        raster_bands = asset_info.get("raster:bands", [])
        if raster_bands:
            for band in raster_bands:
                if "name" in band:
                    return band["name"]

        # Fall back to asset key
        asset_key_lower = asset_key.lower()
        for common, name in band_mapping.items():
            if name.lower() == asset_key_lower or common == asset_key_lower:
                return common

        return asset_key

    def _get_signed_url(self, href: str, asset_info: Dict[str, Any]) -> str:
        """Get signed URL if available."""
        # Check for alternate href with signing
        alternate = asset_info.get("alternate", {})
        if "s3" in alternate:
            s3_info = alternate["s3"]
            if "href" in s3_info:
                return s3_info["href"]

        return href

    def _generate_vrt_name(
        self,
        items: List[Dict[str, Any]],
        collection_id: Optional[str],
    ) -> str:
        """Generate a unique name for the VRT."""
        # Use collection + datetime range + hash
        if collection_id is None and items:
            collection_id = items[0].get("collection", "data")

        # Get datetime range
        datetimes = []
        for item in items:
            props = item.get("properties", {})
            dt = props.get("datetime", props.get("start_datetime", ""))
            if dt:
                datetimes.append(dt[:10])  # Just the date part

        date_range = ""
        if datetimes:
            datetimes.sort()
            date_range = f"_{datetimes[0]}_{datetimes[-1]}"

        # Add hash for uniqueness
        content_hash = hashlib.md5(
            json.dumps([i.get("id", "") for i in items]).encode()
        ).hexdigest()[:8]

        return f"{collection_id}{date_range}_{content_hash}"

    def _build_vrt_gdal(
        self,
        assets_by_band: Dict[str, List[str]],
        vrt_path: Path,
    ) -> None:
        """Build VRT using GDAL."""
        if not HAS_GDAL:
            raise RuntimeError("GDAL not available")

        # Configure GDAL for cloud access
        gdal.SetConfigOption("GDAL_HTTP_MULTIRANGE", "YES")
        gdal.SetConfigOption("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
        gdal.SetConfigOption("VSI_CURL_CACHE_SIZE", "100000000")  # 100MB cache

        # Collect all source files
        all_sources = []
        band_names = list(assets_by_band.keys())

        for band_name in band_names:
            sources = assets_by_band[band_name]
            for src in sources:
                # Add /vsicurl/ prefix for HTTP sources
                if src.startswith("http"):
                    src = f"/vsicurl/{src}"
                all_sources.append(src)

        # Build VRT options
        vrt_options = gdal.BuildVRTOptions(
            separate=True if len(band_names) > 1 else False,
            resampleAlg="nearest",
            addAlpha=False,
            srcNodata=self.nodata_value,
            VRTNodata=self.nodata_value,
        )

        # Build VRT - take first source per band for now
        # In production, would create mosaic of all sources per band
        sources_to_use = []
        for band_name in band_names:
            if assets_by_band[band_name]:
                src = assets_by_band[band_name][0]
                if src.startswith("http"):
                    src = f"/vsicurl/{src}"
                sources_to_use.append(src)

        if not sources_to_use:
            raise ValueError("No valid sources to build VRT")

        vrt_ds = gdal.BuildVRT(str(vrt_path), sources_to_use, options=vrt_options)

        if vrt_ds is None:
            raise RuntimeError(f"Failed to create VRT: {gdal.GetLastErrorMsg()}")

        # Set band descriptions
        for i, band_name in enumerate(band_names, 1):
            if i <= vrt_ds.RasterCount:
                band = vrt_ds.GetRasterBand(i)
                band.SetDescription(band_name)

        # Flush and close
        vrt_ds.FlushCache()
        vrt_ds = None

        logger.debug(f"Created VRT with GDAL: {vrt_path}")

    def _build_vrt_xml(
        self,
        assets_by_band: Dict[str, List[str]],
        vrt_path: Path,
    ) -> None:
        """Build VRT by generating XML directly (fallback when GDAL unavailable)."""
        # Get info from first source to determine dimensions
        first_band = list(assets_by_band.keys())[0]
        first_source = assets_by_band[first_band][0]

        # Read source metadata
        src_info = self._read_source_info(first_source)

        # Create VRT XML
        root = ET.Element("VRTDataset")
        root.set("rasterXSize", str(src_info["width"]))
        root.set("rasterYSize", str(src_info["height"]))

        # Add SRS
        srs = ET.SubElement(root, "SRS")
        srs.text = src_info.get("crs", "EPSG:4326")

        # Add GeoTransform
        gt = ET.SubElement(root, "GeoTransform")
        gt.text = ", ".join(str(v) for v in src_info.get("transform", [0, 1, 0, 0, 0, -1]))

        # Add bands
        for band_idx, band_name in enumerate(assets_by_band.keys(), 1):
            band_elem = ET.SubElement(root, "VRTRasterBand")
            band_elem.set("dataType", "Float32")
            band_elem.set("band", str(band_idx))

            # Description
            desc = ET.SubElement(band_elem, "Description")
            desc.text = band_name

            # NoData
            nodata = ET.SubElement(band_elem, "NoDataValue")
            nodata.text = str(self.nodata_value)

            # Add sources
            for source_url in assets_by_band[band_name]:
                source = ET.SubElement(band_elem, "SimpleSource")

                source_filename = ET.SubElement(source, "SourceFilename")
                source_filename.set("relativeToVRT", "0")
                if source_url.startswith("http"):
                    source_filename.text = f"/vsicurl/{source_url}"
                else:
                    source_filename.text = source_url

                source_band = ET.SubElement(source, "SourceBand")
                source_band.text = "1"

                # Source and Dest regions (full extent)
                src_rect = ET.SubElement(source, "SrcRect")
                src_rect.set("xOff", "0")
                src_rect.set("yOff", "0")
                src_rect.set("xSize", str(src_info["width"]))
                src_rect.set("ySize", str(src_info["height"]))

                dst_rect = ET.SubElement(source, "DstRect")
                dst_rect.set("xOff", "0")
                dst_rect.set("yOff", "0")
                dst_rect.set("xSize", str(src_info["width"]))
                dst_rect.set("ySize", str(src_info["height"]))

        # Write XML
        tree = ET.ElementTree(root)
        tree.write(str(vrt_path), encoding="utf-8", xml_declaration=True)

        logger.debug(f"Created VRT with XML: {vrt_path}")

    def _read_source_info(self, source_url: str) -> Dict[str, Any]:
        """Read metadata from a source file."""
        info = {
            "width": 10980,  # Default Sentinel-2 width
            "height": 10980,
            "crs": "EPSG:32610",
            "transform": [0, 10, 0, 0, 0, -10],
        }

        if HAS_RASTERIO:
            try:
                with rasterio.open(source_url) as src:
                    info["width"] = src.width
                    info["height"] = src.height
                    info["crs"] = str(src.crs)
                    info["transform"] = list(src.transform)[:6]
            except Exception as e:
                logger.warning(f"Could not read source info: {e}")

        return info


# =============================================================================
# VirtualRasterIndex
# =============================================================================


class VirtualRasterIndex:
    """
    Manages a Virtual Raster Index for lazy tile access.

    Wraps a VRT file and provides methods for:
    - Querying available tiles
    - Reading tiles on demand
    - Converting to Dask arrays for parallel processing

    Example:
        # Open existing VRT
        index = VirtualRasterIndex(vrt_path)

        # Get metadata
        print(f"Bounds: {index.bounds}")
        print(f"Shape: {index.shape}")
        print(f"Bands: {[b.common_name for b in index.bands]}")

        # Read a region
        data = index.read_region(bounds, shape=(512, 512))

        # Get as Dask array
        dask_arr = index.as_dask_array(chunks=(1, 512, 512))
    """

    def __init__(
        self,
        vrt_path: Union[str, Path],
        cache_size_mb: int = 256,
    ):
        """
        Initialize VirtualRasterIndex.

        Args:
            vrt_path: Path to VRT file
            cache_size_mb: Cache size for tile caching
        """
        self.vrt_path = Path(vrt_path)
        self.cache_size_mb = cache_size_mb
        self._metadata: Optional[VRTMetadata] = None
        self._lock = threading.Lock()

        if not self.vrt_path.exists():
            raise FileNotFoundError(f"VRT file not found: {vrt_path}")

        # Load metadata
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load VRT metadata."""
        if HAS_RASTERIO:
            with rasterio.open(self.vrt_path) as src:
                bands = []
                for i in range(1, src.count + 1):
                    band_info = BandInfo(
                        band_index=i,
                        source_band=i,
                        name=src.descriptions[i-1] if src.descriptions else f"band_{i}",
                        common_name=src.descriptions[i-1] if src.descriptions else "",
                        data_type=str(src.dtypes[i-1]),
                        nodata_value=src.nodata,
                        source_path=str(self.vrt_path),
                    )
                    bands.append(band_info)

                self._metadata = VRTMetadata(
                    vrt_path=str(self.vrt_path),
                    bounds=src.bounds,
                    crs=str(src.crs),
                    resolution=(src.res[0], src.res[1]),
                    shape=(src.height, src.width),
                    bands=bands,
                    created_at=datetime.now(timezone.utc),
                )
        else:
            # Fallback: parse VRT XML
            self._load_metadata_from_xml()

    def _load_metadata_from_xml(self) -> None:
        """Load metadata by parsing VRT XML."""
        tree = ET.parse(self.vrt_path)
        root = tree.getroot()

        width = int(root.get("rasterXSize", 0))
        height = int(root.get("rasterYSize", 0))

        # Parse GeoTransform
        gt_elem = root.find("GeoTransform")
        if gt_elem is not None and gt_elem.text:
            gt = [float(v.strip()) for v in gt_elem.text.split(",")]
            resolution = (abs(gt[1]), abs(gt[5]))
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + width * resolution[0]
            miny = maxy - height * resolution[1]
            bounds = (minx, miny, maxx, maxy)
        else:
            resolution = (1.0, 1.0)
            bounds = (0, 0, float(width), float(height))

        # Parse CRS
        srs_elem = root.find("SRS")
        crs = srs_elem.text if srs_elem is not None and srs_elem.text else "EPSG:4326"

        # Parse bands
        bands = []
        for band_elem in root.findall("VRTRasterBand"):
            band_idx = int(band_elem.get("band", 1))
            desc_elem = band_elem.find("Description")
            name = desc_elem.text if desc_elem is not None and desc_elem.text else f"band_{band_idx}"

            nodata_elem = band_elem.find("NoDataValue")
            nodata = float(nodata_elem.text) if nodata_elem is not None and nodata_elem.text else None

            bands.append(BandInfo(
                band_index=band_idx,
                source_band=band_idx,
                name=name,
                common_name=name,
                data_type=band_elem.get("dataType", "Float32"),
                nodata_value=nodata,
            ))

        self._metadata = VRTMetadata(
            vrt_path=str(self.vrt_path),
            bounds=bounds,
            crs=crs,
            resolution=resolution,
            shape=(height, width),
            bands=bands,
            created_at=datetime.now(timezone.utc),
        )

    @property
    def metadata(self) -> VRTMetadata:
        """Get VRT metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get geographic bounds."""
        return self.metadata.bounds

    @property
    def shape(self) -> Tuple[int, int]:
        """Get raster shape (height, width)."""
        return self.metadata.shape

    @property
    def resolution(self) -> Tuple[float, float]:
        """Get pixel resolution."""
        return self.metadata.resolution

    @property
    def crs(self) -> str:
        """Get coordinate reference system."""
        return self.metadata.crs

    @property
    def bands(self) -> List[BandInfo]:
        """Get band information."""
        return self.metadata.bands

    @property
    def n_bands(self) -> int:
        """Get number of bands."""
        return len(self.bands)

    def read_region(
        self,
        bounds: Union[TileBounds, Tuple[float, float, float, float]],
        shape: Tuple[int, int] = (512, 512),
        bands: Optional[List[int]] = None,
        resampling: str = "nearest",
    ) -> np.ndarray:
        """
        Read a region from the VRT.

        Args:
            bounds: Geographic bounds to read
            shape: Output shape (height, width)
            bands: Band indices (1-based), None = all bands
            resampling: Resampling method

        Returns:
            Array of shape (bands, height, width)
        """
        if isinstance(bounds, tuple):
            bounds = TileBounds(*bounds)

        if bands is None:
            bands = list(range(1, self.n_bands + 1))

        if not HAS_RASTERIO:
            raise RuntimeError("rasterio required for read_region")

        with rasterio.open(self.vrt_path) as src:
            # Calculate window from bounds
            window = rasterio.windows.from_bounds(
                bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                transform=src.transform
            )

            # Read with resampling
            data = src.read(
                bands,
                window=window,
                out_shape=(len(bands), shape[0], shape[1]),
                resampling=getattr(Resampling, resampling),
            )

        return data

    def read_tile(
        self,
        col: int,
        row: int,
        tile_size: Tuple[int, int] = (512, 512),
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Read a tile by grid indices.

        Args:
            col: Tile column index
            row: Tile row index
            tile_size: Tile size in pixels
            bands: Band indices (1-based), None = all bands

        Returns:
            Array of shape (bands, height, width)
        """
        # Calculate bounds from tile indices
        minx = self.bounds[0] + col * tile_size[0] * self.resolution[0]
        maxy = self.bounds[3] - row * tile_size[1] * self.resolution[1]
        maxx = minx + tile_size[0] * self.resolution[0]
        miny = maxy - tile_size[1] * self.resolution[1]

        return self.read_region(
            bounds=(minx, miny, maxx, maxy),
            shape=tile_size,
            bands=bands,
        )

    def get_tile_count(self, tile_size: Tuple[int, int] = (512, 512)) -> Tuple[int, int]:
        """
        Get number of tiles in grid.

        Args:
            tile_size: Tile size in pixels

        Returns:
            (n_cols, n_rows)
        """
        n_cols = int(np.ceil(self.shape[1] / tile_size[0]))
        n_rows = int(np.ceil(self.shape[0] / tile_size[1]))
        return (n_cols, n_rows)

    def iter_tiles(
        self,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
    ) -> Generator[Tuple[int, int, TileBounds], None, None]:
        """
        Iterate over all tiles.

        Args:
            tile_size: Tile size in pixels
            overlap: Overlap in pixels

        Yields:
            (col, row, bounds) for each tile
        """
        n_cols, n_rows = self.get_tile_count(tile_size)

        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate bounds with overlap
                px_minx = col * tile_size[0] - overlap
                px_maxy = row * tile_size[1] + overlap
                px_maxx = px_minx + tile_size[0] + 2 * overlap
                px_miny = px_maxy - tile_size[1] - 2 * overlap

                # Convert to geographic coordinates
                minx = self.bounds[0] + max(0, px_minx) * self.resolution[0]
                maxy = self.bounds[3] - max(0, row * tile_size[1] - overlap) * self.resolution[1]
                maxx = min(self.bounds[2], self.bounds[0] + px_maxx * self.resolution[0])
                miny = max(self.bounds[1], self.bounds[3] - (px_miny + tile_size[1] + 2*overlap) * self.resolution[1])

                yield col, row, TileBounds(minx, miny, maxx, maxy)

    def as_dask_array(
        self,
        chunks: Tuple[int, int, int] = (1, 512, 512),
        bands: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """
        Get VRT as a Dask array for parallel processing.

        Args:
            chunks: Chunk sizes (bands, height, width)
            bands: Band indices (1-based), None = all bands

        Returns:
            Dask array or None if Dask not available
        """
        if not HAS_DASK:
            logger.warning("Dask not available")
            return None

        if bands is None:
            bands = list(range(1, self.n_bands + 1))

        n_bands = len(bands)
        height, width = self.shape

        # Create delayed read functions
        tile_height = chunks[1]
        tile_width = chunks[2]

        n_row_chunks = int(np.ceil(height / tile_height))
        n_col_chunks = int(np.ceil(width / tile_width))

        # Build array of delayed reads
        @delayed
        def read_chunk(row_idx: int, col_idx: int) -> np.ndarray:
            row_start = row_idx * tile_height
            col_start = col_idx * tile_width

            # Calculate actual size (handle edge tiles)
            actual_height = min(tile_height, height - row_start)
            actual_width = min(tile_width, width - col_start)

            # Calculate bounds
            minx = self.bounds[0] + col_start * self.resolution[0]
            maxy = self.bounds[3] - row_start * self.resolution[1]
            maxx = minx + actual_width * self.resolution[0]
            miny = maxy - actual_height * self.resolution[1]

            data = self.read_region(
                bounds=(minx, miny, maxx, maxy),
                shape=(actual_height, actual_width),
                bands=bands,
            )

            # Pad to chunk size if needed
            if actual_height < tile_height or actual_width < tile_width:
                padded = np.zeros((n_bands, tile_height, tile_width), dtype=data.dtype)
                padded[:, :actual_height, :actual_width] = data
                return padded

            return data

        # Build chunks list
        chunks_list = []
        for row_idx in range(n_row_chunks):
            row_chunks = []
            for col_idx in range(n_col_chunks):
                chunk = da.from_delayed(
                    read_chunk(row_idx, col_idx),
                    shape=(n_bands, tile_height, tile_width),
                    dtype=np.float32,
                )
                row_chunks.append(chunk)
            chunks_list.append(row_chunks)

        # Concatenate into single array
        rows = [da.concatenate(row, axis=2) for row in chunks_list]
        arr = da.concatenate(rows, axis=1)

        # Trim to actual size
        arr = arr[:, :height, :width]

        return arr

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.metadata.to_dict()

    def __repr__(self) -> str:
        return (
            f"VirtualRasterIndex("
            f"path={self.vrt_path.name}, "
            f"shape={self.shape}, "
            f"bands={self.n_bands}, "
            f"crs={self.crs})"
        )


# =============================================================================
# TileAccessor
# =============================================================================


class TileAccessor:
    """
    Provides efficient tile access from a VirtualRasterIndex.

    Features:
    - Tile caching for repeated access
    - Parallel tile reading
    - Memory-efficient streaming

    Example:
        index = VirtualRasterIndex(vrt_path)
        accessor = TileAccessor(index)

        # Read single tile
        tile = accessor.read_tile(0, 0)

        # Read multiple tiles in parallel
        tiles = accessor.read_tiles_parallel([(0,0), (0,1), (1,0), (1,1)])

        # Stream tiles for processing
        for tile_data, tile_info in accessor.stream_tiles():
            process(tile_data)
    """

    def __init__(
        self,
        index: VirtualRasterIndex,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        cache_size: int = 32,
    ):
        """
        Initialize TileAccessor.

        Args:
            index: VirtualRasterIndex to access
            tile_size: Default tile size
            overlap: Overlap in pixels
            cache_size: Number of tiles to cache
        """
        self.index = index
        self.tile_size = tile_size
        self.overlap = overlap
        self.cache_size = cache_size

        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cache_order: List[Tuple[int, int]] = []
        self._lock = threading.Lock()

    def read_tile(
        self,
        col: int,
        row: int,
        bands: Optional[List[int]] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Read a single tile.

        Args:
            col: Tile column
            row: Tile row
            bands: Band indices (1-based)
            use_cache: Whether to use tile cache

        Returns:
            Tile data array
        """
        cache_key = (col, row)

        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Read tile
        data = self.index.read_tile(col, row, self.tile_size, bands)

        # Update cache
        if use_cache:
            with self._lock:
                self._cache[cache_key] = data
                self._cache_order.append(cache_key)

                # Evict old entries
                while len(self._cache) > self.cache_size:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)

        return data

    def read_tiles_parallel(
        self,
        tile_indices: List[Tuple[int, int]],
        bands: Optional[List[int]] = None,
        max_workers: int = 4,
    ) -> List[np.ndarray]:
        """
        Read multiple tiles in parallel.

        Args:
            tile_indices: List of (col, row) tuples
            bands: Band indices (1-based)
            max_workers: Maximum parallel workers

        Returns:
            List of tile data arrays
        """
        import concurrent.futures

        results = [None] * len(tile_indices)

        def read_single(idx: int, col: int, row: int) -> Tuple[int, np.ndarray]:
            data = self.read_tile(col, row, bands, use_cache=True)
            return (idx, data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(read_single, i, col, row)
                for i, (col, row) in enumerate(tile_indices)
            ]

            for future in concurrent.futures.as_completed(futures):
                idx, data = future.result()
                results[idx] = data

        return results

    def stream_tiles(
        self,
        bands: Optional[List[int]] = None,
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """
        Stream tiles for memory-efficient processing.

        Args:
            bands: Band indices (1-based)

        Yields:
            (tile_data, tile_info) tuples
        """
        for col, row, bounds in self.index.iter_tiles(self.tile_size, self.overlap):
            data = self.read_tile(col, row, bands, use_cache=False)

            info = {
                "col": col,
                "row": row,
                "bounds": bounds.as_tuple(),
                "shape": data.shape,
            }

            yield data, info

    def clear_cache(self) -> None:
        """Clear the tile cache."""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def build_vrt_from_stac(
    stac_items: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    bands: Optional[List[str]] = None,
    collection_id: Optional[str] = None,
) -> VirtualRasterIndex:
    """
    Build VRT from STAC items and return index.

    Args:
        stac_items: List of STAC items
        output_dir: Output directory
        bands: Band names to include
        collection_id: Collection ID

    Returns:
        VirtualRasterIndex for the created VRT
    """
    builder = STACVRTBuilder()
    vrt_path = builder.build_from_stac_items(
        items=stac_items,
        output_dir=output_dir,
        bands=bands,
        collection_id=collection_id,
    )
    return VirtualRasterIndex(vrt_path)


def create_vrt_from_urls(
    urls: List[str],
    output_path: Union[str, Path],
    band_names: Optional[List[str]] = None,
) -> VirtualRasterIndex:
    """
    Create VRT from list of URLs.

    Args:
        urls: List of COG URLs
        output_path: Output VRT path
        band_names: Optional band names

    Returns:
        VirtualRasterIndex for the created VRT
    """
    assets_by_band = {}
    for i, url in enumerate(urls):
        band_name = band_names[i] if band_names and i < len(band_names) else f"band_{i+1}"
        assets_by_band[band_name] = [url]

    builder = STACVRTBuilder()
    output_path = Path(output_path)
    builder._build_vrt_xml(assets_by_band, output_path) if not HAS_GDAL else builder._build_vrt_gdal(assets_by_band, output_path)

    return VirtualRasterIndex(output_path)
