"""
STAC Metadata Generator.

Generates SpatioTemporal Asset Catalog (STAC) metadata for geospatial
assets. Supports STAC 1.0.0 specification with common extensions.

STAC Benefits:
- Standardized metadata for search and discovery
- Rich extension ecosystem (EO, SAR, Processing, etc.)
- Interoperability across catalogs
- Self-describing assets
"""

import hashlib
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# STAC schema versions
STAC_VERSION = "1.0.0"
STAC_EXTENSIONS = {
    "eo": "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
    "sar": "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
    "proj": "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
    "view": "https://stac-extensions.github.io/view/v1.0.0/schema.json",
    "processing": "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
    "scientific": "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    "raster": "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
}


class AssetRole(Enum):
    """Standard STAC asset roles."""

    DATA = "data"
    METADATA = "metadata"
    THUMBNAIL = "thumbnail"
    OVERVIEW = "overview"
    VISUAL = "visual"
    DATE = "date"
    GRAPHIC = "graphic"
    SNOW_ICE = "snow-ice"
    LAND_WATER = "land-water"
    WATER_MASK = "water-mask"
    CLOUD = "cloud"
    CLOUD_SHADOW = "cloud-shadow"


class MediaType(Enum):
    """Common media types for geospatial assets."""

    GEOTIFF = "image/tiff; application=geotiff"
    COG = "image/tiff; application=geotiff; profile=cloud-optimized"
    JPEG2000 = "image/jp2"
    PNG = "image/png"
    JPEG = "image/jpeg"
    JSON = "application/json"
    GEOJSON = "application/geo+json"
    GEOPACKAGE = "application/geopackage+sqlite3"
    PARQUET = "application/x-parquet"
    ZARR = "application/x-zarr"
    NETCDF = "application/x-netcdf"
    HDF5 = "application/x-hdf5"
    XML = "application/xml"
    TEXT = "text/plain"


@dataclass
class BandInfo:
    """Information about a raster band for EO extension."""

    name: str
    common_name: Optional[str] = None  # blue, green, red, nir, etc.
    description: Optional[str] = None
    center_wavelength: Optional[float] = None  # micrometers
    full_width_half_max: Optional[float] = None
    solar_illumination: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to STAC EO band object."""
        band = {"name": self.name}
        if self.common_name:
            band["common_name"] = self.common_name
        if self.description:
            band["description"] = self.description
        if self.center_wavelength is not None:
            band["center_wavelength"] = self.center_wavelength
        if self.full_width_half_max is not None:
            band["full_width_half_max"] = self.full_width_half_max
        if self.solar_illumination is not None:
            band["solar_illumination"] = self.solar_illumination
        return band


@dataclass
class RasterBandInfo:
    """Raster band statistics for raster extension."""

    data_type: str
    nodata: Optional[float] = None
    statistics: Optional[Dict[str, float]] = None  # min, max, mean, stddev
    unit: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    histogram: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to STAC raster band object."""
        band = {"data_type": self.data_type}
        if self.nodata is not None:
            band["nodata"] = self.nodata
        if self.statistics:
            band["statistics"] = self.statistics
        if self.unit:
            band["unit"] = self.unit
        if self.scale is not None:
            band["scale"] = self.scale
        if self.offset is not None:
            band["offset"] = self.offset
        if self.histogram:
            band["histogram"] = self.histogram
        return band


@dataclass
class AssetInfo:
    """Information about a STAC asset."""

    href: str
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[MediaType] = None
    roles: List[AssetRole] = field(default_factory=list)
    eo_bands: Optional[List[int]] = None  # Indices into item's eo:bands
    raster_bands: Optional[List[RasterBandInfo]] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to STAC asset object."""
        asset = {"href": self.href}
        if self.title:
            asset["title"] = self.title
        if self.description:
            asset["description"] = self.description
        if self.type:
            asset["type"] = self.type.value
        if self.roles:
            asset["roles"] = [r.value for r in self.roles]
        if self.eo_bands is not None:
            asset["eo:bands"] = self.eo_bands
        if self.raster_bands:
            asset["raster:bands"] = [b.to_dict() for b in self.raster_bands]
        asset.update(self.extra_fields)
        return asset


@dataclass
class STACItemConfig:
    """
    Configuration for STAC Item generation.

    Attributes:
        extensions: List of STAC extensions to use
        include_checksum: Calculate and include file checksums
        checksum_algorithm: Hash algorithm for checksums
        include_file_size: Include file size in assets
        datetime_precision: Datetime string precision
        self_link: Include self link
        root_link: Root catalog link
        parent_link: Parent collection link
    """

    extensions: List[str] = field(default_factory=lambda: ["proj", "processing"])
    include_checksum: bool = True
    checksum_algorithm: str = "sha256"
    include_file_size: bool = True
    datetime_precision: str = "milliseconds"
    self_link: Optional[str] = None
    root_link: Optional[str] = None
    parent_link: Optional[str] = None

    def get_extension_schemas(self) -> List[str]:
        """Get list of extension schema URLs."""
        return [
            STAC_EXTENSIONS[ext]
            for ext in self.extensions
            if ext in STAC_EXTENSIONS
        ]


@dataclass
class STACItem:
    """
    STAC Item representation.

    Represents a single spatiotemporal asset with metadata.
    """

    id: str
    geometry: Dict[str, Any]
    bbox: Tuple[float, float, float, float]
    datetime: datetime
    properties: Dict[str, Any]
    assets: Dict[str, AssetInfo]
    links: List[Dict[str, str]] = field(default_factory=list)
    stac_extensions: List[str] = field(default_factory=list)
    collection: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to STAC Item dictionary."""
        item = {
            "type": "Feature",
            "stac_version": STAC_VERSION,
            "stac_extensions": self.stac_extensions,
            "id": self.id,
            "geometry": self.geometry,
            "bbox": list(self.bbox),
            "properties": {
                "datetime": self.datetime.isoformat().replace("+00:00", "Z"),
                **self.properties,
            },
            "links": self.links,
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
        }
        if self.collection:
            item["collection"] = self.collection
        item.update(self.extra_fields)
        return item

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class STACItemGenerator:
    """
    STAC Item metadata generator.

    Creates STAC Item metadata from raster files, combining file
    information with user-provided metadata and optional extensions.

    Example:
        generator = STACItemGenerator(STACItemConfig(
            extensions=["eo", "proj", "processing"]
        ))

        # Generate from COG file
        item = generator.generate_from_raster(
            raster_path="output.tif",
            item_id="sentinel2-scene-123",
            datetime=datetime(2024, 9, 15, 10, 30),
            collection="sentinel-2-l2a"
        )

        # Add EO bands
        item = generator.add_eo_bands(item, [
            BandInfo("B02", "blue", center_wavelength=0.490),
            BandInfo("B03", "green", center_wavelength=0.560),
            BandInfo("B04", "red", center_wavelength=0.665),
            BandInfo("B08", "nir", center_wavelength=0.842),
        ])
    """

    def __init__(self, config: Optional[STACItemConfig] = None):
        """
        Initialize STAC Item generator.

        Args:
            config: STAC Item configuration (uses defaults if None)
        """
        self.config = config or STACItemConfig()

    def generate_from_raster(
        self,
        raster_path: Union[str, Path],
        item_id: str,
        dt: Optional[datetime] = None,
        collection: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        asset_roles: Optional[List[AssetRole]] = None,
    ) -> STACItem:
        """
        Generate a STAC Item from a raster file.

        Args:
            raster_path: Path to raster file
            item_id: Unique identifier for the item
            dt: Datetime for the item (uses file mtime if None)
            collection: Parent collection ID
            properties: Additional properties
            asset_roles: Roles for the main asset

        Returns:
            STACItem with metadata from raster

        Raises:
            FileNotFoundError: If raster file doesn't exist
            ImportError: If rasterio is not available
        """
        try:
            import rasterio
            from rasterio.crs import CRS
        except ImportError:
            raise ImportError("rasterio is required for raster metadata extraction")

        raster_path = Path(raster_path)
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        logger.info(f"Generating STAC Item from: {raster_path}")

        with rasterio.open(raster_path) as src:
            # Get bounds and convert to WGS84 if needed
            bounds = src.bounds
            src_crs = src.crs

            if src_crs and not src_crs.is_geographic:
                from rasterio.warp import transform_bounds

                wgs84 = CRS.from_epsg(4326)
                bounds = transform_bounds(src_crs, wgs84, *bounds)

            bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)

            # Create geometry from bounds
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]],
            }

            # Get datetime
            if dt is None:
                mtime = raster_path.stat().st_mtime
                dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            elif dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            # Build properties
            item_props = properties.copy() if properties else {}

            # Add projection extension properties
            if "proj" in self.config.extensions:
                item_props["proj:epsg"] = src_crs.to_epsg() if src_crs else None
                item_props["proj:shape"] = [src.height, src.width]
                item_props["proj:transform"] = list(src.transform)[:6]
                if src_crs:
                    item_props["proj:wkt2"] = src_crs.to_wkt()

            # Add processing extension
            if "processing" in self.config.extensions:
                item_props["processing:software"] = {
                    "multiverse_dive": "1.0.0"
                }
                item_props["processing:level"] = "L2A"

            # Determine media type
            driver = src.driver
            if driver == "GTiff":
                # Check for COG profile
                if src.profile.get("tiled"):
                    media_type = MediaType.COG
                else:
                    media_type = MediaType.GEOTIFF
            else:
                media_type = MediaType.GEOTIFF

            # Create raster band info
            raster_bands = []
            for i in range(1, src.count + 1):
                band_info = RasterBandInfo(
                    data_type=str(src.dtypes[i - 1]),
                    nodata=src.nodatavals[i - 1],
                )
                raster_bands.append(band_info)

        # Create main asset
        asset_href = str(raster_path.absolute())
        main_asset = AssetInfo(
            href=asset_href,
            title=raster_path.name,
            type=media_type,
            roles=asset_roles or [AssetRole.DATA],
        )

        # Add raster extension bands
        if "raster" in self.config.extensions:
            main_asset.raster_bands = raster_bands

        # Add file info
        if self.config.include_file_size:
            main_asset.extra_fields["file:size"] = raster_path.stat().st_size

        if self.config.include_checksum:
            checksum = self._calculate_checksum(raster_path)
            main_asset.extra_fields[f"file:checksum"] = checksum

        # Build links
        links = []
        if self.config.self_link:
            links.append({
                "rel": "self",
                "href": self.config.self_link,
                "type": "application/geo+json",
            })
        if self.config.root_link:
            links.append({
                "rel": "root",
                "href": self.config.root_link,
                "type": "application/json",
            })
        if self.config.parent_link:
            links.append({
                "rel": "parent",
                "href": self.config.parent_link,
                "type": "application/json",
            })
        if collection:
            links.append({
                "rel": "collection",
                "href": f"../collection.json",
                "type": "application/json",
            })

        return STACItem(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=dt,
            properties=item_props,
            assets={"data": main_asset},
            links=links,
            stac_extensions=self.config.get_extension_schemas(),
            collection=collection,
        )

    def generate_from_zarr(
        self,
        zarr_path: Union[str, Path],
        item_id: str,
        dt: Optional[datetime] = None,
        datetime_range: Optional[Tuple[datetime, datetime]] = None,
        collection: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> STACItem:
        """
        Generate a STAC Item from a Zarr store.

        Args:
            zarr_path: Path to Zarr store
            item_id: Unique identifier for the item
            dt: Datetime for the item
            datetime_range: Optional datetime range (start, end)
            collection: Parent collection ID
            properties: Additional properties

        Returns:
            STACItem with metadata from Zarr store
        """
        try:
            import zarr
            import xarray as xr
        except ImportError:
            raise ImportError("zarr and xarray are required for Zarr metadata extraction")

        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        logger.info(f"Generating STAC Item from Zarr: {zarr_path}")

        # Open Zarr store
        ds = xr.open_zarr(zarr_path)

        # Get bounds from coordinates
        if "x" in ds.coords and "y" in ds.coords:
            x_vals = ds.coords["x"].values
            y_vals = ds.coords["y"].values
            bbox = (float(x_vals.min()), float(y_vals.min()),
                    float(x_vals.max()), float(y_vals.max()))
        else:
            bbox = (-180.0, -90.0, 180.0, 90.0)

        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]],
        }

        # Handle datetime
        if datetime_range:
            start_dt, end_dt = datetime_range
            item_props = {
                "start_datetime": start_dt.isoformat().replace("+00:00", "Z"),
                "end_datetime": end_dt.isoformat().replace("+00:00", "Z"),
            }
            dt = None  # Set to None when using range
        else:
            item_props = {}
            if dt is None:
                if "time" in ds.coords and len(ds.coords["time"]) > 0:
                    # Use first time value
                    import pandas as pd
                    first_time = pd.Timestamp(ds.coords["time"].values[0])
                    dt = first_time.to_pydatetime()
                else:
                    dt = datetime.now(timezone.utc)

        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if properties:
            item_props.update(properties)

        # Add zarr-specific properties
        item_props["zarr:chunks"] = {
            var: list(ds[var].encoding.get("chunks", ds[var].shape))
            for var in ds.data_vars
        }
        item_props["zarr:dimensions"] = dict(ds.dims)

        # Create main asset
        asset_href = str(zarr_path.absolute())
        main_asset = AssetInfo(
            href=asset_href,
            title=zarr_path.name,
            type=MediaType.ZARR,
            roles=[AssetRole.DATA],
        )

        if self.config.include_file_size:
            # Calculate total store size
            total_size = sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            )
            main_asset.extra_fields["file:size"] = total_size

        # Build links
        links = []
        if collection:
            links.append({
                "rel": "collection",
                "href": f"../collection.json",
                "type": "application/json",
            })

        return STACItem(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=dt or datetime.now(timezone.utc),  # Default if None
            properties=item_props,
            assets={"data": main_asset},
            links=links,
            stac_extensions=self.config.get_extension_schemas(),
            collection=collection,
        )

    def add_eo_bands(
        self,
        item: STACItem,
        bands: List[BandInfo],
        asset_key: str = "data",
    ) -> STACItem:
        """
        Add EO extension band information to an item.

        Args:
            item: STAC Item to modify
            bands: List of band information
            asset_key: Key of asset to associate bands with

        Returns:
            Modified STAC Item
        """
        # Add eo:bands to properties
        item.properties["eo:bands"] = [b.to_dict() for b in bands]

        # Link asset to bands by index
        if asset_key in item.assets:
            item.assets[asset_key].eo_bands = list(range(len(bands)))

        # Ensure EO extension is listed
        eo_schema = STAC_EXTENSIONS.get("eo")
        if eo_schema and eo_schema not in item.stac_extensions:
            item.stac_extensions.append(eo_schema)

        return item

    def add_sar_properties(
        self,
        item: STACItem,
        instrument_mode: str,
        frequency_band: str,
        polarizations: List[str],
        product_type: str,
        resolution_range: Optional[float] = None,
        resolution_azimuth: Optional[float] = None,
        looks_range: Optional[int] = None,
        looks_azimuth: Optional[int] = None,
    ) -> STACItem:
        """
        Add SAR extension properties to an item.

        Args:
            item: STAC Item to modify
            instrument_mode: SAR instrument mode (IW, EW, SM, etc.)
            frequency_band: Frequency band (C, L, X, etc.)
            polarizations: List of polarizations (VV, VH, HH, HV)
            product_type: Product type (GRD, SLC, etc.)
            resolution_range: Range resolution in meters
            resolution_azimuth: Azimuth resolution in meters
            looks_range: Number of range looks
            looks_azimuth: Number of azimuth looks

        Returns:
            Modified STAC Item
        """
        item.properties["sar:instrument_mode"] = instrument_mode
        item.properties["sar:frequency_band"] = frequency_band
        item.properties["sar:polarizations"] = polarizations
        item.properties["sar:product_type"] = product_type

        if resolution_range is not None:
            item.properties["sar:resolution_range"] = resolution_range
        if resolution_azimuth is not None:
            item.properties["sar:resolution_azimuth"] = resolution_azimuth
        if looks_range is not None:
            item.properties["sar:looks_range"] = looks_range
        if looks_azimuth is not None:
            item.properties["sar:looks_azimuth"] = looks_azimuth

        # Ensure SAR extension is listed
        sar_schema = STAC_EXTENSIONS.get("sar")
        if sar_schema and sar_schema not in item.stac_extensions:
            item.stac_extensions.append(sar_schema)

        return item

    def add_asset(
        self,
        item: STACItem,
        key: str,
        href: str,
        title: Optional[str] = None,
        media_type: Optional[MediaType] = None,
        roles: Optional[List[AssetRole]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> STACItem:
        """
        Add an asset to an item.

        Args:
            item: STAC Item to modify
            key: Asset key
            href: Asset URL or path
            title: Asset title
            media_type: Asset media type
            roles: Asset roles
            extra_fields: Additional asset fields

        Returns:
            Modified STAC Item
        """
        asset = AssetInfo(
            href=href,
            title=title,
            type=media_type,
            roles=roles or [],
            extra_fields=extra_fields or {},
        )
        item.assets[key] = asset
        return item

    def write_item(
        self,
        item: STACItem,
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> Path:
        """
        Write a STAC Item to a JSON file.

        Args:
            item: STAC Item to write
            output_path: Output file path
            indent: JSON indentation

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(item.to_json(indent=indent))

        logger.info(f"STAC Item written: {output_path}")
        return output_path

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        algo = self.config.checksum_algorithm
        hasher = hashlib.new(algo)

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        # Format: multihash-style prefix
        digest = hasher.hexdigest()
        return f"{algo}-{digest}"


def generate_stac_item(
    raster_path: Union[str, Path],
    item_id: str,
    dt: Optional[datetime] = None,
    collection: Optional[str] = None,
) -> STACItem:
    """
    Convenience function to generate a STAC Item from a raster file.

    Args:
        raster_path: Path to raster file
        item_id: Unique identifier for the item
        dt: Datetime for the item
        collection: Parent collection ID

    Returns:
        STACItem with metadata from raster
    """
    generator = STACItemGenerator()
    return generator.generate_from_raster(raster_path, item_id, dt, collection)
