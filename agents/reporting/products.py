"""
Product Generation for Reporting Agent.

Handles generation of all output product types:
- GeoTIFF/COG raster products
- GeoJSON vector products
- PDF/HTML QA reports
- Provenance documents
- Thumbnail/preview images
- Metadata sidecar files

Uses existing infrastructure from:
- core/data/ingestion/formats/ for COG, Zarr, GeoParquet, STAC
- core/quality/reporting/ for QA reports
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ThumbnailColormap(Enum):
    """Colormaps for thumbnail generation."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    BLUES = "blues"
    REDS = "reds"
    BINARY = "binary"


@dataclass
class GeoTIFFResult:
    """Result of GeoTIFF generation."""

    width: int
    height: int
    bands: int
    dtype: str
    crs: Optional[str]
    bounds: Optional[Tuple[float, float, float, float]]
    compression: str
    cog: bool
    overview_levels: List[int]
    file_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": self.bounds,
            "compression": self.compression,
            "cog": self.cog,
            "overview_levels": self.overview_levels,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class GeoJSONResult:
    """Result of GeoJSON generation."""

    feature_count: int
    geometry_types: List[str]
    crs: str
    bbox: Optional[Tuple[float, float, float, float]]
    properties_schema: Dict[str, str]
    file_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_count": self.feature_count,
            "geometry_types": self.geometry_types,
            "crs": self.crs,
            "bbox": self.bbox,
            "properties_schema": self.properties_schema,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class COGConfig:
    """Configuration for COG generation."""

    blocksize: int = 512
    compression: str = "deflate"
    predictor: int = 2  # Horizontal differencing
    quality: int = 75  # For lossy compression
    overview_factors: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    overview_resampling: str = "average"
    bigtiff: bool = False

    def to_creation_options(self) -> Dict[str, Any]:
        """Convert to GDAL/rasterio creation options."""
        options = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": self.blocksize,
            "blockysize": self.blocksize,
            "compress": self.compression.upper(),
            "interleave": "band",
        }

        if self.compression.lower() in ("deflate", "lzw", "zstd"):
            options["predictor"] = self.predictor

        if self.compression.lower() in ("jpeg", "webp"):
            options["quality"] = self.quality

        if self.bigtiff:
            options["bigtiff"] = "yes"

        return options


class ProductGenerator:
    """
    Generator for output products.

    Handles creation of GeoTIFF/COG rasters, GeoJSON vectors,
    QA reports, and thumbnail images.

    Example:
        generator = ProductGenerator(output_dir=Path("/output"))

        # Generate COG
        result = generator.generate_geotiff(
            data=flood_extent,
            output_path=Path("flood_extent.tif"),
            metadata={"crs": "EPSG:4326", "transform": transform},
            as_cog=True,
        )

        # Generate GeoJSON
        result = generator.generate_geojson(
            features=features,
            output_path=Path("flood_extent.geojson"),
        )

        # Generate thumbnail
        thumbnail = generator.generate_thumbnail(data=flood_extent, size=(512, 512))
        generator.save_thumbnail(thumbnail, Path("thumbnail.png"))
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cog_config: Optional[COGConfig] = None,
    ):
        """
        Initialize ProductGenerator.

        Args:
            output_dir: Default output directory
            cog_config: COG generation configuration
        """
        self.output_dir = output_dir or Path("/tmp/multiverse_dive/products")
        self.cog_config = cog_config or COGConfig()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_geotiff(
        self,
        data: np.ndarray,
        output_path: Path,
        metadata: Dict[str, Any],
        as_cog: bool = True,
        compression: str = "deflate",
        nodata: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a GeoTIFF/COG product.

        Args:
            data: Raster data (2D or 3D array)
            output_path: Output file path
            metadata: Metadata including transform and CRS
            as_cog: Generate as Cloud-Optimized GeoTIFF
            compression: Compression method
            nodata: NoData value

        Returns:
            GeoTIFFResult as dictionary
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize data shape
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        bands, height, width = data.shape

        # Get or create transform and CRS
        transform = metadata.get("transform")
        crs = metadata.get("crs", "EPSG:4326")

        if transform is None:
            # Create default transform if not provided
            bounds = metadata.get("bounds", [-180, -90, 180, 90])
            transform = self._create_transform(bounds, width, height)

        # Build profile
        profile = self._build_geotiff_profile(
            width=width,
            height=height,
            bands=bands,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            as_cog=as_cog,
            compression=compression,
            nodata=nodata,
        )

        # Write file
        self._write_geotiff(data, output_path, profile, as_cog)

        # Calculate bounds
        bounds = self._calculate_bounds(transform, width, height)

        return GeoTIFFResult(
            width=width,
            height=height,
            bands=bands,
            dtype=str(data.dtype),
            crs=str(crs),
            bounds=bounds,
            compression=compression,
            cog=as_cog,
            overview_levels=self.cog_config.overview_factors if as_cog else [],
            file_size_bytes=output_path.stat().st_size,
        ).to_dict()

    def _create_transform(
        self,
        bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
    ):
        """Create affine transform from bounds."""
        try:
            from affine import Affine
        except ImportError:
            # Create simple transform tuple
            west, south, east, north = bounds
            pixel_width = (east - west) / width
            pixel_height = (south - north) / height  # Negative for north-up
            return (pixel_width, 0.0, west, 0.0, pixel_height, north)

        west, south, east, north = bounds
        pixel_width = (east - west) / width
        pixel_height = (north - south) / height
        return Affine.translation(west, north) * Affine.scale(pixel_width, -pixel_height)

    def _build_geotiff_profile(
        self,
        width: int,
        height: int,
        bands: int,
        dtype: np.dtype,
        crs: str,
        transform: Any,
        as_cog: bool,
        compression: str,
        nodata: Optional[float],
    ) -> Dict[str, Any]:
        """Build GeoTIFF profile."""
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": bands,
            "dtype": str(dtype),
            "crs": crs,
            "transform": transform,
        }

        if nodata is not None:
            profile["nodata"] = nodata

        if as_cog:
            config = COGConfig(compression=compression)
            profile.update(config.to_creation_options())

        return profile

    def _write_geotiff(
        self,
        data: np.ndarray,
        output_path: Path,
        profile: Dict[str, Any],
        as_cog: bool,
    ):
        """Write GeoTIFF file."""
        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

                # Build overviews for COG
                if as_cog and self.cog_config.overview_factors:
                    resampling = getattr(
                        Resampling,
                        self.cog_config.overview_resampling.upper(),
                        Resampling.AVERAGE,
                    )
                    dst.build_overviews(self.cog_config.overview_factors, resampling)
                    dst.update_tags(ns="rio_overview", resampling=resampling.name)

            logger.info(f"Written GeoTIFF: {output_path}")

        except ImportError:
            # Fallback: write minimal TIFF without rasterio
            logger.warning("rasterio not available, writing minimal TIFF")
            self._write_minimal_tiff(data, output_path, profile)

    def _write_minimal_tiff(
        self,
        data: np.ndarray,
        output_path: Path,
        profile: Dict[str, Any],
    ):
        """Write minimal TIFF without rasterio (fallback)."""
        try:
            from PIL import Image

            # Normalize data to uint8 for simple TIFF
            if data.dtype != np.uint8:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                if data_max > data_min:
                    data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)

            # Handle multi-band
            if data.ndim == 3:
                if data.shape[0] == 1:
                    data = data[0]
                elif data.shape[0] == 3:
                    data = np.moveaxis(data, 0, -1)
                else:
                    data = data[0]

            img = Image.fromarray(data)
            img.save(output_path)

        except ImportError:
            # Last resort: save as raw numpy
            np.save(str(output_path).replace(".tif", ".npy"), data)
            logger.warning(f"Saved as numpy array: {output_path}")

    def _calculate_bounds(
        self,
        transform: Any,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Calculate bounds from transform."""
        try:
            if hasattr(transform, "c"):
                # Affine transform
                west = transform.c
                north = transform.f
                east = west + transform.a * width
                south = north + transform.e * height
                return (west, min(south, north), east, max(south, north))
            elif isinstance(transform, tuple) and len(transform) == 6:
                # Tuple transform
                pixel_width, _, west, _, pixel_height, north = transform
                east = west + pixel_width * width
                south = north + pixel_height * height
                return (west, min(south, north), east, max(south, north))
        except Exception as e:
            logger.warning(f"Could not calculate bounds: {e}")
        return None

    def generate_geojson(
        self,
        features: List[Dict[str, Any]],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """
        Generate a GeoJSON product.

        Args:
            features: List of GeoJSON feature dictionaries
            output_path: Output file path
            metadata: Additional metadata
            crs: Coordinate reference system

        Returns:
            GeoJSONResult as dictionary
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = metadata or {}

        # Build feature collection
        feature_collection = {
            "type": "FeatureCollection",
            "features": features,
        }

        # Add CRS if not WGS84
        if crs != "EPSG:4326":
            feature_collection["crs"] = {
                "type": "name",
                "properties": {"name": crs},
            }

        # Add metadata
        if metadata:
            feature_collection["properties"] = metadata

        # Calculate bbox
        bbox = self._calculate_geojson_bbox(features)
        if bbox:
            feature_collection["bbox"] = bbox

        # Write file
        with open(output_path, "w") as f:
            json.dump(feature_collection, f, indent=2)

        logger.info(f"Written GeoJSON: {output_path} ({len(features)} features)")

        # Collect geometry types
        geometry_types = list(set(
            f.get("geometry", {}).get("type", "Unknown")
            for f in features
            if f.get("geometry")
        ))

        # Build properties schema
        properties_schema = self._infer_properties_schema(features)

        return GeoJSONResult(
            feature_count=len(features),
            geometry_types=geometry_types,
            crs=crs,
            bbox=bbox,
            properties_schema=properties_schema,
            file_size_bytes=output_path.stat().st_size,
        ).to_dict()

    def _calculate_geojson_bbox(
        self,
        features: List[Dict[str, Any]],
    ) -> Optional[Tuple[float, float, float, float]]:
        """Calculate bounding box from features."""
        if not features:
            return None

        all_coords = []
        for feature in features:
            geometry = feature.get("geometry", {})
            coords = self._extract_coordinates(geometry)
            all_coords.extend(coords)

        if not all_coords:
            return None

        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        return (min(xs), min(ys), max(xs), max(ys))

    def _extract_coordinates(self, geometry: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract all coordinates from a geometry."""
        coords = []
        geom_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])

        if geom_type == "Point":
            coords.append(tuple(coordinates[:2]))
        elif geom_type == "MultiPoint":
            coords.extend(tuple(c[:2]) for c in coordinates)
        elif geom_type == "LineString":
            coords.extend(tuple(c[:2]) for c in coordinates)
        elif geom_type == "MultiLineString":
            for line in coordinates:
                coords.extend(tuple(c[:2]) for c in line)
        elif geom_type == "Polygon":
            for ring in coordinates:
                coords.extend(tuple(c[:2]) for c in ring)
        elif geom_type == "MultiPolygon":
            for polygon in coordinates:
                for ring in polygon:
                    coords.extend(tuple(c[:2]) for c in ring)
        elif geom_type == "GeometryCollection":
            for geom in geometry.get("geometries", []):
                coords.extend(self._extract_coordinates(geom))

        return coords

    def _infer_properties_schema(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Infer properties schema from features."""
        schema = {}
        for feature in features[:10]:  # Sample first 10
            props = feature.get("properties", {})
            for key, value in props.items():
                if key not in schema:
                    schema[key] = type(value).__name__
        return schema

    def raster_to_features(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        threshold: float = 0.5,
        simplify_tolerance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert raster data to GeoJSON features.

        Args:
            data: Raster data (2D array)
            metadata: Metadata with transform and CRS
            threshold: Threshold for binary classification
            simplify_tolerance: Tolerance for geometry simplification

        Returns:
            List of GeoJSON features
        """
        # Ensure 2D
        if data.ndim == 3:
            data = data[0]

        # Create binary mask
        mask = (data >= threshold).astype(np.uint8)

        # Get transform
        transform = metadata.get("transform")
        if transform is None:
            bounds = metadata.get("bounds", [-180, -90, 180, 90])
            transform = self._create_transform(bounds, data.shape[1], data.shape[0])

        try:
            return self._vectorize_with_rasterio(mask, transform, simplify_tolerance)
        except ImportError:
            return self._vectorize_simple(mask, metadata, threshold)

    def _vectorize_with_rasterio(
        self,
        mask: np.ndarray,
        transform: Any,
        simplify_tolerance: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Vectorize using rasterio/shapely."""
        import rasterio.features
        from shapely.geometry import shape, mapping

        features = []
        for geom, value in rasterio.features.shapes(mask, transform=transform):
            if value == 1:  # Positive mask
                geometry = shape(geom)

                if simplify_tolerance:
                    geometry = geometry.simplify(simplify_tolerance)

                feature = {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": {
                        "value": 1,
                        "area_m2": geometry.area if hasattr(geometry, "area") else 0,
                    },
                }
                features.append(feature)

        return features

    def _vectorize_simple(
        self,
        mask: np.ndarray,
        metadata: Dict[str, Any],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Simple vectorization without rasterio (fallback)."""
        # Find connected components
        from scipy import ndimage

        labeled, num_features = ndimage.label(mask)

        features = []
        bounds = metadata.get("bounds", [-180, -90, 180, 90])
        height, width = mask.shape
        pixel_width = (bounds[2] - bounds[0]) / width
        pixel_height = (bounds[3] - bounds[1]) / height

        for label_id in range(1, num_features + 1):
            # Find bounding box of component
            rows, cols = np.where(labeled == label_id)
            if len(rows) == 0:
                continue

            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Convert to coordinates
            west = bounds[0] + min_col * pixel_width
            east = bounds[0] + (max_col + 1) * pixel_width
            south = bounds[3] - (max_row + 1) * pixel_height
            north = bounds[3] - min_row * pixel_height

            # Create simple polygon (bounding box)
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]],
                },
                "properties": {
                    "label_id": int(label_id),
                    "pixel_count": int(len(rows)),
                },
            }
            features.append(feature)

        return features

    def generate_thumbnail(
        self,
        data: np.ndarray,
        size: Tuple[int, int] = (512, 512),
        colormap: ThumbnailColormap = ThumbnailColormap.VIRIDIS,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Generate thumbnail image from raster data.

        Args:
            data: Raster data (2D or 3D array)
            size: Output size (width, height)
            colormap: Colormap to apply
            background_color: Background color for nodata

        Returns:
            RGB thumbnail array (height, width, 3)
        """
        # Ensure 2D
        if data.ndim == 3:
            if data.shape[0] == 3:
                # Already RGB
                data = np.moveaxis(data, 0, -1)
            else:
                data = data[0]

        # Handle RGB data
        if data.ndim == 3 and data.shape[2] == 3:
            return self._resize_rgb(data, size)

        # Normalize to 0-1
        valid_mask = np.isfinite(data)
        if valid_mask.any():
            data_min = np.nanmin(data[valid_mask])
            data_max = np.nanmax(data[valid_mask])
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(data)
        else:
            normalized = np.zeros_like(data)

        # Apply colormap
        rgb = self._apply_colormap(normalized, colormap, valid_mask, background_color)

        # Resize
        return self._resize_rgb(rgb, size)

    def _apply_colormap(
        self,
        data: np.ndarray,
        colormap: ThumbnailColormap,
        valid_mask: np.ndarray,
        background_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Apply colormap to normalized data."""
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap.value)
            rgba = cmap(data)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            # Simple fallback colormap
            rgb = self._simple_colormap(data, colormap)

        # Apply background for invalid pixels
        for i in range(3):
            rgb[:, :, i] = np.where(valid_mask, rgb[:, :, i], background_color[i])

        return rgb

    def _simple_colormap(
        self,
        data: np.ndarray,
        colormap: ThumbnailColormap,
    ) -> np.ndarray:
        """Simple colormap without matplotlib."""
        # Simple gradient colormaps
        if colormap == ThumbnailColormap.BLUES:
            r = (255 - data * 200).astype(np.uint8)
            g = (255 - data * 150).astype(np.uint8)
            b = np.full_like(data, 255, dtype=np.uint8)
        elif colormap == ThumbnailColormap.REDS:
            r = np.full_like(data, 255, dtype=np.uint8)
            g = (255 - data * 200).astype(np.uint8)
            b = (255 - data * 200).astype(np.uint8)
        elif colormap == ThumbnailColormap.BINARY:
            val = (data * 255).astype(np.uint8)
            r = g = b = val
        else:
            # Default viridis-like
            r = (68 + data * 150).astype(np.uint8)
            g = (1 + data * 200).astype(np.uint8)
            b = (84 + data * 100).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)

    def _resize_rgb(
        self,
        data: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize RGB image."""
        try:
            from PIL import Image
            img = Image.fromarray(data.astype(np.uint8))
            img = img.resize(size, Image.Resampling.LANCZOS)
            return np.array(img)
        except ImportError:
            # Simple resize with numpy
            from scipy import ndimage
            height, width = data.shape[:2]
            target_width, target_height = size
            zoom_y = target_height / height
            zoom_x = target_width / width

            if data.ndim == 3:
                zoom = (zoom_y, zoom_x, 1)
            else:
                zoom = (zoom_y, zoom_x)

            return ndimage.zoom(data, zoom, order=1).astype(np.uint8)

    def save_thumbnail(
        self,
        thumbnail: np.ndarray,
        output_path: Path,
    ):
        """
        Save thumbnail to file.

        Args:
            thumbnail: RGB thumbnail array
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image
            img = Image.fromarray(thumbnail.astype(np.uint8))
            img.save(output_path)
        except ImportError:
            # Fallback to saving as numpy
            np.save(str(output_path).replace(".png", ".npy"), thumbnail)
            logger.warning(f"PIL not available, saved as numpy: {output_path}")

        logger.info(f"Saved thumbnail: {output_path}")

    def generate_qa_report(
        self,
        qa_report: Dict[str, Any],
        results: Dict[str, Any],
        output_path: Path,
        format: "OutputFormat",
    ):
        """
        Generate QA report in specified format.

        Args:
            qa_report: QA report data
            results: Analysis results
            output_path: Output file path
            format: Output format (HTML or PDF)
        """
        from agents.reporting.main import OutputFormat

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == OutputFormat.HTML:
            self._generate_html_report(qa_report, results, output_path)
        elif format == OutputFormat.PDF:
            self._generate_pdf_report(qa_report, results, output_path)
        else:
            # Default to JSON
            with open(output_path, "w") as f:
                json.dump(qa_report, f, indent=2)

        logger.info(f"Generated QA report: {output_path}")

    def _generate_html_report(
        self,
        qa_report: Dict[str, Any],
        results: Dict[str, Any],
        output_path: Path,
    ):
        """Generate HTML QA report."""
        # Try to use existing QA report generator
        try:
            from core.quality.reporting.qa_report import QAReportGenerator, ReportFormat

            generator = QAReportGenerator()
            report = generator.generate(
                event_id=qa_report.get("event_id", "unknown"),
                product_id=qa_report.get("product_id", "unknown"),
                sanity_result=results.get("sanity"),
                validation_result=results.get("validation"),
                uncertainty_result=results.get("uncertainty"),
                gating_decision=results.get("gating"),
            )
            report.save(output_path)
            return
        except ImportError:
            pass

        # Fallback: generate simple HTML
        html = self._build_simple_html_report(qa_report)

        with open(output_path, "w") as f:
            f.write(html)

    def _build_simple_html_report(self, qa_report: Dict[str, Any]) -> str:
        """Build simple HTML report."""
        status = qa_report.get("overall_status", "UNKNOWN")
        confidence = qa_report.get("confidence_score", 0)
        event_id = qa_report.get("event_id", "unknown")
        product_id = qa_report.get("product_id", "unknown")

        status_colors = {
            "PASS": "#28a745",
            "PASS_WITH_WARNINGS": "#ffc107",
            "REVIEW_REQUIRED": "#fd7e14",
            "BLOCKED": "#dc3545",
        }
        status_color = status_colors.get(status, "#6c757d")

        checks_html = ""
        for check in qa_report.get("checks", []):
            check_status = check.get("status", "unknown")
            checks_html += f"""
            <tr>
                <td>{check.get("check_name", "")}</td>
                <td>{check.get("category", "")}</td>
                <td class="{check_status}">{check_status.upper()}</td>
                <td>{check.get("details", "")}</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>QA Report: {product_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 4px; color: white; font-weight: bold; background-color: {status_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .pass {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .soft_fail {{ color: #fd7e14; }}
        .hard_fail {{ color: #dc3545; }}
    </style>
</head>
<body>
<div class="container">
    <h1>QA Report</h1>
    <p><strong>Event ID:</strong> {event_id}</p>
    <p><strong>Product ID:</strong> {product_id}</p>
    <p><strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

    <p>
        <span class="status-badge">{status}</span>
        &nbsp;&nbsp;
        <strong>Confidence:</strong> {confidence:.1%}
    </p>

    <h2>Quality Checks</h2>
    <table>
        <tr><th>Check</th><th>Category</th><th>Status</th><th>Details</th></tr>
        {checks_html}
    </table>
</div>
</body>
</html>"""

    def _generate_pdf_report(
        self,
        qa_report: Dict[str, Any],
        results: Dict[str, Any],
        output_path: Path,
    ):
        """Generate PDF QA report."""
        # First generate HTML
        html_path = output_path.with_suffix(".html")
        self._generate_html_report(qa_report, results, html_path)

        # Try to convert to PDF
        try:
            import weasyprint
            doc = weasyprint.HTML(filename=str(html_path))
            doc.write_pdf(str(output_path))
            html_path.unlink()  # Remove HTML
        except ImportError:
            # Keep HTML if PDF conversion not available
            logger.warning("weasyprint not available, keeping HTML report")
            html_path.rename(output_path)

    def generate_provenance(
        self,
        event_id: str,
        product_id: str,
        lineage: List[Dict[str, Any]],
        input_datasets: List[Dict[str, Any]],
        algorithms_used: List[Dict[str, Any]],
        quality_summary: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate provenance record following provenance.schema.json.

        Args:
            event_id: Event identifier
            product_id: Product identifier
            lineage: Processing chain
            input_datasets: Input datasets
            algorithms_used: Algorithms used
            quality_summary: Quality summary
            output_path: Output file path

        Returns:
            Provenance record as dictionary
        """
        now = datetime.now(timezone.utc)

        provenance = {
            "product_id": product_id,
            "event_id": event_id,
            "lineage": lineage,
            "input_datasets": input_datasets,
            "algorithms_used": algorithms_used,
            "quality_summary": quality_summary,
            "reproducibility": {
                "deterministic": True,
                "selection_hash": self._compute_selection_hash(lineage, algorithms_used),
                "environment_hash": self._compute_environment_hash(),
            },
            "metadata": {
                "created_at": now.isoformat(),
                "created_by": "reporting_agent",
                "total_processing_time_seconds": sum(
                    step.get("execution_time_seconds", 0) for step in lineage
                ),
            },
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(provenance, f, indent=2)
            logger.info(f"Generated provenance: {output_path}")

        return provenance

    def _compute_selection_hash(
        self,
        lineage: List[Dict[str, Any]],
        algorithms: List[Dict[str, Any]],
    ) -> str:
        """Compute hash for selection reproducibility."""
        data = {
            "steps": [s.get("step") for s in lineage],
            "algorithms": [a.get("algorithm_id") for a in algorithms],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def _compute_environment_hash(self) -> str:
        """Compute hash of software environment."""
        import platform
        import sys

        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }

        try:
            import numpy
            env["numpy"] = numpy.__version__
        except ImportError:
            pass

        return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]

    def generate_metadata_sidecar(
        self,
        product_path: Path,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Generate metadata sidecar file.

        Args:
            product_path: Path to product file
            metadata: Metadata to write

        Returns:
            Path to sidecar file
        """
        sidecar_path = product_path.with_suffix(product_path.suffix + ".json")

        with open(sidecar_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Generated metadata sidecar: {sidecar_path}")
        return sidecar_path
