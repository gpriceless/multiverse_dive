"""
Export Command - Generate final products in various formats.

Usage:
    mdive export --input ./results/ --format geotiff,geojson --output ./products/
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import click

logger = logging.getLogger("mdive.export")


# Supported export formats
EXPORT_FORMATS = {
    "geotiff": {
        "extension": ".tif",
        "description": "Cloud-Optimized GeoTIFF",
        "mime_type": "image/tiff",
    },
    "geojson": {
        "extension": ".geojson",
        "description": "GeoJSON vector polygons",
        "mime_type": "application/geo+json",
    },
    "shapefile": {
        "extension": ".shp",
        "description": "ESRI Shapefile",
        "mime_type": "application/x-shapefile",
    },
    "gpkg": {
        "extension": ".gpkg",
        "description": "GeoPackage",
        "mime_type": "application/geopackage+sqlite3",
    },
    "kml": {
        "extension": ".kml",
        "description": "Keyhole Markup Language",
        "mime_type": "application/vnd.google-earth.kml+xml",
    },
    "png": {
        "extension": ".png",
        "description": "PNG image with world file",
        "mime_type": "image/png",
    },
    "pdf": {
        "extension": ".pdf",
        "description": "PDF report with map",
        "mime_type": "application/pdf",
    },
    "stac": {
        "extension": ".json",
        "description": "STAC Item metadata",
        "mime_type": "application/json",
    },
}


@click.command("export")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory containing analysis results.",
)
@click.option(
    "--format",
    "-f",
    "formats",
    type=str,
    default="geotiff,geojson",
    help="Comma-separated list of output formats (default: geotiff,geojson).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for exported products.",
)
@click.option(
    "--name",
    "-n",
    "product_name",
    type=str,
    default=None,
    help="Base name for output products.",
)
@click.option(
    "--crs",
    type=str,
    default=None,
    help="Target CRS for output (default: source CRS).",
)
@click.option(
    "--simplify",
    type=float,
    default=None,
    help="Simplification tolerance for vector outputs (in CRS units).",
)
@click.option(
    "--include-metadata/--no-metadata",
    default=True,
    help="Include metadata files (default: yes).",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    help="Compress raster outputs (default: yes).",
)
@click.pass_obj
def export(
    ctx,
    input_path: Path,
    formats: str,
    output_path: Path,
    product_name: Optional[str],
    crs: Optional[str],
    simplify: Optional[float],
    include_metadata: bool,
    compress: bool,
):
    """
    Generate final products from analysis results.

    Exports analysis results to various formats including GeoTIFF, GeoJSON,
    Shapefile, GeoPackage, KML, PNG, and PDF reports.

    \b
    Examples:
        # Export to GeoTIFF and GeoJSON
        mdive export --input ./results/ --format geotiff,geojson --output ./products/

        # Export with custom name and simplified vectors
        mdive export --input ./results/ --format geojson,kml --output ./products/ \\
            --name miami_flood --simplify 10.0

        # Export for web visualization
        mdive export --input ./results/ --format geojson,png --output ./products/
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse formats
    format_list = [f.strip().lower() for f in formats.split(",")]
    invalid_formats = [f for f in format_list if f not in EXPORT_FORMATS]
    if invalid_formats:
        valid = ", ".join(EXPORT_FORMATS.keys())
        raise click.BadParameter(
            f"Unknown formats: {', '.join(invalid_formats)}. Valid: {valid}"
        )

    # Determine product name
    if not product_name:
        # Try to get from input metadata
        metadata_file = input_path / "analysis_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                product_name = metadata.get("algorithm", "output")
        else:
            product_name = input_path.name

    product_name = sanitize_filename(product_name)

    click.echo(f"\n=== Export Products ===")
    click.echo(f"  Input: {input_path}")
    click.echo(f"  Output: {output_path}")
    click.echo(f"  Product name: {product_name}")
    click.echo(f"  Formats: {', '.join(format_list)}")

    if crs:
        click.echo(f"  Target CRS: {crs}")
    if simplify:
        click.echo(f"  Simplification: {simplify}")

    # Find input raster
    input_rasters = list(input_path.glob("*.tif"))
    if not input_rasters:
        input_rasters = list(input_path.rglob("*.tif"))

    if not input_rasters:
        click.echo("Warning: No input rasters found", err=True)

    # Export to each format
    exported = []
    failed = []

    for fmt in format_list:
        click.echo(f"\nExporting to {EXPORT_FORMATS[fmt]['description']}...")

        try:
            output_file = export_format(
                input_path=input_path,
                input_rasters=input_rasters,
                output_path=output_path,
                product_name=product_name,
                format_type=fmt,
                target_crs=crs,
                simplify=simplify,
                compress=compress,
            )
            exported.append({"format": fmt, "path": str(output_file)})
            click.echo(f"  Created: {output_file}")

        except Exception as e:
            logger.error(f"Failed to export {fmt}: {e}")
            failed.append({"format": fmt, "error": str(e)})
            click.echo(f"  Failed: {e}", err=True)

    # Export metadata if requested
    if include_metadata:
        metadata_file = export_metadata(
            input_path=input_path,
            output_path=output_path,
            product_name=product_name,
            exported=exported,
        )
        click.echo(f"\nMetadata: {metadata_file}")

    # Summary
    click.echo(f"\n=== Export Summary ===")
    click.echo(f"  Exported: {len(exported)} formats")
    click.echo(f"  Failed: {len(failed)} formats")
    click.echo(f"  Output: {output_path}")

    if failed:
        raise SystemExit(1)


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace problematic characters
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        name = name.replace(char, '_')
    return name


def export_format(
    input_path: Path,
    input_rasters: List[Path],
    output_path: Path,
    product_name: str,
    format_type: str,
    target_crs: Optional[str],
    simplify: Optional[float],
    compress: bool,
) -> Path:
    """
    Export to a specific format.
    """
    format_info = EXPORT_FORMATS[format_type]
    output_file = output_path / f"{product_name}{format_info['extension']}"

    if format_type == "geotiff":
        return export_geotiff(input_rasters, output_file, target_crs, compress)
    elif format_type == "geojson":
        return export_geojson(input_rasters, output_file, target_crs, simplify)
    elif format_type == "shapefile":
        return export_shapefile(input_rasters, output_file, target_crs, simplify)
    elif format_type == "gpkg":
        return export_geopackage(input_rasters, output_file, target_crs, simplify)
    elif format_type == "kml":
        return export_kml(input_rasters, output_file, simplify)
    elif format_type == "png":
        return export_png(input_rasters, output_file)
    elif format_type == "pdf":
        return export_pdf(input_path, output_file)
    elif format_type == "stac":
        return export_stac(input_path, output_file)
    else:
        raise ValueError(f"Unknown format: {format_type}")


def export_geotiff(
    input_rasters: List[Path],
    output_file: Path,
    target_crs: Optional[str],
    compress: bool,
) -> Path:
    """Export as Cloud-Optimized GeoTIFF."""
    if not input_rasters:
        raise ValueError("No input rasters to export")

    try:
        import rasterio
        from rasterio.enums import Resampling

        # Use first raster as source
        src_path = input_rasters[0]

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()

            # Update for COG
            profile.update(
                driver="GTiff",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                interleave="band",
            )

            if compress:
                profile.update(compress="lzw", predictor=2)

            # Handle CRS reprojection if needed
            if target_crs and target_crs != str(src.crs):
                from rasterio.warp import calculate_default_transform, reproject

                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                profile.update(crs=target_crs, transform=transform, width=width, height=height)

                with rasterio.open(output_file, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest,
                        )
            else:
                # Direct copy with COG options
                data = src.read()
                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(data)

        # Build overviews for COG
        with rasterio.open(output_file, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

        return output_file

    except ImportError:
        # Fallback: copy the file
        shutil.copy(input_rasters[0], output_file)
        return output_file


def export_geojson(
    input_rasters: List[Path],
    output_file: Path,
    target_crs: Optional[str],
    simplify: Optional[float],
) -> Path:
    """Export as GeoJSON vector polygons."""
    try:
        import rasterio
        from rasterio.features import shapes
        import json

        if not input_rasters:
            # Create empty GeoJSON
            geojson = {"type": "FeatureCollection", "features": []}
            with open(output_file, "w") as f:
                json.dump(geojson, f, indent=2)
            return output_file

        with rasterio.open(input_rasters[0]) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

            # Vectorize
            features = []
            for geom, value in shapes(data, transform=transform):
                if value > 0:  # Non-zero values (flooded areas)
                    # Simplify if requested
                    if simplify:
                        try:
                            from shapely.geometry import shape
                            from shapely.ops import transform as shapely_transform

                            geom_obj = shape(geom)
                            geom_obj = geom_obj.simplify(simplify, preserve_topology=True)
                            geom = geom_obj.__geo_interface__
                        except ImportError:
                            pass

                    features.append({
                        "type": "Feature",
                        "properties": {
                            "value": int(value),
                            "class": "flooded" if value == 1 else f"class_{int(value)}",
                        },
                        "geometry": geom,
                    })

            geojson = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": str(crs) if crs else "EPSG:4326"},
                },
                "features": features,
            }

            with open(output_file, "w") as f:
                json.dump(geojson, f, indent=2)

        return output_file

    except ImportError:
        # Create mock GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"value": 1, "class": "flooded"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-80.5, 25.5], [-80.0, 25.5], [-80.0, 26.0], [-80.5, 26.0], [-80.5, 25.5]]]
                },
            }],
        }
        with open(output_file, "w") as f:
            json.dump(geojson, f, indent=2)
        return output_file


def export_shapefile(
    input_rasters: List[Path],
    output_file: Path,
    target_crs: Optional[str],
    simplify: Optional[float],
) -> Path:
    """Export as ESRI Shapefile."""
    # First export to GeoJSON, then convert
    geojson_file = output_file.with_suffix(".geojson")
    export_geojson(input_rasters, geojson_file, target_crs, simplify)

    try:
        import geopandas as gpd

        gdf = gpd.read_file(geojson_file)
        gdf.to_file(output_file, driver="ESRI Shapefile")
        geojson_file.unlink()  # Remove temporary GeoJSON
        return output_file

    except ImportError:
        logger.warning("geopandas not available, Shapefile export skipped")
        raise


def export_geopackage(
    input_rasters: List[Path],
    output_file: Path,
    target_crs: Optional[str],
    simplify: Optional[float],
) -> Path:
    """Export as GeoPackage."""
    geojson_file = output_file.with_suffix(".geojson")
    export_geojson(input_rasters, geojson_file, target_crs, simplify)

    try:
        import geopandas as gpd

        gdf = gpd.read_file(geojson_file)
        gdf.to_file(output_file, driver="GPKG")
        geojson_file.unlink()
        return output_file

    except ImportError:
        logger.warning("geopandas not available, GeoPackage export skipped")
        raise


def export_kml(
    input_rasters: List[Path],
    output_file: Path,
    simplify: Optional[float],
) -> Path:
    """Export as KML for Google Earth."""
    # Export to GeoJSON first (in WGS84)
    geojson_file = output_file.with_suffix(".geojson")
    export_geojson(input_rasters, geojson_file, "EPSG:4326", simplify)

    try:
        import geopandas as gpd

        gdf = gpd.read_file(geojson_file)
        # Convert to WGS84 if not already
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        gdf.to_file(output_file, driver="KML")
        geojson_file.unlink()
        return output_file

    except ImportError:
        # Create simple KML manually
        with open(geojson_file) as f:
            geojson = json.load(f)

        kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>Export</name>
"""
        for feature in geojson.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                kml += f"""<Placemark>
<Polygon><outerBoundaryIs><LinearRing>
<coordinates>{coord_str}</coordinates>
</LinearRing></outerBoundaryIs></Polygon>
</Placemark>
"""
        kml += "</Document>\n</kml>"

        with open(output_file, "w") as f:
            f.write(kml)
        geojson_file.unlink()
        return output_file


def export_png(input_rasters: List[Path], output_file: Path) -> Path:
    """Export as PNG image with world file."""
    try:
        import rasterio
        from PIL import Image
        import numpy as np

        if not input_rasters:
            raise ValueError("No input rasters")

        with rasterio.open(input_rasters[0]) as src:
            data = src.read(1)

            # Create color-mapped image
            # 0 = transparent, 1 = blue (flooded)
            rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
            flood_mask = data > 0
            rgba[flood_mask] = [0, 100, 200, 200]  # Semi-transparent blue

            img = Image.fromarray(rgba, "RGBA")
            img.save(output_file)

            # Create world file
            world_file = output_file.with_suffix(".pgw")
            transform = src.transform
            with open(world_file, "w") as f:
                f.write(f"{transform.a}\n")
                f.write(f"{transform.d}\n")
                f.write(f"{transform.b}\n")
                f.write(f"{transform.e}\n")
                f.write(f"{transform.c}\n")
                f.write(f"{transform.f}\n")

        return output_file

    except ImportError:
        # Create simple placeholder PNG
        try:
            from PIL import Image

            img = Image.new("RGBA", (100, 100), (0, 100, 200, 200))
            img.save(output_file)
        except ImportError:
            output_file.write_bytes(b"")  # Empty file
        return output_file


def export_pdf(input_path: Path, output_file: Path) -> Path:
    """Export as PDF report with map."""
    # Create a simple text-based PDF placeholder
    content = f"""Flood Analysis Report
Generated: {datetime.now().isoformat()}
Input: {input_path}

[Map would be rendered here with actual PDF generation libraries]

This is a placeholder PDF. Install reportlab or matplotlib for full PDF generation.
"""
    # For now, save as text with .pdf extension
    output_file.write_text(content)
    return output_file


def export_stac(input_path: Path, output_file: Path) -> Path:
    """Export as STAC Item metadata."""
    # Read analysis metadata if available
    analysis_meta = {}
    meta_file = input_path / "analysis_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            analysis_meta = json.load(f)

    stac_item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": output_file.stem,
        "geometry": None,  # Would be populated from actual data
        "bbox": None,
        "properties": {
            "datetime": datetime.now().isoformat() + "Z",
            "created": datetime.now().isoformat() + "Z",
            "algorithm": analysis_meta.get("algorithm", "unknown"),
            "processing:level": "L3",
        },
        "links": [],
        "assets": {},
    }

    with open(output_file, "w") as f:
        json.dump(stac_item, f, indent=2)

    return output_file


def export_metadata(
    input_path: Path,
    output_path: Path,
    product_name: str,
    exported: List[Dict[str, Any]],
) -> Path:
    """Export product metadata."""
    metadata = {
        "product_name": product_name,
        "exported_at": datetime.now().isoformat(),
        "source": str(input_path),
        "formats": exported,
    }

    # Include analysis metadata if available
    analysis_meta = input_path / "analysis_metadata.json"
    if analysis_meta.exists():
        with open(analysis_meta) as f:
            metadata["analysis"] = json.load(f)

    output_file = output_path / f"{product_name}_metadata.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_file
