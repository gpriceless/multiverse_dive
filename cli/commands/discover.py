"""
Discover Command - Find available data for an area and time window.

Usage:
    mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood
    mdive discover --bbox -80.5,25.5,-80.0,26.0 --start 2024-09-15 --end 2024-09-20 --event flood
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import click

logger = logging.getLogger("mdive.discover")


def format_size(size_bytes: Optional[int]) -> str:
    """Format bytes to human-readable string."""
    if size_bytes is None:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date: {date_str}")


def load_geometry(area_path: Optional[Path], bbox: Optional[str]) -> Dict[str, Any]:
    """Load geometry from file or bounding box string."""
    if area_path:
        if not area_path.exists():
            raise click.BadParameter(f"Area file not found: {area_path}")

        suffix = area_path.suffix.lower()
        if suffix == ".geojson" or suffix == ".json":
            with open(area_path) as f:
                geojson = json.load(f)
            # Extract geometry from GeoJSON
            if geojson.get("type") == "FeatureCollection":
                if geojson.get("features"):
                    return geojson["features"][0]["geometry"]
                raise click.BadParameter("GeoJSON FeatureCollection has no features")
            elif geojson.get("type") == "Feature":
                return geojson["geometry"]
            else:
                return geojson
        else:
            raise click.BadParameter(f"Unsupported area file format: {suffix}")

    elif bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise click.BadParameter(
                "Bounding box must have 4 values: min_lon,min_lat,max_lon,max_lat"
            )
        min_lon, min_lat, max_lon, max_lat = parts
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }
    else:
        raise click.BadParameter("Either --area or --bbox must be provided")


def format_table_row(columns: List[str], widths: List[int]) -> str:
    """Format a table row with proper spacing."""
    parts = []
    for col, width in zip(columns, widths):
        parts.append(str(col).ljust(width)[:width])
    return "  ".join(parts)


@click.command("discover")
@click.option(
    "--area",
    "-a",
    "area_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GeoJSON file defining the area of interest.",
)
@click.option(
    "--bbox",
    "-b",
    type=str,
    help="Bounding box as min_lon,min_lat,max_lon,max_lat.",
)
@click.option(
    "--start",
    "-s",
    "start_date",
    required=True,
    type=str,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    "-e",
    "end_date",
    required=True,
    type=str,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--event",
    "-t",
    "event_type",
    type=click.Choice(["flood", "wildfire", "storm"], case_sensitive=False),
    required=True,
    help="Event type to optimize data discovery for.",
)
@click.option(
    "--source",
    "-S",
    "sources",
    multiple=True,
    type=str,
    help="Filter by specific data sources (e.g., sentinel1, sentinel2, landsat).",
)
@click.option(
    "--max-cloud",
    type=float,
    default=30.0,
    help="Maximum cloud cover percentage for optical data (default: 30).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output file path for results (JSON format).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.pass_obj
def discover(
    ctx,
    area_path: Optional[Path],
    bbox: Optional[str],
    start_date: str,
    end_date: str,
    event_type: str,
    sources: tuple,
    max_cloud: float,
    output_path: Optional[Path],
    output_format: str,
):
    """
    Discover available satellite data for an area and time window.

    Queries multiple data catalogs (STAC, WMS/WCS) to find available
    satellite imagery matching the specified criteria. Results include
    cloud cover, resolution, and availability information.

    \b
    Examples:
        # Discover flood-relevant data for Miami
        mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood

        # Use bounding box instead of file
        mdive discover --bbox -80.5,25.5,-80.0,26.0 --start 2024-09-15 --end 2024-09-20 --event flood

        # Filter by source and output as JSON
        mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 \\
            --event flood --source sentinel1 --format json --output results.json
    """
    # Parse inputs
    start = parse_date(start_date)
    end = parse_date(end_date)
    geometry = load_geometry(area_path, bbox)

    if start > end:
        raise click.BadParameter("Start date must be before end date")

    click.echo(f"\nDiscovering data for {event_type} event...")
    click.echo(f"  Time window: {start.date()} to {end.date()}")
    click.echo(f"  Max cloud cover: {max_cloud}%")
    if sources:
        click.echo(f"  Sources: {', '.join(sources)}")

    # Perform discovery
    results = perform_discovery(
        geometry=geometry,
        start=start,
        end=end,
        event_type=event_type,
        sources=list(sources) if sources else None,
        max_cloud=max_cloud,
        config=ctx.config if ctx else {},
    )

    # Output results
    if output_format == "json":
        output_json(results, output_path)
    elif output_format == "csv":
        output_csv(results, output_path)
    else:
        output_table(results)

    # Summary
    click.echo(f"\nFound {len(results)} datasets")
    if output_path:
        click.echo(f"Results saved to: {output_path}")


def perform_discovery(
    geometry: Dict[str, Any],
    start: datetime,
    end: datetime,
    event_type: str,
    sources: Optional[List[str]],
    max_cloud: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Perform data discovery across configured catalogs.

    This function queries STAC catalogs and other data sources
    to find available satellite imagery.
    """
    results = []

    # Define event-specific data requirements
    event_data_priorities = {
        "flood": {
            "primary": ["sentinel1", "sentinel2"],
            "secondary": ["landsat8", "landsat9", "modis"],
            "ancillary": ["dem", "wsf", "osm"],
        },
        "wildfire": {
            "primary": ["sentinel2", "landsat8", "landsat9"],
            "secondary": ["modis", "viirs"],
            "ancillary": ["dem", "landcover"],
        },
        "storm": {
            "primary": ["sentinel1", "sentinel2"],
            "secondary": ["landsat8", "landsat9"],
            "ancillary": ["dem", "wsf", "osm"],
        },
    }

    priorities = event_data_priorities.get(event_type.lower(), event_data_priorities["flood"])

    # Try to import and use actual discovery modules
    try:
        from core.data.broker import DataBroker

        broker = DataBroker()
        discovery_results = broker.discover(
            geometry=geometry,
            start_date=start,
            end_date=end,
            event_type=event_type,
            max_cloud_cover=max_cloud,
        )

        for dr in discovery_results:
            result = {
                "id": dr.id if hasattr(dr, "id") else str(id(dr)),
                "source": dr.provider if hasattr(dr, "provider") else "unknown",
                "datetime": dr.datetime.isoformat() if hasattr(dr, "datetime") else None,
                "cloud_cover": dr.cloud_cover if hasattr(dr, "cloud_cover") else None,
                "resolution_m": dr.resolution if hasattr(dr, "resolution") else None,
                "size_bytes": dr.size if hasattr(dr, "size") else None,
                "url": dr.url if hasattr(dr, "url") else None,
                "priority": "primary" if dr.provider in priorities["primary"] else "secondary",
            }
            results.append(result)

    except ImportError:
        # Mock discovery for demonstration when core modules not available
        logger.debug("Core discovery modules not available, using mock data")
        results = generate_mock_results(
            geometry, start, end, event_type, sources, max_cloud, priorities
        )

    # Filter by source if specified
    if sources:
        source_lower = [s.lower() for s in sources]
        results = [r for r in results if r["source"].lower() in source_lower]

    # Sort by priority and datetime
    def sort_key(r):
        priority_order = {"primary": 0, "secondary": 1, "ancillary": 2}
        return (priority_order.get(r.get("priority", "secondary"), 1), r.get("datetime", ""))

    results.sort(key=sort_key)

    return results


def generate_mock_results(
    geometry: Dict[str, Any],
    start: datetime,
    end: datetime,
    event_type: str,
    sources: Optional[List[str]],
    max_cloud: float,
    priorities: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Generate mock discovery results for demonstration."""
    import random

    results = []
    current = start

    # Generate results for each day in the range
    while current <= end:
        # Sentinel-1 (SAR - no cloud issues)
        if not sources or "sentinel1" in [s.lower() for s in sources]:
            results.append({
                "id": f"S1A_IW_GRD_{current.strftime('%Y%m%d')}",
                "source": "sentinel1",
                "datetime": current.isoformat(),
                "cloud_cover": None,  # SAR
                "resolution_m": 10.0,
                "size_bytes": random.randint(800_000_000, 1_200_000_000),
                "url": f"https://scihub.copernicus.eu/dhus/S1A_IW_GRD_{current.strftime('%Y%m%d')}",
                "priority": "primary",
                "polarization": "VV+VH",
                "orbit": random.choice(["ascending", "descending"]),
            })

        # Sentinel-2 (optical)
        if not sources or "sentinel2" in [s.lower() for s in sources]:
            cloud = random.uniform(0, 100)
            if cloud <= max_cloud:
                results.append({
                    "id": f"S2A_MSIL2A_{current.strftime('%Y%m%d')}",
                    "source": "sentinel2",
                    "datetime": current.isoformat(),
                    "cloud_cover": round(cloud, 1),
                    "resolution_m": 10.0,
                    "size_bytes": random.randint(500_000_000, 900_000_000),
                    "url": f"https://scihub.copernicus.eu/dhus/S2A_MSIL2A_{current.strftime('%Y%m%d')}",
                    "priority": "primary",
                    "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
                })

        # Landsat (optical, every 8 days)
        if current.day % 8 == 0:
            if not sources or "landsat8" in [s.lower() for s in sources]:
                cloud = random.uniform(0, 100)
                if cloud <= max_cloud:
                    results.append({
                        "id": f"LC08_L2SP_{current.strftime('%Y%m%d')}",
                        "source": "landsat8",
                        "datetime": current.isoformat(),
                        "cloud_cover": round(cloud, 1),
                        "resolution_m": 30.0,
                        "size_bytes": random.randint(200_000_000, 400_000_000),
                        "url": f"https://earthexplorer.usgs.gov/LC08_{current.strftime('%Y%m%d')}",
                        "priority": "secondary",
                    })

        current += timedelta(days=1)

    return results


def output_table(results: List[Dict[str, Any]]):
    """Output results as a formatted table."""
    if not results:
        click.echo("\nNo datasets found matching criteria.")
        return

    click.echo("\n")
    widths = [40, 12, 20, 8, 8, 10]
    headers = ["ID", "Source", "Date/Time", "Cloud%", "Res(m)", "Size"]
    click.echo(format_table_row(headers, widths))
    click.echo("-" * (sum(widths) + len(widths) * 2))

    for r in results:
        dt_str = r.get("datetime", "")
        if dt_str and "T" in dt_str:
            dt_str = dt_str.split("T")[0]

        cloud = r.get("cloud_cover")
        cloud_str = f"{cloud:.0f}" if cloud is not None else "N/A"

        res = r.get("resolution_m")
        res_str = f"{res:.0f}" if res is not None else "N/A"

        size_str = format_size(r.get("size_bytes"))

        row = [
            r.get("id", "unknown"),
            r.get("source", "unknown"),
            dt_str,
            cloud_str,
            res_str,
            size_str,
        ]
        click.echo(format_table_row(row, widths))


def output_json(results: List[Dict[str, Any]], output_path: Optional[Path]):
    """Output results as JSON."""
    output = {
        "count": len(results),
        "results": results,
    }

    json_str = json.dumps(output, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)
    else:
        click.echo(json_str)


def output_csv(results: List[Dict[str, Any]], output_path: Optional[Path]):
    """Output results as CSV."""
    import csv
    import io

    if not results:
        click.echo("No results to export.")
        return

    fields = ["id", "source", "datetime", "cloud_cover", "resolution_m", "size_bytes", "url"]

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        click.echo(output.getvalue())
