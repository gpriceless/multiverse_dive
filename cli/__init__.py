"""
Multiverse Dive CLI Package

Command-line interface for the geospatial event intelligence platform.
Provides commands for data discovery, ingestion, analysis, and export.

Usage:
    mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood
    mdive ingest --area miami.geojson --source sentinel1 --output ./data/
    mdive analyze --input ./data/ --algorithm sar_threshold --output ./results/
    mdive run --area miami.geojson --event flood --profile laptop --output ./products/
"""

__version__ = "0.1.0"
__author__ = "Multiverse Dive Team"

from cli.main import app

__all__ = ["app", "__version__"]
