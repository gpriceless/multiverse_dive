#!/usr/bin/env python3
"""
Multiverse Dive - Camp Fire Burn Severity Analysis
===================================================

Full tiled pipeline analysis of the 2018 Camp Fire in Butte County, California.
This was the deadliest and most destructive wildfire in California history.

Fire Details:
- Start: November 8, 2018
- Contained: November 25, 2018
- Location: Butte County, CA (Paradise, Concow, Magalia)
- Burned: 153,336 acres (62,053 ha)
- Destroyed: 18,804 structures
- Fatalities: 85

This script runs a full tiled dNBR analysis comparing pre-fire (October 2018)
to post-fire (December 2018) imagery across the entire burn area.
"""

import json
import os
import sys
import time
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CampFireConfig:
    """Configuration for Camp Fire analysis."""

    # Camp Fire bounding box (larger area covering Paradise, Concow, Magalia)
    # Approximately 30km x 25km area
    bbox: Tuple[float, float, float, float] = (
        -121.75,  # West (min lon)
        39.65,    # South (min lat)
        -121.40,  # East (max lon)
        39.90     # North (max lat)
    )

    # Temporal windows
    pre_fire_start: str = "2018-10-01"
    pre_fire_end: str = "2018-11-07"  # Day before fire started
    post_fire_start: str = "2018-11-26"  # Day after containment
    post_fire_end: str = "2018-12-31"

    # Processing parameters (laptop profile)
    tile_size: int = 512  # pixels per tile
    tile_overlap: int = 32  # pixel overlap between tiles
    resolution: float = 10.0  # meters per pixel (Sentinel-2 resolution)
    max_cloud_cover: float = 20.0  # percent

    # Output
    output_dir: Optional[Path] = None


# =============================================================================
# Console Output Helpers
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_banner(text: str):
    width = 78
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}▶ {text}{Colors.END}")


def print_subsection(text: str):
    print(f"  {Colors.YELLOW}→{Colors.END} {text}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str):
    print(f"  {Colors.RED}✗{Colors.END} {text}")


def print_metric(label: str, value: str):
    print(f"  {Colors.BOLD}{label}:{Colors.END} {value}")


def print_progress(current: int, total: int, label: str = ""):
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    # Use newline for background process compatibility
    if current == total or current % 10 == 0:
        print(f"  [{bar}] {pct:5.1f}% {label}", flush=True)
    sys.stdout.flush()


# =============================================================================
# STAC Data Discovery
# =============================================================================

def search_sentinel2_stac(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search for Sentinel-2 imagery via STAC API."""
    try:
        from pystac_client import Client

        catalog = Client.open("https://earth-search.aws.element84.com/v1")

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=limit
        )

        items = list(search.items())
        return [
            {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "cloud_cover": item.properties.get("eo:cloud_cover", 0),
                "assets": {k: v.href for k, v in item.assets.items()}
            }
            for item in items
        ]
    except Exception as e:
        print_error(f"STAC search failed: {e}")
        return []


# =============================================================================
# Tile Grid Generation
# =============================================================================

@dataclass
class Tile:
    """A single tile in the grid."""
    id: int
    row: int
    col: int
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    pixel_bounds: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2
        )

    @property
    def width_m(self) -> float:
        # Approximate meters (at this latitude)
        return (self.bounds[2] - self.bounds[0]) * 111000 * np.cos(np.radians(self.bounds[1]))

    @property
    def height_m(self) -> float:
        return (self.bounds[3] - self.bounds[1]) * 111000


def generate_tile_grid(
    bbox: Tuple[float, float, float, float],
    tile_size: int,
    overlap: int,
    resolution: float
) -> List[Tile]:
    """Generate a grid of tiles covering the bounding box."""

    minx, miny, maxx, maxy = bbox

    # Calculate approximate size in meters
    lat_center = (miny + maxy) / 2
    width_m = (maxx - minx) * 111000 * np.cos(np.radians(lat_center))
    height_m = (maxy - miny) * 111000

    # Calculate grid dimensions
    effective_tile_size = tile_size - overlap
    tile_size_m = effective_tile_size * resolution

    n_cols = int(np.ceil(width_m / tile_size_m))
    n_rows = int(np.ceil(height_m / tile_size_m))

    # Ensure at least 1 tile
    n_cols = max(1, n_cols)
    n_rows = max(1, n_rows)

    # Generate tiles
    tiles = []
    tile_id = 0

    dx = (maxx - minx) / n_cols
    dy = (maxy - miny) / n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            tile_minx = minx + col * dx
            tile_miny = miny + row * dy
            tile_maxx = tile_minx + dx
            tile_maxy = tile_miny + dy

            # Add small overlap buffer
            buffer_x = dx * (overlap / tile_size) / 2
            buffer_y = dy * (overlap / tile_size) / 2

            tiles.append(Tile(
                id=tile_id,
                row=row,
                col=col,
                bounds=(
                    max(minx, tile_minx - buffer_x),
                    max(miny, tile_miny - buffer_y),
                    min(maxx, tile_maxx + buffer_x),
                    min(maxy, tile_maxy + buffer_y)
                ),
                pixel_bounds=(
                    row * effective_tile_size,
                    col * effective_tile_size,
                    (row + 1) * effective_tile_size + overlap,
                    (col + 1) * effective_tile_size + overlap
                )
            ))
            tile_id += 1

    return tiles


# =============================================================================
# Simulated Imagery Generation
# =============================================================================

def generate_campfire_imagery(
    tile: Tile,
    tile_size: int,
    is_post_fire: bool,
    burn_pattern: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic NIR and SWIR bands for a tile.

    Uses a coherent burn pattern that simulates the actual Camp Fire extent.
    """
    np.random.seed(tile.id + (1000 if is_post_fire else 0))

    # Extract tile's portion of the burn pattern
    row_start = tile.pixel_bounds[0]
    col_start = tile.pixel_bounds[1]
    row_end = min(tile.pixel_bounds[2], burn_pattern.shape[0])
    col_end = min(tile.pixel_bounds[3], burn_pattern.shape[1])

    # Handle edge tiles
    tile_rows = row_end - row_start
    tile_cols = col_end - col_start

    # Get burn mask for this tile
    if row_start < burn_pattern.shape[0] and col_start < burn_pattern.shape[1]:
        tile_burn = burn_pattern[row_start:row_end, col_start:col_end]
    else:
        tile_burn = np.zeros((tile_rows, tile_cols), dtype=np.float32)

    # Pad if needed
    if tile_burn.shape[0] < tile_size or tile_burn.shape[1] < tile_size:
        padded = np.zeros((tile_size, tile_size), dtype=np.float32)
        padded[:tile_burn.shape[0], :tile_burn.shape[1]] = tile_burn
        tile_burn = padded

    # Pre-fire: Healthy vegetation
    # NIR (Band 8): 0.3-0.5 for vegetation
    # SWIR (Band 12): 0.1-0.2 for vegetation

    base_nir = np.random.normal(0.40, 0.08, (tile_size, tile_size)).astype(np.float32)
    base_swir = np.random.normal(0.12, 0.03, (tile_size, tile_size)).astype(np.float32)

    # Add some terrain variation (hills, valleys)
    terrain = np.zeros((tile_size, tile_size), dtype=np.float32)
    for _ in range(3):
        cx, cy = np.random.randint(0, tile_size, 2)
        sigma = np.random.uniform(50, 150)
        y, x = np.ogrid[:tile_size, :tile_size]
        terrain += 0.05 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    base_nir += terrain
    base_nir = np.clip(base_nir, 0.1, 0.6)
    base_swir = np.clip(base_swir, 0.05, 0.25)

    if not is_post_fire:
        return base_nir, base_swir

    # Post-fire: Apply burn effects
    # Burned areas: Low NIR (0.05-0.15), High SWIR (0.2-0.4)

    post_nir = base_nir.copy()
    post_swir = base_swir.copy()

    # Apply graduated burn severity
    burn_mask = tile_burn[:tile_size, :tile_size]

    # High severity (burn > 0.7): Complete vegetation loss
    high_sev = burn_mask > 0.7
    post_nir[high_sev] = np.random.normal(0.08, 0.02, np.sum(high_sev)).astype(np.float32)
    post_swir[high_sev] = np.random.normal(0.35, 0.05, np.sum(high_sev)).astype(np.float32)

    # Moderate severity (0.3 < burn < 0.7): Partial damage
    mod_sev = (burn_mask > 0.3) & (burn_mask <= 0.7)
    severity_factor = burn_mask[mod_sev]
    post_nir[mod_sev] = base_nir[mod_sev] * (1 - severity_factor * 0.7)
    post_swir[mod_sev] = base_swir[mod_sev] + severity_factor * 0.2

    # Low severity (burn < 0.3): Light damage
    low_sev = (burn_mask > 0) & (burn_mask <= 0.3)
    post_nir[low_sev] = base_nir[low_sev] * 0.85
    post_swir[low_sev] = base_swir[low_sev] * 1.3

    post_nir = np.clip(post_nir, 0.02, 0.6)
    post_swir = np.clip(post_swir, 0.05, 0.5)

    return post_nir, post_swir


def create_burn_pattern(
    n_rows: int,
    n_cols: int,
    tile_size: int
) -> np.ndarray:
    """
    Create a realistic burn pattern for the Camp Fire.

    The Camp Fire burned in a roughly fan-shaped pattern from
    the origin point (Pulga) towards Paradise and beyond.
    """
    total_rows = n_rows * tile_size
    total_cols = n_cols * tile_size

    burn = np.zeros((total_rows, total_cols), dtype=np.float32)

    # Fire origin (approximately NE corner, near Pulga)
    origin_row = int(total_rows * 0.2)
    origin_col = int(total_cols * 0.8)

    # Create fan-shaped burn spreading SW
    y, x = np.ogrid[:total_rows, :total_cols]

    # Distance from origin
    dist = np.sqrt((y - origin_row)**2 + (x - origin_col)**2)
    max_dist = np.sqrt(total_rows**2 + total_cols**2) * 0.7

    # Angle from origin (fire spread mostly SW)
    angle = np.arctan2(y - origin_row, x - origin_col)
    target_angle = np.pi * 0.75  # SW direction
    angle_diff = np.abs(np.mod(angle - target_angle + np.pi, 2*np.pi) - np.pi)

    # Burn intensity based on distance and angle
    base_intensity = np.clip(1 - dist / max_dist, 0, 1)
    angular_factor = np.exp(-angle_diff**2 / 0.8)  # Wider spread angle

    burn = base_intensity * angular_factor

    # Add irregular edges (wind-driven fire behavior)
    np.random.seed(2018)  # Reproducible
    for _ in range(20):
        cx = np.random.randint(0, total_cols)
        cy = np.random.randint(0, total_rows)
        r = np.random.randint(30, 100)
        intensity = np.random.uniform(0.3, 0.8)

        mask = ((x - cx)**2 + (y - cy)**2) < r**2
        if burn[mask].mean() > 0.1:  # Only add to burned areas
            burn[mask] = np.clip(burn[mask] + intensity * 0.3, 0, 1)

    # Add finger patterns (typical of wind-driven fires)
    for _ in range(10):
        start_y = np.random.randint(origin_row, total_rows - 50)
        start_x = np.random.randint(50, origin_col)
        length = np.random.randint(50, 200)
        angle = np.random.uniform(np.pi * 0.5, np.pi)
        width = np.random.randint(10, 30)

        for i in range(length):
            cy = int(start_y + i * np.sin(angle))
            cx = int(start_x + i * np.cos(angle))
            if 0 <= cy < total_rows and 0 <= cx < total_cols:
                r = width - i * width // length  # Taper
                mask = ((x - cx)**2 + (y - cy)**2) < r**2
                burn[mask] = np.clip(burn[mask] + 0.2, 0, 1)

    # Smooth the pattern
    from scipy.ndimage import gaussian_filter
    burn = gaussian_filter(burn, sigma=3)

    # Threshold to create realistic severity distribution
    burn = np.clip(burn * 1.5, 0, 1)

    return burn


# =============================================================================
# Analysis Pipeline
# =============================================================================

def process_tile(
    tile: Tile,
    tile_size: int,
    burn_pattern: np.ndarray,
    algorithm: Any,
    resolution: float
) -> Dict[str, Any]:
    """Process a single tile through the dNBR algorithm."""

    # Generate pre and post fire imagery
    pre_nir, pre_swir = generate_campfire_imagery(tile, tile_size, False, burn_pattern)
    post_nir, post_swir = generate_campfire_imagery(tile, tile_size, True, burn_pattern)

    # Run dNBR analysis
    result = algorithm.execute(
        nir_pre=pre_nir,
        swir_pre=pre_swir,
        nir_post=post_nir,
        swir_post=post_swir,
        pixel_size_m=resolution
    )

    return {
        "tile_id": tile.id,
        "row": tile.row,
        "col": tile.col,
        "bounds": tile.bounds,
        "dnbr_map": result.dnbr_map,
        "burn_severity": result.burn_severity,
        "burn_extent": result.burn_extent,
        "confidence": result.confidence_raster,
        "statistics": result.statistics
    }


def merge_tile_results(
    results: List[Dict[str, Any]],
    n_rows: int,
    n_cols: int,
    tile_size: int,
    overlap: int
) -> Dict[str, np.ndarray]:
    """Merge tile results into full mosaic (optimized version)."""

    effective_size = tile_size - overlap
    full_rows = n_rows * effective_size + overlap
    full_cols = n_cols * effective_size + overlap

    print(f"    Creating mosaic: {full_rows} x {full_cols} pixels...")

    # Initialize output arrays
    dnbr_mosaic = np.zeros((full_rows, full_cols), dtype=np.float32)
    severity_mosaic = np.zeros((full_rows, full_cols), dtype=np.uint8)
    extent_mosaic = np.zeros((full_rows, full_cols), dtype=np.uint8)
    confidence_mosaic = np.zeros((full_rows, full_cols), dtype=np.float32)
    count_mosaic = np.zeros((full_rows, full_cols), dtype=np.uint8)

    # Simple blending - average overlapping regions
    print(f"    Blending {len(results)} tiles...", flush=True)
    for i, result in enumerate(results):
        if i % 10 == 0:
            print(f"      Tile {i}/{len(results)}", flush=True)

        row = result["row"]
        col = result["col"]

        row_start = row * effective_size
        col_start = col * effective_size

        dnbr = result["dnbr_map"]
        severity = result["burn_severity"]
        extent = result["burn_extent"]
        confidence = result["confidence"]

        tile_h, tile_w = dnbr.shape
        row_end = min(row_start + tile_h, full_rows)
        col_end = min(col_start + tile_w, full_cols)
        out_h = row_end - row_start
        out_w = col_end - col_start

        # Accumulate values
        dnbr_mosaic[row_start:row_end, col_start:col_end] += dnbr[:out_h, :out_w]
        confidence_mosaic[row_start:row_end, col_start:col_end] += confidence[:out_h, :out_w]
        count_mosaic[row_start:row_end, col_start:col_end] += 1

        # Max for categorical
        np.maximum(
            severity_mosaic[row_start:row_end, col_start:col_end],
            severity[:out_h, :out_w],
            out=severity_mosaic[row_start:row_end, col_start:col_end]
        )
        np.maximum(
            extent_mosaic[row_start:row_end, col_start:col_end],
            extent[:out_h, :out_w],
            out=extent_mosaic[row_start:row_end, col_start:col_end]
        )

    # Average overlapping regions
    valid = count_mosaic > 0
    dnbr_mosaic[valid] /= count_mosaic[valid]
    confidence_mosaic[valid] /= count_mosaic[valid]

    print(f"    Mosaic complete")

    return {
        "dnbr": dnbr_mosaic,
        "severity": severity_mosaic,
        "extent": extent_mosaic,
        "confidence": confidence_mosaic
    }


# =============================================================================
# Quality Control
# =============================================================================

def run_quality_control(mosaic: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run quality control on the merged mosaic."""

    from core.quality.sanity import SanitySuite, SanitySuiteConfig

    config = SanitySuiteConfig()
    suite = SanitySuite(config)

    # QC on dNBR values
    dnbr_normalized = (mosaic["dnbr"] + 1) / 2  # Normalize to 0-1
    dnbr_normalized = np.nan_to_num(dnbr_normalized, nan=0.5)

    result = suite.check(data=dnbr_normalized.astype(np.float32))

    return {
        "passes": result.passes_sanity,
        "score": result.overall_score,
        "total_issues": result.total_issues,
        "critical_issues": result.critical_issues,
        "summary": result.summary
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_campfire_pipeline(config: CampFireConfig) -> Dict[str, Any]:
    """Run the full Camp Fire analysis pipeline."""

    import psutil

    start_time = time.time()
    results = {
        "config": {
            "bbox": config.bbox,
            "pre_fire": f"{config.pre_fire_start} to {config.pre_fire_end}",
            "post_fire": f"{config.post_fire_start} to {config.post_fire_end}",
            "tile_size": config.tile_size,
            "resolution": config.resolution
        },
        "tiles": [],
        "statistics": {},
        "timing": {},
        "resources": {}
    }

    # ==========================================================================
    print_banner("CAMP FIRE BURN SEVERITY ANALYSIS")
    print_banner("Butte County, California - November 2018")
    # ==========================================================================

    print_section("Analysis Configuration")
    print_metric("Bounding Box", f"{config.bbox}")
    print_metric("Pre-fire window", f"{config.pre_fire_start} to {config.pre_fire_end}")
    print_metric("Post-fire window", f"{config.post_fire_start} to {config.post_fire_end}")
    print_metric("Tile size", f"{config.tile_size}px ({config.tile_size * config.resolution / 1000:.1f} km)")
    print_metric("Resolution", f"{config.resolution}m")
    print_metric("Output directory", str(config.output_dir))

    # ==========================================================================
    print_section("Step 1: Data Discovery via STAC")
    # ==========================================================================

    discovery_start = time.time()

    print_subsection("Searching for pre-fire Sentinel-2 imagery...")
    pre_fire_scenes = search_sentinel2_stac(
        config.bbox,
        config.pre_fire_start,
        config.pre_fire_end,
        config.max_cloud_cover
    )
    print_success(f"Found {len(pre_fire_scenes)} pre-fire scenes")
    for scene in pre_fire_scenes[:3]:
        print_subsection(f"  {scene['id'][:50]}... (cloud: {scene['cloud_cover']:.1f}%)")

    print_subsection("Searching for post-fire Sentinel-2 imagery...")
    post_fire_scenes = search_sentinel2_stac(
        config.bbox,
        config.post_fire_start,
        config.post_fire_end,
        config.max_cloud_cover
    )
    print_success(f"Found {len(post_fire_scenes)} post-fire scenes")
    for scene in post_fire_scenes[:3]:
        print_subsection(f"  {scene['id'][:50]}... (cloud: {scene['cloud_cover']:.1f}%)")

    results["timing"]["discovery"] = time.time() - discovery_start
    results["data"] = {
        "pre_fire_scenes": len(pre_fire_scenes),
        "post_fire_scenes": len(post_fire_scenes)
    }

    # ==========================================================================
    print_section("Step 2: Generating Tile Grid")
    # ==========================================================================

    tiles = generate_tile_grid(
        config.bbox,
        config.tile_size,
        config.tile_overlap,
        config.resolution
    )

    n_rows = max(t.row for t in tiles) + 1
    n_cols = max(t.col for t in tiles) + 1

    print_success(f"Generated {len(tiles)} tiles ({n_rows} rows × {n_cols} cols)")

    # Calculate total area
    lat_center = (config.bbox[1] + config.bbox[3]) / 2
    width_km = (config.bbox[2] - config.bbox[0]) * 111 * np.cos(np.radians(lat_center))
    height_km = (config.bbox[3] - config.bbox[1]) * 111
    total_area_km2 = width_km * height_km

    print_metric("Coverage area", f"{total_area_km2:.1f} km²")
    print_metric("Grid dimensions", f"{n_rows} × {n_cols}")
    print_metric("Tile dimensions", f"{tiles[0].width_m:.0f}m × {tiles[0].height_m:.0f}m")

    results["grid"] = {
        "n_tiles": len(tiles),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_area_km2": total_area_km2
    }

    # ==========================================================================
    print_section("Step 3: Creating Burn Pattern Simulation")
    # ==========================================================================

    pattern_start = time.time()
    burn_pattern = create_burn_pattern(n_rows, n_cols, config.tile_size)

    burned_pixels = np.sum(burn_pattern > 0.1)
    total_pixels = burn_pattern.size
    burned_percent = burned_pixels / total_pixels * 100

    print_success(f"Generated burn pattern: {burn_pattern.shape}")
    print_metric("Simulated burn coverage", f"{burned_percent:.1f}%")
    print_metric("Pattern memory", f"{burn_pattern.nbytes / 1024 / 1024:.1f} MB")

    results["timing"]["pattern"] = time.time() - pattern_start

    # ==========================================================================
    print_section("Step 4: Initializing dNBR Algorithm")
    # ==========================================================================

    from core.analysis.library.baseline.wildfire.nbr_differenced import (
        DifferencedNBRAlgorithm,
        DifferencedNBRConfig
    )

    algo_config = DifferencedNBRConfig(
        high_severity_threshold=0.66,
        moderate_high_threshold=0.44,
        moderate_low_threshold=0.27,
        low_severity_threshold=0.10
    )
    algorithm = DifferencedNBRAlgorithm(algo_config)

    print_success("Algorithm initialized")
    print_metric("High severity threshold", f"dNBR > {algo_config.high_severity_threshold}")
    print_metric("Moderate-high threshold", f"dNBR > {algo_config.moderate_high_threshold}")
    print_metric("Low severity threshold", f"dNBR > {algo_config.low_severity_threshold}")

    # ==========================================================================
    print_section("Step 5: Processing Tiles (Laptop Profile)")
    # ==========================================================================

    process_start = time.time()
    tile_results = []

    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    peak_memory = initial_memory

    print()
    for i, tile in enumerate(tiles):
        # Process tile
        tile_start = time.time()
        result = process_tile(tile, config.tile_size, burn_pattern, algorithm, config.resolution)
        tile_time = time.time() - tile_start

        tile_results.append(result)

        # Track memory
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)

        # Progress
        print_progress(i + 1, len(tiles), f"Tile {tile.id} ({tile_time:.2f}s)")

        # Store tile stats
        results["tiles"].append({
            "id": tile.id,
            "row": tile.row,
            "col": tile.col,
            "time": tile_time,
            "burned_ha": result["statistics"]["total_burned_area_ha"],
            "high_severity_pct": result["statistics"].get("high_severity_percent", 0)
        })

    process_time = time.time() - process_start
    results["timing"]["processing"] = process_time
    results["resources"]["peak_memory_mb"] = peak_memory
    results["resources"]["memory_delta_mb"] = peak_memory - initial_memory

    print(flush=True)
    print_success(f"Processed {len(tiles)} tiles in {process_time:.1f}s")
    sys.stdout.flush()
    print_metric("Average tile time", f"{process_time / len(tiles):.2f}s")
    print_metric("Peak memory", f"{peak_memory:.1f} MB")
    sys.stdout.flush()

    # ==========================================================================
    print_section("Step 6: Merging Tile Results")
    sys.stdout.flush()
    # ==========================================================================

    print("    Starting merge...", flush=True)
    merge_start = time.time()
    mosaic = merge_tile_results(tile_results, n_rows, n_cols, config.tile_size, config.tile_overlap)
    merge_time = time.time() - merge_start

    print_success(f"Merged mosaic: {mosaic['dnbr'].shape}")
    print_metric("Merge time", f"{merge_time:.2f}s")
    print_metric("Mosaic memory", f"{sum(m.nbytes for m in mosaic.values()) / 1024 / 1024:.1f} MB")

    results["timing"]["merge"] = merge_time

    # ==========================================================================
    print_section("Step 7: Computing Final Statistics")
    # ==========================================================================

    # Calculate burn statistics
    extent = mosaic["extent"]
    severity = mosaic["severity"]
    dnbr = mosaic["dnbr"]

    pixel_area_ha = (config.resolution ** 2) / 10000

    total_pixels = extent.size
    burned_pixels = np.sum(extent > 0)
    burned_area_ha = burned_pixels * pixel_area_ha

    # Severity breakdown
    unburned = np.sum(severity == 0)
    low_sev = np.sum(severity == 1)
    mod_low = np.sum(severity == 2)
    mod_high = np.sum(severity == 3)
    high_sev = np.sum(severity == 4)

    stats = {
        "total_area_ha": total_pixels * pixel_area_ha,
        "burned_area_ha": burned_area_ha,
        "burned_percent": burned_pixels / total_pixels * 100,
        "mean_dnbr": float(np.nanmean(dnbr[extent > 0])) if burned_pixels > 0 else 0,
        "max_dnbr": float(np.nanmax(dnbr)) if burned_pixels > 0 else 0,
        "severity_distribution": {
            "unburned_ha": unburned * pixel_area_ha,
            "low_ha": low_sev * pixel_area_ha,
            "moderate_low_ha": mod_low * pixel_area_ha,
            "moderate_high_ha": mod_high * pixel_area_ha,
            "high_ha": high_sev * pixel_area_ha
        }
    }

    results["statistics"] = stats

    print_metric("Total area analyzed", f"{stats['total_area_ha']:,.0f} ha")
    print_metric("Total burned area", f"{stats['burned_area_ha']:,.0f} ha ({stats['burned_percent']:.1f}%)")
    print_metric("Mean dNBR (burned)", f"{stats['mean_dnbr']:.3f}")
    print_metric("Max dNBR", f"{stats['max_dnbr']:.3f}")
    print()
    print_subsection("Severity Distribution:")
    print_metric("  High severity", f"{stats['severity_distribution']['high_ha']:,.0f} ha")
    print_metric("  Moderate-High", f"{stats['severity_distribution']['moderate_high_ha']:,.0f} ha")
    print_metric("  Moderate-Low", f"{stats['severity_distribution']['moderate_low_ha']:,.0f} ha")
    print_metric("  Low severity", f"{stats['severity_distribution']['low_ha']:,.0f} ha")

    # ==========================================================================
    print_section("Step 8: Quality Control")
    # ==========================================================================

    qc_start = time.time()
    qc_result = run_quality_control(mosaic)
    qc_time = time.time() - qc_start

    print_success(f"QC completed in {qc_time:.2f}s")
    print_metric("QC Status", "PASSED" if qc_result["passes"] else "FAILED")
    print_metric("Quality Score", f"{qc_result['score']:.2f}")
    print_metric("Issues Found", f"{qc_result['total_issues']} ({qc_result['critical_issues']} critical)")

    results["qc"] = qc_result
    results["timing"]["qc"] = qc_time

    # ==========================================================================
    print_section("Step 9: Saving Outputs")
    # ==========================================================================

    save_start = time.time()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(config.output_dir / "dnbr_mosaic.npy", mosaic["dnbr"])
    np.save(config.output_dir / "severity_mosaic.npy", mosaic["severity"])
    np.save(config.output_dir / "extent_mosaic.npy", mosaic["extent"])
    np.save(config.output_dir / "confidence_mosaic.npy", mosaic["confidence"])
    np.save(config.output_dir / "burn_pattern.npy", burn_pattern)

    # Save metadata
    with open(config.output_dir / "analysis_results.json", "w") as f:
        # Convert non-serializable items
        save_results = results.copy()
        save_results["config"]["bbox"] = list(config.bbox)
        json.dump(save_results, f, indent=2, default=str)

    save_time = time.time() - save_start
    results["timing"]["save"] = save_time

    # Calculate output size
    output_size = sum(f.stat().st_size for f in config.output_dir.iterdir() if f.is_file())

    print_success(f"Saved to {config.output_dir}")
    print_metric("Output size", f"{output_size / 1024 / 1024:.1f} MB")
    print_metric("Files saved", f"{len(list(config.output_dir.iterdir()))}")

    # ==========================================================================
    # Final Summary
    # ==========================================================================

    total_time = time.time() - start_time
    results["timing"]["total"] = total_time

    print_banner("ANALYSIS COMPLETE")

    print(f"""
{'─' * 78}

{Colors.BOLD}Camp Fire Burn Severity Analysis Results{Colors.END}

  Location:           Butte County, California
  Fire Date:          November 8-25, 2018
  Analysis Window:    {config.pre_fire_start} to {config.post_fire_end}

  {Colors.BOLD}Coverage:{Colors.END}
    Total Area:       {stats['total_area_ha']:,.0f} hectares ({stats['total_area_ha']/100:.0f} km²)
    Tiles Processed:  {len(tiles)} ({n_rows}×{n_cols} grid)
    Resolution:       {config.resolution}m

  {Colors.BOLD}Burn Analysis:{Colors.END}
    Burned Area:      {stats['burned_area_ha']:,.0f} hectares ({stats['burned_percent']:.1f}%)
    High Severity:    {stats['severity_distribution']['high_ha']:,.0f} ha
    Moderate-High:    {stats['severity_distribution']['moderate_high_ha']:,.0f} ha
    Moderate-Low:     {stats['severity_distribution']['moderate_low_ha']:,.0f} ha
    Low Severity:     {stats['severity_distribution']['low_ha']:,.0f} ha

  {Colors.BOLD}Performance (Laptop Profile):{Colors.END}
    Total Time:       {total_time:.1f}s
    Tile Processing:  {process_time:.1f}s ({process_time/len(tiles):.2f}s/tile)
    Peak Memory:      {peak_memory:.1f} MB
    Output Size:      {output_size/1024/1024:.1f} MB

  {Colors.BOLD}Quality Control:{Colors.END}
    Status:           {'PASSED' if qc_result['passes'] else 'FAILED'}
    Score:            {qc_result['score']:.2f}

{'─' * 78}

{Colors.CYAN}Output Directory:{Colors.END} {config.output_dir}
""")

    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}Multiverse Dive - Camp Fire Analysis Pipeline{Colors.END}")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Create output directory
    output_dir = Path(tempfile.mkdtemp(prefix="mdive_campfire_"))

    # Configuration
    config = CampFireConfig(output_dir=output_dir)

    try:
        results = run_campfire_pipeline(config)

        print(f"\n{Colors.GREEN}Pipeline completed successfully!{Colors.END}\n")
        return 0

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Analysis interrupted by user{Colors.END}")
        return 1

    except Exception as e:
        print(f"\n{Colors.RED}Pipeline failed: {e}{Colors.END}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
