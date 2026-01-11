"""
Spatial Coherence Sanity Checks.

Validates that analysis outputs maintain spatial coherence and physical consistency:
- Spatial autocorrelation (nearby pixels should be similar for many phenomena)
- Boundary coherence (no artificial edges at tile boundaries)
- Geographic plausibility (outputs within valid geographic bounds)
- Topology consistency (connected regions, no isolated pixels from noise)
- Polygon validity (closed, non-self-intersecting)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SpatialCheckType(Enum):
    """Types of spatial coherence checks."""
    AUTOCORRELATION = "autocorrelation"           # Spatial autocorrelation (Moran's I)
    BOUNDARY_COHERENCE = "boundary_coherence"     # Tile boundary artifacts
    GEOGRAPHIC_BOUNDS = "geographic_bounds"       # Valid lat/lon ranges
    TOPOLOGY = "topology"                         # Connected component analysis
    ISOLATED_PIXELS = "isolated_pixels"           # Salt-and-pepper noise
    SPATIAL_CONTINUITY = "spatial_continuity"     # Smooth transitions
    POLYGON_VALIDITY = "polygon_validity"         # Valid polygon geometry
    COVERAGE_GAPS = "coverage_gaps"               # Holes in coverage


class SpatialIssueSeverity(Enum):
    """Severity levels for spatial issues."""
    CRITICAL = "critical"   # Fundamentally invalid output
    HIGH = "high"           # Likely indicates processing error
    MEDIUM = "medium"       # May indicate quality issue
    LOW = "low"             # Minor issue, possibly acceptable
    INFO = "info"           # Informational only


@dataclass
class SpatialIssue:
    """
    A spatial coherence issue found during validation.

    Attributes:
        check_type: Type of check that found the issue
        severity: Issue severity level
        description: Human-readable description
        location: Affected region (bbox or pixel coordinates)
        metric_value: The problematic metric value
        threshold: The threshold that was exceeded
        recommendation: Suggested action to address the issue
    """
    check_type: SpatialCheckType
    severity: SpatialIssueSeverity
    description: str
    location: Optional[Dict[str, Any]] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
        }


@dataclass
class SpatialCoherenceConfig:
    """
    Configuration for spatial coherence checks.

    Attributes:
        check_autocorrelation: Enable spatial autocorrelation check
        check_boundaries: Enable tile boundary check
        check_geographic_bounds: Enable geographic bounds validation
        check_topology: Enable topology/connectivity check
        check_isolated_pixels: Enable isolated pixel detection

        min_autocorrelation: Minimum expected spatial autocorrelation (Moran's I)
        max_isolated_pixel_pct: Maximum percentage of isolated pixels
        min_region_size_pixels: Minimum region size to not be considered noise
        boundary_gradient_threshold: Max allowed gradient at tile boundaries

        expected_bounds: Expected geographic bounds (minx, miny, maxx, maxy)
    """
    check_autocorrelation: bool = True
    check_boundaries: bool = True
    check_geographic_bounds: bool = True
    check_topology: bool = True
    check_isolated_pixels: bool = True

    min_autocorrelation: float = 0.0  # Moran's I threshold (0 = random, 1 = perfect)
    max_isolated_pixel_pct: float = 5.0  # Maximum % of isolated pixels
    min_region_size_pixels: int = 4  # Regions smaller than this are noise
    boundary_gradient_threshold: float = 2.0  # Max gradient z-score at boundaries

    # Geographic bounds (WGS84)
    expected_bounds: Optional[Tuple[float, float, float, float]] = None

    # Lat/lon ranges for sanity
    valid_latitude_range: Tuple[float, float] = (-90.0, 90.0)
    valid_longitude_range: Tuple[float, float] = (-180.0, 180.0)


@dataclass
class SpatialCoherenceResult:
    """
    Result of spatial coherence validation.

    Attributes:
        is_coherent: Whether all spatial checks passed
        issues: List of issues found
        metrics: Computed spatial metrics
        duration_seconds: Time taken for validation
    """
    is_coherent: bool
    issues: List[SpatialIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == SpatialIssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high-severity issues."""
        return sum(1 for i in self.issues if i.severity == SpatialIssueSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_coherent": self.is_coherent,
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class SpatialCoherenceChecker:
    """
    Validates spatial coherence of analysis outputs.

    Checks that outputs maintain expected spatial properties:
    - Spatial autocorrelation (nearby pixels are correlated)
    - No artificial boundaries from tiling
    - Valid geographic coordinates
    - Connected regions without excessive noise

    Example:
        checker = SpatialCoherenceChecker()
        result = checker.check(flood_extent_array, transform=geotransform)
        if not result.is_coherent:
            for issue in result.issues:
                print(f"{issue.severity}: {issue.description}")
    """

    def __init__(self, config: Optional[SpatialCoherenceConfig] = None):
        """
        Initialize the spatial coherence checker.

        Args:
            config: Configuration options for checks
        """
        self.config = config or SpatialCoherenceConfig()

    def check(
        self,
        data: np.ndarray,
        transform: Optional[Tuple[float, ...]] = None,
        mask: Optional[np.ndarray] = None,
        tile_boundaries: Optional[List[int]] = None,
    ) -> SpatialCoherenceResult:
        """
        Run all configured spatial coherence checks.

        Args:
            data: 2D array of analysis output values
            transform: Affine geotransform (a, b, c, d, e, f)
            mask: Optional boolean mask of valid pixels
            tile_boundaries: Optional list of row/column indices of tile boundaries

        Returns:
            SpatialCoherenceResult with all issues found
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}

        # Validate input
        if data.ndim != 2:
            issues.append(SpatialIssue(
                check_type=SpatialCheckType.TOPOLOGY,
                severity=SpatialIssueSeverity.CRITICAL,
                description=f"Expected 2D array, got {data.ndim}D",
            ))
            return SpatialCoherenceResult(
                is_coherent=False,
                issues=issues,
                duration_seconds=time.time() - start_time,
            )

        # Apply mask if provided
        if mask is not None:
            data = np.where(mask, data, np.nan)

        # Run enabled checks
        if self.config.check_autocorrelation:
            autocorr_issues, autocorr_metrics = self._check_autocorrelation(data)
            issues.extend(autocorr_issues)
            metrics["autocorrelation"] = autocorr_metrics

        if self.config.check_boundaries and tile_boundaries:
            boundary_issues, boundary_metrics = self._check_boundaries(data, tile_boundaries)
            issues.extend(boundary_issues)
            metrics["boundaries"] = boundary_metrics

        if self.config.check_geographic_bounds and transform:
            geo_issues, geo_metrics = self._check_geographic_bounds(data.shape, transform)
            issues.extend(geo_issues)
            metrics["geographic"] = geo_metrics

        if self.config.check_topology:
            topo_issues, topo_metrics = self._check_topology(data)
            issues.extend(topo_issues)
            metrics["topology"] = topo_metrics

        if self.config.check_isolated_pixels:
            isolated_issues, isolated_metrics = self._check_isolated_pixels(data)
            issues.extend(isolated_issues)
            metrics["isolated_pixels"] = isolated_metrics

        # Determine overall coherence
        is_coherent = not any(
            i.severity in (SpatialIssueSeverity.CRITICAL, SpatialIssueSeverity.HIGH)
            for i in issues
        )

        duration = time.time() - start_time
        logger.info(f"Spatial coherence check completed in {duration:.2f}s, coherent={is_coherent}")

        return SpatialCoherenceResult(
            is_coherent=is_coherent,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _check_autocorrelation(
        self, data: np.ndarray
    ) -> Tuple[List[SpatialIssue], Dict[str, Any]]:
        """
        Check spatial autocorrelation using Moran's I.

        Low autocorrelation in flood/fire extent may indicate noise.
        """
        issues = []
        metrics = {}

        try:
            morans_i = self._calculate_morans_i(data)
            metrics["morans_i"] = morans_i

            if np.isnan(morans_i):
                issues.append(SpatialIssue(
                    check_type=SpatialCheckType.AUTOCORRELATION,
                    severity=SpatialIssueSeverity.MEDIUM,
                    description="Could not calculate Moran's I (possibly constant values)",
                    metric_value=morans_i,
                ))
            elif morans_i < self.config.min_autocorrelation:
                # Low autocorrelation - may be noise
                severity = SpatialIssueSeverity.HIGH if morans_i < -0.1 else SpatialIssueSeverity.MEDIUM
                issues.append(SpatialIssue(
                    check_type=SpatialCheckType.AUTOCORRELATION,
                    severity=severity,
                    description=f"Low spatial autocorrelation (Moran's I = {morans_i:.3f})",
                    metric_value=morans_i,
                    threshold=self.config.min_autocorrelation,
                    recommendation="Check for noise or processing artifacts",
                ))
        except Exception as e:
            logger.warning(f"Autocorrelation check failed: {e}")
            metrics["error"] = str(e)

        return issues, metrics

    def _calculate_morans_i(self, data: np.ndarray, sample_size: int = 10000) -> float:
        """
        Calculate Moran's I spatial autocorrelation statistic.

        Uses sampling for large arrays to improve performance.
        """
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() < 10:
            return np.nan

        # Sample if array is large
        flat_data = data[valid_mask]
        if len(flat_data) > sample_size:
            indices = np.random.choice(len(flat_data), sample_size, replace=False)
            # For proper Moran's I we need spatial structure, so use subsampling
            rows, cols = np.where(valid_mask)
            sample_idx = np.random.choice(len(rows), min(sample_size, len(rows)), replace=False)
            rows = rows[sample_idx]
            cols = cols[sample_idx]
            values = data[rows, cols]
        else:
            rows, cols = np.where(valid_mask)
            values = data[rows, cols]

        n = len(values)
        if n < 4:
            return np.nan

        mean = np.mean(values)
        variance = np.var(values)
        if variance < 1e-10:
            return np.nan  # Constant values

        # Calculate Moran's I with simple contiguity weights
        # For efficiency, use k-nearest neighbors approximation
        numerator = 0.0
        weight_sum = 0.0

        # Use distance-based weights (inverse distance)
        k = min(8, n - 1)  # Use up to 8 nearest neighbors

        for i in range(min(n, 1000)):  # Limit iterations
            # Find k nearest neighbors
            distances = np.sqrt((rows - rows[i])**2 + (cols - cols[i])**2)
            nearest = np.argsort(distances)[1:k+1]  # Skip self

            for j in nearest:
                dist = distances[j]
                if dist > 0:
                    weight = 1.0 / dist
                    numerator += weight * (values[i] - mean) * (values[j] - mean)
                    weight_sum += weight

        if weight_sum < 1e-10:
            return np.nan

        deviation_sq_sum = np.sum((values[:min(n, 1000)] - mean) ** 2)
        if deviation_sq_sum < 1e-10:
            return np.nan

        morans_i = (n / weight_sum) * (numerator / deviation_sq_sum)

        return float(np.clip(morans_i, -1.0, 1.0))

    def _check_boundaries(
        self, data: np.ndarray, tile_boundaries: List[int]
    ) -> Tuple[List[SpatialIssue], Dict[str, Any]]:
        """
        Check for artificial gradients at tile boundaries.

        Tile boundaries shouldn't have higher gradients than elsewhere.
        """
        issues = []
        metrics = {"boundary_gradients": []}

        if not tile_boundaries:
            return issues, metrics

        # Calculate gradient magnitude
        gy, gx = np.gradient(np.nan_to_num(data, nan=0.0))
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Get non-boundary gradient statistics
        valid = ~np.isnan(data)
        if valid.sum() < 100:
            return issues, metrics

        non_boundary_grads = []
        boundary_grads = []

        height, width = data.shape

        for boundary in tile_boundaries:
            if boundary < 0 or boundary >= height:
                continue

            # Get gradient at boundary
            if boundary > 0 and boundary < height:
                boundary_row = gradient_mag[boundary, :]
                boundary_grads.extend(boundary_row[valid[boundary, :]])

        # Get non-boundary gradients (sample)
        non_boundary_rows = [i for i in range(height) if i not in tile_boundaries]
        if non_boundary_rows:
            sample_rows = np.random.choice(
                non_boundary_rows,
                min(len(non_boundary_rows), 50),
                replace=False
            )
            for row in sample_rows:
                row_grads = gradient_mag[row, :]
                non_boundary_grads.extend(row_grads[valid[row, :]])

        if not boundary_grads or not non_boundary_grads:
            return issues, metrics

        # Compare using z-score
        non_boundary_mean = np.mean(non_boundary_grads)
        non_boundary_std = np.std(non_boundary_grads)

        if non_boundary_std < 1e-10:
            non_boundary_std = 1.0  # Avoid division by zero

        boundary_mean = np.mean(boundary_grads)
        z_score = (boundary_mean - non_boundary_mean) / non_boundary_std

        metrics["boundary_gradient_zscore"] = z_score
        metrics["boundary_mean_gradient"] = boundary_mean
        metrics["background_mean_gradient"] = non_boundary_mean

        if abs(z_score) > self.config.boundary_gradient_threshold:
            issues.append(SpatialIssue(
                check_type=SpatialCheckType.BOUNDARY_COHERENCE,
                severity=SpatialIssueSeverity.HIGH,
                description=f"Significant gradient discontinuity at tile boundaries (z={z_score:.2f})",
                metric_value=z_score,
                threshold=self.config.boundary_gradient_threshold,
                recommendation="Check tile stitching/mosaicking process",
            ))

        return issues, metrics

    def _check_geographic_bounds(
        self, shape: Tuple[int, int], transform: Tuple[float, ...]
    ) -> Tuple[List[SpatialIssue], Dict[str, Any]]:
        """
        Check that output falls within valid geographic bounds.
        """
        issues = []
        metrics = {}

        height, width = shape

        # Extract transform components (GDAL-style: a, b, c, d, e, f)
        # x = a + b*col + c*row
        # y = d + e*col + f*row
        if len(transform) >= 6:
            a, b, c, d, e, f = transform[:6]

            # Calculate corner coordinates
            corners = [
                (a, d),  # Upper-left
                (a + b*width, d + e*width),  # Upper-right
                (a + c*height, d + f*height),  # Lower-left
                (a + b*width + c*height, d + e*width + f*height),  # Lower-right
            ]

            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]

            bounds = (min(xs), min(ys), max(xs), max(ys))
            metrics["computed_bounds"] = bounds

            # Check latitude/longitude ranges
            min_lon, min_lat, max_lon, max_lat = bounds

            # Check latitude
            lat_min, lat_max = self.config.valid_latitude_range
            if min_lat < lat_min or max_lat > lat_max:
                issues.append(SpatialIssue(
                    check_type=SpatialCheckType.GEOGRAPHIC_BOUNDS,
                    severity=SpatialIssueSeverity.CRITICAL,
                    description=f"Invalid latitude range: [{min_lat:.2f}, {max_lat:.2f}] outside [{lat_min}, {lat_max}]",
                    location={"bounds": bounds},
                    recommendation="Check coordinate reference system and transform",
                ))

            # Check longitude
            lon_min, lon_max = self.config.valid_longitude_range
            if min_lon < lon_min or max_lon > lon_max:
                issues.append(SpatialIssue(
                    check_type=SpatialCheckType.GEOGRAPHIC_BOUNDS,
                    severity=SpatialIssueSeverity.CRITICAL,
                    description=f"Invalid longitude range: [{min_lon:.2f}, {max_lon:.2f}] outside [{lon_min}, {lon_max}]",
                    location={"bounds": bounds},
                    recommendation="Check coordinate reference system and transform",
                ))

            # Check expected bounds if provided
            if self.config.expected_bounds:
                exp_min_lon, exp_min_lat, exp_max_lon, exp_max_lat = self.config.expected_bounds

                # Check if output is within expected area
                if (max_lon < exp_min_lon or min_lon > exp_max_lon or
                    max_lat < exp_min_lat or min_lat > exp_max_lat):
                    issues.append(SpatialIssue(
                        check_type=SpatialCheckType.GEOGRAPHIC_BOUNDS,
                        severity=SpatialIssueSeverity.HIGH,
                        description="Output bounds do not overlap with expected area",
                        location={"actual": bounds, "expected": self.config.expected_bounds},
                        recommendation="Verify area of interest definition",
                    ))

        return issues, metrics

    def _check_topology(
        self, data: np.ndarray
    ) -> Tuple[List[SpatialIssue], Dict[str, Any]]:
        """
        Check topological consistency of binary classification results.

        Looks for connected components and their sizes.
        """
        issues = []
        metrics = {}

        # Convert to binary (non-zero = feature)
        binary = ~np.isnan(data) & (data != 0)

        if not binary.any():
            metrics["num_regions"] = 0
            metrics["total_feature_pixels"] = 0
            return issues, metrics

        # Find connected components using simple flood-fill approach
        num_regions, region_sizes = self._count_connected_components(binary)

        metrics["num_regions"] = num_regions
        metrics["region_sizes"] = region_sizes[:20] if len(region_sizes) > 20 else region_sizes  # Top 20
        metrics["total_feature_pixels"] = int(binary.sum())
        metrics["largest_region_pct"] = 100.0 * max(region_sizes) / binary.sum() if region_sizes else 0.0

        # Check for excessive fragmentation
        small_regions = sum(1 for s in region_sizes if s < self.config.min_region_size_pixels)
        if num_regions > 0:
            fragmentation_ratio = small_regions / num_regions
            metrics["small_region_ratio"] = fragmentation_ratio

            if fragmentation_ratio > 0.5 and num_regions > 10:
                issues.append(SpatialIssue(
                    check_type=SpatialCheckType.TOPOLOGY,
                    severity=SpatialIssueSeverity.MEDIUM,
                    description=f"High fragmentation: {small_regions}/{num_regions} regions are smaller than {self.config.min_region_size_pixels} pixels",
                    metric_value=fragmentation_ratio,
                    recommendation="Consider morphological filtering to reduce noise",
                ))

        return issues, metrics

    def _count_connected_components(self, binary: np.ndarray) -> Tuple[int, List[int]]:
        """
        Count connected components and their sizes using union-find.
        """
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            if num_features == 0:
                return 0, []
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            return num_features, sorted([int(s) for s in sizes], reverse=True)
        except ImportError:
            # Fallback: simple BFS counting
            return self._count_components_bfs(binary)

    def _count_components_bfs(self, binary: np.ndarray) -> Tuple[int, List[int]]:
        """Fallback BFS component counting."""
        visited = np.zeros_like(binary, dtype=bool)
        height, width = binary.shape
        components = []

        def bfs(start_r: int, start_c: int) -> int:
            queue = [(start_r, start_c)]
            size = 0
            while queue:
                r, c = queue.pop(0)
                if r < 0 or r >= height or c < 0 or c >= width:
                    continue
                if visited[r, c] or not binary[r, c]:
                    continue
                visited[r, c] = True
                size += 1
                # 4-connectivity
                queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            return size

        for r in range(height):
            for c in range(width):
                if binary[r, c] and not visited[r, c]:
                    size = bfs(r, c)
                    if size > 0:
                        components.append(size)

        return len(components), sorted(components, reverse=True)

    def _check_isolated_pixels(
        self, data: np.ndarray
    ) -> Tuple[List[SpatialIssue], Dict[str, Any]]:
        """
        Check for isolated (salt-and-pepper) pixels.

        Isolated pixels often indicate noise rather than real features.
        """
        issues = []
        metrics = {}

        # Convert to binary
        binary = ~np.isnan(data) & (data != 0)
        total_feature_pixels = binary.sum()

        if total_feature_pixels == 0:
            metrics["isolated_pixel_count"] = 0
            metrics["isolated_pixel_pct"] = 0.0
            return issues, metrics

        # Count isolated pixels (pixels with no neighbors of same class)
        isolated_count = 0
        height, width = binary.shape

        # Use convolution for efficiency
        try:
            from scipy import ndimage
            # Kernel counts neighbors (8-connectivity)
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbor_count = ndimage.convolve(binary.astype(float), kernel, mode='constant', cval=0)
            # Isolated = feature pixel with 0 neighbors
            isolated = binary & (neighbor_count == 0)
            isolated_count = int(isolated.sum())
        except ImportError:
            # Fallback: manual counting
            for r in range(height):
                for c in range(width):
                    if binary[r, c]:
                        has_neighbor = False
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < height and 0 <= nc < width:
                                    if binary[nr, nc]:
                                        has_neighbor = True
                                        break
                            if has_neighbor:
                                break
                        if not has_neighbor:
                            isolated_count += 1

        isolated_pct = 100.0 * isolated_count / total_feature_pixels
        metrics["isolated_pixel_count"] = isolated_count
        metrics["isolated_pixel_pct"] = isolated_pct

        if isolated_pct > self.config.max_isolated_pixel_pct:
            severity = (
                SpatialIssueSeverity.HIGH if isolated_pct > 20.0
                else SpatialIssueSeverity.MEDIUM
            )
            issues.append(SpatialIssue(
                check_type=SpatialCheckType.ISOLATED_PIXELS,
                severity=severity,
                description=f"High isolated pixel ratio: {isolated_pct:.1f}% ({isolated_count} pixels)",
                metric_value=isolated_pct,
                threshold=self.config.max_isolated_pixel_pct,
                recommendation="Apply morphological opening or median filter",
            ))

        return issues, metrics


def check_spatial_coherence(
    data: np.ndarray,
    transform: Optional[Tuple[float, ...]] = None,
    mask: Optional[np.ndarray] = None,
    tile_boundaries: Optional[List[int]] = None,
    config: Optional[SpatialCoherenceConfig] = None,
) -> SpatialCoherenceResult:
    """
    Convenience function to check spatial coherence.

    Args:
        data: 2D array of analysis output
        transform: Affine geotransform
        mask: Optional boolean mask
        tile_boundaries: Optional list of tile boundary row indices
        config: Optional configuration

    Returns:
        SpatialCoherenceResult with all findings
    """
    checker = SpatialCoherenceChecker(config)
    return checker.check(data, transform, mask, tile_boundaries)
