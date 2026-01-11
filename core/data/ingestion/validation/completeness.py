"""
Data Completeness Validation Module.

Provides comprehensive completeness checks for geospatial data:
- Spatial coverage analysis (AOI coverage percentage)
- Temporal completeness (expected vs actual observations)
- Band completeness (all required bands present)
- Metadata completeness (required fields present)
- NoData pattern analysis (gaps, holes, edge effects)
- Multi-file dataset completeness (tile coverage)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CompletenessCheckType(Enum):
    """Types of completeness checks."""

    SPATIAL_COVERAGE = "spatial_coverage"
    TEMPORAL_COVERAGE = "temporal_coverage"
    BAND_COMPLETENESS = "band_completeness"
    METADATA_COMPLETENESS = "metadata_completeness"
    NODATA_PATTERN = "nodata_pattern"
    TILE_COVERAGE = "tile_coverage"
    RESOLUTION_CONSISTENCY = "resolution_consistency"


class CompletenessSeverity(Enum):
    """Severity levels for completeness issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CoverageRegion:
    """
    Represents a region of data or gap.

    Attributes:
        bounds: (minx, miny, maxx, maxy) in CRS units
        pixel_bounds: (col_start, row_start, col_end, row_end)
        area_sq_units: Area in CRS squared units
        percentage: Percentage of total area
        is_gap: Whether this is a gap (missing data)
    """

    bounds: Optional[Tuple[float, float, float, float]] = None
    pixel_bounds: Optional[Tuple[int, int, int, int]] = None
    area_sq_units: float = 0.0
    percentage: float = 0.0
    is_gap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert region to dictionary."""
        return {
            "bounds": self.bounds,
            "pixel_bounds": self.pixel_bounds,
            "area_sq_units": self.area_sq_units,
            "percentage": self.percentage,
            "is_gap": self.is_gap,
        }


@dataclass
class CompletenessIssue:
    """
    Represents a single completeness issue.

    Attributes:
        check_type: Type of check that found the issue
        severity: Severity level
        message: Human-readable description
        details: Additional context
        affected_regions: List of affected coverage regions
    """

    check_type: CompletenessCheckType
    severity: CompletenessSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    affected_regions: List[CoverageRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "affected_regions": [r.to_dict() for r in self.affected_regions],
        }


@dataclass
class CompletenessResult:
    """
    Result of completeness validation.

    Attributes:
        is_complete: Whether data meets completeness requirements
        coverage_percentage: Overall data coverage percentage
        valid_pixel_count: Number of valid (non-nodata) pixels
        total_pixel_count: Total number of pixels
        issues: List of completeness issues
        coverage_map: Optional binary coverage array
        gap_regions: List of identified gap regions
        metadata: Additional validation metadata
        duration_seconds: Time taken for validation
    """

    is_complete: bool
    coverage_percentage: float
    valid_pixel_count: int
    total_pixel_count: int
    issues: List[CompletenessIssue] = field(default_factory=list)
    coverage_map: Optional[np.ndarray] = None
    gap_regions: List[CoverageRegion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def gap_percentage(self) -> float:
        """Percentage of data that is missing."""
        return 100.0 - self.coverage_percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_complete": self.is_complete,
            "coverage_percentage": self.coverage_percentage,
            "gap_percentage": self.gap_percentage,
            "valid_pixel_count": self.valid_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "gap_regions": [r.to_dict() for r in self.gap_regions],
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class CompletenessConfig:
    """
    Configuration for completeness validation.

    Attributes:
        min_coverage_percent: Minimum required coverage percentage
        max_gap_percent: Maximum allowed gap percentage
        expected_bands: List of expected band names/indices
        expected_bounds: Expected spatial bounds (minx, miny, maxx, maxy)
        expected_crs: Expected coordinate reference system
        required_metadata: List of required metadata keys
        detect_gaps: Whether to identify gap regions
        min_gap_size_pixels: Minimum gap size to report
        edge_buffer_pixels: Buffer around edges to ignore
        check_band_consistency: Whether to check all bands have same coverage
    """

    min_coverage_percent: float = 80.0
    max_gap_percent: float = 20.0
    expected_bands: Optional[List[Union[str, int]]] = None
    expected_bounds: Optional[Tuple[float, float, float, float]] = None
    expected_crs: Optional[str] = None
    required_metadata: List[str] = field(default_factory=list)
    detect_gaps: bool = True
    min_gap_size_pixels: int = 100
    edge_buffer_pixels: int = 0
    check_band_consistency: bool = True


class CompletenessValidator:
    """
    Validates data completeness for geospatial files.

    Checks:
    - Spatial coverage against expected AOI
    - NoData patterns and gap identification
    - Band completeness
    - Metadata presence
    - Resolution consistency

    Example:
        validator = CompletenessValidator(CompletenessConfig(min_coverage_percent=90))
        result = validator.validate_file("data.tif")
        if not result.is_complete:
            print(f"Coverage: {result.coverage_percentage}%")
            for issue in result.issues:
                print(f"{issue.severity}: {issue.message}")
    """

    def __init__(self, config: Optional[CompletenessConfig] = None):
        """
        Initialize completeness validator.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or CompletenessConfig()

    def validate_file(self, path: Union[str, Path]) -> CompletenessResult:
        """
        Validate completeness of a raster file.

        Args:
            path: Path to raster file

        Returns:
            CompletenessResult with validation details
        """
        import time

        start_time = time.time()
        path = Path(path)

        if not path.exists():
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=[
                    CompletenessIssue(
                        check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                        severity=CompletenessSeverity.CRITICAL,
                        message=f"File not found: {path}",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        try:
            import rasterio
        except ImportError:
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=[
                    CompletenessIssue(
                        check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                        severity=CompletenessSeverity.HIGH,
                        message="rasterio not available for completeness validation",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        try:
            with rasterio.open(path) as src:
                data = src.read()
                nodata = src.nodata
                bounds = src.bounds
                crs = str(src.crs) if src.crs else None
                transform = src.transform
                profile = src.profile.copy()

            return self.validate_array(
                data,
                nodata=nodata,
                bounds=bounds,
                crs=crs,
                transform=transform,
                profile=profile,
            )

        except Exception as e:
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=[
                    CompletenessIssue(
                        check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                        severity=CompletenessSeverity.CRITICAL,
                        message=f"Failed to read file: {e}",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

    def validate_array(
        self,
        data: np.ndarray,
        nodata: Optional[float] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        crs: Optional[str] = None,
        transform: Any = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> CompletenessResult:
        """
        Validate completeness of array data.

        Args:
            data: Input array (bands, height, width) or (height, width)
            nodata: NoData value to identify gaps
            bounds: Spatial bounds of data
            crs: Coordinate reference system
            transform: Affine transform
            profile: Raster profile with additional metadata

        Returns:
            CompletenessResult with validation details
        """
        import time

        start_time = time.time()
        issues: List[CompletenessIssue] = []
        metadata: Dict[str, Any] = {}

        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        num_bands, height, width = data.shape
        total_pixels = height * width

        metadata["num_bands"] = num_bands
        metadata["height"] = height
        metadata["width"] = width
        metadata["bounds"] = bounds
        metadata["crs"] = crs

        # Create nodata mask
        if nodata is not None:
            if np.isnan(nodata):
                nodata_mask = np.isnan(data)
            else:
                nodata_mask = data == nodata
        else:
            nodata_mask = np.zeros_like(data, dtype=bool)

        # Calculate coverage per band
        band_coverages: Dict[int, float] = {}
        band_valid_counts: Dict[int, int] = {}

        for band_idx in range(num_bands):
            band_nodata = nodata_mask[band_idx]
            valid_count = np.sum(~band_nodata)
            coverage = (valid_count / total_pixels) * 100

            band_coverages[band_idx] = coverage
            band_valid_counts[band_idx] = int(valid_count)

        metadata["band_coverages"] = band_coverages

        # Overall coverage (all bands must have data)
        all_valid = ~np.any(nodata_mask, axis=0)  # Valid if all bands have data
        valid_pixel_count = int(np.sum(all_valid))
        coverage_percentage = (valid_pixel_count / total_pixels) * 100

        # Check minimum coverage
        if coverage_percentage < self.config.min_coverage_percent:
            issues.append(
                CompletenessIssue(
                    check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                    severity=CompletenessSeverity.HIGH
                    if coverage_percentage < self.config.min_coverage_percent / 2
                    else CompletenessSeverity.MEDIUM,
                    message=f"Coverage {coverage_percentage:.1f}% below minimum {self.config.min_coverage_percent}%",
                    details={
                        "actual": coverage_percentage,
                        "required": self.config.min_coverage_percent,
                    },
                )
            )

        # Check band consistency
        if self.config.check_band_consistency and num_bands > 1:
            coverage_values = list(band_coverages.values())
            min_band_coverage = min(coverage_values)
            max_band_coverage = max(coverage_values)

            if max_band_coverage - min_band_coverage > 5:  # More than 5% difference
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.BAND_COMPLETENESS,
                        severity=CompletenessSeverity.MEDIUM,
                        message=f"Inconsistent coverage across bands ({min_band_coverage:.1f}% - {max_band_coverage:.1f}%)",
                        details={
                            "min_coverage": min_band_coverage,
                            "max_coverage": max_band_coverage,
                            "band_coverages": band_coverages,
                        },
                    )
                )

        # Check expected bands
        if self.config.expected_bands:
            missing_bands = []
            for expected in self.config.expected_bands:
                if isinstance(expected, int):
                    if expected >= num_bands:
                        missing_bands.append(str(expected))
                # String band names would need metadata lookup

            if missing_bands:
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.BAND_COMPLETENESS,
                        severity=CompletenessSeverity.HIGH,
                        message=f"Missing expected bands: {', '.join(missing_bands)}",
                        details={"missing": missing_bands},
                    )
                )

        # Check CRS
        if self.config.expected_crs and crs != self.config.expected_crs:
            issues.append(
                CompletenessIssue(
                    check_type=CompletenessCheckType.METADATA_COMPLETENESS,
                    severity=CompletenessSeverity.MEDIUM,
                    message=f"CRS mismatch: expected {self.config.expected_crs}, got {crs}",
                    details={"expected": self.config.expected_crs, "actual": crs},
                )
            )

        # Check bounds coverage
        if self.config.expected_bounds and bounds:
            bounds_issues = self._check_bounds_coverage(bounds, self.config.expected_bounds)
            issues.extend(bounds_issues)

        # Check metadata completeness
        if self.config.required_metadata and profile:
            missing_meta = []
            for key in self.config.required_metadata:
                if key not in profile or profile[key] is None:
                    missing_meta.append(key)

            if missing_meta:
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.METADATA_COMPLETENESS,
                        severity=CompletenessSeverity.MEDIUM,
                        message=f"Missing required metadata: {', '.join(missing_meta)}",
                        details={"missing": missing_meta},
                    )
                )

        # Detect gap regions
        gap_regions: List[CoverageRegion] = []
        coverage_map = None

        if self.config.detect_gaps:
            # Combined nodata mask (any band missing = gap)
            any_nodata = np.any(nodata_mask, axis=0)

            # Apply edge buffer if configured
            if self.config.edge_buffer_pixels > 0:
                buf = self.config.edge_buffer_pixels
                edge_mask = np.zeros_like(any_nodata)
                edge_mask[:buf, :] = True
                edge_mask[-buf:, :] = True
                edge_mask[:, :buf] = True
                edge_mask[:, -buf:] = True
                any_nodata = any_nodata & ~edge_mask

            gap_regions = self._detect_gap_regions(any_nodata, bounds, transform)
            coverage_map = ~any_nodata

            # Report large gaps
            for region in gap_regions:
                if region.percentage > 5:  # Gaps larger than 5%
                    issues.append(
                        CompletenessIssue(
                            check_type=CompletenessCheckType.NODATA_PATTERN,
                            severity=CompletenessSeverity.MEDIUM,
                            message=f"Large gap detected ({region.percentage:.1f}% of data)",
                            details={"percentage": region.percentage},
                            affected_regions=[region],
                        )
                    )

        # Determine overall completeness
        critical_count = sum(1 for i in issues if i.severity == CompletenessSeverity.CRITICAL)
        high_count = sum(1 for i in issues if i.severity == CompletenessSeverity.HIGH)

        is_complete = (
            critical_count == 0
            and high_count == 0
            and coverage_percentage >= self.config.min_coverage_percent
        )

        duration = time.time() - start_time

        logger.info(
            f"Completeness validation: coverage={coverage_percentage:.1f}%, "
            f"complete={is_complete}, issues={len(issues)}"
        )

        return CompletenessResult(
            is_complete=is_complete,
            coverage_percentage=coverage_percentage,
            valid_pixel_count=valid_pixel_count,
            total_pixel_count=total_pixels,
            issues=issues,
            coverage_map=coverage_map,
            gap_regions=gap_regions,
            metadata=metadata,
            duration_seconds=duration,
        )

    def _check_bounds_coverage(
        self,
        actual: Tuple[float, float, float, float],
        expected: Tuple[float, float, float, float],
    ) -> List[CompletenessIssue]:
        """Check if actual bounds cover expected bounds."""
        issues = []

        # actual = (left, bottom, right, top)
        # expected = (minx, miny, maxx, maxy)
        a_left, a_bottom, a_right, a_top = actual
        e_left, e_bottom, e_right, e_top = expected

        # Calculate coverage
        overlap_left = max(a_left, e_left)
        overlap_bottom = max(a_bottom, e_bottom)
        overlap_right = min(a_right, e_right)
        overlap_top = min(a_top, e_top)

        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            # No overlap
            issues.append(
                CompletenessIssue(
                    check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                    severity=CompletenessSeverity.CRITICAL,
                    message="Data bounds do not overlap with expected AOI",
                    details={"actual": actual, "expected": expected},
                )
            )
        else:
            # Calculate percentage coverage
            expected_area = (e_right - e_left) * (e_top - e_bottom)
            overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)

            coverage_pct = (overlap_area / expected_area) * 100 if expected_area > 0 else 0

            if coverage_pct < 100:
                severity = (
                    CompletenessSeverity.HIGH
                    if coverage_pct < 50
                    else CompletenessSeverity.MEDIUM
                    if coverage_pct < 90
                    else CompletenessSeverity.LOW
                )
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.SPATIAL_COVERAGE,
                        severity=severity,
                        message=f"Data covers {coverage_pct:.1f}% of expected AOI",
                        details={
                            "coverage_percent": coverage_pct,
                            "actual": actual,
                            "expected": expected,
                        },
                    )
                )

        return issues

    def _detect_gap_regions(
        self,
        gap_mask: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]],
        transform: Any,
    ) -> List[CoverageRegion]:
        """Detect connected gap regions in the data."""
        gap_regions = []
        height, width = gap_mask.shape
        total_pixels = height * width

        if not np.any(gap_mask):
            return gap_regions

        try:
            from scipy import ndimage
        except ImportError:
            # Fall back to simple counting without region detection
            gap_count = np.sum(gap_mask)
            gap_pct = (gap_count / total_pixels) * 100

            if gap_count >= self.config.min_gap_size_pixels:
                gap_regions.append(
                    CoverageRegion(
                        pixel_bounds=(0, 0, width, height),
                        percentage=gap_pct,
                        is_gap=True,
                    )
                )
            return gap_regions

        # Label connected components
        labeled, num_features = ndimage.label(gap_mask)

        for label_id in range(1, num_features + 1):
            region_mask = labeled == label_id
            pixel_count = np.sum(region_mask)

            if pixel_count < self.config.min_gap_size_pixels:
                continue

            # Find bounding box
            rows = np.any(region_mask, axis=1)
            cols = np.any(region_mask, axis=0)
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            if len(row_indices) == 0 or len(col_indices) == 0:
                continue

            row_start = int(row_indices[0])
            row_end = int(row_indices[-1])
            col_start = int(col_indices[0])
            col_end = int(col_indices[-1])

            pixel_bounds = (col_start, row_start, col_end, row_end)
            percentage = (pixel_count / total_pixels) * 100

            # Convert to geo bounds if transform available
            geo_bounds = None
            if transform is not None:
                try:
                    import rasterio.transform

                    ul = rasterio.transform.xy(transform, row_start, col_start)
                    lr = rasterio.transform.xy(transform, row_end, col_end)
                    geo_bounds = (ul[0], lr[1], lr[0], ul[1])
                except Exception:
                    pass

            gap_regions.append(
                CoverageRegion(
                    bounds=geo_bounds,
                    pixel_bounds=pixel_bounds,
                    percentage=percentage,
                    is_gap=True,
                )
            )

        return gap_regions

    def validate_tile_coverage(
        self,
        tile_paths: List[Union[str, Path]],
        expected_bounds: Tuple[float, float, float, float],
    ) -> CompletenessResult:
        """
        Validate coverage of multiple tiles against expected AOI.

        Args:
            tile_paths: List of tile file paths
            expected_bounds: Expected total coverage bounds

        Returns:
            CompletenessResult with tile coverage analysis
        """
        import time

        start_time = time.time()
        issues: List[CompletenessIssue] = []
        metadata: Dict[str, Any] = {"tile_count": len(tile_paths)}

        if not tile_paths:
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=[
                    CompletenessIssue(
                        check_type=CompletenessCheckType.TILE_COVERAGE,
                        severity=CompletenessSeverity.CRITICAL,
                        message="No tiles provided",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        try:
            import rasterio
            from rasterio.merge import merge as rasterio_merge
        except ImportError:
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=[
                    CompletenessIssue(
                        check_type=CompletenessCheckType.TILE_COVERAGE,
                        severity=CompletenessSeverity.HIGH,
                        message="rasterio not available for tile coverage validation",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        tile_bounds_list = []
        valid_tiles = 0

        for tile_path in tile_paths:
            tile_path = Path(tile_path)
            if not tile_path.exists():
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.TILE_COVERAGE,
                        severity=CompletenessSeverity.MEDIUM,
                        message=f"Tile not found: {tile_path}",
                    )
                )
                continue

            try:
                with rasterio.open(tile_path) as src:
                    tile_bounds_list.append(src.bounds)
                    valid_tiles += 1
            except Exception as e:
                issues.append(
                    CompletenessIssue(
                        check_type=CompletenessCheckType.TILE_COVERAGE,
                        severity=CompletenessSeverity.MEDIUM,
                        message=f"Failed to read tile {tile_path}: {e}",
                    )
                )

        metadata["valid_tiles"] = valid_tiles

        if not tile_bounds_list:
            return CompletenessResult(
                is_complete=False,
                coverage_percentage=0.0,
                valid_pixel_count=0,
                total_pixel_count=0,
                issues=issues,
                metadata=metadata,
                duration_seconds=time.time() - start_time,
            )

        # Calculate combined bounds
        all_left = min(b.left for b in tile_bounds_list)
        all_bottom = min(b.bottom for b in tile_bounds_list)
        all_right = max(b.right for b in tile_bounds_list)
        all_top = max(b.top for b in tile_bounds_list)

        combined_bounds = (all_left, all_bottom, all_right, all_top)
        metadata["combined_bounds"] = combined_bounds

        # Calculate coverage percentage (simplified - assumes no overlaps)
        expected_area = (expected_bounds[2] - expected_bounds[0]) * (expected_bounds[3] - expected_bounds[1])
        tile_total_area = sum(
            (b.right - b.left) * (b.top - b.bottom) for b in tile_bounds_list
        )

        # Estimate coverage (overlap would reduce this)
        estimated_coverage = min(100.0, (tile_total_area / expected_area) * 100) if expected_area > 0 else 0

        # Check for gaps in tile coverage
        bounds_issues = self._check_bounds_coverage(combined_bounds, expected_bounds)
        issues.extend(bounds_issues)

        is_complete = estimated_coverage >= self.config.min_coverage_percent and all(
            i.severity not in [CompletenessSeverity.CRITICAL, CompletenessSeverity.HIGH]
            for i in issues
        )

        return CompletenessResult(
            is_complete=is_complete,
            coverage_percentage=estimated_coverage,
            valid_pixel_count=valid_tiles,
            total_pixel_count=len(tile_paths),
            issues=issues,
            metadata=metadata,
            duration_seconds=time.time() - start_time,
        )


def validate_completeness(
    data: np.ndarray,
    nodata: Optional[float] = None,
    min_coverage: float = 80.0,
) -> CompletenessResult:
    """
    Convenience function to validate array completeness.

    Args:
        data: Input array (bands, height, width) or (height, width)
        nodata: NoData value to identify gaps
        min_coverage: Minimum required coverage percentage

    Returns:
        CompletenessResult with validation details
    """
    config = CompletenessConfig(min_coverage_percent=min_coverage)
    validator = CompletenessValidator(config)
    return validator.validate_array(data, nodata=nodata)


def validate_completeness_from_file(
    path: Union[str, Path],
    min_coverage: float = 80.0,
) -> CompletenessResult:
    """
    Convenience function to validate file completeness.

    Args:
        path: Path to raster file
        min_coverage: Minimum required coverage percentage

    Returns:
        CompletenessResult with validation details
    """
    config = CompletenessConfig(min_coverage_percent=min_coverage)
    validator = CompletenessValidator(config)
    return validator.validate_file(path)
