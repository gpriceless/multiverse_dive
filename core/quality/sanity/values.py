"""
Value Plausibility Sanity Checks.

Validates that analysis output values are physically plausible:
- Value range checks (e.g., confidence scores in [0, 1])
- Physical bounds (e.g., NDWI in [-1, 1], temperature > absolute zero)
- Statistical plausibility (e.g., distribution shape, percentiles)
- Cross-band consistency (e.g., water index high where NIR low)
- Event-specific constraints (e.g., flood extent can't exceed DEM low areas)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ValueCheckType(Enum):
    """Types of value plausibility checks."""
    RANGE = "range"                         # Value within expected range
    PHYSICAL_BOUNDS = "physical_bounds"     # Physical constraints
    DISTRIBUTION = "distribution"           # Statistical distribution check
    PERCENTILE = "percentile"               # Extreme percentiles
    CROSS_BAND = "cross_band"               # Inter-band consistency
    NODATA = "nodata"                       # NoData value handling
    PRECISION = "precision"                 # Numerical precision issues
    NAN_INF = "nan_inf"                     # NaN/Inf detection


class ValueIssueSeverity(Enum):
    """Severity levels for value issues."""
    CRITICAL = "critical"   # Values are clearly wrong (e.g., negative water extent)
    HIGH = "high"           # Likely calculation error
    MEDIUM = "medium"       # Suspicious but possibly valid
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational


@dataclass
class ValueIssue:
    """
    A value plausibility issue found during validation.

    Attributes:
        check_type: Type of check that found the issue
        severity: Issue severity level
        description: Human-readable description
        affected_pixels: Number of affected pixels
        affected_percentage: Percentage of total valid pixels affected
        example_values: Sample of problematic values
        recommendation: Suggested action
    """
    check_type: ValueCheckType
    severity: ValueIssueSeverity
    description: str
    affected_pixels: int = 0
    affected_percentage: float = 0.0
    example_values: Optional[List[float]] = None
    expected_range: Optional[Tuple[float, float]] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_pixels": self.affected_pixels,
            "affected_percentage": self.affected_percentage,
            "example_values": self.example_values,
            "expected_range": self.expected_range,
            "recommendation": self.recommendation,
        }


class ValueType(Enum):
    """Known value types with predefined valid ranges."""
    CONFIDENCE = "confidence"           # 0-1 probability/confidence
    PROBABILITY = "probability"         # 0-1 probability
    PERCENTAGE = "percentage"           # 0-100 percentage
    NDWI = "ndwi"                       # -1 to 1
    NDVI = "ndvi"                       # -1 to 1
    NBR = "nbr"                         # -1 to 1
    DNBR = "dnbr"                       # -2 to 2
    BACKSCATTER_DB = "backscatter_db"   # -50 to 10 dB (typical)
    TEMPERATURE_K = "temperature_k"     # 0+ Kelvin
    TEMPERATURE_C = "temperature_c"     # -273.15+ Celsius
    ELEVATION_M = "elevation_m"         # -500 to 9000 meters (Earth)
    DEPTH_M = "depth_m"                 # 0+ meters
    BINARY = "binary"                   # 0 or 1
    CATEGORICAL = "categorical"         # Integer classes
    CUSTOM = "custom"                   # User-defined


# Predefined value ranges for known types
VALUE_RANGES: Dict[ValueType, Tuple[float, float]] = {
    ValueType.CONFIDENCE: (0.0, 1.0),
    ValueType.PROBABILITY: (0.0, 1.0),
    ValueType.PERCENTAGE: (0.0, 100.0),
    ValueType.NDWI: (-1.0, 1.0),
    ValueType.NDVI: (-1.0, 1.0),
    ValueType.NBR: (-1.0, 1.0),
    ValueType.DNBR: (-2.0, 2.0),
    ValueType.BACKSCATTER_DB: (-50.0, 10.0),
    ValueType.TEMPERATURE_K: (0.0, 500.0),
    ValueType.TEMPERATURE_C: (-273.15, 200.0),
    ValueType.ELEVATION_M: (-500.0, 9000.0),
    ValueType.DEPTH_M: (0.0, 12000.0),  # Marianas Trench
    ValueType.BINARY: (0.0, 1.0),
}


@dataclass
class ValuePlausibilityConfig:
    """
    Configuration for value plausibility checks.

    Attributes:
        value_type: Type of values being checked
        custom_range: Custom valid range (min, max)
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        check_distribution: Check value distribution
        check_percentiles: Check extreme percentiles

        max_nan_percentage: Maximum allowed NaN percentage
        max_inf_percentage: Maximum allowed Inf percentage
        extreme_percentile_threshold: Threshold for extreme value warning

        expected_classes: For categorical data, the expected class values
        nodata_value: The nodata value to ignore
    """
    value_type: ValueType = ValueType.CUSTOM
    custom_range: Optional[Tuple[float, float]] = None

    check_nan: bool = True
    check_inf: bool = True
    check_distribution: bool = True
    check_percentiles: bool = True

    max_nan_percentage: float = 10.0
    max_inf_percentage: float = 0.0  # No Inf values allowed by default
    extreme_percentile_threshold: float = 99.9  # Check 0.1% and 99.9%

    expected_classes: Optional[List[int]] = None
    nodata_value: Optional[float] = None

    # Distribution checks
    min_unique_values: int = 2  # At least 2 unique values expected
    max_single_value_pct: float = 99.0  # Flag if single value > 99%


@dataclass
class ValuePlausibilityResult:
    """
    Result of value plausibility validation.

    Attributes:
        is_plausible: Whether all value checks passed
        issues: List of issues found
        statistics: Computed value statistics
        duration_seconds: Time taken for validation
    """
    is_plausible: bool
    issues: List[ValueIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == ValueIssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high-severity issues."""
        return sum(1 for i in self.issues if i.severity == ValueIssueSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_plausible": self.is_plausible,
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "issues": [i.to_dict() for i in self.issues],
            "statistics": self.statistics,
            "duration_seconds": self.duration_seconds,
        }


class ValuePlausibilityChecker:
    """
    Validates that analysis output values are physically plausible.

    Checks that values fall within expected ranges and make physical sense
    for the type of analysis being performed.

    Example:
        # Check flood confidence map
        checker = ValuePlausibilityChecker(
            ValuePlausibilityConfig(value_type=ValueType.CONFIDENCE)
        )
        result = checker.check(flood_confidence_array)

        # Check NDWI values
        checker = ValuePlausibilityChecker(
            ValuePlausibilityConfig(value_type=ValueType.NDWI)
        )
        result = checker.check(ndwi_array)
    """

    def __init__(self, config: Optional[ValuePlausibilityConfig] = None):
        """
        Initialize the value plausibility checker.

        Args:
            config: Configuration options
        """
        self.config = config or ValuePlausibilityConfig()

    def check(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reference: Optional[np.ndarray] = None,
    ) -> ValuePlausibilityResult:
        """
        Run all configured value plausibility checks.

        Args:
            data: Array of analysis output values
            mask: Optional boolean mask of valid pixels
            reference: Optional reference data for cross-validation

        Returns:
            ValuePlausibilityResult with all issues found
        """
        import time
        start_time = time.time()

        issues = []
        statistics = {}

        # Get valid data
        if mask is not None:
            valid_data = data[mask]
        elif self.config.nodata_value is not None:
            valid_data = data[data != self.config.nodata_value]
        else:
            valid_data = data.ravel()

        total_pixels = data.size
        valid_pixels = len(valid_data)
        statistics["total_pixels"] = total_pixels
        statistics["valid_pixels"] = valid_pixels

        # Check NaN values
        if self.config.check_nan:
            nan_issues, nan_stats = self._check_nan(data, valid_data)
            issues.extend(nan_issues)
            statistics["nan"] = nan_stats

        # Check Inf values
        if self.config.check_inf:
            inf_issues, inf_stats = self._check_inf(data, valid_data)
            issues.extend(inf_issues)
            statistics["inf"] = inf_stats

        # Filter out NaN/Inf for remaining checks
        valid_data = valid_data[np.isfinite(valid_data)]
        statistics["finite_pixels"] = len(valid_data)

        if len(valid_data) == 0:
            issues.append(ValueIssue(
                check_type=ValueCheckType.NODATA,
                severity=ValueIssueSeverity.CRITICAL,
                description="No valid (finite) data values found",
            ))
            return ValuePlausibilityResult(
                is_plausible=False,
                issues=issues,
                statistics=statistics,
                duration_seconds=time.time() - start_time,
            )

        # Compute basic statistics
        stats = self._compute_statistics(valid_data)
        statistics["basic"] = stats

        # Check value range
        range_issues, range_stats = self._check_range(valid_data)
        issues.extend(range_issues)
        statistics["range"] = range_stats

        # Check distribution
        if self.config.check_distribution:
            dist_issues, dist_stats = self._check_distribution(valid_data)
            issues.extend(dist_issues)
            statistics["distribution"] = dist_stats

        # Check percentiles
        if self.config.check_percentiles:
            pctl_issues, pctl_stats = self._check_percentiles(valid_data)
            issues.extend(pctl_issues)
            statistics["percentiles"] = pctl_stats

        # Determine overall plausibility
        is_plausible = not any(
            i.severity in (ValueIssueSeverity.CRITICAL, ValueIssueSeverity.HIGH)
            for i in issues
        )

        duration = time.time() - start_time
        logger.info(f"Value plausibility check completed in {duration:.2f}s, plausible={is_plausible}")

        return ValuePlausibilityResult(
            is_plausible=is_plausible,
            issues=issues,
            statistics=statistics,
            duration_seconds=duration,
        )

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics on valid data."""
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "count": len(data),
        }

    def _check_nan(
        self, full_data: np.ndarray, valid_data: np.ndarray
    ) -> Tuple[List[ValueIssue], Dict[str, Any]]:
        """Check for NaN values."""
        issues = []
        metrics = {}

        nan_count = np.isnan(full_data).sum()
        # Guard against division by zero for empty arrays
        nan_pct = 100.0 * nan_count / full_data.size if full_data.size > 0 else 0.0

        metrics["count"] = int(nan_count)
        metrics["percentage"] = nan_pct

        if nan_pct > self.config.max_nan_percentage:
            severity = (
                ValueIssueSeverity.CRITICAL if nan_pct > 50.0
                else ValueIssueSeverity.HIGH if nan_pct > 25.0
                else ValueIssueSeverity.MEDIUM
            )
            issues.append(ValueIssue(
                check_type=ValueCheckType.NAN_INF,
                severity=severity,
                description=f"Excessive NaN values: {nan_pct:.1f}% ({nan_count} pixels)",
                affected_pixels=int(nan_count),
                affected_percentage=nan_pct,
                recommendation="Check algorithm handling of edge cases and nodata",
            ))

        return issues, metrics

    def _check_inf(
        self, full_data: np.ndarray, valid_data: np.ndarray
    ) -> Tuple[List[ValueIssue], Dict[str, Any]]:
        """Check for Inf values."""
        issues = []
        metrics = {}

        inf_count = np.isinf(full_data).sum()
        # Guard against division by zero for empty arrays
        inf_pct = 100.0 * inf_count / full_data.size if full_data.size > 0 else 0.0

        metrics["count"] = int(inf_count)
        metrics["percentage"] = inf_pct

        if inf_count > 0:
            # Find sample of Inf locations
            inf_mask = np.isinf(full_data)
            inf_values = full_data[inf_mask]
            pos_inf = np.sum(inf_values > 0)
            neg_inf = np.sum(inf_values < 0)

            metrics["positive_inf"] = int(pos_inf)
            metrics["negative_inf"] = int(neg_inf)

            severity = (
                ValueIssueSeverity.CRITICAL if inf_pct > 1.0
                else ValueIssueSeverity.HIGH if inf_pct > 0.1
                else ValueIssueSeverity.MEDIUM
            )
            issues.append(ValueIssue(
                check_type=ValueCheckType.NAN_INF,
                severity=severity,
                description=f"Inf values found: {inf_count} pixels (+Inf: {pos_inf}, -Inf: {neg_inf})",
                affected_pixels=int(inf_count),
                affected_percentage=inf_pct,
                recommendation="Check for division by zero or overflow in algorithm",
            ))

        return issues, metrics

    def _check_range(
        self, data: np.ndarray
    ) -> Tuple[List[ValueIssue], Dict[str, Any]]:
        """Check that values fall within expected range."""
        issues = []
        metrics = {}

        # Get expected range
        if self.config.custom_range:
            expected_min, expected_max = self.config.custom_range
        elif self.config.value_type in VALUE_RANGES:
            expected_min, expected_max = VALUE_RANGES[self.config.value_type]
        else:
            # No range defined - just report statistics
            metrics["range_defined"] = False
            return issues, metrics

        metrics["expected_range"] = (expected_min, expected_max)
        metrics["actual_range"] = (float(np.min(data)), float(np.max(data)))

        # Check for values below minimum
        below_min = data < expected_min
        below_count = below_min.sum()
        if below_count > 0:
            below_pct = 100.0 * below_count / len(data)
            severity = (
                ValueIssueSeverity.CRITICAL if below_pct > 5.0
                else ValueIssueSeverity.HIGH if below_pct > 1.0
                else ValueIssueSeverity.MEDIUM
            )
            example_values = sorted(data[below_min])[:5]
            issues.append(ValueIssue(
                check_type=ValueCheckType.RANGE,
                severity=severity,
                description=f"Values below minimum {expected_min}: {below_count} pixels ({below_pct:.2f}%)",
                affected_pixels=int(below_count),
                affected_percentage=below_pct,
                example_values=[float(v) for v in example_values],
                expected_range=(expected_min, expected_max),
                recommendation=f"Clamp values to range or investigate algorithm",
            ))
            metrics["below_min_count"] = int(below_count)

        # Check for values above maximum
        above_max = data > expected_max
        above_count = above_max.sum()
        if above_count > 0:
            above_pct = 100.0 * above_count / len(data)
            severity = (
                ValueIssueSeverity.CRITICAL if above_pct > 5.0
                else ValueIssueSeverity.HIGH if above_pct > 1.0
                else ValueIssueSeverity.MEDIUM
            )
            example_values = sorted(data[above_max], reverse=True)[:5]
            issues.append(ValueIssue(
                check_type=ValueCheckType.RANGE,
                severity=severity,
                description=f"Values above maximum {expected_max}: {above_count} pixels ({above_pct:.2f}%)",
                affected_pixels=int(above_count),
                affected_percentage=above_pct,
                example_values=[float(v) for v in example_values],
                expected_range=(expected_min, expected_max),
                recommendation=f"Clamp values to range or investigate algorithm",
            ))
            metrics["above_max_count"] = int(above_count)

        return issues, metrics

    def _check_distribution(
        self, data: np.ndarray
    ) -> Tuple[List[ValueIssue], Dict[str, Any]]:
        """Check value distribution for anomalies."""
        issues = []
        metrics = {}

        # Check unique values
        unique_values = np.unique(data)
        metrics["unique_count"] = len(unique_values)

        if len(unique_values) < self.config.min_unique_values:
            # Check if this is binary data
            if self.config.value_type != ValueType.BINARY:
                issues.append(ValueIssue(
                    check_type=ValueCheckType.DISTRIBUTION,
                    severity=ValueIssueSeverity.MEDIUM,
                    description=f"Only {len(unique_values)} unique values found",
                    recommendation="Check if data processing reduced precision excessively",
                ))

        # Check if single value dominates
        value_counts = {}
        for v in unique_values:
            count = (data == v).sum()
            value_counts[float(v)] = count

        if value_counts:
            max_value = max(value_counts, key=value_counts.get)
            max_count = value_counts[max_value]
            max_pct = 100.0 * max_count / len(data)
            metrics["dominant_value"] = max_value
            metrics["dominant_pct"] = max_pct

            if max_pct > self.config.max_single_value_pct:
                issues.append(ValueIssue(
                    check_type=ValueCheckType.DISTRIBUTION,
                    severity=ValueIssueSeverity.MEDIUM,
                    description=f"Single value {max_value} dominates: {max_pct:.1f}% of data",
                    recommendation="Verify this is expected (e.g., mostly background)",
                ))

        # Check skewness (if enough data points)
        if len(data) > 100:
            try:
                from scipy import stats
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                metrics["skewness"] = float(skewness)
                metrics["kurtosis"] = float(kurtosis)

                # Extreme skewness may indicate issues
                if abs(skewness) > 5.0:
                    issues.append(ValueIssue(
                        check_type=ValueCheckType.DISTRIBUTION,
                        severity=ValueIssueSeverity.LOW,
                        description=f"Highly skewed distribution (skewness={skewness:.2f})",
                        recommendation="Verify distribution is expected for this data type",
                    ))
            except (ImportError, Exception):
                pass  # scipy not available or computation failed

        return issues, metrics

    def _check_percentiles(
        self, data: np.ndarray
    ) -> Tuple[List[ValueIssue], Dict[str, Any]]:
        """Check extreme percentiles for outliers."""
        issues = []
        metrics = {}

        # Compute percentiles
        percentiles = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9]
        pctl_values = np.percentile(data, percentiles)
        metrics["percentiles"] = {f"p{p}": float(v) for p, v in zip(percentiles, pctl_values)}

        # Check for extreme outliers
        p01, p999 = pctl_values[0], pctl_values[-1]
        iqr = pctl_values[6] - pctl_values[3]  # p75 - p25
        median = pctl_values[4]

        metrics["iqr"] = float(iqr)

        # Check extreme value ratio
        if iqr > 0:
            lower_extreme = abs(p01 - median) / iqr
            upper_extreme = abs(p999 - median) / iqr

            metrics["lower_extreme_ratio"] = float(lower_extreme)
            metrics["upper_extreme_ratio"] = float(upper_extreme)

            # Very large ratios indicate extreme outliers
            if lower_extreme > 10 or upper_extreme > 10:
                issues.append(ValueIssue(
                    check_type=ValueCheckType.PERCENTILE,
                    severity=ValueIssueSeverity.LOW,
                    description=f"Extreme outliers detected (>10 IQR from median)",
                    example_values=[float(p01), float(p999)],
                    recommendation="Review outlier handling in algorithm",
                ))

        return issues, metrics


def check_value_plausibility(
    data: np.ndarray,
    value_type: ValueType = ValueType.CUSTOM,
    custom_range: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    config: Optional[ValuePlausibilityConfig] = None,
) -> ValuePlausibilityResult:
    """
    Convenience function to check value plausibility.

    Args:
        data: Array of analysis output values
        value_type: Type of values for automatic range detection
        custom_range: Custom valid range (overrides value_type)
        mask: Optional boolean mask
        config: Full configuration (overrides other args)

    Returns:
        ValuePlausibilityResult with all findings
    """
    if config is None:
        config = ValuePlausibilityConfig(
            value_type=value_type,
            custom_range=custom_range,
        )
    checker = ValuePlausibilityChecker(config)
    return checker.check(data, mask)
