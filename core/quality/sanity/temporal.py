"""
Temporal Consistency Sanity Checks.

Validates that analysis outputs are temporally consistent:
- Time series monotonicity (e.g., flood extent should evolve smoothly)
- Temporal derivative bounds (rate of change limits)
- Event timing plausibility (e.g., fire before pre-event image)
- Acquisition timestamp consistency
- Temporal gap handling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TemporalCheckType(Enum):
    """Types of temporal consistency checks."""
    MONOTONICITY = "monotonicity"           # Expected monotonic change
    RATE_OF_CHANGE = "rate_of_change"       # Change rate within bounds
    TIMESTAMP_ORDER = "timestamp_order"     # Timestamps in correct order
    TIMESTAMP_GAPS = "timestamp_gaps"       # Excessive gaps in time series
    EVENT_TIMING = "event_timing"           # Pre/post event timing
    TEMPORAL_CORRELATION = "temporal_correlation"  # Temporal autocorrelation
    SUDDEN_CHANGE = "sudden_change"         # Unexpected sudden changes
    STABILITY = "stability"                 # Values should be stable (no event)


class TemporalIssueSeverity(Enum):
    """Severity levels for temporal issues."""
    CRITICAL = "critical"   # Temporal logic violated (e.g., effect before cause)
    HIGH = "high"           # Likely processing error
    MEDIUM = "medium"       # Suspicious pattern
    LOW = "low"             # Minor inconsistency
    INFO = "info"           # Informational


@dataclass
class TemporalIssue:
    """
    A temporal consistency issue found during validation.

    Attributes:
        check_type: Type of check that found the issue
        severity: Issue severity level
        description: Human-readable description
        timestamps: Affected timestamps
        metric_value: The problematic metric value
        threshold: The threshold that was exceeded
        recommendation: Suggested action
    """
    check_type: TemporalCheckType
    severity: TemporalIssueSeverity
    description: str
    timestamps: Optional[List[datetime]] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    location: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "timestamps": [t.isoformat() if t else None for t in (self.timestamps or [])],
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "location": self.location,
            "recommendation": self.recommendation,
        }


class ChangeDirection(Enum):
    """Expected direction of change over time."""
    INCREASING = "increasing"     # Values should increase
    DECREASING = "decreasing"     # Values should decrease
    STABLE = "stable"             # Values should remain stable
    ANY = "any"                   # No constraint


@dataclass
class TemporalConsistencyConfig:
    """
    Configuration for temporal consistency checks.

    Attributes:
        check_monotonicity: Check for expected monotonic behavior
        check_rate_of_change: Check rate of change limits
        check_timestamps: Check timestamp ordering and gaps
        check_sudden_changes: Check for unexpected sudden changes

        expected_direction: Expected change direction (if known)
        max_rate_per_hour: Maximum allowed change rate per hour
        max_gap_hours: Maximum allowed gap in time series
        min_temporal_correlation: Minimum expected temporal autocorrelation

        pre_event_time: Pre-event acquisition time
        post_event_time: Post-event acquisition time
        event_time: Known event occurrence time
    """
    check_monotonicity: bool = True
    check_rate_of_change: bool = True
    check_timestamps: bool = True
    check_sudden_changes: bool = True

    expected_direction: ChangeDirection = ChangeDirection.ANY
    max_rate_per_hour: Optional[float] = None  # None = auto-detect
    max_gap_hours: float = 48.0  # 2 days
    min_temporal_correlation: float = 0.3

    # Event timing
    pre_event_time: Optional[datetime] = None
    post_event_time: Optional[datetime] = None
    event_time: Optional[datetime] = None

    # Sudden change detection
    sudden_change_zscore: float = 3.0  # Z-score threshold for sudden changes
    min_stable_periods: int = 3  # Minimum periods for stability check


@dataclass
class TemporalConsistencyResult:
    """
    Result of temporal consistency validation.

    Attributes:
        is_consistent: Whether all temporal checks passed
        issues: List of issues found
        metrics: Computed temporal metrics
        duration_seconds: Time taken for validation
    """
    is_consistent: bool
    issues: List[TemporalIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == TemporalIssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high-severity issues."""
        return sum(1 for i in self.issues if i.severity == TemporalIssueSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_consistent": self.is_consistent,
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class TemporalConsistencyChecker:
    """
    Validates temporal consistency of analysis outputs.

    Checks that outputs are temporally consistent:
    - Time series follow expected patterns
    - Changes occur at plausible rates
    - Event timing is logical

    Example:
        checker = TemporalConsistencyChecker(config)
        result = checker.check_time_series(
            values=flood_extents,
            timestamps=acquisition_times
        )
    """

    def __init__(self, config: Optional[TemporalConsistencyConfig] = None):
        """
        Initialize the temporal consistency checker.

        Args:
            config: Configuration options
        """
        self.config = config or TemporalConsistencyConfig()

    def check_time_series(
        self,
        values: Union[List[float], np.ndarray],
        timestamps: List[datetime],
    ) -> TemporalConsistencyResult:
        """
        Check temporal consistency of a time series.

        Args:
            values: Time series values (e.g., total flood extent per timestep)
            timestamps: Corresponding timestamps

        Returns:
            TemporalConsistencyResult with all issues found
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}

        values = np.asarray(values)

        # Validate input
        if len(values) != len(timestamps):
            issues.append(TemporalIssue(
                check_type=TemporalCheckType.TIMESTAMP_ORDER,
                severity=TemporalIssueSeverity.CRITICAL,
                description=f"Mismatch: {len(values)} values but {len(timestamps)} timestamps",
            ))
            return TemporalConsistencyResult(
                is_consistent=False,
                issues=issues,
                duration_seconds=time.time() - start_time,
            )

        if len(values) < 2:
            metrics["note"] = "Insufficient data points for temporal analysis"
            return TemporalConsistencyResult(
                is_consistent=True,
                issues=issues,
                metrics=metrics,
                duration_seconds=time.time() - start_time,
            )

        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        values = values[sorted_indices]
        timestamps = [timestamps[i] for i in sorted_indices]

        metrics["num_timesteps"] = len(values)
        metrics["time_range"] = {
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
        }

        # Check timestamp ordering and gaps
        if self.config.check_timestamps:
            ts_issues, ts_metrics = self._check_timestamps(timestamps)
            issues.extend(ts_issues)
            metrics["timestamps"] = ts_metrics

        # Check monotonicity
        if self.config.check_monotonicity:
            mono_issues, mono_metrics = self._check_monotonicity(values, timestamps)
            issues.extend(mono_issues)
            metrics["monotonicity"] = mono_metrics

        # Check rate of change
        if self.config.check_rate_of_change:
            rate_issues, rate_metrics = self._check_rate_of_change(values, timestamps)
            issues.extend(rate_issues)
            metrics["rate_of_change"] = rate_metrics

        # Check for sudden changes
        if self.config.check_sudden_changes:
            sudden_issues, sudden_metrics = self._check_sudden_changes(values, timestamps)
            issues.extend(sudden_issues)
            metrics["sudden_changes"] = sudden_metrics

        # Check event timing
        if self.config.event_time:
            event_issues, event_metrics = self._check_event_timing(timestamps)
            issues.extend(event_issues)
            metrics["event_timing"] = event_metrics

        # Determine overall consistency
        is_consistent = not any(
            i.severity in (TemporalIssueSeverity.CRITICAL, TemporalIssueSeverity.HIGH)
            for i in issues
        )

        duration = time.time() - start_time
        logger.info(f"Temporal consistency check completed in {duration:.2f}s, consistent={is_consistent}")

        return TemporalConsistencyResult(
            is_consistent=is_consistent,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def check_raster_series(
        self,
        rasters: List[np.ndarray],
        timestamps: List[datetime],
        sample_size: int = 1000,
    ) -> TemporalConsistencyResult:
        """
        Check temporal consistency of a raster time series.

        Samples pixels and checks temporal consistency at each location.

        Args:
            rasters: List of 2D arrays for each timestep
            timestamps: Corresponding timestamps
            sample_size: Number of pixels to sample

        Returns:
            TemporalConsistencyResult with all issues found
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}

        if not rasters:
            return TemporalConsistencyResult(
                is_consistent=True,
                metrics={"note": "No rasters provided"},
                duration_seconds=time.time() - start_time,
            )

        # Validate shapes
        shape = rasters[0].shape
        for i, r in enumerate(rasters):
            if r.shape != shape:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.TIMESTAMP_ORDER,
                    severity=TemporalIssueSeverity.CRITICAL,
                    description=f"Raster {i} has shape {r.shape}, expected {shape}",
                ))
                return TemporalConsistencyResult(
                    is_consistent=False,
                    issues=issues,
                    duration_seconds=time.time() - start_time,
                )

        # Stack rasters
        stack = np.stack(rasters, axis=0)  # Shape: (T, H, W)
        T, H, W = stack.shape

        # Find valid pixels (not NaN in any timestep)
        valid_all = np.all(np.isfinite(stack), axis=0)
        valid_indices = np.where(valid_all)

        if len(valid_indices[0]) == 0:
            issues.append(TemporalIssue(
                check_type=TemporalCheckType.STABILITY,
                severity=TemporalIssueSeverity.HIGH,
                description="No pixels are valid across all timesteps",
            ))
            return TemporalConsistencyResult(
                is_consistent=False,
                issues=issues,
                duration_seconds=time.time() - start_time,
            )

        # Sample pixels
        n_valid = len(valid_indices[0])
        sample_n = min(sample_size, n_valid)
        sample_idx = np.random.choice(n_valid, sample_n, replace=False)
        sample_rows = valid_indices[0][sample_idx]
        sample_cols = valid_indices[1][sample_idx]

        metrics["sampled_pixels"] = sample_n
        metrics["valid_pixels"] = n_valid

        # Analyze mean time series
        mean_ts = np.nanmean(stack, axis=(1, 2))
        metrics["mean_time_series"] = mean_ts.tolist()

        # Check mean time series
        mean_result = self.check_time_series(mean_ts, timestamps)
        issues.extend(mean_result.issues)
        metrics["mean_series_metrics"] = mean_result.metrics

        # Check temporal correlation across sampled pixels
        correlations = []
        for i in range(sample_n):
            r, c = sample_rows[i], sample_cols[i]
            pixel_ts = stack[:, r, c]
            if len(pixel_ts) > 2:
                # Lag-1 autocorrelation
                autocorr = self._lag1_autocorrelation(pixel_ts)
                if np.isfinite(autocorr):
                    correlations.append(autocorr)

        if correlations:
            mean_corr = np.mean(correlations)
            metrics["mean_temporal_autocorrelation"] = mean_corr

            if mean_corr < self.config.min_temporal_correlation:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.TEMPORAL_CORRELATION,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Low temporal autocorrelation: {mean_corr:.3f}",
                    metric_value=mean_corr,
                    threshold=self.config.min_temporal_correlation,
                    recommendation="Check for temporal noise or registration errors",
                ))

        # Determine overall consistency
        is_consistent = not any(
            i.severity in (TemporalIssueSeverity.CRITICAL, TemporalIssueSeverity.HIGH)
            for i in issues
        )

        duration = time.time() - start_time

        return TemporalConsistencyResult(
            is_consistent=is_consistent,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _check_timestamps(
        self, timestamps: List[datetime]
    ) -> Tuple[List[TemporalIssue], Dict[str, Any]]:
        """Check timestamp ordering and gaps."""
        issues = []
        metrics = {}

        # Calculate gaps
        gaps_hours = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600.0
            gaps_hours.append(gap)

        if gaps_hours:
            metrics["mean_gap_hours"] = np.mean(gaps_hours)
            metrics["max_gap_hours"] = max(gaps_hours)
            metrics["min_gap_hours"] = min(gaps_hours)

            # Check for excessive gaps
            max_gap = max(gaps_hours)
            if max_gap > self.config.max_gap_hours:
                gap_idx = gaps_hours.index(max_gap)
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.TIMESTAMP_GAPS,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Large temporal gap: {max_gap:.1f} hours between observations",
                    timestamps=[timestamps[gap_idx], timestamps[gap_idx + 1]],
                    metric_value=max_gap,
                    threshold=self.config.max_gap_hours,
                    recommendation="Consider interpolating or flagging gap period",
                ))

            # Check for negative gaps (time travel)
            negative_gaps = [(i, g) for i, g in enumerate(gaps_hours) if g < 0]
            if negative_gaps:
                for idx, gap in negative_gaps:
                    issues.append(TemporalIssue(
                        check_type=TemporalCheckType.TIMESTAMP_ORDER,
                        severity=TemporalIssueSeverity.CRITICAL,
                        description=f"Timestamps out of order at index {idx}: {gap:.1f} hour negative gap",
                        timestamps=[timestamps[idx], timestamps[idx + 1]],
                        recommendation="Sort timestamps or check data ingestion",
                    ))

        return issues, metrics

    def _check_monotonicity(
        self, values: np.ndarray, timestamps: List[datetime]
    ) -> Tuple[List[TemporalIssue], Dict[str, Any]]:
        """Check for expected monotonic behavior."""
        issues = []
        metrics = {}

        # Calculate differences
        diffs = np.diff(values)

        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        stable = np.sum(diffs == 0)

        metrics["increasing_transitions"] = int(increasing)
        metrics["decreasing_transitions"] = int(decreasing)
        metrics["stable_transitions"] = int(stable)

        total_transitions = len(diffs)
        if total_transitions == 0:
            return issues, metrics

        # Check against expected direction
        if self.config.expected_direction == ChangeDirection.INCREASING:
            violation_rate = decreasing / total_transitions
            metrics["violation_rate"] = violation_rate
            if violation_rate > 0.2:  # More than 20% violations
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.MONOTONICITY,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Expected increasing values, but {100*violation_rate:.1f}% of transitions decrease",
                    metric_value=violation_rate,
                    recommendation="Review if monotonic increase is truly expected",
                ))

        elif self.config.expected_direction == ChangeDirection.DECREASING:
            violation_rate = increasing / total_transitions
            metrics["violation_rate"] = violation_rate
            if violation_rate > 0.2:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.MONOTONICITY,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Expected decreasing values, but {100*violation_rate:.1f}% of transitions increase",
                    metric_value=violation_rate,
                    recommendation="Review if monotonic decrease is truly expected",
                ))

        elif self.config.expected_direction == ChangeDirection.STABLE:
            change_rate = (increasing + decreasing) / total_transitions
            metrics["change_rate"] = change_rate
            if change_rate > 0.1:  # More than 10% unstable
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.STABILITY,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Expected stable values, but {100*change_rate:.1f}% of transitions change",
                    metric_value=change_rate,
                    recommendation="Check for noise or unexpected events",
                ))

        return issues, metrics

    def _check_rate_of_change(
        self, values: np.ndarray, timestamps: List[datetime]
    ) -> Tuple[List[TemporalIssue], Dict[str, Any]]:
        """Check that rate of change is within bounds."""
        issues = []
        metrics = {}

        # Calculate rates per hour
        rates_per_hour = []
        for i in range(1, len(values)):
            dt_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600.0
            if dt_hours > 0:
                rate = abs(values[i] - values[i-1]) / dt_hours
                rates_per_hour.append(rate)

        if not rates_per_hour:
            return issues, metrics

        max_rate = max(rates_per_hour)
        mean_rate = np.mean(rates_per_hour)
        metrics["max_rate_per_hour"] = max_rate
        metrics["mean_rate_per_hour"] = mean_rate

        # Determine threshold
        threshold = self.config.max_rate_per_hour
        if threshold is None:
            # Auto-detect: flag rates > 5 standard deviations
            std_rate = np.std(rates_per_hour)
            threshold = mean_rate + 5 * std_rate if std_rate > 0 else mean_rate * 10
            metrics["auto_threshold"] = threshold

        metrics["threshold"] = threshold

        # Check for excessive rates
        excessive = [(i, r) for i, r in enumerate(rates_per_hour) if r > threshold]
        if excessive:
            for idx, rate in excessive[:3]:  # Report first 3
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.RATE_OF_CHANGE,
                    severity=TemporalIssueSeverity.MEDIUM,
                    description=f"Excessive change rate: {rate:.2f}/hour between timesteps {idx} and {idx+1}",
                    timestamps=[timestamps[idx], timestamps[idx + 1]],
                    metric_value=rate,
                    threshold=threshold,
                    recommendation="Verify rapid change is real, not artifact",
                ))

        return issues, metrics

    def _check_sudden_changes(
        self, values: np.ndarray, timestamps: List[datetime]
    ) -> Tuple[List[TemporalIssue], Dict[str, Any]]:
        """Check for unexpected sudden changes."""
        issues = []
        metrics = {}

        if len(values) < self.config.min_stable_periods + 1:
            return issues, metrics

        # Calculate differences
        diffs = np.diff(values)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        if std_diff < 1e-10:
            metrics["note"] = "Constant differences, no sudden changes"
            return issues, metrics

        # Find sudden changes (z-score based)
        z_scores = (diffs - mean_diff) / std_diff
        sudden_mask = np.abs(z_scores) > self.config.sudden_change_zscore

        sudden_indices = np.where(sudden_mask)[0]
        metrics["sudden_change_count"] = len(sudden_indices)

        for idx in sudden_indices[:5]:  # Report first 5
            z = z_scores[idx]
            diff = diffs[idx]
            issues.append(TemporalIssue(
                check_type=TemporalCheckType.SUDDEN_CHANGE,
                severity=TemporalIssueSeverity.MEDIUM if abs(z) < 5 else TemporalIssueSeverity.HIGH,
                description=f"Sudden change at timestep {idx}: diff={diff:.3f}, z-score={z:.2f}",
                timestamps=[timestamps[idx], timestamps[idx + 1]],
                metric_value=z,
                threshold=self.config.sudden_change_zscore,
                recommendation="Verify if sudden change corresponds to real event",
            ))

        return issues, metrics

    def _check_event_timing(
        self, timestamps: List[datetime]
    ) -> Tuple[List[TemporalIssue], Dict[str, Any]]:
        """Check event timing logic."""
        issues = []
        metrics = {}

        event_time = self.config.event_time
        pre_time = self.config.pre_event_time
        post_time = self.config.post_event_time

        # Check pre-event before event
        if pre_time and event_time:
            if pre_time > event_time:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.EVENT_TIMING,
                    severity=TemporalIssueSeverity.CRITICAL,
                    description="Pre-event time is after event time",
                    timestamps=[pre_time, event_time],
                    recommendation="Check event and acquisition timestamps",
                ))
            else:
                gap_hours = (event_time - pre_time).total_seconds() / 3600.0
                metrics["pre_event_gap_hours"] = gap_hours

        # Check post-event after event
        if post_time and event_time:
            if post_time < event_time:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.EVENT_TIMING,
                    severity=TemporalIssueSeverity.CRITICAL,
                    description="Post-event time is before event time",
                    timestamps=[event_time, post_time],
                    recommendation="Check event and acquisition timestamps",
                ))
            else:
                gap_hours = (post_time - event_time).total_seconds() / 3600.0
                metrics["post_event_gap_hours"] = gap_hours

        # Check data coverage spans event
        if event_time and timestamps:
            first_obs = min(timestamps)
            last_obs = max(timestamps)

            if first_obs > event_time:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.EVENT_TIMING,
                    severity=TemporalIssueSeverity.HIGH,
                    description="All observations are after the event (no baseline)",
                    timestamps=[event_time, first_obs],
                    recommendation="Acquire pre-event imagery for change detection",
                ))

            if last_obs < event_time:
                issues.append(TemporalIssue(
                    check_type=TemporalCheckType.EVENT_TIMING,
                    severity=TemporalIssueSeverity.HIGH,
                    description="All observations are before the event (no post-event data)",
                    timestamps=[last_obs, event_time],
                    recommendation="Acquire post-event imagery to assess impact",
                ))

        return issues, metrics

    def _lag1_autocorrelation(self, values: np.ndarray) -> float:
        """Calculate lag-1 autocorrelation."""
        if len(values) < 3:
            return np.nan
        mean = np.mean(values)
        var = np.var(values)
        if var < 1e-10:
            return np.nan
        cov = np.mean((values[:-1] - mean) * (values[1:] - mean))
        return cov / var


def check_temporal_consistency(
    values: Union[List[float], np.ndarray],
    timestamps: List[datetime],
    config: Optional[TemporalConsistencyConfig] = None,
) -> TemporalConsistencyResult:
    """
    Convenience function to check temporal consistency.

    Args:
        values: Time series values
        timestamps: Corresponding timestamps
        config: Optional configuration

    Returns:
        TemporalConsistencyResult with all findings
    """
    checker = TemporalConsistencyChecker(config)
    return checker.check_time_series(values, timestamps)


def check_raster_temporal_consistency(
    rasters: List[np.ndarray],
    timestamps: List[datetime],
    config: Optional[TemporalConsistencyConfig] = None,
    sample_size: int = 1000,
) -> TemporalConsistencyResult:
    """
    Convenience function to check raster temporal consistency.

    Args:
        rasters: List of 2D arrays for each timestep
        timestamps: Corresponding timestamps
        config: Optional configuration
        sample_size: Number of pixels to sample

    Returns:
        TemporalConsistencyResult with all findings
    """
    checker = TemporalConsistencyChecker(config)
    return checker.check_raster_series(rasters, timestamps, sample_size)
