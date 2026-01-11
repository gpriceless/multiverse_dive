"""
Sanity Check Module for Quality Control.

Provides comprehensive sanity checks to validate analysis outputs:
- Spatial: Spatial coherence, autocorrelation, boundary artifacts
- Values: Value ranges, physical plausibility, statistical sanity
- Temporal: Temporal consistency, rate of change, event timing
- Artifacts: Stripe detection, hot pixels, compression artifacts

Each checker follows a consistent pattern:
- Config dataclass for check options
- Result dataclass with findings and metrics
- Checker class with check methods
- Convenience function for simple usage

Example:
    from core.quality.sanity import (
        SanitySuite,
        check_spatial_coherence,
        check_value_plausibility,
        check_temporal_consistency,
        detect_artifacts,
    )

    # Run individual checks
    spatial_result = check_spatial_coherence(flood_extent)
    value_result = check_value_plausibility(confidence_map, value_type=ValueType.CONFIDENCE)
    temporal_result = check_temporal_consistency(extent_series, timestamps)
    artifact_result = detect_artifacts(raw_output)

    # Run combined sanity suite
    suite = SanitySuite()
    result = suite.check(flood_extent, timestamps=acquisition_times)
"""

# Spatial coherence checks
from core.quality.sanity.spatial import (
    SpatialCheckType,
    SpatialIssueSeverity,
    SpatialIssue,
    SpatialCoherenceConfig,
    SpatialCoherenceResult,
    SpatialCoherenceChecker,
    check_spatial_coherence,
)

# Value plausibility checks
from core.quality.sanity.values import (
    ValueCheckType,
    ValueIssueSeverity,
    ValueIssue,
    ValueType,
    VALUE_RANGES,
    ValuePlausibilityConfig,
    ValuePlausibilityResult,
    ValuePlausibilityChecker,
    check_value_plausibility,
)

# Temporal consistency checks
from core.quality.sanity.temporal import (
    TemporalCheckType,
    TemporalIssueSeverity,
    TemporalIssue,
    ChangeDirection,
    TemporalConsistencyConfig,
    TemporalConsistencyResult,
    TemporalConsistencyChecker,
    check_temporal_consistency,
    check_raster_temporal_consistency,
)

# Artifact detection
from core.quality.sanity.artifacts import (
    ArtifactType,
    ArtifactSeverity,
    ArtifactLocation,
    DetectedArtifact,
    ArtifactDetectionConfig,
    ArtifactDetectionResult,
    ArtifactDetector,
    detect_artifacts,
)

__all__ = [
    # Spatial
    "SpatialCheckType",
    "SpatialIssueSeverity",
    "SpatialIssue",
    "SpatialCoherenceConfig",
    "SpatialCoherenceResult",
    "SpatialCoherenceChecker",
    "check_spatial_coherence",
    # Values
    "ValueCheckType",
    "ValueIssueSeverity",
    "ValueIssue",
    "ValueType",
    "VALUE_RANGES",
    "ValuePlausibilityConfig",
    "ValuePlausibilityResult",
    "ValuePlausibilityChecker",
    "check_value_plausibility",
    # Temporal
    "TemporalCheckType",
    "TemporalIssueSeverity",
    "TemporalIssue",
    "ChangeDirection",
    "TemporalConsistencyConfig",
    "TemporalConsistencyResult",
    "TemporalConsistencyChecker",
    "check_temporal_consistency",
    "check_raster_temporal_consistency",
    # Artifacts
    "ArtifactType",
    "ArtifactSeverity",
    "ArtifactLocation",
    "DetectedArtifact",
    "ArtifactDetectionConfig",
    "ArtifactDetectionResult",
    "ArtifactDetector",
    "detect_artifacts",
    # Combined
    "SanitySuite",
    "SanitySuiteConfig",
    "SanitySuiteResult",
]


from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SanitySuiteConfig:
    """
    Configuration for the combined sanity suite.

    Attributes:
        run_spatial: Run spatial coherence checks
        run_values: Run value plausibility checks
        run_temporal: Run temporal consistency checks
        run_artifacts: Run artifact detection

        spatial_config: Configuration for spatial checks
        value_config: Configuration for value checks
        temporal_config: Configuration for temporal checks
        artifact_config: Configuration for artifact detection
    """
    run_spatial: bool = True
    run_values: bool = True
    run_temporal: bool = True
    run_artifacts: bool = True

    spatial_config: Optional[SpatialCoherenceConfig] = None
    value_config: Optional[ValuePlausibilityConfig] = None
    temporal_config: Optional[TemporalConsistencyConfig] = None
    artifact_config: Optional[ArtifactDetectionConfig] = None


@dataclass
class SanitySuiteResult:
    """
    Combined result from all sanity checks.

    Attributes:
        passes_sanity: Whether all critical checks passed
        spatial: Spatial coherence result
        values: Value plausibility result
        temporal: Temporal consistency result
        artifacts: Artifact detection result
        overall_score: Combined quality score (0-1)
        summary: Human-readable summary
        duration_seconds: Total time for all checks
    """
    passes_sanity: bool
    spatial: Optional[SpatialCoherenceResult] = None
    values: Optional[ValuePlausibilityResult] = None
    temporal: Optional[TemporalConsistencyResult] = None
    artifacts: Optional[ArtifactDetectionResult] = None
    overall_score: float = 0.0
    summary: str = ""
    duration_seconds: float = 0.0

    @property
    def total_issues(self) -> int:
        """Total number of issues across all checks."""
        count = 0
        if self.spatial:
            count += len(self.spatial.issues)
        if self.values:
            count += len(self.values.issues)
        if self.temporal:
            count += len(self.temporal.issues)
        if self.artifacts:
            count += len(self.artifacts.artifacts)
        return count

    @property
    def critical_issues(self) -> int:
        """Total number of critical issues."""
        count = 0
        if self.spatial:
            count += self.spatial.critical_count
        if self.values:
            count += self.values.critical_count
        if self.temporal:
            count += self.temporal.critical_count
        if self.artifacts:
            count += self.artifacts.critical_count
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passes_sanity": self.passes_sanity,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "spatial": self.spatial.to_dict() if self.spatial else None,
            "values": self.values.to_dict() if self.values else None,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "duration_seconds": self.duration_seconds,
        }


class SanitySuite:
    """
    Combined sanity check suite.

    Runs spatial, value, temporal, and artifact checks on analysis outputs
    and provides an overall quality assessment.

    Example:
        suite = SanitySuite()
        result = suite.check(
            data=flood_extent_array,
            transform=geotransform,
            timestamps=acquisition_times,
            time_series_values=extent_values,
        )
        if not result.passes_sanity:
            print(f"Sanity check failed: {result.summary}")
    """

    def __init__(self, config: Optional[SanitySuiteConfig] = None):
        """
        Initialize the sanity suite.

        Args:
            config: Configuration options
        """
        self.config = config or SanitySuiteConfig()

        # Initialize checkers
        self.spatial_checker = SpatialCoherenceChecker(self.config.spatial_config)
        self.value_checker = ValuePlausibilityChecker(self.config.value_config)
        self.temporal_checker = TemporalConsistencyChecker(self.config.temporal_config)
        self.artifact_detector = ArtifactDetector(self.config.artifact_config)

    def check(
        self,
        data: np.ndarray,
        transform: Optional[Tuple[float, ...]] = None,
        mask: Optional[np.ndarray] = None,
        timestamps: Optional[List[datetime]] = None,
        time_series_values: Optional[Union[List[float], np.ndarray]] = None,
        tile_boundaries: Optional[List[int]] = None,
    ) -> SanitySuiteResult:
        """
        Run all configured sanity checks.

        Args:
            data: 2D array of analysis output
            transform: Affine geotransform for spatial checks
            mask: Optional boolean mask of valid pixels
            timestamps: Optional timestamps for temporal checks
            time_series_values: Optional time series values (if different from data mean)
            tile_boundaries: Optional tile boundary indices for spatial checks

        Returns:
            SanitySuiteResult with all check results
        """
        import time
        start_time = time.time()

        spatial_result = None
        value_result = None
        temporal_result = None
        artifact_result = None

        # Run spatial checks
        if self.config.run_spatial:
            logger.info("Running spatial coherence checks...")
            spatial_result = self.spatial_checker.check(
                data, transform, mask, tile_boundaries
            )

        # Run value checks
        if self.config.run_values:
            logger.info("Running value plausibility checks...")
            value_result = self.value_checker.check(data, mask)

        # Run temporal checks
        if self.config.run_temporal and timestamps is not None:
            logger.info("Running temporal consistency checks...")
            # Use provided time series values or compute from data
            if time_series_values is not None:
                values = time_series_values
            elif data.ndim == 3:
                # 3D data: assume first dimension is time
                values = [np.nanmean(data[t]) for t in range(data.shape[0])]
            else:
                # 2D data: can't do temporal check without time series
                values = None

            if values is not None and len(values) == len(timestamps):
                temporal_result = self.temporal_checker.check_time_series(
                    values, timestamps
                )

        # Run artifact detection
        if self.config.run_artifacts:
            logger.info("Running artifact detection...")
            artifact_result = self.artifact_detector.detect(data, mask)

        # Calculate overall pass/fail
        passes_sanity = True
        if spatial_result and not spatial_result.is_coherent:
            passes_sanity = False
        if value_result and not value_result.is_plausible:
            passes_sanity = False
        if temporal_result and not temporal_result.is_consistent:
            passes_sanity = False
        if artifact_result and artifact_result.critical_count > 0:
            passes_sanity = False

        # Calculate overall score
        scores = []
        if spatial_result:
            # Convert issues to score
            spatial_score = 1.0 - min(1.0, (spatial_result.critical_count * 0.3 + spatial_result.high_count * 0.15))
            scores.append(spatial_score)
        if value_result:
            value_score = 1.0 - min(1.0, (value_result.critical_count * 0.3 + value_result.high_count * 0.15))
            scores.append(value_score)
        if temporal_result:
            temporal_score = 1.0 - min(1.0, (temporal_result.critical_count * 0.3 + temporal_result.high_count * 0.15))
            scores.append(temporal_score)
        if artifact_result:
            artifact_score = 1.0 - min(1.0, (artifact_result.critical_count * 0.3 + artifact_result.high_count * 0.15))
            scores.append(artifact_score)

        overall_score = sum(scores) / len(scores) if scores else 1.0

        # Generate summary
        summary_parts = []
        if spatial_result:
            status = "OK" if spatial_result.is_coherent else f"{len(spatial_result.issues)} issues"
            summary_parts.append(f"Spatial: {status}")
        if value_result:
            status = "OK" if value_result.is_plausible else f"{len(value_result.issues)} issues"
            summary_parts.append(f"Values: {status}")
        if temporal_result:
            status = "OK" if temporal_result.is_consistent else f"{len(temporal_result.issues)} issues"
            summary_parts.append(f"Temporal: {status}")
        if artifact_result:
            status = "OK" if not artifact_result.has_artifacts else f"{len(artifact_result.artifacts)} artifacts"
            summary_parts.append(f"Artifacts: {status}")

        summary = "; ".join(summary_parts)
        duration = time.time() - start_time

        logger.info(f"Sanity suite completed in {duration:.2f}s, passes={passes_sanity}, score={overall_score:.2f}")

        return SanitySuiteResult(
            passes_sanity=passes_sanity,
            spatial=spatial_result,
            values=value_result,
            temporal=temporal_result,
            artifacts=artifact_result,
            overall_score=round(overall_score, 3),
            summary=summary,
            duration_seconds=duration,
        )
