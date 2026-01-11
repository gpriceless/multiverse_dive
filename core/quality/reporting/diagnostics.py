"""
Diagnostic Outputs for Quality Control.

Provides detailed diagnostic information for debugging and analysis:
- Quality metric visualizations (histograms, spatial maps, time series)
- Issue detail dumps with full context
- Performance profiling of QC operations
- Comparison diagnostics between runs
- Export formats for external analysis tools

Key Features:
- Machine-readable diagnostic exports (JSON, CSV, Parquet)
- Human-readable diagnostic summaries
- Spatial diagnostic outputs (GeoTIFF, GeoJSON)
- Temporal diagnostic plots data
- Integration with logging and monitoring systems

Example:
    from core.quality.reporting import (
        DiagnosticGenerator,
        DiagnosticLevel,
        DiagnosticType,
        generate_diagnostics,
    )

    # Generate full diagnostics
    generator = DiagnosticGenerator()
    diagnostics = generator.generate(
        sanity_result=sanity_result,
        validation_result=validation_result,
        uncertainty_result=uncertainty_result,
        level=DiagnosticLevel.DETAILED,
    )

    # Export to various formats
    diagnostics.export_json("diagnostics.json")
    diagnostics.export_csv("diagnostics/")
    diagnostics.export_spatial("diagnostics/spatial/")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a value, returning None if NaN, Inf, or None."""
    if value is None:
        return None
    try:
        if np.isnan(value) or np.isinf(value):
            return None
        return round(value, digits)
    except (TypeError, ValueError):
        # Handle non-numeric types gracefully
        return None


class DiagnosticLevel(Enum):
    """Level of diagnostic detail."""
    MINIMAL = "minimal"       # Only errors and critical issues
    STANDARD = "standard"     # Normal operational diagnostics
    DETAILED = "detailed"     # Full detail for debugging
    TRACE = "trace"           # Maximum verbosity with timing


class DiagnosticType(Enum):
    """Types of diagnostic outputs."""
    SUMMARY = "summary"           # High-level summary
    METRICS = "metrics"           # Quality metrics
    ISSUES = "issues"             # Detected issues
    SPATIAL = "spatial"           # Spatial diagnostics
    TEMPORAL = "temporal"         # Temporal diagnostics
    PERFORMANCE = "performance"   # Performance profiling
    COMPARISON = "comparison"     # Run comparison


class MetricCategory(Enum):
    """Categories of diagnostic metrics."""
    SANITY = "sanity"
    VALIDATION = "validation"
    UNCERTAINTY = "uncertainty"
    GATING = "gating"
    PERFORMANCE = "performance"


@dataclass
class DiagnosticMetric:
    """
    A single diagnostic metric.

    Attributes:
        name: Metric name
        category: Metric category
        value: Metric value (can be scalar, array, or dict)
        unit: Unit of measurement
        description: What this metric means
        threshold: Expected threshold if applicable
        status: Pass/warning/fail based on threshold
        timestamp: When metric was computed
    """
    name: str
    category: MetricCategory
    value: Union[float, int, str, List, Dict, np.ndarray]
    unit: str = ""
    description: str = ""
    threshold: Optional[float] = None
    status: str = "info"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        val = self.value
        if isinstance(val, np.ndarray):
            val = val.tolist()
        return {
            "name": self.name,
            "category": self.category.value,
            "value": val,
            "unit": self.unit,
            "description": self.description,
            "threshold": self.threshold,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IssueDetail:
    """
    Detailed information about a detected issue.

    Attributes:
        issue_id: Unique identifier
        issue_type: Type of issue
        severity: Severity level
        category: Issue category
        description: Human-readable description
        location: Where the issue was found
        spatial_extent: GeoJSON geometry if spatial
        pixel_coordinates: Pixel coordinates if applicable
        affected_values: Sample of affected values
        context: Additional context information
        suggested_action: Recommended resolution
        related_issues: IDs of related issues
    """
    issue_id: str
    issue_type: str
    severity: str
    category: str
    description: str
    location: str = ""
    spatial_extent: Optional[Dict[str, Any]] = None
    pixel_coordinates: Optional[Tuple[int, int, int, int]] = None  # (min_row, max_row, min_col, max_col)
    affected_values: Optional[List[float]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    related_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
        }
        if self.location:
            result["location"] = self.location
        if self.spatial_extent:
            result["spatial_extent"] = self.spatial_extent
        if self.pixel_coordinates:
            result["pixel_coordinates"] = {
                "min_row": self.pixel_coordinates[0],
                "max_row": self.pixel_coordinates[1],
                "min_col": self.pixel_coordinates[2],
                "max_col": self.pixel_coordinates[3],
            }
        if self.affected_values:
            result["affected_values"] = self.affected_values[:100]  # Limit size
        if self.context:
            result["context"] = self.context
        if self.suggested_action:
            result["suggested_action"] = self.suggested_action
        if self.related_issues:
            result["related_issues"] = self.related_issues
        return result


@dataclass
class SpatialDiagnostic:
    """
    Spatial diagnostic data.

    Attributes:
        name: Diagnostic name
        description: What this represents
        data: 2D array of values
        crs: Coordinate reference system
        transform: Affine transform
        nodata: NoData value
        statistics: Summary statistics
    """
    name: str
    description: str
    data: np.ndarray
    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    nodata: float = np.nan
    statistics: Dict[str, float] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistics for the data."""
        try:
            # Handle potential None values by converting to float array first
            data = np.asarray(self.data, dtype=np.float64)
            valid = data[~np.isnan(data) & ~np.isinf(data)]
        except (TypeError, ValueError):
            return {"count": 0}
        if len(valid) == 0:
            return {"count": 0}
        return {
            "count": len(valid),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "median": float(np.median(valid)),
            "p05": float(np.percentile(valid, 5)),
            "p95": float(np.percentile(valid, 95)),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without full data array)."""
        return {
            "name": self.name,
            "description": self.description,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "nodata": self.nodata if not np.isnan(self.nodata) else None,
            "statistics": self.statistics or self.compute_statistics(),
        }


@dataclass
class TemporalDiagnostic:
    """
    Temporal diagnostic data.

    Attributes:
        name: Diagnostic name
        description: What this represents
        timestamps: List of timestamps
        values: Values at each timestamp
        unit: Unit of measurement
        statistics: Summary statistics
    """
    name: str
    description: str
    timestamps: List[datetime]
    values: List[float]
    unit: str = ""
    statistics: Dict[str, float] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistics for the time series."""
        valid = [v for v in self.values if v is not None and not np.isnan(v) and not np.isinf(v)]
        if len(valid) == 0:
            return {"count": 0}

        stats = {
            "count": len(valid),
            "min": float(min(valid)),
            "max": float(max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
        }

        # Compute rate of change
        if len(valid) > 1:
            diffs = [valid[i+1] - valid[i] for i in range(len(valid)-1)]
            stats["mean_change"] = float(np.mean(diffs))
            stats["max_change"] = float(max(abs(d) for d in diffs))

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "values": self.values,
            "unit": self.unit,
            "statistics": self.statistics or self.compute_statistics(),
        }


@dataclass
class PerformanceMetric:
    """
    Performance profiling metric.

    Attributes:
        operation: Operation name
        duration_seconds: Time taken
        memory_mb: Memory used in MB
        input_size: Input data size
        output_size: Output data size
        details: Additional performance details
    """
    operation: str
    duration_seconds: float
    memory_mb: float = 0.0
    input_size: int = 0
    output_size: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "duration_seconds": _safe_round(self.duration_seconds, 4) or 0.0,
            "memory_mb": _safe_round(self.memory_mb, 2) or 0.0,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "details": self.details,
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing two diagnostic runs.

    Attributes:
        baseline_id: Identifier of baseline run
        compare_id: Identifier of comparison run
        metric_changes: Changes in metrics
        new_issues: Issues in compare not in baseline
        resolved_issues: Issues in baseline not in compare
        status_changed: Whether overall status changed
        summary: Comparison summary
    """
    baseline_id: str
    compare_id: str
    metric_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    new_issues: List[str] = field(default_factory=list)
    resolved_issues: List[str] = field(default_factory=list)
    status_changed: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "compare_id": self.compare_id,
            "metric_changes": self.metric_changes,
            "new_issues_count": len(self.new_issues),
            "resolved_issues_count": len(self.resolved_issues),
            "new_issues": self.new_issues,
            "resolved_issues": self.resolved_issues,
            "status_changed": self.status_changed,
            "summary": self.summary,
        }


@dataclass
class DiagnosticConfig:
    """
    Configuration for diagnostic generation.

    Attributes:
        level: Diagnostic detail level
        include_spatial: Include spatial diagnostics
        include_temporal: Include temporal diagnostics
        include_performance: Include performance metrics
        max_issues: Maximum issues to include
        max_values_per_issue: Maximum sample values per issue
        export_formats: Formats to export
    """
    level: DiagnosticLevel = DiagnosticLevel.STANDARD
    include_spatial: bool = True
    include_temporal: bool = True
    include_performance: bool = True
    max_issues: int = 100
    max_values_per_issue: int = 50
    export_formats: List[str] = field(default_factory=lambda: ["json"])


@dataclass
class Diagnostics:
    """
    Complete diagnostic output.

    Attributes:
        run_id: Unique identifier for this run
        timestamp: When diagnostics were generated
        level: Diagnostic level used
        metrics: List of diagnostic metrics
        issues: List of detailed issues
        spatial: Spatial diagnostics
        temporal: Temporal diagnostics
        performance: Performance metrics
        comparison: Comparison with previous run
        raw_data: Raw diagnostic data for export
    """
    run_id: str
    timestamp: datetime
    level: DiagnosticLevel
    metrics: List[DiagnosticMetric] = field(default_factory=list)
    issues: List[IssueDetail] = field(default_factory=list)
    spatial: List[SpatialDiagnostic] = field(default_factory=list)
    temporal: List[TemporalDiagnostic] = field(default_factory=list)
    performance: List[PerformanceMetric] = field(default_factory=list)
    comparison: Optional[ComparisonResult] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "issues": [i.to_dict() for i in self.issues],
            "spatial": [s.to_dict() for s in self.spatial],
            "temporal": [t.to_dict() for t in self.temporal],
            "performance": [p.to_dict() for p in self.performance],
        }
        if self.comparison:
            result["comparison"] = self.comparison.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def export_json(self, path: Union[str, Path]) -> None:
        """Export diagnostics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info(f"Exported diagnostics to {path}")

    def export_csv(self, directory: Union[str, Path]) -> None:
        """Export diagnostics to CSV files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Export metrics
        if self.metrics:
            metrics_path = directory / "metrics.csv"
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write("name,category,value,unit,threshold,status,timestamp\n")
                for m in self.metrics:
                    val = m.value if not isinstance(m.value, (list, dict, np.ndarray)) else str(m.value)
                    f.write(f"{m.name},{m.category.value},{val},{m.unit},{m.threshold},{m.status},{m.timestamp.isoformat()}\n")
            logger.info(f"Exported metrics to {metrics_path}")

        # Export issues
        if self.issues:
            issues_path = directory / "issues.csv"
            with open(issues_path, "w", encoding="utf-8") as f:
                f.write("issue_id,issue_type,severity,category,description,location,suggested_action\n")
                for i in self.issues:
                    desc = i.description.replace('"', '""')
                    f.write(f'{i.issue_id},{i.issue_type},{i.severity},{i.category},"{desc}",{i.location},{i.suggested_action}\n')
            logger.info(f"Exported issues to {issues_path}")

        # Export performance
        if self.performance:
            perf_path = directory / "performance.csv"
            with open(perf_path, "w", encoding="utf-8") as f:
                f.write("operation,duration_seconds,memory_mb,input_size,output_size\n")
                for p in self.performance:
                    f.write(f"{p.operation},{p.duration_seconds},{p.memory_mb},{p.input_size},{p.output_size}\n")
            logger.info(f"Exported performance to {perf_path}")

    def export_spatial(self, directory: Union[str, Path]) -> None:
        """Export spatial diagnostics to files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for spatial in self.spatial:
            # Export as numpy array
            np_path = directory / f"{spatial.name}.npy"
            np.save(np_path, spatial.data)

            # Export metadata
            meta_path = directory / f"{spatial.name}.json"
            meta_path.write_text(json.dumps(spatial.to_dict(), indent=2), encoding="utf-8")

            logger.info(f"Exported spatial diagnostic {spatial.name} to {directory}")

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Diagnostic Summary (Run: {self.run_id})",
            "=" * 50,
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Level: {self.level.value}",
            "",
            f"Metrics: {len(self.metrics)}",
            f"Issues: {len(self.issues)}",
            f"  - Critical: {len([i for i in self.issues if i.severity == 'critical'])}",
            f"  - High: {len([i for i in self.issues if i.severity == 'high'])}",
            f"  - Medium: {len([i for i in self.issues if i.severity == 'medium'])}",
            f"  - Low: {len([i for i in self.issues if i.severity == 'low'])}",
            f"Spatial Diagnostics: {len(self.spatial)}",
            f"Temporal Diagnostics: {len(self.temporal)}",
            "",
        ]

        # Add key metrics
        if self.metrics:
            lines.append("Key Metrics:")
            for m in self.metrics[:10]:
                val = f"{m.value:.4f}" if isinstance(m.value, float) else str(m.value)
                lines.append(f"  {m.name}: {val} {m.unit} [{m.status}]")
            if len(self.metrics) > 10:
                lines.append(f"  ... and {len(self.metrics) - 10} more")
            lines.append("")

        # Add performance summary
        if self.performance:
            total_time = sum(p.duration_seconds for p in self.performance)
            total_memory = max((p.memory_mb for p in self.performance), default=0)
            lines.append("Performance:")
            lines.append(f"  Total Duration: {total_time:.2f}s")
            lines.append(f"  Peak Memory: {total_memory:.1f} MB")
            lines.append("")

        # Add comparison summary
        if self.comparison:
            lines.append("Comparison with Baseline:")
            lines.append(f"  Baseline: {self.comparison.baseline_id}")
            lines.append(f"  New Issues: {len(self.comparison.new_issues)}")
            lines.append(f"  Resolved Issues: {len(self.comparison.resolved_issues)}")
            lines.append(f"  Status Changed: {'Yes' if self.comparison.status_changed else 'No'}")

        return "\n".join(lines)


class DiagnosticGenerator:
    """
    Generator for diagnostic outputs.

    Extracts detailed diagnostic information from quality control results
    for debugging, analysis, and monitoring.

    Example:
        generator = DiagnosticGenerator()
        diagnostics = generator.generate(
            sanity_result=sanity_result,
            validation_result=validation_result,
            uncertainty_result=uncertainty_result,
        )
        print(diagnostics.get_summary())
        diagnostics.export_json("diagnostics.json")
    """

    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize diagnostic generator.

        Args:
            config: Diagnostic configuration
        """
        self.config = config or DiagnosticConfig()
        self._issue_counter = 0

    def generate(
        self,
        run_id: Optional[str] = None,
        sanity_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
        uncertainty_result: Optional[Any] = None,
        gating_decision: Optional[Any] = None,
        performance_data: Optional[Dict[str, float]] = None,
        baseline_diagnostics: Optional[Diagnostics] = None,
    ) -> Diagnostics:
        """
        Generate diagnostics from quality control results.

        Args:
            run_id: Unique identifier for this run
            sanity_result: Result from SanitySuite
            validation_result: Result from cross-validation
            uncertainty_result: Result from uncertainty quantification
            gating_decision: Result from QualityGate
            performance_data: Performance timing data
            baseline_diagnostics: Previous run for comparison

        Returns:
            Complete Diagnostics object
        """
        import time
        start_time = time.time()

        if run_id is None:
            run_id = f"diag_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating diagnostics for run {run_id}")

        metrics: List[DiagnosticMetric] = []
        issues: List[IssueDetail] = []
        spatial: List[SpatialDiagnostic] = []
        temporal: List[TemporalDiagnostic] = []
        performance: List[PerformanceMetric] = []

        # Extract sanity diagnostics
        if sanity_result is not None:
            sanity_metrics, sanity_issues, sanity_spatial = self._extract_sanity_diagnostics(sanity_result)
            metrics.extend(sanity_metrics)
            issues.extend(sanity_issues[:self.config.max_issues])
            if self.config.include_spatial:
                spatial.extend(sanity_spatial)

        # Extract validation diagnostics
        if validation_result is not None:
            val_metrics, val_issues, val_spatial = self._extract_validation_diagnostics(validation_result)
            metrics.extend(val_metrics)
            issues.extend(val_issues[:self.config.max_issues - len(issues)])
            if self.config.include_spatial:
                spatial.extend(val_spatial)

        # Extract uncertainty diagnostics
        if uncertainty_result is not None:
            unc_metrics, unc_issues, unc_spatial = self._extract_uncertainty_diagnostics(uncertainty_result)
            metrics.extend(unc_metrics)
            issues.extend(unc_issues[:self.config.max_issues - len(issues)])
            if self.config.include_spatial:
                spatial.extend(unc_spatial)

        # Extract gating diagnostics
        if gating_decision is not None:
            gate_metrics = self._extract_gating_diagnostics(gating_decision)
            metrics.extend(gate_metrics)

        # Add performance metrics
        if self.config.include_performance:
            if performance_data:
                for op, duration in performance_data.items():
                    performance.append(PerformanceMetric(
                        operation=op,
                        duration_seconds=duration,
                    ))

            # Add diagnostic generation time
            performance.append(PerformanceMetric(
                operation="diagnostic_generation",
                duration_seconds=time.time() - start_time,
            ))

        # Compare with baseline
        comparison = None
        if baseline_diagnostics is not None:
            comparison = self._compare_with_baseline(
                baseline_diagnostics, metrics, issues
            )

        diagnostics = Diagnostics(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            level=self.config.level,
            metrics=metrics,
            issues=issues,
            spatial=spatial,
            temporal=temporal,
            performance=performance,
            comparison=comparison,
        )

        logger.info(f"Generated {len(metrics)} metrics, {len(issues)} issues in {time.time() - start_time:.2f}s")
        return diagnostics

    def _next_issue_id(self) -> str:
        """Generate next unique issue ID."""
        self._issue_counter += 1
        return f"ISS-{self._issue_counter:04d}"

    def _extract_sanity_diagnostics(
        self, result: Any
    ) -> Tuple[List[DiagnosticMetric], List[IssueDetail], List[SpatialDiagnostic]]:
        """Extract diagnostics from sanity suite result."""
        metrics = []
        issues = []
        spatial = []

        # Overall metrics
        if hasattr(result, 'overall_score'):
            metrics.append(DiagnosticMetric(
                name="sanity_overall_score",
                category=MetricCategory.SANITY,
                value=result.overall_score,
                unit="score",
                description="Overall sanity check score",
                threshold=0.7,
                status="pass" if result.overall_score >= 0.7 else "warning" if result.overall_score >= 0.5 else "fail",
            ))

        if hasattr(result, 'total_issues'):
            metrics.append(DiagnosticMetric(
                name="sanity_total_issues",
                category=MetricCategory.SANITY,
                value=result.total_issues,
                unit="count",
                description="Total number of sanity issues detected",
            ))

        if hasattr(result, 'duration_seconds'):
            metrics.append(DiagnosticMetric(
                name="sanity_duration",
                category=MetricCategory.PERFORMANCE,
                value=result.duration_seconds,
                unit="seconds",
                description="Time taken for sanity checks",
            ))

        # Extract spatial issues
        if hasattr(result, 'spatial') and result.spatial is not None:
            spatial_result = result.spatial
            if hasattr(spatial_result, 'autocorrelation'):
                metrics.append(DiagnosticMetric(
                    name="spatial_autocorrelation",
                    category=MetricCategory.SANITY,
                    value=spatial_result.autocorrelation,
                    description="Moran's I spatial autocorrelation",
                ))

            for issue in getattr(spatial_result, 'issues', []):
                issues.append(IssueDetail(
                    issue_id=self._next_issue_id(),
                    issue_type=getattr(issue, 'check_type', 'spatial').value if hasattr(getattr(issue, 'check_type', None), 'value') else str(getattr(issue, 'check_type', 'spatial')),
                    severity=getattr(issue, 'severity', 'medium').value if hasattr(getattr(issue, 'severity', None), 'value') else str(getattr(issue, 'severity', 'medium')),
                    category="spatial",
                    description=getattr(issue, 'description', ''),
                    location=getattr(issue, 'location', ''),
                    spatial_extent=getattr(issue, 'extent', None),
                    context={
                        "metric_value": getattr(issue, 'metric_value', None),
                        "threshold": getattr(issue, 'threshold', None),
                    },
                ))

        # Extract value issues
        if hasattr(result, 'values') and result.values is not None:
            value_result = result.values
            for issue in getattr(value_result, 'issues', []):
                issues.append(IssueDetail(
                    issue_id=self._next_issue_id(),
                    issue_type=getattr(issue, 'check_type', 'value').value if hasattr(getattr(issue, 'check_type', None), 'value') else str(getattr(issue, 'check_type', 'value')),
                    severity=getattr(issue, 'severity', 'medium').value if hasattr(getattr(issue, 'severity', None), 'value') else str(getattr(issue, 'severity', 'medium')),
                    category="value",
                    description=getattr(issue, 'description', ''),
                    affected_values=getattr(issue, 'sample_values', None),
                    context={
                        "metric_value": getattr(issue, 'metric_value', None),
                        "threshold": getattr(issue, 'threshold', None),
                    },
                ))

        # Extract artifact issues
        if hasattr(result, 'artifacts') and result.artifacts is not None:
            artifact_result = result.artifacts
            for artifact in getattr(artifact_result, 'artifacts', []):
                issues.append(IssueDetail(
                    issue_id=self._next_issue_id(),
                    issue_type=f"artifact_{getattr(artifact, 'artifact_type', 'unknown').value if hasattr(getattr(artifact, 'artifact_type', None), 'value') else str(getattr(artifact, 'artifact_type', 'unknown'))}",
                    severity=getattr(artifact, 'severity', 'medium').value if hasattr(getattr(artifact, 'severity', None), 'value') else str(getattr(artifact, 'severity', 'medium')),
                    category="artifact",
                    description=getattr(artifact, 'description', ''),
                    pixel_coordinates=getattr(artifact, 'location', None),
                    context={
                        "confidence": getattr(artifact, 'confidence', None),
                    },
                    suggested_action="Review artifact region and consider masking or reprocessing",
                ))

        return metrics, issues, spatial

    def _extract_validation_diagnostics(
        self, result: Any
    ) -> Tuple[List[DiagnosticMetric], List[IssueDetail], List[SpatialDiagnostic]]:
        """Extract diagnostics from validation result."""
        metrics = []
        issues = []
        spatial = []

        # Cross-model validation
        if hasattr(result, 'agreement_score'):
            metrics.append(DiagnosticMetric(
                name="validation_agreement_score",
                category=MetricCategory.VALIDATION,
                value=result.agreement_score,
                unit="score",
                description="Overall agreement between models/sensors",
                threshold=0.7,
                status="pass" if result.agreement_score >= 0.7 else "warning" if result.agreement_score >= 0.5 else "fail",
            ))

        if hasattr(result, 'iou'):
            metrics.append(DiagnosticMetric(
                name="validation_iou",
                category=MetricCategory.VALIDATION,
                value=result.iou,
                description="Intersection over Union",
            ))

        if hasattr(result, 'kappa'):
            metrics.append(DiagnosticMetric(
                name="validation_kappa",
                category=MetricCategory.VALIDATION,
                value=result.kappa,
                description="Cohen's Kappa coefficient",
            ))

        # Extract disagreement regions as issues
        if hasattr(result, 'disagreement_regions'):
            for i, region in enumerate(result.disagreement_regions[:20]):
                issues.append(IssueDetail(
                    issue_id=self._next_issue_id(),
                    issue_type="cross_validation_disagreement",
                    severity="medium",
                    category="validation",
                    description=f"Disagreement region {i+1}",
                    spatial_extent=getattr(region, 'geometry', None),
                    context={
                        "area_km2": getattr(region, 'area_km2', None),
                        "disagreeing_sources": getattr(region, 'sources', []),
                    },
                ))

        # Extract pairwise comparisons
        if hasattr(result, 'pairwise_comparisons'):
            for comp in result.pairwise_comparisons:
                model_a = getattr(comp, 'model_a', 'model_a')
                model_b = getattr(comp, 'model_b', 'model_b')
                agreement = getattr(comp, 'agreement_score', 0)

                metrics.append(DiagnosticMetric(
                    name=f"pairwise_agreement_{model_a}_vs_{model_b}",
                    category=MetricCategory.VALIDATION,
                    value=agreement,
                    description=f"Agreement between {model_a} and {model_b}",
                ))

                if agreement < 0.5:
                    issues.append(IssueDetail(
                        issue_id=self._next_issue_id(),
                        issue_type="low_model_agreement",
                        severity="high",
                        category="validation",
                        description=f"Low agreement ({agreement:.1%}) between {model_a} and {model_b}",
                        context={
                            "model_a": model_a,
                            "model_b": model_b,
                            "agreement": agreement,
                        },
                        suggested_action="Review model outputs and consider ensemble approach",
                    ))

        return metrics, issues, spatial

    def _extract_uncertainty_diagnostics(
        self, result: Any
    ) -> Tuple[List[DiagnosticMetric], List[IssueDetail], List[SpatialDiagnostic]]:
        """Extract diagnostics from uncertainty result."""
        metrics = []
        issues = []
        spatial = []

        mean_unc = getattr(result, 'mean_uncertainty', None) or getattr(result, 'mean', None)
        max_unc = getattr(result, 'max_uncertainty', None) or getattr(result, 'max', None)
        std_unc = getattr(result, 'std_uncertainty', None) or getattr(result, 'std', None)

        if mean_unc is not None:
            metrics.append(DiagnosticMetric(
                name="uncertainty_mean",
                category=MetricCategory.UNCERTAINTY,
                value=mean_unc,
                description="Mean uncertainty",
                threshold=0.3,
                status="pass" if mean_unc < 0.2 else "warning" if mean_unc < 0.3 else "fail",
            ))

        if max_unc is not None:
            metrics.append(DiagnosticMetric(
                name="uncertainty_max",
                category=MetricCategory.UNCERTAINTY,
                value=max_unc,
                description="Maximum uncertainty",
            ))

        if std_unc is not None:
            metrics.append(DiagnosticMetric(
                name="uncertainty_std",
                category=MetricCategory.UNCERTAINTY,
                value=std_unc,
                description="Uncertainty standard deviation",
            ))

        # High uncertainty as issue
        if mean_unc is not None and mean_unc > 0.3:
            issues.append(IssueDetail(
                issue_id=self._next_issue_id(),
                issue_type="high_uncertainty",
                severity="high",
                category="uncertainty",
                description=f"Mean uncertainty ({mean_unc:.2%}) exceeds threshold",
                context={
                    "mean": mean_unc,
                    "max": max_unc,
                    "threshold": 0.3,
                },
                suggested_action="Consider additional validation or flagging high-uncertainty regions",
            ))

        # Extract hotspots
        if hasattr(result, 'hotspots'):
            for i, hotspot in enumerate(result.hotspots[:10]):
                issues.append(IssueDetail(
                    issue_id=self._next_issue_id(),
                    issue_type="uncertainty_hotspot",
                    severity="medium",
                    category="uncertainty",
                    description=f"Uncertainty hotspot {i+1}",
                    spatial_extent=getattr(hotspot, 'geometry', None),
                    pixel_coordinates=getattr(hotspot, 'location', None),
                    context={
                        "peak_uncertainty": getattr(hotspot, 'peak_value', None),
                        "area_pixels": getattr(hotspot, 'area', None),
                    },
                ))

        # Add spatial uncertainty surface
        if hasattr(result, 'uncertainty_surface') and self.config.include_spatial:
            surface = result.uncertainty_surface
            if isinstance(surface, np.ndarray):
                spatial.append(SpatialDiagnostic(
                    name="uncertainty_surface",
                    description="Spatial uncertainty distribution",
                    data=surface,
                ))

        return metrics, issues, spatial

    def _extract_gating_diagnostics(self, decision: Any) -> List[DiagnosticMetric]:
        """Extract diagnostics from gating decision."""
        metrics = []

        if hasattr(decision, 'status'):
            status_val = decision.status.value if hasattr(decision.status, 'value') else str(decision.status)
            metrics.append(DiagnosticMetric(
                name="gating_status",
                category=MetricCategory.GATING,
                value=status_val,
                description="Quality gate status",
                status="pass" if status_val == "PASS" else "warning" if status_val == "PASS_WITH_WARNINGS" else "fail",
            ))

        if hasattr(decision, 'confidence_score'):
            metrics.append(DiagnosticMetric(
                name="gating_confidence",
                category=MetricCategory.GATING,
                value=decision.confidence_score,
                description="Confidence score from gating",
                threshold=0.6,
                status="pass" if decision.confidence_score >= 0.6 else "warning" if decision.confidence_score >= 0.4 else "fail",
            ))

        if hasattr(decision, 'rules_evaluated'):
            metrics.append(DiagnosticMetric(
                name="gating_rules_evaluated",
                category=MetricCategory.GATING,
                value=decision.rules_evaluated,
                description="Number of rules evaluated",
            ))

        if hasattr(decision, 'rules_passed'):
            metrics.append(DiagnosticMetric(
                name="gating_rules_passed",
                category=MetricCategory.GATING,
                value=decision.rules_passed,
                description="Number of rules passed",
            ))

        return metrics

    def _compare_with_baseline(
        self,
        baseline: Diagnostics,
        current_metrics: List[DiagnosticMetric],
        current_issues: List[IssueDetail],
    ) -> ComparisonResult:
        """Compare current diagnostics with baseline."""
        metric_changes: Dict[str, Dict[str, float]] = {}
        new_issues: List[str] = []
        resolved_issues: List[str] = []
        status_changed = False

        # Build lookup for baseline metrics
        baseline_metrics = {m.name: m.value for m in baseline.metrics}

        # Compare metrics
        for metric in current_metrics:
            if metric.name in baseline_metrics:
                old_val = baseline_metrics[metric.name]
                new_val = metric.value
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if abs(old_val) > 1e-10:
                        change_pct = (new_val - old_val) / abs(old_val) * 100
                    else:
                        change_pct = 0 if abs(new_val) < 1e-10 else 100
                    metric_changes[metric.name] = {
                        "old": old_val,
                        "new": new_val,
                        "change_percent": _safe_round(change_pct, 2) or 0.0,
                    }

        # Compare issues
        baseline_issue_types = {(i.issue_type, i.category) for i in baseline.issues}
        current_issue_types = {(i.issue_type, i.category) for i in current_issues}

        for issue_type, category in current_issue_types - baseline_issue_types:
            new_issues.append(f"{category}:{issue_type}")

        for issue_type, category in baseline_issue_types - current_issue_types:
            resolved_issues.append(f"{category}:{issue_type}")

        # Check status change
        baseline_status = next((m.value for m in baseline.metrics if m.name == "gating_status"), None)
        current_status = next((m.value for m in current_metrics if m.name == "gating_status"), None)
        if baseline_status and current_status and baseline_status != current_status:
            status_changed = True

        # Generate summary
        summary_parts = []
        if new_issues:
            summary_parts.append(f"{len(new_issues)} new issues")
        if resolved_issues:
            summary_parts.append(f"{len(resolved_issues)} resolved issues")
        if status_changed:
            summary_parts.append(f"status changed from {baseline_status} to {current_status}")
        if not summary_parts:
            summary_parts.append("no significant changes")

        return ComparisonResult(
            baseline_id=baseline.run_id,
            compare_id="current",
            metric_changes=metric_changes,
            new_issues=new_issues,
            resolved_issues=resolved_issues,
            status_changed=status_changed,
            summary="; ".join(summary_parts),
        )


def generate_diagnostics(
    results: Dict[str, Any],
    run_id: Optional[str] = None,
    level: DiagnosticLevel = DiagnosticLevel.STANDARD,
) -> Diagnostics:
    """
    Convenience function to generate diagnostics.

    Args:
        results: Dictionary with keys: sanity, validation, uncertainty, gating, performance
        run_id: Optional run identifier
        level: Diagnostic detail level

    Returns:
        Generated Diagnostics object
    """
    config = DiagnosticConfig(level=level)
    generator = DiagnosticGenerator(config)

    return generator.generate(
        run_id=run_id,
        sanity_result=results.get("sanity"),
        validation_result=results.get("validation"),
        uncertainty_result=results.get("uncertainty"),
        gating_decision=results.get("gating"),
        performance_data=results.get("performance"),
        baseline_diagnostics=results.get("baseline"),
    )
