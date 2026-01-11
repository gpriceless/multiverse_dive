"""
Tests for Quality Reporting Module.

Tests cover:
- QA report generation in multiple formats (JSON, HTML, Markdown, Text)
- Diagnostic output generation
- Report data extraction from various QC results
- Edge cases and error handling
"""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from core.quality.reporting import (
    # QA Report
    QAReportGenerator,
    ReportFormat,
    ReportLevel,
    ReportSection,
    ReportConfig,
    ReportMetadata,
    QualitySummary,
    CheckReport,
    CrossValidationReport,
    UncertaintySummaryReport,
    GatingReport,
    FlagReport,
    ActionReport,
    ExpertReviewReport,
    Recommendation,
    QAReport,
    generate_qa_report,
    # Diagnostics
    DiagnosticGenerator,
    DiagnosticLevel,
    DiagnosticType,
    MetricCategory,
    DiagnosticConfig,
    DiagnosticMetric,
    IssueDetail,
    SpatialDiagnostic,
    TemporalDiagnostic,
    PerformanceMetric,
    ComparisonResult,
    Diagnostics,
    generate_diagnostics,
)


# =============================================================================
# Mock Data Classes for Testing
# =============================================================================

class MockSeverity(Enum):
    """Mock severity enum."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MockCheckType(Enum):
    """Mock check type enum."""
    VALUE_RANGE = "value_range"
    SPATIAL_COHERENCE = "spatial_coherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class MockArtifactType(Enum):
    """Mock artifact type."""
    STRIPE = "stripe"
    HOT_PIXEL = "hot_pixel"


class MockGateStatus(Enum):
    """Mock gate status."""
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass
class MockIssue:
    """Mock issue for testing."""
    check_type: MockCheckType
    severity: MockSeverity
    description: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class MockArtifact:
    """Mock artifact for testing."""
    artifact_type: MockArtifactType
    severity: MockSeverity
    description: str
    confidence: float = 0.8


@dataclass
class MockSpatialResult:
    """Mock spatial check result."""
    is_coherent: bool = True
    issues: List[MockIssue] = field(default_factory=list)
    autocorrelation: float = 0.7
    critical_count: int = 0
    high_count: int = 0


@dataclass
class MockValueResult:
    """Mock value check result."""
    is_plausible: bool = True
    issues: List[MockIssue] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0


@dataclass
class MockArtifactResult:
    """Mock artifact detection result."""
    has_artifacts: bool = False
    artifacts: List[MockArtifact] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0


@dataclass
class MockSanityResult:
    """Mock sanity suite result."""
    passes_sanity: bool = True
    overall_score: float = 0.9
    total_issues: int = 0
    critical_issues: int = 0
    spatial: Optional[MockSpatialResult] = None
    values: Optional[MockValueResult] = None
    temporal: Optional[Any] = None
    artifacts: Optional[MockArtifactResult] = None
    summary: str = "All checks passed"
    duration_seconds: float = 1.5


@dataclass
class MockPairwiseComparison:
    """Mock pairwise comparison."""
    model_a: str
    model_b: str
    agreement_score: float
    iou: float = 0.0
    kappa: float = 0.0


@dataclass
class MockValidationResult:
    """Mock validation result."""
    agreement_score: float = 0.85
    iou: float = 0.78
    kappa: float = 0.72
    pairwise_comparisons: List[MockPairwiseComparison] = field(default_factory=list)
    disagreement_regions: List[Dict[str, Any]] = field(default_factory=list)
    consensus_method: str = "weighted_mean"


@dataclass
class MockUncertaintyResult:
    """Mock uncertainty result."""
    mean_uncertainty: float = 0.15
    max_uncertainty: float = 0.45
    std_uncertainty: float = 0.08
    high_uncertainty_percent: float = 12.5
    high_uncertainty_area_km2: float = 45.2
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    calibration_score: float = 0.92


@dataclass
class MockGatingDecision:
    """Mock gating decision."""
    status: MockGateStatus = MockGateStatus.PASS
    confidence_score: float = 0.88
    rules_evaluated: int = 10
    rules_passed: int = 9
    degraded_mode: bool = False
    degraded_level: int = 0
    rule_results: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class MockFlag:
    """Mock quality flag."""
    flag_id: str
    flag_name: str
    severity: MockSeverity
    affected_percent: float = 5.0
    affected_area_km2: float = 12.3
    reason: str = "Low confidence region"


@dataclass
class MockAction:
    """Mock action."""
    action: str
    reason: str
    affected_area: Optional[Dict[str, Any]] = None


@dataclass
class MockExpertReview:
    """Mock expert review."""
    required: bool = False
    reason: str = ""
    priority: str = "normal"
    deadline: Optional[datetime] = None
    reviewer_assigned: Optional[str] = None
    status: str = "not_required"


# =============================================================================
# QA Report Tests
# =============================================================================

class TestReportMetadata:
    """Tests for ReportMetadata."""

    def test_default_creation(self):
        """Test default metadata creation."""
        meta = ReportMetadata(
            event_id="evt_test_001",
            product_id="prod_test_001",
        )
        assert meta.event_id == "evt_test_001"
        assert meta.product_id == "prod_test_001"
        assert meta.format == ReportFormat.JSON
        assert meta.level == ReportLevel.STANDARD
        assert meta.generated_at is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = ReportMetadata(
            event_id="evt_test_001",
            product_id="prod_test_001",
            format=ReportFormat.HTML,
            level=ReportLevel.DETAILED,
        )
        d = meta.to_dict()
        assert d["event_id"] == "evt_test_001"
        assert d["format"] == "html"
        assert d["level"] == "detailed"


class TestQualitySummary:
    """Tests for QualitySummary."""

    def test_default_creation(self):
        """Test default summary creation."""
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.95,
        )
        assert summary.overall_status == "PASS"
        assert summary.confidence_score == 0.95
        assert summary.total_checks == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = QualitySummary(
            overall_status="PASS_WITH_WARNINGS",
            confidence_score=0.85,
            total_checks=10,
            passed_checks=8,
            warning_checks=2,
            failed_checks=0,
            key_findings=["Minor spatial artifacts detected"],
        )
        d = summary.to_dict()
        assert d["overall_status"] == "PASS_WITH_WARNINGS"
        assert d["confidence_score"] == 0.85
        assert len(d["key_findings"]) == 1


class TestCheckReport:
    """Tests for CheckReport."""

    def test_basic_check(self):
        """Test basic check report."""
        check = CheckReport(
            check_name="value_range",
            category="value",
            status="pass",
            details="All values within expected range",
        )
        assert check.check_name == "value_range"
        assert check.status == "pass"

    def test_check_with_metrics(self):
        """Test check report with metrics."""
        check = CheckReport(
            check_name="spatial_coherence",
            category="spatial",
            status="warning",
            metric_value=0.55,
            threshold=0.6,
            details="Spatial coherence below threshold",
        )
        d = check.to_dict()
        assert d["metric_value"] == 0.55
        assert d["threshold"] == 0.6

    def test_nan_metric_handling(self):
        """Test NaN metric value handling."""
        check = CheckReport(
            check_name="test_check",
            category="value",
            status="pass",
            metric_value=float("nan"),
        )
        d = check.to_dict()
        assert d.get("metric_value") is None


class TestCrossValidationReport:
    """Tests for CrossValidationReport."""

    def test_basic_creation(self):
        """Test basic cross-validation report."""
        cv = CrossValidationReport(
            methods_compared=["sar_threshold", "ndwi_optical"],
            agreement_score=0.85,
            iou=0.78,
            kappa=0.72,
        )
        assert len(cv.methods_compared) == 2
        assert cv.agreement_score == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cv = CrossValidationReport(
            methods_compared=["model_a", "model_b"],
            agreement_score=0.9,
            consensus_method="weighted_mean",
        )
        d = cv.to_dict()
        assert "agreement_metrics" in d
        assert d["consensus_method_used"] == "weighted_mean"


class TestQAReport:
    """Tests for QAReport."""

    def test_basic_report(self):
        """Test basic report creation."""
        metadata = ReportMetadata(
            event_id="evt_flood_001",
            product_id="prod_extent_001",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.92,
        )
        report = QAReport(metadata=metadata, summary=summary)

        assert report.metadata.event_id == "evt_flood_001"
        assert report.summary.overall_status == "PASS"

    def test_to_dict_schema_compliance(self):
        """Test report dictionary matches schema structure."""
        metadata = ReportMetadata(
            event_id="evt_flood_001",
            product_id="prod_extent_001",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.92,
        )
        report = QAReport(metadata=metadata, summary=summary)
        d = report.to_dict()

        # Check required schema fields
        assert "event_id" in d
        assert "product_id" in d
        assert "timestamp" in d
        assert "overall_status" in d
        assert "confidence_score" in d
        assert "checks" in d

    def test_to_json(self):
        """Test JSON conversion."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
        )
        report = QAReport(metadata=metadata, summary=summary)
        json_str = report.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["event_id"] == "evt_test"

    def test_to_markdown(self):
        """Test Markdown conversion."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
            total_checks=5,
            passed_checks=5,
        )
        report = QAReport(
            metadata=metadata,
            summary=summary,
            checks=[
                CheckReport(
                    check_name="test_check",
                    category="value",
                    status="pass",
                    details="Test passed",
                )
            ],
        )
        md = report.to_markdown()

        assert "# QA Report" in md
        assert "prod_test" in md
        assert "PASS" in md

    def test_to_html(self):
        """Test HTML conversion."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS_WITH_WARNINGS",
            confidence_score=0.85,
        )
        report = QAReport(metadata=metadata, summary=summary)
        html = report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "PASS_WITH_WARNINGS" in html

    def test_to_text(self):
        """Test plain text conversion."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="BLOCKED",
            confidence_score=0.3,
        )
        report = QAReport(metadata=metadata, summary=summary)
        text = report.to_text()

        assert "QA REPORT" in text
        assert "BLOCKED" in text

    def test_save_json(self):
        """Test saving to JSON file."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
        )
        report = QAReport(metadata=metadata, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["event_id"] == "evt_test"

    def test_save_html(self):
        """Test saving to HTML file."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
        )
        report = QAReport(metadata=metadata, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            report.save(path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content


class TestQAReportGenerator:
    """Tests for QAReportGenerator."""

    def test_basic_generation(self):
        """Test basic report generation."""
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_flood_001",
            product_id="prod_extent_001",
        )

        assert report.metadata.event_id == "evt_flood_001"
        assert report.summary.overall_status == "PASS"

    def test_generation_with_sanity_result(self):
        """Test generation with sanity result."""
        sanity = MockSanityResult(
            passes_sanity=True,
            overall_score=0.85,
            total_issues=2,
            spatial=MockSpatialResult(
                is_coherent=True,
                issues=[
                    MockIssue(
                        check_type=MockCheckType.SPATIAL_COHERENCE,
                        severity=MockSeverity.MEDIUM,
                        description="Minor coherence issue",
                    )
                ],
            ),
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            sanity_result=sanity,
        )

        assert len(report.checks) > 0

    def test_generation_with_validation_result(self):
        """Test generation with validation result."""
        validation = MockValidationResult(
            agreement_score=0.85,
            pairwise_comparisons=[
                MockPairwiseComparison("model_a", "model_b", 0.9),
            ],
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            validation_result=validation,
        )

        assert report.cross_validation is not None
        # Agreement score is averaged from pairwise comparisons
        assert report.cross_validation.agreement_score == 0.9

    def test_generation_with_uncertainty_result(self):
        """Test generation with uncertainty result."""
        uncertainty = MockUncertaintyResult(
            mean_uncertainty=0.15,
            max_uncertainty=0.45,
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            uncertainty_result=uncertainty,
        )

        assert report.uncertainty is not None
        assert report.uncertainty.mean_uncertainty == 0.15

    def test_generation_with_gating_decision(self):
        """Test generation with gating decision."""
        gating = MockGatingDecision(
            status=MockGateStatus.PASS_WITH_WARNINGS,
            confidence_score=0.75,
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            gating_decision=gating,
        )

        assert report.summary.overall_status == "PASS_WITH_WARNINGS"
        assert report.gating is not None

    def test_generation_with_flags(self):
        """Test generation with quality flags."""
        flags = [
            MockFlag(
                flag_id="FLG_001",
                flag_name="Low Confidence",
                severity=MockSeverity.MEDIUM,
            )
        ]

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            flags=flags,
        )

        assert len(report.flags) == 1
        assert report.flags[0].flag_id == "FLG_001"

    def test_generation_with_expert_review(self):
        """Test generation with expert review requirement."""
        review = MockExpertReview(
            required=True,
            reason="High uncertainty in coastal regions",
            priority="high",
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            expert_review=review,
        )

        assert report.expert_review is not None
        assert report.expert_review.required

    def test_key_findings_generation(self):
        """Test key findings are generated."""
        sanity = MockSanityResult(
            passes_sanity=False,
            critical_issues=2,
            artifacts=MockArtifactResult(
                has_artifacts=True,
                artifacts=[MockArtifact(MockArtifactType.STRIPE, MockSeverity.HIGH, "Stripe artifact")],
            ),
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            sanity_result=sanity,
        )

        assert len(report.summary.key_findings) > 0

    def test_recommendations_generation(self):
        """Test recommendations are generated."""
        uncertainty = MockUncertaintyResult(
            mean_uncertainty=0.4,  # High uncertainty
        )

        config = ReportConfig(include_recommendations=True)
        generator = QAReportGenerator(config)
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            uncertainty_result=uncertainty,
        )

        assert len(report.recommendations) > 0


class TestGenerateQAReport:
    """Tests for generate_qa_report convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        report = generate_qa_report(
            event_id="evt_test",
            product_id="prod_test",
            results={},
        )

        assert report.metadata.event_id == "evt_test"

    def test_with_results_dict(self):
        """Test with results dictionary."""
        results = {
            "sanity": MockSanityResult(),
            "validation": MockValidationResult(),
            "uncertainty": MockUncertaintyResult(),
        }

        report = generate_qa_report(
            event_id="evt_test",
            product_id="prod_test",
            results=results,
        )

        assert report.cross_validation is not None
        assert report.uncertainty is not None


# =============================================================================
# Diagnostic Tests
# =============================================================================

class TestDiagnosticMetric:
    """Tests for DiagnosticMetric."""

    def test_scalar_metric(self):
        """Test scalar metric."""
        metric = DiagnosticMetric(
            name="test_metric",
            category=MetricCategory.SANITY,
            value=0.85,
            unit="score",
            description="Test metric",
        )
        assert metric.value == 0.85
        assert metric.category == MetricCategory.SANITY

    def test_array_metric(self):
        """Test array metric."""
        metric = DiagnosticMetric(
            name="histogram",
            category=MetricCategory.VALIDATION,
            value=np.array([1, 2, 3, 4, 5]),
        )
        d = metric.to_dict()
        assert d["value"] == [1, 2, 3, 4, 5]


class TestIssueDetail:
    """Tests for IssueDetail."""

    def test_basic_issue(self):
        """Test basic issue creation."""
        issue = IssueDetail(
            issue_id="ISS-0001",
            issue_type="spatial_coherence",
            severity="high",
            category="spatial",
            description="Low spatial coherence detected",
        )
        assert issue.issue_id == "ISS-0001"

    def test_issue_with_spatial_extent(self):
        """Test issue with spatial extent."""
        issue = IssueDetail(
            issue_id="ISS-0002",
            issue_type="artifact",
            severity="medium",
            category="artifact",
            description="Stripe artifact",
            spatial_extent={"type": "Polygon", "coordinates": [[]]},
        )
        d = issue.to_dict()
        assert "spatial_extent" in d


class TestSpatialDiagnostic:
    """Tests for SpatialDiagnostic."""

    def test_basic_spatial(self):
        """Test basic spatial diagnostic."""
        data = np.random.rand(100, 100)
        spatial = SpatialDiagnostic(
            name="uncertainty_surface",
            description="Spatial uncertainty distribution",
            data=data,
        )
        assert spatial.data.shape == (100, 100)

    def test_compute_statistics(self):
        """Test statistics computation."""
        data = np.array([[1, 2], [3, 4]], dtype=float)
        spatial = SpatialDiagnostic(
            name="test",
            description="Test",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["mean"] == 2.5

    def test_statistics_with_nan(self):
        """Test statistics with NaN values."""
        data = np.array([[1, np.nan], [3, 4]], dtype=float)
        spatial = SpatialDiagnostic(
            name="test",
            description="Test",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["count"] == 3
        assert np.isclose(stats["mean"], 8/3)


class TestTemporalDiagnostic:
    """Tests for TemporalDiagnostic."""

    def test_basic_temporal(self):
        """Test basic temporal diagnostic."""
        temporal = TemporalDiagnostic(
            name="confidence_time_series",
            description="Confidence over time",
            timestamps=[datetime(2024, 1, i) for i in range(1, 6)],
            values=[0.8, 0.82, 0.85, 0.83, 0.87],
        )
        assert len(temporal.timestamps) == 5
        assert len(temporal.values) == 5

    def test_compute_statistics(self):
        """Test statistics computation."""
        temporal = TemporalDiagnostic(
            name="test",
            description="Test",
            timestamps=[datetime(2024, 1, i) for i in range(1, 4)],
            values=[1.0, 2.0, 3.0],
        )
        stats = temporal.compute_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["mean_change"] == 1.0


class TestDiagnostics:
    """Tests for Diagnostics."""

    def test_basic_diagnostics(self):
        """Test basic diagnostics creation."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
        )
        assert diag.run_id == "test_001"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="test_metric",
                    category=MetricCategory.SANITY,
                    value=0.9,
                )
            ],
        )
        d = diag.to_dict()
        assert d["run_id"] == "test_001"
        assert len(d["metrics"]) == 1

    def test_to_json(self):
        """Test JSON conversion."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
        )
        json_str = diag.to_json()
        data = json.loads(json_str)
        assert data["run_id"] == "test_001"

    def test_export_json(self):
        """Test JSON export."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "diagnostics.json"
            diag.export_json(path)
            assert path.exists()

    def test_export_csv(self):
        """Test CSV export."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="test_metric",
                    category=MetricCategory.SANITY,
                    value=0.9,
                )
            ],
            issues=[
                IssueDetail(
                    issue_id="ISS-0001",
                    issue_type="test",
                    severity="medium",
                    category="test",
                    description="Test issue",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            diag.export_csv(tmpdir)
            assert (Path(tmpdir) / "metrics.csv").exists()
            assert (Path(tmpdir) / "issues.csv").exists()

    def test_get_summary(self):
        """Test summary generation."""
        diag = Diagnostics(
            run_id="test_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="test_metric",
                    category=MetricCategory.SANITY,
                    value=0.9,
                )
            ],
            issues=[
                IssueDetail(
                    issue_id="ISS-0001",
                    issue_type="test",
                    severity="high",
                    category="test",
                    description="Test issue",
                )
            ],
        )
        summary = diag.get_summary()
        assert "test_001" in summary
        assert "Metrics: 1" in summary
        assert "Issues: 1" in summary


class TestDiagnosticGenerator:
    """Tests for DiagnosticGenerator."""

    def test_basic_generation(self):
        """Test basic diagnostic generation."""
        generator = DiagnosticGenerator()
        diag = generator.generate(run_id="test_001")

        assert diag.run_id == "test_001"
        assert diag.level == DiagnosticLevel.STANDARD

    def test_generation_with_sanity_result(self):
        """Test generation with sanity result."""
        sanity = MockSanityResult(
            overall_score=0.85,
            duration_seconds=1.5,
            spatial=MockSpatialResult(
                autocorrelation=0.7,
                issues=[
                    MockIssue(
                        check_type=MockCheckType.SPATIAL_COHERENCE,
                        severity=MockSeverity.MEDIUM,
                        description="Test issue",
                    )
                ],
            ),
        )

        generator = DiagnosticGenerator()
        diag = generator.generate(sanity_result=sanity)

        assert len(diag.metrics) > 0
        assert any(m.name == "sanity_overall_score" for m in diag.metrics)

    def test_generation_with_validation_result(self):
        """Test generation with validation result."""
        validation = MockValidationResult(
            agreement_score=0.85,
            iou=0.78,
            kappa=0.72,
            pairwise_comparisons=[
                MockPairwiseComparison("a", "b", 0.4),  # Low agreement
            ],
        )

        generator = DiagnosticGenerator()
        diag = generator.generate(validation_result=validation)

        assert len(diag.metrics) > 0
        assert len(diag.issues) > 0  # Should flag low agreement

    def test_generation_with_uncertainty_result(self):
        """Test generation with uncertainty result."""
        uncertainty = MockUncertaintyResult(
            mean_uncertainty=0.35,  # High - should create issue
        )

        generator = DiagnosticGenerator()
        diag = generator.generate(uncertainty_result=uncertainty)

        assert len(diag.metrics) > 0
        assert any(i.issue_type == "high_uncertainty" for i in diag.issues)

    def test_generation_with_gating_decision(self):
        """Test generation with gating decision."""
        gating = MockGatingDecision(
            status=MockGateStatus.REVIEW_REQUIRED,
            confidence_score=0.55,
        )

        generator = DiagnosticGenerator()
        diag = generator.generate(gating_decision=gating)

        assert any(m.name == "gating_status" for m in diag.metrics)
        assert any(m.name == "gating_confidence" for m in diag.metrics)

    def test_generation_with_performance_data(self):
        """Test generation with performance data."""
        config = DiagnosticConfig(include_performance=True)
        generator = DiagnosticGenerator(config)
        diag = generator.generate(
            performance_data={
                "sanity_check": 1.5,
                "validation": 2.3,
            }
        )

        assert len(diag.performance) >= 2  # Includes diagnostic_generation

    def test_comparison_with_baseline(self):
        """Test comparison with baseline diagnostics."""
        # Create baseline
        baseline = Diagnostics(
            run_id="baseline_001",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="sanity_overall_score",
                    category=MetricCategory.SANITY,
                    value=0.85,
                )
            ],
            issues=[
                IssueDetail(
                    issue_id="ISS-0001",
                    issue_type="old_issue",
                    severity="medium",
                    category="sanity",
                    description="Old issue",
                )
            ],
        )

        # Generate new with different issues
        sanity = MockSanityResult(
            overall_score=0.90,  # Improved
            spatial=MockSpatialResult(
                issues=[
                    MockIssue(
                        check_type=MockCheckType.VALUE_RANGE,
                        severity=MockSeverity.LOW,
                        description="New issue",
                    )
                ]
            ),
        )

        generator = DiagnosticGenerator()
        diag = generator.generate(
            sanity_result=sanity,
            baseline_diagnostics=baseline,
        )

        assert diag.comparison is not None
        assert diag.comparison.baseline_id == "baseline_001"


class TestGenerateDiagnostics:
    """Tests for generate_diagnostics convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        diag = generate_diagnostics({})
        assert diag.run_id is not None

    def test_with_results_dict(self):
        """Test with results dictionary."""
        results = {
            "sanity": MockSanityResult(),
            "validation": MockValidationResult(),
            "uncertainty": MockUncertaintyResult(),
        }

        diag = generate_diagnostics(results)
        assert len(diag.metrics) > 0

    def test_with_level(self):
        """Test with different diagnostic levels."""
        diag = generate_diagnostics({}, level=DiagnosticLevel.DETAILED)
        assert diag.level == DiagnosticLevel.DETAILED


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test with empty/None results."""
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
        )
        assert report.summary.overall_status == "PASS"

    def test_none_values(self):
        """Test handling of None values in results."""
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            sanity_result=None,
            validation_result=None,
            uncertainty_result=None,
        )
        assert report is not None

    def test_inf_values_in_metrics(self):
        """Test handling of infinite values."""
        check = CheckReport(
            check_name="test",
            category="test",
            status="pass",
            metric_value=float("inf"),
        )
        d = check.to_dict()
        assert d.get("metric_value") is None

    def test_empty_sanity_issues(self):
        """Test sanity result with empty issues."""
        sanity = MockSanityResult(
            spatial=MockSpatialResult(issues=[]),
            values=MockValueResult(issues=[]),
        )

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
            sanity_result=sanity,
        )
        assert report is not None

    def test_diagnostic_with_empty_arrays(self):
        """Test spatial diagnostic with empty/zero arrays."""
        data = np.array([])
        spatial = SpatialDiagnostic(
            name="test",
            description="Test",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["count"] == 0

    def test_temporal_with_single_point(self):
        """Test temporal diagnostic with single data point."""
        temporal = TemporalDiagnostic(
            name="test",
            description="Test",
            timestamps=[datetime(2024, 1, 1)],
            values=[0.5],
        )
        stats = temporal.compute_statistics()
        assert stats["count"] == 1
        assert "mean_change" not in stats  # Can't compute with single point


class TestIntegration:
    """Integration tests for reporting module."""

    def test_full_report_generation_workflow(self):
        """Test complete report generation workflow."""
        # Create mock results
        sanity = MockSanityResult(
            passes_sanity=True,
            overall_score=0.88,
            spatial=MockSpatialResult(
                issues=[
                    MockIssue(
                        MockCheckType.SPATIAL_COHERENCE,
                        MockSeverity.LOW,
                        "Minor coherence variation",
                    )
                ]
            ),
        )

        validation = MockValidationResult(
            agreement_score=0.85,
            pairwise_comparisons=[
                MockPairwiseComparison("sar", "optical", 0.82),
            ],
        )

        uncertainty = MockUncertaintyResult(
            mean_uncertainty=0.12,
        )

        gating = MockGatingDecision(
            status=MockGateStatus.PASS_WITH_WARNINGS,
            confidence_score=0.82,
        )

        # Generate report
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_flood_miami_2024",
            product_id="prod_flood_extent_001",
            sanity_result=sanity,
            validation_result=validation,
            uncertainty_result=uncertainty,
            gating_decision=gating,
        )

        # Verify report content
        assert report.summary.overall_status == "PASS_WITH_WARNINGS"
        assert report.cross_validation is not None
        assert report.uncertainty is not None
        assert len(report.checks) > 0

        # Test all output formats
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            report.save(tmpdir / "report.json")
            report.save(tmpdir / "report.html")
            report.save(tmpdir / "report.md")
            report.save(tmpdir / "report.txt")

            assert (tmpdir / "report.json").exists()
            assert (tmpdir / "report.html").exists()
            assert (tmpdir / "report.md").exists()
            assert (tmpdir / "report.txt").exists()

    def test_full_diagnostics_workflow(self):
        """Test complete diagnostics workflow."""
        # Create mock results
        sanity = MockSanityResult(
            overall_score=0.9,
            duration_seconds=2.5,
        )

        validation = MockValidationResult(
            agreement_score=0.75,
        )

        # Generate diagnostics
        generator = DiagnosticGenerator()
        diag = generator.generate(
            run_id="diag_test_001",
            sanity_result=sanity,
            validation_result=validation,
            performance_data={"sanity": 2.5, "validation": 3.2},
        )

        # Verify content
        assert len(diag.metrics) > 0
        assert len(diag.performance) > 0

        # Get summary
        summary = diag.get_summary()
        assert "diag_test_001" in summary

        # Test exports
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            diag.export_json(tmpdir / "diagnostics.json")
            diag.export_csv(tmpdir / "csv")

            assert (tmpdir / "diagnostics.json").exists()
            assert (tmpdir / "csv" / "metrics.csv").exists()


# =============================================================================
# Additional Edge Case Tests for Bug Fixes
# =============================================================================

class TestNoneHandling:
    """Tests for None value handling across all reporting components."""

    def test_safe_round_with_none(self):
        """Test _safe_round handles None gracefully."""
        from core.quality.reporting.qa_report import _safe_round

        assert _safe_round(None) is None
        assert _safe_round(1.234) == 1.234
        assert _safe_round(float('nan')) is None
        assert _safe_round(float('inf')) is None
        assert _safe_round(float('-inf')) is None

    def test_quality_summary_with_nan_confidence(self):
        """Test QualitySummary handles NaN confidence_score."""
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=float('nan'),
        )
        d = summary.to_dict()
        # Should not crash, and should return None or 0.0
        assert d.get("confidence_score") == 0.0  # or 0.0 is returned

    def test_quality_summary_with_inf_confidence(self):
        """Test QualitySummary handles Inf confidence_score."""
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=float('inf'),
        )
        d = summary.to_dict()
        assert d.get("confidence_score") == 0.0

    def test_uncertainty_report_with_none_calibration(self):
        """Test UncertaintySummaryReport with None calibration_score."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc = UncertaintySummaryReport(
            mean_uncertainty=0.15,
            calibration_score=None,
        )
        d = unc.to_dict()
        assert "calibration_score" not in d  # Should be omitted

    def test_uncertainty_report_with_nan_values(self):
        """Test UncertaintySummaryReport with NaN values."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc = UncertaintySummaryReport(
            mean_uncertainty=float('nan'),
            max_uncertainty=float('inf'),
        )
        d = unc.to_dict()
        # NaN/Inf should be replaced with 0.0
        assert d["mean_uncertainty"] == 0.0
        assert d["max_uncertainty"] == 0.0

    def test_temporal_diagnostic_with_none_values(self):
        """Test TemporalDiagnostic.compute_statistics with None in values."""
        temporal = TemporalDiagnostic(
            name="test",
            description="Test with None",
            timestamps=[datetime(2024, 1, i) for i in range(1, 5)],
            values=[1.0, None, 2.0, 3.0],
        )
        stats = temporal.compute_statistics()
        # Should skip None and compute stats on valid values
        assert stats["count"] == 3
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_temporal_diagnostic_all_none(self):
        """Test TemporalDiagnostic.compute_statistics with all None values."""
        temporal = TemporalDiagnostic(
            name="test",
            description="Test all None",
            timestamps=[datetime(2024, 1, i) for i in range(1, 4)],
            values=[None, None, None],
        )
        stats = temporal.compute_statistics()
        assert stats["count"] == 0

    def test_spatial_diagnostic_with_none_array(self):
        """Test SpatialDiagnostic with array containing object None."""
        # Create an object array with None (unusual but possible)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        spatial = SpatialDiagnostic(
            name="test",
            description="Normal test",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["count"] == 4
        assert stats["mean"] == 2.5

    def test_spatial_diagnostic_all_nan(self):
        """Test SpatialDiagnostic with all NaN values."""
        data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        spatial = SpatialDiagnostic(
            name="test",
            description="All NaN",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["count"] == 0

    def test_spatial_diagnostic_mixed_nan_inf(self):
        """Test SpatialDiagnostic with mixed NaN, Inf, and valid values."""
        data = np.array([[1.0, np.nan], [np.inf, 2.0]])
        spatial = SpatialDiagnostic(
            name="test",
            description="Mixed values",
            data=data,
        )
        stats = spatial.compute_statistics()
        assert stats["count"] == 2  # Only 1.0 and 2.0 are valid
        assert stats["min"] == 1.0
        assert stats["max"] == 2.0


class TestCheckReportEdgeCases:
    """Additional edge case tests for CheckReport."""

    def test_check_report_negative_inf(self):
        """Test CheckReport with negative infinity."""
        check = CheckReport(
            check_name="test",
            category="test",
            status="pass",
            metric_value=float("-inf"),
            threshold=float("-inf"),
        )
        d = check.to_dict()
        assert d.get("metric_value") is None
        assert d.get("threshold") is None

    def test_check_report_very_small_values(self):
        """Test CheckReport with very small values (not zero)."""
        check = CheckReport(
            check_name="test",
            category="test",
            status="pass",
            metric_value=1e-10,
            threshold=1e-15,
        )
        d = check.to_dict()
        assert d["metric_value"] is not None
        assert d["threshold"] is not None


class TestDiagnosticComparisonEdgeCases:
    """Tests for diagnostic comparison edge cases."""

    def test_comparison_with_zero_baseline(self):
        """Test comparison when baseline metric is zero."""
        # Create baseline with zero metric
        baseline = Diagnostics(
            run_id="baseline",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="zero_metric",
                    category=MetricCategory.SANITY,
                    value=0.0,
                )
            ],
        )

        # Create current with non-zero metric
        generator = DiagnosticGenerator()
        sanity = MockSanityResult(overall_score=0.5)
        diag = generator.generate(
            sanity_result=sanity,
            baseline_diagnostics=baseline,
        )

        # Should not crash on division
        assert diag.comparison is not None

    def test_comparison_with_same_metrics(self):
        """Test comparison when metrics are identical."""
        baseline = Diagnostics(
            run_id="baseline",
            timestamp=datetime.now(timezone.utc),
            level=DiagnosticLevel.STANDARD,
            metrics=[
                DiagnosticMetric(
                    name="sanity_overall_score",
                    category=MetricCategory.SANITY,
                    value=0.9,
                )
            ],
        )

        sanity = MockSanityResult(overall_score=0.9)
        generator = DiagnosticGenerator()
        diag = generator.generate(
            sanity_result=sanity,
            baseline_diagnostics=baseline,
        )

        assert diag.comparison is not None
        # Same value should show 0% change
        if "sanity_overall_score" in diag.comparison.metric_changes:
            assert diag.comparison.metric_changes["sanity_overall_score"]["change_percent"] == 0.0


class TestReportSaveFormats:
    """Additional tests for report saving edge cases."""

    def test_save_with_nested_directory(self):
        """Test saving to nested directory that doesn't exist."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
        )
        report = QAReport(metadata=metadata, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "report.json"
            report.save(nested_path)
            assert nested_path.exists()

    def test_save_format_inference(self):
        """Test format inference from file extension."""
        metadata = ReportMetadata(
            event_id="evt_test",
            product_id="prod_test",
        )
        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.9,
        )
        report = QAReport(metadata=metadata, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test each extension
            for ext, expected_content in [
                (".json", "{"),
                (".html", "<!DOCTYPE"),
                (".md", "# QA Report"),
                (".txt", "====="),
            ]:
                path = Path(tmpdir) / f"report{ext}"
                report.save(path)
                content = path.read_text()
                assert expected_content in content, f"Failed for extension {ext}"
