"""
Tests for Quality Control Reporting Module (Group I, Track 5/6).

Tests for QA report generation:
- Report formats (JSON, HTML, Markdown, Text)
- Report detail levels (Summary, Standard, Detailed, Debug)
- Data structures and serialization
- Integration with quality control results
- Edge cases and error handling
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ============================================================================
# REPORT ENUMS AND DATA STRUCTURES
# ============================================================================

class TestReportEnums:
    """Test report enumeration types."""

    def test_report_format_enum(self):
        """Test ReportFormat enum values."""
        from core.quality.reporting.qa_report import ReportFormat

        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.TEXT.value == "text"

    def test_report_level_enum(self):
        """Test ReportLevel enum values."""
        from core.quality.reporting.qa_report import ReportLevel

        assert ReportLevel.SUMMARY.value == "summary"
        assert ReportLevel.STANDARD.value == "standard"
        assert ReportLevel.DETAILED.value == "detailed"
        assert ReportLevel.DEBUG.value == "debug"

    def test_report_section_enum(self):
        """Test ReportSection enum values."""
        from core.quality.reporting.qa_report import ReportSection

        assert ReportSection.OVERVIEW.value == "overview"
        assert ReportSection.SANITY_CHECKS.value == "sanity_checks"
        assert ReportSection.CROSS_VALIDATION.value == "cross_validation"
        assert ReportSection.UNCERTAINTY.value == "uncertainty"
        assert ReportSection.GATING.value == "gating"


class TestReportMetadata:
    """Test ReportMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating ReportMetadata."""
        from core.quality.reporting.qa_report import ReportMetadata, ReportFormat, ReportLevel

        metadata = ReportMetadata(
            event_id="evt_test_001",
            product_id="prod_flood_001",
        )

        assert metadata.event_id == "evt_test_001"
        assert metadata.product_id == "prod_flood_001"
        assert metadata.format == ReportFormat.JSON
        assert metadata.level == ReportLevel.STANDARD

    def test_metadata_to_dict(self):
        """Test ReportMetadata serialization."""
        from core.quality.reporting.qa_report import ReportMetadata

        metadata = ReportMetadata(
            event_id="evt_test_001",
            product_id="prod_flood_001",
        )

        d = metadata.to_dict()
        assert d["event_id"] == "evt_test_001"
        assert d["product_id"] == "prod_flood_001"
        assert "generated_at" in d
        assert d["format"] == "json"
        assert d["level"] == "standard"


class TestQualitySummary:
    """Test QualitySummary dataclass."""

    def test_summary_creation(self):
        """Test creating QualitySummary."""
        from core.quality.reporting.qa_report import QualitySummary

        summary = QualitySummary(
            overall_status="PASS",
            confidence_score=0.95,
            total_checks=10,
            passed_checks=9,
            warning_checks=1,
            failed_checks=0,
        )

        assert summary.overall_status == "PASS"
        assert summary.confidence_score == 0.95
        assert summary.total_checks == 10
        assert summary.passed_checks == 9

    def test_summary_to_dict(self):
        """Test QualitySummary serialization."""
        from core.quality.reporting.qa_report import QualitySummary

        summary = QualitySummary(
            overall_status="PASS_WITH_WARNINGS",
            confidence_score=0.85,
            total_checks=5,
            passed_checks=4,
            warning_checks=1,
            failed_checks=0,
            key_findings=["Minor artifact detected"],
        )

        d = summary.to_dict()
        assert d["overall_status"] == "PASS_WITH_WARNINGS"
        assert d["confidence_score"] == 0.85
        assert d["key_findings"] == ["Minor artifact detected"]


class TestCheckReport:
    """Test CheckReport dataclass."""

    def test_check_report_creation(self):
        """Test creating CheckReport."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="spatial_autocorrelation",
            category="spatial",
            status="pass",
            metric_value=0.85,
            threshold=0.7,
            details="Spatial coherence is acceptable",
        )

        assert check.check_name == "spatial_autocorrelation"
        assert check.category == "spatial"
        assert check.status == "pass"
        assert check.metric_value == 0.85

    def test_check_report_to_dict(self):
        """Test CheckReport serialization."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="value_range",
            category="value",
            status="warning",
            metric_value=1.05,
            threshold=1.0,
            details="Some values slightly exceed expected range",
            recommendations=["Apply clipping to [0, 1]"],
        )

        d = check.to_dict()
        assert d["check_name"] == "value_range"
        assert d["category"] == "value"
        assert d["status"] == "warning"
        assert d["metric_value"] == 1.05
        assert d["recommendations"] == ["Apply clipping to [0, 1]"]

    def test_check_report_with_nan_metric(self):
        """Test CheckReport with NaN metric value."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="test_check",
            category="test",
            status="pass",
            metric_value=np.nan,
        )

        d = check.to_dict()
        assert d["metric_value"] is None  # NaN should be converted to None

    def test_check_report_with_inf_metric(self):
        """Test CheckReport with Inf metric value."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="test_check",
            category="test",
            status="pass",
            metric_value=np.inf,
        )

        d = check.to_dict()
        assert d["metric_value"] is None  # Inf should be converted to None

    def test_check_report_with_nan_threshold(self):
        """Test CheckReport with NaN threshold value."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="test_check",
            category="test",
            status="pass",
            metric_value=0.5,
            threshold=np.nan,
        )

        d = check.to_dict()
        assert d["threshold"] is None  # NaN should be converted to None

    def test_check_report_with_inf_threshold(self):
        """Test CheckReport with Inf threshold value."""
        from core.quality.reporting.qa_report import CheckReport

        check = CheckReport(
            check_name="test_check",
            category="test",
            status="pass",
            metric_value=0.5,
            threshold=np.inf,
        )

        d = check.to_dict()
        assert d["threshold"] is None  # Inf should be converted to None


class TestCrossValidationReport:
    """Test CrossValidationReport dataclass."""

    def test_cv_report_creation(self):
        """Test creating CrossValidationReport."""
        from core.quality.reporting.qa_report import CrossValidationReport

        cv_report = CrossValidationReport(
            methods_compared=["sar_threshold", "ndwi_optical"],
            agreement_score=0.82,
            iou=0.75,
            kappa=0.68,
            consensus_method="weighted_average",
        )

        assert cv_report.methods_compared == ["sar_threshold", "ndwi_optical"]
        assert cv_report.agreement_score == 0.82
        assert cv_report.iou == 0.75

    def test_cv_report_to_dict(self):
        """Test CrossValidationReport serialization."""
        from core.quality.reporting.qa_report import CrossValidationReport

        cv_report = CrossValidationReport(
            methods_compared=["model_a", "model_b"],
            agreement_score=0.90,
            kappa=0.85,
            consensus_method="majority_vote",
        )

        d = cv_report.to_dict()
        assert d["methods_compared"] == ["model_a", "model_b"]
        assert d["agreement_metrics"]["agreement_score"] == 0.90
        assert d["agreement_metrics"]["kappa"] == 0.85
        assert d["consensus_method_used"] == "majority_vote"

    def test_cv_report_with_nan_iou(self):
        """Test CrossValidationReport with NaN IoU value."""
        from core.quality.reporting.qa_report import CrossValidationReport

        cv_report = CrossValidationReport(
            methods_compared=["model_a", "model_b"],
            agreement_score=0.85,
            iou=np.nan,
        )

        d = cv_report.to_dict()
        assert "iou" not in d["agreement_metrics"]  # NaN should be omitted

    def test_cv_report_with_inf_kappa(self):
        """Test CrossValidationReport with Inf kappa value."""
        from core.quality.reporting.qa_report import CrossValidationReport

        cv_report = CrossValidationReport(
            methods_compared=["model_a", "model_b"],
            agreement_score=0.85,
            kappa=np.inf,
        )

        d = cv_report.to_dict()
        assert "kappa" not in d["agreement_metrics"]  # Inf should be omitted

    def test_cv_report_with_nan_agreement_score(self):
        """Test CrossValidationReport with NaN agreement score defaults to 0.0."""
        from core.quality.reporting.qa_report import CrossValidationReport

        cv_report = CrossValidationReport(
            methods_compared=["model_a", "model_b"],
            agreement_score=np.nan,
        )

        d = cv_report.to_dict()
        assert d["agreement_metrics"]["agreement_score"] == 0.0  # NaN defaults to 0.0


class TestUncertaintySummaryReport:
    """Test UncertaintySummaryReport dataclass."""

    def test_uncertainty_report_creation(self):
        """Test creating UncertaintySummaryReport."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc_report = UncertaintySummaryReport(
            mean_uncertainty=0.15,
            max_uncertainty=0.45,
            std_uncertainty=0.08,
            high_uncertainty_percent=12.5,
            hotspot_count=3,
        )

        assert unc_report.mean_uncertainty == 0.15
        assert unc_report.max_uncertainty == 0.45
        assert unc_report.hotspot_count == 3

    def test_uncertainty_report_to_dict(self):
        """Test UncertaintySummaryReport serialization."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc_report = UncertaintySummaryReport(
            mean_uncertainty=0.20,
            max_uncertainty=0.60,
            std_uncertainty=0.10,
            high_uncertainty_percent=15.0,
            high_uncertainty_area_km2=25.5,
            calibration_score=0.92,
        )

        d = unc_report.to_dict()
        assert d["mean_uncertainty"] == 0.20
        assert d["high_uncertainty_area_km2"] == 25.5
        assert d["calibration_score"] == 0.92

    def test_uncertainty_report_with_nan_values(self):
        """Test UncertaintySummaryReport with NaN values defaults to 0.0."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc_report = UncertaintySummaryReport(
            mean_uncertainty=np.nan,
            max_uncertainty=np.inf,
            std_uncertainty=0.1,
            high_uncertainty_percent=-np.inf,
            high_uncertainty_area_km2=25.5,
        )

        d = unc_report.to_dict()
        assert d["mean_uncertainty"] == 0.0  # NaN defaults to 0.0
        assert d["max_uncertainty"] == 0.0  # Inf defaults to 0.0
        assert d["high_uncertainty_percent"] == 0.0  # -Inf defaults to 0.0
        assert d["std_uncertainty"] == 0.1  # Valid value preserved

    def test_uncertainty_report_with_nan_calibration_score(self):
        """Test UncertaintySummaryReport with NaN calibration_score."""
        from core.quality.reporting.qa_report import UncertaintySummaryReport

        unc_report = UncertaintySummaryReport(
            mean_uncertainty=0.15,
            max_uncertainty=0.45,
            std_uncertainty=0.08,
            high_uncertainty_percent=12.5,
            calibration_score=np.nan,
        )

        d = unc_report.to_dict()
        assert d.get("calibration_score") is None  # NaN calibration_score becomes None


class TestGatingReport:
    """Test GatingReport dataclass."""

    def test_gating_report_creation(self):
        """Test creating GatingReport."""
        from core.quality.reporting.qa_report import GatingReport

        gating = GatingReport(
            status="PASS_WITH_WARNINGS",
            rules_evaluated=10,
            rules_passed=9,
            warning_rules=["rule_cloud_cover: Cloud cover 45%"],
        )

        assert gating.status == "PASS_WITH_WARNINGS"
        assert gating.rules_evaluated == 10
        assert len(gating.warning_rules) == 1

    def test_gating_report_to_dict(self):
        """Test GatingReport serialization."""
        from core.quality.reporting.qa_report import GatingReport

        gating = GatingReport(
            status="BLOCKED",
            rules_evaluated=5,
            rules_passed=3,
            blocking_rules=["mandatory_qa: Critical sanity failure"],
            degraded_mode=True,
            degraded_level=2,
        )

        d = gating.to_dict()
        assert d["status"] == "BLOCKED"
        assert d["blocking_rules"] == ["mandatory_qa: Critical sanity failure"]
        assert d["degraded_mode"]["active"] is True
        assert d["degraded_mode"]["level"] == 2


class TestFlagReport:
    """Test FlagReport dataclass."""

    def test_flag_report_creation(self):
        """Test creating FlagReport."""
        from core.quality.reporting.qa_report import FlagReport

        flag = FlagReport(
            flag_id="FLAG_CLOUD_COVER",
            flag_name="High Cloud Cover",
            severity="medium",
            affected_percent=35.0,
            reason="Cloud cover exceeds 30% threshold",
        )

        assert flag.flag_id == "FLAG_CLOUD_COVER"
        assert flag.severity == "medium"
        assert flag.affected_percent == 35.0

    def test_flag_report_to_dict(self):
        """Test FlagReport serialization."""
        from core.quality.reporting.qa_report import FlagReport

        flag = FlagReport(
            flag_id="FLAG_001",
            flag_name="Test Flag",
            severity="high",
            affected_percent=10.5,
            affected_area_km2=55.3,
            reason="Test reason",
        )

        d = flag.to_dict()
        assert d["flag_id"] == "FLAG_001"
        assert d["affected_percent"] == 10.5
        assert d["affected_area_km2"] == 55.3

    def test_flag_report_with_nan_values(self):
        """Test FlagReport with NaN/Inf values defaults to 0.0."""
        from core.quality.reporting.qa_report import FlagReport

        flag = FlagReport(
            flag_id="FLAG_001",
            flag_name="Test Flag",
            severity="high",
            affected_percent=np.nan,
            affected_area_km2=np.inf,
            reason="Test reason",
        )

        d = flag.to_dict()
        assert d["affected_percent"] == 0.0  # NaN defaults to 0.0
        assert d["affected_area_km2"] == 0.0  # Inf defaults to 0.0


class TestActionReport:
    """Test ActionReport dataclass."""

    def test_action_report_creation(self):
        """Test creating ActionReport."""
        from core.quality.reporting.qa_report import ActionReport

        action = ActionReport(
            action="mask_region",
            reason="High uncertainty region masked",
        )

        assert action.action == "mask_region"
        assert action.reason == "High uncertainty region masked"

    def test_action_report_to_dict(self):
        """Test ActionReport serialization."""
        from core.quality.reporting.qa_report import ActionReport

        action = ActionReport(
            action="interpolate_gap",
            reason="Temporal gap filled by interpolation",
            affected_area={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        )

        d = action.to_dict()
        assert d["action"] == "interpolate_gap"
        assert d["affected_area"]["type"] == "Polygon"
        assert "timestamp" in d


class TestExpertReviewReport:
    """Test ExpertReviewReport dataclass."""

    def test_expert_review_not_required(self):
        """Test ExpertReviewReport when not required."""
        from core.quality.reporting.qa_report import ExpertReviewReport

        review = ExpertReviewReport(required=False)

        assert review.required is False
        d = review.to_dict()
        assert d["required"] is False
        assert "reason" not in d

    def test_expert_review_required(self):
        """Test ExpertReviewReport when required."""
        from core.quality.reporting.qa_report import ExpertReviewReport

        review = ExpertReviewReport(
            required=True,
            reason="Low cross-validation agreement",
            priority="high",
            deadline=datetime(2024, 1, 15, tzinfo=timezone.utc),
            reviewer_assigned="expert@example.com",
            status="pending",
        )

        d = review.to_dict()
        assert d["required"] is True
        assert d["reason"] == "Low cross-validation agreement"
        assert d["priority"] == "high"
        assert "deadline" in d


class TestRecommendation:
    """Test Recommendation dataclass."""

    def test_recommendation_creation(self):
        """Test creating Recommendation."""
        from core.quality.reporting.qa_report import Recommendation

        rec = Recommendation(
            category="Data Quality",
            priority="high",
            recommendation="Apply additional atmospheric correction",
            rationale="Haze detected in optical imagery",
            impact="Improved surface reflectance accuracy",
        )

        assert rec.category == "Data Quality"
        assert rec.priority == "high"
        assert "atmospheric" in rec.recommendation

    def test_recommendation_to_dict(self):
        """Test Recommendation serialization."""
        from core.quality.reporting.qa_report import Recommendation

        rec = Recommendation(
            category="Methodology",
            priority="medium",
            recommendation="Use ensemble approach",
            rationale="Models disagree",
            impact="Better consensus",
        )

        d = rec.to_dict()
        assert d["category"] == "Methodology"
        assert d["priority"] == "medium"


# ============================================================================
# QA REPORT CLASS
# ============================================================================

class TestQAReport:
    """Test QAReport dataclass and output formats."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample QA report for testing."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary, CheckReport,
            CrossValidationReport, UncertaintySummaryReport, GatingReport,
            FlagReport, ExpertReviewReport, Recommendation,
        )

        metadata = ReportMetadata(
            event_id="evt_flood_2024",
            product_id="prod_flood_extent_001",
        )

        summary = QualitySummary(
            overall_status="PASS_WITH_WARNINGS",
            confidence_score=0.85,
            total_checks=5,
            passed_checks=4,
            warning_checks=1,
            failed_checks=0,
            key_findings=["Minor cloud cover issue"],
        )

        checks = [
            CheckReport(
                check_name="spatial_coherence",
                category="spatial",
                status="pass",
                metric_value=0.92,
                threshold=0.7,
                details="Good spatial coherence",
            ),
            CheckReport(
                check_name="cloud_cover",
                category="value",
                status="warning",
                metric_value=0.35,
                threshold=0.30,
                details="Cloud cover slightly above threshold",
            ),
        ]

        cross_validation = CrossValidationReport(
            methods_compared=["sar", "optical"],
            agreement_score=0.88,
            iou=0.82,
            consensus_method="weighted_average",
        )

        uncertainty = UncertaintySummaryReport(
            mean_uncertainty=0.12,
            max_uncertainty=0.35,
            std_uncertainty=0.05,
            high_uncertainty_percent=8.0,
        )

        gating = GatingReport(
            status="PASS_WITH_WARNINGS",
            rules_evaluated=8,
            rules_passed=8,
            warning_rules=["cloud_cover: 35%"],
        )

        flags = [
            FlagReport(
                flag_id="FLAG_001",
                flag_name="Elevated Cloud Cover",
                severity="medium",
                affected_percent=35.0,
                reason="Cloud cover above normal",
            ),
        ]

        expert_review = ExpertReviewReport(required=False)

        recommendations = [
            Recommendation(
                category="Data Quality",
                priority="medium",
                recommendation="Consider SAR-only analysis for cloudy regions",
            ),
        ]

        return QAReport(
            metadata=metadata,
            summary=summary,
            checks=checks,
            cross_validation=cross_validation,
            uncertainty=uncertainty,
            gating=gating,
            flags=flags,
            expert_review=expert_review,
            recommendations=recommendations,
        )

    def test_report_to_dict(self, sample_report):
        """Test QAReport to_dict method."""
        d = sample_report.to_dict()

        assert d["event_id"] == "evt_flood_2024"
        assert d["product_id"] == "prod_flood_extent_001"
        assert d["overall_status"] == "PASS_WITH_WARNINGS"
        assert d["confidence_score"] == 0.85
        assert "checks" in d
        assert len(d["checks"]) == 2

    def test_report_to_json(self, sample_report):
        """Test QAReport to_json method."""
        json_str = sample_report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "evt_flood_2024"
        assert "checks" in parsed

    def test_report_to_markdown(self, sample_report):
        """Test QAReport to_markdown method."""
        md = sample_report.to_markdown()

        assert "# QA Report:" in md
        assert "evt_flood_2024" in md
        assert "PASS_WITH_WARNINGS" in md
        assert "Key Findings" in md
        assert "Quality Checks" in md
        assert "Cross-Validation" in md
        assert "Uncertainty" in md

    def test_report_to_html(self, sample_report):
        """Test QAReport to_html method."""
        html = sample_report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "QA Report" in html
        assert "evt_flood_2024" in html
        assert "PASS_WITH_WARNINGS" in html
        # Check for CSS styles
        assert "<style>" in html
        assert "summary-card" in html

    def test_report_to_text(self, sample_report):
        """Test QAReport to_text method."""
        text = sample_report.to_text()

        assert "QA REPORT" in text
        assert "evt_flood_2024" in text
        assert "PASS_WITH_WARNINGS" in text
        assert "Total Checks:" in text

    def test_report_save_json(self, sample_report, tmp_path):
        """Test saving report as JSON."""
        output_path = tmp_path / "report.json"
        sample_report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        parsed = json.loads(content)
        assert parsed["event_id"] == "evt_flood_2024"

    def test_report_save_html(self, sample_report, tmp_path):
        """Test saving report as HTML."""
        output_path = tmp_path / "report.html"
        sample_report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_report_save_markdown(self, sample_report, tmp_path):
        """Test saving report as Markdown."""
        output_path = tmp_path / "report.md"
        sample_report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# QA Report:" in content

    def test_report_save_text(self, sample_report, tmp_path):
        """Test saving report as plain text."""
        output_path = tmp_path / "report.txt"
        sample_report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "QA REPORT" in content

    def test_report_creates_parent_dirs(self, sample_report, tmp_path):
        """Test that save creates parent directories."""
        output_path = tmp_path / "nested" / "dirs" / "report.json"
        sample_report.save(output_path)

        assert output_path.exists()


# ============================================================================
# QA REPORT GENERATOR
# ============================================================================

class TestQAReportGenerator:
    """Test QAReportGenerator class."""

    def test_generator_initialization(self):
        """Test QAReportGenerator initialization."""
        from core.quality.reporting.qa_report import QAReportGenerator, ReportConfig

        generator = QAReportGenerator()
        assert generator.config is not None

        config = ReportConfig()
        generator = QAReportGenerator(config)
        assert generator.config == config

    def test_generate_minimal_report(self):
        """Test generating a minimal report."""
        from core.quality.reporting.qa_report import QAReportGenerator

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_test",
            product_id="prod_test",
        )

        assert report.metadata.event_id == "evt_test"
        assert report.metadata.product_id == "prod_test"
        assert report.summary.overall_status == "PASS"

    def test_generate_with_sanity_result(self):
        """Test generating report with sanity results."""
        from core.quality.reporting.qa_report import QAReportGenerator
        from core.quality.sanity import SanitySuite

        # Create a mock sanity result
        data = np.random.rand(50, 50) * 0.8 + 0.1
        suite = SanitySuite()
        sanity_result = suite.check(data)

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_sanity",
            product_id="prod_sanity",
            sanity_result=sanity_result,
        )

        assert report.summary.total_checks >= 1

    def test_generate_with_all_components(self):
        """Test generating report with all quality components."""
        from core.quality.reporting.qa_report import QAReportGenerator

        # Mock results as simple objects with expected attributes
        class MockSanityResult:
            passes_sanity = True
            overall_score = 0.9
            spatial = None
            values = None
            temporal = None
            artifacts = None

        class MockValidationResult:
            agreement_score = 0.85

        class MockUncertaintyResult:
            mean_uncertainty = 0.15
            max_uncertainty = 0.4
            std_uncertainty = 0.08

        class MockGatingDecision:
            status = "PASS_WITH_WARNINGS"
            confidence_score = 0.88
            rules_evaluated = 5
            rules_passed = 5
            degraded_mode = False
            degraded_level = 0

        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_full",
            product_id="prod_full",
            sanity_result=MockSanityResult(),
            validation_result=MockValidationResult(),
            uncertainty_result=MockUncertaintyResult(),
            gating_decision=MockGatingDecision(),
        )

        assert report.metadata.event_id == "evt_full"
        assert report.summary.overall_status == "PASS_WITH_WARNINGS"
        assert report.uncertainty is not None
        assert report.uncertainty.mean_uncertainty == 0.15

    def test_severity_to_status_mapping(self):
        """Test severity to status conversion."""
        from core.quality.reporting.qa_report import QAReportGenerator

        generator = QAReportGenerator()

        assert generator._severity_to_status(None) == "pass"

        # Mock severity enum
        class MockSeverity:
            def __init__(self, value):
                self.value = value

        assert generator._severity_to_status(MockSeverity("critical")) == "hard_fail"
        assert generator._severity_to_status(MockSeverity("high")) == "soft_fail"
        assert generator._severity_to_status(MockSeverity("medium")) == "warning"
        assert generator._severity_to_status(MockSeverity("low")) == "pass"
        assert generator._severity_to_status(MockSeverity("info")) == "pass"


class TestConvenienceFunction:
    """Test generate_qa_report convenience function."""

    def test_generate_qa_report_basic(self):
        """Test basic usage of generate_qa_report."""
        from core.quality.reporting.qa_report import generate_qa_report, ReportFormat

        results = {}
        report = generate_qa_report(
            event_id="evt_conv",
            product_id="prod_conv",
            results=results,
        )

        assert report.metadata.event_id == "evt_conv"

    def test_generate_qa_report_with_format(self):
        """Test generate_qa_report with specified format."""
        from core.quality.reporting.qa_report import generate_qa_report, ReportFormat, ReportLevel

        results = {}
        report = generate_qa_report(
            event_id="evt_html",
            product_id="prod_html",
            results=results,
            format=ReportFormat.HTML,
            level=ReportLevel.DETAILED,
        )

        assert report.metadata.format == ReportFormat.HTML
        assert report.metadata.level == ReportLevel.DETAILED


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestReportEdgeCases:
    """Test edge cases in report generation."""

    def test_empty_checks_list(self):
        """Test report with empty checks list."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary
        )

        report = QAReport(
            metadata=ReportMetadata(event_id="evt", product_id="prod"),
            summary=QualitySummary(overall_status="PASS", confidence_score=1.0),
            checks=[],
        )

        d = report.to_dict()
        assert d["checks"] == []

    def test_report_with_special_characters(self):
        """Test report with special characters in text."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary, CheckReport
        )

        report = QAReport(
            metadata=ReportMetadata(
                event_id="evt_special<>&\"'",
                product_id="prod_special",
            ),
            summary=QualitySummary(
                overall_status="PASS",
                confidence_score=1.0,
                key_findings=["Finding with <html> & \"quotes\""],
            ),
            checks=[
                CheckReport(
                    check_name="check_with_special",
                    category="test",
                    status="pass",
                    details="Details with <script>alert('xss')</script>",
                ),
            ],
        )

        # Should not crash on any format
        _ = report.to_json()
        _ = report.to_html()
        _ = report.to_markdown()
        _ = report.to_text()

    def test_report_with_very_long_lists(self):
        """Test report with many checks."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary, CheckReport,
            QAReportGenerator, ReportConfig, ReportLevel
        )

        # Create many checks
        checks = [
            CheckReport(
                check_name=f"check_{i}",
                category="test",
                status="pass",
                details=f"Check {i} passed",
            )
            for i in range(100)
        ]

        report = QAReport(
            metadata=ReportMetadata(event_id="evt", product_id="prod"),
            summary=QualitySummary(
                overall_status="PASS",
                confidence_score=1.0,
                total_checks=100,
                passed_checks=100,
            ),
            checks=checks,
        )

        d = report.to_dict()
        assert len(d["checks"]) == 100

    def test_report_detailed_level_includes_metadata(self):
        """Test that detailed level includes extra metadata."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary, ReportLevel
        )

        report = QAReport(
            metadata=ReportMetadata(
                event_id="evt",
                product_id="prod",
                level=ReportLevel.DETAILED,
            ),
            summary=QualitySummary(overall_status="PASS", confidence_score=1.0),
        )

        d = report.to_dict()
        assert "_metadata" in d
        assert "_summary" in d

    def test_degraded_mode_confidence_modifier(self):
        """Test degraded mode confidence modifier calculation."""
        from core.quality.reporting.qa_report import (
            QAReport, ReportMetadata, QualitySummary, GatingReport
        )

        report = QAReport(
            metadata=ReportMetadata(event_id="evt", product_id="prod"),
            summary=QualitySummary(overall_status="PASS", confidence_score=1.0),
            gating=GatingReport(
                status="PASS",
                degraded_mode=True,
                degraded_level=2,
            ),
        )

        d = report.to_dict()
        assert d["degraded_mode"]["active"] is True
        assert d["degraded_mode"]["level"] == 2
        # 0.8 ** 2 = 0.64
        assert abs(d["degraded_mode"]["confidence_modifier"] - 0.64) < 0.01


class TestReportIntegration:
    """Integration tests with actual quality control modules."""

    def test_integration_with_sanity_suite(self):
        """Test integration with SanitySuite results."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.reporting.qa_report import QAReportGenerator

        # Create test data
        data = np.random.rand(100, 100) * 0.5 + 0.25

        # Run sanity suite
        config = SanitySuiteConfig(run_temporal=False)
        suite = SanitySuite(config)
        sanity_result = suite.check(data)

        # Generate report
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_integration",
            product_id="prod_integration",
            sanity_result=sanity_result,
        )

        assert report.summary.total_checks >= 1
        json_output = report.to_json()
        assert "evt_integration" in json_output

    def test_integration_with_gating(self):
        """Test integration with gating decision results."""
        from core.quality.actions.gating import (
            QualityGate, GatingThresholds, QCCheck, CheckStatus, CheckCategory
        )
        from core.quality.reporting.qa_report import QAReportGenerator

        # Create mock QC checks
        checks = [
            QCCheck(
                check_name="sanity_spatial",
                category=CheckCategory.SPATIAL,
                status=CheckStatus.PASS,
                metric_value=0.92,
                details="Spatial coherence acceptable",
            ),
            QCCheck(
                check_name="sanity_values",
                category=CheckCategory.VALUE,
                status=CheckStatus.PASS,
                metric_value=0.88,
                details="Values within expected range",
            ),
        ]

        # Run gating
        thresholds = GatingThresholds()
        gate = QualityGate(thresholds)
        decision = gate.evaluate(checks, event_id="evt_gating", product_id="prod_gating")

        # Generate report
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_gating",
            product_id="prod_gating",
            gating_decision=decision,
        )

        assert report.gating is not None

    def test_full_pipeline_integration(self):
        """Test full pipeline with all quality components."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.actions.gating import (
            QualityGate, GatingThresholds, QCCheck, CheckStatus, CheckCategory
        )
        from core.quality.actions.flagging import QualityFlagger, StandardFlag
        from core.quality.reporting.qa_report import QAReportGenerator

        # Create test data
        data = np.random.rand(50, 50)

        # Run sanity checks
        sanity_suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        sanity_result = sanity_suite.check(data)

        # Create QC checks from sanity result for gating
        checks = [
            QCCheck(
                check_name="sanity_overall",
                category=CheckCategory.SPATIAL,  # Use SPATIAL as closest category
                status=CheckStatus.PASS if sanity_result.passes_sanity else CheckStatus.SOFT_FAIL,
                metric_value=sanity_result.overall_score,
                details=sanity_result.summary,
            ),
        ]

        # Run gating
        gate = QualityGate(GatingThresholds())
        gating_decision = gate.evaluate(
            checks,
            event_id="evt_full_pipeline",
            product_id="prod_full_pipeline",
        )

        # Run flagging - apply a flag manually
        flagger = QualityFlagger()
        applied_flag = flagger.apply_standard_flag(
            product_id="prod_full_pipeline",
            flag=StandardFlag.SINGLE_SENSOR_MODE,
            reason="Test flag application",
        )
        applied_flags = [applied_flag]

        # Generate comprehensive report
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_full_pipeline",
            product_id="prod_full_pipeline",
            sanity_result=sanity_result,
            gating_decision=gating_decision,
            flags=applied_flags,
        )

        # Verify report completeness
        assert report.metadata.event_id == "evt_full_pipeline"
        assert report.summary is not None
        assert report.gating is not None
        assert len(report.flags) > 0

        # Export all formats
        _ = report.to_json()
        _ = report.to_html()
        _ = report.to_markdown()
        _ = report.to_text()


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_report_config_defaults(self):
        """Test ReportConfig default values."""
        from core.quality.reporting.qa_report import ReportConfig, ReportFormat, ReportLevel

        config = ReportConfig()
        assert config.format == ReportFormat.JSON
        assert config.level == ReportLevel.STANDARD
        assert config.include_recommendations is True
        assert config.max_checks_displayed == 50

    def test_report_config_custom(self):
        """Test ReportConfig with custom values."""
        from core.quality.reporting.qa_report import (
            ReportConfig, ReportFormat, ReportLevel, ReportSection
        )

        config = ReportConfig(
            format=ReportFormat.HTML,
            level=ReportLevel.DEBUG,
            sections=[ReportSection.OVERVIEW, ReportSection.SANITY_CHECKS],
            include_recommendations=False,
            max_checks_displayed=100,
        )

        assert config.format == ReportFormat.HTML
        assert config.level == ReportLevel.DEBUG
        assert len(config.sections) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
