"""
QA Report Generation for Quality Control.

Provides comprehensive QA report generation in multiple formats:
- JSON: Machine-readable report conforming to quality.schema.json
- HTML: Human-readable report with visualizations
- Markdown: Lightweight text report for documentation
- PDF: Formal report for distribution (via HTML conversion)

Reports aggregate results from:
- Sanity checks (spatial, value, temporal, artifact)
- Cross-validation (model comparison, sensor comparison)
- Uncertainty quantification
- Gating decisions
- Expert review requirements

Key Features:
- Schema-compliant JSON output
- Configurable detail levels (summary, standard, detailed)
- Multi-language support for human-readable reports
- Template-based HTML generation
- Provenance tracking for reproducibility

Example:
    from core.quality.reporting import (
        QAReportGenerator,
        ReportFormat,
        ReportLevel,
        generate_qa_report,
    )

    # Create report from quality results
    generator = QAReportGenerator()
    report = generator.generate(
        sanity_result=sanity_result,
        validation_result=validation_result,
        uncertainty_result=uncertainty_result,
        gating_decision=gating_decision,
        format=ReportFormat.JSON,
        level=ReportLevel.STANDARD,
    )

    # Save report
    report.save("qa_report.json")

    # Quick generation
    json_report = generate_qa_report(results, format=ReportFormat.JSON)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


class ReportFormat(Enum):
    """Output format for QA reports."""
    JSON = "json"           # Schema-compliant JSON
    HTML = "html"           # Human-readable HTML
    MARKDOWN = "markdown"   # Lightweight markdown
    TEXT = "text"           # Plain text summary


class ReportLevel(Enum):
    """Detail level for reports."""
    SUMMARY = "summary"     # High-level pass/fail only
    STANDARD = "standard"   # Standard detail for operational use
    DETAILED = "detailed"   # Full detail for debugging
    DEBUG = "debug"         # Maximum verbosity including internal state


class ReportSection(Enum):
    """Sections that can be included in reports."""
    OVERVIEW = "overview"
    SANITY_CHECKS = "sanity_checks"
    CROSS_VALIDATION = "cross_validation"
    UNCERTAINTY = "uncertainty"
    GATING = "gating"
    FLAGS = "flags"
    ACTIONS = "actions"
    EXPERT_REVIEW = "expert_review"
    PROVENANCE = "provenance"
    RECOMMENDATIONS = "recommendations"


@dataclass
class ReportMetadata:
    """
    Metadata for a QA report.

    Attributes:
        event_id: Event identifier
        product_id: Product being validated
        generated_at: Report generation timestamp
        generator_version: Version of report generator
        format: Report format
        level: Report detail level
        sections: Sections included
    """
    event_id: str
    product_id: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generator_version: str = "1.0.0"
    format: ReportFormat = ReportFormat.JSON
    level: ReportLevel = ReportLevel.STANDARD
    sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "product_id": self.product_id,
            "generated_at": self.generated_at.isoformat(),
            "generator_version": self.generator_version,
            "format": self.format.value,
            "level": self.level.value,
            "sections": [s.value for s in self.sections],
        }


@dataclass
class QualitySummary:
    """
    Summary of quality assessment.

    Attributes:
        overall_status: Pass/fail/review status
        confidence_score: Overall confidence (0-1)
        total_checks: Number of checks performed
        passed_checks: Number of checks passed
        warning_checks: Number of checks with warnings
        failed_checks: Number of checks failed
        issues_by_category: Issue counts by category
        key_findings: Most important findings
    """
    overall_status: str  # PASS, PASS_WITH_WARNINGS, REVIEW_REQUIRED, BLOCKED
    confidence_score: float
    total_checks: int = 0
    passed_checks: int = 0
    warning_checks: int = 0
    failed_checks: int = 0
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status,
            "confidence_score": _safe_round(self.confidence_score, 4) or 0.0,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "warning_checks": self.warning_checks,
            "failed_checks": self.failed_checks,
            "issues_by_category": self.issues_by_category,
            "key_findings": self.key_findings,
        }


@dataclass
class CheckReport:
    """
    Report for a single QC check.

    Attributes:
        check_name: Name of the check
        category: Check category
        status: Pass/warning/fail status
        metric_value: Numeric metric if applicable
        threshold: Threshold used
        details: Human-readable explanation
        spatial_extent: Affected area (GeoJSON)
        recommendations: Suggested actions
    """
    check_name: str
    category: str
    status: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: str = ""
    spatial_extent: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "check_name": self.check_name,
            "category": self.category,
            "status": self.status,
            "details": self.details,
        }
        if self.metric_value is not None:
            result["metric_value"] = (
                round(self.metric_value, 6)
                if not np.isnan(self.metric_value) and not np.isinf(self.metric_value)
                else None
            )
        if self.threshold is not None:
            result["threshold"] = (
                round(self.threshold, 6)
                if not np.isnan(self.threshold) and not np.isinf(self.threshold)
                else None
            )
        if self.spatial_extent is not None:
            result["spatial_extent"] = self.spatial_extent
        if self.recommendations:
            result["recommendations"] = self.recommendations
        return result


@dataclass
class CrossValidationReport:
    """
    Report section for cross-validation results.

    Attributes:
        methods_compared: List of methods/models compared
        agreement_score: Overall agreement (0-1)
        iou: Intersection over Union
        kappa: Cohen's Kappa coefficient
        disagreement_area_km2: Area of disagreement
        disagreement_regions: List of disagreement region details
        consensus_method: Method used for consensus
        confidence_by_method: Confidence per method
    """
    methods_compared: List[str] = field(default_factory=list)
    agreement_score: float = 0.0
    iou: Optional[float] = None
    kappa: Optional[float] = None
    disagreement_area_km2: Optional[float] = None
    disagreement_regions: List[Dict[str, Any]] = field(default_factory=list)
    consensus_method: str = ""
    confidence_by_method: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "methods_compared": self.methods_compared,
            "agreement_metrics": {
                "agreement_score": _safe_round(self.agreement_score, 4) or 0.0,
            },
        }
        if self.iou is not None:
            safe_iou = _safe_round(self.iou, 4)
            if safe_iou is not None:
                result["agreement_metrics"]["iou"] = safe_iou
        if self.kappa is not None:
            safe_kappa = _safe_round(self.kappa, 4)
            if safe_kappa is not None:
                result["agreement_metrics"]["kappa"] = safe_kappa
        if self.disagreement_area_km2 is not None:
            safe_area = _safe_round(self.disagreement_area_km2, 2)
            if safe_area is not None:
                result["agreement_metrics"]["disagreement_area_km2"] = safe_area
        if self.disagreement_regions:
            result["disagreement_regions"] = self.disagreement_regions
        if self.consensus_method:
            result["consensus_method_used"] = self.consensus_method
        if self.confidence_by_method:
            result["confidence_by_method"] = {
                k: _safe_round(v, 4) or 0.0 for k, v in self.confidence_by_method.items()
            }
        return result


@dataclass
class UncertaintySummaryReport:
    """
    Report section for uncertainty quantification.

    Attributes:
        mean_uncertainty: Mean uncertainty value
        max_uncertainty: Maximum uncertainty value
        std_uncertainty: Standard deviation of uncertainty
        high_uncertainty_percent: Percentage above threshold
        high_uncertainty_area_km2: Area above threshold
        hotspot_count: Number of uncertainty hotspots
        hotspot_regions: Details of hotspot regions
        calibration_score: Uncertainty calibration score
    """
    mean_uncertainty: float = 0.0
    max_uncertainty: float = 0.0
    std_uncertainty: float = 0.0
    high_uncertainty_percent: float = 0.0
    high_uncertainty_area_km2: float = 0.0
    hotspot_count: int = 0
    hotspot_regions: List[Dict[str, Any]] = field(default_factory=list)
    calibration_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mean_uncertainty": _safe_round(self.mean_uncertainty, 4) or 0.0,
            "max_uncertainty": _safe_round(self.max_uncertainty, 4) or 0.0,
            "std_uncertainty": _safe_round(self.std_uncertainty, 4) or 0.0,
            "high_uncertainty_percent": _safe_round(self.high_uncertainty_percent, 2) or 0.0,
            "high_uncertainty_area_km2": _safe_round(self.high_uncertainty_area_km2, 2) or 0.0,
            "hotspot_count": self.hotspot_count,
        }
        if self.hotspot_regions:
            result["hotspot_regions"] = self.hotspot_regions
        if self.calibration_score is not None:
            result["calibration_score"] = _safe_round(self.calibration_score, 4)
        return result


@dataclass
class GatingReport:
    """
    Report section for gating decisions.

    Attributes:
        status: Gating outcome
        rules_evaluated: Number of rules evaluated
        rules_passed: Number of rules passed
        blocking_rules: Rules that caused blocking
        warning_rules: Rules that generated warnings
        thresholds_used: Thresholds applied
        degraded_mode: Whether degraded mode was active
        degraded_level: Degraded mode level if active
    """
    status: str = "PASS"
    rules_evaluated: int = 0
    rules_passed: int = 0
    blocking_rules: List[str] = field(default_factory=list)
    warning_rules: List[str] = field(default_factory=list)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    degraded_mode: bool = False
    degraded_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status,
            "rules_evaluated": self.rules_evaluated,
            "rules_passed": self.rules_passed,
        }
        if self.blocking_rules:
            result["blocking_rules"] = self.blocking_rules
        if self.warning_rules:
            result["warning_rules"] = self.warning_rules
        if self.thresholds_used:
            result["thresholds_used"] = {
                k: _safe_round(v, 4) if isinstance(v, float) else v
                for k, v in self.thresholds_used.items()
            }
        result["degraded_mode"] = {
            "active": self.degraded_mode,
            "level": self.degraded_level,
        }
        return result


@dataclass
class FlagReport:
    """
    Report for applied quality flags.

    Attributes:
        flag_id: Flag identifier
        flag_name: Human-readable name
        severity: Flag severity level
        affected_percent: Percentage of product affected
        affected_area_km2: Area affected
        reason: Why flag was applied
    """
    flag_id: str
    flag_name: str
    severity: str
    affected_percent: float = 0.0
    affected_area_km2: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_id": self.flag_id,
            "flag_name": self.flag_name,
            "severity": self.severity,
            "affected_percent": _safe_round(self.affected_percent, 2) or 0.0,
            "affected_area_km2": _safe_round(self.affected_area_km2, 2) or 0.0,
            "reason": self.reason,
        }


@dataclass
class ActionReport:
    """
    Report for actions taken.

    Attributes:
        action: Action type
        reason: Why action was taken
        affected_area: GeoJSON of affected area
        timestamp: When action was taken
    """
    action: str
    reason: str
    affected_area: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "action": self.action,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.affected_area is not None:
            result["affected_area"] = self.affected_area
        return result


@dataclass
class ExpertReviewReport:
    """
    Report for expert review requirements.

    Attributes:
        required: Whether review is required
        reason: Why review is required
        priority: Review priority
        deadline: Review deadline
        reviewer_assigned: Assigned reviewer
        status: Review status
        review_notes: Notes from reviewer
    """
    required: bool = False
    reason: str = ""
    priority: str = "normal"
    deadline: Optional[datetime] = None
    reviewer_assigned: Optional[str] = None
    status: str = "not_required"
    review_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "required": self.required,
            "status": self.status,
        }
        if self.required:
            result["reason"] = self.reason
            result["priority"] = self.priority
            if self.deadline:
                result["deadline"] = self.deadline.isoformat()
            if self.reviewer_assigned:
                result["reviewer_assigned"] = self.reviewer_assigned
            if self.review_notes:
                result["review_notes"] = self.review_notes
        return result


@dataclass
class Recommendation:
    """
    A recommendation for quality improvement.

    Attributes:
        category: Recommendation category
        priority: Priority (high, medium, low)
        recommendation: The recommendation text
        rationale: Why this is recommended
        impact: Expected impact of implementing
    """
    category: str
    priority: str
    recommendation: str
    rationale: str = ""
    impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "priority": self.priority,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "impact": self.impact,
        }


@dataclass
class QAReport:
    """
    Complete QA report.

    Attributes:
        metadata: Report metadata
        summary: Quality summary
        checks: Individual check reports
        cross_validation: Cross-validation results
        uncertainty: Uncertainty summary
        gating: Gating decision
        flags: Applied flags
        actions: Actions taken
        expert_review: Expert review requirements
        recommendations: Improvement recommendations
    """
    metadata: ReportMetadata
    summary: QualitySummary
    checks: List[CheckReport] = field(default_factory=list)
    cross_validation: Optional[CrossValidationReport] = None
    uncertainty: Optional[UncertaintySummaryReport] = None
    gating: Optional[GatingReport] = None
    flags: List[FlagReport] = field(default_factory=list)
    actions: List[ActionReport] = field(default_factory=list)
    expert_review: Optional[ExpertReviewReport] = None
    recommendations: List[Recommendation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary conforming to quality.schema.json.

        Returns:
            Schema-compliant dictionary
        """
        result = {
            "event_id": self.metadata.event_id,
            "product_id": self.metadata.product_id,
            "timestamp": self.metadata.generated_at.isoformat(),
            "overall_status": self.summary.overall_status,
            "confidence_score": self.summary.confidence_score,
            "checks": [c.to_dict() for c in self.checks],
        }

        if self.cross_validation:
            result["cross_validation"] = self.cross_validation.to_dict()

        if self.uncertainty:
            result["uncertainty_summary"] = self.uncertainty.to_dict()

        if self.flags:
            result["quality_flags"] = [f.to_dict() for f in self.flags]

        if self.actions:
            result["actions_taken"] = [a.to_dict() for a in self.actions]

        if self.expert_review:
            result["expert_review"] = self.expert_review.to_dict()

        if self.gating:
            result["degraded_mode"] = {
                "active": self.gating.degraded_mode,
                "level": self.gating.degraded_level,
                "confidence_modifier": 0.8 ** self.gating.degraded_level if self.gating.degraded_mode else 1.0,
            }

        # Add detailed sections based on level
        if self.metadata.level in (ReportLevel.DETAILED, ReportLevel.DEBUG):
            result["_metadata"] = self.metadata.to_dict()
            result["_summary"] = self.summary.to_dict()
            if self.gating:
                result["_gating_details"] = self.gating.to_dict()
            if self.recommendations:
                result["_recommendations"] = [r.to_dict() for r in self.recommendations]

        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """
        Convert to markdown format.

        Returns:
            Markdown string
        """
        lines = [
            f"# QA Report: {self.metadata.product_id}",
            "",
            f"**Event ID:** {self.metadata.event_id}",
            f"**Generated:** {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Status:** {self.summary.overall_status}",
            f"**Confidence:** {self.summary.confidence_score:.2%}",
            "",
            "## Summary",
            "",
            f"- **Total Checks:** {self.summary.total_checks}",
            f"- **Passed:** {self.summary.passed_checks}",
            f"- **Warnings:** {self.summary.warning_checks}",
            f"- **Failed:** {self.summary.failed_checks}",
            "",
        ]

        if self.summary.key_findings:
            lines.append("### Key Findings")
            lines.append("")
            for finding in self.summary.key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        if self.checks:
            lines.append("## Quality Checks")
            lines.append("")
            lines.append("| Check | Category | Status | Details |")
            lines.append("|-------|----------|--------|---------|")
            for check in self.checks:
                status_icon = {"pass": "PASS", "warning": "WARNING", "soft_fail": "SOFT_FAIL", "hard_fail": "HARD_FAIL"}.get(check.status, check.status)
                lines.append(f"| {check.check_name} | {check.category} | {status_icon} | {check.details[:50]}... |")
            lines.append("")

        if self.cross_validation:
            lines.append("## Cross-Validation")
            lines.append("")
            lines.append(f"- **Methods Compared:** {', '.join(self.cross_validation.methods_compared)}")
            lines.append(f"- **Agreement Score:** {self.cross_validation.agreement_score:.2%}")
            if self.cross_validation.iou is not None:
                lines.append(f"- **IoU:** {self.cross_validation.iou:.4f}")
            if self.cross_validation.kappa is not None:
                lines.append(f"- **Kappa:** {self.cross_validation.kappa:.4f}")
            lines.append("")

        if self.uncertainty:
            lines.append("## Uncertainty")
            lines.append("")
            lines.append(f"- **Mean:** {self.uncertainty.mean_uncertainty:.4f}")
            lines.append(f"- **Max:** {self.uncertainty.max_uncertainty:.4f}")
            lines.append(f"- **High Uncertainty Area:** {self.uncertainty.high_uncertainty_percent:.1f}%")
            lines.append("")

        if self.flags:
            lines.append("## Quality Flags")
            lines.append("")
            for flag in self.flags:
                lines.append(f"- **{flag.flag_name}** ({flag.severity}): {flag.reason}")
            lines.append("")

        if self.expert_review and self.expert_review.required:
            lines.append("## Expert Review Required")
            lines.append("")
            lines.append(f"- **Reason:** {self.expert_review.reason}")
            lines.append(f"- **Priority:** {self.expert_review.priority}")
            if self.expert_review.deadline:
                lines.append(f"- **Deadline:** {self.expert_review.deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"### {rec.category} ({rec.priority} priority)")
                lines.append("")
                lines.append(rec.recommendation)
                if rec.rationale:
                    lines.append(f"\n*Rationale:* {rec.rationale}")
                lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """
        Convert to HTML format.

        Returns:
            HTML string
        """
        status_colors = {
            "PASS": "#28a745",
            "PASS_WITH_WARNINGS": "#ffc107",
            "REVIEW_REQUIRED": "#fd7e14",
            "BLOCKED": "#dc3545",
        }
        status_color = status_colors.get(self.summary.overall_status, "#6c757d")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>QA Report: {self.metadata.product_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #495057; margin-top: 30px; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 4px; color: white; font-weight: bold; }}
        .metadata {{ color: #6c757d; font-size: 0.9em; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; }}
        .summary-card .number {{ font-size: 2em; font-weight: bold; color: #333; }}
        .summary-card .label {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .pass {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .soft_fail {{ color: #fd7e14; }}
        .hard_fail {{ color: #dc3545; }}
        .finding {{ background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .recommendation {{ background: #e7f5ff; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .high-priority {{ border-left: 4px solid #dc3545; }}
        .medium-priority {{ border-left: 4px solid #ffc107; }}
        .low-priority {{ border-left: 4px solid #28a745; }}
    </style>
</head>
<body>
<div class="container">
    <h1>QA Report</h1>
    <p class="metadata">
        <strong>Product:</strong> {self.metadata.product_id}<br>
        <strong>Event:</strong> {self.metadata.event_id}<br>
        <strong>Generated:</strong> {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
    </p>

    <p>
        <span class="status-badge" style="background-color: {status_color}">
            {self.summary.overall_status}
        </span>
        &nbsp;&nbsp;
        <strong>Confidence:</strong> {self.summary.confidence_score:.1%}
    </p>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="number">{self.summary.total_checks}</div>
            <div class="label">Total Checks</div>
        </div>
        <div class="summary-card">
            <div class="number pass">{self.summary.passed_checks}</div>
            <div class="label">Passed</div>
        </div>
        <div class="summary-card">
            <div class="number warning">{self.summary.warning_checks}</div>
            <div class="label">Warnings</div>
        </div>
        <div class="summary-card">
            <div class="number hard_fail">{self.summary.failed_checks}</div>
            <div class="label">Failed</div>
        </div>
    </div>
"""

        # Key findings
        if self.summary.key_findings:
            html += "<h2>Key Findings</h2>\n"
            for finding in self.summary.key_findings:
                html += f'<div class="finding">{finding}</div>\n'

        # Checks table
        if self.checks:
            html += """<h2>Quality Checks</h2>
<table>
<tr><th>Check</th><th>Category</th><th>Status</th><th>Metric</th><th>Details</th></tr>
"""
            for check in self.checks:
                metric_str = f"{check.metric_value:.4f}" if check.metric_value is not None else "-"
                threshold_str = f" / {check.threshold:.4f}" if check.threshold is not None else ""
                html += f'<tr><td>{check.check_name}</td><td>{check.category}</td><td class="{check.status}">{check.status.upper()}</td><td>{metric_str}{threshold_str}</td><td>{check.details}</td></tr>\n'
            html += "</table>\n"

        # Cross-validation
        if self.cross_validation:
            html += f"""<h2>Cross-Validation</h2>
<p><strong>Methods:</strong> {', '.join(self.cross_validation.methods_compared)}</p>
<p><strong>Agreement Score:</strong> {self.cross_validation.agreement_score:.2%}</p>
"""
            if self.cross_validation.iou is not None:
                html += f"<p><strong>IoU:</strong> {self.cross_validation.iou:.4f}</p>\n"
            if self.cross_validation.kappa is not None:
                html += f"<p><strong>Cohen's Kappa:</strong> {self.cross_validation.kappa:.4f}</p>\n"

        # Uncertainty
        if self.uncertainty:
            html += f"""<h2>Uncertainty Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean Uncertainty</td><td>{self.uncertainty.mean_uncertainty:.4f}</td></tr>
<tr><td>Max Uncertainty</td><td>{self.uncertainty.max_uncertainty:.4f}</td></tr>
<tr><td>High Uncertainty Area</td><td>{self.uncertainty.high_uncertainty_percent:.1f}%</td></tr>
<tr><td>Hotspot Count</td><td>{self.uncertainty.hotspot_count}</td></tr>
</table>
"""

        # Flags
        if self.flags:
            html += "<h2>Quality Flags</h2>\n<ul>\n"
            for flag in self.flags:
                html += f"<li><strong>{flag.flag_name}</strong> ({flag.severity}): {flag.reason}</li>\n"
            html += "</ul>\n"

        # Expert review
        if self.expert_review and self.expert_review.required:
            html += f"""<h2>Expert Review Required</h2>
<p><strong>Reason:</strong> {self.expert_review.reason}</p>
<p><strong>Priority:</strong> {self.expert_review.priority}</p>
"""
            if self.expert_review.deadline:
                html += f"<p><strong>Deadline:</strong> {self.expert_review.deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>\n"

        # Recommendations
        if self.recommendations:
            html += "<h2>Recommendations</h2>\n"
            for rec in self.recommendations:
                priority_class = f"{rec.priority}-priority"
                html += f'<div class="recommendation {priority_class}"><strong>{rec.category}</strong><br>{rec.recommendation}'
                if rec.rationale:
                    html += f"<br><em>Rationale: {rec.rationale}</em>"
                html += "</div>\n"

        html += """
</div>
</body>
</html>"""
        return html

    def to_text(self) -> str:
        """
        Convert to plain text format.

        Returns:
            Plain text string
        """
        lines = [
            "=" * 60,
            f"QA REPORT: {self.metadata.product_id}",
            "=" * 60,
            "",
            f"Event ID:    {self.metadata.event_id}",
            f"Generated:   {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Status:      {self.summary.overall_status}",
            f"Confidence:  {self.summary.confidence_score:.1%}",
            "",
            "-" * 60,
            "SUMMARY",
            "-" * 60,
            f"Total Checks: {self.summary.total_checks}",
            f"Passed:       {self.summary.passed_checks}",
            f"Warnings:     {self.summary.warning_checks}",
            f"Failed:       {self.summary.failed_checks}",
            "",
        ]

        if self.summary.key_findings:
            lines.append("KEY FINDINGS:")
            for finding in self.summary.key_findings:
                lines.append(f"  * {finding}")
            lines.append("")

        if self.expert_review and self.expert_review.required:
            lines.append("-" * 60)
            lines.append("EXPERT REVIEW REQUIRED")
            lines.append("-" * 60)
            lines.append(f"Reason:   {self.expert_review.reason}")
            lines.append(f"Priority: {self.expert_review.priority}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Union[str, Path], format: Optional[ReportFormat] = None) -> None:
        """
        Save report to file.

        Args:
            path: Output file path
            format: Override format (default: infer from extension)
        """
        path = Path(path)
        if format is None:
            ext_map = {
                ".json": ReportFormat.JSON,
                ".html": ReportFormat.HTML,
                ".md": ReportFormat.MARKDOWN,
                ".txt": ReportFormat.TEXT,
            }
            format = ext_map.get(path.suffix.lower(), ReportFormat.JSON)

        content = {
            ReportFormat.JSON: self.to_json,
            ReportFormat.HTML: self.to_html,
            ReportFormat.MARKDOWN: self.to_markdown,
            ReportFormat.TEXT: self.to_text,
        }[format]()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Saved QA report to {path}")


@dataclass
class ReportConfig:
    """
    Configuration for report generation.

    Attributes:
        format: Output format
        level: Detail level
        sections: Sections to include
        include_recommendations: Generate recommendations
        include_spatial_details: Include spatial extent info
        max_checks_displayed: Limit checks in summary reports
    """
    format: ReportFormat = ReportFormat.JSON
    level: ReportLevel = ReportLevel.STANDARD
    sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))
    include_recommendations: bool = True
    include_spatial_details: bool = True
    max_checks_displayed: int = 50


class QAReportGenerator:
    """
    Generator for QA reports.

    Aggregates results from multiple quality control systems and
    produces comprehensive reports in various formats.

    Example:
        generator = QAReportGenerator()
        report = generator.generate(
            event_id="evt_flood_miami_2024",
            product_id="prod_flood_extent_001",
            sanity_result=sanity_suite_result,
            validation_result=cross_validation_result,
            uncertainty_result=uncertainty_quantification_result,
            gating_decision=quality_gate_decision,
        )
        report.save("qa_report.json")
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()

    def generate(
        self,
        event_id: str,
        product_id: str,
        sanity_result: Optional[Any] = None,
        validation_result: Optional[Any] = None,
        uncertainty_result: Optional[Any] = None,
        gating_decision: Optional[Any] = None,
        flags: Optional[List[Any]] = None,
        actions: Optional[List[Any]] = None,
        expert_review: Optional[Any] = None,
    ) -> QAReport:
        """
        Generate a QA report from quality control results.

        Args:
            event_id: Event identifier
            product_id: Product identifier
            sanity_result: Result from SanitySuite
            validation_result: Result from cross-validation
            uncertainty_result: Result from uncertainty quantification
            gating_decision: Result from QualityGate
            flags: Applied quality flags
            actions: Actions taken
            expert_review: Expert review request

        Returns:
            Complete QAReport
        """
        logger.info(f"Generating QA report for {product_id}")

        # Create metadata
        metadata = ReportMetadata(
            event_id=event_id,
            product_id=product_id,
            format=self.config.format,
            level=self.config.level,
            sections=self.config.sections,
        )

        # Build checks list
        checks: List[CheckReport] = []
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        issues_by_category: Dict[str, int] = {}

        # Process sanity checks
        if sanity_result is not None:
            sanity_checks = self._extract_sanity_checks(sanity_result)
            checks.extend(sanity_checks)
            for check in sanity_checks:
                total_checks += 1
                cat = check.category
                if check.status == "pass":
                    passed_checks += 1
                elif check.status == "warning":
                    warning_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
                else:
                    failed_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        # Process validation results
        cross_validation_report = None
        if validation_result is not None:
            cross_validation_report, validation_checks = self._extract_validation(validation_result)
            checks.extend(validation_checks)
            for check in validation_checks:
                total_checks += 1
                cat = check.category
                if check.status == "pass":
                    passed_checks += 1
                elif check.status == "warning":
                    warning_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
                else:
                    failed_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        # Process uncertainty
        uncertainty_report = None
        if uncertainty_result is not None:
            uncertainty_report, uncertainty_checks = self._extract_uncertainty(uncertainty_result)
            checks.extend(uncertainty_checks)
            for check in uncertainty_checks:
                total_checks += 1
                cat = check.category
                if check.status == "pass":
                    passed_checks += 1
                elif check.status == "warning":
                    warning_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
                else:
                    failed_checks += 1
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        # Process gating decision
        gating_report = None
        overall_status = "PASS"
        confidence_score = 1.0

        if gating_decision is not None:
            gating_report, overall_status, confidence_score = self._extract_gating(gating_decision)
        else:
            # Infer status from checks
            if failed_checks > 0:
                overall_status = "BLOCKED" if failed_checks > 2 else "REVIEW_REQUIRED"
            elif warning_checks > 0:
                overall_status = "PASS_WITH_WARNINGS"
            confidence_score = max(0.0, 1.0 - (failed_checks * 0.2) - (warning_checks * 0.05))

        # Generate key findings
        key_findings = self._generate_key_findings(
            sanity_result, validation_result, uncertainty_result, gating_decision
        )

        # Create summary
        summary = QualitySummary(
            overall_status=overall_status,
            confidence_score=confidence_score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            warning_checks=warning_checks,
            failed_checks=failed_checks,
            issues_by_category=issues_by_category,
            key_findings=key_findings,
        )

        # Process flags
        flag_reports = []
        if flags:
            flag_reports = self._extract_flags(flags)

        # Process actions
        action_reports = []
        if actions:
            action_reports = self._extract_actions(actions)

        # Process expert review
        expert_review_report = None
        if expert_review:
            expert_review_report = self._extract_expert_review(expert_review)

        # Generate recommendations
        recommendations = []
        if self.config.include_recommendations:
            recommendations = self._generate_recommendations(
                sanity_result, validation_result, uncertainty_result, gating_decision
            )

        # Limit checks if needed
        if len(checks) > self.config.max_checks_displayed and self.config.level == ReportLevel.SUMMARY:
            # Keep most important checks
            failed = [c for c in checks if c.status in ("hard_fail", "soft_fail")]
            warnings = [c for c in checks if c.status == "warning"]
            passed = [c for c in checks if c.status == "pass"]
            checks = (
                failed[:20] +
                warnings[:min(20, self.config.max_checks_displayed - len(failed))] +
                passed[:max(0, self.config.max_checks_displayed - len(failed) - len(warnings))]
            )

        return QAReport(
            metadata=metadata,
            summary=summary,
            checks=checks,
            cross_validation=cross_validation_report,
            uncertainty=uncertainty_report,
            gating=gating_report,
            flags=flag_reports,
            actions=action_reports,
            expert_review=expert_review_report,
            recommendations=recommendations,
        )

    def _extract_sanity_checks(self, result: Any) -> List[CheckReport]:
        """Extract check reports from sanity suite result."""
        checks = []

        # Handle SanitySuiteResult
        if hasattr(result, 'spatial') and result.spatial is not None:
            spatial = result.spatial
            for issue in getattr(spatial, 'issues', []):
                checks.append(CheckReport(
                    check_name=getattr(issue, 'check_type', 'spatial_check').value if hasattr(getattr(issue, 'check_type', None), 'value') else str(getattr(issue, 'check_type', 'spatial_check')),
                    category="spatial",
                    status=self._severity_to_status(getattr(issue, 'severity', None)),
                    metric_value=getattr(issue, 'metric_value', None),
                    threshold=getattr(issue, 'threshold', None),
                    details=getattr(issue, 'description', ''),
                ))

        if hasattr(result, 'values') and result.values is not None:
            values = result.values
            for issue in getattr(values, 'issues', []):
                checks.append(CheckReport(
                    check_name=getattr(issue, 'check_type', 'value_check').value if hasattr(getattr(issue, 'check_type', None), 'value') else str(getattr(issue, 'check_type', 'value_check')),
                    category="value",
                    status=self._severity_to_status(getattr(issue, 'severity', None)),
                    metric_value=getattr(issue, 'metric_value', None),
                    threshold=getattr(issue, 'threshold', None),
                    details=getattr(issue, 'description', ''),
                ))

        if hasattr(result, 'temporal') and result.temporal is not None:
            temporal = result.temporal
            for issue in getattr(temporal, 'issues', []):
                checks.append(CheckReport(
                    check_name=getattr(issue, 'check_type', 'temporal_check').value if hasattr(getattr(issue, 'check_type', None), 'value') else str(getattr(issue, 'check_type', 'temporal_check')),
                    category="temporal",
                    status=self._severity_to_status(getattr(issue, 'severity', None)),
                    metric_value=getattr(issue, 'metric_value', None),
                    threshold=getattr(issue, 'threshold', None),
                    details=getattr(issue, 'description', ''),
                ))

        if hasattr(result, 'artifacts') and result.artifacts is not None:
            artifacts = result.artifacts
            for artifact in getattr(artifacts, 'artifacts', []):
                checks.append(CheckReport(
                    check_name=f"artifact_{getattr(artifact, 'artifact_type', 'unknown').value if hasattr(getattr(artifact, 'artifact_type', None), 'value') else str(getattr(artifact, 'artifact_type', 'unknown'))}",
                    category="artifact",
                    status=self._severity_to_status(getattr(artifact, 'severity', None)),
                    metric_value=getattr(artifact, 'confidence', None),
                    details=getattr(artifact, 'description', ''),
                ))

        # If no issues found but we have a result, add summary checks
        if not checks and hasattr(result, 'passes_sanity'):
            if result.passes_sanity:
                checks.append(CheckReport(
                    check_name="sanity_suite",
                    category="value",
                    status="pass",
                    details="All sanity checks passed",
                ))
            else:
                checks.append(CheckReport(
                    check_name="sanity_suite",
                    category="value",
                    status="soft_fail",
                    details=getattr(result, 'summary', 'Sanity checks failed'),
                ))

        return checks

    def _severity_to_status(self, severity: Any) -> str:
        """Convert severity enum to status string."""
        if severity is None:
            return "pass"
        severity_val = severity.value if hasattr(severity, 'value') else str(severity)
        mapping = {
            "critical": "hard_fail",
            "high": "soft_fail",
            "medium": "warning",
            "low": "pass",
            "info": "pass",
            "informational": "pass",
        }
        return mapping.get(severity_val.lower(), "warning")

    def _extract_validation(self, result: Any) -> tuple[Optional[CrossValidationReport], List[CheckReport]]:
        """Extract cross-validation report and checks."""
        checks = []
        cv_report = None

        # Handle CrossModelResult
        if hasattr(result, 'pairwise_comparisons'):
            methods = []
            agreement_scores = []
            ious = []
            kappas = []

            for comp in result.pairwise_comparisons:
                methods.extend([comp.model_a, comp.model_b] if hasattr(comp, 'model_a') else [])
                if hasattr(comp, 'agreement_score'):
                    agreement_scores.append(comp.agreement_score)
                if hasattr(comp, 'iou'):
                    ious.append(comp.iou)
                if hasattr(comp, 'kappa'):
                    kappas.append(comp.kappa)

                # Add check for each comparison
                status = "pass" if getattr(comp, 'agreement_score', 0) > 0.7 else "warning" if getattr(comp, 'agreement_score', 0) > 0.5 else "soft_fail"
                checks.append(CheckReport(
                    check_name=f"cross_model_{getattr(comp, 'model_a', 'a')}_vs_{getattr(comp, 'model_b', 'b')}",
                    category="cross_validation",
                    status=status,
                    metric_value=getattr(comp, 'agreement_score', None),
                    threshold=0.7,
                    details=f"Agreement between models",
                ))

            cv_report = CrossValidationReport(
                methods_compared=list(set(methods)),
                agreement_score=sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0,
                iou=sum(ious) / len(ious) if ious else None,
                kappa=sum(kappas) / len(kappas) if kappas else None,
                consensus_method=getattr(result, 'consensus_method', ''),
            )

        # Handle ConsensusResult
        elif hasattr(result, 'quality'):
            agreement = getattr(result.quality, 'agreement_score', 0.0) if hasattr(result, 'quality') else 0.0
            status = "pass" if agreement > 0.7 else "warning" if agreement > 0.5 else "soft_fail"
            checks.append(CheckReport(
                check_name="consensus_agreement",
                category="cross_validation",
                status=status,
                metric_value=agreement,
                threshold=0.7,
                details="Consensus agreement across sources",
            ))

            sources = getattr(result, 'sources_used', [])
            cv_report = CrossValidationReport(
                methods_compared=[str(s) for s in sources],
                agreement_score=agreement,
                consensus_method=getattr(result, 'strategy', '').value if hasattr(getattr(result, 'strategy', ''), 'value') else str(getattr(result, 'strategy', '')),
            )

        return cv_report, checks

    def _extract_uncertainty(self, result: Any) -> tuple[Optional[UncertaintySummaryReport], List[CheckReport]]:
        """Extract uncertainty report and checks."""
        checks = []
        unc_report = None

        # Handle UncertaintyMetrics or similar
        mean_unc = getattr(result, 'mean_uncertainty', None) or getattr(result, 'mean', None)
        max_unc = getattr(result, 'max_uncertainty', None) or getattr(result, 'max', None)
        std_unc = getattr(result, 'std_uncertainty', None) or getattr(result, 'std', None)

        if mean_unc is not None:
            status = "pass" if mean_unc < 0.2 else "warning" if mean_unc < 0.3 else "soft_fail"
            checks.append(CheckReport(
                check_name="uncertainty_level",
                category="uncertainty",
                status=status,
                metric_value=mean_unc,
                threshold=0.3,
                details=f"Mean uncertainty level",
            ))

            unc_report = UncertaintySummaryReport(
                mean_uncertainty=mean_unc,
                max_uncertainty=max_unc or 0.0,
                std_uncertainty=std_unc or 0.0,
                high_uncertainty_percent=getattr(result, 'high_uncertainty_percent', 0.0),
                high_uncertainty_area_km2=getattr(result, 'high_uncertainty_area_km2', 0.0),
                hotspot_count=len(getattr(result, 'hotspots', [])),
                calibration_score=getattr(result, 'calibration_score', None),
            )

        return unc_report, checks

    def _extract_gating(self, decision: Any) -> tuple[Optional[GatingReport], str, float]:
        """Extract gating report, status, and confidence."""
        status = "PASS"
        confidence = 1.0

        # Handle GatingDecision
        if hasattr(decision, 'status'):
            status_val = decision.status
            status = status_val.value if hasattr(status_val, 'value') else str(status_val)

        if hasattr(decision, 'confidence_score'):
            confidence = decision.confidence_score

        blocking_rules = []
        warning_rules = []

        if hasattr(decision, 'rule_results'):
            for rule_id, (passed, reason) in decision.rule_results.items():
                if not passed:
                    if 'block' in reason.lower() or 'mandatory' in reason.lower():
                        blocking_rules.append(f"{rule_id}: {reason}")
                    else:
                        warning_rules.append(f"{rule_id}: {reason}")

        gating_report = GatingReport(
            status=status,
            rules_evaluated=getattr(decision, 'rules_evaluated', 0),
            rules_passed=getattr(decision, 'rules_passed', 0),
            blocking_rules=blocking_rules,
            warning_rules=warning_rules,
            degraded_mode=getattr(decision, 'degraded_mode', False),
            degraded_level=getattr(decision, 'degraded_level', 0),
        )

        return gating_report, status, confidence

    def _extract_flags(self, flags: List[Any]) -> List[FlagReport]:
        """Extract flag reports from applied flags."""
        reports = []
        for flag in flags:
            reports.append(FlagReport(
                flag_id=getattr(flag, 'flag_id', str(flag)),
                flag_name=getattr(flag, 'flag_name', '') or getattr(flag, 'name', ''),
                severity=getattr(flag, 'severity', 'medium').value if hasattr(getattr(flag, 'severity', None), 'value') else str(getattr(flag, 'severity', 'medium')),
                affected_percent=getattr(flag, 'affected_percent', 0.0),
                affected_area_km2=getattr(flag, 'affected_area_km2', 0.0),
                reason=getattr(flag, 'reason', ''),
            ))
        return reports

    def _extract_actions(self, actions: List[Any]) -> List[ActionReport]:
        """Extract action reports."""
        reports = []
        for action in actions:
            reports.append(ActionReport(
                action=getattr(action, 'action', str(action)),
                reason=getattr(action, 'reason', ''),
                affected_area=getattr(action, 'affected_area', None),
            ))
        return reports

    def _extract_expert_review(self, review: Any) -> ExpertReviewReport:
        """Extract expert review report."""
        return ExpertReviewReport(
            required=getattr(review, 'required', True),
            reason=getattr(review, 'reason', ''),
            priority=getattr(review, 'priority', 'normal').value if hasattr(getattr(review, 'priority', None), 'value') else str(getattr(review, 'priority', 'normal')),
            deadline=getattr(review, 'deadline', None),
            reviewer_assigned=getattr(review, 'reviewer', None) or getattr(review, 'reviewer_assigned', None),
            status=getattr(review, 'status', 'pending').value if hasattr(getattr(review, 'status', None), 'value') else str(getattr(review, 'status', 'pending')),
        )

    def _generate_key_findings(
        self,
        sanity_result: Optional[Any],
        validation_result: Optional[Any],
        uncertainty_result: Optional[Any],
        gating_decision: Optional[Any],
    ) -> List[str]:
        """Generate key findings from results."""
        findings = []

        # Sanity findings
        if sanity_result is not None:
            if hasattr(sanity_result, 'critical_issues') and sanity_result.critical_issues > 0:
                findings.append(f"{sanity_result.critical_issues} critical sanity issues detected")
            if hasattr(sanity_result, 'artifacts') and sanity_result.artifacts:
                artifact_count = len(getattr(sanity_result.artifacts, 'artifacts', []))
                if artifact_count > 0:
                    findings.append(f"{artifact_count} artifacts detected in output")

        # Validation findings
        if validation_result is not None:
            if hasattr(validation_result, 'agreement_score'):
                score = validation_result.agreement_score
                if score < 0.5:
                    findings.append(f"Low cross-validation agreement ({score:.1%})")
            if hasattr(validation_result, 'disagreement_regions'):
                regions = validation_result.disagreement_regions
                if len(regions) > 5:
                    findings.append(f"Significant disagreement in {len(regions)} regions")

        # Uncertainty findings
        if uncertainty_result is not None:
            mean_unc = getattr(uncertainty_result, 'mean_uncertainty', None) or getattr(uncertainty_result, 'mean', None)
            if mean_unc is not None and mean_unc > 0.3:
                findings.append(f"High mean uncertainty ({mean_unc:.2%})")

        # Gating findings
        if gating_decision is not None:
            if hasattr(gating_decision, 'status'):
                status = gating_decision.status
                status_val = status.value if hasattr(status, 'value') else str(status)
                if status_val == "BLOCKED":
                    findings.append("Output blocked by quality gate")
                elif status_val == "REVIEW_REQUIRED":
                    findings.append("Expert review required before release")

        return findings

    def _generate_recommendations(
        self,
        sanity_result: Optional[Any],
        validation_result: Optional[Any],
        uncertainty_result: Optional[Any],
        gating_decision: Optional[Any],
    ) -> List[Recommendation]:
        """Generate recommendations for quality improvement."""
        recommendations = []

        # Sanity-based recommendations
        if sanity_result is not None:
            if hasattr(sanity_result, 'artifacts') and sanity_result.artifacts:
                artifacts = getattr(sanity_result.artifacts, 'artifacts', [])
                if artifacts:
                    recommendations.append(Recommendation(
                        category="Data Quality",
                        priority="high",
                        recommendation="Review detected artifacts and consider reprocessing with artifact correction",
                        rationale="Artifacts can propagate to downstream analysis",
                        impact="Improved spatial accuracy",
                    ))

        # Validation-based recommendations
        if validation_result is not None:
            agreement = getattr(validation_result, 'agreement_score', 1.0)
            if agreement < 0.7:
                recommendations.append(Recommendation(
                    category="Methodology",
                    priority="medium",
                    recommendation="Consider ensemble approach to reconcile model disagreements",
                    rationale="Multiple models show significant disagreement",
                    impact="More robust consensus output",
                ))

        # Uncertainty-based recommendations
        if uncertainty_result is not None:
            mean_unc = getattr(uncertainty_result, 'mean_uncertainty', None) or getattr(uncertainty_result, 'mean', 0.0)
            if mean_unc > 0.3:
                recommendations.append(Recommendation(
                    category="Uncertainty",
                    priority="high",
                    recommendation="Flag high-uncertainty regions in output product",
                    rationale="Mean uncertainty exceeds acceptable threshold",
                    impact="Users can make informed decisions",
                ))

        return recommendations


def generate_qa_report(
    event_id: str,
    product_id: str,
    results: Dict[str, Any],
    format: ReportFormat = ReportFormat.JSON,
    level: ReportLevel = ReportLevel.STANDARD,
) -> QAReport:
    """
    Convenience function to generate a QA report.

    Args:
        event_id: Event identifier
        product_id: Product identifier
        results: Dictionary with keys: sanity, validation, uncertainty, gating, flags, actions, expert_review
        format: Output format
        level: Detail level

    Returns:
        Generated QAReport
    """
    config = ReportConfig(format=format, level=level)
    generator = QAReportGenerator(config)

    return generator.generate(
        event_id=event_id,
        product_id=product_id,
        sanity_result=results.get("sanity"),
        validation_result=results.get("validation"),
        uncertainty_result=results.get("uncertainty"),
        gating_decision=results.get("gating"),
        flags=results.get("flags"),
        actions=results.get("actions"),
        expert_review=results.get("expert_review"),
    )
