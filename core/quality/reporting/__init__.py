"""
Quality Reporting Module.

Provides comprehensive reporting capabilities for quality control results:
- **QA Reports**: Schema-compliant reports in JSON, HTML, Markdown formats
- **Diagnostics**: Detailed diagnostic outputs for debugging and analysis

QA reports aggregate results from all quality control systems (sanity checks,
cross-validation, uncertainty quantification, gating) into human-readable
and machine-readable formats suitable for operational use and documentation.

Diagnostics provide detailed technical information for debugging, performance
analysis, and comparison between runs.

Example Usage:
    from core.quality.reporting import (
        # QA Report generation
        QAReportGenerator,
        ReportFormat,
        ReportLevel,
        generate_qa_report,

        # Diagnostics
        DiagnosticGenerator,
        DiagnosticLevel,
        generate_diagnostics,
    )

    # Generate QA report from quality results
    generator = QAReportGenerator()
    report = generator.generate(
        event_id="evt_flood_miami_2024",
        product_id="prod_flood_extent_001",
        sanity_result=sanity_result,
        validation_result=validation_result,
        uncertainty_result=uncertainty_result,
        gating_decision=gating_decision,
    )

    # Save in multiple formats
    report.save("qa_report.json")
    report.save("qa_report.html")
    report.save("qa_report.md")

    # Generate diagnostics for debugging
    diagnostics = generate_diagnostics({
        "sanity": sanity_result,
        "validation": validation_result,
        "uncertainty": uncertainty_result,
    })
    print(diagnostics.get_summary())
    diagnostics.export_json("diagnostics.json")
"""

# QA Report exports
from core.quality.reporting.qa_report import (
    # Enums
    ReportFormat,
    ReportLevel,
    ReportSection,
    # Config dataclasses
    ReportConfig,
    ReportMetadata,
    # Summary dataclasses
    QualitySummary,
    # Detail dataclasses
    CheckReport,
    CrossValidationReport,
    UncertaintySummaryReport,
    GatingReport,
    FlagReport,
    ActionReport,
    ExpertReviewReport,
    Recommendation,
    # Main report class
    QAReport,
    # Generator class
    QAReportGenerator,
    # Convenience function
    generate_qa_report,
)

# Diagnostics exports - conditional import (module in progress)
try:
    from core.quality.reporting.diagnostics import (
        # Enums
        DiagnosticLevel,
        DiagnosticType,
        MetricCategory,
        # Config dataclasses
        DiagnosticConfig,
        # Data dataclasses
        DiagnosticMetric,
        IssueDetail,
        SpatialDiagnostic,
        TemporalDiagnostic,
        PerformanceMetric,
        ComparisonResult,
        # Main diagnostics class
        Diagnostics,
        # Generator class
        DiagnosticGenerator,
        # Convenience function
        generate_diagnostics,
    )
    _DIAGNOSTICS_AVAILABLE = True
except ImportError:
    _DIAGNOSTICS_AVAILABLE = False
    # Placeholder values to avoid NameError in __all__
    DiagnosticLevel = None
    DiagnosticType = None
    MetricCategory = None
    DiagnosticConfig = None
    DiagnosticMetric = None
    IssueDetail = None
    SpatialDiagnostic = None
    TemporalDiagnostic = None
    PerformanceMetric = None
    ComparisonResult = None
    Diagnostics = None
    DiagnosticGenerator = None
    generate_diagnostics = None

__all__ = [
    # QA Report - Enums
    "ReportFormat",
    "ReportLevel",
    "ReportSection",
    # QA Report - Config
    "ReportConfig",
    "ReportMetadata",
    # QA Report - Summary
    "QualitySummary",
    # QA Report - Detail
    "CheckReport",
    "CrossValidationReport",
    "UncertaintySummaryReport",
    "GatingReport",
    "FlagReport",
    "ActionReport",
    "ExpertReviewReport",
    "Recommendation",
    # QA Report - Main
    "QAReport",
    # QA Report - Generator
    "QAReportGenerator",
    # QA Report - Function
    "generate_qa_report",
    # Diagnostics - Enums
    "DiagnosticLevel",
    "DiagnosticType",
    "MetricCategory",
    # Diagnostics - Config
    "DiagnosticConfig",
    # Diagnostics - Data
    "DiagnosticMetric",
    "IssueDetail",
    "SpatialDiagnostic",
    "TemporalDiagnostic",
    "PerformanceMetric",
    "ComparisonResult",
    # Diagnostics - Main
    "Diagnostics",
    # Diagnostics - Generator
    "DiagnosticGenerator",
    # Diagnostics - Function
    "generate_diagnostics",
]
