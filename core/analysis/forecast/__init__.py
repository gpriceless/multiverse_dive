"""
Forecast Integration Module.

Provides comprehensive forecast data handling and analysis for the
geospatial event intelligence platform. Integrates weather forecasts
with observational data for enhanced hazard assessment.

Submodules:
- ingestion: Forecast data loading and normalization
- validation: Forecast verification against observations
- scenarios: Scenario analysis and uncertainty quantification
- projection: Forecast-to-impact translation

Example:
    from core.analysis.forecast import (
        ForecastIngester,
        ForecastVariable,
        ingest_forecast,
        validate_forecast,
        analyze_scenarios,
        project_impacts,
    )

    # Ingest forecast data
    result = ingest_forecast(
        providers=[ForecastProvider.GFS],
        variables=[ForecastVariable.PRECIPITATION, ForecastVariable.WIND_SPEED],
        spatial_bounds=(-82, 24, -79, 27),
    )

    if result.success:
        forecast = result.dataset

        # Analyze scenarios
        scenario_result = analyze_scenarios(
            forecast,
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [25, 50, 100]},
        )

        # Project impacts
        impact_result = project_impacts(
            forecast,
            hazards=[HazardType.FLOOD, HazardType.STORM],
        )
"""

# Ingestion
from core.analysis.forecast.ingestion import (
    ForecastProvider,
    ForecastVariable,
    ForecastType,
    ForecastMetadata,
    ForecastTimestep,
    ForecastDataset,
    ForecastIngestionConfig,
    ForecastIngestionResult,
    ForecastIngester,
    ingest_forecast,
)

# Validation
from core.analysis.forecast.validation import (
    MetricType,
    ValidationLevel,
    ObservationData,
    ContingencyTable,
    VariableMetrics,
    LeadTimePerformance,
    EnsembleMetrics,
    SpatialVerification,
    ValidationConfig,
    ValidationResult,
    ForecastValidator,
    validate_forecast,
)

# Scenarios
from core.analysis.forecast.scenarios import (
    ScenarioType,
    ImpactDirection,
    ScenarioDefinition,
    ScenarioData,
    ThresholdExceedance,
    ScenarioComparison,
    SensitivityResult,
    ScenarioAnalysisConfig,
    ScenarioAnalysisResult,
    ScenarioAnalyzer,
    analyze_scenarios,
)

# Projection
from core.analysis.forecast.projection import (
    HazardType,
    ImpactSeverity,
    ImpactCategory,
    ImpactThreshold,
    ImpactModel,
    ImpactTimestep,
    ImpactProjection,
    CompoundImpact,
    ProjectionConfig,
    ProjectionResult,
    ImpactProjector,
    project_impacts,
    get_impact_model,
    create_custom_impact_model,
)

__all__ = [
    # Ingestion
    "ForecastProvider",
    "ForecastVariable",
    "ForecastType",
    "ForecastMetadata",
    "ForecastTimestep",
    "ForecastDataset",
    "ForecastIngestionConfig",
    "ForecastIngestionResult",
    "ForecastIngester",
    "ingest_forecast",
    # Validation
    "MetricType",
    "ValidationLevel",
    "ObservationData",
    "ContingencyTable",
    "VariableMetrics",
    "LeadTimePerformance",
    "EnsembleMetrics",
    "SpatialVerification",
    "ValidationConfig",
    "ValidationResult",
    "ForecastValidator",
    "validate_forecast",
    # Scenarios
    "ScenarioType",
    "ImpactDirection",
    "ScenarioDefinition",
    "ScenarioData",
    "ThresholdExceedance",
    "ScenarioComparison",
    "SensitivityResult",
    "ScenarioAnalysisConfig",
    "ScenarioAnalysisResult",
    "ScenarioAnalyzer",
    "analyze_scenarios",
    # Projection
    "HazardType",
    "ImpactSeverity",
    "ImpactCategory",
    "ImpactThreshold",
    "ImpactModel",
    "ImpactTimestep",
    "ImpactProjection",
    "CompoundImpact",
    "ProjectionConfig",
    "ProjectionResult",
    "ImpactProjector",
    "project_impacts",
    "get_impact_model",
    "create_custom_impact_model",
]
