"""
Impact Projection Module.

Projects forecast data into potential hazard impacts using configurable
impact models. Translates meteorological forecasts into actionable
hazard assessments (flood extent, fire risk, wind damage potential).

Key Capabilities:
- Forecast-to-impact translation models
- Hazard-specific impact projections (flood, wildfire, storm)
- Configurable impact thresholds and severity levels
- Temporal impact evolution tracking
- Uncertainty propagation from forecast to impact
- Multi-hazard compound impact analysis
- Impact summary and visualization support
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from core.analysis.forecast.ingestion import (
    ForecastDataset,
    ForecastTimestep,
    ForecastVariable,
)
from core.analysis.forecast.scenarios import ScenarioData

logger = logging.getLogger(__name__)


class HazardType(Enum):
    """Types of hazards for impact projection."""

    FLOOD = "flood"
    WILDFIRE = "wildfire"
    STORM = "storm"
    DROUGHT = "drought"
    HEATWAVE = "heatwave"
    COLDWAVE = "coldwave"
    LANDSLIDE = "landslide"


class ImpactSeverity(Enum):
    """Severity levels for impacts."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"

    @property
    def level(self) -> int:
        """Numeric severity level (0-6)."""
        levels = {
            "none": 0,
            "minor": 1,
            "moderate": 2,
            "significant": 3,
            "severe": 4,
            "extreme": 5,
            "catastrophic": 6,
        }
        return levels[self.value]


class ImpactCategory(Enum):
    """Categories of impacts."""

    INFRASTRUCTURE = "infrastructure"
    POPULATION = "population"
    AGRICULTURE = "agriculture"
    ENVIRONMENT = "environment"
    ECONOMIC = "economic"
    TRANSPORTATION = "transportation"
    UTILITIES = "utilities"
    HEALTH = "health"


@dataclass
class ImpactThreshold:
    """
    Threshold definition for impact severity determination.

    Attributes:
        variable: Forecast variable to threshold
        min_value: Minimum value for this threshold (inclusive)
        max_value: Maximum value for this threshold (exclusive)
        severity: Resulting severity level
        duration_hours: Required duration to trigger (optional)
        description: Human-readable description
    """

    variable: str
    min_value: float
    max_value: float
    severity: ImpactSeverity
    duration_hours: Optional[float] = None
    description: str = ""

    def matches(self, value: float) -> bool:
        """Check if value falls within threshold range."""
        return self.min_value <= value < self.max_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "severity": self.severity.value,
            "duration_hours": self.duration_hours,
            "description": self.description,
        }


@dataclass
class ImpactModel:
    """
    Model for projecting forecast values to impact severity.

    Attributes:
        hazard_type: Type of hazard this model applies to
        name: Model name
        description: Model description
        variables: Required forecast variables
        thresholds: Impact thresholds
        aggregation_method: How to aggregate across time/space
        uncertainty_factor: Uncertainty scaling factor
    """

    hazard_type: HazardType
    name: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    thresholds: List[ImpactThreshold] = field(default_factory=list)
    aggregation_method: str = "max"  # max, sum, mean
    uncertainty_factor: float = 0.2

    def get_severity(self, value: float, variable: str) -> ImpactSeverity:
        """Get severity level for a value."""
        for thresh in self.thresholds:
            if thresh.variable == variable and thresh.matches(value):
                return thresh.severity
        return ImpactSeverity.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hazard_type": self.hazard_type.value,
            "name": self.name,
            "description": self.description,
            "variables": self.variables,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "aggregation_method": self.aggregation_method,
            "uncertainty_factor": self.uncertainty_factor,
        }


@dataclass
class ImpactTimestep:
    """
    Impact assessment at a single timestep.

    Attributes:
        valid_time: Time of assessment
        severity_field: Spatial severity field (enum values as ints)
        confidence_field: Spatial confidence (0-1)
        affected_area_km2: Total affected area
        peak_severity: Maximum severity in domain
        driver_values: Values of driving variables
    """

    valid_time: datetime
    severity_field: np.ndarray
    confidence_field: np.ndarray
    affected_area_km2: float = 0.0
    peak_severity: ImpactSeverity = ImpactSeverity.NONE
    driver_values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timezone."""
        if self.valid_time.tzinfo is None:
            self.valid_time = self.valid_time.replace(tzinfo=timezone.utc)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across domain."""
        return float(np.nanmean(self.confidence_field))

    def get_severity_distribution(self) -> Dict[ImpactSeverity, float]:
        """Get distribution of severity levels (fraction of area)."""
        total = np.sum(~np.isnan(self.severity_field))
        if total == 0:
            return {}

        dist = {}
        for sev in ImpactSeverity:
            count = np.sum(self.severity_field == sev.level)
            dist[sev] = float(count / total)
        return dist

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid_time": self.valid_time.isoformat(),
            "affected_area_km2": self.affected_area_km2,
            "peak_severity": self.peak_severity.value,
            "mean_confidence": self.mean_confidence,
            "driver_values": self.driver_values,
            "severity_distribution": {k.value: v for k, v in self.get_severity_distribution().items()},
        }


@dataclass
class ImpactProjection:
    """
    Complete impact projection result.

    Attributes:
        hazard_type: Type of hazard
        model: Impact model used
        timesteps: Impact assessment per timestep
        grid_lat: Latitude grid
        grid_lon: Longitude grid
        max_severity: Maximum severity across all timesteps
        total_affected_area_km2: Maximum affected area
        peak_time: Time of peak impact
        summary: Summary statistics
    """

    hazard_type: HazardType
    model: ImpactModel
    timesteps: List[ImpactTimestep]
    grid_lat: np.ndarray
    grid_lon: np.ndarray
    max_severity: ImpactSeverity = ImpactSeverity.NONE
    total_affected_area_km2: float = 0.0
    peak_time: Optional[datetime] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute aggregate statistics."""
        if self.timesteps:
            self.max_severity = max(self.timesteps, key=lambda ts: ts.peak_severity.level).peak_severity
            self.total_affected_area_km2 = max(ts.affected_area_km2 for ts in self.timesteps)

            # Find peak time
            peak_ts = max(self.timesteps, key=lambda ts: ts.peak_severity.level)
            self.peak_time = peak_ts.valid_time

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        """Time range of projection."""
        if not self.timesteps:
            return (datetime.now(timezone.utc), datetime.now(timezone.utc))
        return (self.timesteps[0].valid_time, self.timesteps[-1].valid_time)

    def get_timestep(self, valid_time: datetime) -> Optional[ImpactTimestep]:
        """Get impact at specific time."""
        for ts in self.timesteps:
            if ts.valid_time == valid_time:
                return ts
        return None

    def get_cumulative_severity(self) -> np.ndarray:
        """Get maximum severity at each point across all timesteps."""
        if not self.timesteps:
            return np.array([])

        stacked = np.stack([ts.severity_field for ts in self.timesteps], axis=0)
        return np.nanmax(stacked, axis=0)

    def get_impact_evolution(self) -> Dict[str, List]:
        """Get time series of impact metrics."""
        return {
            "times": [ts.valid_time for ts in self.timesteps],
            "peak_severity": [ts.peak_severity.level for ts in self.timesteps],
            "affected_area_km2": [ts.affected_area_km2 for ts in self.timesteps],
            "mean_confidence": [ts.mean_confidence for ts in self.timesteps],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hazard_type": self.hazard_type.value,
            "model": self.model.to_dict(),
            "max_severity": self.max_severity.value,
            "total_affected_area_km2": self.total_affected_area_km2,
            "peak_time": self.peak_time.isoformat() if self.peak_time else None,
            "time_range": [t.isoformat() for t in self.time_range],
            "n_timesteps": len(self.timesteps),
            "summary": self.summary,
        }


@dataclass
class CompoundImpact:
    """
    Assessment of compound/cascading impacts from multiple hazards.

    Attributes:
        hazards: List of hazard types involved
        projections: Individual projections
        compound_severity_field: Combined severity field
        interaction_effects: Interaction between hazards
        compound_probability: Probability of compound occurrence
    """

    hazards: List[HazardType]
    projections: Dict[HazardType, ImpactProjection]
    compound_severity_field: Optional[np.ndarray] = None
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    compound_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hazards": [h.value for h in self.hazards],
            "projections": {h.value: p.to_dict() for h, p in self.projections.items()},
            "interaction_effects": self.interaction_effects,
            "compound_probability": self.compound_probability,
        }


@dataclass
class ProjectionConfig:
    """
    Configuration for impact projection.

    Attributes:
        hazard_types: Hazards to project
        models: Custom impact models (overrides defaults)
        cell_area_km2: Grid cell area for area calculations
        min_severity: Minimum severity to include in results
        propagate_uncertainty: Whether to propagate forecast uncertainty
        compute_compound: Whether to assess compound impacts
    """

    hazard_types: List[HazardType] = field(default_factory=lambda: [HazardType.FLOOD])
    models: Dict[HazardType, ImpactModel] = field(default_factory=dict)
    cell_area_km2: float = 1.0
    min_severity: ImpactSeverity = ImpactSeverity.MINOR
    propagate_uncertainty: bool = True
    compute_compound: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hazard_types": [h.value for h in self.hazard_types],
            "models": {h.value: m.to_dict() for h, m in self.models.items()},
            "cell_area_km2": self.cell_area_km2,
            "min_severity": self.min_severity.value,
            "propagate_uncertainty": self.propagate_uncertainty,
            "compute_compound": self.compute_compound,
        }


@dataclass
class ProjectionResult:
    """
    Complete result of impact projection.

    Attributes:
        success: Whether projection completed successfully
        projections: Impact projections per hazard type
        compound: Compound impact assessment (if computed)
        summary: Overall summary
        issues: Issues encountered
        projection_time: When projection was performed
    """

    success: bool
    projections: Dict[HazardType, ImpactProjection] = field(default_factory=dict)
    compound: Optional[CompoundImpact] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    projection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def max_severity(self) -> ImpactSeverity:
        """Maximum severity across all hazards."""
        if not self.projections:
            return ImpactSeverity.NONE
        return max(self.projections.values(), key=lambda p: p.max_severity.level).max_severity

    def get_projection(self, hazard: HazardType) -> Optional[ImpactProjection]:
        """Get projection for specific hazard."""
        return self.projections.get(hazard)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "projections": {h.value: p.to_dict() for h, p in self.projections.items()},
            "compound": self.compound.to_dict() if self.compound else None,
            "summary": self.summary,
            "issues": self.issues,
            "projection_time": self.projection_time.isoformat(),
            "max_severity": self.max_severity.value,
        }


class ImpactProjector:
    """
    Projects forecast data into hazard impacts.

    Translates meteorological forecasts into actionable impact
    assessments using configurable impact models.

    Example:
        projector = ImpactProjector()
        config = ProjectionConfig(hazard_types=[HazardType.FLOOD])
        result = projector.project(forecast, config)
        flood_impact = result.get_projection(HazardType.FLOOD)
    """

    # Default impact models
    DEFAULT_MODELS: Dict[HazardType, ImpactModel] = {
        HazardType.FLOOD: ImpactModel(
            hazard_type=HazardType.FLOOD,
            name="Precipitation-Based Flood Impact",
            description="Estimates flood impact from precipitation accumulation",
            variables=["precipitation", "accumulated_precip"],
            thresholds=[
                ImpactThreshold("accumulated_precip", 0, 10, ImpactSeverity.NONE, description="Light rain"),
                ImpactThreshold("accumulated_precip", 10, 25, ImpactSeverity.MINOR, description="Minor ponding possible"),
                ImpactThreshold("accumulated_precip", 25, 50, ImpactSeverity.MODERATE, description="Urban flooding likely"),
                ImpactThreshold("accumulated_precip", 50, 100, ImpactSeverity.SIGNIFICANT, description="Significant flooding"),
                ImpactThreshold("accumulated_precip", 100, 200, ImpactSeverity.SEVERE, description="Severe flooding"),
                ImpactThreshold("accumulated_precip", 200, 500, ImpactSeverity.EXTREME, description="Extreme flooding"),
                ImpactThreshold("accumulated_precip", 500, float('inf'), ImpactSeverity.CATASTROPHIC, description="Catastrophic flooding"),
            ],
            aggregation_method="sum",
        ),
        HazardType.STORM: ImpactModel(
            hazard_type=HazardType.STORM,
            name="Wind-Based Storm Impact",
            description="Estimates storm impact from wind speed and gusts",
            variables=["wind_speed", "wind_gust"],
            thresholds=[
                ImpactThreshold("wind_speed", 0, 10, ImpactSeverity.NONE, description="Light winds"),
                ImpactThreshold("wind_speed", 10, 18, ImpactSeverity.MINOR, description="Fresh breeze"),
                ImpactThreshold("wind_speed", 18, 25, ImpactSeverity.MODERATE, description="Strong winds, small branches break"),
                ImpactThreshold("wind_speed", 25, 33, ImpactSeverity.SIGNIFICANT, description="High winds, trees may uproot"),
                ImpactThreshold("wind_speed", 33, 42, ImpactSeverity.SEVERE, description="Storm force, structural damage"),
                ImpactThreshold("wind_speed", 42, 55, ImpactSeverity.EXTREME, description="Hurricane force winds"),
                ImpactThreshold("wind_speed", 55, float('inf'), ImpactSeverity.CATASTROPHIC, description="Major hurricane"),
            ],
            aggregation_method="max",
        ),
        HazardType.WILDFIRE: ImpactModel(
            hazard_type=HazardType.WILDFIRE,
            name="Fire Weather Impact",
            description="Estimates fire weather risk from temperature, humidity, and wind",
            variables=["temperature", "humidity", "wind_speed"],
            thresholds=[
                ImpactThreshold("humidity", 50, 100, ImpactSeverity.NONE, description="High humidity"),
                ImpactThreshold("humidity", 25, 50, ImpactSeverity.MINOR, description="Moderate humidity"),
                ImpactThreshold("humidity", 15, 25, ImpactSeverity.MODERATE, description="Low humidity"),
                ImpactThreshold("humidity", 10, 15, ImpactSeverity.SIGNIFICANT, description="Very low humidity"),
                ImpactThreshold("humidity", 5, 10, ImpactSeverity.SEVERE, description="Critically low humidity"),
                ImpactThreshold("humidity", 0, 5, ImpactSeverity.EXTREME, description="Extreme fire danger"),
            ],
            aggregation_method="min",  # Lower humidity = worse
        ),
        HazardType.HEATWAVE: ImpactModel(
            hazard_type=HazardType.HEATWAVE,
            name="Heat Impact",
            description="Estimates heat stress impact from temperature",
            variables=["temperature", "temperature_2m"],
            thresholds=[
                ImpactThreshold("temperature_2m", 0, 303.15, ImpactSeverity.NONE, description="Normal (<30C)"),
                ImpactThreshold("temperature_2m", 303.15, 308.15, ImpactSeverity.MINOR, description="Warm (30-35C)"),
                ImpactThreshold("temperature_2m", 308.15, 313.15, ImpactSeverity.MODERATE, description="Hot (35-40C)"),
                ImpactThreshold("temperature_2m", 313.15, 318.15, ImpactSeverity.SIGNIFICANT, description="Very hot (40-45C)"),
                ImpactThreshold("temperature_2m", 318.15, 323.15, ImpactSeverity.SEVERE, description="Extreme heat (45-50C)"),
                ImpactThreshold("temperature_2m", 323.15, float('inf'), ImpactSeverity.EXTREME, description="Dangerous heat (>50C)"),
            ],
            aggregation_method="max",
        ),
    }

    def __init__(self, config: Optional[ProjectionConfig] = None):
        """
        Initialize projector.

        Args:
            config: Projection configuration
        """
        self.config = config or ProjectionConfig()
        self._models = dict(self.DEFAULT_MODELS)
        self._models.update(self.config.models)

    def project(
        self,
        forecast: ForecastDataset,
        config: Optional[ProjectionConfig] = None,
    ) -> ProjectionResult:
        """
        Project forecast into impacts.

        Args:
            forecast: Forecast dataset to project
            config: Projection configuration (overrides instance config)

        Returns:
            ProjectionResult with all impact projections
        """
        config = config or self.config
        logger.info("Starting impact projection")

        issues = []
        projections = {}

        for hazard in config.hazard_types:
            if hazard not in self._models:
                issues.append(f"No impact model for {hazard.value}")
                continue

            try:
                projection = self._project_hazard(forecast, hazard, config)
                projections[hazard] = projection
            except Exception as e:
                issues.append(f"Failed to project {hazard.value}: {e}")
                logger.exception(f"Error projecting {hazard.value}")

        # Compound impact assessment
        compound = None
        if config.compute_compound and len(projections) > 1:
            try:
                compound = self._assess_compound_impact(projections)
            except Exception as e:
                issues.append(f"Failed to assess compound impacts: {e}")

        # Summary
        summary = self._compute_summary(projections, compound)

        return ProjectionResult(
            success=len(issues) == 0 or len(projections) > 0,
            projections=projections,
            compound=compound,
            summary=summary,
            issues=issues,
        )

    def project_scenario(
        self,
        scenario: ScenarioData,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        hazard: HazardType,
    ) -> ImpactProjection:
        """
        Project a specific scenario into impacts.

        Args:
            scenario: Scenario to project
            grid_lat: Latitude grid
            grid_lon: Longitude grid
            hazard: Hazard type to project

        Returns:
            ImpactProjection for the scenario
        """
        model = self._models.get(hazard)
        if model is None:
            raise ValueError(f"No model for {hazard.value}")

        timesteps = []

        for ts in scenario.timesteps:
            impact_ts = self._project_timestep(ts, model, grid_lat, grid_lon)
            timesteps.append(impact_ts)

        return ImpactProjection(
            hazard_type=hazard,
            model=model,
            timesteps=timesteps,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )

    def _project_hazard(
        self,
        forecast: ForecastDataset,
        hazard: HazardType,
        config: ProjectionConfig,
    ) -> ImpactProjection:
        """Project single hazard type."""
        model = self._models[hazard]
        timesteps = []

        # Compute cumulative values if needed
        cumulative = {}
        if model.aggregation_method == "sum":
            for var in model.variables:
                cumulative[var] = np.zeros(forecast.shape)

        for ts in forecast.timesteps:
            # Update cumulative
            if model.aggregation_method == "sum":
                for var in model.variables:
                    data = ts.get_variable(var)
                    if data is not None:
                        if data.shape == cumulative[var].shape:
                            cumulative[var] += data

            impact_ts = self._project_timestep(
                ts, model, forecast.grid_lat, forecast.grid_lon,
                cumulative=cumulative if model.aggregation_method == "sum" else None,
                cell_area_km2=config.cell_area_km2,
                propagate_uncertainty=config.propagate_uncertainty,
            )
            timesteps.append(impact_ts)

        projection = ImpactProjection(
            hazard_type=hazard,
            model=model,
            timesteps=timesteps,
            grid_lat=forecast.grid_lat,
            grid_lon=forecast.grid_lon,
        )

        # Compute summary
        projection.summary = self._compute_projection_summary(projection)

        return projection

    def _project_timestep(
        self,
        ts: ForecastTimestep,
        model: ImpactModel,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        cumulative: Optional[Dict[str, np.ndarray]] = None,
        cell_area_km2: float = 1.0,
        propagate_uncertainty: bool = True,
    ) -> ImpactTimestep:
        """Project single timestep."""
        # Get shape from first variable
        shape = None
        for var in model.variables:
            data = ts.get_variable(var)
            if data is not None:
                shape = data.shape
                break

        if shape is None:
            shape = grid_lat.shape if grid_lat.ndim > 1 else (len(grid_lat), len(grid_lon))

        severity_field = np.zeros(shape, dtype=int)
        confidence_field = np.ones(shape)
        driver_values = {}

        # Evaluate each variable against thresholds
        for var in model.variables:
            # Use cumulative or instantaneous
            if cumulative and var in cumulative:
                data = cumulative[var]
            else:
                data = ts.get_variable(var)

            if data is None:
                continue

            driver_values[var] = float(np.nanmean(data))

            # Ensure data shape matches severity field, reshape if needed
            if data.shape != shape:
                # If data is 1D and shape is 2D, broadcast appropriately
                if data.ndim == 1 and len(shape) == 2:
                    # Assume data corresponds to first dimension
                    data = np.broadcast_to(data[:, np.newaxis], shape)
                elif data.ndim == 2 and data.shape != shape:
                    # Skip if shapes don't match and can't be resolved
                    continue

            # Apply thresholds
            for thresh in model.thresholds:
                if thresh.variable != var:
                    continue

                mask = (data >= thresh.min_value) & (data < thresh.max_value)
                # Take maximum severity
                severity_field = np.maximum(
                    severity_field,
                    np.where(mask, thresh.severity.level, 0)
                )

        # Compute confidence based on forecast quality flags
        if propagate_uncertainty:
            quality_sum = sum(ts.quality_flags.get(v, 1.0) for v in model.variables)
            mean_quality = quality_sum / len(model.variables) if model.variables else 1.0
            confidence_field = confidence_field * mean_quality * (1 - model.uncertainty_factor)

        # Compute affected area
        affected_cells = np.sum(severity_field > ImpactSeverity.NONE.level)
        affected_area_km2 = affected_cells * cell_area_km2

        # Peak severity
        peak_severity = ImpactSeverity.NONE
        max_sev = np.nanmax(severity_field)
        for sev in ImpactSeverity:
            if sev.level == max_sev:
                peak_severity = sev
                break

        return ImpactTimestep(
            valid_time=ts.valid_time,
            severity_field=severity_field,
            confidence_field=confidence_field,
            affected_area_km2=affected_area_km2,
            peak_severity=peak_severity,
            driver_values=driver_values,
        )

    def _assess_compound_impact(
        self,
        projections: Dict[HazardType, ImpactProjection],
    ) -> CompoundImpact:
        """Assess compound impacts from multiple hazards."""
        hazards = list(projections.keys())

        # Get common timesteps
        all_times = set()
        for proj in projections.values():
            all_times.update(ts.valid_time for ts in proj.timesteps)

        # Compute compound severity (maximum across hazards)
        compound_severity = None

        for hazard, proj in projections.items():
            cumulative = proj.get_cumulative_severity()
            if compound_severity is None:
                compound_severity = cumulative.copy()
            else:
                if cumulative.shape == compound_severity.shape:
                    compound_severity = np.maximum(compound_severity, cumulative)

        # Compute interaction effects (simplified)
        interactions = {}
        if HazardType.FLOOD in projections and HazardType.STORM in projections:
            interactions["flood_storm_synergy"] = 1.2  # Storms enhance flood impact

        if HazardType.HEATWAVE in projections and HazardType.WILDFIRE in projections:
            interactions["heat_fire_synergy"] = 1.5  # Heat enhances fire risk

        # Compound probability (simplified: joint occurrence)
        probs = []
        for proj in projections.values():
            # Estimate probability based on severity
            if proj.max_severity.level >= ImpactSeverity.MODERATE.level:
                probs.append(0.5)
            else:
                probs.append(0.1)

        compound_prob = np.prod(probs) if probs else 0.0

        return CompoundImpact(
            hazards=hazards,
            projections=projections,
            compound_severity_field=compound_severity,
            interaction_effects=interactions,
            compound_probability=compound_prob,
        )

    def _compute_projection_summary(
        self,
        projection: ImpactProjection,
    ) -> Dict[str, Any]:
        """Compute summary statistics for a projection."""
        summary = {
            "n_timesteps": len(projection.timesteps),
            "max_severity": projection.max_severity.value,
            "total_affected_area_km2": projection.total_affected_area_km2,
        }

        if projection.timesteps:
            # Time of max impact
            max_ts = max(projection.timesteps, key=lambda ts: ts.peak_severity.level)
            summary["peak_time"] = max_ts.valid_time.isoformat()
            summary["peak_drivers"] = max_ts.driver_values

            # Mean confidence
            confidences = [ts.mean_confidence for ts in projection.timesteps]
            summary["mean_confidence"] = float(np.mean(confidences))

            # Severity duration
            severe_hours = sum(
                1 for ts in projection.timesteps
                if ts.peak_severity.level >= ImpactSeverity.SEVERE.level
            )
            summary["severe_duration_hours"] = severe_hours * 3  # Assuming 3-hour timesteps

        return summary

    def _compute_summary(
        self,
        projections: Dict[HazardType, ImpactProjection],
        compound: Optional[CompoundImpact],
    ) -> Dict[str, Any]:
        """Compute overall summary."""
        summary = {
            "n_hazards": len(projections),
            "hazards_assessed": [h.value for h in projections.keys()],
        }

        if projections:
            max_sev = max(projections.values(), key=lambda p: p.max_severity.level).max_severity
            summary["overall_max_severity"] = max_sev.value

            total_area = sum(p.total_affected_area_km2 for p in projections.values())
            summary["total_affected_area_km2"] = total_area

        if compound:
            summary["compound_probability"] = compound.compound_probability
            summary["interaction_effects"] = compound.interaction_effects

        return summary

    def register_model(
        self,
        hazard: HazardType,
        model: ImpactModel,
    ):
        """Register a custom impact model."""
        self._models[hazard] = model


def project_impacts(
    forecast: ForecastDataset,
    hazards: Optional[List[HazardType]] = None,
    cell_area_km2: float = 1.0,
) -> ProjectionResult:
    """
    Convenience function for impact projection.

    Args:
        forecast: Forecast dataset to project
        hazards: Hazard types to assess (default: [FLOOD])
        cell_area_km2: Grid cell area for calculations

    Returns:
        ProjectionResult with all impact projections
    """
    config = ProjectionConfig(
        hazard_types=hazards or [HazardType.FLOOD],
        cell_area_km2=cell_area_km2,
    )

    projector = ImpactProjector(config)
    return projector.project(forecast, config)


def get_impact_model(hazard: HazardType) -> Optional[ImpactModel]:
    """Get the default impact model for a hazard type."""
    return ImpactProjector.DEFAULT_MODELS.get(hazard)


def create_custom_impact_model(
    hazard: HazardType,
    name: str,
    variables: List[str],
    thresholds: List[Tuple[str, float, float, ImpactSeverity]],
    aggregation: str = "max",
) -> ImpactModel:
    """
    Create a custom impact model.

    Args:
        hazard: Hazard type
        name: Model name
        variables: Required variables
        thresholds: List of (variable, min, max, severity) tuples
        aggregation: Aggregation method ("max", "sum", "min", "mean")

    Returns:
        ImpactModel configured with custom thresholds
    """
    thresh_objects = [
        ImpactThreshold(
            variable=var,
            min_value=min_val,
            max_value=max_val,
            severity=severity,
        )
        for var, min_val, max_val, severity in thresholds
    ]

    return ImpactModel(
        hazard_type=hazard,
        name=name,
        variables=variables,
        thresholds=thresh_objects,
        aggregation_method=aggregation,
    )
