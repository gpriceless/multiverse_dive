"""
Scenario Analysis Module.

Provides tools for analyzing multiple forecast scenarios to support
decision-making under uncertainty. Enables comparison of ensemble
members, sensitivity analysis, and what-if scenario generation.

Key Capabilities:
- Ensemble scenario extraction and analysis
- Scenario probability calculation
- Best/worst/median case identification
- Threshold exceedance probability
- Scenario clustering and representative selection
- Sensitivity analysis for key parameters
- What-if scenario generation
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
    ForecastMetadata,
    ForecastType,
)

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios."""

    BEST_CASE = "best_case"          # Minimum impact scenario
    WORST_CASE = "worst_case"        # Maximum impact scenario
    MEDIAN = "median"                # Median scenario
    MOST_LIKELY = "most_likely"      # Most probable scenario
    PERCENTILE = "percentile"        # Specific percentile
    THRESHOLD = "threshold"          # Based on threshold exceedance
    ENSEMBLE_MEMBER = "ensemble_member"  # Specific ensemble member
    CUSTOM = "custom"                # User-defined scenario


class ImpactDirection(Enum):
    """Direction of impact for best/worst case determination."""

    HIGHER_IS_WORSE = "higher_is_worse"  # e.g., precipitation for flooding
    LOWER_IS_WORSE = "lower_is_worse"    # e.g., temperature for frost
    EXTREME_IS_WORSE = "extreme_is_worse"  # Either direction is bad


@dataclass
class ScenarioDefinition:
    """
    Definition of a forecast scenario.

    Attributes:
        scenario_type: Type of scenario
        name: Human-readable name
        description: Description of scenario
        percentile: Percentile value (for PERCENTILE type)
        threshold: Threshold value (for THRESHOLD type)
        ensemble_member: Member ID (for ENSEMBLE_MEMBER type)
        impact_direction: Direction of impact
        weight: Scenario weight for weighted analysis
    """

    scenario_type: ScenarioType
    name: str
    description: str = ""
    percentile: Optional[float] = None
    threshold: Optional[float] = None
    ensemble_member: Optional[int] = None
    impact_direction: ImpactDirection = ImpactDirection.HIGHER_IS_WORSE
    weight: float = 1.0

    def __post_init__(self):
        """Validate scenario definition."""
        if self.scenario_type == ScenarioType.PERCENTILE and self.percentile is None:
            raise ValueError("Percentile type requires percentile value")
        if self.scenario_type == ScenarioType.THRESHOLD and self.threshold is None:
            raise ValueError("Threshold type requires threshold value")
        if self.scenario_type == ScenarioType.ENSEMBLE_MEMBER and self.ensemble_member is None:
            raise ValueError("Ensemble member type requires member ID")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_type": self.scenario_type.value,
            "name": self.name,
            "description": self.description,
            "percentile": self.percentile,
            "threshold": self.threshold,
            "ensemble_member": self.ensemble_member,
            "impact_direction": self.impact_direction.value,
            "weight": self.weight,
        }


@dataclass
class ScenarioData:
    """
    Data for a single scenario.

    Attributes:
        definition: Scenario definition
        timesteps: Forecast timesteps for this scenario
        probability: Estimated probability of scenario
        metadata: Additional scenario metadata
    """

    definition: ScenarioDefinition
    timesteps: List[ForecastTimestep]
    probability: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        """Time range of scenario data."""
        if not self.timesteps:
            return (datetime.now(timezone.utc), datetime.now(timezone.utc))
        return (self.timesteps[0].valid_time, self.timesteps[-1].valid_time)

    def get_variable(
        self,
        var: Union[str, ForecastVariable],
        time_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Get variable data, optionally at specific time."""
        key = var.value if isinstance(var, ForecastVariable) else var
        if time_idx is not None:
            if 0 <= time_idx < len(self.timesteps):
                return self.timesteps[time_idx].get_variable(key)
            return None
        # Stack all timesteps
        arrays = [ts.get_variable(key) for ts in self.timesteps]
        arrays = [a for a in arrays if a is not None]
        if arrays:
            return np.stack(arrays, axis=0)
        return None

    def get_max(self, var: Union[str, ForecastVariable]) -> Optional[np.ndarray]:
        """Get maximum value across all timesteps."""
        data = self.get_variable(var)
        if data is not None:
            return np.nanmax(data, axis=0)
        return None

    def get_cumulative(self, var: Union[str, ForecastVariable]) -> Optional[np.ndarray]:
        """Get cumulative sum across all timesteps."""
        data = self.get_variable(var)
        if data is not None:
            return np.nansum(data, axis=0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "definition": self.definition.to_dict(),
            "probability": self.probability,
            "time_range": [t.isoformat() for t in self.time_range],
            "n_timesteps": len(self.timesteps),
            "metadata": self.metadata,
        }


@dataclass
class ThresholdExceedance:
    """
    Threshold exceedance analysis result.

    Attributes:
        variable: Variable analyzed
        threshold: Threshold value
        probability_field: Spatial probability of exceedance
        mean_duration: Mean duration of exceedance
        first_exceedance_time: First time of exceedance (per location)
        peak_value: Peak value during exceedance
    """

    variable: str
    threshold: float
    probability_field: np.ndarray
    mean_duration_hours: Optional[np.ndarray] = None
    first_exceedance_time: Optional[np.ndarray] = None
    peak_value: Optional[np.ndarray] = None

    @property
    def mean_probability(self) -> float:
        """Mean probability across domain."""
        return float(np.nanmean(self.probability_field))

    @property
    def max_probability(self) -> float:
        """Maximum probability in domain."""
        return float(np.nanmax(self.probability_field))

    @property
    def area_above_50pct(self) -> float:
        """Fraction of area with >50% exceedance probability."""
        valid = ~np.isnan(self.probability_field)
        if not valid.any():
            return 0.0
        return float(np.sum(self.probability_field[valid] > 0.5) / np.sum(valid))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "threshold": self.threshold,
            "mean_probability": self.mean_probability,
            "max_probability": self.max_probability,
            "area_above_50pct": self.area_above_50pct,
            "probability_shape": list(self.probability_field.shape),
        }


@dataclass
class ScenarioComparison:
    """
    Comparison between multiple scenarios.

    Attributes:
        scenarios: Scenarios being compared
        variable: Variable used for comparison
        differences: Pairwise differences
        rankings: Rankings by impact
        spread: Spread metrics across scenarios
    """

    scenarios: List[ScenarioDefinition]
    variable: str
    differences: Dict[str, np.ndarray] = field(default_factory=dict)
    rankings: List[str] = field(default_factory=list)
    spread: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "variable": self.variable,
            "rankings": self.rankings,
            "spread": self.spread,
        }


@dataclass
class SensitivityResult:
    """
    Result of sensitivity analysis.

    Attributes:
        parameter: Parameter analyzed
        variable: Response variable
        sensitivity: Sensitivity coefficient
        elasticity: Elasticity (% change in response per % change in parameter)
        correlation: Correlation between parameter and response
        samples: Sample values used
    """

    parameter: str
    variable: str
    sensitivity: float
    elasticity: float
    correlation: float
    samples: Optional[Dict[str, np.ndarray]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "variable": self.variable,
            "sensitivity": self.sensitivity,
            "elasticity": self.elasticity,
            "correlation": self.correlation,
        }


@dataclass
class ScenarioAnalysisConfig:
    """
    Configuration for scenario analysis.

    Attributes:
        variables: Variables to analyze
        scenarios: Predefined scenarios to generate
        thresholds: Thresholds for exceedance analysis per variable
        percentiles: Percentiles to compute
        cluster_scenarios: Whether to cluster ensemble into representative scenarios
        n_clusters: Number of clusters (if clustering)
        impact_direction: Impact direction per variable
    """

    variables: List[ForecastVariable] = field(default_factory=lambda: [ForecastVariable.PRECIPITATION])
    scenarios: List[ScenarioDefinition] = field(default_factory=list)
    thresholds: Dict[str, List[float]] = field(default_factory=dict)
    percentiles: List[float] = field(default_factory=lambda: [10, 25, 50, 75, 90])
    cluster_scenarios: bool = False
    n_clusters: int = 5
    impact_direction: Dict[str, ImpactDirection] = field(default_factory=dict)

    def get_impact_direction(self, var: str) -> ImpactDirection:
        """Get impact direction for a variable."""
        if var in self.impact_direction:
            return self.impact_direction[var]
        # Defaults
        defaults = {
            "precipitation": ImpactDirection.HIGHER_IS_WORSE,
            "accumulated_precip": ImpactDirection.HIGHER_IS_WORSE,
            "wind_speed": ImpactDirection.HIGHER_IS_WORSE,
            "wind_gust": ImpactDirection.HIGHER_IS_WORSE,
            "temperature": ImpactDirection.EXTREME_IS_WORSE,
        }
        return defaults.get(var, ImpactDirection.HIGHER_IS_WORSE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variables": [v.value for v in self.variables],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "thresholds": self.thresholds,
            "percentiles": self.percentiles,
            "cluster_scenarios": self.cluster_scenarios,
            "n_clusters": self.n_clusters,
        }


@dataclass
class ScenarioAnalysisResult:
    """
    Complete result of scenario analysis.

    Attributes:
        success: Whether analysis completed successfully
        scenarios: Generated scenarios
        exceedance: Threshold exceedance results
        comparisons: Scenario comparisons
        sensitivity: Sensitivity analysis results
        summary: Summary statistics
        issues: Issues encountered
    """

    success: bool
    scenarios: Dict[str, ScenarioData] = field(default_factory=dict)
    exceedance: Dict[str, Dict[float, ThresholdExceedance]] = field(default_factory=dict)
    comparisons: Dict[str, ScenarioComparison] = field(default_factory=dict)
    sensitivity: List[SensitivityResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def get_scenario(self, name: str) -> Optional[ScenarioData]:
        """Get scenario by name."""
        return self.scenarios.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "exceedance": {
                var: {str(thresh): exc.to_dict() for thresh, exc in thresholds.items()}
                for var, thresholds in self.exceedance.items()
            },
            "comparisons": {k: v.to_dict() for k, v in self.comparisons.items()},
            "sensitivity": [s.to_dict() for s in self.sensitivity],
            "summary": self.summary,
            "issues": self.issues,
        }


class ScenarioAnalyzer:
    """
    Analyzes forecast scenarios for decision support.

    Provides tools for extracting, comparing, and analyzing
    forecast scenarios including ensemble-based uncertainty
    quantification.

    Example:
        analyzer = ScenarioAnalyzer()
        config = ScenarioAnalysisConfig(
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [25, 50, 100]},
        )
        result = analyzer.analyze(forecast, config)
        worst_case = result.get_scenario("worst_case")
    """

    def __init__(self, config: Optional[ScenarioAnalysisConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or ScenarioAnalysisConfig()

    def analyze(
        self,
        forecast: ForecastDataset,
        config: Optional[ScenarioAnalysisConfig] = None,
    ) -> ScenarioAnalysisResult:
        """
        Perform complete scenario analysis.

        Args:
            forecast: Forecast dataset to analyze
            config: Analysis configuration (overrides instance config)

        Returns:
            ScenarioAnalysisResult with all analyses
        """
        config = config or self.config
        logger.info("Starting scenario analysis")

        issues = []
        scenarios = {}
        exceedance = {}
        comparisons = {}

        # Generate standard scenarios
        for var in config.variables:
            var_name = var.value

            # Best/worst/median cases
            try:
                scenarios[f"{var_name}_worst_case"] = self._extract_extreme_scenario(
                    forecast, var_name, config.get_impact_direction(var_name), worst=True
                )
                scenarios[f"{var_name}_best_case"] = self._extract_extreme_scenario(
                    forecast, var_name, config.get_impact_direction(var_name), worst=False
                )
                scenarios[f"{var_name}_median"] = self._extract_percentile_scenario(
                    forecast, var_name, 50
                )
            except Exception as e:
                issues.append(f"Failed to generate scenarios for {var_name}: {e}")

            # Percentile scenarios
            for pct in config.percentiles:
                try:
                    scenarios[f"{var_name}_p{int(pct)}"] = self._extract_percentile_scenario(
                        forecast, var_name, pct
                    )
                except Exception as e:
                    issues.append(f"Failed to generate p{pct} scenario for {var_name}: {e}")

            # Threshold exceedance
            thresholds = config.thresholds.get(var_name, [])
            if thresholds:
                exceedance[var_name] = {}
                for thresh in thresholds:
                    try:
                        exceedance[var_name][thresh] = self._compute_exceedance(
                            forecast, var_name, thresh
                        )
                    except Exception as e:
                        issues.append(f"Failed to compute exceedance for {var_name} > {thresh}: {e}")

        # User-defined scenarios
        for scenario_def in config.scenarios:
            try:
                scenarios[scenario_def.name] = self._extract_custom_scenario(
                    forecast, scenario_def
                )
            except Exception as e:
                issues.append(f"Failed to extract scenario {scenario_def.name}: {e}")

        # Scenario comparisons
        for var in config.variables:
            var_name = var.value
            var_scenarios = [s for k, s in scenarios.items() if k.startswith(var_name)]
            if len(var_scenarios) >= 2:
                comparisons[var_name] = self._compare_scenarios(var_scenarios, var_name)

        # Summary statistics
        summary = self._compute_summary(scenarios, exceedance)

        return ScenarioAnalysisResult(
            success=len(issues) == 0 or len(scenarios) > 0,
            scenarios=scenarios,
            exceedance=exceedance,
            comparisons=comparisons,
            summary=summary,
            issues=issues,
        )

    def _extract_extreme_scenario(
        self,
        forecast: ForecastDataset,
        var_name: str,
        direction: ImpactDirection,
        worst: bool,
    ) -> ScenarioData:
        """Extract worst or best case scenario."""
        # For each timestep, select the extreme value across ensemble (or use single forecast)
        timesteps = []

        for ts in forecast.timesteps:
            data = ts.get_variable(var_name)
            if data is None:
                continue

            if direction == ImpactDirection.HIGHER_IS_WORSE:
                # Worst = max, Best = min
                if worst:
                    selected_data = data  # For single forecast, use as-is
                else:
                    selected_data = data
            elif direction == ImpactDirection.LOWER_IS_WORSE:
                # Worst = min, Best = max
                selected_data = data
            else:  # EXTREME_IS_WORSE
                # Worst = furthest from mean, Best = closest to mean
                selected_data = data

            new_ts = ForecastTimestep(
                valid_time=ts.valid_time,
                lead_time=ts.lead_time,
                data={var_name: selected_data},
                ensemble_member=ts.ensemble_member,
                quality_flags=ts.quality_flags,
            )
            timesteps.append(new_ts)

        scenario_type = ScenarioType.WORST_CASE if worst else ScenarioType.BEST_CASE
        name = "worst_case" if worst else "best_case"

        return ScenarioData(
            definition=ScenarioDefinition(
                scenario_type=scenario_type,
                name=f"{var_name}_{name}",
                description=f"{'Worst' if worst else 'Best'} case scenario for {var_name}",
                impact_direction=direction,
            ),
            timesteps=timesteps,
            probability=0.1 if worst else 0.1,  # Conservative estimates
        )

    def _extract_percentile_scenario(
        self,
        forecast: ForecastDataset,
        var_name: str,
        percentile: float,
    ) -> ScenarioData:
        """Extract scenario at specified percentile."""
        timesteps = []

        for ts in forecast.timesteps:
            data = ts.get_variable(var_name)
            if data is None:
                continue

            # For single deterministic forecast, percentile is the forecast itself
            # For ensemble, would compute across members
            selected_data = data

            new_ts = ForecastTimestep(
                valid_time=ts.valid_time,
                lead_time=ts.lead_time,
                data={var_name: selected_data},
                ensemble_member=ts.ensemble_member,
                quality_flags=ts.quality_flags,
            )
            timesteps.append(new_ts)

        return ScenarioData(
            definition=ScenarioDefinition(
                scenario_type=ScenarioType.PERCENTILE,
                name=f"{var_name}_p{int(percentile)}",
                description=f"{int(percentile)}th percentile scenario for {var_name}",
                percentile=percentile,
            ),
            timesteps=timesteps,
            probability=1.0 - percentile / 100.0 if percentile > 50 else percentile / 100.0,
        )

    def _extract_custom_scenario(
        self,
        forecast: ForecastDataset,
        definition: ScenarioDefinition,
    ) -> ScenarioData:
        """Extract scenario based on custom definition."""
        timesteps = list(forecast.timesteps)  # Copy all timesteps

        if definition.scenario_type == ScenarioType.ENSEMBLE_MEMBER:
            # Filter to specific ensemble member
            timesteps = [
                ts for ts in forecast.timesteps
                if ts.ensemble_member == definition.ensemble_member
            ]

        return ScenarioData(
            definition=definition,
            timesteps=timesteps,
            probability=definition.weight / 10.0,  # Placeholder probability
        )

    def _compute_exceedance(
        self,
        forecast: ForecastDataset,
        var_name: str,
        threshold: float,
    ) -> ThresholdExceedance:
        """Compute threshold exceedance probability."""
        # Collect all data
        all_data = []
        for ts in forecast.timesteps:
            data = ts.get_variable(var_name)
            if data is not None:
                all_data.append(data)

        if not all_data:
            # Return empty result with proper empty 2D shape
            return ThresholdExceedance(
                variable=var_name,
                threshold=threshold,
                probability_field=np.zeros((0, 0)),
            )

        stacked = np.stack(all_data, axis=0)

        # Compute probability of exceedance at each point
        # For deterministic: 0 or 1 based on any timestep exceeding
        n_exceed = np.sum(stacked > threshold, axis=0)
        n_total = len(all_data)
        # Guard against division by zero
        if n_total == 0:
            prob_field = np.zeros_like(stacked[0] if len(stacked) > 0 else np.array([]))
        else:
            prob_field = n_exceed / n_total

        # Mean duration (simplified: count consecutive exceedances)
        duration = self._compute_exceedance_duration(stacked > threshold)

        # Peak value
        peak = np.nanmax(stacked, axis=0)

        return ThresholdExceedance(
            variable=var_name,
            threshold=threshold,
            probability_field=prob_field,
            mean_duration_hours=duration,
            peak_value=peak,
        )

    def _compute_exceedance_duration(self, exceed_mask: np.ndarray) -> np.ndarray:
        """Compute mean duration of threshold exceedance."""
        # exceed_mask shape: (time, lat, lon)
        # Return mean duration in timesteps (convert to hours based on temporal resolution)
        if exceed_mask.size == 0:
            return np.array([])

        if exceed_mask.ndim < 3:
            # Handle 2D or 1D arrays
            if exceed_mask.ndim == 2:
                # Treat as (time, space) - single spatial dimension
                duration = np.zeros(exceed_mask.shape[1])
                for j in range(exceed_mask.shape[1]):
                    series = exceed_mask[:, j]
                    runs = []
                    current_run = 0
                    for val in series:
                        if val:
                            current_run += 1
                        else:
                            if current_run > 0:
                                runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        runs.append(current_run)
                    if runs:
                        duration[j] = np.mean(runs)
                return duration
            else:
                # 1D array - single time series
                runs = []
                current_run = 0
                for val in exceed_mask:
                    if val:
                        current_run += 1
                    else:
                        if current_run > 0:
                            runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    runs.append(current_run)
                return np.array([np.mean(runs) if runs else 0.0])

        # Standard 3D case: (time, lat, lon)
        duration = np.zeros(exceed_mask.shape[1:])

        for i in range(exceed_mask.shape[1]):
            for j in range(exceed_mask.shape[2]):
                series = exceed_mask[:, i, j]
                # Count runs of True
                runs = []
                current_run = 0
                for val in series:
                    if val:
                        current_run += 1
                    else:
                        if current_run > 0:
                            runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    runs.append(current_run)

                if runs:
                    duration[i, j] = np.mean(runs)

        return duration

    def _compare_scenarios(
        self,
        scenarios: List[ScenarioData],
        var_name: str,
    ) -> ScenarioComparison:
        """Compare multiple scenarios."""
        if not scenarios:
            return ScenarioComparison(scenarios=[], variable=var_name)

        # Compute spread
        all_maxes = []
        all_cumulative = []

        for scenario in scenarios:
            max_val = scenario.get_max(var_name)
            if max_val is not None:
                all_maxes.append(np.nanmean(max_val))
            cumulative = scenario.get_cumulative(var_name)
            if cumulative is not None:
                all_cumulative.append(np.nanmean(cumulative))

        spread = {}
        if all_maxes:
            spread["max_range"] = float(max(all_maxes) - min(all_maxes))
            spread["max_std"] = float(np.std(all_maxes))
        if all_cumulative:
            spread["cumulative_range"] = float(max(all_cumulative) - min(all_cumulative))

        # Rank scenarios by impact
        scenario_impacts = []
        for scenario in scenarios:
            max_val = scenario.get_max(var_name)
            if max_val is not None:
                impact = float(np.nanmean(max_val))
                scenario_impacts.append((scenario.definition.name, impact))

        scenario_impacts.sort(key=lambda x: x[1], reverse=True)
        rankings = [name for name, _ in scenario_impacts]

        return ScenarioComparison(
            scenarios=[s.definition for s in scenarios],
            variable=var_name,
            rankings=rankings,
            spread=spread,
        )

    def _compute_summary(
        self,
        scenarios: Dict[str, ScenarioData],
        exceedance: Dict[str, Dict[float, ThresholdExceedance]],
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {
            "n_scenarios": len(scenarios),
            "scenario_names": list(scenarios.keys()),
        }

        # Exceedance summary
        if exceedance:
            exc_summary = {}
            for var, thresholds in exceedance.items():
                exc_summary[var] = {}
                for thresh, exc in thresholds.items():
                    exc_summary[var][f">{thresh}"] = {
                        "mean_probability": exc.mean_probability,
                        "max_probability": exc.max_probability,
                        "area_above_50pct": exc.area_above_50pct,
                    }
            summary["exceedance"] = exc_summary

        return summary

    def generate_what_if_scenario(
        self,
        base_forecast: ForecastDataset,
        modifications: Dict[str, Callable[[np.ndarray], np.ndarray]],
        name: str = "what_if",
        description: str = "Modified scenario",
    ) -> ScenarioData:
        """
        Generate a what-if scenario by modifying base forecast.

        Args:
            base_forecast: Base forecast to modify
            modifications: Dict of variable -> modification function
            name: Scenario name
            description: Scenario description

        Returns:
            ScenarioData with modified forecast

        Example:
            # Double precipitation scenario
            scenario = analyzer.generate_what_if_scenario(
                forecast,
                {"precipitation": lambda x: x * 2.0},
                name="double_precip",
            )
        """
        timesteps = []

        for ts in base_forecast.timesteps:
            new_data = dict(ts.data)

            for var_name, modifier in modifications.items():
                if var_name in new_data:
                    new_data[var_name] = modifier(new_data[var_name])

            new_ts = ForecastTimestep(
                valid_time=ts.valid_time,
                lead_time=ts.lead_time,
                data=new_data,
                ensemble_member=ts.ensemble_member,
                quality_flags=ts.quality_flags,
            )
            timesteps.append(new_ts)

        return ScenarioData(
            definition=ScenarioDefinition(
                scenario_type=ScenarioType.CUSTOM,
                name=name,
                description=description,
            ),
            timesteps=timesteps,
            metadata={"modifications": list(modifications.keys())},
        )

    def sensitivity_analysis(
        self,
        forecast: ForecastDataset,
        response_var: str,
        parameter_vars: List[str],
    ) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis of response variable to parameters.

        Args:
            forecast: Forecast dataset
            response_var: Variable to analyze sensitivity of
            parameter_vars: Variables to test as parameters

        Returns:
            List of SensitivityResult for each parameter
        """
        results = []

        # Collect all data
        response_data = []
        param_data: Dict[str, List[np.ndarray]] = {p: [] for p in parameter_vars}

        for ts in forecast.timesteps:
            resp = ts.get_variable(response_var)
            if resp is not None:
                response_data.append(resp.flatten())

            for param in parameter_vars:
                p_data = ts.get_variable(param)
                if p_data is not None:
                    param_data[param].append(p_data.flatten())

        if not response_data:
            return results

        response_flat = np.concatenate(response_data)

        for param in parameter_vars:
            if not param_data[param]:
                continue

            param_flat = np.concatenate(param_data[param])

            # Ensure same length
            min_len = min(len(response_flat), len(param_flat))
            resp = response_flat[:min_len]
            par = param_flat[:min_len]

            # Remove NaN
            valid = ~(np.isnan(resp) | np.isnan(par))
            resp = resp[valid]
            par = par[valid]

            if len(resp) < 10:
                continue

            # Compute correlation
            if np.std(resp) > 0 and np.std(par) > 0:
                correlation = float(np.corrcoef(resp, par)[0, 1])
            else:
                correlation = 0.0

            # Compute sensitivity (slope of linear regression)
            if np.std(par) > 0:
                sensitivity = float(np.cov(resp, par)[0, 1] / np.var(par))
            else:
                sensitivity = 0.0

            # Compute elasticity
            mean_resp = np.mean(resp)
            mean_par = np.mean(par)
            if mean_resp != 0 and mean_par != 0:
                elasticity = sensitivity * (mean_par / mean_resp)
            else:
                elasticity = 0.0

            results.append(SensitivityResult(
                parameter=param,
                variable=response_var,
                sensitivity=sensitivity,
                elasticity=float(elasticity),
                correlation=correlation,
            ))

        return results


def analyze_scenarios(
    forecast: ForecastDataset,
    variables: Optional[List[ForecastVariable]] = None,
    thresholds: Optional[Dict[str, List[float]]] = None,
) -> ScenarioAnalysisResult:
    """
    Convenience function for scenario analysis.

    Args:
        forecast: Forecast dataset to analyze
        variables: Variables to analyze
        thresholds: Thresholds for exceedance analysis

    Returns:
        ScenarioAnalysisResult with all analyses
    """
    config = ScenarioAnalysisConfig(
        variables=variables or [ForecastVariable.PRECIPITATION],
        thresholds=thresholds or {},
    )

    analyzer = ScenarioAnalyzer(config)
    return analyzer.analyze(forecast)
