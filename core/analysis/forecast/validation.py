"""
Forecast Validation Module.

Validates forecast data against observations and provides comprehensive
metrics for forecast skill assessment. Supports multiple verification
approaches including point-based, spatial, and ensemble validation.

Key Capabilities:
- Forecast vs observation comparison
- Standard verification metrics (RMSE, MAE, bias, correlation)
- Skill scores (BSS, HSS, POD, FAR, CSI)
- Spatial verification (FSS, SAL)
- Ensemble spread-skill analysis
- Lead time dependent performance tracking
- Temporal validation across forecast windows
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
    ForecastType,
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of verification metrics."""

    # Continuous metrics
    RMSE = "rmse"                      # Root Mean Square Error
    MAE = "mae"                        # Mean Absolute Error
    BIAS = "bias"                      # Mean bias (forecast - observed)
    CORRELATION = "correlation"        # Pearson correlation coefficient
    EXPLAINED_VARIANCE = "explained_variance"

    # Binary/categorical metrics
    POD = "pod"                        # Probability of Detection (hit rate)
    FAR = "far"                        # False Alarm Ratio
    CSI = "csi"                        # Critical Success Index
    HSS = "hss"                        # Heidke Skill Score
    PSS = "pss"                        # Peirce Skill Score
    BIAS_RATIO = "bias_ratio"          # Frequency bias

    # Probabilistic metrics
    BSS = "bss"                        # Brier Skill Score
    CRPS = "crps"                      # Continuous Ranked Probability Score
    RELIABILITY = "reliability"
    RESOLUTION = "resolution"
    ROC_AUC = "roc_auc"                # Area under ROC curve

    # Spatial metrics
    FSS = "fss"                        # Fractions Skill Score
    SAL_STRUCTURE = "sal_structure"    # SAL Structure component
    SAL_AMPLITUDE = "sal_amplitude"    # SAL Amplitude component
    SAL_LOCATION = "sal_location"      # SAL Location component


class ValidationLevel(Enum):
    """Level of validation detail."""

    BASIC = "basic"           # Simple aggregate metrics
    STANDARD = "standard"     # Time-varying metrics
    COMPREHENSIVE = "comprehensive"  # Full spatial/temporal analysis


@dataclass
class ObservationData:
    """
    Observation data for forecast validation.

    Attributes:
        valid_time: Time of observation
        data: Dictionary of variable name -> numpy array
        grid_lat: Latitude grid
        grid_lon: Longitude grid
        quality_mask: Boolean mask of valid observations
        source: Source of observations
    """

    valid_time: datetime
    data: Dict[str, np.ndarray]
    grid_lat: np.ndarray
    grid_lon: np.ndarray
    quality_mask: Optional[np.ndarray] = None
    source: str = "unknown"

    def __post_init__(self):
        """Ensure timezone."""
        if self.valid_time.tzinfo is None:
            self.valid_time = self.valid_time.replace(tzinfo=timezone.utc)

        if self.quality_mask is None:
            # Default: all valid
            first_var = next(iter(self.data.values()))
            self.quality_mask = np.ones(first_var.shape, dtype=bool)

    def get_variable(self, var: Union[str, ForecastVariable]) -> Optional[np.ndarray]:
        """Get data for a variable."""
        key = var.value if isinstance(var, ForecastVariable) else var
        return self.data.get(key)


@dataclass
class ContingencyTable:
    """
    2x2 contingency table for binary verification.

    Attributes:
        hits: Forecast yes, observed yes
        misses: Forecast no, observed yes
        false_alarms: Forecast yes, observed no
        correct_negatives: Forecast no, observed no
    """

    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_negatives: int = 0

    @property
    def n(self) -> int:
        """Total sample size."""
        return self.hits + self.misses + self.false_alarms + self.correct_negatives

    @property
    def observed_yes(self) -> int:
        """Total observed yes."""
        return self.hits + self.misses

    @property
    def forecast_yes(self) -> int:
        """Total forecast yes."""
        return self.hits + self.false_alarms

    @property
    def pod(self) -> float:
        """Probability of Detection (hit rate)."""
        if self.observed_yes == 0:
            return np.nan
        return self.hits / self.observed_yes

    @property
    def far(self) -> float:
        """False Alarm Ratio."""
        if self.forecast_yes == 0:
            return np.nan
        return self.false_alarms / self.forecast_yes

    @property
    def csi(self) -> float:
        """Critical Success Index (Threat Score)."""
        denom = self.hits + self.misses + self.false_alarms
        if denom == 0:
            return np.nan
        return self.hits / denom

    @property
    def bias_ratio(self) -> float:
        """Frequency Bias."""
        if self.observed_yes == 0:
            return np.nan
        return self.forecast_yes / self.observed_yes

    @property
    def hss(self) -> float:
        """Heidke Skill Score."""
        if self.n == 0:
            return np.nan
        # Expected correct by chance
        a = (self.hits + self.misses) * (self.hits + self.false_alarms)
        b = (self.correct_negatives + self.misses) * (self.correct_negatives + self.false_alarms)
        expected = (a + b) / self.n
        correct = self.hits + self.correct_negatives
        if self.n == expected:
            return np.nan
        return (correct - expected) / (self.n - expected)

    @property
    def pss(self) -> float:
        """Peirce Skill Score (True Skill Statistic)."""
        obs_no = self.false_alarms + self.correct_negatives
        if self.observed_yes == 0 or obs_no == 0:
            return np.nan
        return self.pod - (self.false_alarms / obs_no)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "n": self.n,
            "pod": float(self.pod) if not np.isnan(self.pod) else None,
            "far": float(self.far) if not np.isnan(self.far) else None,
            "csi": float(self.csi) if not np.isnan(self.csi) else None,
            "bias_ratio": float(self.bias_ratio) if not np.isnan(self.bias_ratio) else None,
            "hss": float(self.hss) if not np.isnan(self.hss) else None,
            "pss": float(self.pss) if not np.isnan(self.pss) else None,
        }


@dataclass
class VariableMetrics:
    """
    Verification metrics for a single variable.

    Attributes:
        variable: Variable name
        continuous_metrics: Continuous verification metrics
        categorical_metrics: Categorical verification metrics
        contingency_table: Contingency table (if applicable)
        n_samples: Number of valid sample points
        lead_time_hours: Lead time this applies to (if specific)
    """

    variable: str
    continuous_metrics: Dict[str, float] = field(default_factory=dict)
    categorical_metrics: Dict[str, float] = field(default_factory=dict)
    contingency_table: Optional[ContingencyTable] = None
    n_samples: int = 0
    lead_time_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "continuous_metrics": self.continuous_metrics,
            "categorical_metrics": self.categorical_metrics,
            "contingency_table": self.contingency_table.to_dict() if self.contingency_table else None,
            "n_samples": self.n_samples,
            "lead_time_hours": self.lead_time_hours,
        }


@dataclass
class LeadTimePerformance:
    """
    Performance metrics as a function of lead time.

    Attributes:
        variable: Variable name
        lead_times_hours: List of lead times
        metrics: Dict of metric name -> list of values by lead time
    """

    variable: str
    lead_times_hours: List[float]
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def get_metric_at_lead(self, metric: str, lead_hours: float) -> Optional[float]:
        """Get metric value at specific lead time."""
        if metric not in self.metrics:
            return None
        try:
            idx = self.lead_times_hours.index(lead_hours)
            return self.metrics[metric][idx]
        except (ValueError, IndexError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "lead_times_hours": self.lead_times_hours,
            "metrics": self.metrics,
        }


@dataclass
class EnsembleMetrics:
    """
    Metrics specific to ensemble forecasts.

    Attributes:
        variable: Variable name
        spread: Ensemble spread (std across members)
        skill: Skill (RMSE of ensemble mean)
        spread_skill_ratio: Ratio of spread to skill
        crps: Continuous Ranked Probability Score
        reliability: Reliability diagram data
        rank_histogram: Rank histogram counts
    """

    variable: str
    spread: float = 0.0
    skill: float = 0.0
    spread_skill_ratio: float = 0.0
    crps: float = 0.0
    reliability: Optional[Dict[str, List[float]]] = None
    rank_histogram: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "spread": self.spread,
            "skill": self.skill,
            "spread_skill_ratio": self.spread_skill_ratio,
            "crps": self.crps,
            "reliability": self.reliability,
            "rank_histogram": self.rank_histogram,
        }


@dataclass
class SpatialVerification:
    """
    Spatial verification metrics.

    Attributes:
        variable: Variable name
        fss_by_scale: Fractions Skill Score by scale (scale_km -> FSS)
        sal: SAL components (structure, amplitude, location)
        displacement: Mean displacement error
        area_ratio: Ratio of forecast to observed area
    """

    variable: str
    fss_by_scale: Dict[float, float] = field(default_factory=dict)
    sal: Optional[Dict[str, float]] = None
    displacement_km: Optional[float] = None
    area_ratio: Optional[float] = None

    def get_fss(self, scale_km: float) -> Optional[float]:
        """Get FSS at specific scale."""
        return self.fss_by_scale.get(scale_km)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "fss_by_scale": self.fss_by_scale,
            "sal": self.sal,
            "displacement_km": self.displacement_km,
            "area_ratio": self.area_ratio,
        }


@dataclass
class ValidationConfig:
    """
    Configuration for forecast validation.

    Attributes:
        level: Level of validation detail
        variables: Variables to validate (None = all common)
        thresholds: Thresholds for binary metrics per variable
        spatial_scales_km: Scales for spatial verification
        compute_ensemble_metrics: Whether to compute ensemble metrics
        require_matching_grid: Whether to require exact grid matching
        interpolation_method: Method for grid interpolation
    """

    level: ValidationLevel = ValidationLevel.STANDARD
    variables: Optional[List[ForecastVariable]] = None
    thresholds: Dict[str, List[float]] = field(default_factory=dict)
    spatial_scales_km: List[float] = field(default_factory=lambda: [10, 25, 50, 100])
    compute_ensemble_metrics: bool = True
    require_matching_grid: bool = False
    interpolation_method: str = "nearest"

    def get_thresholds(self, var: str) -> List[float]:
        """Get thresholds for a variable."""
        if var in self.thresholds:
            return self.thresholds[var]
        # Default thresholds
        defaults = {
            "precipitation": [0.1, 1.0, 5.0, 10.0, 25.0],
            "accumulated_precip": [1.0, 5.0, 10.0, 25.0, 50.0],
            "wind_speed": [5.0, 10.0, 15.0, 20.0],
            "temperature": [273.15, 283.15, 293.15, 303.15],  # K
        }
        return defaults.get(var, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "variables": [v.value for v in self.variables] if self.variables else None,
            "thresholds": self.thresholds,
            "spatial_scales_km": self.spatial_scales_km,
            "compute_ensemble_metrics": self.compute_ensemble_metrics,
            "require_matching_grid": self.require_matching_grid,
            "interpolation_method": self.interpolation_method,
        }


@dataclass
class ValidationResult:
    """
    Complete validation result.

    Attributes:
        success: Whether validation completed successfully
        variable_metrics: Metrics per variable
        lead_time_performance: Lead time dependent performance
        ensemble_metrics: Ensemble-specific metrics
        spatial_verification: Spatial verification results
        overall_skill: Overall skill summary
        issues: List of issues encountered
        validation_time: When validation was performed
    """

    success: bool
    variable_metrics: Dict[str, VariableMetrics] = field(default_factory=dict)
    lead_time_performance: Dict[str, LeadTimePerformance] = field(default_factory=dict)
    ensemble_metrics: Dict[str, EnsembleMetrics] = field(default_factory=dict)
    spatial_verification: Dict[str, SpatialVerification] = field(default_factory=dict)
    overall_skill: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mean_rmse(self) -> Optional[float]:
        """Mean RMSE across all variables."""
        rmses = [
            m.continuous_metrics.get("rmse")
            for m in self.variable_metrics.values()
            if m.continuous_metrics.get("rmse") is not None
        ]
        return np.mean(rmses) if rmses else None

    @property
    def mean_correlation(self) -> Optional[float]:
        """Mean correlation across all variables."""
        corrs = [
            m.continuous_metrics.get("correlation")
            for m in self.variable_metrics.values()
            if m.continuous_metrics.get("correlation") is not None
        ]
        return np.mean(corrs) if corrs else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "variable_metrics": {k: v.to_dict() for k, v in self.variable_metrics.items()},
            "lead_time_performance": {k: v.to_dict() for k, v in self.lead_time_performance.items()},
            "ensemble_metrics": {k: v.to_dict() for k, v in self.ensemble_metrics.items()},
            "spatial_verification": {k: v.to_dict() for k, v in self.spatial_verification.items()},
            "overall_skill": self.overall_skill,
            "issues": self.issues,
            "validation_time": self.validation_time.isoformat(),
            "mean_rmse": self.mean_rmse,
            "mean_correlation": self.mean_correlation,
        }


class ForecastValidator:
    """
    Validates forecast data against observations.

    Provides comprehensive verification of forecast skill using
    standard meteorological verification metrics.

    Example:
        validator = ForecastValidator()
        result = validator.validate(forecast_dataset, observations)
        print(f"RMSE: {result.mean_rmse:.2f}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
    ) -> ValidationResult:
        """
        Validate forecast against observations.

        Args:
            forecast: Forecast dataset to validate
            observations: List of observations to compare against

        Returns:
            ValidationResult with all metrics
        """
        logger.info(f"Validating forecast from {forecast.metadata.provider.value}")

        issues = []
        variable_metrics = {}
        lead_time_performance = {}
        ensemble_metrics = {}
        spatial_verification = {}

        # Determine variables to validate
        variables = self.config.variables or [
            ForecastVariable(v) for v in forecast.variables
            if v in [fv.value for fv in ForecastVariable]
        ]

        # Match observations to forecast timesteps
        matched_pairs = self._match_times(forecast, observations)

        if not matched_pairs:
            issues.append("No matching forecast/observation times found")
            return ValidationResult(success=False, issues=issues)

        # Validate each variable
        for var in variables:
            var_name = var.value

            if var_name not in forecast.variables:
                issues.append(f"Variable {var_name} not in forecast")
                continue

            # Compute continuous metrics
            metrics = self._compute_continuous_metrics(
                forecast, observations, var_name, matched_pairs
            )
            variable_metrics[var_name] = metrics

            # Compute categorical metrics if thresholds defined
            thresholds = self.config.get_thresholds(var_name)
            if thresholds:
                cat_metrics = self._compute_categorical_metrics(
                    forecast, observations, var_name, matched_pairs, thresholds
                )
                metrics.categorical_metrics = cat_metrics

            # Lead time performance
            if self.config.level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                lt_perf = self._compute_lead_time_performance(
                    forecast, observations, var_name, matched_pairs
                )
                lead_time_performance[var_name] = lt_perf

            # Spatial verification
            if self.config.level == ValidationLevel.COMPREHENSIVE:
                spatial = self._compute_spatial_verification(
                    forecast, observations, var_name, matched_pairs
                )
                spatial_verification[var_name] = spatial

        # Ensemble metrics
        if (forecast.metadata.forecast_type == ForecastType.ENSEMBLE and
                self.config.compute_ensemble_metrics):
            for var in variables:
                ens_metrics = self._compute_ensemble_metrics(
                    forecast, observations, var.value, matched_pairs
                )
                if ens_metrics:
                    ensemble_metrics[var.value] = ens_metrics

        # Compute overall skill
        overall_skill = self._compute_overall_skill(variable_metrics)

        return ValidationResult(
            success=True,
            variable_metrics=variable_metrics,
            lead_time_performance=lead_time_performance,
            ensemble_metrics=ensemble_metrics,
            spatial_verification=spatial_verification,
            overall_skill=overall_skill,
            issues=issues,
        )

    def _match_times(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
    ) -> List[Tuple[ForecastTimestep, ObservationData]]:
        """Match forecast timesteps to observations."""
        matched = []
        obs_by_time = {obs.valid_time: obs for obs in observations}

        for ts in forecast.timesteps:
            if ts.valid_time in obs_by_time:
                matched.append((ts, obs_by_time[ts.valid_time]))
            else:
                # Find nearest within tolerance (30 minutes)
                for obs_time, obs in obs_by_time.items():
                    if abs((ts.valid_time - obs_time).total_seconds()) < 1800:
                        matched.append((ts, obs))
                        break

        return matched

    def _interpolate_to_grid(
        self,
        data: np.ndarray,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
    ) -> np.ndarray:
        """Interpolate data to a different grid."""
        from scipy.interpolate import RegularGridInterpolator

        # Handle 1D grids
        if src_lat.ndim == 1 and src_lon.ndim == 1:
            interp = RegularGridInterpolator(
                (src_lat, src_lon),
                data,
                method=self.config.interpolation_method,
                bounds_error=False,
                fill_value=np.nan,
            )

            if dst_lat.ndim == 1:
                dst_xx, dst_yy = np.meshgrid(dst_lat, dst_lon, indexing='ij')
            else:
                dst_xx, dst_yy = dst_lat, dst_lon

            points = np.column_stack([dst_xx.ravel(), dst_yy.ravel()])
            result = interp(points).reshape(dst_xx.shape)
            return result

        # 2D grids - need scattered interpolation
        from scipy.interpolate import griddata

        src_points = np.column_stack([src_lat.ravel(), src_lon.ravel()])
        src_values = data.ravel()

        if dst_lat.ndim == 1:
            dst_xx, dst_yy = np.meshgrid(dst_lat, dst_lon, indexing='ij')
        else:
            dst_xx, dst_yy = dst_lat, dst_lon

        dst_points = np.column_stack([dst_xx.ravel(), dst_yy.ravel()])
        result = griddata(src_points, src_values, dst_points, method='nearest')
        return result.reshape(dst_xx.shape)

    def _compute_continuous_metrics(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
        var_name: str,
        matched_pairs: List[Tuple[ForecastTimestep, ObservationData]],
    ) -> VariableMetrics:
        """Compute continuous verification metrics."""
        all_fcst = []
        all_obs = []

        for fc_ts, obs in matched_pairs:
            fc_data = fc_ts.get_variable(var_name)
            obs_data = obs.get_variable(var_name)

            if fc_data is None or obs_data is None:
                continue

            # Interpolate if grids don't match
            if fc_data.shape != obs_data.shape:
                if not self.config.require_matching_grid:
                    fc_data = self._interpolate_to_grid(
                        fc_data,
                        forecast.grid_lat, forecast.grid_lon,
                        obs.grid_lat, obs.grid_lon,
                    )
                else:
                    continue

            # Apply quality mask
            mask = obs.quality_mask
            if mask is not None and mask.shape == fc_data.shape:
                valid = mask & ~np.isnan(fc_data) & ~np.isnan(obs_data)
            else:
                valid = ~np.isnan(fc_data) & ~np.isnan(obs_data)

            all_fcst.extend(fc_data[valid].flatten())
            all_obs.extend(obs_data[valid].flatten())

        if not all_fcst:
            return VariableMetrics(variable=var_name, n_samples=0)

        fcst = np.array(all_fcst)
        obs = np.array(all_obs)

        # Compute metrics
        metrics = {}

        # RMSE
        metrics["rmse"] = float(np.sqrt(np.mean((fcst - obs) ** 2)))

        # MAE
        metrics["mae"] = float(np.mean(np.abs(fcst - obs)))

        # Bias
        metrics["bias"] = float(np.mean(fcst - obs))

        # Correlation
        if len(fcst) > 1 and np.std(fcst) > 0 and np.std(obs) > 0:
            metrics["correlation"] = float(np.corrcoef(fcst, obs)[0, 1])
        else:
            metrics["correlation"] = np.nan

        # Explained variance
        if np.var(obs) > 0:
            metrics["explained_variance"] = float(1 - np.var(fcst - obs) / np.var(obs))
        else:
            metrics["explained_variance"] = np.nan

        return VariableMetrics(
            variable=var_name,
            continuous_metrics=metrics,
            n_samples=len(fcst),
        )

    def _compute_categorical_metrics(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
        var_name: str,
        matched_pairs: List[Tuple[ForecastTimestep, ObservationData]],
        thresholds: List[float],
    ) -> Dict[str, float]:
        """Compute categorical verification metrics for each threshold."""
        metrics = {}

        for thresh in thresholds:
            table = ContingencyTable()

            for fc_ts, obs in matched_pairs:
                fc_data = fc_ts.get_variable(var_name)
                obs_data = obs.get_variable(var_name)

                if fc_data is None or obs_data is None:
                    continue

                # Interpolate if needed
                if fc_data.shape != obs_data.shape:
                    fc_data = self._interpolate_to_grid(
                        fc_data,
                        forecast.grid_lat, forecast.grid_lon,
                        obs.grid_lat, obs.grid_lon,
                    )

                # Apply quality mask
                mask = obs.quality_mask
                if mask is not None and mask.shape == fc_data.shape:
                    valid = mask
                else:
                    valid = np.ones(fc_data.shape, dtype=bool)

                fc_above = fc_data[valid] >= thresh
                obs_above = obs_data[valid] >= thresh

                table.hits += int(np.sum(fc_above & obs_above))
                table.misses += int(np.sum(~fc_above & obs_above))
                table.false_alarms += int(np.sum(fc_above & ~obs_above))
                table.correct_negatives += int(np.sum(~fc_above & ~obs_above))

            suffix = f"_{thresh}"
            metrics[f"pod{suffix}"] = float(table.pod) if not np.isnan(table.pod) else None
            metrics[f"far{suffix}"] = float(table.far) if not np.isnan(table.far) else None
            metrics[f"csi{suffix}"] = float(table.csi) if not np.isnan(table.csi) else None
            metrics[f"hss{suffix}"] = float(table.hss) if not np.isnan(table.hss) else None

        return metrics

    def _compute_lead_time_performance(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
        var_name: str,
        matched_pairs: List[Tuple[ForecastTimestep, ObservationData]],
    ) -> LeadTimePerformance:
        """Compute performance as function of lead time."""
        # Group by lead time
        by_lead: Dict[float, List[Tuple[ForecastTimestep, ObservationData]]] = {}

        for fc_ts, obs in matched_pairs:
            lead_hours = fc_ts.lead_hours
            # Round to nearest hour
            lead_hours = round(lead_hours)
            if lead_hours not in by_lead:
                by_lead[lead_hours] = []
            by_lead[lead_hours].append((fc_ts, obs))

        lead_times = sorted(by_lead.keys())
        metrics: Dict[str, List[float]] = {
            "rmse": [],
            "bias": [],
            "correlation": [],
        }

        for lead in lead_times:
            pairs = by_lead[lead]

            # Compute metrics for this lead time
            var_metrics = self._compute_continuous_metrics(
                forecast, observations, var_name, pairs
            )

            metrics["rmse"].append(var_metrics.continuous_metrics.get("rmse", np.nan))
            metrics["bias"].append(var_metrics.continuous_metrics.get("bias", np.nan))
            metrics["correlation"].append(var_metrics.continuous_metrics.get("correlation", np.nan))

        return LeadTimePerformance(
            variable=var_name,
            lead_times_hours=lead_times,
            metrics=metrics,
        )

    def _compute_spatial_verification(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
        var_name: str,
        matched_pairs: List[Tuple[ForecastTimestep, ObservationData]],
    ) -> SpatialVerification:
        """Compute spatial verification metrics."""
        fss_by_scale = {}

        # Compute FSS at different scales
        for scale_km in self.config.spatial_scales_km:
            fss_values = []

            for fc_ts, obs in matched_pairs:
                fc_data = fc_ts.get_variable(var_name)
                obs_data = obs.get_variable(var_name)

                if fc_data is None or obs_data is None:
                    continue

                # Interpolate if needed
                if fc_data.shape != obs_data.shape:
                    fc_data = self._interpolate_to_grid(
                        fc_data,
                        forecast.grid_lat, forecast.grid_lon,
                        obs.grid_lat, obs.grid_lon,
                    )

                # Compute FSS
                fss = self._fractions_skill_score(
                    fc_data, obs_data, scale_km,
                    forecast.metadata.spatial_resolution_m / 1000,
                )
                if not np.isnan(fss):
                    fss_values.append(fss)

            if fss_values:
                fss_by_scale[scale_km] = float(np.mean(fss_values))

        return SpatialVerification(
            variable=var_name,
            fss_by_scale=fss_by_scale,
        )

    def _fractions_skill_score(
        self,
        forecast: np.ndarray,
        observed: np.ndarray,
        scale_km: float,
        resolution_km: float,
    ) -> float:
        """
        Compute Fractions Skill Score.

        Args:
            forecast: Forecast field
            observed: Observed field
            scale_km: Scale for fraction computation (km)
            resolution_km: Grid resolution (km)

        Returns:
            FSS value (0-1), or NaN if computation fails
        """
        # Guard against empty or all-NaN arrays
        if forecast.size == 0 or observed.size == 0:
            return np.nan

        if np.all(np.isnan(observed)) or np.all(np.isnan(forecast)):
            return np.nan

        # Guard against zero/invalid resolution
        if resolution_km <= 0:
            return np.nan

        # Determine window size
        window = max(1, int(scale_km / resolution_km))

        # Simple threshold for binary conversion (median)
        threshold = np.nanmedian(observed)

        # Handle case where median is NaN
        if np.isnan(threshold):
            return np.nan

        # Binary fields - handle NaN values by treating them as False (0)
        fc_binary = np.where(np.isnan(forecast), 0.0, (forecast >= threshold).astype(float))
        ob_binary = np.where(np.isnan(observed), 0.0, (observed >= threshold).astype(float))

        # Compute fractions using uniform filter
        from scipy.ndimage import uniform_filter

        fc_frac = uniform_filter(fc_binary, size=window, mode='nearest')
        ob_frac = uniform_filter(ob_binary, size=window, mode='nearest')

        # FSS
        mse_fc = np.nanmean((fc_frac - ob_frac) ** 2)
        mse_ref = np.nanmean(fc_frac ** 2) + np.nanmean(ob_frac ** 2)

        # Handle NaN results
        if np.isnan(mse_fc) or np.isnan(mse_ref):
            return np.nan

        # Handle edge cases: both mse values are zero or very small
        if mse_ref < 1e-10:
            return 1.0 if mse_fc < 1e-10 else 0.0

        fss = 1 - mse_fc / mse_ref
        return float(np.clip(fss, 0.0, 1.0))  # Ensure result is in valid range

    def _compute_ensemble_metrics(
        self,
        forecast: ForecastDataset,
        observations: List[ObservationData],
        var_name: str,
        matched_pairs: List[Tuple[ForecastTimestep, ObservationData]],
    ) -> Optional[EnsembleMetrics]:
        """Compute ensemble-specific metrics."""
        # For ensemble forecasts, we need member data
        # This is a simplified implementation

        spreads = []
        skills = []

        for fc_ts, obs in matched_pairs:
            fc_data = fc_ts.get_variable(var_name)
            obs_data = obs.get_variable(var_name)

            if fc_data is None or obs_data is None:
                continue

            # Simplified: treat single member as ensemble mean
            # Real implementation would have member data
            skill = np.sqrt(np.nanmean((fc_data - obs_data) ** 2))
            spread = np.nanstd(fc_data) * 0.1  # Placeholder

            spreads.append(spread)
            skills.append(skill)

        if not spreads:
            return None

        mean_spread = float(np.mean(spreads))
        mean_skill = float(np.mean(skills))

        # Guard against division by zero
        if mean_skill > 1e-10:
            spread_skill_ratio = mean_spread / mean_skill
        else:
            spread_skill_ratio = np.nan

        return EnsembleMetrics(
            variable=var_name,
            spread=mean_spread,
            skill=mean_skill,
            spread_skill_ratio=spread_skill_ratio,
        )

    def _compute_overall_skill(
        self,
        variable_metrics: Dict[str, VariableMetrics],
    ) -> Dict[str, float]:
        """Compute overall skill summary."""
        skill = {}

        rmses = [m.continuous_metrics.get("rmse") for m in variable_metrics.values()
                 if m.continuous_metrics.get("rmse") is not None]
        if rmses:
            skill["mean_rmse"] = float(np.mean(rmses))

        corrs = [m.continuous_metrics.get("correlation") for m in variable_metrics.values()
                 if m.continuous_metrics.get("correlation") is not None]
        if corrs:
            skill["mean_correlation"] = float(np.mean(corrs))

        biases = [m.continuous_metrics.get("bias") for m in variable_metrics.values()
                  if m.continuous_metrics.get("bias") is not None]
        if biases:
            skill["mean_abs_bias"] = float(np.mean(np.abs(biases)))

        return skill


def validate_forecast(
    forecast: ForecastDataset,
    observations: List[ObservationData],
    level: ValidationLevel = ValidationLevel.STANDARD,
    variables: Optional[List[ForecastVariable]] = None,
) -> ValidationResult:
    """
    Convenience function to validate a forecast.

    Args:
        forecast: Forecast dataset to validate
        observations: Observations to compare against
        level: Validation detail level
        variables: Variables to validate (None = all)

    Returns:
        ValidationResult with all metrics
    """
    config = ValidationConfig(level=level, variables=variables)
    validator = ForecastValidator(config)
    return validator.validate(forecast, observations)
