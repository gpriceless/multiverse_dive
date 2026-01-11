"""
Comprehensive tests for the Forecast Integration Module (Group H, Track 4).

Tests cover:
- Forecast ingestion (ingestion.py)
- Forecast validation (validation.py)
- Scenario analysis (scenarios.py)
- Impact projection (projection.py)
"""

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_grid():
    """Create sample latitude/longitude grids."""
    grid_lat = np.linspace(24.0, 27.0, 30)
    grid_lon = np.linspace(-82.0, -79.0, 30)
    return grid_lat, grid_lon


@pytest.fixture
def sample_forecast_data(sample_grid):
    """Create sample forecast data arrays."""
    grid_lat, grid_lon = sample_grid
    n_lat, n_lon = len(grid_lat), len(grid_lon)

    # Create spatially varying precipitation data
    xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, n_lon), np.linspace(0, 2*np.pi, n_lat))
    precipitation = 20 + 15 * np.sin(xx) * np.cos(yy)
    wind_speed = 10 + 5 * np.cos(xx)
    temperature = 295 + 10 * np.sin(yy)
    humidity = np.clip(60 + 20 * np.sin(xx + yy), 0, 100)

    return {
        "precipitation": precipitation,
        "accumulated_precip": precipitation * 3,  # 3-hour accumulation
        "wind_speed": wind_speed,
        "temperature": temperature,
        "humidity": humidity,
    }


@pytest.fixture
def forecast_dataset(sample_grid, sample_forecast_data):
    """Create a complete ForecastDataset for testing."""
    from core.analysis.forecast.ingestion import (
        ForecastDataset,
        ForecastMetadata,
        ForecastTimestep,
        ForecastProvider,
        ForecastType,
        ForecastVariable,
    )

    grid_lat, grid_lon = sample_grid
    now = datetime.now(timezone.utc)
    init_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Create timesteps
    timesteps = []
    for i in range(8):  # 24 hours of 3-hourly data
        lead_time = timedelta(hours=i * 3)
        valid_time = init_time + lead_time

        # Vary data slightly per timestep
        data = {}
        for var_name, arr in sample_forecast_data.items():
            # Add temporal variation
            phase = i * 0.5
            data[var_name] = arr + np.random.randn(*arr.shape) * 2 + np.sin(phase) * 5

        timesteps.append(ForecastTimestep(
            valid_time=valid_time,
            lead_time=lead_time,
            data=data,
            quality_flags={var: 0.9 for var in data.keys()},
        ))

    metadata = ForecastMetadata(
        provider=ForecastProvider.GFS,
        forecast_type=ForecastType.DETERMINISTIC,
        initialization_time=init_time,
        variables=[
            ForecastVariable.PRECIPITATION,
            ForecastVariable.WIND_SPEED,
            ForecastVariable.TEMPERATURE,
        ],
        lead_times=[ts.lead_time for ts in timesteps],
        spatial_resolution_m=10000,
        temporal_resolution=timedelta(hours=3),
        bounds=(-82.0, 24.0, -79.0, 27.0),
    )

    return ForecastDataset(
        metadata=metadata,
        timesteps=timesteps,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        units={
            "precipitation": "mm",
            "accumulated_precip": "mm",
            "wind_speed": "m/s",
            "temperature": "K",
            "humidity": "%",
        },
    )


@pytest.fixture
def observation_data(sample_grid, sample_forecast_data):
    """Create observation data for validation tests."""
    from core.analysis.forecast.validation import ObservationData
    from datetime import datetime, timezone

    grid_lat, grid_lon = sample_grid
    now = datetime.now(timezone.utc)
    init_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

    observations = []
    for i in range(8):
        valid_time = init_time + timedelta(hours=i * 3)

        # Create observation data with some bias from forecast
        obs_data = {}
        for var_name, arr in sample_forecast_data.items():
            # Observations have some random error compared to "truth"
            obs_data[var_name] = arr + np.random.randn(*arr.shape) * 1.5

        observations.append(ObservationData(
            valid_time=valid_time,
            data=obs_data,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            source="test_observations",
        ))

    return observations


# ============================================================================
# Ingestion Tests
# ============================================================================

class TestForecastIngestion:
    """Tests for forecast data ingestion."""

    def test_forecast_provider_enum(self):
        """Test ForecastProvider enum values."""
        from core.analysis.forecast.ingestion import ForecastProvider

        assert ForecastProvider.GFS.value == "gfs"
        assert ForecastProvider.ECMWF_HRES.value == "ecmwf_hres"
        assert ForecastProvider.ERA5.value == "era5"

    def test_forecast_variable_enum(self):
        """Test ForecastVariable enum values."""
        from core.analysis.forecast.ingestion import ForecastVariable

        assert ForecastVariable.PRECIPITATION.value == "precipitation"
        assert ForecastVariable.WIND_SPEED.value == "wind_speed"
        assert ForecastVariable.TEMPERATURE.value == "temperature"

    def test_forecast_metadata_creation(self):
        """Test ForecastMetadata dataclass."""
        from core.analysis.forecast.ingestion import (
            ForecastMetadata,
            ForecastProvider,
            ForecastType,
            ForecastVariable,
        )

        now = datetime.now(timezone.utc)
        metadata = ForecastMetadata(
            provider=ForecastProvider.GFS,
            forecast_type=ForecastType.DETERMINISTIC,
            initialization_time=now,
            variables=[ForecastVariable.PRECIPITATION],
            lead_times=[timedelta(hours=0), timedelta(hours=3)],
            spatial_resolution_m=10000,
            temporal_resolution=timedelta(hours=3),
        )

        assert metadata.provider == ForecastProvider.GFS
        assert metadata.max_lead_time == timedelta(hours=3)
        assert len(metadata.valid_times()) == 2

    def test_forecast_metadata_timezone_handling(self):
        """Test that naive datetimes are converted to UTC."""
        from core.analysis.forecast.ingestion import (
            ForecastMetadata,
            ForecastProvider,
            ForecastType,
        )

        naive_time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone
        metadata = ForecastMetadata(
            provider=ForecastProvider.GFS,
            forecast_type=ForecastType.DETERMINISTIC,
            initialization_time=naive_time,
            variables=[],
            lead_times=[],
            spatial_resolution_m=10000,
            temporal_resolution=timedelta(hours=3),
        )

        assert metadata.initialization_time.tzinfo == timezone.utc

    def test_forecast_timestep_creation(self):
        """Test ForecastTimestep dataclass."""
        from core.analysis.forecast.ingestion import ForecastTimestep, ForecastVariable

        now = datetime.now(timezone.utc)
        data = {"precipitation": np.random.rand(10, 10)}

        ts = ForecastTimestep(
            valid_time=now,
            lead_time=timedelta(hours=6),
            data=data,
        )

        assert ts.lead_hours == 6.0
        assert ts.has_variable("precipitation")
        assert ts.has_variable(ForecastVariable.PRECIPITATION)
        assert not ts.has_variable("nonexistent")
        assert ts.get_variable("precipitation").shape == (10, 10)

    def test_forecast_dataset_creation(self, forecast_dataset):
        """Test ForecastDataset creation and properties."""
        assert len(forecast_dataset.timesteps) == 8
        assert len(forecast_dataset.variables) > 0
        assert forecast_dataset.shape == (30,)  # 1D lat grid

        time_range = forecast_dataset.time_range
        assert time_range[1] > time_range[0]

    def test_forecast_dataset_time_slice(self, forecast_dataset):
        """Test time slicing of forecast dataset."""
        start = forecast_dataset.timesteps[2].valid_time
        end = forecast_dataset.timesteps[5].valid_time

        sliced = forecast_dataset.slice_time(start, end)

        assert len(sliced.timesteps) <= len(forecast_dataset.timesteps)
        for ts in sliced.timesteps:
            assert start <= ts.valid_time <= end

    def test_forecast_dataset_spatial_slice(self, forecast_dataset):
        """Test spatial subsetting of forecast dataset."""
        # Subset to smaller region
        west, south, east, north = -81.0, 25.0, -80.0, 26.0
        sliced = forecast_dataset.slice_spatial(west, south, east, north)

        assert sliced.metadata.bounds == (west, south, east, north)
        assert len(sliced.timesteps) == len(forecast_dataset.timesteps)

    def test_forecast_dataset_get_timestep(self, forecast_dataset):
        """Test retrieving specific timestep."""
        target_time = forecast_dataset.timesteps[3].valid_time

        # Exact match
        ts = forecast_dataset.get_timestep(target_time)
        assert ts is not None
        assert ts.valid_time == target_time

        # Nearest match
        ts_nearest = forecast_dataset.get_timestep_nearest(
            target_time + timedelta(minutes=30)
        )
        assert ts_nearest is not None

    def test_forecast_ingester_synthetic(self):
        """Test ForecastIngester with synthetic data."""
        from core.analysis.forecast.ingestion import (
            ForecastIngester,
            ForecastIngestionConfig,
            ForecastProvider,
            ForecastVariable,
        )

        config = ForecastIngestionConfig(
            providers=[ForecastProvider.GFS],
            variables=[ForecastVariable.PRECIPITATION, ForecastVariable.WIND_SPEED],
            spatial_bounds=(-82, 24, -79, 27),
            max_lead_time_hours=24,
        )

        ingester = ForecastIngester()
        result = ingester.ingest(config)

        assert result.success
        assert result.dataset is not None
        assert result.provider_used == ForecastProvider.GFS
        assert len(result.errors) == 0

    def test_ingest_forecast_convenience_function(self):
        """Test the ingest_forecast convenience function."""
        from core.analysis.forecast.ingestion import (
            ingest_forecast,
            ForecastProvider,
            ForecastVariable,
        )

        result = ingest_forecast(
            providers=[ForecastProvider.GFS],
            variables=[ForecastVariable.PRECIPITATION],
            max_lead_hours=12,
        )

        assert result.success
        assert result.dataset is not None


class TestForecastIngestionEdgeCases:
    """Edge case tests for forecast ingestion."""

    def test_empty_variables_list(self):
        """Test handling of empty variables list."""
        from core.analysis.forecast.ingestion import (
            ForecastIngester,
            ForecastIngestionConfig,
            ForecastProvider,
        )

        config = ForecastIngestionConfig(
            providers=[ForecastProvider.GFS],
            variables=[],  # Empty
        )

        ingester = ForecastIngester()
        result = ingester.ingest(config)

        # Should still succeed, just with no variable data
        assert result.success

    def test_nonexistent_provider_fallback(self):
        """Test fallback when provider fails."""
        from core.analysis.forecast.ingestion import (
            ForecastIngester,
            ForecastIngestionConfig,
            ForecastProvider,
        )

        # ECMWF requires API key, so should fail and fall back to CUSTOM
        config = ForecastIngestionConfig(
            providers=[ForecastProvider.ECMWF_HRES, ForecastProvider.GFS],
        )

        ingester = ForecastIngester()
        result = ingester.ingest(config)

        # Should fall back to GFS
        assert result.success
        assert result.provider_used == ForecastProvider.GFS

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        from core.analysis.forecast.ingestion import (
            ForecastMetadata,
            ForecastProvider,
            ForecastType,
            ForecastVariable,
        )

        metadata = ForecastMetadata(
            provider=ForecastProvider.GFS,
            forecast_type=ForecastType.DETERMINISTIC,
            initialization_time=datetime.now(timezone.utc),
            variables=[ForecastVariable.PRECIPITATION],
            lead_times=[timedelta(hours=3)],
            spatial_resolution_m=10000,
            temporal_resolution=timedelta(hours=3),
        )

        d = metadata.to_dict()
        assert "provider" in d
        assert d["provider"] == "gfs"
        assert "lead_times_hours" in d
        assert d["lead_times_hours"] == [3.0]


# ============================================================================
# Validation Tests
# ============================================================================

class TestForecastValidation:
    """Tests for forecast validation."""

    def test_contingency_table_metrics(self):
        """Test ContingencyTable score calculations."""
        from core.analysis.forecast.validation import ContingencyTable

        # Perfect forecast
        table = ContingencyTable(hits=100, misses=0, false_alarms=0, correct_negatives=100)
        assert table.pod == 1.0
        assert table.far == 0.0
        assert table.csi == 1.0
        assert table.bias_ratio == 1.0

        # 50% hit rate
        table2 = ContingencyTable(hits=50, misses=50, false_alarms=50, correct_negatives=50)
        assert table2.pod == 0.5
        assert table2.far == 0.5

    def test_contingency_table_edge_cases(self):
        """Test ContingencyTable with edge cases."""
        from core.analysis.forecast.validation import ContingencyTable

        # No observed events
        table = ContingencyTable(hits=0, misses=0, false_alarms=10, correct_negatives=90)
        assert np.isnan(table.pod)  # Division by zero guarded
        assert table.far == 1.0

        # No forecast events
        table2 = ContingencyTable(hits=0, misses=10, false_alarms=0, correct_negatives=90)
        assert table2.pod == 0.0
        assert np.isnan(table2.far)  # Division by zero guarded

    def test_variable_metrics_creation(self):
        """Test VariableMetrics dataclass."""
        from core.analysis.forecast.validation import VariableMetrics

        metrics = VariableMetrics(
            variable="precipitation",
            continuous_metrics={"rmse": 5.0, "bias": -0.5, "correlation": 0.9},
            n_samples=1000,
        )

        assert metrics.variable == "precipitation"
        assert metrics.continuous_metrics["rmse"] == 5.0

    def test_forecast_validator_basic(self, forecast_dataset, observation_data):
        """Test basic forecast validation."""
        from core.analysis.forecast.validation import (
            ForecastValidator,
            ValidationConfig,
            ValidationLevel,
        )

        config = ValidationConfig(level=ValidationLevel.BASIC)
        validator = ForecastValidator(config)

        result = validator.validate(forecast_dataset, observation_data)

        assert result.success
        assert len(result.variable_metrics) > 0
        assert result.mean_rmse is not None

    def test_forecast_validator_standard(self, forecast_dataset, observation_data):
        """Test standard forecast validation with lead time performance."""
        from core.analysis.forecast.validation import (
            ForecastValidator,
            ValidationConfig,
            ValidationLevel,
        )

        config = ValidationConfig(level=ValidationLevel.STANDARD)
        validator = ForecastValidator(config)

        result = validator.validate(forecast_dataset, observation_data)

        assert result.success
        assert len(result.lead_time_performance) > 0

    def test_validate_forecast_convenience_function(self, forecast_dataset, observation_data):
        """Test the validate_forecast convenience function."""
        from core.analysis.forecast.validation import validate_forecast, ValidationLevel

        result = validate_forecast(
            forecast_dataset,
            observation_data,
            level=ValidationLevel.BASIC,
        )

        assert result.success


class TestForecastValidationEdgeCases:
    """Edge case tests for forecast validation."""

    def test_validation_with_no_matching_times(self, forecast_dataset):
        """Test validation when no times match."""
        from core.analysis.forecast.validation import (
            ForecastValidator,
            ValidationConfig,
            ObservationData,
        )

        # Create observations at non-matching times
        observations = [
            ObservationData(
                valid_time=datetime(2000, 1, 1, tzinfo=timezone.utc),  # Far in past
                data={"precipitation": np.random.rand(30, 30)},
                grid_lat=np.linspace(24, 27, 30),
                grid_lon=np.linspace(-82, -79, 30),
            )
        ]

        validator = ForecastValidator()
        result = validator.validate(forecast_dataset, observations)

        # Should fail gracefully
        assert not result.success
        assert len(result.issues) > 0

    def test_validation_with_nan_data(self, forecast_dataset, sample_grid):
        """Test validation with NaN values in data."""
        from core.analysis.forecast.validation import (
            ForecastValidator,
            ObservationData,
        )

        grid_lat, grid_lon = sample_grid

        # Create observation with NaN values
        data_with_nan = np.random.rand(30, 30)
        data_with_nan[10:15, 10:15] = np.nan  # Some NaN region

        observations = [
            ObservationData(
                valid_time=forecast_dataset.timesteps[0].valid_time,
                data={"precipitation": data_with_nan},
                grid_lat=grid_lat,
                grid_lon=grid_lon,
            )
        ]

        validator = ForecastValidator()
        result = validator.validate(forecast_dataset, observations)

        # Should handle NaN gracefully
        assert result.success or len(result.issues) > 0

    def test_fractions_skill_score_empty_array(self):
        """Test FSS with empty arrays."""
        from core.analysis.forecast.validation import ForecastValidator

        validator = ForecastValidator()

        # Empty arrays
        fss = validator._fractions_skill_score(
            np.array([]),
            np.array([]),
            scale_km=10.0,
            resolution_km=1.0,
        )

        assert np.isnan(fss)

    def test_fractions_skill_score_all_nan(self):
        """Test FSS with all-NaN arrays."""
        from core.analysis.forecast.validation import ForecastValidator

        validator = ForecastValidator()

        nan_array = np.full((10, 10), np.nan)
        fss = validator._fractions_skill_score(
            nan_array,
            nan_array,
            scale_km=10.0,
            resolution_km=1.0,
        )

        assert np.isnan(fss)

    def test_fractions_skill_score_zero_resolution(self):
        """Test FSS with zero resolution."""
        from core.analysis.forecast.validation import ForecastValidator

        validator = ForecastValidator()

        fss = validator._fractions_skill_score(
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            scale_km=10.0,
            resolution_km=0.0,  # Invalid
        )

        assert np.isnan(fss)


# ============================================================================
# Scenario Analysis Tests
# ============================================================================

class TestScenarioAnalysis:
    """Tests for scenario analysis."""

    def test_scenario_type_enum(self):
        """Test ScenarioType enum values."""
        from core.analysis.forecast.scenarios import ScenarioType

        assert ScenarioType.BEST_CASE.value == "best_case"
        assert ScenarioType.WORST_CASE.value == "worst_case"
        assert ScenarioType.MEDIAN.value == "median"

    def test_scenario_definition_creation(self):
        """Test ScenarioDefinition dataclass."""
        from core.analysis.forecast.scenarios import (
            ScenarioDefinition,
            ScenarioType,
            ImpactDirection,
        )

        definition = ScenarioDefinition(
            scenario_type=ScenarioType.PERCENTILE,
            name="p90_precip",
            percentile=90.0,
        )

        assert definition.percentile == 90.0

    def test_scenario_definition_validation(self):
        """Test ScenarioDefinition validation."""
        from core.analysis.forecast.scenarios import ScenarioDefinition, ScenarioType

        # Should raise ValueError for PERCENTILE without percentile value
        with pytest.raises(ValueError, match="percentile"):
            ScenarioDefinition(
                scenario_type=ScenarioType.PERCENTILE,
                name="invalid",
            )

        # Should raise ValueError for THRESHOLD without threshold value
        with pytest.raises(ValueError, match="threshold"):
            ScenarioDefinition(
                scenario_type=ScenarioType.THRESHOLD,
                name="invalid",
            )

    def test_scenario_analyzer_basic(self, forecast_dataset):
        """Test basic scenario analysis."""
        from core.analysis.forecast.scenarios import (
            ScenarioAnalyzer,
            ScenarioAnalysisConfig,
        )
        from core.analysis.forecast.ingestion import ForecastVariable

        config = ScenarioAnalysisConfig(
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [10.0, 25.0, 50.0]},
        )

        analyzer = ScenarioAnalyzer(config)
        result = analyzer.analyze(forecast_dataset)

        assert result.success
        assert len(result.scenarios) > 0

    def test_scenario_analyzer_exceedance(self, forecast_dataset):
        """Test threshold exceedance analysis."""
        from core.analysis.forecast.scenarios import (
            ScenarioAnalyzer,
            ScenarioAnalysisConfig,
        )
        from core.analysis.forecast.ingestion import ForecastVariable

        config = ScenarioAnalysisConfig(
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [10.0, 25.0]},
        )

        analyzer = ScenarioAnalyzer(config)
        result = analyzer.analyze(forecast_dataset)

        assert "precipitation" in result.exceedance
        assert len(result.exceedance["precipitation"]) == 2

    def test_analyze_scenarios_convenience_function(self, forecast_dataset):
        """Test the analyze_scenarios convenience function."""
        from core.analysis.forecast.scenarios import analyze_scenarios
        from core.analysis.forecast.ingestion import ForecastVariable

        result = analyze_scenarios(
            forecast_dataset,
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [10.0]},
        )

        assert result.success

    def test_what_if_scenario_generation(self, forecast_dataset):
        """Test what-if scenario generation."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer()

        # Double precipitation scenario
        scenario = analyzer.generate_what_if_scenario(
            forecast_dataset,
            modifications={"precipitation": lambda x: x * 2.0},
            name="double_precip",
            description="Test scenario with doubled precipitation",
        )

        assert scenario.definition.name == "double_precip"
        assert len(scenario.timesteps) == len(forecast_dataset.timesteps)

    def test_sensitivity_analysis(self, forecast_dataset):
        """Test sensitivity analysis."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer()

        results = analyzer.sensitivity_analysis(
            forecast_dataset,
            response_var="precipitation",
            parameter_vars=["temperature", "humidity"],
        )

        # Should return results for parameters that exist
        assert isinstance(results, list)


class TestScenarioAnalysisEdgeCases:
    """Edge case tests for scenario analysis."""

    def test_exceedance_with_empty_data(self):
        """Test exceedance computation with no data."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer
        from core.analysis.forecast.ingestion import (
            ForecastDataset,
            ForecastMetadata,
            ForecastProvider,
            ForecastType,
        )

        # Create empty forecast
        metadata = ForecastMetadata(
            provider=ForecastProvider.GFS,
            forecast_type=ForecastType.DETERMINISTIC,
            initialization_time=datetime.now(timezone.utc),
            variables=[],
            lead_times=[],
            spatial_resolution_m=10000,
            temporal_resolution=timedelta(hours=3),
        )

        empty_forecast = ForecastDataset(
            metadata=metadata,
            timesteps=[],
            grid_lat=np.array([]),
            grid_lon=np.array([]),
        )

        analyzer = ScenarioAnalyzer()
        exceedance = analyzer._compute_exceedance(empty_forecast, "precipitation", 10.0)

        # Should return empty result
        assert exceedance.probability_field.size == 0 or exceedance.probability_field.shape == (0, 0)

    def test_exceedance_duration_1d(self):
        """Test exceedance duration with 1D data."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer()

        # 1D boolean array
        mask = np.array([True, True, False, True, True, True, False, False])
        duration = analyzer._compute_exceedance_duration(mask)

        assert duration.size > 0

    def test_exceedance_duration_2d(self):
        """Test exceedance duration with 2D data."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer()

        # 2D boolean array (time, space)
        mask = np.array([
            [True, False, True],
            [True, True, False],
            [False, True, True],
        ])
        duration = analyzer._compute_exceedance_duration(mask)

        assert duration.shape == (3,)  # One per spatial point

    def test_exceedance_duration_empty(self):
        """Test exceedance duration with empty data."""
        from core.analysis.forecast.scenarios import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer()

        mask = np.array([])
        duration = analyzer._compute_exceedance_duration(mask)

        assert duration.size == 0


# ============================================================================
# Impact Projection Tests
# ============================================================================

class TestImpactProjection:
    """Tests for impact projection."""

    def test_hazard_type_enum(self):
        """Test HazardType enum values."""
        from core.analysis.forecast.projection import HazardType

        assert HazardType.FLOOD.value == "flood"
        assert HazardType.WILDFIRE.value == "wildfire"
        assert HazardType.STORM.value == "storm"

    def test_impact_severity_levels(self):
        """Test ImpactSeverity level ordering."""
        from core.analysis.forecast.projection import ImpactSeverity

        assert ImpactSeverity.NONE.level == 0
        assert ImpactSeverity.MINOR.level == 1
        assert ImpactSeverity.CATASTROPHIC.level == 6

        # Verify ordering
        assert ImpactSeverity.MODERATE.level < ImpactSeverity.SEVERE.level

    def test_impact_threshold_matching(self):
        """Test ImpactThreshold value matching."""
        from core.analysis.forecast.projection import ImpactThreshold, ImpactSeverity

        threshold = ImpactThreshold(
            variable="precipitation",
            min_value=25.0,
            max_value=50.0,
            severity=ImpactSeverity.MODERATE,
        )

        assert threshold.matches(25.0)  # min is inclusive
        assert threshold.matches(35.0)
        assert not threshold.matches(50.0)  # max is exclusive
        assert not threshold.matches(10.0)

    def test_impact_model_get_severity(self):
        """Test ImpactModel severity lookup."""
        from core.analysis.forecast.projection import (
            ImpactModel,
            ImpactThreshold,
            ImpactSeverity,
            HazardType,
        )

        model = ImpactModel(
            hazard_type=HazardType.FLOOD,
            name="Test Model",
            variables=["precipitation"],
            thresholds=[
                ImpactThreshold("precipitation", 0, 10, ImpactSeverity.NONE),
                ImpactThreshold("precipitation", 10, 25, ImpactSeverity.MINOR),
                ImpactThreshold("precipitation", 25, 50, ImpactSeverity.MODERATE),
            ],
        )

        assert model.get_severity(5.0, "precipitation") == ImpactSeverity.NONE
        assert model.get_severity(15.0, "precipitation") == ImpactSeverity.MINOR
        assert model.get_severity(30.0, "precipitation") == ImpactSeverity.MODERATE

    def test_impact_projector_basic(self, forecast_dataset):
        """Test basic impact projection."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        config = ProjectionConfig(
            hazard_types=[HazardType.FLOOD],
            cell_area_km2=1.0,
        )

        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        assert result.success
        assert HazardType.FLOOD in result.projections

    def test_impact_projector_multiple_hazards(self, forecast_dataset):
        """Test projection with multiple hazard types."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        config = ProjectionConfig(
            hazard_types=[HazardType.FLOOD, HazardType.STORM],
            cell_area_km2=1.0,
        )

        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        assert result.success
        assert len(result.projections) >= 1  # At least one should succeed

    def test_impact_projector_compound(self, forecast_dataset):
        """Test compound impact assessment."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        config = ProjectionConfig(
            hazard_types=[HazardType.FLOOD, HazardType.STORM],
            compute_compound=True,
        )

        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        assert result.success
        # Compound should be computed if multiple hazards projected
        if len(result.projections) > 1:
            assert result.compound is not None

    def test_project_impacts_convenience_function(self, forecast_dataset):
        """Test the project_impacts convenience function."""
        from core.analysis.forecast.projection import project_impacts, HazardType

        result = project_impacts(
            forecast_dataset,
            hazards=[HazardType.FLOOD],
            cell_area_km2=1.0,
        )

        assert result.success

    def test_get_impact_model(self):
        """Test get_impact_model function."""
        from core.analysis.forecast.projection import get_impact_model, HazardType

        flood_model = get_impact_model(HazardType.FLOOD)
        assert flood_model is not None
        assert flood_model.hazard_type == HazardType.FLOOD

        storm_model = get_impact_model(HazardType.STORM)
        assert storm_model is not None

    def test_create_custom_impact_model(self):
        """Test creating custom impact models."""
        from core.analysis.forecast.projection import (
            create_custom_impact_model,
            HazardType,
            ImpactSeverity,
        )

        model = create_custom_impact_model(
            hazard=HazardType.FLOOD,
            name="Custom Flood Model",
            variables=["precipitation"],
            thresholds=[
                ("precipitation", 0, 50, ImpactSeverity.MINOR),
                ("precipitation", 50, 100, ImpactSeverity.SEVERE),
            ],
            aggregation="sum",
        )

        assert model.name == "Custom Flood Model"
        assert len(model.thresholds) == 2


class TestImpactProjectionEdgeCases:
    """Edge case tests for impact projection."""

    def test_impact_timestep_severity_distribution(self):
        """Test severity distribution calculation."""
        from core.analysis.forecast.projection import ImpactTimestep, ImpactSeverity

        severity_field = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        confidence_field = np.ones((3, 3))

        ts = ImpactTimestep(
            valid_time=datetime.now(timezone.utc),
            severity_field=severity_field,
            confidence_field=confidence_field,
        )

        dist = ts.get_severity_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_impact_projection_get_cumulative(self, forecast_dataset):
        """Test cumulative severity calculation."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        config = ProjectionConfig(hazard_types=[HazardType.FLOOD])
        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        if result.success and HazardType.FLOOD in result.projections:
            projection = result.projections[HazardType.FLOOD]
            cumulative = projection.get_cumulative_severity()

            assert cumulative.size > 0 or len(projection.timesteps) == 0

    def test_impact_projection_evolution(self, forecast_dataset):
        """Test impact evolution time series."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        config = ProjectionConfig(hazard_types=[HazardType.FLOOD])
        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        if result.success and HazardType.FLOOD in result.projections:
            projection = result.projections[HazardType.FLOOD]
            evolution = projection.get_impact_evolution()

            assert "times" in evolution
            assert "peak_severity" in evolution
            assert len(evolution["times"]) == len(projection.timesteps)

    def test_projection_with_missing_variables(self, forecast_dataset):
        """Test projection when required variables are missing."""
        from core.analysis.forecast.projection import (
            ImpactProjector,
            ProjectionConfig,
            HazardType,
        )

        # Wildfire model needs humidity which might not be present
        config = ProjectionConfig(hazard_types=[HazardType.WILDFIRE])
        projector = ImpactProjector(config)
        result = projector.project(forecast_dataset)

        # Should handle missing variables gracefully
        assert result.success or len(result.issues) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestForecastIntegration:
    """Integration tests for complete forecast workflow."""

    def test_end_to_end_workflow(self, sample_grid):
        """Test complete forecast-to-impact workflow."""
        from core.analysis.forecast import (
            ingest_forecast,
            validate_forecast,
            analyze_scenarios,
            project_impacts,
            ForecastProvider,
            ForecastVariable,
            HazardType,
        )
        from core.analysis.forecast.validation import ObservationData, ValidationLevel

        # Step 1: Ingest forecast
        ingest_result = ingest_forecast(
            providers=[ForecastProvider.GFS],
            variables=[ForecastVariable.PRECIPITATION, ForecastVariable.WIND_SPEED],
            spatial_bounds=(-82, 24, -79, 27),
            max_lead_hours=24,
        )

        assert ingest_result.success
        forecast = ingest_result.dataset

        # Step 2: Create synthetic observations
        grid_lat, grid_lon = sample_grid
        observations = []
        for ts in forecast.timesteps[:3]:  # Just first few
            obs_data = {}
            for var in forecast.variables:
                fc_data = ts.get_variable(var)
                if fc_data is not None:
                    obs_data[var] = fc_data + np.random.randn(*fc_data.shape) * 2

            observations.append(ObservationData(
                valid_time=ts.valid_time,
                data=obs_data,
                grid_lat=forecast.grid_lat,
                grid_lon=forecast.grid_lon,
            ))

        # Step 3: Validate forecast
        val_result = validate_forecast(forecast, observations, level=ValidationLevel.BASIC)
        assert val_result.success

        # Step 4: Analyze scenarios
        scenario_result = analyze_scenarios(
            forecast,
            variables=[ForecastVariable.PRECIPITATION],
            thresholds={"precipitation": [10, 25]},
        )
        assert scenario_result.success

        # Step 5: Project impacts
        impact_result = project_impacts(
            forecast,
            hazards=[HazardType.FLOOD],
        )
        assert impact_result.success

    def test_module_exports(self):
        """Test that all expected exports are available."""
        from core.analysis.forecast import (
            # Ingestion
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
            # Validation
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
            # Scenarios
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
            # Projection
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

        # Just verify imports work
        assert ForecastProvider is not None
        assert HazardType is not None


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for data serialization (to_dict methods)."""

    def test_ingestion_result_to_dict(self):
        """Test ForecastIngestionResult serialization."""
        from core.analysis.forecast import ingest_forecast, ForecastProvider

        result = ingest_forecast(providers=[ForecastProvider.GFS], max_lead_hours=6)
        d = result.to_dict()

        assert "success" in d
        assert "provider_used" in d
        assert "errors" in d

    def test_validation_result_to_dict(self, forecast_dataset, observation_data):
        """Test ValidationResult serialization."""
        from core.analysis.forecast import validate_forecast, ValidationLevel

        result = validate_forecast(forecast_dataset, observation_data, level=ValidationLevel.BASIC)
        d = result.to_dict()

        assert "success" in d
        assert "variable_metrics" in d
        assert "overall_skill" in d

    def test_scenario_result_to_dict(self, forecast_dataset):
        """Test ScenarioAnalysisResult serialization."""
        from core.analysis.forecast import analyze_scenarios, ForecastVariable

        result = analyze_scenarios(
            forecast_dataset,
            variables=[ForecastVariable.PRECIPITATION],
        )
        d = result.to_dict()

        assert "success" in d
        assert "scenarios" in d
        assert "summary" in d

    def test_projection_result_to_dict(self, forecast_dataset):
        """Test ProjectionResult serialization."""
        from core.analysis.forecast import project_impacts, HazardType

        result = project_impacts(forecast_dataset, hazards=[HazardType.FLOOD])
        d = result.to_dict()

        assert "success" in d
        assert "projections" in d
        assert "max_severity" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
