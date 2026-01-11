"""
Tests for fusion strategy.

Tests multi-sensor blending rules, complementary vs redundant strategies,
and temporal densification for improved time series coverage.
"""

import pytest
from datetime import datetime, timedelta, timezone

from core.data.discovery.base import DiscoveryResult
from core.data.selection.fusion import (
    FusionStrategy,
    SensorRole,
    BlendingMethod,
    SensorContribution,
    FusionConfiguration,
    FusionPlan,
    TemporalGap,
    FusionStrategyEngine,
    determine_fusion_strategy,
    create_fusion_plan,
    identify_temporal_gaps,
)


class TestFusionEnums:
    """Test fusion enum definitions."""

    def test_fusion_strategy_values(self):
        """Test FusionStrategy enum values."""
        assert FusionStrategy.COMPLEMENTARY.value == "complementary"
        assert FusionStrategy.REDUNDANT.value == "redundant"
        assert FusionStrategy.TEMPORAL.value == "temporal"
        assert FusionStrategy.HIERARCHICAL.value == "hierarchical"
        assert FusionStrategy.ENSEMBLE.value == "ensemble"

    def test_sensor_role_values(self):
        """Test SensorRole enum values."""
        assert SensorRole.PRIMARY.value == "primary"
        assert SensorRole.SECONDARY.value == "secondary"
        assert SensorRole.FALLBACK.value == "fallback"
        assert SensorRole.GAP_FILL.value == "gap_fill"
        assert SensorRole.VALIDATION.value == "validation"

    def test_blending_method_values(self):
        """Test BlendingMethod enum values."""
        assert BlendingMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert BlendingMethod.QUALITY_MOSAIC.value == "quality_mosaic"
        assert BlendingMethod.TEMPORAL_COMPOSITE.value == "temporal_composite"
        assert BlendingMethod.CONSENSUS.value == "consensus"
        assert BlendingMethod.PRIORITY_STACK.value == "priority_stack"
        assert BlendingMethod.KALMAN_FILTER.value == "kalman_filter"


class TestSensorContribution:
    """Test sensor contribution dataclass."""

    def test_basic_creation(self):
        """Test creating a basic sensor contribution."""
        contrib = SensorContribution(
            sensor_type="sar",
            role=SensorRole.PRIMARY,
            weight=0.8
        )

        assert contrib.sensor_type == "sar"
        assert contrib.role == SensorRole.PRIMARY
        assert contrib.weight == 0.8
        assert contrib.provides == []
        assert contrib.requirements == {}

    def test_full_creation(self):
        """Test creating a full sensor contribution."""
        contrib = SensorContribution(
            sensor_type="optical",
            role=SensorRole.SECONDARY,
            weight=0.5,
            provides=["burn_severity", "vegetation_index"],
            requirements={"max_cloud_cover": 30},
            fallback_for=["sar"]
        )

        assert contrib.provides == ["burn_severity", "vegetation_index"]
        assert contrib.requirements == {"max_cloud_cover": 30}
        assert contrib.fallback_for == ["sar"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        contrib = SensorContribution(
            sensor_type="sar",
            role=SensorRole.PRIMARY,
            weight=0.8,
            provides=["water_extent"]
        )

        data = contrib.to_dict()

        assert data["sensor_type"] == "sar"
        assert data["role"] == "primary"
        assert data["weight"] == 0.8
        assert data["provides"] == ["water_extent"]


class TestFusionConfiguration:
    """Test fusion configuration dataclass."""

    def test_basic_configuration(self):
        """Test creating a basic fusion configuration."""
        config = FusionConfiguration(
            name="test_config",
            strategy=FusionStrategy.COMPLEMENTARY,
            sensors=[
                SensorContribution(sensor_type="sar", role=SensorRole.PRIMARY),
                SensorContribution(sensor_type="optical", role=SensorRole.SECONDARY),
            ]
        )

        assert config.name == "test_config"
        assert config.strategy == FusionStrategy.COMPLEMENTARY
        assert len(config.sensors) == 2
        assert config.blending_method == BlendingMethod.WEIGHTED_AVERAGE

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = FusionConfiguration(
            name="test_config",
            strategy=FusionStrategy.REDUNDANT,
            sensors=[
                SensorContribution(sensor_type="sar", role=SensorRole.PRIMARY),
            ],
            blending_method=BlendingMethod.CONSENSUS,
            confidence_threshold=0.7
        )

        data = config.to_dict()

        assert data["name"] == "test_config"
        assert data["strategy"] == "redundant"
        assert data["blending_method"] == "consensus"
        assert data["confidence_threshold"] == 0.7


class TestTemporalGap:
    """Test temporal gap dataclass."""

    def test_basic_gap(self):
        """Test creating a basic temporal gap."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=12)

        gap = TemporalGap(
            start=start,
            end=end,
            duration_hours=12.0
        )

        assert gap.duration_hours == 12.0
        assert gap.sensor_types_available == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)  # Next day

        gap = TemporalGap(
            start=start,
            end=end,
            duration_hours=24.0,
            sensor_types_available=["sar", "optical"]
        )

        data = gap.to_dict()

        assert "start" in data
        assert "end" in data
        assert data["duration_hours"] == 24.0
        assert data["sensor_types_available"] == ["sar", "optical"]


class TestFusionStrategyEngine:
    """Test fusion strategy engine."""

    @pytest.fixture
    def engine(self):
        """Create fusion strategy engine."""
        return FusionStrategyEngine()

    @pytest.fixture
    def sample_candidates(self):
        """Create sample discovery result candidates."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        return [
            DiscoveryResult(
                dataset_id="sar_001",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=95.0,
                resolution_m=10.0,
                quality_flag="good"
            ),
            DiscoveryResult(
                dataset_id="optical_001",
                provider="sentinel2",
                data_type="optical",
                source_uri="s3://bucket/optical_001.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=2),
                spatial_coverage_percent=90.0,
                resolution_m=10.0,
                cloud_cover_percent=15.0,
                quality_flag="good"
            ),
            DiscoveryResult(
                dataset_id="dem_001",
                provider="copernicus",
                data_type="dem",
                source_uri="s3://bucket/dem_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100.0,
                resolution_m=30.0,
                quality_flag="excellent"
            ),
        ]

    def test_default_configurations_exist(self, engine):
        """Test that default configurations are initialized."""
        assert "flood_complementary" in engine.configurations
        assert "flood_sar_only" in engine.configurations
        assert "wildfire_complementary" in engine.configurations
        assert "storm_redundant" in engine.configurations
        assert "temporal_densification" in engine.configurations
        assert "ensemble_consensus" in engine.configurations

    def test_get_configuration(self, engine):
        """Test getting a configuration by name."""
        config = engine.get_configuration("flood_complementary")

        assert config is not None
        assert config.name == "flood_complementary"
        assert config.strategy == FusionStrategy.COMPLEMENTARY

    def test_get_nonexistent_configuration(self, engine):
        """Test getting a non-existent configuration."""
        config = engine.get_configuration("nonexistent")

        assert config is None

    def test_determine_strategy_flood_clear(self, engine):
        """Test strategy determination for flood with clear conditions."""
        config = engine.determine_strategy(
            event_class="flood.riverine",
            available_sensors={"sar", "optical"},
            atmospheric_conditions={"cloud_cover_percent": 10}
        )

        assert config.name == "flood_complementary"
        assert config.strategy == FusionStrategy.COMPLEMENTARY

    def test_determine_strategy_flood_cloudy(self, engine):
        """Test strategy determination for flood with cloudy conditions."""
        config = engine.determine_strategy(
            event_class="flood.coastal",
            available_sensors={"sar", "optical"},
            atmospheric_conditions={"cloud_cover_percent": 90}
        )

        assert config.name == "flood_sar_only"
        assert config.strategy == FusionStrategy.HIERARCHICAL

    def test_determine_strategy_wildfire(self, engine):
        """Test strategy determination for wildfire."""
        config = engine.determine_strategy(
            event_class="wildfire.forest",
            available_sensors={"thermal", "optical", "sar"},
            atmospheric_conditions={}
        )

        assert config.name == "wildfire_complementary"

    def test_determine_strategy_storm(self, engine):
        """Test strategy determination for storm."""
        config = engine.determine_strategy(
            event_class="storm.hurricane",
            available_sensors={"sar", "optical"},
            atmospheric_conditions={}
        )

        assert config.name == "storm_redundant"
        assert config.strategy == FusionStrategy.REDUNDANT

    def test_determine_strategy_unknown_event(self, engine):
        """Test strategy determination for unknown event class."""
        config = engine.determine_strategy(
            event_class="earthquake.magnitude7",
            available_sensors={"sar", "optical"},
            atmospheric_conditions={}
        )

        # Should fall back to temporal densification
        assert config.name == "temporal_densification"

    def test_create_fusion_plan(self, engine, sample_candidates):
        """Test creating a fusion plan."""
        config = engine.get_configuration("flood_complementary")

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=sample_candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        assert isinstance(plan, FusionPlan)
        assert plan.configuration == config
        assert "sar" in plan.datasets
        assert len(plan.blending_weights) > 0
        assert 0.0 <= plan.confidence <= 1.0
        assert len(plan.rationale) > 0

    def test_fusion_plan_to_dict(self, engine, sample_candidates):
        """Test fusion plan conversion to dictionary."""
        config = engine.get_configuration("flood_complementary")

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=sample_candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        data = plan.to_dict()

        assert "configuration" in data
        assert "datasets" in data
        assert "blending_weights" in data
        assert "temporal_coverage" in data
        assert "spatial_coverage" in data
        assert "confidence" in data
        assert "rationale" in data

    def test_is_complementary(self, engine):
        """Test complementary sensor pair detection."""
        assert engine.is_complementary("sar", "optical") is True
        assert engine.is_complementary("optical", "sar") is True
        assert engine.is_complementary("thermal", "optical") is True
        assert engine.is_complementary("sar", "dem") is True
        assert engine.is_complementary("sar", "sar") is False
        assert engine.is_complementary("optical", "optical") is False

    def test_is_redundant(self, engine):
        """Test redundant sensor pair detection."""
        assert engine.is_redundant("sar", "sar") is True
        assert engine.is_redundant("optical", "optical") is True
        assert engine.is_redundant("thermal", "thermal") is True
        assert engine.is_redundant("sar", "optical") is False


class TestTemporalGapIdentification:
    """Test temporal gap identification."""

    @pytest.fixture
    def engine(self):
        """Create fusion strategy engine."""
        return FusionStrategyEngine()

    def test_identify_gaps_no_data(self, engine):
        """Test gap identification with no data."""
        gaps = engine.identify_temporal_gaps(
            datasets={},
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        # Empty dataset means no acquisitions to compare
        assert len(gaps) == 0

    def test_identify_gaps_single_acquisition(self, engine):
        """Test gap identification with single acquisition."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
            ]
        }

        gaps = engine.identify_temporal_gaps(
            datasets=datasets,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            },
            max_gap_hours=6.0
        )

        # Should have gaps at start and end
        assert len(gaps) >= 1

    def test_identify_gaps_dense_coverage(self, engine):
        """Test gap identification with dense temporal coverage."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        # Acquisitions every 4 hours
        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id=f"sar_{i}",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri=f"s3://bucket/sar_{i}.tif",
                    format="cog",
                    acquisition_time=base_time + timedelta(hours=i*4),
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
                for i in range(6)  # 0, 4, 8, 12, 16, 20 hours
            ]
        }

        gaps = engine.identify_temporal_gaps(
            datasets=datasets,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            },
            max_gap_hours=6.0  # No gaps should exceed 6 hours
        )

        # All gaps should be <= 4 hours (within threshold)
        assert len(gaps) == 0

    def test_identify_gaps_sparse_coverage(self, engine):
        """Test gap identification with sparse temporal coverage."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                ),
                DiscoveryResult(
                    dataset_id="sar_002",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_002.tif",
                    format="cog",
                    acquisition_time=base_time + timedelta(hours=48),  # 48-hour gap
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                ),
            ]
        }

        gaps = engine.identify_temporal_gaps(
            datasets=datasets,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-17T00:00:00Z"
            },
            max_gap_hours=12.0
        )

        # Should identify the 48-hour gap between acquisitions
        assert len(gaps) >= 1
        assert any(g.duration_hours > 24 for g in gaps)


class TestTemporalDensification:
    """Test temporal densification."""

    @pytest.fixture
    def engine(self):
        """Create fusion strategy engine."""
        return FusionStrategyEngine()

    def test_densify_empty_gap_fill(self, engine):
        """Test densification with no gap-fill data."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        primary = [
            DiscoveryResult(
                dataset_id="sar_001",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            ),
            DiscoveryResult(
                dataset_id="sar_002",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_002.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=48),
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            ),
        ]

        result = engine.densify_temporal_coverage(
            primary_datasets=primary,
            gap_fill_datasets=[],
            max_gap_hours=12.0
        )

        # No gap fill available
        assert len(result) == 2

    def test_densify_with_gap_fill(self, engine):
        """Test densification with gap-fill data available."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        primary = [
            DiscoveryResult(
                dataset_id="sar_001",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            ),
            DiscoveryResult(
                dataset_id="sar_002",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_002.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=48),
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            ),
        ]

        gap_fill = [
            DiscoveryResult(
                dataset_id="optical_001",
                provider="sentinel2",
                data_type="optical",
                source_uri="s3://bucket/optical_001.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=24),  # In the gap
                spatial_coverage_percent=90.0,
                resolution_m=10.0,
                cloud_cover_percent=10.0
            ),
        ]

        result = engine.densify_temporal_coverage(
            primary_datasets=primary,
            gap_fill_datasets=gap_fill,
            max_gap_hours=12.0
        )

        # Should include the gap-fill dataset
        assert len(result) == 3
        assert any(d.dataset_id == "optical_001" for d in result)


class TestBlendingWeightComputation:
    """Test blending weight computation."""

    @pytest.fixture
    def engine(self):
        """Create fusion strategy engine."""
        return FusionStrategyEngine()

    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets grouped by type."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        return {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0,
                    quality_flag="excellent"
                ),
            ],
            "optical": [
                DiscoveryResult(
                    dataset_id="optical_001",
                    provider="sentinel2",
                    data_type="optical",
                    source_uri="s3://bucket/optical_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=80.0,
                    resolution_m=10.0,
                    cloud_cover_percent=20.0,
                    quality_flag="good"
                ),
            ]
        }

    def test_weighted_average_weights(self, engine, sample_datasets):
        """Test weighted average blending weights."""
        weights = engine.compute_blending_weights(
            datasets=sample_datasets,
            method=BlendingMethod.WEIGHTED_AVERAGE,
            base_weights={"sar": 0.6, "optical": 0.4}
        )

        assert "sar" in weights
        assert "optical" in weights
        assert "sar_001" in weights["sar"]
        assert "optical_001" in weights["optical"]

        # Check normalization within sensor type
        for sensor_type in weights:
            total = sum(weights[sensor_type].values())
            assert abs(total - 1.0) < 0.01

    def test_priority_stack_weights(self, engine, sample_datasets):
        """Test priority stack blending weights."""
        weights = engine.compute_blending_weights(
            datasets=sample_datasets,
            method=BlendingMethod.PRIORITY_STACK
        )

        # First dataset should have higher weight
        assert weights["sar"]["sar_001"] >= weights["optical"]["optical_001"]

    def test_consensus_weights(self, engine, sample_datasets):
        """Test consensus blending weights."""
        weights = engine.compute_blending_weights(
            datasets=sample_datasets,
            method=BlendingMethod.CONSENSUS
        )

        # Equal weights for voting
        for sensor_type in weights:
            for dataset_id in weights[sensor_type]:
                assert weights[sensor_type][dataset_id] == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample discovery result candidates."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        return [
            DiscoveryResult(
                dataset_id="sar_001",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            ),
            DiscoveryResult(
                dataset_id="optical_001",
                provider="sentinel2",
                data_type="optical",
                source_uri="s3://bucket/optical_001.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=2),
                spatial_coverage_percent=90.0,
                resolution_m=10.0,
                cloud_cover_percent=15.0
            ),
        ]

    def test_determine_fusion_strategy_convenience(self):
        """Test determine_fusion_strategy convenience function."""
        config = determine_fusion_strategy(
            event_class="flood.coastal",
            available_sensors={"sar", "optical"},
            atmospheric_conditions={"cloud_cover_percent": 20}
        )

        assert isinstance(config, FusionConfiguration)
        assert config.strategy == FusionStrategy.COMPLEMENTARY

    def test_create_fusion_plan_convenience(self, sample_candidates):
        """Test create_fusion_plan convenience function."""
        plan = create_fusion_plan(
            event_class="flood.riverine",
            candidates=sample_candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            },
            atmospheric_conditions={"cloud_cover_percent": 10}
        )

        assert isinstance(plan, FusionPlan)
        assert len(plan.datasets) > 0

    def test_identify_temporal_gaps_convenience(self, sample_candidates):
        """Test identify_temporal_gaps convenience function."""
        gaps = identify_temporal_gaps(
            datasets=sample_candidates,
            temporal_extent={
                "start": "2024-01-14T00:00:00Z",
                "end": "2024-01-17T00:00:00Z"
            },
            max_gap_hours=12.0
        )

        assert isinstance(gaps, list)
        # Should find gaps at start and end of extent
        assert len(gaps) >= 1


class TestFusionEdgeCases:
    """Test edge cases for fusion strategy."""

    @pytest.fixture
    def engine(self):
        """Create fusion strategy engine."""
        return FusionStrategyEngine()

    def test_empty_candidates(self, engine):
        """Test fusion plan with empty candidates."""
        config = engine.get_configuration("flood_complementary")

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=[],
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        assert len(plan.datasets) == 0
        assert plan.confidence == 0.0

    def test_missing_required_sensor(self, engine):
        """Test fusion plan when required sensor is missing."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        # Only DEM, no SAR or optical
        candidates = [
            DiscoveryResult(
                dataset_id="dem_001",
                provider="copernicus",
                data_type="dem",
                source_uri="s3://bucket/dem_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100.0,
                resolution_m=30.0
            ),
        ]

        config = engine.get_configuration("flood_complementary")

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        # Only DEM should be in datasets
        assert "dem" in plan.datasets
        assert "sar" not in plan.datasets
        # Confidence should be reduced (missing primary sensor)
        # With only 1 of 3 sensors available, coverage ratio is 0.33
        # Even with high quality DEM, overall confidence is limited
        assert plan.confidence < 1.0  # Not full confidence

    def test_candidates_not_meeting_requirements(self, engine):
        """Test fusion plan when candidates don't meet requirements."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        # Optical with very high cloud cover
        candidates = [
            DiscoveryResult(
                dataset_id="optical_001",
                provider="sentinel2",
                data_type="optical",
                source_uri="s3://bucket/optical_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=90.0,
                resolution_m=10.0,
                cloud_cover_percent=80.0  # Above typical threshold
            ),
        ]

        config = engine.get_configuration("flood_complementary")

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        # Optical should be filtered out due to cloud cover requirement
        assert "optical" not in plan.datasets

    def test_quality_factor_calculation(self, engine):
        """Test quality factor calculation for different datasets."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        excellent = DiscoveryResult(
            dataset_id="excellent",
            provider="provider",
            data_type="sar",
            source_uri="s3://bucket/excellent.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=100.0,
            resolution_m=5.0,
            quality_flag="excellent"
        )

        poor = DiscoveryResult(
            dataset_id="poor",
            provider="provider",
            data_type="sar",
            source_uri="s3://bucket/poor.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=50.0,
            resolution_m=100.0,
            quality_flag="poor"
        )

        excellent_factor = engine._quality_factor(excellent)
        poor_factor = engine._quality_factor(poor)

        assert excellent_factor > poor_factor

    def test_temporal_coverage_calculation(self, engine):
        """Test temporal coverage calculation."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id=f"sar_{i}",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri=f"s3://bucket/sar_{i}.tif",
                    format="cog",
                    acquisition_time=base_time + timedelta(hours=i*6),
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
                for i in range(4)
            ]
        }

        coverage = engine._calculate_temporal_coverage(
            datasets=datasets,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        assert coverage["total_hours"] == 24.0
        assert coverage["total_acquisitions"] == 4
        assert coverage["by_sensor"]["sar"] == 4
        assert "avg_revisit_hours" in coverage
        assert coverage["avg_revisit_hours"] == 6.0

    def test_none_atmospheric_conditions(self, engine):
        """Test strategy determination with None atmospheric conditions."""
        config = engine.determine_strategy(
            event_class="flood.riverine",
            available_sensors={"sar", "optical"},
            atmospheric_conditions=None
        )

        # Should still work with empty conditions
        assert config is not None

    def test_empty_available_sensors(self, engine):
        """Test strategy determination with empty available sensors."""
        config = engine.determine_strategy(
            event_class="flood.riverine",
            available_sensors=set(),
            atmospheric_conditions={}
        )

        # Should still return a configuration
        assert config is not None

    def test_weight_normalization(self, engine):
        """Test that weights are properly normalized."""
        weights = {"sar": 0.6, "optical": 0.3, "thermal": 0.1}

        normalized = engine._normalize_weights(weights)

        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001

    def test_empty_weight_normalization(self, engine):
        """Test normalization of empty weights."""
        weights = {}

        normalized = engine._normalize_weights(weights)

        assert normalized == {}

    def test_spatial_coverage_empty_datasets(self, engine):
        """Test spatial coverage calculation with empty datasets."""
        coverage = engine._calculate_spatial_coverage({})

        assert coverage == 0.0

    def test_generate_rationale_content(self, engine):
        """Test that rationale generation produces meaningful content."""
        config = engine.get_configuration("flood_complementary")
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
            ]
        }

        rationale = engine._generate_rationale(
            config, datasets, confidence=0.8
        )

        assert "complementary" in rationale
        assert "weighted_average" in rationale
        assert "0.80" in rationale

    def test_empty_sensor_configuration_confidence(self, engine):
        """Test confidence calculation with empty sensor configuration (division by zero guard)."""
        # Create a configuration with no sensors
        empty_config = FusionConfiguration(
            name="empty_test",
            strategy=FusionStrategy.COMPLEMENTARY,
            sensors=[]  # Empty sensors list
        )

        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
            ]
        }

        # Should not raise ZeroDivisionError
        confidence = engine._calculate_fusion_confidence(
            empty_config, datasets, weights={"sar": 1.0}
        )

        # With no required sensors, coverage_ratio is 0, so confidence is purely quality-based
        assert 0.0 <= confidence <= 1.0

    def test_quality_factor_no_optional_fields(self, engine):
        """Test quality factor calculation with minimal dataset (no cloud cover, no quality flag)."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        minimal = DiscoveryResult(
            dataset_id="minimal",
            provider="provider",
            data_type="sar",
            source_uri="s3://bucket/minimal.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=75.0,
            resolution_m=20.0
            # No cloud_cover_percent, no quality_flag
        )

        factor = engine._quality_factor(minimal)

        # Should compute without errors and return valid score
        assert 0.0 <= factor <= 1.0

    def test_temporal_gap_gap_duration_at_boundaries(self, engine):
        """Test temporal gap identification exactly at boundary conditions."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        # Acquisition exactly at start
        datasets = {
            "sar": [
                DiscoveryResult(
                    dataset_id="sar_001",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_001.tif",
                    format="cog",
                    acquisition_time=base_time,  # Exactly at start
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                ),
                DiscoveryResult(
                    dataset_id="sar_002",
                    provider="sentinel1",
                    data_type="sar",
                    source_uri="s3://bucket/sar_002.tif",
                    format="cog",
                    acquisition_time=base_time + timedelta(hours=24),  # Exactly at end
                    spatial_coverage_percent=95.0,
                    resolution_m=10.0
                )
            ]
        }

        gaps = engine.identify_temporal_gaps(
            datasets=datasets,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            },
            max_gap_hours=12.0
        )

        # One gap of 24 hours between acquisitions
        assert len(gaps) == 1
        assert gaps[0].duration_hours == 24.0

    def test_densify_single_primary_dataset(self, engine):
        """Test densification with only one primary dataset (no gaps to fill)."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        primary = [
            DiscoveryResult(
                dataset_id="sar_001",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/sar_001.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=95.0,
                resolution_m=10.0
            )
        ]

        gap_fill = [
            DiscoveryResult(
                dataset_id="optical_001",
                provider="sentinel2",
                data_type="optical",
                source_uri="s3://bucket/optical_001.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=12),
                spatial_coverage_percent=90.0,
                resolution_m=10.0
            )
        ]

        result = engine.densify_temporal_coverage(
            primary_datasets=primary,
            gap_fill_datasets=gap_fill,
            max_gap_hours=6.0
        )

        # With only one primary, no gaps to fill (range(0) = no iterations)
        assert len(result) == 1

    def test_blending_weights_empty_datasets(self, engine):
        """Test blending weight computation with empty datasets."""
        weights = engine.compute_blending_weights(
            datasets={},
            method=BlendingMethod.WEIGHTED_AVERAGE,
            base_weights={}
        )

        assert weights == {}

    def test_filter_by_requirements_empty(self, engine):
        """Test filtering with no requirements passes all candidates."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        candidates = [
            DiscoveryResult(
                dataset_id="test_001",
                provider="test",
                data_type="sar",
                source_uri="s3://bucket/test.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=50.0,
                resolution_m=100.0,
                cloud_cover_percent=90.0
            )
        ]

        filtered = engine._filter_by_requirements(candidates, {})

        assert len(filtered) == 1

    def test_select_best_candidates_hierarchical(self, engine):
        """Test candidate selection with hierarchical strategy."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        candidates = [
            DiscoveryResult(
                dataset_id="low_quality",
                provider="test",
                data_type="sar",
                source_uri="s3://bucket/low.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=50.0,
                resolution_m=100.0,
                quality_flag="poor"
            ),
            DiscoveryResult(
                dataset_id="high_quality",
                provider="test",
                data_type="sar",
                source_uri="s3://bucket/high.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=1),
                spatial_coverage_percent=95.0,
                resolution_m=10.0,
                quality_flag="excellent"
            )
        ]

        selected = engine._select_best_candidates(
            candidates,
            FusionStrategy.HIERARCHICAL,
            {"start": "2024-01-15T00:00:00Z", "end": "2024-01-16T00:00:00Z"}
        )

        # Should select only the best one for hierarchical
        assert len(selected) == 1
        assert selected[0].dataset_id == "high_quality"

    def test_is_redundant_different_types(self, engine):
        """Test that different sensor types are not redundant."""
        assert engine.is_redundant("sar", "optical") is False
        assert engine.is_redundant("thermal", "dem") is False
        assert engine.is_redundant("optical", "dem") is False

    def test_fusion_plan_multiple_datasets_per_sensor(self, engine):
        """Test fusion plan with multiple datasets per sensor type."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

        candidates = [
            DiscoveryResult(
                dataset_id=f"sar_{i}",
                provider="sentinel1",
                data_type="sar",
                source_uri=f"s3://bucket/sar_{i}.tif",
                format="cog",
                acquisition_time=base_time + timedelta(hours=i*6),
                spatial_coverage_percent=90.0 + i,
                resolution_m=10.0,
                quality_flag="good"
            )
            for i in range(4)
        ]

        config = FusionConfiguration(
            name="test_temporal",
            strategy=FusionStrategy.TEMPORAL,
            sensors=[
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.GAP_FILL,
                    weight=1.0
                )
            ]
        )

        plan = engine.create_fusion_plan(
            configuration=config,
            candidates=candidates,
            temporal_extent={
                "start": "2024-01-15T00:00:00Z",
                "end": "2024-01-16T00:00:00Z"
            }
        )

        # Temporal strategy selects all candidates
        assert "sar" in plan.datasets
        assert len(plan.datasets["sar"]) == 4
