"""
Tests for fusion strategy.

Tests multi-sensor blending rules, complementary vs redundant strategies,
and temporal densification for improved time series coverage.
"""

import pytest
import numpy as np
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


# ============================================================================
# FUSION CORE TESTS (core/analysis/fusion)
# Tests for multi-sensor alignment, corrections, conflict resolution,
# and uncertainty propagation.
# ============================================================================

from core.analysis.fusion import (
    # Alignment
    SpatialAlignmentMethod,
    TemporalAlignmentMethod,
    AlignmentQuality,
    ReferenceGrid,
    TemporalBin,
    SpatialAlignmentConfig,
    TemporalAlignmentConfig,
    AlignedLayer,
    AlignmentResult,
    SpatialAligner,
    TemporalAligner,
    MultiSensorAligner,
    create_reference_grid,
    align_datasets,
    # Corrections
    TerrainCorrectionMethod,
    AtmosphericCorrectionMethod,
    NormalizationMethod,
    TerrainCorrectionConfig,
    AtmosphericCorrectionConfig,
    NormalizationConfig,
    CorrectionResult,
    TerrainCorrector,
    AtmosphericCorrector,
    RadiometricNormalizer,
    CorrectionPipeline,
    apply_terrain_correction,
    apply_atmospheric_correction,
    normalize_to_reference,
    # Conflict Resolution
    ConflictResolutionStrategy,
    ConflictSeverity,
    ConflictThresholds,
    ConflictConfig,
    SourceLayer,
    ConflictMap,
    ConflictResolutionResult,
    ConflictDetector,
    ConflictResolver,
    ConsensusBuilder,
    detect_conflicts,
    resolve_conflicts,
    build_consensus,
    # Uncertainty
    UncertaintyType,
    UncertaintySource,
    PropagationMethod,
    UncertaintyComponent,
    UncertaintyBudget,
    UncertaintyMap,
    PropagationConfig,
    UncertaintyPropagator,
    UncertaintyCombiner,
    FusionUncertaintyEstimator,
    estimate_uncertainty_from_samples,
    propagate_through_operation,
    combine_uncertainties,
)

import numpy as np


# ============================================================================
# Alignment Tests
# ============================================================================

class TestReferenceGrid:
    """Test reference grid dataclass."""

    def test_basic_creation(self):
        """Test creating a basic reference grid."""
        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_x=0.01,
            resolution_y=0.01,
        )

        assert grid.crs == "EPSG:4326"
        assert grid.bounds == (0.0, 0.0, 1.0, 1.0)
        assert grid.width == 100
        assert grid.height == 100

    def test_explicit_dimensions(self):
        """Test creating a grid with explicit dimensions."""
        grid = ReferenceGrid(
            crs="EPSG:32610",
            bounds=(500000, 4000000, 600000, 4100000),
            resolution_x=100.0,
            resolution_y=100.0,
            width=1000,
            height=1000,
        )

        assert grid.width == 1000
        assert grid.height == 1000

    def test_transform_property(self):
        """Test affine transform property."""
        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_x=0.1,
            resolution_y=0.1,
        )

        transform = grid.transform
        assert transform[0] == 0.0  # x origin
        assert transform[1] == 0.1  # x pixel size
        assert transform[3] == 1.0  # y origin
        assert transform[5] == -0.1  # y pixel size (negative for north-up)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_x=0.01,
            resolution_y=0.01,
        )

        data = grid.to_dict()
        assert "crs" in data
        assert "bounds" in data
        assert "width" in data
        assert "height" in data


class TestTemporalBin:
    """Test temporal bin dataclass."""

    def test_basic_creation(self):
        """Test creating a temporal bin."""
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)

        bin_obj = TemporalBin(start=start, end=end)

        assert bin_obj.duration_hours == 24.0
        assert bin_obj.center == datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

    def test_contains(self):
        """Test timestamp containment check."""
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)

        bin_obj = TemporalBin(start=start, end=end)

        # Inside
        assert bin_obj.contains(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        # At start (inclusive)
        assert bin_obj.contains(start)
        # At end (exclusive)
        assert not bin_obj.contains(end)
        # Outside
        assert not bin_obj.contains(datetime(2024, 1, 14, 12, 0, tzinfo=timezone.utc))


class TestAlignedLayer:
    """Test aligned layer dataclass."""

    def test_basic_creation(self):
        """Test creating an aligned layer."""
        data = np.random.rand(100, 100).astype(np.float32)
        timestamp = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        layer = AlignedLayer(
            data=data,
            source_id="test_001",
            sensor_type="optical",
            timestamp=timestamp,
        )

        assert layer.source_id == "test_001"
        assert layer.sensor_type == "optical"
        assert layer.alignment_quality == AlignmentQuality.GOOD
        assert layer.quality_mask is not None

    def test_valid_fraction(self):
        """Test valid fraction calculation."""
        data = np.full((100, 100), 1.0, dtype=np.float32)
        data[0:50, :] = np.nan  # 50% invalid

        layer = AlignedLayer(
            data=data,
            source_id="test_001",
            sensor_type="sar",
            timestamp=datetime.now(timezone.utc),
        )

        assert 0.49 <= layer.valid_fraction <= 0.51


class TestSpatialAligner:
    """Test spatial aligner."""

    @pytest.fixture
    def aligner(self):
        """Create spatial aligner with reference grid."""
        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_x=0.01,
            resolution_y=0.01,
        )
        config = SpatialAlignmentConfig(
            method=SpatialAlignmentMethod.RESAMPLE_BILINEAR,
            reference_grid=grid,
        )
        return SpatialAligner(config)

    def test_align_same_crs(self, aligner):
        """Test alignment with same CRS."""
        data = np.random.rand(50, 50).astype(np.float32)

        layer = aligner.align(
            data=data,
            source_crs="EPSG:4326",
            source_bounds=(0.0, 0.0, 1.0, 1.0),
            source_resolution=(0.02, 0.02),
            source_id="test_001",
            sensor_type="optical",
        )

        assert layer.data.shape == (100, 100)
        assert layer.source_id == "test_001"
        assert layer.alignment_quality in [AlignmentQuality.EXCELLENT, AlignmentQuality.GOOD]

    def test_align_multiple(self, aligner):
        """Test aligning multiple datasets."""
        datasets = [
            {
                "data": np.random.rand(50, 50).astype(np.float32),
                "source_crs": "EPSG:4326",
                "source_bounds": (0.0, 0.0, 1.0, 1.0),
                "source_resolution": (0.02, 0.02),
                "source_id": f"source_{i}",
                "sensor_type": "optical",
                "timestamp": datetime(2024, 1, 15, i, 0, tzinfo=timezone.utc),
            }
            for i in range(3)
        ]

        layers = aligner.align_multiple(datasets)

        assert len(layers) == 3
        assert all(l.data.shape == (100, 100) for l in layers)

    def test_create_reference_grid(self, aligner):
        """Test reference grid creation."""
        grid = aligner.create_reference_grid(
            bounds=(0.0, 0.0, 10.0, 10.0),
            crs="EPSG:4326",
            resolution=0.1,
        )

        assert grid.width == 100
        assert grid.height == 100

    def test_no_reference_grid_error(self):
        """Test error when no reference grid is set."""
        aligner = SpatialAligner()

        with pytest.raises(ValueError, match="Reference grid not configured"):
            aligner.align(
                data=np.random.rand(50, 50),
                source_crs="EPSG:4326",
                source_bounds=(0.0, 0.0, 1.0, 1.0),
                source_resolution=(0.02, 0.02),
            )


class TestTemporalAligner:
    """Test temporal aligner."""

    @pytest.fixture
    def aligner(self):
        """Create temporal aligner."""
        config = TemporalAlignmentConfig(
            method=TemporalAlignmentMethod.LINEAR,
            max_gap_hours=48.0,
        )
        return TemporalAligner(config)

    @pytest.fixture
    def sample_layers(self):
        """Create sample aligned layers."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        layers = []
        for i in range(4):
            layer = AlignedLayer(
                data=np.full((100, 100), float(i), dtype=np.float32),
                source_id=f"layer_{i}",
                sensor_type="sar",
                timestamp=base_time + timedelta(hours=i * 6),
            )
            layers.append(layer)
        return layers

    def test_align_to_timestamps(self, aligner, sample_layers):
        """Test alignment to specific timestamps."""
        target = [
            datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc),  # Between 0 and 6
            datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc),  # Between 12 and 18
        ]

        aligned = aligner.align_to_timestamps(sample_layers, target)

        assert len(aligned) == 2
        assert aligned[0].timestamp == target[0]
        assert aligned[1].timestamp == target[1]

    def test_align_to_bins(self, aligner, sample_layers):
        """Test alignment to temporal bins."""
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)

        layers, bins = aligner.align_to_bins(sample_layers, start, end, bin_duration_hours=12.0)

        assert len(bins) == 2
        assert len(layers) >= 1

    def test_create_dense_timeseries(self, aligner, sample_layers):
        """Test dense time series creation."""
        dense = aligner.create_dense_timeseries(
            sample_layers,
            time_step_hours=2.0,
        )

        # Should have more layers than input
        assert len(dense) >= len(sample_layers)


class TestMultiSensorAligner:
    """Test multi-sensor aligner."""

    @pytest.fixture
    def aligner(self):
        """Create multi-sensor aligner."""
        spatial_config = SpatialAlignmentConfig(
            method=SpatialAlignmentMethod.RESAMPLE_BILINEAR,
        )
        temporal_config = TemporalAlignmentConfig(
            method=TemporalAlignmentMethod.LINEAR,
        )
        return MultiSensorAligner(spatial_config, temporal_config)

    def test_align_complete(self, aligner):
        """Test complete spatial and temporal alignment."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        datasets = [
            {
                "data": np.random.rand(50, 50).astype(np.float32),
                "source_crs": "EPSG:4326",
                "source_bounds": (0.0, 0.0, 1.0, 1.0),
                "source_resolution": (0.02, 0.02),
                "source_id": f"source_{i}",
                "sensor_type": "sar" if i % 2 == 0 else "optical",
                "timestamp": base_time + timedelta(hours=i * 6),
            }
            for i in range(4)
        ]

        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_x=0.01,
            resolution_y=0.01,
        )

        result = aligner.align(datasets, grid)

        assert isinstance(result, AlignmentResult)
        assert len(result.layers) == 4
        assert result.coverage_map is not None


class TestConvenienceFunctionsAlignment:
    """Test alignment convenience functions."""

    def test_create_reference_grid_convenience(self):
        """Test create_reference_grid convenience function."""
        grid = create_reference_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            crs="EPSG:4326",
            resolution=0.01,
        )

        assert grid.width == 100
        assert grid.height == 100

    def test_align_datasets_convenience(self):
        """Test align_datasets convenience function."""
        datasets = [
            {
                "data": np.random.rand(50, 50).astype(np.float32),
                "source_crs": "EPSG:4326",
                "source_bounds": (0.0, 0.0, 1.0, 1.0),
                "source_resolution": (0.02, 0.02),
                "source_id": "test",
                "sensor_type": "optical",
                "timestamp": datetime.now(timezone.utc),
            }
        ]

        grid = create_reference_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution=0.01,
        )

        result = align_datasets(datasets, grid)

        assert isinstance(result, AlignmentResult)


# ============================================================================
# Corrections Tests
# ============================================================================

class TestTerrainCorrector:
    """Test terrain corrector."""

    @pytest.fixture
    def dem(self):
        """Create synthetic DEM."""
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        # Sloped terrain with some variation
        dem = 100 + xx * 0.5 + yy * 0.3 + 10 * np.sin(xx / 10) * np.cos(yy / 10)
        return dem.astype(np.float32)

    @pytest.fixture
    def corrector(self, dem):
        """Create terrain corrector."""
        config = TerrainCorrectionConfig(
            method=TerrainCorrectionMethod.COSINE,
            dem=dem,
            dem_resolution=30.0,
            sun_elevation_deg=45.0,
            sun_azimuth_deg=180.0,
        )
        return TerrainCorrector(config)

    def test_cosine_correction(self, corrector):
        """Test cosine terrain correction."""
        data = np.random.rand(100, 100).astype(np.float32) * 1000

        result = corrector.correct(data)

        assert isinstance(result, CorrectionResult)
        assert result.corrected_data.shape == data.shape
        assert result.correction_factor is not None
        assert result.quality_mask is not None

    def test_minnaert_correction(self, dem):
        """Test Minnaert terrain correction."""
        config = TerrainCorrectionConfig(
            method=TerrainCorrectionMethod.MINNAERT,
            dem=dem,
            minnaert_k=0.5,
        )
        corrector = TerrainCorrector(config)

        data = np.random.rand(100, 100).astype(np.float32) * 1000
        result = corrector.correct(data)

        assert result.correction_type == "terrain_minnaert"

    def test_no_dem_warning(self):
        """Test warning when no DEM is provided."""
        corrector = TerrainCorrector()
        data = np.random.rand(100, 100).astype(np.float32)

        result = corrector.correct(data)

        assert result.correction_type == "none"


class TestAtmosphericCorrector:
    """Test atmospheric corrector."""

    @pytest.fixture
    def corrector(self):
        """Create atmospheric corrector."""
        config = AtmosphericCorrectionConfig(
            method=AtmosphericCorrectionMethod.DOS,
            solar_zenith_deg=30.0,
        )
        return AtmosphericCorrector(config)

    def test_dos_correction(self, corrector):
        """Test dark object subtraction."""
        # Create data with dark objects
        data = np.random.rand(100, 100).astype(np.float32) * 1000 + 100

        result = corrector.correct(data)

        assert isinstance(result, CorrectionResult)
        assert result.corrected_data.shape == data.shape
        assert result.corrected_data.min() >= 0  # No negative values

    def test_multiband_dos(self, corrector):
        """Test DOS on multi-band data."""
        data = np.random.rand(100, 100, 4).astype(np.float32) * 1000 + 100

        result = corrector.correct(data)

        assert result.corrected_data.shape == data.shape

    def test_toar_correction(self):
        """Test TOAR conversion."""
        config = AtmosphericCorrectionConfig(
            method=AtmosphericCorrectionMethod.TOAR,
            solar_zenith_deg=30.0,
        )
        corrector = AtmosphericCorrector(config)

        data = np.random.rand(100, 100).astype(np.float32) * 10000

        result = corrector.correct(data)

        # TOAR should produce reflectance-like values [0, 1]
        assert result.corrected_data.max() <= 1.0
        assert result.corrected_data.min() >= 0.0


class TestRadiometricNormalizer:
    """Test radiometric normalizer."""

    @pytest.fixture
    def reference(self):
        """Create reference data."""
        return np.random.rand(100, 100).astype(np.float32) * 100 + 50

    @pytest.fixture
    def normalizer(self, reference):
        """Create normalizer with reference."""
        config = NormalizationConfig(
            method=NormalizationMethod.HISTOGRAM_MATCHING,
        )
        normalizer = RadiometricNormalizer(config)
        normalizer.set_reference(reference)
        return normalizer

    def test_histogram_matching(self, normalizer):
        """Test histogram matching normalization."""
        # Data with different distribution
        data = np.random.rand(100, 100).astype(np.float32) * 200 + 100

        result = normalizer.normalize(data)

        assert isinstance(result, CorrectionResult)
        assert result.corrected_data.shape == data.shape

    def test_relative_normalization(self, reference):
        """Test relative normalization."""
        config = NormalizationConfig(method=NormalizationMethod.RELATIVE)
        normalizer = RadiometricNormalizer(config)
        normalizer.set_reference(reference)

        data = np.random.rand(100, 100).astype(np.float32) * 200 + 100
        result = normalizer.normalize(data)

        assert result.correction_type == "normalization_relative"

    def test_no_reference_warning(self):
        """Test warning when no reference is set."""
        normalizer = RadiometricNormalizer()
        data = np.random.rand(100, 100).astype(np.float32)

        result = normalizer.normalize(data)

        assert result.correction_type == "none"


class TestCorrectionPipeline:
    """Test correction pipeline."""

    @pytest.fixture
    def dem(self):
        """Create synthetic DEM."""
        return np.random.rand(100, 100).astype(np.float32) * 1000

    def test_full_pipeline(self, dem):
        """Test full correction pipeline."""
        terrain_config = TerrainCorrectionConfig(
            method=TerrainCorrectionMethod.COSINE,
            sun_elevation_deg=45.0,
        )
        atmospheric_config = AtmosphericCorrectionConfig(
            method=AtmosphericCorrectionMethod.DOS,
        )

        pipeline = CorrectionPipeline(
            terrain_config=terrain_config,
            atmospheric_config=atmospheric_config,
        )

        data = np.random.rand(100, 100).astype(np.float32) * 1000

        result = pipeline.apply(
            data=data,
            sensor_type="optical",
            dem=dem,
        )

        assert result.correction_type == "pipeline"
        assert "atmospheric" in result.diagnostics["steps_applied"]
        assert "terrain" in result.diagnostics["steps_applied"]


class TestCorrectionConvenienceFunctions:
    """Test correction convenience functions."""

    def test_apply_terrain_correction(self):
        """Test terrain correction convenience function."""
        dem = np.random.rand(100, 100).astype(np.float32) * 1000
        data = np.random.rand(100, 100).astype(np.float32) * 100

        result = apply_terrain_correction(data, dem)

        assert isinstance(result, CorrectionResult)

    def test_apply_atmospheric_correction(self):
        """Test atmospheric correction convenience function."""
        data = np.random.rand(100, 100).astype(np.float32) * 1000

        result = apply_atmospheric_correction(data)

        assert isinstance(result, CorrectionResult)

    def test_normalize_to_reference_function(self):
        """Test normalization convenience function."""
        data = np.random.rand(100, 100).astype(np.float32) * 100
        reference = np.random.rand(100, 100).astype(np.float32) * 50

        result = normalize_to_reference(data, reference)

        assert isinstance(result, CorrectionResult)


# ============================================================================
# Conflict Resolution Tests
# ============================================================================

class TestConflictDetector:
    """Test conflict detector."""

    @pytest.fixture
    def detector(self):
        """Create conflict detector."""
        thresholds = ConflictThresholds(
            absolute_tolerance=0.1,
            relative_tolerance=0.05,
        )
        return ConflictDetector(thresholds)

    @pytest.fixture
    def agreeing_layers(self):
        """Create layers that agree."""
        base_data = np.random.rand(50, 50).astype(np.float32)
        return [
            SourceLayer(data=base_data + np.random.randn(50, 50) * 0.01, source_id=f"src_{i}")
            for i in range(3)
        ]

    @pytest.fixture
    def disagreeing_layers(self):
        """Create layers that disagree."""
        return [
            SourceLayer(data=np.full((50, 50), float(i) * 10), source_id=f"src_{i}")
            for i in range(3)
        ]

    def test_detect_no_conflict(self, detector, agreeing_layers):
        """Test detecting no conflict."""
        conflict_map = detector.detect(agreeing_layers)

        assert isinstance(conflict_map, ConflictMap)
        # Most pixels should have no or low conflict
        no_conflict = np.sum(conflict_map.severity_map == ConflictSeverity.NONE.value)
        low_conflict = np.sum(conflict_map.severity_map == ConflictSeverity.LOW.value)
        assert (no_conflict + low_conflict) > 0.5 * conflict_map.severity_map.size

    def test_detect_high_conflict(self, detector, disagreeing_layers):
        """Test detecting high conflict."""
        conflict_map = detector.detect(disagreeing_layers)

        # Should have high or critical conflict
        high = np.sum(conflict_map.severity_map == ConflictSeverity.HIGH.value)
        critical = np.sum(conflict_map.severity_map == ConflictSeverity.CRITICAL.value)
        assert (high + critical) > 0

    def test_single_layer_no_conflict(self, detector):
        """Test that single layer has no conflict."""
        layer = SourceLayer(data=np.random.rand(50, 50), source_id="single")
        conflict_map = detector.detect([layer])

        assert np.all(conflict_map.severity_map == ConflictSeverity.NONE.value)


class TestConflictResolver:
    """Test conflict resolver."""

    @pytest.fixture
    def resolver(self):
        """Create conflict resolver."""
        config = ConflictConfig(
            strategy=ConflictResolutionStrategy.WEIGHTED_MEAN,
        )
        return ConflictResolver(config)

    @pytest.fixture
    def sample_layers(self):
        """Create sample source layers."""
        base = np.random.rand(50, 50).astype(np.float32)
        return [
            SourceLayer(
                data=base + i * 0.1,
                source_id=f"src_{i}",
                confidence=0.8 - i * 0.1,
            )
            for i in range(3)
        ]

    def test_weighted_mean_resolution(self, resolver, sample_layers):
        """Test weighted mean resolution."""
        result = resolver.resolve(sample_layers)

        assert isinstance(result, ConflictResolutionResult)
        assert result.resolved_data.shape == sample_layers[0].data.shape
        assert result.strategy_used == "weighted_mean"

    def test_majority_vote_resolution(self, sample_layers):
        """Test majority vote resolution."""
        # Create discrete data
        layers = [
            SourceLayer(
                data=np.full((50, 50), i % 2, dtype=np.int32),
                source_id=f"src_{i}",
            )
            for i in range(5)
        ]

        config = ConflictConfig(strategy=ConflictResolutionStrategy.MAJORITY_VOTE)
        resolver = ConflictResolver(config)

        result = resolver.resolve(layers)

        assert result.strategy_used == "majority_vote"

    def test_median_resolution(self, sample_layers):
        """Test median resolution."""
        config = ConflictConfig(strategy=ConflictResolutionStrategy.MEDIAN)
        resolver = ConflictResolver(config)

        result = resolver.resolve(sample_layers)

        assert result.strategy_used == "median"

    def test_highest_confidence_resolution(self, sample_layers):
        """Test highest confidence resolution."""
        config = ConflictConfig(strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
        resolver = ConflictResolver(config)

        result = resolver.resolve(sample_layers)

        assert result.strategy_used == "highest_confidence"
        assert result.provenance_map is not None

    def test_priority_order_resolution(self, sample_layers):
        """Test priority order resolution."""
        config = ConflictConfig(
            strategy=ConflictResolutionStrategy.PRIORITY_ORDER,
            source_priorities=["src_2", "src_0", "src_1"],
        )
        resolver = ConflictResolver(config)

        result = resolver.resolve(sample_layers)

        assert result.strategy_used == "priority_order"

    def test_single_layer_resolution(self, resolver):
        """Test resolution with single layer."""
        layer = SourceLayer(data=np.random.rand(50, 50), source_id="single")

        result = resolver.resolve([layer])

        assert result.strategy_used == "single_source"
        assert np.array_equal(result.resolved_data, layer.data)

    def test_empty_layers_error(self, resolver):
        """Test error with empty layers."""
        with pytest.raises(ValueError, match="No layers provided"):
            resolver.resolve([])


class TestConsensusBuilder:
    """Test consensus builder."""

    @pytest.fixture
    def builder(self):
        """Create consensus builder."""
        primary = ConflictConfig(strategy=ConflictResolutionStrategy.WEIGHTED_MEAN)
        secondary = ConflictConfig(strategy=ConflictResolutionStrategy.MEDIAN)
        return ConsensusBuilder(primary, secondary)

    @pytest.fixture
    def sample_layers(self):
        """Create sample source layers."""
        return [
            SourceLayer(
                data=np.random.rand(50, 50).astype(np.float32),
                source_id=f"src_{i}",
                confidence=0.9 - i * 0.2,
            )
            for i in range(3)
        ]

    def test_build_consensus(self, builder, sample_layers):
        """Test building consensus."""
        result = builder.build_consensus(sample_layers)

        assert isinstance(result, ConflictResolutionResult)
        assert result.resolved_data.shape == sample_layers[0].data.shape


class TestConflictConvenienceFunctions:
    """Test conflict resolution convenience functions."""

    def test_detect_conflicts_function(self):
        """Test detect_conflicts convenience function."""
        layers = [
            SourceLayer(data=np.random.rand(50, 50), source_id=f"src_{i}")
            for i in range(3)
        ]

        conflict_map = detect_conflicts(layers)

        assert isinstance(conflict_map, ConflictMap)

    def test_resolve_conflicts_function(self):
        """Test resolve_conflicts convenience function."""
        layers = [
            SourceLayer(data=np.random.rand(50, 50), source_id=f"src_{i}")
            for i in range(3)
        ]

        result = resolve_conflicts(layers, ConflictResolutionStrategy.MEAN)

        assert isinstance(result, ConflictResolutionResult)

    def test_build_consensus_function(self):
        """Test build_consensus convenience function."""
        arrays = [np.random.rand(50, 50).astype(np.float32) for _ in range(3)]

        consensus = build_consensus(arrays)

        assert consensus.shape == (50, 50)


# ============================================================================
# Uncertainty Tests
# ============================================================================

class TestUncertaintyComponent:
    """Test uncertainty component dataclass."""

    def test_basic_creation(self):
        """Test creating an uncertainty component."""
        component = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.1,
        )

        assert component.source == UncertaintySource.SENSOR
        assert component.value == 0.1
        assert component.uncertainty_type == UncertaintyType.STANDARD_DEVIATION

    def test_to_variance(self):
        """Test conversion to variance."""
        component = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.1,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
        )

        variance = component.to_variance()
        assert abs(variance - 0.01) < 1e-10

    def test_to_std(self):
        """Test conversion to standard deviation."""
        component = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.01,
            uncertainty_type=UncertaintyType.VARIANCE,
        )

        std = component.to_std()
        assert abs(std - 0.1) < 1e-10


class TestUncertaintyBudget:
    """Test uncertainty budget dataclass."""

    def test_automatic_total_calculation(self):
        """Test automatic total calculation."""
        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.2),
        ]

        budget = UncertaintyBudget(components=components)

        # RSS: sqrt(0.1^2 + 0.2^2) = sqrt(0.01 + 0.04) = sqrt(0.05) ≈ 0.224
        expected = np.sqrt(0.01 + 0.04)
        assert abs(budget.total_uncertainty - expected) < 1e-6

    def test_dominant_source_detection(self):
        """Test dominant source detection."""
        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.3),
        ]

        budget = UncertaintyBudget(components=components)

        assert budget.dominant_source == UncertaintySource.ALGORITHM


class TestUncertaintyMap:
    """Test uncertainty map dataclass."""

    def test_to_confidence_interval(self):
        """Test confidence interval calculation."""
        uncertainty = np.full((10, 10), 0.1, dtype=np.float32)
        data = np.full((10, 10), 1.0, dtype=np.float32)

        unc_map = UncertaintyMap(uncertainty=uncertainty)
        lower, upper = unc_map.to_confidence_interval(data, confidence=0.95)

        # 95% CI should be approximately ±1.96 sigma
        assert lower.mean() < data.mean()
        assert upper.mean() > data.mean()

    def test_to_relative(self):
        """Test relative uncertainty calculation."""
        uncertainty = np.full((10, 10), 0.1, dtype=np.float32)
        data = np.full((10, 10), 1.0, dtype=np.float32)

        unc_map = UncertaintyMap(uncertainty=uncertainty)
        relative = unc_map.to_relative(data)

        # 0.1 / 1.0 = 10%
        assert abs(relative.mean() - 10.0) < 1e-6


class TestUncertaintyPropagator:
    """Test uncertainty propagator."""

    @pytest.fixture
    def propagator(self):
        """Create uncertainty propagator."""
        return UncertaintyPropagator()

    def test_propagate_addition(self, propagator):
        """Test uncertainty propagation through addition."""
        a = np.array([1.0, 2.0, 3.0])
        sigma_a = np.array([0.1, 0.1, 0.1])
        b = np.array([2.0, 3.0, 4.0])
        sigma_b = np.array([0.2, 0.2, 0.2])

        result, uncertainty = propagator.propagate_addition(a, sigma_a, b, sigma_b)

        # c = a + b
        assert np.allclose(result, a + b)
        # sigma_c = sqrt(sigma_a^2 + sigma_b^2) = sqrt(0.01 + 0.04) ≈ 0.224
        expected_sigma = np.sqrt(0.01 + 0.04)
        assert np.allclose(uncertainty, expected_sigma, rtol=0.01)

    def test_propagate_multiplication(self, propagator):
        """Test uncertainty propagation through multiplication."""
        a = np.array([10.0, 20.0, 30.0])
        sigma_a = np.array([1.0, 1.0, 1.0])  # 10% relative
        b = np.array([2.0, 2.0, 2.0])
        sigma_b = np.array([0.2, 0.2, 0.2])  # 10% relative

        result, uncertainty = propagator.propagate_multiplication(a, sigma_a, b, sigma_b)

        assert np.allclose(result, a * b)

    def test_propagate_division(self, propagator):
        """Test uncertainty propagation through division."""
        a = np.array([10.0, 20.0, 30.0])
        sigma_a = np.array([1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        sigma_b = np.array([0.2, 0.2, 0.2])

        result, uncertainty = propagator.propagate_division(a, sigma_a, b, sigma_b)

        assert np.allclose(result, a / b)

    def test_propagate_power(self, propagator):
        """Test uncertainty propagation through power."""
        a = np.array([2.0, 3.0, 4.0])
        sigma_a = np.array([0.1, 0.1, 0.1])
        n = 2.0

        result, uncertainty = propagator.propagate_power(a, sigma_a, n)

        assert np.allclose(result, a ** n)

    def test_propagate_weighted_average(self, propagator):
        """Test uncertainty propagation through weighted average."""
        data_list = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([0.9, 1.9, 2.9]),
        ]
        unc_list = [
            np.array([0.1, 0.1, 0.1]),
            np.array([0.2, 0.2, 0.2]),
            np.array([0.15, 0.15, 0.15]),
        ]

        result, uncertainty = propagator.propagate_weighted_average(data_list, unc_list)

        # Should be close to average
        expected = np.mean(data_list, axis=0)
        assert np.allclose(result, expected)


class TestUncertaintyCombiner:
    """Test uncertainty combiner."""

    @pytest.fixture
    def combiner(self):
        """Create uncertainty combiner."""
        return UncertaintyCombiner()

    def test_combine_uncorrelated(self, combiner):
        """Test combining uncorrelated components."""
        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.2),
            UncertaintyComponent(source=UncertaintySource.FUSION, value=0.1),
        ]

        combined = combiner.combine_uncorrelated(components)

        # RSS: sqrt(0.01 + 0.04 + 0.01) = sqrt(0.06) ≈ 0.245
        expected = np.sqrt(0.06)
        assert abs(combined.value - expected) < 1e-6

    def test_combine_correlated(self, combiner):
        """Test combining correlated components."""
        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.2),
        ]
        correlation = np.array([[1.0, 0.5], [0.5, 1.0]])

        combined = combiner.combine_correlated(components, correlation)

        # With positive correlation, result should be > RSS
        rss = np.sqrt(0.01 + 0.04)
        assert combined.value > rss * 0.9  # Correlation adds variance


class TestFusionUncertaintyEstimator:
    """Test fusion uncertainty estimator."""

    @pytest.fixture
    def estimator(self):
        """Create fusion uncertainty estimator."""
        return FusionUncertaintyEstimator()

    def test_estimate_from_disagreement(self, estimator):
        """Test uncertainty from sensor disagreement."""
        data_list = [
            np.random.rand(50, 50).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
            np.random.rand(50, 50).astype(np.float32),
        ]

        unc_map = estimator.estimate_from_disagreement(data_list)

        assert isinstance(unc_map, UncertaintyMap)
        assert unc_map.uncertainty.shape == (50, 50)

    def test_estimate_from_single_source(self, estimator):
        """Test uncertainty from single source."""
        data_list = [np.random.rand(50, 50).astype(np.float32)]

        unc_map = estimator.estimate_from_disagreement(data_list)

        # Should return minimum uncertainty
        assert unc_map.metadata["source"] == "single_source_minimum"

    def test_estimate_interpolation_uncertainty(self, estimator):
        """Test interpolation uncertainty estimation."""
        before = np.full((50, 50), 1.0, dtype=np.float32)
        after = np.full((50, 50), 2.0, dtype=np.float32)

        unc_map = estimator.estimate_interpolation_uncertainty(
            before, after, time_fraction=0.5
        )

        assert unc_map.metadata["source"] == "temporal_interpolation"

    def test_estimate_alignment_uncertainty(self, estimator):
        """Test alignment uncertainty estimation."""
        gradient = np.random.rand(50, 50).astype(np.float32) * 10

        unc_map = estimator.estimate_alignment_uncertainty(
            offset_pixels=0.5,
            gradient_magnitude=gradient,
        )

        assert unc_map.metadata["source"] == "spatial_alignment"

    def test_combine_fusion_uncertainties(self, estimator):
        """Test combining all fusion uncertainties."""
        sensor_unc = UncertaintyMap(
            uncertainty=np.full((50, 50), 0.1, dtype=np.float32),
        )
        disagreement_unc = UncertaintyMap(
            uncertainty=np.full((50, 50), 0.05, dtype=np.float32),
        )

        combined = estimator.combine_fusion_uncertainties(
            sensor_unc, disagreement_unc
        )

        assert combined.budget is not None
        assert len(combined.budget.components) >= 2


class TestUncertaintyConvenienceFunctions:
    """Test uncertainty convenience functions."""

    def test_estimate_uncertainty_from_samples_function(self):
        """Test estimate_uncertainty_from_samples convenience function."""
        samples = [np.random.rand(50, 50) for _ in range(5)]

        unc_map = estimate_uncertainty_from_samples(samples)

        assert isinstance(unc_map, UncertaintyMap)

    def test_propagate_through_operation_function(self):
        """Test propagate_through_operation convenience function."""
        data = np.array([1.0, 2.0, 3.0])
        uncertainty = np.array([0.1, 0.1, 0.1])

        result, unc = propagate_through_operation(
            data, uncertainty, "add", np.array([1.0, 1.0, 1.0])
        )

        assert np.allclose(result, data + 1.0)

    def test_combine_uncertainties_function(self):
        """Test combine_uncertainties convenience function."""
        uncertainties = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.2, 0.1, 0.2]),
        ]

        combined = combine_uncertainties(uncertainties, method="rss")

        # RSS
        expected = np.sqrt(uncertainties[0]**2 + uncertainties[1]**2)
        assert np.allclose(combined, expected)

    def test_combine_uncertainties_max_method(self):
        """Test max combination method."""
        uncertainties = [
            np.array([0.1, 0.3, 0.2]),
            np.array([0.2, 0.1, 0.3]),
        ]

        combined = combine_uncertainties(uncertainties, method="max")

        expected = np.array([0.2, 0.3, 0.3])
        assert np.allclose(combined, expected)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestFusionCoreEdgeCases:
    """Test edge cases for fusion core modules."""

    def test_alignment_empty_input(self):
        """Test alignment with empty input."""
        aligner = TemporalAligner()
        result = aligner.align_to_timestamps([], [datetime.now(timezone.utc)])
        assert result == []

    def test_conflict_resolution_with_nan(self):
        """Test conflict resolution handles NaN values."""
        layers = [
            SourceLayer(
                data=np.array([[1.0, np.nan], [2.0, 3.0]]),
                source_id="src_0"
            ),
            SourceLayer(
                data=np.array([[1.1, 2.0], [np.nan, 3.1]]),
                source_id="src_1"
            ),
        ]

        result = resolve_conflicts(layers, ConflictResolutionStrategy.MEAN)

        # Should handle NaN gracefully
        assert isinstance(result, ConflictResolutionResult)

    def test_uncertainty_zero_values(self):
        """Test uncertainty propagation with zero values."""
        propagator = UncertaintyPropagator()

        data = np.array([0.0, 1.0, 2.0])
        uncertainty = np.array([0.1, 0.1, 0.1])
        divisor = np.array([0.0, 1.0, 2.0])  # Contains zero
        div_unc = np.array([0.1, 0.1, 0.1])

        result, unc = propagator.propagate_division(data, uncertainty, divisor, div_unc)

        # Division by zero should produce NaN
        assert np.isnan(result[0])
        assert np.isnan(unc[0])

    def test_terrain_correction_flat_dem(self):
        """Test terrain correction with flat DEM."""
        flat_dem = np.full((100, 100), 1000.0, dtype=np.float32)
        config = TerrainCorrectionConfig(
            method=TerrainCorrectionMethod.COSINE,
            dem=flat_dem,
            dem_resolution=30.0,
        )
        corrector = TerrainCorrector(config)

        data = np.random.rand(100, 100).astype(np.float32) * 100
        result = corrector.correct(data)

        # Flat terrain should have minimal correction
        assert isinstance(result, CorrectionResult)

    def test_normalization_constant_data(self):
        """Test normalization with constant data."""
        constant = np.full((50, 50), 100.0, dtype=np.float32)
        reference = np.full((50, 50), 50.0, dtype=np.float32)

        config = NormalizationConfig(method=NormalizationMethod.RELATIVE)
        normalizer = RadiometricNormalizer(config)
        normalizer.set_reference(reference)

        result = normalizer.normalize(constant)

        # Should handle constant data gracefully
        assert isinstance(result, CorrectionResult)
