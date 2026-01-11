"""
Tests for Fusion Core Module (Group H, Track 2).

Tests the core fusion capabilities:
- alignment.py: Spatial and temporal alignment
- corrections.py: Terrain and atmospheric corrections
- conflict.py: Conflict detection and resolution
- uncertainty.py: Uncertainty propagation
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone


# ============================================================================
# Alignment Module Tests
# ============================================================================

class TestSpatialAlignmentEnums:
    """Test spatial alignment enums."""

    def test_spatial_alignment_method_values(self):
        """Test SpatialAlignmentMethod enum values."""
        from core.analysis.fusion.alignment import SpatialAlignmentMethod

        assert SpatialAlignmentMethod.REPROJECT.value == "reproject"
        assert SpatialAlignmentMethod.COREGISTER.value == "coregister"
        assert SpatialAlignmentMethod.RESAMPLE_NEAREST.value == "resample_nearest"
        assert SpatialAlignmentMethod.RESAMPLE_BILINEAR.value == "resample_bilinear"
        assert SpatialAlignmentMethod.RESAMPLE_CUBIC.value == "resample_cubic"
        assert SpatialAlignmentMethod.RESAMPLE_LANCZOS.value == "resample_lanczos"

    def test_temporal_alignment_method_values(self):
        """Test TemporalAlignmentMethod enum values."""
        from core.analysis.fusion.alignment import TemporalAlignmentMethod

        assert TemporalAlignmentMethod.NEAREST.value == "nearest"
        assert TemporalAlignmentMethod.LINEAR.value == "linear"
        assert TemporalAlignmentMethod.CUBIC.value == "cubic"
        assert TemporalAlignmentMethod.STEP.value == "step"
        assert TemporalAlignmentMethod.MEAN.value == "mean"
        assert TemporalAlignmentMethod.MEDIAN.value == "median"
        assert TemporalAlignmentMethod.WEIGHTED.value == "weighted"

    def test_alignment_quality_values(self):
        """Test AlignmentQuality enum values."""
        from core.analysis.fusion.alignment import AlignmentQuality

        assert AlignmentQuality.EXCELLENT.value == "excellent"
        assert AlignmentQuality.GOOD.value == "good"
        assert AlignmentQuality.FAIR.value == "fair"
        assert AlignmentQuality.POOR.value == "poor"
        assert AlignmentQuality.DEGRADED.value == "degraded"


class TestReferenceGrid:
    """Test ReferenceGrid dataclass."""

    def test_basic_creation(self):
        """Test creating a basic reference grid."""
        from core.analysis.fusion.alignment import ReferenceGrid

        grid = ReferenceGrid(
            crs="EPSG:32618",
            bounds=(500000, 4500000, 600000, 4600000),
            resolution_x=10.0,
            resolution_y=10.0,
        )

        assert grid.crs == "EPSG:32618"
        assert grid.bounds == (500000, 4500000, 600000, 4600000)
        assert grid.resolution_x == 10.0
        assert grid.resolution_y == 10.0

    def test_dimension_calculation(self):
        """Test automatic dimension calculation."""
        from core.analysis.fusion.alignment import ReferenceGrid

        grid = ReferenceGrid(
            crs="EPSG:32618",
            bounds=(0, 0, 1000, 500),
            resolution_x=10.0,
            resolution_y=10.0,
        )

        assert grid.width == 100
        assert grid.height == 50

    def test_transform_property(self):
        """Test affine transform generation."""
        from core.analysis.fusion.alignment import ReferenceGrid

        grid = ReferenceGrid(
            crs="EPSG:32618",
            bounds=(500000, 4500000, 600000, 4600000),
            resolution_x=10.0,
            resolution_y=10.0,
        )

        transform = grid.transform
        assert transform[0] == 500000  # x origin
        assert transform[1] == 10.0    # x pixel size
        assert transform[3] == 4600000  # y origin (top)
        assert transform[5] == -10.0   # y pixel size (negative)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.alignment import ReferenceGrid

        grid = ReferenceGrid(
            crs="EPSG:32618",
            bounds=(0, 0, 100, 100),
            resolution_x=10.0,
            resolution_y=10.0,
        )

        data = grid.to_dict()

        assert data["crs"] == "EPSG:32618"
        assert data["bounds"] == (0, 0, 100, 100)
        assert data["width"] == 10
        assert data["height"] == 10


class TestTemporalBin:
    """Test TemporalBin dataclass."""

    def test_basic_creation(self):
        """Test creating a temporal bin."""
        from core.analysis.fusion.alignment import TemporalBin

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        bin = TemporalBin(start=start, end=end)

        assert bin.start == start
        assert bin.end == end

    def test_center_calculation(self):
        """Test automatic center calculation."""
        from core.analysis.fusion.alignment import TemporalBin

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        bin = TemporalBin(start=start, end=end)

        expected_center = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        assert bin.center == expected_center

    def test_duration_hours(self):
        """Test duration calculation."""
        from core.analysis.fusion.alignment import TemporalBin

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

        bin = TemporalBin(start=start, end=end)

        assert bin.duration_hours == 24.0

    def test_contains(self):
        """Test timestamp containment check."""
        from core.analysis.fusion.alignment import TemporalBin

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        bin = TemporalBin(start=start, end=end)

        inside = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        before = datetime(2023, 12, 31, tzinfo=timezone.utc)
        after = datetime(2024, 1, 2, tzinfo=timezone.utc)  # end is exclusive

        assert bin.contains(inside)
        assert not bin.contains(before)
        assert not bin.contains(after)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.alignment import TemporalBin

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        bin = TemporalBin(start=start, end=end, label="day1")

        data = bin.to_dict()

        assert "start" in data
        assert "end" in data
        assert "center" in data
        assert data["label"] == "day1"
        assert data["duration_hours"] == 24.0


class TestAlignmentConfig:
    """Test alignment configuration dataclasses."""

    def test_spatial_config_defaults(self):
        """Test SpatialAlignmentConfig defaults."""
        from core.analysis.fusion.alignment import SpatialAlignmentConfig, SpatialAlignmentMethod

        config = SpatialAlignmentConfig()

        assert config.method == SpatialAlignmentMethod.RESAMPLE_BILINEAR
        assert config.max_offset_pixels == 5.0
        assert config.subpixel_precision is True
        assert config.nodata_handling == "mask"
        assert config.quality_threshold == 0.5

    def test_temporal_config_defaults(self):
        """Test TemporalAlignmentConfig defaults."""
        from core.analysis.fusion.alignment import TemporalAlignmentConfig, TemporalAlignmentMethod

        config = TemporalAlignmentConfig()

        assert config.method == TemporalAlignmentMethod.LINEAR
        assert config.bin_duration_hours == 24.0
        assert config.max_gap_hours == 72.0
        assert config.extrapolate is False
        assert config.quality_weights is True


class TestSpatialAligner:
    """Test SpatialAligner class."""

    def test_creation_default_config(self):
        """Test creating a spatial aligner with defaults."""
        from core.analysis.fusion.alignment import SpatialAligner

        aligner = SpatialAligner()
        assert aligner is not None
        assert aligner.config is not None

    def test_creation_custom_config(self):
        """Test creating with custom configuration."""
        from core.analysis.fusion.alignment import (
            SpatialAligner,
            SpatialAlignmentConfig,
            SpatialAlignmentMethod,
        )

        config = SpatialAlignmentConfig(
            method=SpatialAlignmentMethod.RESAMPLE_NEAREST,
            subpixel_precision=False,
        )
        aligner = SpatialAligner(config=config)

        assert aligner.config.method == SpatialAlignmentMethod.RESAMPLE_NEAREST
        assert aligner.config.subpixel_precision is False


class TestTemporalAligner:
    """Test TemporalAligner class."""

    def test_creation_default_config(self):
        """Test creating a temporal aligner with defaults."""
        from core.analysis.fusion.alignment import TemporalAligner

        aligner = TemporalAligner()
        assert aligner is not None
        assert aligner.config is not None

    def test_creation_custom_config(self):
        """Test creating with custom configuration."""
        from core.analysis.fusion.alignment import (
            TemporalAligner,
            TemporalAlignmentConfig,
            TemporalAlignmentMethod,
        )

        config = TemporalAlignmentConfig(
            method=TemporalAlignmentMethod.MEDIAN,
            bin_duration_hours=6.0,
        )
        aligner = TemporalAligner(config=config)

        assert aligner.config.method == TemporalAlignmentMethod.MEDIAN
        assert aligner.config.bin_duration_hours == 6.0


class TestMultiSensorAligner:
    """Test MultiSensorAligner class."""

    def test_creation(self):
        """Test creating multi-sensor aligner."""
        from core.analysis.fusion.alignment import MultiSensorAligner

        aligner = MultiSensorAligner()
        assert aligner is not None


class TestAlignmentConvenienceFunctions:
    """Test convenience functions."""

    def test_create_reference_grid(self):
        """Test create_reference_grid function."""
        from core.analysis.fusion.alignment import create_reference_grid

        grid = create_reference_grid(
            crs="EPSG:4326",
            bounds=(-180, -90, 180, 90),
            resolution=1.0,
        )

        assert grid.crs == "EPSG:4326"
        assert grid.width == 360
        assert grid.height == 180


# ============================================================================
# Corrections Module Tests
# ============================================================================

class TestCorrectionEnums:
    """Test correction enums."""

    def test_terrain_correction_method_values(self):
        """Test TerrainCorrectionMethod enum values."""
        from core.analysis.fusion.corrections import TerrainCorrectionMethod

        assert TerrainCorrectionMethod.COSINE.value == "cosine"
        assert TerrainCorrectionMethod.MINNAERT.value == "minnaert"
        assert TerrainCorrectionMethod.C_CORRECTION.value == "c_correction"
        assert TerrainCorrectionMethod.SCS.value == "scs"
        assert TerrainCorrectionMethod.GAMMA.value == "gamma"
        assert TerrainCorrectionMethod.FLAT_EARTH.value == "flat_earth"

    def test_atmospheric_correction_method_values(self):
        """Test AtmosphericCorrectionMethod enum values."""
        from core.analysis.fusion.corrections import AtmosphericCorrectionMethod

        assert AtmosphericCorrectionMethod.DOS.value == "dos"
        assert AtmosphericCorrectionMethod.DOS_IMPROVED.value == "dos_improved"
        assert AtmosphericCorrectionMethod.COST.value == "cost"
        assert AtmosphericCorrectionMethod.TOAR.value == "toar"
        assert AtmosphericCorrectionMethod.FLAASH.value == "flaash"
        assert AtmosphericCorrectionMethod.SEN2COR.value == "sen2cor"

    def test_normalization_method_values(self):
        """Test NormalizationMethod enum values."""
        from core.analysis.fusion.corrections import NormalizationMethod

        assert NormalizationMethod.HISTOGRAM_MATCHING.value == "histogram_matching"
        assert NormalizationMethod.PIF.value == "pif"
        assert NormalizationMethod.RELATIVE.value == "relative"
        assert NormalizationMethod.ABSOLUTE.value == "absolute"
        assert NormalizationMethod.IR_MAD.value == "ir_mad"


class TestCorrectionConfigs:
    """Test correction configuration dataclasses."""

    def test_terrain_config_defaults(self):
        """Test TerrainCorrectionConfig defaults."""
        from core.analysis.fusion.corrections import (
            TerrainCorrectionConfig,
            TerrainCorrectionMethod,
        )

        config = TerrainCorrectionConfig()

        assert config.method == TerrainCorrectionMethod.COSINE
        assert config.dem is None
        assert config.dem_resolution == 30.0
        assert config.sun_elevation_deg == 45.0
        assert config.shadow_detection is True

    def test_atmospheric_config_defaults(self):
        """Test AtmosphericCorrectionConfig defaults."""
        from core.analysis.fusion.corrections import (
            AtmosphericCorrectionConfig,
            AtmosphericCorrectionMethod,
        )

        config = AtmosphericCorrectionConfig()

        assert config.method == AtmosphericCorrectionMethod.DOS
        assert config.sensor_type == "generic"
        assert config.solar_zenith_deg == 30.0
        assert config.aod550 == 0.1

    def test_normalization_config_defaults(self):
        """Test NormalizationConfig defaults."""
        from core.analysis.fusion.corrections import (
            NormalizationConfig,
            NormalizationMethod,
        )

        config = NormalizationConfig()

        assert config.method == NormalizationMethod.HISTOGRAM_MATCHING
        assert config.use_pif is True
        assert config.pif_threshold == 0.95
        assert config.clip_outliers is True


class TestCorrectionResult:
    """Test CorrectionResult dataclass."""

    def test_creation(self):
        """Test creating a correction result."""
        from core.analysis.fusion.corrections import CorrectionResult

        data = np.random.rand(100, 100).astype(np.float32)
        result = CorrectionResult(
            corrected_data=data,
            correction_type="terrain",
            parameters={"method": "cosine"},
        )

        assert result.corrected_data.shape == (100, 100)
        assert result.correction_type == "terrain"
        assert result.parameters["method"] == "cosine"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.corrections import CorrectionResult

        data = np.random.rand(100, 100).astype(np.float32)
        result = CorrectionResult(
            corrected_data=data,
            correction_type="atmospheric",
        )

        data_dict = result.to_dict()

        assert data_dict["correction_type"] == "atmospheric"
        assert data_dict["data_shape"] == [100, 100]
        assert "float32" in data_dict["data_dtype"]


class TestTerrainCorrector:
    """Test TerrainCorrector class."""

    def test_creation_default_config(self):
        """Test creating terrain corrector with defaults."""
        from core.analysis.fusion.corrections import TerrainCorrector

        corrector = TerrainCorrector()
        assert corrector is not None
        assert corrector.config is not None

    def test_creation_custom_config(self):
        """Test creating with custom configuration."""
        from core.analysis.fusion.corrections import (
            TerrainCorrector,
            TerrainCorrectionConfig,
            TerrainCorrectionMethod,
        )

        config = TerrainCorrectionConfig(
            method=TerrainCorrectionMethod.MINNAERT,
            minnaert_k=0.7,
        )
        corrector = TerrainCorrector(config=config)

        assert corrector.config.method == TerrainCorrectionMethod.MINNAERT
        assert corrector.config.minnaert_k == 0.7


class TestAtmosphericCorrector:
    """Test AtmosphericCorrector class."""

    def test_creation_default_config(self):
        """Test creating atmospheric corrector with defaults."""
        from core.analysis.fusion.corrections import AtmosphericCorrector

        corrector = AtmosphericCorrector()
        assert corrector is not None
        assert corrector.config is not None


class TestCorrectionPipeline:
    """Test CorrectionPipeline class."""

    def test_creation(self):
        """Test creating correction pipeline."""
        from core.analysis.fusion.corrections import CorrectionPipeline

        pipeline = CorrectionPipeline()
        assert pipeline is not None


class TestCorrectionConvenienceFunctions:
    """Test convenience functions."""

    def test_apply_terrain_correction(self):
        """Test apply_terrain_correction function."""
        from core.analysis.fusion.corrections import apply_terrain_correction

        data = np.random.rand(50, 50).astype(np.float32)
        dem = np.random.rand(50, 50).astype(np.float32) * 1000  # 0-1000m elevation

        result = apply_terrain_correction(data, dem=dem)

        assert result.corrected_data.shape == data.shape
        # Correction type includes method name
        assert result.correction_type.startswith("terrain")

    def test_apply_atmospheric_correction(self):
        """Test apply_atmospheric_correction function."""
        from core.analysis.fusion.corrections import apply_atmospheric_correction

        data = np.random.rand(50, 50).astype(np.float32) * 10000  # DN values

        result = apply_atmospheric_correction(data)

        assert result.corrected_data.shape == data.shape
        # Correction type includes method name
        assert result.correction_type.startswith("atmospheric")


# ============================================================================
# Conflict Resolution Module Tests
# ============================================================================

class TestConflictEnums:
    """Test conflict enums."""

    def test_conflict_resolution_strategy_values(self):
        """Test ConflictResolutionStrategy enum values."""
        from core.analysis.fusion.conflict import ConflictResolutionStrategy

        assert ConflictResolutionStrategy.MAJORITY_VOTE.value == "majority_vote"
        assert ConflictResolutionStrategy.WEIGHTED_VOTE.value == "weighted_vote"
        assert ConflictResolutionStrategy.MEAN.value == "mean"
        assert ConflictResolutionStrategy.WEIGHTED_MEAN.value == "weighted_mean"
        assert ConflictResolutionStrategy.MEDIAN.value == "median"
        assert ConflictResolutionStrategy.HIGHEST_CONFIDENCE.value == "highest_confidence"
        assert ConflictResolutionStrategy.PRIORITY_ORDER.value == "priority_order"

    def test_conflict_severity_values(self):
        """Test ConflictSeverity enum values."""
        from core.analysis.fusion.conflict import ConflictSeverity

        assert ConflictSeverity.NONE.value == "none"
        assert ConflictSeverity.LOW.value == "low"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.CRITICAL.value == "critical"


class TestConflictThresholds:
    """Test ConflictThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        from core.analysis.fusion.conflict import ConflictThresholds

        thresholds = ConflictThresholds()

        assert thresholds.absolute_tolerance == 0.1
        assert thresholds.relative_tolerance == 0.05
        assert thresholds.min_agreement_ratio == 0.5
        assert thresholds.outlier_sigma == 2.0
        assert thresholds.min_sources_for_consensus == 2

    def test_custom_values(self):
        """Test custom threshold values."""
        from core.analysis.fusion.conflict import ConflictThresholds

        thresholds = ConflictThresholds(
            absolute_tolerance=0.2,
            outlier_sigma=3.0,
        )

        assert thresholds.absolute_tolerance == 0.2
        assert thresholds.outlier_sigma == 3.0


class TestConflictConfig:
    """Test ConflictConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        from core.analysis.fusion.conflict import (
            ConflictConfig,
            ConflictResolutionStrategy,
        )

        config = ConflictConfig()

        assert config.strategy == ConflictResolutionStrategy.WEIGHTED_MEAN
        assert config.use_quality_weights is True
        assert config.track_provenance is True

    def test_custom_values(self):
        """Test custom config values."""
        from core.analysis.fusion.conflict import (
            ConflictConfig,
            ConflictResolutionStrategy,
        )

        config = ConflictConfig(
            strategy=ConflictResolutionStrategy.MEDIAN,
            source_priorities=["sar", "optical"],
        )

        assert config.strategy == ConflictResolutionStrategy.MEDIAN
        assert config.source_priorities == ["sar", "optical"]


class TestSourceLayer:
    """Test SourceLayer dataclass."""

    def test_basic_creation(self):
        """Test creating a basic source layer."""
        from core.analysis.fusion.conflict import SourceLayer

        data = np.random.rand(50, 50).astype(np.float32)
        layer = SourceLayer(data=data, source_id="sentinel1")

        assert layer.data.shape == (50, 50)
        assert layer.source_id == "sentinel1"
        assert layer.confidence == 1.0

    def test_quality_auto_init(self):
        """Test automatic quality initialization."""
        from core.analysis.fusion.conflict import SourceLayer

        data = np.random.rand(50, 50).astype(np.float32)
        data[10:20, 10:20] = np.nan  # Add some NaN values

        layer = SourceLayer(data=data, source_id="test")

        assert layer.quality is not None
        assert layer.quality.shape == (50, 50)
        # Quality should be 0 where data is NaN
        assert np.all(layer.quality[10:20, 10:20] == 0)

    def test_full_creation(self):
        """Test full creation with all attributes."""
        from core.analysis.fusion.conflict import SourceLayer

        data = np.random.rand(50, 50).astype(np.float32)
        quality = np.ones((50, 50), dtype=np.float32)
        timestamp = datetime.now(timezone.utc)

        layer = SourceLayer(
            data=data,
            source_id="sentinel2",
            quality=quality,
            confidence=0.9,
            sensor_type="optical",
            timestamp=timestamp,
            metadata={"cloud_cover": 10},
        )

        assert layer.confidence == 0.9
        assert layer.sensor_type == "optical"
        assert layer.timestamp == timestamp
        assert layer.metadata["cloud_cover"] == 10


class TestConflictMap:
    """Test ConflictMap dataclass."""

    def test_creation(self):
        """Test creating a conflict map."""
        from core.analysis.fusion.conflict import ConflictMap

        severity = np.full((50, 50), "none")
        disagreement = np.zeros((50, 50))
        source_count = np.full((50, 50), 3)
        agreement = np.ones((50, 50))

        cmap = ConflictMap(
            severity_map=severity,
            disagreement_map=disagreement,
            source_count_map=source_count,
            agreement_ratio_map=agreement,
        )

        assert cmap.severity_map.shape == (50, 50)
        assert cmap.disagreement_map.shape == (50, 50)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.conflict import ConflictMap

        severity = np.full((50, 50), "none")
        disagreement = np.zeros((50, 50))
        source_count = np.full((50, 50), 3)
        agreement = np.ones((50, 50))

        cmap = ConflictMap(
            severity_map=severity,
            disagreement_map=disagreement,
            source_count_map=source_count,
            agreement_ratio_map=agreement,
        )

        data = cmap.to_dict()

        assert data["total_pixels"] == 2500
        assert data["mean_disagreement"] == 0.0
        assert data["mean_source_count"] == 3.0


class TestConflictResolutionResult:
    """Test ConflictResolutionResult dataclass."""

    def test_creation(self):
        """Test creating a resolution result."""
        from core.analysis.fusion.conflict import ConflictResolutionResult

        resolved = np.random.rand(50, 50).astype(np.float32)
        confidence = np.ones((50, 50), dtype=np.float32)

        result = ConflictResolutionResult(
            resolved_data=resolved,
            confidence_map=confidence,
            strategy_used="weighted_mean",
        )

        assert result.resolved_data.shape == (50, 50)
        assert result.strategy_used == "weighted_mean"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.conflict import ConflictResolutionResult

        resolved = np.random.rand(50, 50).astype(np.float32)
        confidence = np.ones((50, 50), dtype=np.float32) * 0.8

        result = ConflictResolutionResult(
            resolved_data=resolved,
            confidence_map=confidence,
            strategy_used="median",
        )

        data = result.to_dict()

        assert data["data_shape"] == [50, 50]
        assert data["strategy_used"] == "median"
        assert abs(data["mean_confidence"] - 0.8) < 0.01


class TestConflictDetector:
    """Test ConflictDetector class."""

    def test_creation_default(self):
        """Test creating conflict detector with defaults."""
        from core.analysis.fusion.conflict import ConflictDetector

        detector = ConflictDetector()
        assert detector is not None
        assert detector.thresholds is not None

    def test_creation_custom_thresholds(self):
        """Test creating with custom thresholds."""
        from core.analysis.fusion.conflict import ConflictDetector, ConflictThresholds

        thresholds = ConflictThresholds(absolute_tolerance=0.5)
        detector = ConflictDetector(thresholds=thresholds)

        assert detector.thresholds.absolute_tolerance == 0.5


class TestConflictResolver:
    """Test ConflictResolver class."""

    def test_creation(self):
        """Test creating conflict resolver."""
        from core.analysis.fusion.conflict import ConflictResolver

        resolver = ConflictResolver()
        assert resolver is not None


class TestConsensusBuilder:
    """Test ConsensusBuilder class."""

    def test_creation(self):
        """Test creating consensus builder."""
        from core.analysis.fusion.conflict import ConsensusBuilder

        builder = ConsensusBuilder()
        assert builder is not None


class TestConflictConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_conflicts(self):
        """Test detect_conflicts function."""
        from core.analysis.fusion.conflict import detect_conflicts, SourceLayer

        # Create two agreeing layers
        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.5

        layer1 = SourceLayer(data=data1, source_id="source1")
        layer2 = SourceLayer(data=data2, source_id="source2")

        conflict_map = detect_conflicts([layer1, layer2])

        assert conflict_map is not None
        assert conflict_map.severity_map.shape == (50, 50)

    def test_resolve_conflicts(self):
        """Test resolve_conflicts function."""
        from core.analysis.fusion.conflict import resolve_conflicts, SourceLayer

        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.6

        layer1 = SourceLayer(data=data1, source_id="source1")
        layer2 = SourceLayer(data=data2, source_id="source2")

        result = resolve_conflicts([layer1, layer2])

        assert result.resolved_data.shape == (50, 50)
        # Mean of 0.5 and 0.6 should be approximately 0.55
        assert np.abs(np.mean(result.resolved_data) - 0.55) < 0.01

    def test_build_consensus(self):
        """Test build_consensus function."""
        from core.analysis.fusion.conflict import build_consensus

        # build_consensus takes arrays, not SourceLayer objects
        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.5
        data3 = np.ones((50, 50), dtype=np.float32) * 0.5

        result = build_consensus([data1, data2, data3])

        # Returns array directly
        assert result.shape == (50, 50)


# ============================================================================
# Uncertainty Module Tests
# ============================================================================

class TestUncertaintyEnums:
    """Test uncertainty enums."""

    def test_uncertainty_type_values(self):
        """Test UncertaintyType enum values."""
        from core.analysis.fusion.uncertainty import UncertaintyType

        assert UncertaintyType.STANDARD_DEVIATION.value == "std"
        assert UncertaintyType.VARIANCE.value == "variance"
        assert UncertaintyType.CONFIDENCE_INTERVAL.value == "ci"
        assert UncertaintyType.RELATIVE.value == "relative"
        assert UncertaintyType.QUANTILES.value == "quantiles"
        assert UncertaintyType.ENSEMBLE.value == "ensemble"

    def test_uncertainty_source_values(self):
        """Test UncertaintySource enum values."""
        from core.analysis.fusion.uncertainty import UncertaintySource

        assert UncertaintySource.SENSOR.value == "sensor"
        assert UncertaintySource.ATMOSPHERIC.value == "atmospheric"
        assert UncertaintySource.GEOMETRIC.value == "geometric"
        assert UncertaintySource.ALGORITHM.value == "algorithm"
        assert UncertaintySource.FUSION.value == "fusion"
        assert UncertaintySource.INTERPOLATION.value == "interpolation"

    def test_propagation_method_values(self):
        """Test PropagationMethod enum values."""
        from core.analysis.fusion.uncertainty import PropagationMethod

        assert PropagationMethod.LINEAR.value == "linear"
        assert PropagationMethod.MONTE_CARLO.value == "monte_carlo"
        assert PropagationMethod.ANALYTICAL.value == "analytical"
        assert PropagationMethod.ENSEMBLE.value == "ensemble"
        assert PropagationMethod.TAYLOR.value == "taylor"


class TestUncertaintyComponent:
    """Test UncertaintyComponent dataclass."""

    def test_basic_creation(self):
        """Test creating a basic uncertainty component."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyComponent,
            UncertaintySource,
            UncertaintyType,
        )

        component = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.1,
        )

        assert component.source == UncertaintySource.SENSOR
        assert component.value == 0.1
        assert component.uncertainty_type == UncertaintyType.STANDARD_DEVIATION
        assert component.is_systematic is False

    def test_to_variance(self):
        """Test conversion to variance."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyComponent,
            UncertaintySource,
            UncertaintyType,
        )

        # From std dev
        comp_std = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.1,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
        )
        assert np.isclose(comp_std.to_variance(), 0.01)

        # From variance
        comp_var = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.04,
            uncertainty_type=UncertaintyType.VARIANCE,
        )
        assert comp_var.to_variance() == 0.04

    def test_to_std(self):
        """Test conversion to standard deviation."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyComponent,
            UncertaintySource,
            UncertaintyType,
        )

        # From variance
        comp_var = UncertaintyComponent(
            source=UncertaintySource.SENSOR,
            value=0.04,
            uncertainty_type=UncertaintyType.VARIANCE,
        )
        assert np.isclose(comp_var.to_std(), 0.2)

    def test_array_uncertainty(self):
        """Test with array uncertainty values."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyComponent,
            UncertaintySource,
        )

        uncertainty_array = np.random.rand(50, 50) * 0.1
        component = UncertaintyComponent(
            source=UncertaintySource.ALGORITHM,
            value=uncertainty_array,
        )

        assert component.value.shape == (50, 50)
        assert component.to_variance().shape == (50, 50)


class TestUncertaintyBudget:
    """Test UncertaintyBudget dataclass."""

    def test_basic_creation(self):
        """Test creating an uncertainty budget."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.05),
        ]

        budget = UncertaintyBudget(components=components)

        assert len(budget.components) == 2
        assert budget.total_uncertainty is not None

    def test_total_calculation(self):
        """Test total uncertainty calculation."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        # Two components with std dev 0.3 and 0.4
        # Total should be sqrt(0.3^2 + 0.4^2) = 0.5
        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.3),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.4),
        ]

        budget = UncertaintyBudget(components=components)

        assert np.isclose(budget.total_uncertainty, 0.5)

    def test_dominant_source(self):
        """Test finding dominant uncertainty source."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
            UncertaintyComponent(source=UncertaintySource.ALGORITHM, value=0.5),  # Larger
        ]

        budget = UncertaintyBudget(components=components)

        assert budget.dominant_source == UncertaintySource.ALGORITHM

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.1),
        ]

        budget = UncertaintyBudget(components=components)
        data = budget.to_dict()

        assert data["num_components"] == 1
        assert data["dominant_source"] == "sensor"
        assert "components" in data


class TestUncertaintyMap:
    """Test UncertaintyMap dataclass."""

    def test_creation(self):
        """Test creating an uncertainty map."""
        from core.analysis.fusion.uncertainty import UncertaintyMap

        uncertainty = np.random.rand(50, 50) * 0.1
        umap = UncertaintyMap(uncertainty=uncertainty)

        assert umap.uncertainty.shape == (50, 50)
        assert umap.confidence_level == 0.68  # Default 1-sigma

    def test_with_bounds(self):
        """Test with confidence bounds."""
        from core.analysis.fusion.uncertainty import UncertaintyMap

        uncertainty = np.ones((50, 50)) * 0.1
        lower = np.zeros((50, 50))
        upper = np.ones((50, 50))

        umap = UncertaintyMap(
            uncertainty=uncertainty,
            confidence_level=0.95,
            lower_bound=lower,
            upper_bound=upper,
        )

        assert umap.confidence_level == 0.95
        assert umap.lower_bound is not None
        assert umap.upper_bound is not None


class TestUncertaintyPropagator:
    """Test UncertaintyPropagator class."""

    def test_creation(self):
        """Test creating uncertainty propagator."""
        from core.analysis.fusion.uncertainty import UncertaintyPropagator

        propagator = UncertaintyPropagator()
        assert propagator is not None


class TestUncertaintyCombiner:
    """Test UncertaintyCombiner class."""

    def test_creation(self):
        """Test creating uncertainty combiner."""
        from core.analysis.fusion.uncertainty import UncertaintyCombiner

        combiner = UncertaintyCombiner()
        assert combiner is not None


class TestFusionUncertaintyEstimator:
    """Test FusionUncertaintyEstimator class."""

    def test_creation(self):
        """Test creating fusion uncertainty estimator."""
        from core.analysis.fusion.uncertainty import FusionUncertaintyEstimator

        estimator = FusionUncertaintyEstimator()
        assert estimator is not None


class TestUncertaintyConvenienceFunctions:
    """Test convenience functions."""

    def test_estimate_uncertainty_from_samples(self):
        """Test estimate_uncertainty_from_samples function."""
        from core.analysis.fusion.uncertainty import estimate_uncertainty_from_samples

        # Create samples with known statistics
        np.random.seed(42)
        samples = np.random.randn(100, 50, 50) * 0.1 + 1.0  # Mean 1.0, std ~0.1

        uncertainty_map = estimate_uncertainty_from_samples(samples)

        assert uncertainty_map.uncertainty.shape == (50, 50)
        # Should be close to 0.1
        assert np.abs(np.mean(uncertainty_map.uncertainty) - 0.1) < 0.02

    def test_combine_uncertainties(self):
        """Test combine_uncertainties function."""
        from core.analysis.fusion.uncertainty import combine_uncertainties

        # combine_uncertainties takes arrays, not UncertaintyMap objects
        u1 = np.ones((50, 50)) * 0.3
        u2 = np.ones((50, 50)) * 0.4

        combined = combine_uncertainties([u1, u2])

        # Should be sqrt(0.3^2 + 0.4^2) = 0.5
        assert np.isclose(np.mean(combined), 0.5, atol=0.01)


# ============================================================================
# Integration Tests
# ============================================================================

class TestFusionCoreIntegration:
    """Integration tests for fusion core modules."""

    def test_module_imports(self):
        """Test all module imports work correctly."""
        from core.analysis.fusion import (
            # Alignment
            SpatialAlignmentMethod,
            TemporalAlignmentMethod,
            AlignmentQuality,
            ReferenceGrid,
            TemporalBin,
            MultiSensorAligner,
            create_reference_grid,
            align_datasets,
            # Corrections
            TerrainCorrectionMethod,
            AtmosphericCorrectionMethod,
            NormalizationMethod,
            TerrainCorrector,
            AtmosphericCorrector,
            CorrectionPipeline,
            apply_terrain_correction,
            apply_atmospheric_correction,
            # Conflict
            ConflictResolutionStrategy,
            ConflictSeverity,
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
            UncertaintyPropagator,
            UncertaintyCombiner,
            FusionUncertaintyEstimator,
            combine_uncertainties,
        )

        # All imports succeeded
        assert True

    def test_alignment_to_conflict_pipeline(self):
        """Test a pipeline from alignment to conflict resolution."""
        from core.analysis.fusion import (
            ReferenceGrid,
            SourceLayer,
            resolve_conflicts,
        )

        # Create reference grid
        grid = ReferenceGrid(
            crs="EPSG:32618",
            bounds=(0, 0, 1000, 1000),
            resolution_x=10.0,
            resolution_y=10.0,
        )

        # Create layers as if they were aligned
        data1 = np.random.rand(100, 100).astype(np.float32)
        data2 = data1 + np.random.rand(100, 100).astype(np.float32) * 0.1  # Similar data

        layer1 = SourceLayer(data=data1, source_id="sar")
        layer2 = SourceLayer(data=data2, source_id="optical")

        # Resolve conflicts
        result = resolve_conflicts([layer1, layer2])

        assert result.resolved_data.shape == (100, 100)
        assert result.confidence_map.shape == (100, 100)

    def test_correction_to_uncertainty_pipeline(self):
        """Test corrections with uncertainty propagation."""
        from core.analysis.fusion import (
            apply_terrain_correction,
            UncertaintyComponent,
            UncertaintySource,
            UncertaintyBudget,
        )

        # Apply terrain correction
        data = np.random.rand(50, 50).astype(np.float32) * 1000
        dem = np.random.rand(50, 50).astype(np.float32) * 500

        result = apply_terrain_correction(data, dem=dem)

        # Create uncertainty budget
        components = [
            UncertaintyComponent(
                source=UncertaintySource.SENSOR,
                value=np.ones((50, 50)) * 0.05,
            ),
            UncertaintyComponent(
                source=UncertaintySource.GEOMETRIC,
                value=np.ones((50, 50)) * 0.02,
            ),
        ]

        budget = UncertaintyBudget(components=components)

        assert result.corrected_data.shape == (50, 50)
        assert budget.total_uncertainty is not None


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_source_layers(self):
        """Test handling of empty source layer list."""
        from core.analysis.fusion.conflict import resolve_conflicts

        # Should handle empty list gracefully
        with pytest.raises((ValueError, IndexError)):
            resolve_conflicts([])

    def test_single_source_layer(self):
        """Test conflict resolution with single source."""
        from core.analysis.fusion.conflict import resolve_conflicts, SourceLayer

        data = np.random.rand(50, 50).astype(np.float32)
        layer = SourceLayer(data=data, source_id="only_source")

        result = resolve_conflicts([layer])

        # Single source should pass through
        assert np.allclose(result.resolved_data, data, equal_nan=True)

    def test_all_nan_data(self):
        """Test handling of all-NaN data."""
        from core.analysis.fusion.conflict import resolve_conflicts, SourceLayer

        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data2 = np.full((50, 50), np.nan, dtype=np.float32)

        layer1 = SourceLayer(data=data1, source_id="source1")
        layer2 = SourceLayer(data=data2, source_id="source2")

        result = resolve_conflicts([layer1, layer2])

        # Should return NaN for all-NaN inputs
        assert np.all(np.isnan(result.resolved_data))

    def test_zero_uncertainty(self):
        """Test handling of zero uncertainty."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        components = [
            UncertaintyComponent(source=UncertaintySource.SENSOR, value=0.0),
        ]

        budget = UncertaintyBudget(components=components)

        assert budget.total_uncertainty == 0.0

    def test_negative_resolution_handling(self):
        """Test handling of negative resolution in grid."""
        from core.analysis.fusion.alignment import ReferenceGrid

        # Negative resolution should work (indicates south-up orientation)
        grid = ReferenceGrid(
            crs="EPSG:4326",
            bounds=(-180, -90, 180, 90),
            resolution_x=1.0,
            resolution_y=-1.0,  # Negative
        )

        assert grid.width == 360
        # Height calculation should use absolute value
        assert grid.height == 180


class TestPerformance:
    """Performance-related tests."""

    def test_large_array_conflict_resolution(self):
        """Test conflict resolution with larger arrays."""
        from core.analysis.fusion.conflict import resolve_conflicts, SourceLayer

        # Create larger test data
        shape = (500, 500)
        data1 = np.random.rand(*shape).astype(np.float32)
        data2 = np.random.rand(*shape).astype(np.float32)
        data3 = np.random.rand(*shape).astype(np.float32)

        layers = [
            SourceLayer(data=data1, source_id="s1"),
            SourceLayer(data=data2, source_id="s2"),
            SourceLayer(data=data3, source_id="s3"),
        ]

        # Should complete without issues
        result = resolve_conflicts(layers)

        assert result.resolved_data.shape == shape

    def test_many_uncertainty_components(self):
        """Test uncertainty budget with many components."""
        from core.analysis.fusion.uncertainty import (
            UncertaintyBudget,
            UncertaintyComponent,
            UncertaintySource,
        )

        # Create many components
        sources = [UncertaintySource.SENSOR, UncertaintySource.ALGORITHM,
                   UncertaintySource.ATMOSPHERIC, UncertaintySource.GEOMETRIC]

        components = [
            UncertaintyComponent(
                source=sources[i % len(sources)],
                value=np.random.rand(100, 100) * 0.1,
            )
            for i in range(10)
        ]

        budget = UncertaintyBudget(components=components)

        assert budget.total_uncertainty is not None
        assert budget.dominant_source is not None
