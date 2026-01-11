"""
Tests for multi-criteria ranking system.

Tests weighted scoring across multiple criteria, provider preference integration,
and trade-off documentation for data source selection.
"""

import pytest
from datetime import datetime, timezone, timedelta
from core.data.evaluation.ranking import (
    RankingCriteria,
    RankedCandidate,
    TradeOffRecord,
    MultiCriteriaRanker,
    RESOLUTION_CHARACTERISTIC_LENGTH_M,
    TEMPORAL_HALFLIFE_DAYS,
    SECONDS_PER_DAY,
)
from core.data.discovery.base import DiscoveryResult
from core.data.providers.registry import ProviderRegistry, Provider


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_candidates():
    """Create sample discovery results for testing."""
    base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    return [
        DiscoveryResult(
            dataset_id="sentinel1_high_coverage",
            provider="esa_sentinel1",
            data_type="sar",
            source_uri="s3://sentinel1/data1.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=95.0,
            resolution_m=10.0,
            cloud_cover_percent=None,  # SAR not affected
            quality_flag="excellent",
            cost_tier="open"
        ),
        DiscoveryResult(
            dataset_id="sentinel2_clear_skies",
            provider="esa_sentinel2",
            data_type="optical",
            source_uri="s3://sentinel2/data1.tif",
            format="cog",
            acquisition_time=base_time - timedelta(days=1),
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            cloud_cover_percent=5.0,
            quality_flag="excellent",
            cost_tier="open"
        ),
        DiscoveryResult(
            dataset_id="landsat_cloudy",
            provider="usgs_landsat",
            data_type="optical",
            source_uri="s3://landsat/data1.tif",
            format="cog",
            acquisition_time=base_time - timedelta(days=5),
            spatial_coverage_percent=90.0,
            resolution_m=30.0,
            cloud_cover_percent=45.0,
            quality_flag="fair",
            cost_tier="open"
        ),
        DiscoveryResult(
            dataset_id="planet_high_res",
            provider="planet_labs",
            data_type="optical",
            source_uri="https://planet.com/data1.tif",
            format="geotiff",
            acquisition_time=base_time,
            spatial_coverage_percent=85.0,
            resolution_m=3.0,
            cloud_cover_percent=10.0,
            quality_flag="good",
            cost_tier="commercial"
        ),
        DiscoveryResult(
            dataset_id="copernicus_dem",
            provider="copernicus",
            data_type="dem",
            source_uri="s3://copernicus/dem.tif",
            format="cog",
            acquisition_time=base_time - timedelta(days=365),
            spatial_coverage_percent=100.0,
            resolution_m=30.0,
            cloud_cover_percent=None,
            quality_flag="excellent",
            cost_tier="open"
        ),
    ]


@pytest.fixture
def query_context():
    """Create sample query context."""
    return {
        "temporal": {
            "start": "2024-09-14T00:00:00Z",
            "end": "2024-09-16T00:00:00Z",
            "reference_time": "2024-09-15T12:00:00Z"
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[[-80, 25], [-80, 26], [-79, 26], [-79, 25], [-80, 25]]]
        },
        "intent_class": "flood.coastal"
    }


@pytest.fixture
def default_ranker():
    """Create ranker with default criteria."""
    return MultiCriteriaRanker()


# ============================================================================
# RANKING CRITERIA TESTS
# ============================================================================

class TestRankingCriteria:
    """Test ranking criteria configuration."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        criteria = RankingCriteria()
        criteria.validate()  # Should not raise

    def test_custom_weights_validation_pass(self):
        """Test validation passes for valid weights."""
        criteria = RankingCriteria(
            spatial_coverage=0.3,
            resolution=0.2,
            temporal_proximity=0.2,
            cloud_cover=0.1,
            data_quality=0.1,
            access_cost=0.05,
            provider_preference=0.05
        )
        criteria.validate()  # Should not raise

    def test_custom_weights_validation_fail(self):
        """Test validation fails for invalid weights."""
        criteria = RankingCriteria(
            spatial_coverage=0.5,
            resolution=0.3,
            # Others sum to 0.45, total = 1.25
        )
        with pytest.raises(ValueError, match="must sum to ~1.0"):
            criteria.validate()

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        criteria = RankingCriteria()
        weights = criteria.to_dict()

        assert isinstance(weights, dict)
        assert weights["spatial_coverage"] == 0.25
        assert weights["resolution"] == 0.15
        assert weights["temporal_proximity"] == 0.20
        assert len(weights) == 7

    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        weights = {
            "spatial_coverage": 0.3,
            "resolution": 0.2,
            "temporal_proximity": 0.2,
            "cloud_cover": 0.1,
            "data_quality": 0.1,
            "access_cost": 0.05,
            "provider_preference": 0.05
        }
        criteria = RankingCriteria.from_dict(weights)

        assert criteria.spatial_coverage == 0.3
        assert criteria.resolution == 0.2
        criteria.validate()


# ============================================================================
# RANKED CANDIDATE TESTS
# ============================================================================

class TestRankedCandidate:
    """Test ranked candidate dataclass."""

    def test_creation(self, sample_candidates):
        """Test creation of ranked candidate."""
        candidate = RankedCandidate(
            candidate=sample_candidates[0],
            scores={"spatial_coverage": 0.95, "resolution": 0.9},
            total_score=0.925,
            rank=1
        )

        assert candidate.rank == 1
        assert candidate.total_score == 0.925
        assert candidate.scores["resolution"] == 0.9

    def test_to_dict_conversion(self, sample_candidates):
        """Test conversion to dictionary."""
        candidate = RankedCandidate(
            candidate=sample_candidates[0],
            scores={"spatial_coverage": 0.95},
            total_score=0.8,
            rank=2
        )

        data = candidate.to_dict()

        assert data["rank"] == 2
        assert data["total_score"] == 0.8
        assert "candidate" in data
        assert data["candidate"]["dataset_id"] == "sentinel1_high_coverage"


# ============================================================================
# TRADE-OFF RECORD TESTS
# ============================================================================

class TestTradeOffRecord:
    """Test trade-off record dataclass."""

    def test_timestamp_uses_utc(self):
        """Test that timestamp uses UTC timezone."""
        record = TradeOffRecord(
            decision_context="Test selection",
            selected_id="dataset1",
            selected_score=0.9,
            alternatives=[],
            rationale="Best match"
        )

        # Should have timezone info
        assert record.timestamp.tzinfo is not None
        assert record.timestamp.tzinfo == timezone.utc

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        record = TradeOffRecord(
            decision_context="Select SAR for flood",
            selected_id="sentinel1_data",
            selected_score=0.85,
            alternatives=[
                {"id": "alt1", "score": 0.70, "reason": "lower resolution"}
            ],
            rationale="High coverage and resolution"
        )

        data = record.to_dict()

        assert data["selected_id"] == "sentinel1_data"
        assert data["selected_score"] == 0.85
        assert len(data["alternatives"]) == 1
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)  # ISO format


# ============================================================================
# MULTI-CRITERIA RANKER TESTS
# ============================================================================

class TestMultiCriteriaRanker:
    """Test multi-criteria ranking engine."""

    def test_initialization_default(self):
        """Test initialization with defaults."""
        ranker = MultiCriteriaRanker()

        assert ranker.provider_registry is not None
        assert ranker.criteria is not None
        assert len(ranker._scorers) == 7

    def test_initialization_custom_criteria(self):
        """Test initialization with custom criteria."""
        custom_criteria = RankingCriteria(
            spatial_coverage=0.4,
            resolution=0.3,
            temporal_proximity=0.1,
            cloud_cover=0.1,
            data_quality=0.05,
            access_cost=0.025,
            provider_preference=0.025
        )

        ranker = MultiCriteriaRanker(criteria=custom_criteria)

        assert ranker.criteria.spatial_coverage == 0.4

    def test_rank_candidates_basic(self, default_ranker, sample_candidates, query_context):
        """Test basic ranking of candidates."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)

        assert len(ranked) == len(sample_candidates)
        # Check ranks are assigned sequentially
        assert ranked[0].rank == 1
        assert ranked[-1].rank == len(sample_candidates)
        # Check sorted by total_score descending
        for i in range(len(ranked) - 1):
            assert ranked[i].total_score >= ranked[i + 1].total_score

    def test_rank_candidates_sar_scores_high(self, default_ranker, sample_candidates, query_context):
        """Test that SAR with high coverage scores well."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)

        # Find the SAR candidate
        sar_ranked = [r for r in ranked if r.candidate.dataset_id == "sentinel1_high_coverage"][0]

        # Should have high spatial coverage score
        assert sar_ranked.scores["spatial_coverage"] > 0.9
        # SAR has no cloud cover penalty
        assert sar_ranked.scores["cloud_cover"] == 1.0

    def test_rank_candidates_cloud_cover_penalty(self, default_ranker, sample_candidates, query_context):
        """Test cloud cover penalizes optical imagery."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)

        # Find cloudy Landsat
        landsat = [r for r in ranked if r.candidate.dataset_id == "landsat_cloudy"][0]
        # Find clear Sentinel-2
        sentinel2 = [r for r in ranked if r.candidate.dataset_id == "sentinel2_clear_skies"][0]

        # Sentinel-2 should score higher on cloud cover
        assert sentinel2.scores["cloud_cover"] > landsat.scores["cloud_cover"]

    def test_rank_by_data_type(self, default_ranker, sample_candidates, query_context):
        """Test ranking grouped by data type."""
        ranked_by_type = default_ranker.rank_by_data_type(sample_candidates, query_context)

        assert "sar" in ranked_by_type
        assert "optical" in ranked_by_type
        assert "dem" in ranked_by_type

        # Check optical group has 3 members
        assert len(ranked_by_type["optical"]) == 3
        # Check they're ranked within their type
        assert ranked_by_type["optical"][0].rank == 1

    def test_rank_empty_list(self, default_ranker, query_context):
        """Test ranking empty candidate list."""
        ranked = default_ranker.rank_candidates([], query_context)

        assert len(ranked) == 0

    def test_update_criteria(self, default_ranker):
        """Test updating ranking criteria."""
        new_criteria = RankingCriteria(
            spatial_coverage=0.5,
            resolution=0.2,
            temporal_proximity=0.1,
            cloud_cover=0.1,
            data_quality=0.05,
            access_cost=0.025,
            provider_preference=0.025
        )

        default_ranker.update_criteria(new_criteria)

        assert default_ranker.criteria.spatial_coverage == 0.5

    def test_update_criteria_validation(self, default_ranker):
        """Test that invalid criteria are rejected."""
        bad_criteria = RankingCriteria(
            spatial_coverage=0.9,
            # Others default to much less than 0.1 total
        )

        with pytest.raises(ValueError):
            default_ranker.update_criteria(bad_criteria)

    def test_get_criteria_summary(self, default_ranker):
        """Test criteria summary string generation."""
        summary = default_ranker.get_criteria_summary()

        assert "Current Ranking Criteria:" in summary
        assert "Spatial Coverage:" in summary
        assert "%" in summary  # Percentages


# ============================================================================
# SCORING FUNCTION TESTS
# ============================================================================

class TestScoringFunctions:
    """Test individual scoring functions."""

    def test_score_spatial_coverage_full(self, default_ranker, sample_candidates, query_context):
        """Test spatial coverage scoring at 100%."""
        candidate = sample_candidates[1]  # 100% coverage
        score = default_ranker._score_spatial_coverage(candidate, query_context)

        assert score == 1.0

    def test_score_spatial_coverage_partial(self, default_ranker, sample_candidates, query_context):
        """Test spatial coverage scoring at partial coverage."""
        candidate = sample_candidates[3]  # 85% coverage
        score = default_ranker._score_spatial_coverage(candidate, query_context)

        assert score == 0.85

    def test_score_resolution_high(self, default_ranker, sample_candidates, query_context):
        """Test resolution scoring for high resolution."""
        candidate = sample_candidates[3]  # 3m resolution
        score = default_ranker._score_resolution(candidate, query_context)

        # exp(-3/100) ≈ 0.97
        assert score > 0.95

    def test_score_resolution_medium(self, default_ranker, sample_candidates, query_context):
        """Test resolution scoring for medium resolution."""
        candidate = sample_candidates[0]  # 10m resolution
        score = default_ranker._score_resolution(candidate, query_context)

        # exp(-10/100) ≈ 0.90
        assert 0.89 < score < 0.91

    def test_score_resolution_low(self, default_ranker, sample_candidates, query_context):
        """Test resolution scoring for low resolution."""
        candidate = sample_candidates[2]  # 30m resolution
        score = default_ranker._score_resolution(candidate, query_context)

        # exp(-30/100) ≈ 0.74
        assert 0.73 < score < 0.75

    def test_score_temporal_proximity_exact(self, default_ranker, sample_candidates, query_context):
        """Test temporal proximity at reference time."""
        candidate = sample_candidates[0]  # Exact match
        score = default_ranker._score_temporal_proximity(candidate, query_context)

        assert score == 1.0

    def test_score_temporal_proximity_one_day(self, default_ranker, sample_candidates, query_context):
        """Test temporal proximity 1 day away."""
        candidate = sample_candidates[1]  # 1 day before
        score = default_ranker._score_temporal_proximity(candidate, query_context)

        # exp(-1/7) ≈ 0.87
        assert 0.85 < score < 0.88

    def test_score_temporal_proximity_week(self, default_ranker, sample_candidates, query_context):
        """Test temporal proximity 1 week away."""
        # Create candidate 7 days away
        candidate = DiscoveryResult(
            dataset_id="test",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime(2024, 9, 8, 12, 0, 0, tzinfo=timezone.utc),
            spatial_coverage_percent=100,
            resolution_m=10
        )
        score = default_ranker._score_temporal_proximity(candidate, query_context)

        # exp(-7/7) = exp(-1) ≈ 0.37
        assert 0.35 < score < 0.39

    def test_score_temporal_proximity_no_reference(self, default_ranker, sample_candidates):
        """Test temporal scoring without reference time."""
        candidate = sample_candidates[0]
        context = {"temporal": {}}  # No reference time
        score = default_ranker._score_temporal_proximity(candidate, context)

        assert score == 0.5  # Neutral score

    def test_score_cloud_cover_clear(self, default_ranker, sample_candidates, query_context):
        """Test cloud cover scoring for clear skies."""
        candidate = sample_candidates[1]  # 5% clouds
        score = default_ranker._score_cloud_cover(candidate, query_context)

        assert score == 0.95

    def test_score_cloud_cover_cloudy(self, default_ranker, sample_candidates, query_context):
        """Test cloud cover scoring for cloudy skies."""
        candidate = sample_candidates[2]  # 45% clouds
        score = default_ranker._score_cloud_cover(candidate, query_context)

        assert score == 0.55

    def test_score_cloud_cover_not_applicable(self, default_ranker, sample_candidates, query_context):
        """Test cloud cover scoring for non-optical data."""
        candidate = sample_candidates[0]  # SAR, no cloud cover
        score = default_ranker._score_cloud_cover(candidate, query_context)

        assert score == 1.0  # No penalty

    def test_score_data_quality_excellent(self, default_ranker, sample_candidates, query_context):
        """Test quality scoring for excellent data."""
        candidate = sample_candidates[0]  # excellent
        score = default_ranker._score_data_quality(candidate, query_context)

        assert score == 1.0

    def test_score_data_quality_good(self, default_ranker, sample_candidates, query_context):
        """Test quality scoring for good data."""
        candidate = sample_candidates[3]  # good
        score = default_ranker._score_data_quality(candidate, query_context)

        assert score == 0.8

    def test_score_data_quality_fair(self, default_ranker, sample_candidates, query_context):
        """Test quality scoring for fair data."""
        candidate = sample_candidates[2]  # fair
        score = default_ranker._score_data_quality(candidate, query_context)

        assert score == 0.6

    def test_score_access_cost_open(self, default_ranker, sample_candidates, query_context):
        """Test cost scoring for open data."""
        candidate = sample_candidates[0]  # open
        score = default_ranker._score_access_cost(candidate, query_context)

        assert score == 1.0

    def test_score_access_cost_commercial(self, default_ranker, sample_candidates, query_context):
        """Test cost scoring for commercial data."""
        candidate = sample_candidates[3]  # commercial
        score = default_ranker._score_access_cost(candidate, query_context)

        assert score == 0.3

    def test_score_provider_preference_unknown(self, default_ranker, sample_candidates, query_context):
        """Test provider preference for unknown provider."""
        candidate = sample_candidates[0]
        score = default_ranker._score_provider_preference(candidate, query_context)

        # Unknown provider gets neutral score
        assert score == 0.5


# ============================================================================
# TRADE-OFF DOCUMENTATION TESTS
# ============================================================================

class TestTradeOffDocumentation:
    """Test trade-off documentation generation."""

    def test_document_trade_offs_basic(self, default_ranker, sample_candidates, query_context):
        """Test basic trade-off documentation."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)
        trade_offs = default_ranker.document_trade_offs(ranked, top_n=3)

        assert len(trade_offs) == 1
        record = trade_offs[0]

        assert record.selected_id == ranked[0].candidate.dataset_id
        assert record.selected_score == ranked[0].total_score
        assert len(record.alternatives) <= 3

    def test_document_trade_offs_empty(self, default_ranker, query_context):
        """Test trade-off documentation with empty list."""
        trade_offs = default_ranker.document_trade_offs([])

        assert len(trade_offs) == 0

    def test_document_trade_offs_single_candidate(self, default_ranker, sample_candidates, query_context):
        """Test trade-off documentation with single candidate."""
        ranked = default_ranker.rank_candidates([sample_candidates[0]], query_context)
        trade_offs = default_ranker.document_trade_offs(ranked, top_n=3)

        assert len(trade_offs) == 1
        # No alternatives since only 1 candidate
        assert len(trade_offs[0].alternatives) == 0

    def test_document_trade_offs_rationale(self, default_ranker, sample_candidates, query_context):
        """Test that rationale is generated."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)
        trade_offs = default_ranker.document_trade_offs(ranked)

        assert trade_offs[0].rationale is not None
        assert len(trade_offs[0].rationale) > 0

    def test_document_trade_offs_alternatives_have_reasons(self, default_ranker, sample_candidates, query_context):
        """Test that alternatives include rejection reasons."""
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)
        trade_offs = default_ranker.document_trade_offs(ranked, top_n=2)

        for alt in trade_offs[0].alternatives:
            assert "id" in alt
            assert "score" in alt
            assert "rank" in alt
            assert "reason" in alt
            assert len(alt["reason"]) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRankingIntegration:
    """Integration tests for complete ranking workflows."""

    def test_complete_ranking_workflow(self, default_ranker, sample_candidates, query_context):
        """Test complete ranking workflow."""
        # Rank candidates
        ranked = default_ranker.rank_candidates(sample_candidates, query_context)

        # Generate trade-off documentation
        trade_offs = default_ranker.document_trade_offs(ranked, top_n=3)

        # Verify results
        assert len(ranked) == len(sample_candidates)
        assert len(trade_offs) == 1
        assert ranked[0].rank == 1
        assert trade_offs[0].selected_id == ranked[0].candidate.dataset_id

    def test_optical_vs_sar_preference(self, default_ranker, query_context):
        """Test SAR vs optical preference in cloudy conditions."""
        base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

        candidates = [
            DiscoveryResult(
                dataset_id="sar_data",
                provider="esa",
                data_type="sar",
                source_uri="sar.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100,
                resolution_m=10,
                cost_tier="open",
                quality_flag="excellent"
            ),
            DiscoveryResult(
                dataset_id="optical_cloudy",
                provider="esa",
                data_type="optical",
                source_uri="optical.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100,
                resolution_m=10,
                cloud_cover_percent=70.0,  # Very cloudy
                cost_tier="open",
                quality_flag="excellent"
            ),
        ]

        ranked = default_ranker.rank_candidates(candidates, query_context)

        # SAR should rank higher due to cloud independence
        assert ranked[0].candidate.dataset_id == "sar_data"

    def test_resolution_vs_cost_tradeoff(self, default_ranker, query_context):
        """Test trade-off between high resolution and cost."""
        base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

        candidates = [
            DiscoveryResult(
                dataset_id="high_res_commercial",
                provider="planet",
                data_type="optical",
                source_uri="planet.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100,
                resolution_m=3.0,  # Very high resolution
                cloud_cover_percent=5.0,
                cost_tier="commercial",  # Expensive
                quality_flag="excellent"
            ),
            DiscoveryResult(
                dataset_id="medium_res_open",
                provider="esa",
                data_type="optical",
                source_uri="sentinel.tif",
                format="cog",
                acquisition_time=base_time,
                spatial_coverage_percent=100,
                resolution_m=10.0,
                cloud_cover_percent=5.0,
                cost_tier="open",  # Free
                quality_flag="excellent"
            ),
        ]

        ranked = default_ranker.rank_candidates(candidates, query_context)

        # With default weights, free open data should generally win
        # But test documents the trade-off
        trade_offs = default_ranker.document_trade_offs(ranked)
        assert len(trade_offs[0].alternatives) > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_spatial_coverage(self, default_ranker, query_context):
        """Test handling of zero spatial coverage."""
        candidate = DiscoveryResult(
            dataset_id="no_coverage",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=0.0,
            resolution_m=10
        )

        score = default_ranker._score_spatial_coverage(candidate, query_context)
        assert score == 0.0

    def test_very_high_resolution(self, default_ranker, query_context):
        """Test scoring for very high resolution (<1m)."""
        candidate = DiscoveryResult(
            dataset_id="ultra_high_res",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100,
            resolution_m=0.5  # 50cm
        )

        score = default_ranker._score_resolution(candidate, query_context)
        # Should be capped at 1.0
        assert score <= 1.0
        assert score > 0.99

    def test_very_old_data(self, default_ranker, query_context):
        """Test scoring for very old data."""
        candidate = DiscoveryResult(
            dataset_id="old_data",
            provider="test",
            data_type="dem",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            spatial_coverage_percent=100,
            resolution_m=30
        )

        score = default_ranker._score_temporal_proximity(candidate, query_context)
        # Should be very low but not zero
        assert score < 0.01
        assert score > 0.0

    def test_missing_optional_fields(self, default_ranker, query_context):
        """Test handling of missing optional fields."""
        candidate = DiscoveryResult(
            dataset_id="minimal",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100,
            resolution_m=10
            # All optional fields left as None
        )

        # Should not crash
        ranked = default_ranker.rank_candidates([candidate], query_context)
        assert len(ranked) == 1
        assert ranked[0].total_score > 0


# ============================================================================
# CONSTANTS TESTS
# ============================================================================

class TestConstants:
    """Test that constants are properly defined."""

    def test_resolution_constant_defined(self):
        """Test resolution characteristic length constant."""
        assert RESOLUTION_CHARACTERISTIC_LENGTH_M == 100.0

    def test_temporal_constant_defined(self):
        """Test temporal halflife constant."""
        assert TEMPORAL_HALFLIFE_DAYS == 7.0

    def test_seconds_per_day_constant_defined(self):
        """Test seconds per day constant."""
        assert SECONDS_PER_DAY == 86400.0
