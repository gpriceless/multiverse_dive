"""
Tests for constraint evaluation engine.

Tests hard constraints (pass/fail), soft constraints (scored 0.0-1.0),
and the constraint evaluation engine with structured evaluation results.
"""

import math
import pytest
from datetime import datetime, timezone, timedelta
from core.data.evaluation.constraints import (
    HardConstraint,
    SoftConstraint,
    EvaluationResult,
    ConstraintEvaluator,
    evaluate_candidates,
    get_passing_candidates
)
from core.data.discovery.base import DiscoveryResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_discovery_results():
    """Create sample discovery results for testing."""
    base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    return [
        DiscoveryResult(
            dataset_id="sentinel1_excellent",
            provider="esa_sentinel1",
            data_type="sar",
            source_uri="s3://sentinel1/data1.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=95.0,
            resolution_m=10.0,
            cloud_cover_percent=None,  # SAR not affected
            quality_flag="excellent",
            cost_tier="open",
            metadata={"available": True}
        ),
        DiscoveryResult(
            dataset_id="sentinel2_cloudy",
            provider="esa_sentinel2",
            data_type="optical",
            source_uri="s3://sentinel2/data2.tif",
            format="cog",
            acquisition_time=base_time - timedelta(hours=6),
            spatial_coverage_percent=80.0,
            resolution_m=10.0,
            cloud_cover_percent=35.0,
            quality_flag="good",
            cost_tier="open"
        ),
        DiscoveryResult(
            dataset_id="landsat_very_cloudy",
            provider="usgs_landsat",
            data_type="optical",
            source_uri="s3://landsat/data3.tif",
            format="cog",
            acquisition_time=base_time - timedelta(days=3),
            spatial_coverage_percent=60.0,
            resolution_m=30.0,
            cloud_cover_percent=75.0,
            quality_flag="fair",
            cost_tier="open"
        ),
        DiscoveryResult(
            dataset_id="planet_commercial",
            provider="planet_labs",
            data_type="optical",
            source_uri="https://planet.com/data4.tif",
            format="geotiff",
            acquisition_time=base_time,
            spatial_coverage_percent=100.0,
            resolution_m=3.0,
            cloud_cover_percent=5.0,
            quality_flag="excellent",
            cost_tier="commercial"
        ),
        DiscoveryResult(
            dataset_id="dem_copernicus",
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
        DiscoveryResult(
            dataset_id="low_coverage_sar",
            provider="alos",
            data_type="sar",
            source_uri="s3://alos/data5.tif",
            format="cog",
            acquisition_time=base_time - timedelta(hours=12),
            spatial_coverage_percent=30.0,  # Below typical threshold
            resolution_m=10.0,
            cloud_cover_percent=None,
            quality_flag="good",
            cost_tier="open_restricted"
        ),
        DiscoveryResult(
            dataset_id="unavailable_data",
            provider="test_provider",
            data_type="optical",
            source_uri="s3://test/unavailable.tif",
            format="cog",
            acquisition_time=base_time,
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            cloud_cover_percent=10.0,
            quality_flag="good",
            cost_tier="open",
            metadata={"available": False}
        ),
    ]


@pytest.fixture
def default_context():
    """Create default evaluation context."""
    return {
        "min_spatial_coverage": 0.5,  # 50%
        "max_cloud_cover": 0.5,  # 50%
        "max_resolution_m": 50.0,
        "temporal_window": {
            "start": "2024-09-14T00:00:00Z",
            "end": "2024-09-16T00:00:00Z",
            "reference_time": "2024-09-15T12:00:00Z"
        },
        "soft_weights": {}
    }


@pytest.fixture
def strict_context():
    """Create strict evaluation context."""
    return {
        "min_spatial_coverage": 0.8,  # 80%
        "max_cloud_cover": 0.2,  # 20%
        "max_resolution_m": 15.0,
        "temporal_window": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-15T23:59:59Z",
            "reference_time": "2024-09-15T12:00:00Z"
        }
    }


@pytest.fixture
def evaluator():
    """Create constraint evaluator with default constraints."""
    return ConstraintEvaluator()


# ============================================================================
# HARD CONSTRAINT TESTS
# ============================================================================

class TestHardConstraint:
    """Test hard constraint class."""

    def test_hard_constraint_creation(self):
        """Test creating a hard constraint."""
        constraint = HardConstraint(
            name="test_constraint",
            description="Test constraint description",
            check_function=lambda c, ctx: True
        )

        assert constraint.name == "test_constraint"
        assert constraint.description == "Test constraint description"
        assert constraint.applies_to_data_types is None

    def test_hard_constraint_applies_to_all_types(self):
        """Test constraint applies to all data types by default."""
        constraint = HardConstraint(
            name="universal",
            description="Applies to all",
            check_function=lambda c, ctx: True
        )

        assert constraint.applies_to("optical") is True
        assert constraint.applies_to("sar") is True
        assert constraint.applies_to("dem") is True
        assert constraint.applies_to("weather") is True

    def test_hard_constraint_applies_to_specific_types(self):
        """Test constraint applies only to specified types."""
        constraint = HardConstraint(
            name="optical_only",
            description="Only for optical",
            check_function=lambda c, ctx: True,
            applies_to_data_types=["optical"]
        )

        assert constraint.applies_to("optical") is True
        assert constraint.applies_to("sar") is False
        assert constraint.applies_to("dem") is False

    def test_hard_constraint_evaluate_pass(self, sample_discovery_results):
        """Test hard constraint evaluation passes."""
        constraint = HardConstraint(
            name="always_pass",
            description="Always passes",
            check_function=lambda c, ctx: True
        )

        passed, reason = constraint.evaluate(sample_discovery_results[0], {})

        assert passed is True
        assert "passed" in reason

    def test_hard_constraint_evaluate_fail(self, sample_discovery_results):
        """Test hard constraint evaluation fails."""
        constraint = HardConstraint(
            name="always_fail",
            description="Always fails",
            check_function=lambda c, ctx: False
        )

        passed, reason = constraint.evaluate(sample_discovery_results[0], {})

        assert passed is False
        assert "failed" in reason
        assert constraint.description in reason

    def test_hard_constraint_evaluate_exception(self, sample_discovery_results):
        """Test hard constraint handles exceptions gracefully."""
        def failing_check(candidate, context):
            raise ValueError("Test error")

        constraint = HardConstraint(
            name="error_constraint",
            description="Will error",
            check_function=failing_check
        )

        passed, reason = constraint.evaluate(sample_discovery_results[0], {})

        assert passed is False
        assert "error" in reason.lower()


# ============================================================================
# SOFT CONSTRAINT TESTS
# ============================================================================

class TestSoftConstraint:
    """Test soft constraint class."""

    def test_soft_constraint_creation(self):
        """Test creating a soft constraint."""
        constraint = SoftConstraint(
            name="quality_score",
            description="Scores quality",
            score_function=lambda c, ctx: 0.8,
            weight=1.5
        )

        assert constraint.name == "quality_score"
        assert constraint.weight == 1.5

    def test_soft_constraint_evaluate_returns_score(self, sample_discovery_results):
        """Test soft constraint returns score."""
        constraint = SoftConstraint(
            name="fixed_score",
            description="Returns fixed score",
            score_function=lambda c, ctx: 0.75
        )

        score, explanation = constraint.evaluate(sample_discovery_results[0], {})

        assert score == 0.75
        assert "0.750" in explanation

    def test_soft_constraint_score_clamped_high(self, sample_discovery_results):
        """Test soft constraint clamps scores above 1.0."""
        constraint = SoftConstraint(
            name="high_score",
            description="Returns high score",
            score_function=lambda c, ctx: 1.5
        )

        score, _ = constraint.evaluate(sample_discovery_results[0], {})

        assert score == 1.0

    def test_soft_constraint_score_clamped_low(self, sample_discovery_results):
        """Test soft constraint clamps scores below 0.0."""
        constraint = SoftConstraint(
            name="low_score",
            description="Returns low score",
            score_function=lambda c, ctx: -0.5
        )

        score, _ = constraint.evaluate(sample_discovery_results[0], {})

        assert score == 0.0

    def test_soft_constraint_evaluate_exception(self, sample_discovery_results):
        """Test soft constraint handles exceptions gracefully."""
        def failing_score(candidate, context):
            raise ValueError("Score error")

        constraint = SoftConstraint(
            name="error_score",
            description="Will error",
            score_function=failing_score
        )

        score, explanation = constraint.evaluate(sample_discovery_results[0], {})

        assert score == 0.5  # Neutral score on error
        assert "error" in explanation.lower()

    def test_soft_constraint_applies_to_types(self, sample_discovery_results):
        """Test soft constraint type filtering."""
        constraint = SoftConstraint(
            name="optical_score",
            description="Optical only",
            score_function=lambda c, ctx: 1.0,
            applies_to_data_types=["optical"]
        )

        assert constraint.applies_to("optical") is True
        assert constraint.applies_to("sar") is False


# ============================================================================
# EVALUATION RESULT TESTS
# ============================================================================

class TestEvaluationResult:
    """Test evaluation result class."""

    def test_evaluation_result_creation(self, sample_discovery_results):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            candidate=sample_discovery_results[0],
            passed_hard_constraints=True
        )

        assert result.passed_hard_constraints is True
        assert result.total_soft_score == 0.0
        assert len(result.failure_reasons) == 0

    def test_evaluation_result_to_dict(self, sample_discovery_results):
        """Test converting evaluation result to dict."""
        result = EvaluationResult(
            candidate=sample_discovery_results[0],
            passed_hard_constraints=True,
            hard_constraint_results={
                "spatial": (True, "passed"),
                "temporal": (True, "passed")
            },
            soft_constraint_scores={
                "coverage": (0.95, "coverage: 0.950"),
                "quality": (0.85, "quality: 0.850")
            },
            total_soft_score=0.9,
            failure_reasons=[]
        )

        data = result.to_dict()

        assert data["dataset_id"] == "sentinel1_excellent"
        assert data["passed"] is True
        assert "hard_constraints" in data
        assert data["hard_constraints"]["spatial"]["passed"] is True
        assert "soft_scores" in data
        assert data["soft_scores"]["coverage"]["score"] == 0.95
        assert data["total_score"] == 0.9

    def test_evaluation_result_with_failures(self, sample_discovery_results):
        """Test evaluation result with failures."""
        result = EvaluationResult(
            candidate=sample_discovery_results[2],
            passed_hard_constraints=False,
            failure_reasons=["cloud cover too high", "spatial coverage insufficient"]
        )

        assert result.passed_hard_constraints is False
        assert len(result.failure_reasons) == 2


# ============================================================================
# CONSTRAINT EVALUATOR TESTS
# ============================================================================

class TestConstraintEvaluator:
    """Test constraint evaluator engine."""

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes with default constraints."""
        assert len(evaluator.hard_constraints) > 0
        assert len(evaluator.soft_constraints) > 0

    def test_evaluator_register_hard_constraint(self, evaluator):
        """Test registering a new hard constraint."""
        initial_count = len(evaluator.hard_constraints)

        evaluator.register_hard_constraint(HardConstraint(
            name="custom_check",
            description="Custom constraint",
            check_function=lambda c, ctx: True
        ))

        assert len(evaluator.hard_constraints) == initial_count + 1

    def test_evaluator_register_soft_constraint(self, evaluator):
        """Test registering a new soft constraint."""
        initial_count = len(evaluator.soft_constraints)

        evaluator.register_soft_constraint(SoftConstraint(
            name="custom_score",
            description="Custom score",
            score_function=lambda c, ctx: 0.8
        ))

        assert len(evaluator.soft_constraints) == initial_count + 1

    def test_evaluate_single_candidate_pass(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test evaluating a single passing candidate."""
        candidate = sample_discovery_results[0]  # Excellent SAR
        result = evaluator.evaluate(candidate, default_context)

        assert result.passed_hard_constraints is True
        assert len(result.failure_reasons) == 0
        assert result.total_soft_score > 0

    def test_evaluate_single_candidate_fail(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test evaluating a candidate that fails hard constraints."""
        candidate = sample_discovery_results[2]  # Very cloudy Landsat (75% cloud)
        result = evaluator.evaluate(candidate, default_context)

        # Should fail cloud cover constraint (max 50%)
        assert result.passed_hard_constraints is False
        assert len(result.failure_reasons) > 0

    def test_evaluate_batch(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test batch evaluation of multiple candidates."""
        results = evaluator.evaluate_batch(sample_discovery_results, default_context)

        assert len(results) == len(sample_discovery_results)
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_filter_passing(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test filtering to only passing candidates."""
        results = evaluator.evaluate_batch(sample_discovery_results, default_context)
        passing = evaluator.filter_passing(results)

        assert all(r.passed_hard_constraints for r in passing)
        assert len(passing) < len(results)

    def test_soft_scores_computed_even_on_fail(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test that soft scores are computed even when hard constraints fail."""
        candidate = sample_discovery_results[2]  # Will fail
        result = evaluator.evaluate(candidate, default_context)

        # Soft scores should still be computed for diagnostics
        assert len(result.soft_constraint_scores) > 0

    def test_soft_weights_applied(self, evaluator, sample_discovery_results):
        """Test that custom soft weights are applied."""
        context = {
            "soft_weights": {
                "spatial_coverage_score": 2.0,
                "temporal_proximity_score": 0.5
            }
        }

        candidate = sample_discovery_results[0]
        result = evaluator.evaluate(candidate, context)

        # Total score should be computed with custom weights
        assert result.total_soft_score > 0


# ============================================================================
# DEFAULT CONSTRAINT CHECK TESTS
# ============================================================================

class TestDefaultHardConstraints:
    """Test default hard constraint check functions."""

    def test_min_spatial_coverage_pass(
        self, evaluator, sample_discovery_results
    ):
        """Test minimum spatial coverage constraint passes."""
        candidate = sample_discovery_results[0]  # 95% coverage
        context = {"min_spatial_coverage": 0.5}  # Require 50%

        passed = evaluator._check_min_spatial_coverage(candidate, context)

        assert passed is True

    def test_min_spatial_coverage_fail(
        self, evaluator, sample_discovery_results
    ):
        """Test minimum spatial coverage constraint fails."""
        candidate = sample_discovery_results[5]  # 30% coverage
        context = {"min_spatial_coverage": 0.5}  # Require 50%

        passed = evaluator._check_min_spatial_coverage(candidate, context)

        assert passed is False

    def test_min_spatial_coverage_default(
        self, evaluator, sample_discovery_results
    ):
        """Test minimum spatial coverage uses default."""
        candidate = sample_discovery_results[0]
        context = {}  # No explicit threshold

        # Default is 50%, candidate has 95%
        passed = evaluator._check_min_spatial_coverage(candidate, context)

        assert passed is True

    def test_max_cloud_cover_pass(self, evaluator, sample_discovery_results):
        """Test cloud cover constraint passes for clear imagery."""
        candidate = sample_discovery_results[3]  # 5% cloud cover
        context = {"max_cloud_cover": 0.5}

        passed = evaluator._check_max_cloud_cover(candidate, context)

        assert passed is True

    def test_max_cloud_cover_fail(self, evaluator, sample_discovery_results):
        """Test cloud cover constraint fails for cloudy imagery."""
        candidate = sample_discovery_results[2]  # 75% cloud cover
        context = {"max_cloud_cover": 0.5}

        passed = evaluator._check_max_cloud_cover(candidate, context)

        assert passed is False

    def test_max_cloud_cover_sar_passes(self, evaluator, sample_discovery_results):
        """Test SAR always passes cloud cover (cloud cover is None)."""
        candidate = sample_discovery_results[0]  # SAR, no cloud cover
        context = {"max_cloud_cover": 0.1}  # Very strict

        passed = evaluator._check_max_cloud_cover(candidate, context)

        assert passed is True  # None cloud cover always passes

    def test_max_resolution_pass(self, evaluator, sample_discovery_results):
        """Test resolution constraint passes."""
        candidate = sample_discovery_results[0]  # 10m resolution
        context = {"max_resolution_m": 30.0}

        passed = evaluator._check_max_resolution(candidate, context)

        assert passed is True

    def test_max_resolution_fail(self, evaluator, sample_discovery_results):
        """Test resolution constraint fails."""
        candidate = sample_discovery_results[4]  # 30m resolution
        context = {"max_resolution_m": 15.0}

        passed = evaluator._check_max_resolution(candidate, context)

        assert passed is False

    def test_temporal_window_pass(self, evaluator, sample_discovery_results):
        """Test temporal window constraint passes."""
        candidate = sample_discovery_results[0]  # 2024-09-15T12:00:00Z
        context = {
            "temporal_window": {
                "start": "2024-09-14T00:00:00Z",
                "end": "2024-09-16T00:00:00Z"
            }
        }

        passed = evaluator._check_temporal_window(candidate, context)

        assert passed is True

    def test_temporal_window_fail(self, evaluator, sample_discovery_results):
        """Test temporal window constraint fails."""
        candidate = sample_discovery_results[4]  # 365 days ago
        context = {
            "temporal_window": {
                "start": "2024-09-14T00:00:00Z",
                "end": "2024-09-16T00:00:00Z"
            }
        }

        passed = evaluator._check_temporal_window(candidate, context)

        assert passed is False

    def test_temporal_window_no_constraint(self, evaluator, sample_discovery_results):
        """Test temporal window passes when not specified."""
        candidate = sample_discovery_results[4]  # 365 days ago
        context = {}  # No temporal constraint

        passed = evaluator._check_temporal_window(candidate, context)

        assert passed is True


# ============================================================================
# DEFAULT SOFT CONSTRAINT SCORE TESTS
# ============================================================================

class TestDefaultSoftConstraints:
    """Test default soft constraint scoring functions."""

    def test_score_spatial_coverage_full(self, evaluator, sample_discovery_results):
        """Test spatial coverage score for 100% coverage."""
        candidate = sample_discovery_results[3]  # 100% coverage
        score = evaluator._score_spatial_coverage(candidate, {})

        assert score == 1.0

    def test_score_spatial_coverage_partial(self, evaluator, sample_discovery_results):
        """Test spatial coverage score for partial coverage."""
        candidate = sample_discovery_results[1]  # 80% coverage
        score = evaluator._score_spatial_coverage(candidate, {})

        assert score == 0.8

    def test_score_temporal_proximity_exact(self, evaluator, sample_discovery_results):
        """Test temporal proximity at reference time."""
        candidate = sample_discovery_results[0]  # Exact reference time
        context = {
            "temporal_window": {
                "start": "2024-09-14T00:00:00Z",
                "end": "2024-09-16T00:00:00Z",
                "reference_time": "2024-09-15T12:00:00Z"
            }
        }

        score = evaluator._score_temporal_proximity(candidate, context)

        assert score == 1.0

    def test_score_temporal_proximity_decay(self, evaluator, sample_discovery_results):
        """Test temporal proximity decays with time."""
        candidate = sample_discovery_results[2]  # 3 days away
        context = {
            "temporal_window": {
                "start": "2024-09-10T00:00:00Z",
                "end": "2024-09-20T00:00:00Z",
                "reference_time": "2024-09-15T12:00:00Z"
            }
        }

        score = evaluator._score_temporal_proximity(candidate, context)

        # 3 days with 7-day half-life: exp(-3/7) ≈ 0.65
        assert 0.6 < score < 0.7

    def test_score_temporal_proximity_no_reference(self, evaluator, sample_discovery_results):
        """Test temporal proximity without reference time."""
        candidate = sample_discovery_results[0]
        context = {}  # No temporal context

        score = evaluator._score_temporal_proximity(candidate, context)

        assert score == 1.0  # Default to perfect score

    def test_score_resolution_high(self, evaluator, sample_discovery_results):
        """Test resolution score for high resolution."""
        candidate = sample_discovery_results[3]  # 3m resolution
        score = evaluator._score_resolution(candidate, {})

        # exp(-3/100) ≈ 0.97
        assert score > 0.95

    def test_score_resolution_medium(self, evaluator, sample_discovery_results):
        """Test resolution score for medium resolution."""
        candidate = sample_discovery_results[0]  # 10m resolution
        score = evaluator._score_resolution(candidate, {})

        # exp(-10/100) ≈ 0.90
        assert 0.88 < score < 0.92

    def test_score_cloud_cover_clear(self, evaluator, sample_discovery_results):
        """Test cloud cover score for clear skies."""
        candidate = sample_discovery_results[3]  # 5% cloud cover
        score = evaluator._score_cloud_cover(candidate, {})

        assert score == 0.95

    def test_score_cloud_cover_cloudy(self, evaluator, sample_discovery_results):
        """Test cloud cover score for cloudy skies."""
        candidate = sample_discovery_results[2]  # 75% cloud cover
        score = evaluator._score_cloud_cover(candidate, {})

        assert score == 0.25

    def test_score_cloud_cover_none(self, evaluator, sample_discovery_results):
        """Test cloud cover score for non-optical (None)."""
        candidate = sample_discovery_results[0]  # SAR, None cloud cover
        score = evaluator._score_cloud_cover(candidate, {})

        assert score == 1.0  # Perfect score for N/A

    def test_score_data_availability_available(self, evaluator, sample_discovery_results):
        """Test availability score for available data."""
        candidate = sample_discovery_results[0]  # available=True
        score = evaluator._score_data_availability(candidate, {})

        assert score == 1.0  # Open data

    def test_score_data_availability_unavailable(self, evaluator, sample_discovery_results):
        """Test availability score for unavailable data."""
        candidate = sample_discovery_results[6]  # available=False
        score = evaluator._score_data_availability(candidate, {})

        assert score == 0.0

    def test_score_data_availability_cost_tiers(self, evaluator, sample_discovery_results):
        """Test availability score reflects cost tiers."""
        open_candidate = sample_discovery_results[0]  # open
        commercial_candidate = sample_discovery_results[3]  # commercial
        restricted_candidate = sample_discovery_results[5]  # open_restricted

        open_score = evaluator._score_data_availability(open_candidate, {})
        commercial_score = evaluator._score_data_availability(commercial_candidate, {})
        restricted_score = evaluator._score_data_availability(restricted_candidate, {})

        assert open_score > restricted_score > commercial_score


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_evaluate_candidates_basic(self, sample_discovery_results):
        """Test evaluate_candidates convenience function."""
        results = evaluate_candidates(
            sample_discovery_results,
            min_spatial_coverage=0.5,
            max_cloud_cover=0.5
        )

        assert len(results) == len(sample_discovery_results)

    def test_evaluate_candidates_with_temporal(self, sample_discovery_results):
        """Test evaluate_candidates with temporal window."""
        results = evaluate_candidates(
            sample_discovery_results,
            temporal_window={
                "start": "2024-09-14T00:00:00Z",
                "end": "2024-09-16T00:00:00Z",
                "reference_time": "2024-09-15T12:00:00Z"
            }
        )

        assert len(results) == len(sample_discovery_results)

    def test_get_passing_candidates(self, sample_discovery_results):
        """Test get_passing_candidates convenience function."""
        passing = get_passing_candidates(
            sample_discovery_results,
            min_spatial_coverage=0.5,
            max_cloud_cover=0.5
        )

        assert all(r.passed_hard_constraints for r in passing)
        assert len(passing) < len(sample_discovery_results)

    def test_get_passing_candidates_strict(self, sample_discovery_results):
        """Test get_passing_candidates with strict constraints."""
        passing = get_passing_candidates(
            sample_discovery_results,
            min_spatial_coverage=0.9,
            max_cloud_cover=0.1,
            max_resolution_m=15.0
        )

        # Very strict - fewer should pass than with lenient constraints
        lenient_passing = get_passing_candidates(
            sample_discovery_results,
            min_spatial_coverage=0.3,
            max_cloud_cover=0.8
        )
        assert len(passing) <= len(lenient_passing)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestConstraintIntegration:
    """Integration tests for complete constraint evaluation workflows."""

    def test_full_evaluation_workflow(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test complete evaluation workflow."""
        # Evaluate all candidates
        results = evaluator.evaluate_batch(sample_discovery_results, default_context)

        # Filter to passing only
        passing = evaluator.filter_passing(results)

        # Verify structure
        assert len(results) == len(sample_discovery_results)
        assert all(r.passed_hard_constraints for r in passing)

        # Check diagnostics available for all
        for result in results:
            assert "hard_constraints" in result.to_dict()
            assert "soft_scores" in result.to_dict()

    def test_strict_context_filters_more(
        self, evaluator, sample_discovery_results, default_context, strict_context
    ):
        """Test that stricter context filters more candidates."""
        default_results = evaluator.evaluate_batch(
            sample_discovery_results, default_context
        )
        strict_results = evaluator.evaluate_batch(
            sample_discovery_results, strict_context
        )

        default_passing = len(evaluator.filter_passing(default_results))
        strict_passing = len(evaluator.filter_passing(strict_results))

        assert strict_passing <= default_passing

    def test_data_type_specific_constraints(
        self, evaluator, sample_discovery_results, default_context
    ):
        """Test that data-type-specific constraints only apply to that type."""
        results = evaluator.evaluate_batch(sample_discovery_results, default_context)

        sar_result = next(
            r for r in results
            if r.candidate.data_type == "sar"
        )
        optical_result = next(
            r for r in results
            if r.candidate.data_type == "optical"
        )

        # Cloud cover constraint should only appear for optical
        sar_constraints = sar_result.hard_constraint_results.keys()
        optical_constraints = optical_result.hard_constraint_results.keys()

        assert "cloud_cover_maximum" not in sar_constraints
        assert "cloud_cover_maximum" in optical_constraints


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_candidates_list(self, evaluator, default_context):
        """Test evaluation with empty candidate list."""
        results = evaluator.evaluate_batch([], default_context)

        assert len(results) == 0

    def test_empty_context(self, evaluator, sample_discovery_results):
        """Test evaluation with empty context."""
        candidate = sample_discovery_results[0]
        result = evaluator.evaluate(candidate, {})

        # Should use defaults and not crash
        assert result is not None
        assert isinstance(result.passed_hard_constraints, bool)

    def test_boundary_coverage_50_percent(self, evaluator):
        """Test boundary condition at exactly 50% coverage."""
        candidate = DiscoveryResult(
            dataset_id="boundary_test",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=50.0,  # Exactly at threshold
            resolution_m=10
        )

        # 50% threshold with 50% coverage should pass (>=)
        context = {"min_spatial_coverage": 0.5}
        passed = evaluator._check_min_spatial_coverage(candidate, context)

        assert passed is True

    def test_boundary_cloud_cover_50_percent(self, evaluator):
        """Test boundary condition at exactly 50% cloud cover."""
        candidate = DiscoveryResult(
            dataset_id="boundary_test",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            cloud_cover_percent=50.0  # Exactly at threshold
        )

        # 50% threshold with 50% cloud cover should pass (<=)
        context = {"max_cloud_cover": 0.5}
        passed = evaluator._check_max_cloud_cover(candidate, context)

        assert passed is True

    def test_zero_resolution(self, evaluator):
        """Test handling of zero resolution (edge case)."""
        candidate = DiscoveryResult(
            dataset_id="zero_res",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=0.0  # Edge case
        )

        # Should not crash
        score = evaluator._score_resolution(candidate, {})

        assert score <= 1.0

    def test_very_high_resolution_score_capped(self, evaluator):
        """Test very high resolution (e.g., 10cm) score is capped."""
        candidate = DiscoveryResult(
            dataset_id="ultra_high_res",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=0.1  # 10cm
        )

        score = evaluator._score_resolution(candidate, {})

        # exp(-0.1/100) ≈ 0.999 - capped at 1.0
        assert score <= 1.0
        assert score > 0.99

    def test_coverage_over_100_percent(self, evaluator):
        """Test handling of >100% coverage (data artifact)."""
        candidate = DiscoveryResult(
            dataset_id="over_coverage",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=105.0,  # Over 100%
            resolution_m=10
        )

        score = evaluator._score_spatial_coverage(candidate, {})

        # Should be capped at 1.0
        assert score == 1.0

    def test_negative_cloud_cover(self, evaluator):
        """Test handling of negative cloud cover (invalid data)."""
        candidate = DiscoveryResult(
            dataset_id="neg_clouds",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            cloud_cover_percent=-5.0  # Invalid
        )

        score = evaluator._score_cloud_cover(candidate, {})

        # Invalid negative values should be clamped to 0%, giving score 1.0
        assert score == 1.0

    def test_cloud_cover_over_100_percent(self, evaluator):
        """Test handling of >100% cloud cover (invalid data)."""
        candidate = DiscoveryResult(
            dataset_id="over_clouds",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            cloud_cover_percent=120.0  # Invalid
        )

        score = evaluator._score_cloud_cover(candidate, {})

        # Invalid values >100% should be clamped to 100%, giving score 0.0
        assert score == 0.0

    def test_unknown_cost_tier(self, evaluator):
        """Test handling of unknown cost tier."""
        candidate = DiscoveryResult(
            dataset_id="unknown_tier",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            cost_tier="premium_plus"  # Unknown tier
        )

        score = evaluator._score_data_availability(candidate, {})

        # Unknown tier should get neutral score
        assert score == 0.5


# ============================================================================
# ADDITIONAL SOFT CONSTRAINT TESTS
# ============================================================================

class TestSARNoiseConstraint:
    """Test SAR noise quality scoring."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_sar_noise_with_enl(self, evaluator):
        """Test SAR noise scoring with equivalent number of looks."""
        candidate = DiscoveryResult(
            dataset_id="sar_enl",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"equivalent_number_of_looks": 4.0}
        )

        score = evaluator._score_sar_noise(candidate, {})

        assert score == 1.0

    def test_sar_noise_low_enl(self, evaluator):
        """Test SAR noise scoring with low ENL."""
        candidate = DiscoveryResult(
            dataset_id="sar_low_enl",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"equivalent_number_of_looks": 1.0}
        )

        score = evaluator._score_sar_noise(candidate, {})

        assert score == 0.25

    def test_sar_noise_with_noise_floor(self, evaluator):
        """Test SAR noise scoring with noise floor."""
        candidate = DiscoveryResult(
            dataset_id="sar_noise_floor",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"noise_floor_db": -20.0}
        )

        score = evaluator._score_sar_noise(candidate, {})

        # -20 dB should give score: 1 - (-20 + 25) / 10 = 1 - 0.5 = 0.5
        assert 0.4 < score < 0.6

    def test_sar_noise_with_quality_label(self, evaluator):
        """Test SAR noise scoring with quality label."""
        candidate = DiscoveryResult(
            dataset_id="sar_quality",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"sar_quality": "excellent"}
        )

        score = evaluator._score_sar_noise(candidate, {})

        assert score == 1.0

    def test_sar_noise_no_metadata(self, evaluator):
        """Test SAR noise scoring without metadata."""
        candidate = DiscoveryResult(
            dataset_id="sar_no_meta",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10
        )

        score = evaluator._score_sar_noise(candidate, {})

        assert score == 0.7  # Neutral


class TestGeometricAccuracyConstraint:
    """Test geometric accuracy scoring."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_geometric_accuracy_excellent(self, evaluator):
        """Test geometric accuracy with excellent positioning."""
        candidate = DiscoveryResult(
            dataset_id="geo_excellent",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"absolute_geolocation_accuracy_m": 1.0}
        )

        score = evaluator._score_geometric_accuracy(candidate, {})

        # exp(-1/10) ≈ 0.90
        assert score > 0.85

    def test_geometric_accuracy_poor(self, evaluator):
        """Test geometric accuracy with poor positioning."""
        candidate = DiscoveryResult(
            dataset_id="geo_poor",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"absolute_geolocation_accuracy_m": 50.0}
        )

        score = evaluator._score_geometric_accuracy(candidate, {})

        # exp(-50/10) ≈ 0.007
        assert score < 0.1

    def test_geometric_accuracy_orthorectified(self, evaluator):
        """Test geometric accuracy for orthorectified data."""
        candidate = DiscoveryResult(
            dataset_id="geo_ortho",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"orthorectified": True}
        )

        score = evaluator._score_geometric_accuracy(candidate, {})

        assert score == 1.0

    def test_geometric_accuracy_not_orthorectified(self, evaluator):
        """Test geometric accuracy for non-orthorectified data."""
        candidate = DiscoveryResult(
            dataset_id="geo_not_ortho",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"orthorectified": False}
        )

        score = evaluator._score_geometric_accuracy(candidate, {})

        assert score == 0.6

    def test_geometric_accuracy_processing_level(self, evaluator):
        """Test geometric accuracy from processing level."""
        candidate = DiscoveryResult(
            dataset_id="geo_l2a",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"processing_level": "S2A_MSIL2A"}
        )

        score = evaluator._score_geometric_accuracy(candidate, {})

        assert score == 1.0


class TestAOIProximityConstraint:
    """Test AOI proximity scoring."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_aoi_proximity_exact_match(self, evaluator):
        """Test AOI proximity with exact center match."""
        candidate = DiscoveryResult(
            dataset_id="aoi_exact",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"scene_center": (25.5, -80.0)}
        )

        context = {"aoi_center": (25.5, -80.0)}
        score = evaluator._score_aoi_proximity(candidate, context)

        assert score == 1.0

    def test_aoi_proximity_50km_away(self, evaluator):
        """Test AOI proximity at 50km distance."""
        candidate = DiscoveryResult(
            dataset_id="aoi_50km",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"scene_center": (25.95, -80.0)}  # ~50km north
        )

        context = {"aoi_center": (25.5, -80.0)}
        score = evaluator._score_aoi_proximity(candidate, context)

        # exp(-50/50) ≈ 0.37, but clamped to min 0.2
        assert 0.3 < score < 0.5

    def test_aoi_proximity_tile_match(self, evaluator):
        """Test AOI proximity with tile ID match."""
        candidate = DiscoveryResult(
            dataset_id="aoi_tile",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"tile_id": "17RNH"}
        )

        context = {"target_tile": "17RNH"}
        score = evaluator._score_aoi_proximity(candidate, context)

        assert score == 1.0

    def test_aoi_proximity_high_coverage(self, evaluator):
        """Test AOI proximity inferred from high coverage."""
        candidate = DiscoveryResult(
            dataset_id="aoi_high_cov",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=98.0,
            resolution_m=10,
            metadata={}  # Empty metadata triggers neutral score path
        )

        score = evaluator._score_aoi_proximity(candidate, {})

        # Empty metadata gets neutral score (0.8)
        assert score == 0.8


class TestViewAngleConstraint:
    """Test view/incidence angle scoring."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_optical_nadir(self, evaluator):
        """Test optical nadir view angle."""
        candidate = DiscoveryResult(
            dataset_id="optical_nadir",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"view_angle": 0.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        assert score == 1.0

    def test_optical_off_nadir(self, evaluator):
        """Test optical off-nadir view angle."""
        candidate = DiscoveryResult(
            dataset_id="optical_off_nadir",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"view_angle": 20.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        # exp(-20/20) ≈ 0.37
        assert 0.3 < score < 0.4

    def test_optical_sun_elevation(self, evaluator):
        """Test optical scoring with sun elevation."""
        candidate = DiscoveryResult(
            dataset_id="optical_sun",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"sun_elevation": 60.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        # 60/90 = 0.67
        assert 0.65 < score < 0.7

    def test_sar_optimal_incidence(self, evaluator):
        """Test SAR optimal incidence angle."""
        candidate = DiscoveryResult(
            dataset_id="sar_optimal",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"incidence_angle": 38.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        assert score == 1.0

    def test_sar_steep_incidence(self, evaluator):
        """Test SAR steep incidence angle."""
        candidate = DiscoveryResult(
            dataset_id="sar_steep",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"incidence_angle": 15.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        # Steep angle should score lower
        assert score < 0.8

    def test_sar_grazing_incidence(self, evaluator):
        """Test SAR grazing incidence angle."""
        candidate = DiscoveryResult(
            dataset_id="sar_grazing",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10,
            metadata={"incidence_angle": 65.0}
        )

        score = evaluator._score_view_angle(candidate, {})

        # Grazing angle should score lower
        assert score < 0.7

    def test_view_angle_no_metadata(self, evaluator):
        """Test view angle scoring without metadata."""
        candidate = DiscoveryResult(
            dataset_id="no_angle",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10
        )

        score = evaluator._score_view_angle(candidate, {})

        assert score == 0.75  # Neutral


# ============================================================================
# ADDITIONAL EDGE CASE TESTS (Added during Track 1 review)
# ============================================================================

class TestAdditionalEdgeCases:
    """Additional edge case tests added during Track 1 code review."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_negative_resolution(self, evaluator):
        """Test handling of negative resolution (invalid data)."""
        candidate = DiscoveryResult(
            dataset_id="neg_res",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=-10.0  # Invalid
        )

        score = evaluator._score_resolution(candidate, {})

        # Invalid negative values should be clamped to 0, giving score 1.0
        assert score == 1.0

    def test_nan_handling_in_scores(self, evaluator):
        """Test that NaN values don't propagate through scoring."""
        # Create a candidate with valid data
        candidate = DiscoveryResult(
            dataset_id="valid",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            cloud_cover_percent=20.0
        )

        context = {
            "temporal_window": {
                "start": "2024-09-14T00:00:00Z",
                "end": "2024-09-16T00:00:00Z",
                "reference_time": "2024-09-15T12:00:00Z"
            }
        }

        result = evaluator.evaluate(candidate, context)

        # Check that total score is a valid number
        assert not math.isnan(result.total_soft_score)
        assert not math.isinf(result.total_soft_score)
        assert 0.0 <= result.total_soft_score <= 1.0

    def test_very_large_resolution(self, evaluator):
        """Test handling of very large resolution values."""
        candidate = DiscoveryResult(
            dataset_id="huge_res",
            provider="test",
            data_type="weather",
            source_uri="test.nc",
            format="netcdf",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=100000.0  # 100km resolution
        )

        score = evaluator._score_resolution(candidate, {})

        # Very large resolution should give low but valid score
        assert 0.0 <= score <= 1.0
        assert score < 0.01  # Should be very low

    def test_empty_metadata_dict(self, evaluator):
        """Test handling of empty metadata dictionary."""
        candidate = DiscoveryResult(
            dataset_id="empty_meta",
            provider="test",
            data_type="sar",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            metadata={}
        )

        # All metadata-dependent scores should return neutral values
        sar_score = evaluator._score_sar_noise(candidate, {})
        geo_score = evaluator._score_geometric_accuracy(candidate, {})
        aoi_score = evaluator._score_aoi_proximity(candidate, {})
        view_score = evaluator._score_view_angle(candidate, {})

        assert 0.0 <= sar_score <= 1.0
        assert 0.0 <= geo_score <= 1.0
        assert 0.0 <= aoi_score <= 1.0
        assert 0.0 <= view_score <= 1.0

    def test_context_with_none_values(self, evaluator):
        """Test handling of context with None values."""
        candidate = DiscoveryResult(
            dataset_id="test",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=80.0,
            resolution_m=10.0,
            cloud_cover_percent=30.0
        )

        context = {
            "min_spatial_coverage": None,
            "max_cloud_cover": None,
            "soft_weights": None
        }

        # Should handle None values gracefully
        result = evaluator.evaluate(candidate, context)
        assert result is not None

    def test_special_float_values_in_context(self, evaluator):
        """Test handling of special float values in context."""
        candidate = DiscoveryResult(
            dataset_id="test",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=80.0,
            resolution_m=10.0,
            cloud_cover_percent=30.0
        )

        # Infinity should work for "no limit"
        context = {
            "max_resolution_m": float('inf')
        }

        passed = evaluator._check_max_resolution(candidate, context)
        assert passed is True

    def test_zero_weight_normalization(self, evaluator):
        """Test that zero weights don't cause division by zero."""
        candidate = DiscoveryResult(
            dataset_id="test",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=80.0,
            resolution_m=10.0,
            cloud_cover_percent=30.0
        )

        # Set all weights to zero
        context = {
            "soft_weights": {
                "spatial_coverage_score": 0.0,
                "temporal_proximity_score": 0.0,
                "resolution_score": 0.0,
                "cloud_cover_score": 0.0,
                "data_availability": 0.0,
                "sar_noise_quality": 0.0,
                "geometric_accuracy": 0.0,
                "aoi_proximity": 0.0,
                "view_angle_quality": 0.0,
            }
        }

        result = evaluator.evaluate(candidate, context)

        # Should not crash, total_soft_score should be 0.0
        assert result.total_soft_score == 0.0


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Test thread safety of the constraint evaluator."""

    def test_concurrent_evaluation(self):
        """Test that multiple evaluators can run concurrently."""
        import concurrent.futures

        def evaluate_candidate(index):
            evaluator = ConstraintEvaluator()
            candidate = DiscoveryResult(
                dataset_id=f"test_{index}",
                provider="test",
                data_type="optical",
                source_uri="test.tif",
                format="cog",
                acquisition_time=datetime.now(timezone.utc),
                spatial_coverage_percent=80.0 + (index % 20),
                resolution_m=10.0,
                cloud_cover_percent=30.0 + (index % 10)
            )
            context = {
                "min_spatial_coverage": 0.5,
                "max_cloud_cover": 0.8
            }
            return evaluator.evaluate(candidate, context)

        # Run 20 evaluations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_candidate, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All evaluations should complete successfully
        assert len(results) == 20
        for result in results:
            assert result is not None
            assert isinstance(result.passed_hard_constraints, bool)

    def test_batch_evaluation_with_large_dataset(self):
        """Test batch evaluation with a larger dataset."""
        evaluator = ConstraintEvaluator()

        # Create 100 candidates
        candidates = []
        base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(100):
            candidates.append(DiscoveryResult(
                dataset_id=f"candidate_{i}",
                provider="test",
                data_type=["optical", "sar", "dem"][i % 3],
                source_uri=f"test_{i}.tif",
                format="cog",
                acquisition_time=base_time - timedelta(hours=i),
                spatial_coverage_percent=50.0 + (i % 50),
                resolution_m=5.0 + (i % 30),
                cloud_cover_percent=10.0 + (i % 80) if (i % 3 == 0) else None
            ))

        context = {
            "min_spatial_coverage": 0.6,
            "max_cloud_cover": 0.6,
            "temporal_window": {
                "start": "2024-09-10T00:00:00Z",
                "end": "2024-09-20T00:00:00Z",
                "reference_time": "2024-09-15T12:00:00Z"
            }
        }

        results = evaluator.evaluate_batch(candidates, context)

        assert len(results) == 100
        # Verify all results are valid
        for result in results:
            assert 0.0 <= result.total_soft_score <= 1.0
            assert isinstance(result.passed_hard_constraints, bool)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics of the constraint evaluator."""

    def test_evaluation_performance(self):
        """Test that evaluation performance is reasonable."""
        import time

        evaluator = ConstraintEvaluator()
        candidates = []
        base_time = datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Create 1000 candidates
        for i in range(1000):
            candidates.append(DiscoveryResult(
                dataset_id=f"perf_{i}",
                provider="test",
                data_type="optical",
                source_uri=f"test_{i}.tif",
                format="cog",
                acquisition_time=base_time - timedelta(hours=i % 100),
                spatial_coverage_percent=80.0,
                resolution_m=10.0,
                cloud_cover_percent=30.0
            ))

        context = {"min_spatial_coverage": 0.5}

        start = time.time()
        results = evaluator.evaluate_batch(candidates, context)
        elapsed = time.time() - start

        assert len(results) == 1000
        # Should complete in reasonable time (< 5 seconds for 1000 evaluations)
        assert elapsed < 5.0, f"Evaluation took too long: {elapsed:.2f}s"


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Regression tests for previously found bugs."""

    @pytest.fixture
    def evaluator(self):
        return ConstraintEvaluator()

    def test_regression_cloud_cover_negative(self, evaluator):
        """Regression test: negative cloud cover should not produce scores > 1.0."""
        candidate = DiscoveryResult(
            dataset_id="regression_neg_cloud",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            cloud_cover_percent=-100.0  # Extremely invalid
        )

        score = evaluator._score_cloud_cover(candidate, {})
        assert 0.0 <= score <= 1.0

    def test_regression_resolution_negative(self, evaluator):
        """Regression test: negative resolution should not produce scores > 1.0."""
        candidate = DiscoveryResult(
            dataset_id="regression_neg_res",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=-50.0  # Invalid
        )

        score = evaluator._score_resolution(candidate, {})
        assert 0.0 <= score <= 1.0

    def test_regression_none_soft_weights(self, evaluator):
        """Regression test: None soft_weights should not cause AttributeError."""
        candidate = DiscoveryResult(
            dataset_id="regression_none_weights",
            provider="test",
            data_type="optical",
            source_uri="test.tif",
            format="cog",
            acquisition_time=datetime.now(timezone.utc),
            spatial_coverage_percent=100.0,
            resolution_m=10.0,
            cloud_cover_percent=20.0
        )

        context = {"soft_weights": None}

        # Should not raise AttributeError
        result = evaluator.evaluate(candidate, context)
        assert result is not None
