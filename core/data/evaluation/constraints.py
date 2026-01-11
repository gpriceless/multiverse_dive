"""
Constraint evaluation engine.

Evaluates discovered datasets against hard constraints (pass/fail)
and soft constraints (scored 0.0 to 1.0). Provides structured
evaluation results with clear pass/fail reasons and score breakdowns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import math

from core.data.discovery.base import DiscoveryResult


@dataclass
class HardConstraint:
    """
    Definition of a hard constraint (pass/fail).

    Hard constraints are binary checks that filter out unsuitable datasets.
    Examples: minimum spatial coverage, maximum cloud cover, resolution limits.
    """

    name: str
    description: str
    check_function: callable
    applies_to_data_types: Optional[List[str]] = None  # None = applies to all

    def applies_to(self, data_type: str) -> bool:
        """Check if this constraint applies to the given data type."""
        if self.applies_to_data_types is None:
            return True
        return data_type in self.applies_to_data_types

    def evaluate(self, candidate: DiscoveryResult, context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluate the constraint.

        Args:
            candidate: Dataset candidate to evaluate
            context: Additional context (query parameters, thresholds, etc.)

        Returns:
            (passed, reason) tuple
        """
        try:
            passed = self.check_function(candidate, context)
            if passed:
                return True, f"{self.name}: passed"
            else:
                return False, f"{self.name}: failed - {self.description}"
        except Exception as e:
            # Constraint evaluation error - fail safe
            return False, f"{self.name}: evaluation error - {str(e)}"


@dataclass
class SoftConstraint:
    """
    Definition of a soft constraint (scored 0.0 to 1.0).

    Soft constraints contribute to ranking but don't eliminate candidates.
    Examples: cloud cover quality, temporal proximity, resolution preference.
    """

    name: str
    description: str
    score_function: callable
    weight: float = 1.0
    applies_to_data_types: Optional[List[str]] = None

    def applies_to(self, data_type: str) -> bool:
        """Check if this constraint applies to the given data type."""
        if self.applies_to_data_types is None:
            return True
        return data_type in self.applies_to_data_types

    def evaluate(self, candidate: DiscoveryResult, context: Dict[str, Any]) -> tuple[float, str]:
        """
        Evaluate the constraint and return a score.

        Args:
            candidate: Dataset candidate to evaluate
            context: Additional context (query parameters, preferences, etc.)

        Returns:
            (score, explanation) tuple where score is 0.0 to 1.0
        """
        try:
            score = self.score_function(candidate, context)
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            return score, f"{self.name}: {score:.3f}"
        except Exception as e:
            # Scoring error - return neutral score
            return 0.5, f"{self.name}: scoring error - {str(e)}"


@dataclass
class EvaluationResult:
    """
    Result of evaluating a dataset candidate against constraints.
    """

    candidate: DiscoveryResult
    passed_hard_constraints: bool
    hard_constraint_results: Dict[str, tuple[bool, str]] = field(default_factory=dict)
    soft_constraint_scores: Dict[str, tuple[float, str]] = field(default_factory=dict)
    total_soft_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset_id": self.candidate.dataset_id,
            "provider": self.candidate.provider,
            "data_type": self.candidate.data_type,
            "passed": self.passed_hard_constraints,
            "hard_constraints": {
                name: {"passed": result[0], "reason": result[1]}
                for name, result in self.hard_constraint_results.items()
            },
            "soft_scores": {
                name: {"score": result[0], "explanation": result[1]}
                for name, result in self.soft_constraint_scores.items()
            },
            "total_score": self.total_soft_score,
            "failure_reasons": self.failure_reasons
        }


class ConstraintEvaluator:
    """
    Main constraint evaluation engine.

    Manages sets of hard and soft constraints, evaluates candidates,
    and produces structured evaluation results.
    """

    def __init__(self):
        self.hard_constraints: List[HardConstraint] = []
        self.soft_constraints: List[SoftConstraint] = []
        self._register_default_constraints()

    def _register_default_constraints(self):
        """Register default constraint set."""
        # Hard constraints
        self.register_hard_constraint(HardConstraint(
            name="spatial_coverage_minimum",
            description="Dataset must cover minimum percentage of AOI",
            check_function=self._check_min_spatial_coverage
        ))

        self.register_hard_constraint(HardConstraint(
            name="cloud_cover_maximum",
            description="Cloud cover must not exceed maximum threshold",
            check_function=self._check_max_cloud_cover,
            applies_to_data_types=["optical"]
        ))

        self.register_hard_constraint(HardConstraint(
            name="resolution_maximum",
            description="Spatial resolution must be finer than maximum",
            check_function=self._check_max_resolution
        ))

        self.register_hard_constraint(HardConstraint(
            name="temporal_window",
            description="Acquisition time must fall within temporal window",
            check_function=self._check_temporal_window
        ))

        # Soft constraints
        self.register_soft_constraint(SoftConstraint(
            name="spatial_coverage_score",
            description="Higher coverage is better",
            score_function=self._score_spatial_coverage
        ))

        self.register_soft_constraint(SoftConstraint(
            name="temporal_proximity_score",
            description="Closer to reference time is better",
            score_function=self._score_temporal_proximity
        ))

        self.register_soft_constraint(SoftConstraint(
            name="resolution_score",
            description="Higher resolution is better",
            score_function=self._score_resolution
        ))

        self.register_soft_constraint(SoftConstraint(
            name="cloud_cover_score",
            description="Lower cloud cover is better",
            score_function=self._score_cloud_cover,
            applies_to_data_types=["optical"]
        ))

        self.register_soft_constraint(SoftConstraint(
            name="data_availability",
            description="Data is available and accessible",
            score_function=self._score_data_availability
        ))

    def register_hard_constraint(self, constraint: HardConstraint):
        """Register a hard constraint."""
        self.hard_constraints.append(constraint)

    def register_soft_constraint(self, constraint: SoftConstraint):
        """Register a soft constraint."""
        self.soft_constraints.append(constraint)

    def evaluate(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate a candidate against all applicable constraints.

        Args:
            candidate: Dataset candidate to evaluate
            context: Evaluation context with parameters like:
                - min_spatial_coverage (float): 0.0 to 1.0
                - max_cloud_cover (float): 0.0 to 1.0
                - max_resolution_m (float): meters
                - temporal_window (dict): {start, end, reference_time}
                - soft_weights (dict): weights for soft constraints

        Returns:
            EvaluationResult with detailed scores and pass/fail status
        """
        result = EvaluationResult(candidate=candidate, passed_hard_constraints=True)

        # Evaluate hard constraints
        for constraint in self.hard_constraints:
            if not constraint.applies_to(candidate.data_type):
                continue

            passed, reason = constraint.evaluate(candidate, context)
            result.hard_constraint_results[constraint.name] = (passed, reason)

            if not passed:
                result.passed_hard_constraints = False
                result.failure_reasons.append(reason)

        # Evaluate soft constraints (even if hard constraints failed, for diagnostics)
        total_score = 0.0
        total_weight = 0.0
        weights = context.get("soft_weights", {})

        for constraint in self.soft_constraints:
            if not constraint.applies_to(candidate.data_type):
                continue

            score, explanation = constraint.evaluate(candidate, context)
            result.soft_constraint_scores[constraint.name] = (score, explanation)

            # Apply weight
            weight = weights.get(constraint.name, constraint.weight)
            total_score += score * weight
            total_weight += weight

        # Normalize total score
        if total_weight > 0:
            result.total_soft_score = total_score / total_weight
        else:
            result.total_soft_score = 0.0

        return result

    def evaluate_batch(
        self,
        candidates: List[DiscoveryResult],
        context: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple candidates.

        Args:
            candidates: List of dataset candidates
            context: Evaluation context

        Returns:
            List of EvaluationResult objects
        """
        return [self.evaluate(candidate, context) for candidate in candidates]

    def filter_passing(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """
        Filter to only candidates that passed hard constraints.

        Args:
            evaluation_results: List of evaluation results

        Returns:
            Filtered list containing only passing candidates
        """
        return [result for result in evaluation_results if result.passed_hard_constraints]

    # Default hard constraint check functions

    @staticmethod
    def _check_min_spatial_coverage(candidate: DiscoveryResult, context: Dict[str, Any]) -> bool:
        """Check minimum spatial coverage constraint."""
        min_coverage = context.get("min_spatial_coverage", 0.5)  # Default 50%
        return candidate.spatial_coverage_percent >= min_coverage * 100

    @staticmethod
    def _check_max_cloud_cover(candidate: DiscoveryResult, context: Dict[str, Any]) -> bool:
        """Check maximum cloud cover constraint (optical only)."""
        max_cloud = context.get("max_cloud_cover", 1.0)  # Default 100% (no limit)

        if candidate.cloud_cover_percent is None:
            # No cloud cover data - assume it passes
            return True

        return candidate.cloud_cover_percent <= max_cloud * 100

    @staticmethod
    def _check_max_resolution(candidate: DiscoveryResult, context: Dict[str, Any]) -> bool:
        """Check maximum resolution constraint."""
        max_resolution = context.get("max_resolution_m", float('inf'))
        return candidate.resolution_m <= max_resolution

    @staticmethod
    def _check_temporal_window(candidate: DiscoveryResult, context: Dict[str, Any]) -> bool:
        """Check temporal window constraint."""
        temporal_window = context.get("temporal_window")
        if not temporal_window:
            # No temporal constraint specified
            return True

        start = datetime.fromisoformat(temporal_window["start"].replace('Z', '+00:00'))
        end = datetime.fromisoformat(temporal_window["end"].replace('Z', '+00:00'))

        return start <= candidate.acquisition_time <= end

    # Default soft constraint scoring functions

    @staticmethod
    def _score_spatial_coverage(candidate: DiscoveryResult, context: Dict[str, Any]) -> float:
        """Score spatial coverage (0.0 to 1.0)."""
        # Linear scoring: 50% coverage = 0.5, 100% coverage = 1.0
        return min(candidate.spatial_coverage_percent / 100.0, 1.0)

    @staticmethod
    def _score_temporal_proximity(candidate: DiscoveryResult, context: Dict[str, Any]) -> float:
        """Score temporal proximity to reference time."""
        temporal_window = context.get("temporal_window")
        if not temporal_window:
            return 1.0  # No temporal preference

        reference_time = temporal_window.get("reference_time") or temporal_window["start"]
        ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))

        # Calculate time difference in days
        time_diff_days = abs((candidate.acquisition_time - ref_dt).total_seconds() / 86400)

        # Exponential decay: 7-day half-life
        # Same-day = 1.0, 7 days = 0.5, 14 days = 0.25
        score = math.exp(-time_diff_days / 7.0)

        return score

    @staticmethod
    def _score_resolution(candidate: DiscoveryResult, context: Dict[str, Any]) -> float:
        """Score spatial resolution (higher resolution = better)."""
        # Logarithmic scoring: 10m = 1.0, 30m = 0.74, 100m = 0.37
        score = math.exp(-candidate.resolution_m / 100.0)
        return min(score, 1.0)

    @staticmethod
    def _score_cloud_cover(candidate: DiscoveryResult, context: Dict[str, Any]) -> float:
        """Score cloud cover (lower = better)."""
        if candidate.cloud_cover_percent is None:
            return 1.0  # N/A for non-optical, perfect score

        # Inverse linear: 0% = 1.0, 50% = 0.5, 100% = 0.0
        return 1.0 - (candidate.cloud_cover_percent / 100.0)

    @staticmethod
    def _score_data_availability(candidate: DiscoveryResult, context: Dict[str, Any]) -> float:
        """Score data availability and accessibility."""
        # Check if data is marked as available
        if candidate.metadata and not candidate.metadata.get("available", True):
            return 0.0

        # Check cost tier
        cost_tier = candidate.cost_tier or "open"
        cost_scores = {
            "open": 1.0,
            "open_restricted": 0.8,
            "restricted": 0.5,
            "commercial": 0.3
        }

        return cost_scores.get(cost_tier, 0.5)


# Convenience functions for common use cases

def evaluate_candidates(
    candidates: List[DiscoveryResult],
    min_spatial_coverage: float = 0.5,
    max_cloud_cover: float = 1.0,
    max_resolution_m: float = float('inf'),
    temporal_window: Optional[Dict[str, str]] = None,
    soft_weights: Optional[Dict[str, float]] = None
) -> List[EvaluationResult]:
    """
    Convenience function to evaluate candidates with common constraints.

    Args:
        candidates: List of discovery results to evaluate
        min_spatial_coverage: Minimum spatial coverage (0.0 to 1.0)
        max_cloud_cover: Maximum cloud cover (0.0 to 1.0)
        max_resolution_m: Maximum resolution in meters
        temporal_window: Temporal extent dict with start/end/reference_time
        soft_weights: Optional weights for soft constraints

    Returns:
        List of evaluation results
    """
    evaluator = ConstraintEvaluator()

    context = {
        "min_spatial_coverage": min_spatial_coverage,
        "max_cloud_cover": max_cloud_cover,
        "max_resolution_m": max_resolution_m,
        "temporal_window": temporal_window,
        "soft_weights": soft_weights or {}
    }

    return evaluator.evaluate_batch(candidates, context)


def get_passing_candidates(
    candidates: List[DiscoveryResult],
    **kwargs
) -> List[EvaluationResult]:
    """
    Evaluate candidates and return only those that pass hard constraints.

    Args:
        candidates: List of discovery results
        **kwargs: Arguments passed to evaluate_candidates()

    Returns:
        List of evaluation results for passing candidates only
    """
    results = evaluate_candidates(candidates, **kwargs)
    return [r for r in results if r.passed_hard_constraints]
