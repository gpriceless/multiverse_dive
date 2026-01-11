"""
Multi-criteria ranking system for data source selection.

Provides weighted scoring across multiple criteria (coverage, resolution, quality, cost)
with provider preference integration and trade-off documentation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import math

# Constants for scoring functions
RESOLUTION_CHARACTERISTIC_LENGTH_M = 100.0  # Resolution scoring decay (10m=1.0, 100m≈0.37)
TEMPORAL_HALFLIFE_DAYS = 7.0  # Temporal proximity decay (7-day half-life)
SECONDS_PER_DAY = 86400.0  # For time delta calculations

from core.data.discovery.base import DiscoveryResult
from core.data.providers.registry import ProviderRegistry, Provider


@dataclass
class RankingCriteria:
    """
    Configuration for ranking criteria and weights.

    Each criterion is scored 0.0-1.0, then multiplied by its weight.
    Weights should sum to 1.0 for normalized scoring.
    """

    # Spatial criteria
    spatial_coverage: float = 0.25
    resolution: float = 0.15

    # Temporal criteria
    temporal_proximity: float = 0.20

    # Quality criteria
    cloud_cover: float = 0.15
    data_quality: float = 0.10

    # Cost and access criteria
    access_cost: float = 0.10
    provider_preference: float = 0.05

    def validate(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = sum([
            self.spatial_coverage,
            self.resolution,
            self.temporal_proximity,
            self.cloud_cover,
            self.data_quality,
            self.access_cost,
            self.provider_preference
        ])

        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ranking weights must sum to ~1.0, got {total:.3f}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "spatial_coverage": self.spatial_coverage,
            "resolution": self.resolution,
            "temporal_proximity": self.temporal_proximity,
            "cloud_cover": self.cloud_cover,
            "data_quality": self.data_quality,
            "access_cost": self.access_cost,
            "provider_preference": self.provider_preference
        }

    @classmethod
    def from_dict(cls, weights: Dict[str, float]) -> 'RankingCriteria':
        """Create from dictionary of weights."""
        return cls(**weights)


@dataclass
class RankedCandidate:
    """
    A candidate dataset with computed ranking scores.

    Contains the original discovery result plus all computed scores
    and the final weighted total score.
    """

    candidate: DiscoveryResult
    scores: Dict[str, float]
    total_score: float
    rank: int = 0  # Set during ranking

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "candidate": self.candidate.to_dict(),
            "scores": self.scores,
            "total_score": self.total_score,
            "rank": self.rank
        }


@dataclass
class TradeOffRecord:
    """
    Documents a selection trade-off decision.

    Records what was selected, what alternatives were available,
    and the rationale for the decision.
    """

    decision_context: str  # e.g., "Select optical imagery for flood.coastal"
    selected_id: str
    selected_score: float
    alternatives: List[Dict[str, Any]]  # List of {id, score, reason_rejected}
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_context": self.decision_context,
            "selected_id": self.selected_id,
            "selected_score": self.selected_score,
            "alternatives": self.alternatives,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat()
        }


class MultiCriteriaRanker:
    """
    Multi-criteria ranking engine for data source selection.

    Implements weighted scoring across configurable criteria with
    provider preference integration and comprehensive trade-off documentation.
    """

    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry] = None,
        criteria: Optional[RankingCriteria] = None
    ):
        """
        Initialize ranker.

        Args:
            provider_registry: Provider registry for preference scoring
            criteria: Ranking criteria and weights (uses defaults if None)
        """
        self.provider_registry = provider_registry or ProviderRegistry()
        self.criteria = criteria or RankingCriteria()
        self.criteria.validate()

        # Scoring function registry
        self._scorers: Dict[str, Callable] = {
            "spatial_coverage": self._score_spatial_coverage,
            "resolution": self._score_resolution,
            "temporal_proximity": self._score_temporal_proximity,
            "cloud_cover": self._score_cloud_cover,
            "data_quality": self._score_data_quality,
            "access_cost": self._score_access_cost,
            "provider_preference": self._score_provider_preference
        }

    def rank_candidates(
        self,
        candidates: List[DiscoveryResult],
        query_context: Dict[str, Any]
    ) -> List[RankedCandidate]:
        """
        Rank candidates using multi-criteria weighted scoring.

        Args:
            candidates: List of discovered datasets
            query_context: Query parameters for contextual scoring
                - temporal: {start, end, reference_time}
                - spatial: GeoJSON geometry
                - intent_class: Event classification

        Returns:
            List of RankedCandidate objects sorted by total_score (descending)
        """
        ranked = []

        for candidate in candidates:
            # Compute individual criterion scores
            scores = {}
            for criterion_name, scorer_func in self._scorers.items():
                scores[criterion_name] = scorer_func(candidate, query_context)

            # Compute weighted total score
            total_score = self._compute_total_score(scores)

            ranked.append(RankedCandidate(
                candidate=candidate,
                scores=scores,
                total_score=total_score
            ))

        # Sort by total score (descending) and assign ranks
        ranked.sort(key=lambda x: x.total_score, reverse=True)
        for i, item in enumerate(ranked, start=1):
            item.rank = i

        return ranked

    def rank_by_data_type(
        self,
        candidates: List[DiscoveryResult],
        query_context: Dict[str, Any]
    ) -> Dict[str, List[RankedCandidate]]:
        """
        Rank candidates grouped by data type.

        Returns:
            Dictionary mapping data_type -> ranked candidates
        """
        # Group by data type
        by_type: Dict[str, List[DiscoveryResult]] = {}
        for candidate in candidates:
            if candidate.data_type not in by_type:
                by_type[candidate.data_type] = []
            by_type[candidate.data_type].append(candidate)

        # Rank each group
        ranked_by_type = {}
        for data_type, type_candidates in by_type.items():
            ranked_by_type[data_type] = self.rank_candidates(
                type_candidates,
                query_context
            )

        return ranked_by_type

    def document_trade_offs(
        self,
        ranked: List[RankedCandidate],
        top_n: int = 3,
        context: str = "Dataset selection"
    ) -> List[TradeOffRecord]:
        """
        Document trade-offs between top candidates.

        Args:
            ranked: Ranked candidates (assumed sorted)
            top_n: Number of top alternatives to document
            context: Decision context description

        Returns:
            List of TradeOffRecord objects
        """
        if not ranked:
            return []

        # Select best candidate
        best = ranked[0]
        alternatives = ranked[1:top_n+1]

        # Build alternatives list with rejection reasons
        alternative_records = []
        for alt in alternatives:
            # Identify key score differences
            score_diffs = {
                criterion: best.scores[criterion] - alt.scores[criterion]
                for criterion in best.scores.keys()
            }

            # Find largest differences
            top_diffs = sorted(
                score_diffs.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]

            reason = ", ".join([
                f"{crit.replace('_', ' ')}: {best.scores[crit]:.2f} vs {alt.scores[crit]:.2f}"
                for crit, _ in top_diffs
            ])

            alternative_records.append({
                "id": alt.candidate.dataset_id,
                "score": alt.total_score,
                "rank": alt.rank,
                "reason": reason
            })

        # Generate rationale for selected candidate
        rationale = self._generate_selection_rationale(best)

        trade_off = TradeOffRecord(
            decision_context=context,
            selected_id=best.candidate.dataset_id,
            selected_score=best.total_score,
            alternatives=alternative_records,
            rationale=rationale
        )

        return [trade_off]

    def _compute_total_score(self, scores: Dict[str, float]) -> float:
        """
        Compute weighted total score.

        Args:
            scores: Dictionary of criterion -> score (0.0-1.0)

        Returns:
            Weighted total score
        """
        weights = self.criteria.to_dict()
        total = sum(scores[criterion] * weights[criterion] for criterion in scores)
        return total

    def _generate_selection_rationale(self, ranked: RankedCandidate) -> str:
        """
        Generate human-readable rationale for selection.

        Highlights the top 2-3 scoring criteria.
        """
        # Get top scoring criteria
        top_criteria = sorted(
            ranked.scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Filter to only strong scores (>0.7)
        strong_criteria = [
            (crit, score) for crit, score in top_criteria
            if score > 0.7
        ]

        if not strong_criteria:
            return f"Best overall match (total score: {ranked.total_score:.3f})"

        rationale_parts = [
            f"{crit.replace('_', ' ')}: {score:.2f}"
            for crit, score in strong_criteria
        ]

        return "; ".join(rationale_parts) + f" (total: {ranked.total_score:.3f})"

    # Scoring functions (each returns 0.0-1.0)

    def _score_spatial_coverage(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """Score based on spatial coverage of AOI."""
        return min(candidate.spatial_coverage_percent / 100.0, 1.0)

    def _score_resolution(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on spatial resolution.

        Higher resolution (smaller values) scores better.
        Uses exponential decay: 10m=1.0, 100m≈0.37, 1000m≈0.00005
        """
        # Exponential decay with configurable characteristic length
        score = math.exp(-candidate.resolution_m / RESOLUTION_CHARACTERISTIC_LENGTH_M)
        return min(score, 1.0)

    def _score_temporal_proximity(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on temporal proximity to reference time.

        Uses exponential decay with 7-day half-life.
        """
        temporal = context.get("temporal", {})
        reference_time = temporal.get("reference_time") or temporal.get("start")

        if not reference_time:
            return 0.5  # Neutral score if no reference

        # Parse reference time
        ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))

        # Calculate time difference in days
        time_diff_days = abs((candidate.acquisition_time - ref_dt).total_seconds() / SECONDS_PER_DAY)

        # Exponential decay with configurable half-life
        score = math.exp(-time_diff_days / TEMPORAL_HALFLIFE_DAYS)
        return score

    def _score_cloud_cover(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on cloud cover.

        Lower cloud cover scores better. Returns 1.0 for non-optical data.
        """
        if candidate.cloud_cover_percent is None:
            return 1.0  # N/A for non-optical (SAR, DEM, etc.)

        # Invert cloud cover: 0% = 1.0, 100% = 0.0
        return 1.0 - (candidate.cloud_cover_percent / 100.0)

    def _score_data_quality(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on data quality indicators.

        Maps quality flags to numeric scores.
        """
        quality_flag = candidate.quality_flag or 'good'

        quality_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3,
            'bad': 0.1
        }

        return quality_scores.get(quality_flag, 0.5)

    def _score_access_cost(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on access cost tier.

        Prefers open data over restricted or commercial.
        """
        cost_tier = candidate.cost_tier or 'open'

        cost_scores = {
            'open': 1.0,
            'open_restricted': 0.7,
            'commercial_low': 0.5,
            'commercial': 0.3,
            'commercial_high': 0.1
        }

        return cost_scores.get(cost_tier, 0.5)

    def _score_provider_preference(
        self,
        candidate: DiscoveryResult,
        context: Dict[str, Any]
    ) -> float:
        """
        Score based on provider preference from registry.

        Uses provider metadata preference_score (0.0-1.0).
        """
        provider = self.provider_registry.get_provider(candidate.provider)

        if provider:
            return provider.metadata.get("preference_score", 0.5)

        return 0.5  # Neutral score for unknown providers

    def update_criteria(self, new_criteria: RankingCriteria) -> None:
        """
        Update ranking criteria.

        Args:
            new_criteria: New criteria configuration

        Raises:
            ValueError: If criteria weights don't sum to ~1.0
        """
        new_criteria.validate()
        self.criteria = new_criteria

    def get_criteria_summary(self) -> str:
        """
        Get human-readable summary of current criteria weights.

        Returns:
            Formatted string describing weights
        """
        weights = self.criteria.to_dict()
        lines = ["Current Ranking Criteria:"]

        for criterion, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {criterion.replace('_', ' ').title()}: {weight:.2%}")

        return "\n".join(lines)
