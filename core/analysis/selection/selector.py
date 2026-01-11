"""
Algorithm Selector for intelligent algorithm selection.

Provides rule-based algorithm filtering, data availability matching,
and compute constraint checking for optimal algorithm selection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import logging

from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType,
    get_global_registry,
    load_default_algorithms,
)

logger = logging.getLogger(__name__)


class ComputeProfile(Enum):
    """Predefined compute environment profiles."""
    LAPTOP = "laptop"           # 2GB RAM, no GPU, sequential
    WORKSTATION = "workstation" # 8GB RAM, optional GPU, parallel
    CLOUD = "cloud"             # 32GB+ RAM, GPU available, distributed
    EDGE = "edge"               # 512MB RAM, no GPU, minimal


@dataclass
class ComputeConstraints:
    """
    Compute resource constraints for algorithm filtering.

    Attributes:
        max_memory_mb: Maximum available memory in MB
        gpu_available: Whether GPU is available
        gpu_memory_mb: GPU memory available (if GPU present)
        max_runtime_minutes: Maximum acceptable runtime
        allow_distributed: Whether distributed execution is available
    """
    max_memory_mb: int = 4096
    gpu_available: bool = False
    gpu_memory_mb: Optional[int] = None
    max_runtime_minutes: Optional[int] = None
    allow_distributed: bool = False

    @classmethod
    def from_profile(cls, profile: ComputeProfile) -> 'ComputeConstraints':
        """
        Create constraints from predefined profile.

        Args:
            profile: Compute environment profile

        Returns:
            ComputeConstraints configured for profile
        """
        profiles = {
            ComputeProfile.LAPTOP: cls(
                max_memory_mb=2048,
                gpu_available=False,
                max_runtime_minutes=30,
                allow_distributed=False
            ),
            ComputeProfile.WORKSTATION: cls(
                max_memory_mb=8192,
                gpu_available=True,
                gpu_memory_mb=4096,
                max_runtime_minutes=120,
                allow_distributed=False
            ),
            ComputeProfile.CLOUD: cls(
                max_memory_mb=32768,
                gpu_available=True,
                gpu_memory_mb=16384,
                max_runtime_minutes=None,  # No limit
                allow_distributed=True
            ),
            ComputeProfile.EDGE: cls(
                max_memory_mb=512,
                gpu_available=False,
                max_runtime_minutes=15,
                allow_distributed=False
            ),
        }
        return profiles[profile]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "max_runtime_minutes": self.max_runtime_minutes,
            "allow_distributed": self.allow_distributed
        }


@dataclass
class SelectionCriteria:
    """
    Criteria for algorithm selection.

    Attributes:
        prefer_validated: Prefer algorithms with validation data
        prefer_deterministic: Prefer deterministic algorithms
        prefer_baseline: Prefer baseline over advanced algorithms
        min_accuracy: Minimum required accuracy (if validation data available)
        allowed_categories: Restrict to specific categories
        excluded_algorithms: Explicitly exclude certain algorithm IDs
    """
    prefer_validated: bool = True
    prefer_deterministic: bool = True
    prefer_baseline: bool = False
    min_accuracy: Optional[float] = None
    allowed_categories: Optional[List[AlgorithmCategory]] = None
    excluded_algorithms: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prefer_validated": self.prefer_validated,
            "prefer_deterministic": self.prefer_deterministic,
            "prefer_baseline": self.prefer_baseline,
            "min_accuracy": self.min_accuracy,
            "allowed_categories": [c.value for c in self.allowed_categories] if self.allowed_categories else None,
            "excluded_algorithms": list(self.excluded_algorithms)
        }


@dataclass
class SelectionContext:
    """
    Context for algorithm selection.

    Contains all information needed to select appropriate algorithms.
    """
    event_class: str
    available_data_types: Set[DataType]
    compute_constraints: ComputeConstraints
    criteria: SelectionCriteria = field(default_factory=SelectionCriteria)
    region: Optional[str] = None  # For region-specific validation
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_class": self.event_class,
            "available_data_types": [dt.value for dt in self.available_data_types],
            "compute_constraints": self.compute_constraints.to_dict(),
            "criteria": self.criteria.to_dict(),
            "region": self.region,
            "additional_metadata": self.additional_metadata
        }


@dataclass
class RejectionReason:
    """Reason for rejecting an algorithm."""
    algorithm_id: str
    reason: str
    category: str  # "data", "compute", "validation", "criteria", "deprecated"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm_id": self.algorithm_id,
            "reason": self.reason,
            "category": self.category,
            "details": self.details
        }


@dataclass
class SelectionResult:
    """
    Result of algorithm selection.

    Contains selected algorithm(s), alternatives, and rejection reasons.
    """
    selected: Optional[AlgorithmMetadata]
    alternatives: List[AlgorithmMetadata]
    rejected: List[RejectionReason]
    scores: Dict[str, float]
    context: SelectionContext
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success(self) -> bool:
        """Whether selection was successful."""
        return self.selected is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "rejected": [r.to_dict() for r in self.rejected],
            "scores": self.scores,
            "context": self.context.to_dict(),
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success
        }


class AlgorithmSelector:
    """
    Intelligent algorithm selection engine.

    Provides rule-based filtering, data availability matching,
    and compute constraint checking for optimal algorithm selection.

    Features:
    - Event type matching with wildcard support
    - Data availability validation
    - Compute resource constraint checking
    - Multi-criteria scoring and ranking
    - Comprehensive rejection tracking
    """

    # Scoring weights for algorithm ranking
    DEFAULT_WEIGHTS = {
        "validation_score": 0.25,
        "category_score": 0.15,
        "determinism_score": 0.10,
        "resource_efficiency": 0.20,
        "data_coverage": 0.20,
        "region_match": 0.10,
    }

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize algorithm selector.

        Args:
            registry: Algorithm registry (uses global registry if None)
            weights: Custom scoring weights (uses defaults if None)
        """
        if registry is None:
            registry = get_global_registry()
            # Load default algorithms if registry is empty
            if not registry.algorithms:
                load_default_algorithms()

        self.registry = registry
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        logger.info(
            f"Initialized AlgorithmSelector with {len(self.registry.algorithms)} algorithms"
        )

    def select(self, context: SelectionContext) -> SelectionResult:
        """
        Select optimal algorithm for given context.

        Args:
            context: Selection context with event class, data, constraints

        Returns:
            SelectionResult with selected algorithm and alternatives
        """
        logger.info(f"Selecting algorithm for event class: {context.event_class}")
        logger.info(f"Available data types: {[dt.value for dt in context.available_data_types]}")

        rejected: List[RejectionReason] = []
        candidates: List[AlgorithmMetadata] = []

        # Phase 1: Filter by event type
        event_matches = self.registry.search_by_event_type(context.event_class)
        logger.info(f"Found {len(event_matches)} algorithms matching event type")

        if not event_matches:
            return SelectionResult(
                selected=None,
                alternatives=[],
                rejected=rejected,
                scores={},
                context=context,
                rationale=f"No algorithms found for event class: {context.event_class}"
            )

        # Phase 2: Apply filters
        for algorithm in event_matches:
            rejection = self._check_filters(algorithm, context)
            if rejection:
                rejected.append(rejection)
            else:
                candidates.append(algorithm)

        logger.info(f"After filtering: {len(candidates)} viable candidates, {len(rejected)} rejected")

        if not candidates:
            return SelectionResult(
                selected=None,
                alternatives=[],
                rejected=rejected,
                scores={},
                context=context,
                rationale=self._summarize_rejections(rejected)
            )

        # Phase 3: Score and rank candidates
        scores: Dict[str, float] = {}
        for candidate in candidates:
            scores[candidate.id] = self._score_algorithm(candidate, context)

        # Sort by score descending
        candidates.sort(key=lambda a: scores[a.id], reverse=True)

        # Select best and alternatives
        selected = candidates[0]
        alternatives = candidates[1:min(4, len(candidates))]  # Top 3 alternatives

        # Generate rationale
        rationale = self._generate_rationale(selected, scores[selected.id], context)

        logger.info(
            f"Selected: {selected.id} (score: {scores[selected.id]:.3f})"
        )

        return SelectionResult(
            selected=selected,
            alternatives=alternatives,
            rejected=rejected,
            scores=scores,
            context=context,
            rationale=rationale
        )

    def select_multiple(
        self,
        context: SelectionContext,
        max_algorithms: int = 3
    ) -> List[SelectionResult]:
        """
        Select multiple complementary algorithms.

        Useful when ensemble or multi-algorithm approaches are desired.

        Args:
            context: Selection context
            max_algorithms: Maximum number of algorithms to select

        Returns:
            List of SelectionResult, one per selected algorithm
        """
        results: List[SelectionResult] = []
        used_data_types: Set[DataType] = set()

        for i in range(max_algorithms):
            # Create modified context excluding already-used data types
            modified_context = SelectionContext(
                event_class=context.event_class,
                available_data_types=context.available_data_types - used_data_types,
                compute_constraints=context.compute_constraints,
                criteria=context.criteria,
                region=context.region,
                additional_metadata={
                    **context.additional_metadata,
                    "ensemble_position": i,
                    "excluded_data_types": [dt.value for dt in used_data_types]
                }
            )

            result = self.select(modified_context)

            if result.success:
                results.append(result)
                # Track used data types to encourage diversity
                used_data_types.update(result.selected.required_data_types)
            else:
                break  # No more viable algorithms

        return results

    def _check_filters(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> Optional[RejectionReason]:
        """
        Check all filters and return rejection reason if any fail.

        Returns:
            RejectionReason if algorithm should be rejected, None if viable
        """
        # Check deprecated
        if algorithm.deprecated:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason="Algorithm is deprecated",
                category="deprecated",
                details={"replacement": algorithm.replacement_algorithm}
            )

        # Check explicit exclusion
        if algorithm.id in context.criteria.excluded_algorithms:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason="Explicitly excluded by criteria",
                category="criteria"
            )

        # Check category restriction
        if context.criteria.allowed_categories:
            if algorithm.category not in context.criteria.allowed_categories:
                return RejectionReason(
                    algorithm_id=algorithm.id,
                    reason=f"Category {algorithm.category.value} not in allowed categories",
                    category="criteria",
                    details={"allowed": [c.value for c in context.criteria.allowed_categories]}
                )

        # Check data availability
        data_rejection = self._check_data_availability(algorithm, context)
        if data_rejection:
            return data_rejection

        # Check compute constraints
        compute_rejection = self._check_compute_constraints(algorithm, context)
        if compute_rejection:
            return compute_rejection

        # Check validation requirements
        if context.criteria.min_accuracy is not None:
            validation_rejection = self._check_validation(algorithm, context)
            if validation_rejection:
                return validation_rejection

        return None

    def _check_data_availability(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> Optional[RejectionReason]:
        """Check if required data types are available."""
        required = set(algorithm.required_data_types)
        available = context.available_data_types

        missing = required - available
        if missing:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason=f"Missing required data types: {[dt.value for dt in missing]}",
                category="data",
                details={
                    "required": [dt.value for dt in required],
                    "available": [dt.value for dt in available],
                    "missing": [dt.value for dt in missing]
                }
            )

        return None

    def _check_compute_constraints(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> Optional[RejectionReason]:
        """Check if algorithm fits within compute constraints."""
        resources = algorithm.resources
        constraints = context.compute_constraints

        # Check memory
        if resources.memory_mb and resources.memory_mb > constraints.max_memory_mb:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason=f"Requires {resources.memory_mb}MB memory, only {constraints.max_memory_mb}MB available",
                category="compute",
                details={
                    "required_memory_mb": resources.memory_mb,
                    "available_memory_mb": constraints.max_memory_mb
                }
            )

        # Check GPU requirement
        if resources.gpu_required and not constraints.gpu_available:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason="Requires GPU, but GPU not available",
                category="compute"
            )

        # Check GPU memory if both required and available
        if resources.gpu_memory_mb and constraints.gpu_memory_mb:
            if resources.gpu_memory_mb > constraints.gpu_memory_mb:
                return RejectionReason(
                    algorithm_id=algorithm.id,
                    reason=f"Requires {resources.gpu_memory_mb}MB GPU memory, only {constraints.gpu_memory_mb}MB available",
                    category="compute",
                    details={
                        "required_gpu_memory_mb": resources.gpu_memory_mb,
                        "available_gpu_memory_mb": constraints.gpu_memory_mb
                    }
                )

        # Check runtime
        if (resources.max_runtime_minutes and
            constraints.max_runtime_minutes and
            resources.max_runtime_minutes > constraints.max_runtime_minutes):
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason=f"May take up to {resources.max_runtime_minutes} minutes, limit is {constraints.max_runtime_minutes}",
                category="compute",
                details={
                    "algorithm_runtime_minutes": resources.max_runtime_minutes,
                    "constraint_runtime_minutes": constraints.max_runtime_minutes
                }
            )

        # Check distributed requirement
        if resources.distributed and not constraints.allow_distributed:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason="Requires distributed execution, but not available",
                category="compute"
            )

        return None

    def _check_validation(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> Optional[RejectionReason]:
        """Check if algorithm meets validation requirements."""
        if not algorithm.validation:
            return RejectionReason(
                algorithm_id=algorithm.id,
                reason="No validation data available",
                category="validation"
            )

        min_accuracy = context.criteria.min_accuracy
        if min_accuracy is not None:
            # Use median accuracy if available, otherwise min
            accuracy = algorithm.validation.accuracy_median or algorithm.validation.accuracy_min
            if accuracy is not None and accuracy < min_accuracy:
                return RejectionReason(
                    algorithm_id=algorithm.id,
                    reason=f"Accuracy {accuracy:.2f} below minimum {min_accuracy:.2f}",
                    category="validation",
                    details={
                        "algorithm_accuracy": accuracy,
                        "required_accuracy": min_accuracy
                    }
                )

        return None

    def _score_algorithm(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """
        Compute weighted score for algorithm.

        Scores each criterion 0.0-1.0, then applies weights.
        """
        scores = {}

        # Validation score
        scores["validation_score"] = self._score_validation(algorithm, context)

        # Category score
        scores["category_score"] = self._score_category(algorithm, context)

        # Determinism score
        scores["determinism_score"] = self._score_determinism(algorithm, context)

        # Resource efficiency
        scores["resource_efficiency"] = self._score_resource_efficiency(algorithm, context)

        # Data coverage
        scores["data_coverage"] = self._score_data_coverage(algorithm, context)

        # Region match
        scores["region_match"] = self._score_region_match(algorithm, context)

        # Compute weighted total
        total = sum(
            scores[criterion] * self.weights.get(criterion, 0.0)
            for criterion in scores
        )

        return total

    def _score_validation(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on validation data quality."""
        if not algorithm.validation:
            return 0.3 if not context.criteria.prefer_validated else 0.1

        validation = algorithm.validation

        # Base score from accuracy
        accuracy = validation.accuracy_median or validation.accuracy_min or 0.5
        accuracy_score = accuracy  # Already 0-1

        # Bonus for dataset count
        dataset_bonus = min(validation.validation_dataset_count / 100, 0.2)

        # Bonus for F1 score
        f1_bonus = (validation.f1_score or 0.5) * 0.1

        return min(accuracy_score + dataset_bonus + f1_bonus, 1.0)

    def _score_category(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on algorithm category."""
        category_scores = {
            AlgorithmCategory.BASELINE: 0.8 if context.criteria.prefer_baseline else 0.6,
            AlgorithmCategory.ADVANCED: 0.6 if context.criteria.prefer_baseline else 0.8,
            AlgorithmCategory.EXPERIMENTAL: 0.4,
        }
        return category_scores.get(algorithm.category, 0.5)

    def _score_determinism(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on determinism."""
        if context.criteria.prefer_deterministic:
            return 1.0 if algorithm.deterministic else 0.5
        return 0.8  # Neutral when not preferring

    # Resource efficiency scoring thresholds
    MEMORY_RATIO_HIGH_THRESHOLD = 0.8      # Using most of available memory
    MEMORY_RATIO_BALANCED_THRESHOLD = 0.3  # Good balance of resource usage

    def _score_resource_efficiency(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on how efficiently algorithm uses available resources."""
        constraints = context.compute_constraints
        resources = algorithm.resources

        if not resources.memory_mb:
            return 0.7  # Unknown, assume moderate

        # Guard against division by zero
        if constraints.max_memory_mb <= 0:
            logger.warning("Invalid max_memory_mb constraint (<=0), using default score")
            return 0.5  # Neutral score when constraints are invalid

        # Score higher when algorithm uses available memory well but not excessively
        memory_ratio = resources.memory_mb / constraints.max_memory_mb
        if memory_ratio > 1.0:
            return 0.0  # Should have been rejected
        elif memory_ratio > self.MEMORY_RATIO_HIGH_THRESHOLD:
            return 0.6  # Using most of available memory
        elif memory_ratio > self.MEMORY_RATIO_BALANCED_THRESHOLD:
            return 0.9  # Good balance
        else:
            return 0.8  # Very light, but maybe missing features

    def _score_data_coverage(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on how well algorithm uses available data."""
        required = set(algorithm.required_data_types)
        optional = set(algorithm.optional_data_types)
        available = context.available_data_types

        # Base score: all required data is available (should be if we got here)
        if not required.issubset(available):
            return 0.0

        # Bonus for optional data that's available
        optional_available = optional & available
        if optional:
            optional_ratio = len(optional_available) / len(optional)
        else:
            optional_ratio = 1.0

        # Score: 0.6 base + 0.4 * optional coverage
        return 0.6 + 0.4 * optional_ratio

    def _score_region_match(
        self,
        algorithm: AlgorithmMetadata,
        context: SelectionContext
    ) -> float:
        """Score based on validation in target region."""
        if not context.region:
            return 0.7  # No region specified

        if not algorithm.validation or not algorithm.validation.validated_regions:
            return 0.5  # No region validation data

        # Check if target region is validated
        validated_regions = [r.lower() for r in algorithm.validation.validated_regions]
        target_region = context.region.lower()

        if target_region in validated_regions:
            return 1.0  # Exact match

        # Check for partial matches (e.g., "north_america" matches "america")
        for region in validated_regions:
            if target_region in region or region in target_region:
                return 0.8

        return 0.4  # No match

    def _summarize_rejections(self, rejected: List[RejectionReason]) -> str:
        """Summarize rejection reasons."""
        if not rejected:
            return "No algorithms found for this event type"

        # Group by category
        by_category: Dict[str, int] = {}
        for r in rejected:
            by_category[r.category] = by_category.get(r.category, 0) + 1

        parts = [
            f"{count} rejected due to {category}"
            for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        ]

        return "All candidates rejected: " + ", ".join(parts)

    def _generate_rationale(
        self,
        selected: AlgorithmMetadata,
        score: float,
        context: SelectionContext
    ) -> str:
        """Generate human-readable selection rationale."""
        parts = [
            f"Selected {selected.name} (v{selected.version})",
            f"score={score:.3f}"
        ]

        # Add key factors
        if selected.validation:
            accuracy = selected.validation.accuracy_median or selected.validation.accuracy_min
            if accuracy:
                parts.append(f"accuracy={accuracy:.2%}")

        parts.append(f"category={selected.category.value}")

        if selected.deterministic:
            parts.append("deterministic=yes")

        # Add data types
        data_types = [dt.value for dt in selected.required_data_types]
        parts.append(f"uses={'+'.join(data_types)}")

        return "; ".join(parts)

    def get_supported_event_types(self) -> Set[str]:
        """Get all event types supported by registered algorithms."""
        event_types: Set[str] = set()
        for algorithm in self.registry.list_all():
            event_types.update(algorithm.event_types)
        return event_types

    def get_algorithms_by_data_type(
        self,
        data_type: DataType
    ) -> List[AlgorithmMetadata]:
        """Get algorithms that require a specific data type."""
        return [
            algo for algo in self.registry.list_all()
            if data_type in algo.required_data_types
        ]

    def explain_selection(self, result: SelectionResult) -> str:
        """
        Generate detailed explanation of selection decision.

        Args:
            result: Selection result to explain

        Returns:
            Multi-line explanation string
        """
        lines = [
            "Algorithm Selection Report",
            "=" * 50,
            f"Event Class: {result.context.event_class}",
            f"Available Data: {[dt.value for dt in result.context.available_data_types]}",
            "",
        ]

        if result.success:
            lines.extend([
                f"SELECTED: {result.selected.id}",
                f"  Name: {result.selected.name}",
                f"  Version: {result.selected.version}",
                f"  Category: {result.selected.category.value}",
                f"  Score: {result.scores[result.selected.id]:.3f}",
                "",
            ])

            if result.alternatives:
                lines.append("ALTERNATIVES:")
                for alt in result.alternatives:
                    lines.append(
                        f"  - {alt.id}: {result.scores[alt.id]:.3f}"
                    )
                lines.append("")
        else:
            lines.extend([
                "SELECTION FAILED",
                f"Reason: {result.rationale}",
                "",
            ])

        if result.rejected:
            lines.append("REJECTED:")
            for rejection in result.rejected[:5]:  # Show first 5
                lines.append(
                    f"  - {rejection.algorithm_id}: {rejection.reason}"
                )
            if len(result.rejected) > 5:
                lines.append(f"  ... and {len(result.rejected) - 5} more")

        return "\n".join(lines)
