"""
Conflict Resolution for Multi-Sensor Data Fusion.

Provides tools for resolving disagreements between multiple data sources,
including:
- Per-pixel conflict detection
- Multiple resolution strategies (voting, weighted average, etc.)
- Confidence-based conflict arbitration
- Consensus generation with provenance tracking

Key Concepts:
- Conflict occurs when multiple sources provide different values for same location
- Resolution strategies balance data quality, confidence, and sensor reliability
- Consensus outputs include confidence metrics and provenance
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between data sources."""
    MAJORITY_VOTE = "majority_vote"          # Discrete: most common value wins
    WEIGHTED_VOTE = "weighted_vote"          # Discrete: quality-weighted voting
    MEAN = "mean"                            # Continuous: simple average
    WEIGHTED_MEAN = "weighted_mean"          # Continuous: quality-weighted average
    MEDIAN = "median"                        # Continuous: median value
    HIGHEST_CONFIDENCE = "highest_confidence"  # Use source with highest confidence
    PRIORITY_ORDER = "priority_order"        # Use priority-ordered source list
    MINIMUM = "minimum"                      # Conservative: use minimum value
    MAXIMUM = "maximum"                      # Aggressive: use maximum value
    RANGE_CHECK = "range_check"              # Remove outliers, then average


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""
    NONE = "none"              # No conflict (all sources agree)
    LOW = "low"                # Minor disagreement within tolerance
    MEDIUM = "medium"          # Moderate disagreement requiring resolution
    HIGH = "high"              # Significant disagreement with uncertainty
    CRITICAL = "critical"      # Sources fundamentally disagree


@dataclass
class ConflictThresholds:
    """
    Thresholds for conflict detection.

    Attributes:
        absolute_tolerance: Maximum absolute difference for no conflict
        relative_tolerance: Maximum relative difference (fraction) for no conflict
        min_agreement_ratio: Minimum fraction of sources that must agree
        outlier_sigma: Standard deviations for outlier detection
        min_sources_for_consensus: Minimum sources needed for consensus
    """
    absolute_tolerance: float = 0.1
    relative_tolerance: float = 0.05
    min_agreement_ratio: float = 0.5
    outlier_sigma: float = 2.0
    min_sources_for_consensus: int = 2


@dataclass
class ConflictConfig:
    """
    Configuration for conflict resolution.

    Attributes:
        strategy: Resolution strategy to use
        thresholds: Conflict detection thresholds
        use_quality_weights: Weight sources by quality
        source_priorities: Priority order for sources (highest first)
        fallback_strategy: Strategy if primary fails
        track_provenance: Record which source contributed to each pixel
    """
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_MEAN
    thresholds: ConflictThresholds = field(default_factory=ConflictThresholds)
    use_quality_weights: bool = True
    source_priorities: Optional[List[str]] = None
    fallback_strategy: Optional[ConflictResolutionStrategy] = None
    track_provenance: bool = True


@dataclass
class SourceLayer:
    """
    A data layer from a single source for conflict resolution.

    Attributes:
        data: Data array
        source_id: Identifier for this source
        quality: Per-pixel quality weights (0-1)
        confidence: Overall source confidence (0-1)
        sensor_type: Type of sensor (for priority)
        timestamp: Observation timestamp
        metadata: Additional source metadata
    """
    data: np.ndarray
    source_id: str
    quality: Optional[np.ndarray] = None
    confidence: float = 1.0
    sensor_type: str = "unknown"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize quality if not provided."""
        if self.quality is None:
            # Default to full quality where data is valid
            if self.data.dtype.kind == 'f':
                self.quality = (~np.isnan(self.data)).astype(np.float32)
            else:
                self.quality = np.ones_like(self.data, dtype=np.float32)


@dataclass
class ConflictMap:
    """
    Map of conflict locations and severities.

    Attributes:
        severity_map: Per-pixel conflict severity
        disagreement_map: Quantified disagreement (e.g., std dev)
        source_count_map: Number of valid sources per pixel
        agreement_ratio_map: Fraction of sources that agree
    """
    severity_map: np.ndarray
    disagreement_map: np.ndarray
    source_count_map: np.ndarray
    agreement_ratio_map: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary summary."""
        severity_counts = {}
        for severity in ConflictSeverity:
            severity_counts[severity.value] = int(np.sum(self.severity_map == severity.value))

        return {
            "total_pixels": int(self.severity_map.size),
            "severity_counts": severity_counts,
            "mean_disagreement": float(np.nanmean(self.disagreement_map)),
            "max_disagreement": float(np.nanmax(self.disagreement_map)),
            "mean_source_count": float(np.mean(self.source_count_map)),
            "mean_agreement_ratio": float(np.nanmean(self.agreement_ratio_map)),
        }


@dataclass
class ConflictResolutionResult:
    """
    Result from conflict resolution.

    Attributes:
        resolved_data: Resolved data array
        confidence_map: Per-pixel confidence in resolved value
        provenance_map: Which source contributed to each pixel
        conflict_map: Map of conflict locations and severities
        strategy_used: Strategy that was applied
        diagnostics: Additional diagnostic information
    """
    resolved_data: np.ndarray
    confidence_map: np.ndarray
    provenance_map: Optional[np.ndarray] = None
    conflict_map: Optional[ConflictMap] = None
    strategy_used: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "data_shape": list(self.resolved_data.shape),
            "strategy_used": self.strategy_used,
            "mean_confidence": float(np.nanmean(self.confidence_map)),
            "conflict_summary": self.conflict_map.to_dict() if self.conflict_map else None,
            "diagnostics": self.diagnostics,
        }


class ConflictDetector:
    """
    Detects conflicts between multiple data sources.

    Identifies pixels where sources disagree beyond tolerance thresholds.
    """

    def __init__(self, thresholds: Optional[ConflictThresholds] = None):
        """
        Initialize conflict detector.

        Args:
            thresholds: Conflict detection thresholds
        """
        self.thresholds = thresholds or ConflictThresholds()

    def detect(
        self,
        layers: List[SourceLayer],
    ) -> ConflictMap:
        """
        Detect conflicts between data sources.

        Args:
            layers: List of source layers to compare

        Returns:
            ConflictMap with conflict locations and severities
        """
        if len(layers) < 2:
            # No conflict possible with single source
            shape = layers[0].data.shape if layers else (0, 0)
            return ConflictMap(
                severity_map=np.full(shape, ConflictSeverity.NONE.value, dtype=object),
                disagreement_map=np.zeros(shape),
                source_count_map=np.ones(shape, dtype=np.int32) if layers else np.zeros(shape, dtype=np.int32),
                agreement_ratio_map=np.ones(shape),
            )

        # Stack data for analysis
        shape = layers[0].data.shape
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Count valid sources per pixel
        valid_mask = quality_stack > 0
        source_count = np.sum(valid_mask, axis=0)

        # Calculate disagreement metrics
        # Use masked statistics to ignore invalid sources
        with np.errstate(invalid='ignore'):
            masked_data = np.where(valid_mask, data_stack, np.nan)

            # Mean and standard deviation
            data_mean = np.nanmean(masked_data, axis=0)
            data_std = np.nanstd(masked_data, axis=0)

            # Range (max - min)
            data_range = np.nanmax(masked_data, axis=0) - np.nanmin(masked_data, axis=0)

        # Calculate agreement ratio
        # Pixels that are within tolerance of the mean
        within_tolerance = np.zeros_like(data_stack, dtype=bool)
        abs_tol = self.thresholds.absolute_tolerance
        rel_tol = self.thresholds.relative_tolerance

        for i in range(len(layers)):
            abs_diff = np.abs(data_stack[i] - data_mean)
            rel_diff = np.abs(abs_diff / (np.abs(data_mean) + 1e-10))
            within_tolerance[i] = (abs_diff <= abs_tol) | (rel_diff <= rel_tol)

        # Count sources within tolerance
        sources_agree = np.sum(within_tolerance & valid_mask, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            agreement_ratio = np.where(source_count > 0, sources_agree / source_count, 1.0)

        # Determine severity
        severity_map = self._classify_severity(
            source_count,
            data_std,
            data_range,
            agreement_ratio,
            data_mean
        )

        return ConflictMap(
            severity_map=severity_map,
            disagreement_map=data_std,
            source_count_map=source_count.astype(np.int32),
            agreement_ratio_map=agreement_ratio,
        )

    def _classify_severity(
        self,
        source_count: np.ndarray,
        data_std: np.ndarray,
        data_range: np.ndarray,
        agreement_ratio: np.ndarray,
        data_mean: np.ndarray,
    ) -> np.ndarray:
        """Classify conflict severity per pixel."""
        shape = source_count.shape
        severity = np.full(shape, ConflictSeverity.NONE.value, dtype=object)

        # No conflict if single source or all agree
        single_source = source_count <= 1
        all_agree = agreement_ratio >= 0.99

        # Low conflict: minor disagreement
        abs_tol = self.thresholds.absolute_tolerance
        rel_tol = self.thresholds.relative_tolerance

        with np.errstate(invalid='ignore'):
            relative_std = data_std / (np.abs(data_mean) + 1e-10)

        low_conflict = (
            ~single_source & ~all_agree &
            ((data_std <= abs_tol * 2) | (relative_std <= rel_tol * 2)) &
            (agreement_ratio >= self.thresholds.min_agreement_ratio)
        )

        # Medium conflict: moderate disagreement
        medium_conflict = (
            ~single_source & ~all_agree & ~low_conflict &
            ((data_std <= abs_tol * 5) | (relative_std <= rel_tol * 5)) &
            (agreement_ratio >= self.thresholds.min_agreement_ratio * 0.5)
        )

        # High conflict: significant disagreement
        high_conflict = (
            ~single_source & ~all_agree & ~low_conflict & ~medium_conflict &
            (agreement_ratio >= 0.2)
        )

        # Critical conflict: fundamental disagreement
        critical_conflict = (
            ~single_source & ~all_agree & ~low_conflict & ~medium_conflict & ~high_conflict
        )

        # Apply classifications
        severity[low_conflict] = ConflictSeverity.LOW.value
        severity[medium_conflict] = ConflictSeverity.MEDIUM.value
        severity[high_conflict] = ConflictSeverity.HIGH.value
        severity[critical_conflict] = ConflictSeverity.CRITICAL.value

        return severity


class ConflictResolver:
    """
    Resolves conflicts between multiple data sources.

    Applies configurable resolution strategies to produce consensus output.
    """

    def __init__(self, config: Optional[ConflictConfig] = None):
        """
        Initialize conflict resolver.

        Args:
            config: Conflict resolution configuration
        """
        self.config = config or ConflictConfig()
        self.detector = ConflictDetector(self.config.thresholds)

    def resolve(
        self,
        layers: List[SourceLayer],
    ) -> ConflictResolutionResult:
        """
        Resolve conflicts between data sources.

        Args:
            layers: List of source layers to resolve

        Returns:
            ConflictResolutionResult with consensus data
        """
        if not layers:
            raise ValueError("No layers provided for conflict resolution")

        if len(layers) == 1:
            # Single source - no conflict
            return ConflictResolutionResult(
                resolved_data=layers[0].data.copy(),
                confidence_map=layers[0].quality.copy(),
                provenance_map=np.full(layers[0].data.shape, 0, dtype=np.int8),
                conflict_map=None,
                strategy_used="single_source",
                diagnostics={"num_sources": 1},
            )

        # Detect conflicts
        conflict_map = self.detector.detect(layers)

        # Apply resolution strategy
        strategy = self.config.strategy

        if strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            resolved, confidence, provenance = self._majority_vote(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.WEIGHTED_VOTE:
            resolved, confidence, provenance = self._weighted_vote(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.MEAN:
            resolved, confidence, provenance = self._mean(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.WEIGHTED_MEAN:
            resolved, confidence, provenance = self._weighted_mean(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.MEDIAN:
            resolved, confidence, provenance = self._median(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            resolved, confidence, provenance = self._highest_confidence(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.PRIORITY_ORDER:
            resolved, confidence, provenance = self._priority_order(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.MINIMUM:
            resolved, confidence, provenance = self._minimum(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.MAXIMUM:
            resolved, confidence, provenance = self._maximum(layers, conflict_map)
        elif strategy == ConflictResolutionStrategy.RANGE_CHECK:
            resolved, confidence, provenance = self._range_check(layers, conflict_map)
        else:
            logger.warning(f"Unknown strategy {strategy}, using weighted mean")
            resolved, confidence, provenance = self._weighted_mean(layers, conflict_map)

        # Adjust confidence based on conflict severity
        confidence = self._adjust_confidence_for_conflicts(confidence, conflict_map)

        return ConflictResolutionResult(
            resolved_data=resolved,
            confidence_map=confidence,
            provenance_map=provenance if self.config.track_provenance else None,
            conflict_map=conflict_map,
            strategy_used=strategy.value,
            diagnostics={
                "num_sources": len(layers),
                "source_ids": [l.source_id for l in layers],
                "mean_agreement": float(np.nanmean(conflict_map.agreement_ratio_map)),
            },
        )

    def _majority_vote(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve by majority vote (for discrete/categorical data)."""
        shape = layers[0].data.shape

        # Get unique values and their counts
        data_stack = np.stack([l.data for l in layers])
        quality_stack = np.stack([l.quality for l in layers])
        valid_mask = quality_stack > 0

        # For each pixel, find most common value
        resolved = np.zeros(shape, dtype=layers[0].data.dtype)
        provenance = np.zeros(shape, dtype=np.int8)

        for idx in np.ndindex(shape):
            valid_values = data_stack[:, idx[0], idx[1]][valid_mask[:, idx[0], idx[1]]]
            if len(valid_values) == 0:
                resolved[idx] = np.nan if layers[0].data.dtype.kind == 'f' else 0
                provenance[idx] = -1
            else:
                # Find most common value
                unique, counts = np.unique(valid_values, return_counts=True)
                winner_idx = np.argmax(counts)
                resolved[idx] = unique[winner_idx]
                # Provenance: first source with winning value
                for i, layer in enumerate(layers):
                    if valid_mask[i, idx[0], idx[1]] and layer.data[idx] == unique[winner_idx]:
                        provenance[idx] = i
                        break

        # Confidence based on vote fraction
        confidence = conflict_map.agreement_ratio_map.astype(np.float32)

        return resolved, confidence, provenance

    def _weighted_vote(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve by quality-weighted voting (for discrete data)."""
        shape = layers[0].data.shape

        data_stack = np.stack([l.data for l in layers])
        quality_stack = np.stack([l.quality * l.confidence for l in layers])
        valid_mask = quality_stack > 0

        resolved = np.zeros(shape, dtype=layers[0].data.dtype)
        provenance = np.zeros(shape, dtype=np.int8)
        confidence = np.zeros(shape, dtype=np.float32)

        for idx in np.ndindex(shape):
            valid_indices = np.where(valid_mask[:, idx[0], idx[1]])[0]
            if len(valid_indices) == 0:
                resolved[idx] = np.nan if layers[0].data.dtype.kind == 'f' else 0
                provenance[idx] = -1
                confidence[idx] = 0.0
            else:
                valid_values = data_stack[valid_indices, idx[0], idx[1]]
                valid_weights = quality_stack[valid_indices, idx[0], idx[1]]

                # Weighted vote: sum weights per unique value
                unique = np.unique(valid_values)
                weighted_counts = np.array([
                    np.sum(valid_weights[valid_values == u])
                    for u in unique
                ])
                winner_idx = np.argmax(weighted_counts)
                resolved[idx] = unique[winner_idx]

                # Confidence: fraction of total weight that voted for winner
                total_weight = np.sum(valid_weights)
                if total_weight > 0:
                    confidence[idx] = weighted_counts[winner_idx] / total_weight

                # Provenance: source with highest weight for winning value
                winning_value = unique[winner_idx]
                best_weight = 0
                for i in valid_indices:
                    if data_stack[i, idx[0], idx[1]] == winning_value:
                        if quality_stack[i, idx[0], idx[1]] > best_weight:
                            best_weight = quality_stack[i, idx[0], idx[1]]
                            provenance[idx] = i

        return resolved, confidence, provenance

    def _mean(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve by simple mean (for continuous data)."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Mask invalid data
        masked_data = np.where(quality_stack > 0, data_stack, np.nan)

        # Calculate mean
        resolved = np.nanmean(masked_data, axis=0)

        # Confidence based on number of valid sources
        num_valid = np.sum(quality_stack > 0, axis=0).astype(np.float32)
        confidence = num_valid / len(layers)

        # Provenance: -1 for mean (multiple sources)
        provenance = np.full(resolved.shape, -1, dtype=np.int8)

        return resolved.astype(layers[0].data.dtype), confidence, provenance

    def _weighted_mean(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve by quality-weighted mean (for continuous data)."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality * l.confidence for l in layers])

        # Weighted sum
        weighted_sum = np.nansum(data_stack * quality_stack, axis=0)
        weight_sum = np.nansum(quality_stack, axis=0)

        # Avoid division by zero
        weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        resolved = weighted_sum / weight_sum

        # Handle pixels with no valid data
        no_data_mask = np.all(quality_stack == 0, axis=0)
        resolved[no_data_mask] = np.nan

        # Confidence based on total weight relative to max possible
        max_weight = len(layers)  # If all sources had quality=1 and confidence=1
        confidence = (weight_sum / max_weight).astype(np.float32)
        confidence[no_data_mask] = 0.0

        # Provenance: -1 for weighted mean (multiple sources)
        provenance = np.full(resolved.shape, -1, dtype=np.int8)

        return resolved.astype(layers[0].data.dtype), confidence, provenance

    def _median(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve by median (for continuous data, outlier-robust)."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Mask invalid data
        masked_data = np.where(quality_stack > 0, data_stack, np.nan)

        # Calculate median
        resolved = np.nanmedian(masked_data, axis=0)

        # Confidence based on number of valid sources and agreement
        num_valid = np.sum(quality_stack > 0, axis=0).astype(np.float32)
        confidence = (num_valid / len(layers)) * conflict_map.agreement_ratio_map

        # Provenance: -1 for median (multiple sources)
        provenance = np.full(resolved.shape, -1, dtype=np.int8)

        return resolved.astype(layers[0].data.dtype), confidence.astype(np.float32), provenance

    def _highest_confidence(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use source with highest per-pixel confidence."""
        shape = layers[0].data.shape

        # Stack quality-weighted confidence
        confidence_stack = np.stack([l.quality * l.confidence for l in layers])
        data_stack = np.stack([l.data for l in layers])

        # Find index of max confidence at each pixel
        best_idx = np.argmax(confidence_stack, axis=0)

        # Select data from best source
        resolved = np.take_along_axis(data_stack, best_idx[np.newaxis, ...], axis=0)[0]

        # Confidence is the max confidence value
        confidence = np.take_along_axis(confidence_stack, best_idx[np.newaxis, ...], axis=0)[0]

        # Provenance is the index of the best source
        provenance = best_idx.astype(np.int8)

        return resolved, confidence.astype(np.float32), provenance

    def _priority_order(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use sources in priority order (first valid wins)."""
        shape = layers[0].data.shape

        # Determine priority order
        if self.config.source_priorities:
            # Reorder layers by priority
            priority_map = {sid: i for i, sid in enumerate(self.config.source_priorities)}
            sorted_layers = sorted(
                enumerate(layers),
                key=lambda x: priority_map.get(x[1].source_id, 999)
            )
        else:
            # Use layers in order provided
            sorted_layers = list(enumerate(layers))

        # Initialize with nodata
        resolved = np.full(shape, np.nan if layers[0].data.dtype.kind == 'f' else 0, dtype=layers[0].data.dtype)
        confidence = np.zeros(shape, dtype=np.float32)
        provenance = np.full(shape, -1, dtype=np.int8)
        filled = np.zeros(shape, dtype=bool)

        # Fill from highest to lowest priority
        for original_idx, layer in sorted_layers:
            valid_mask = layer.quality > 0
            to_fill = valid_mask & ~filled

            resolved[to_fill] = layer.data[to_fill]
            confidence[to_fill] = (layer.quality * layer.confidence)[to_fill]
            provenance[to_fill] = original_idx
            filled |= valid_mask

        return resolved, confidence, provenance

    def _minimum(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use minimum value (conservative estimate)."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Mask invalid data with inf for min operation
        masked_data = np.where(quality_stack > 0, data_stack, np.inf)

        # Find minimum (excluding inf)
        resolved = np.nanmin(masked_data, axis=0)
        resolved[np.isinf(resolved)] = np.nan

        # Find which source provided the minimum
        min_idx = np.argmin(masked_data, axis=0)

        # Confidence: lower if sources disagree significantly
        confidence = conflict_map.agreement_ratio_map.astype(np.float32)

        provenance = min_idx.astype(np.int8)

        return resolved.astype(layers[0].data.dtype), confidence, provenance

    def _maximum(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use maximum value (aggressive estimate)."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Mask invalid data with -inf for max operation
        masked_data = np.where(quality_stack > 0, data_stack, -np.inf)

        # Find maximum (excluding -inf)
        resolved = np.nanmax(masked_data, axis=0)
        resolved[np.isinf(resolved)] = np.nan

        # Find which source provided the maximum
        max_idx = np.argmax(masked_data, axis=0)

        # Confidence: lower if sources disagree significantly
        confidence = conflict_map.agreement_ratio_map.astype(np.float32)

        provenance = max_idx.astype(np.int8)

        return resolved.astype(layers[0].data.dtype), confidence, provenance

    def _range_check(
        self,
        layers: List[SourceLayer],
        conflict_map: ConflictMap,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove outliers then average."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])
        quality_stack = np.stack([l.quality for l in layers])

        # Mask invalid data
        masked_data = np.where(quality_stack > 0, data_stack, np.nan)

        # Calculate mean and std for outlier detection
        data_mean = np.nanmean(masked_data, axis=0)
        data_std = np.nanstd(masked_data, axis=0)

        # Mark outliers
        sigma = self.config.thresholds.outlier_sigma
        outlier_mask = np.abs(masked_data - data_mean) > (sigma * data_std)

        # Replace outliers with nan
        cleaned_data = np.where(outlier_mask, np.nan, masked_data)

        # Average cleaned data
        resolved = np.nanmean(cleaned_data, axis=0)

        # Confidence based on fraction of data retained
        num_retained = np.sum(~np.isnan(cleaned_data), axis=0).astype(np.float32)
        num_valid = np.sum(~np.isnan(masked_data), axis=0).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            confidence = np.where(num_valid > 0, num_retained / num_valid, 0.0)

        # Provenance: -1 for averaged data
        provenance = np.full(resolved.shape, -1, dtype=np.int8)

        return resolved.astype(layers[0].data.dtype), confidence.astype(np.float32), provenance

    def _adjust_confidence_for_conflicts(
        self,
        confidence: np.ndarray,
        conflict_map: ConflictMap,
    ) -> np.ndarray:
        """Reduce confidence in areas of high conflict."""
        adjusted = confidence.copy()

        # Penalty factors for conflict severity
        penalties = {
            ConflictSeverity.NONE.value: 1.0,
            ConflictSeverity.LOW.value: 0.95,
            ConflictSeverity.MEDIUM.value: 0.8,
            ConflictSeverity.HIGH.value: 0.6,
            ConflictSeverity.CRITICAL.value: 0.3,
        }

        for severity_val, penalty in penalties.items():
            mask = conflict_map.severity_map == severity_val
            adjusted[mask] *= penalty

        return adjusted


class ConsensusBuilder:
    """
    Builds consensus from multiple conflict resolution results.

    Provides multi-stage conflict resolution and consensus tracking.
    """

    def __init__(
        self,
        primary_config: Optional[ConflictConfig] = None,
        secondary_config: Optional[ConflictConfig] = None,
    ):
        """
        Initialize consensus builder.

        Args:
            primary_config: Primary resolution configuration
            secondary_config: Secondary/fallback configuration
        """
        self.primary_resolver = ConflictResolver(primary_config)
        self.secondary_resolver = ConflictResolver(secondary_config) if secondary_config else None

    def build_consensus(
        self,
        layers: List[SourceLayer],
        confidence_threshold: float = 0.5,
    ) -> ConflictResolutionResult:
        """
        Build consensus from multiple sources with fallback.

        Args:
            layers: List of source layers
            confidence_threshold: Minimum confidence for primary result

        Returns:
            ConflictResolutionResult with consensus
        """
        # Primary resolution
        primary_result = self.primary_resolver.resolve(layers)

        # Check if secondary resolution needed
        if self.secondary_resolver is not None:
            low_confidence_mask = primary_result.confidence_map < confidence_threshold

            if np.any(low_confidence_mask):
                # Apply secondary resolution
                secondary_result = self.secondary_resolver.resolve(layers)

                # Blend results: use secondary where primary confidence is low
                final_data = np.where(
                    low_confidence_mask,
                    secondary_result.resolved_data,
                    primary_result.resolved_data
                )
                final_confidence = np.where(
                    low_confidence_mask,
                    secondary_result.confidence_map,
                    primary_result.confidence_map
                )

                # Update provenance
                if primary_result.provenance_map is not None and secondary_result.provenance_map is not None:
                    # Mark secondary-resolved pixels with negative offset
                    final_provenance = np.where(
                        low_confidence_mask,
                        secondary_result.provenance_map - 100,  # Offset to indicate secondary
                        primary_result.provenance_map
                    )
                else:
                    final_provenance = None

                return ConflictResolutionResult(
                    resolved_data=final_data,
                    confidence_map=final_confidence,
                    provenance_map=final_provenance,
                    conflict_map=primary_result.conflict_map,
                    strategy_used=f"{primary_result.strategy_used}+{secondary_result.strategy_used}",
                    diagnostics={
                        **primary_result.diagnostics,
                        "secondary_pixels": int(np.sum(low_confidence_mask)),
                        "secondary_strategy": secondary_result.strategy_used,
                    },
                )

        return primary_result


# Convenience functions

def detect_conflicts(
    layers: List[SourceLayer],
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.05,
) -> ConflictMap:
    """
    Detect conflicts between data sources.

    Args:
        layers: List of source layers
        absolute_tolerance: Max absolute difference for no conflict
        relative_tolerance: Max relative difference for no conflict

    Returns:
        ConflictMap with conflict locations and severities
    """
    thresholds = ConflictThresholds(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    detector = ConflictDetector(thresholds)
    return detector.detect(layers)


def resolve_conflicts(
    layers: List[SourceLayer],
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_MEAN,
) -> ConflictResolutionResult:
    """
    Resolve conflicts between data sources.

    Args:
        layers: List of source layers
        strategy: Resolution strategy to use

    Returns:
        ConflictResolutionResult with consensus data
    """
    config = ConflictConfig(strategy=strategy)
    resolver = ConflictResolver(config)
    return resolver.resolve(layers)


def build_consensus(
    data_arrays: List[np.ndarray],
    quality_arrays: Optional[List[np.ndarray]] = None,
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_MEAN,
) -> np.ndarray:
    """
    Simple convenience function to build consensus from arrays.

    Args:
        data_arrays: List of data arrays to combine
        quality_arrays: Optional list of quality arrays
        strategy: Resolution strategy

    Returns:
        Consensus array
    """
    layers = []
    for i, data in enumerate(data_arrays):
        quality = quality_arrays[i] if quality_arrays else None
        layers.append(SourceLayer(
            data=data,
            source_id=f"source_{i}",
            quality=quality,
        ))

    result = resolve_conflicts(layers, strategy)
    return result.resolved_data
