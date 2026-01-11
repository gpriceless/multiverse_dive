"""
Ensemble Fusion Flood Detection Algorithm

Combines multiple flood detection algorithms through weighted ensemble fusion.
Improves robustness by leveraging the strengths of different approaches
(SAR threshold, optical NDWI, change detection, ML-based methods).

Algorithm ID: flood.advanced.ensemble_fusion
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Ensemble fusion methods."""
    WEIGHTED_AVERAGE = "weighted_average"  # Simple weighted average
    VOTING = "voting"  # Majority/weighted voting
    BAYESIAN = "bayesian"  # Bayesian combination
    STACKING = "stacking"  # Stacking with meta-learner
    RELIABILITY_WEIGHTED = "reliability_weighted"  # Weight by local reliability


class DisagreementHandling(Enum):
    """How to handle disagreement between algorithms."""
    CONSERVATIVE = "conservative"  # Only flag as flood if majority agrees
    LIBERAL = "liberal"  # Flag as flood if any detector triggers
    CONFIDENCE = "confidence"  # Use confidence-weighted decision
    ADAPTIVE = "adaptive"  # Adapt based on local conditions


@dataclass
class AlgorithmWeight:
    """Weight configuration for a single algorithm."""

    algorithm_id: str  # ID of the algorithm
    base_weight: float = 1.0  # Base weight in ensemble
    reliability_by_region: Dict[str, float] = field(default_factory=dict)
    reliability_by_condition: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.base_weight < 0:
            raise ValueError(f"base_weight must be non-negative, got {self.base_weight}")


@dataclass
class EnsembleFusionConfig:
    """Configuration for ensemble fusion algorithm."""

    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.RELIABILITY_WEIGHTED
    disagreement_handling: DisagreementHandling = DisagreementHandling.CONFIDENCE

    # Algorithm weights (defaults if not specified)
    algorithm_weights: List[AlgorithmWeight] = field(default_factory=list)

    # Voting thresholds
    voting_threshold: float = 0.5  # Fraction of algorithms that must agree
    confidence_threshold: float = 0.5  # Output probability threshold

    # Reliability estimation
    use_spatial_reliability: bool = True  # Adjust weights by spatial context
    use_temporal_reliability: bool = True  # Adjust weights by temporal consistency
    reliability_window_size: int = 5  # Window for local reliability estimation

    # Output settings
    generate_agreement_map: bool = True  # Generate per-pixel agreement statistics
    generate_uncertainty_map: bool = True  # Generate uncertainty estimates
    min_algorithms: int = 2  # Minimum algorithms required for valid output

    def __post_init__(self):
        if not 0.0 <= self.voting_threshold <= 1.0:
            raise ValueError(f"voting_threshold must be in [0, 1], got {self.voting_threshold}")
        if self.min_algorithms < 1:
            raise ValueError(f"min_algorithms must be >= 1, got {self.min_algorithms}")


@dataclass
class AlgorithmResult:
    """Result from a single algorithm in the ensemble."""

    algorithm_id: str
    flood_extent: np.ndarray  # Binary mask (H, W)
    confidence: np.ndarray  # Confidence scores (H, W)
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.flood_extent.shape != self.confidence.shape:
            raise ValueError(
                f"Shape mismatch: flood_extent {self.flood_extent.shape} "
                f"vs confidence {self.confidence.shape}"
            )


@dataclass
class EnsembleFusionResult:
    """Results from ensemble fusion."""

    flood_extent: np.ndarray  # Fused binary mask (H, W)
    flood_probability: np.ndarray  # Fused probability (H, W)
    confidence_raster: np.ndarray  # Confidence in fusion (H, W)
    agreement_map: Optional[np.ndarray]  # Per-pixel algorithm agreement (H, W)
    uncertainty_map: Optional[np.ndarray]  # Uncertainty estimate (H, W)
    per_algorithm_weights: Dict[str, np.ndarray]  # Final weights per algorithm
    metadata: Dict[str, Any]
    statistics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            "flood_extent": self.flood_extent,
            "flood_probability": self.flood_probability,
            "confidence_raster": self.confidence_raster,
            "metadata": self.metadata,
            "statistics": self.statistics
        }
        if self.agreement_map is not None:
            result["agreement_map"] = self.agreement_map
        if self.uncertainty_map is not None:
            result["uncertainty_map"] = self.uncertainty_map
        return result


class EnsembleFusionAlgorithm:
    """
    Ensemble Fusion for Flood Detection.

    This algorithm combines outputs from multiple flood detection algorithms
    into a single, more robust result. Different fusion strategies are available:

    - Weighted Average: Simple combination based on algorithm weights
    - Voting: Majority voting with optional weighting
    - Bayesian: Probabilistic combination using prior performance
    - Reliability-Weighted: Dynamic weights based on local reliability

    Requirements:
        - Outputs from 2+ flood detection algorithms
        - Each algorithm must provide flood_extent and confidence

    Outputs:
        - flood_extent: Fused binary flood mask
        - flood_probability: Continuous probability map
        - agreement_map: Per-pixel algorithm agreement
        - uncertainty_map: Spatial uncertainty estimate
    """

    METADATA = {
        "id": "flood.advanced.ensemble_fusion",
        "name": "Ensemble Fusion",
        "category": "advanced",
        "event_types": ["flood.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "algorithm_outputs": {
                    "min_count": 2,
                    "required_fields": ["flood_extent", "confidence"]
                }
            },
            "compute": {
                "memory_gb": 4,
                "gpu": False
            }
        },
        "validation": {
            "accuracy_range": [0.80, 0.95],
            "validated_regions": ["north_america", "europe", "southeast_asia"],
            "citations": [
                "doi:10.1016/j.rse.2018.04.028",  # Multi-algorithm flood mapping
                "doi:10.1109/TGRS.2019.2952321"  # Ensemble methods for remote sensing
            ]
        }
    }

    # Default weights for common algorithms
    DEFAULT_WEIGHTS = {
        "flood.baseline.threshold_sar": AlgorithmWeight(
            algorithm_id="flood.baseline.threshold_sar",
            base_weight=1.0,
            reliability_by_condition={
                "cloudy": 1.2,  # SAR works well under clouds
                "clear": 0.9
            }
        ),
        "flood.baseline.ndwi_optical": AlgorithmWeight(
            algorithm_id="flood.baseline.ndwi_optical",
            base_weight=1.0,
            reliability_by_condition={
                "cloudy": 0.3,  # Optical degrades under clouds
                "clear": 1.3
            }
        ),
        "flood.baseline.change_detection": AlgorithmWeight(
            algorithm_id="flood.baseline.change_detection",
            base_weight=0.9,
            reliability_by_condition={
                "has_pre_event": 1.2,
                "no_pre_event": 0.0  # Unusable without baseline
            }
        ),
        "flood.advanced.unet_segmentation": AlgorithmWeight(
            algorithm_id="flood.advanced.unet_segmentation",
            base_weight=1.2,  # Generally higher accuracy
            reliability_by_region={
                "urban": 1.3,  # Good in complex scenes
                "rural": 1.0
            }
        )
    }

    def __init__(self, config: Optional[EnsembleFusionConfig] = None):
        """
        Initialize ensemble fusion algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or EnsembleFusionConfig()

        # Build weight lookup with defaults
        self._weights: Dict[str, AlgorithmWeight] = dict(self.DEFAULT_WEIGHTS)
        for w in self.config.algorithm_weights:
            self._weights[w.algorithm_id] = w

        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Fusion method: {self.config.fusion_method.value}")

    def execute(
        self,
        algorithm_results: List[AlgorithmResult],
        valid_mask: Optional[np.ndarray] = None,
        conditions: Optional[Dict[str, Any]] = None,
        pixel_size_m: float = 10.0
    ) -> EnsembleFusionResult:
        """
        Execute ensemble fusion on algorithm outputs.

        Args:
            algorithm_results: List of results from individual algorithms
            valid_mask: Optional valid data mask (H, W)
            conditions: Environmental conditions for weight adjustment
            pixel_size_m: Pixel size in meters

        Returns:
            EnsembleFusionResult containing fused outputs
        """
        logger.info(f"Starting ensemble fusion with {len(algorithm_results)} algorithms")

        # Validate inputs
        if len(algorithm_results) < self.config.min_algorithms:
            raise ValueError(
                f"Need at least {self.config.min_algorithms} algorithms, "
                f"got {len(algorithm_results)}"
            )

        # Get reference shape
        ref_shape = algorithm_results[0].flood_extent.shape
        for result in algorithm_results[1:]:
            if result.flood_extent.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {result.algorithm_id} has shape "
                    f"{result.flood_extent.shape}, expected {ref_shape}"
                )

        # Create valid mask if not provided
        if valid_mask is None:
            valid_mask = np.ones(ref_shape, dtype=bool)

        # Calculate algorithm weights
        weights = self._calculate_weights(algorithm_results, conditions)

        # Apply fusion method
        if self.config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            probability = self._fuse_weighted_average(algorithm_results, weights)
        elif self.config.fusion_method == FusionMethod.VOTING:
            probability = self._fuse_voting(algorithm_results, weights)
        elif self.config.fusion_method == FusionMethod.BAYESIAN:
            probability = self._fuse_bayesian(algorithm_results, weights)
        elif self.config.fusion_method == FusionMethod.RELIABILITY_WEIGHTED:
            probability = self._fuse_reliability_weighted(algorithm_results, weights)
        else:
            # Default to weighted average
            probability = self._fuse_weighted_average(algorithm_results, weights)

        # Apply threshold for binary classification
        flood_extent = probability >= self.config.confidence_threshold

        # Apply valid mask
        flood_extent = flood_extent & valid_mask
        probability = np.where(valid_mask, probability, 0.0)

        # Calculate agreement map
        agreement_map = None
        if self.config.generate_agreement_map:
            agreement_map = self._calculate_agreement(algorithm_results, flood_extent)

        # Calculate uncertainty map
        uncertainty_map = None
        if self.config.generate_uncertainty_map:
            uncertainty_map = self._calculate_uncertainty(algorithm_results, probability)

        # Calculate confidence (based on agreement and probability certainty)
        confidence = self._calculate_confidence(probability, agreement_map)

        # Calculate statistics
        statistics = self._calculate_statistics(
            flood_extent, probability, confidence,
            agreement_map, valid_mask, pixel_size_m,
            len(algorithm_results)
        )

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "fusion_method": self.config.fusion_method.value,
                "disagreement_handling": self.config.disagreement_handling.value,
                "voting_threshold": self.config.voting_threshold,
                "confidence_threshold": self.config.confidence_threshold,
                "pixel_size_m": pixel_size_m
            },
            "execution": {
                "algorithm_count": len(algorithm_results),
                "algorithms": [r.algorithm_id for r in algorithm_results],
                "conditions": conditions
            }
        }

        logger.info(
            f"Fusion complete: {statistics['flood_area_ha']:.2f} ha flood extent "
            f"(mean agreement: {statistics['mean_agreement']:.2f})"
        )

        return EnsembleFusionResult(
            flood_extent=flood_extent,
            flood_probability=probability,
            confidence_raster=confidence,
            agreement_map=agreement_map,
            uncertainty_map=uncertainty_map,
            per_algorithm_weights={
                r.algorithm_id: weights[i] for i, r in enumerate(algorithm_results)
            },
            metadata=metadata,
            statistics=statistics
        )

    def _calculate_weights(
        self,
        results: List[AlgorithmResult],
        conditions: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate weights for each algorithm.

        Args:
            results: Algorithm results
            conditions: Environmental conditions

        Returns:
            Array of weights, shape (n_algorithms,)
        """
        weights = np.zeros(len(results))

        for i, result in enumerate(results):
            alg_id = result.algorithm_id
            weight_config = self._weights.get(
                alg_id,
                AlgorithmWeight(algorithm_id=alg_id, base_weight=1.0)
            )

            # Start with base weight
            w = weight_config.base_weight

            # Adjust for conditions
            if conditions:
                for cond_name, cond_value in conditions.items():
                    if cond_name in weight_config.reliability_by_condition:
                        if isinstance(cond_value, bool):
                            key = cond_name if cond_value else f"no_{cond_name}"
                        else:
                            key = str(cond_value)
                        if key in weight_config.reliability_by_condition:
                            w *= weight_config.reliability_by_condition[key]

            weights[i] = w

        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Equal weights if all zeros
            weights = np.ones(len(results)) / len(results)

        logger.debug(f"Algorithm weights: {dict(zip([r.algorithm_id for r in results], weights))}")
        return weights

    def _fuse_weighted_average(
        self,
        results: List[AlgorithmResult],
        weights: np.ndarray
    ) -> np.ndarray:
        """Fuse using weighted average of confidences."""
        shape = results[0].confidence.shape
        probability = np.zeros(shape, dtype=np.float64)

        for i, result in enumerate(results):
            # Use confidence where flood detected, complement where not
            prob = np.where(result.flood_extent, result.confidence, 1 - result.confidence)
            probability += weights[i] * prob

        return probability.astype(np.float32)

    def _fuse_voting(
        self,
        results: List[AlgorithmResult],
        weights: np.ndarray
    ) -> np.ndarray:
        """Fuse using weighted voting."""
        shape = results[0].flood_extent.shape
        votes = np.zeros(shape, dtype=np.float64)

        for i, result in enumerate(results):
            votes += weights[i] * result.flood_extent.astype(float)

        # Normalize to [0, 1]
        # votes is already normalized since weights sum to 1
        return votes.astype(np.float32)

    def _fuse_bayesian(
        self,
        results: List[AlgorithmResult],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Fuse using Bayesian combination.

        Assumes conditional independence of algorithms given true flood state.
        P(flood|d1,d2,...) ∝ P(flood) * Π P(di|flood)
        """
        shape = results[0].confidence.shape

        # Prior probability of flooding (uniform prior)
        prior = 0.5

        # Log-odds ratio combination
        log_odds = np.log(prior / (1 - prior)) * np.ones(shape)

        for i, result in enumerate(results):
            # Avoid log(0) by clipping confidence
            conf = np.clip(result.confidence, 1e-6, 1 - 1e-6)

            # Update log-odds based on detection
            detected = result.flood_extent.astype(float)
            not_detected = 1 - detected

            # Weight the contribution
            contribution = weights[i] * (
                detected * np.log(conf / (1 - conf)) +
                not_detected * np.log((1 - conf) / conf)
            )
            log_odds += contribution

        # Convert back to probability
        probability = 1 / (1 + np.exp(-log_odds))
        probability = np.clip(probability, 0, 1)

        return probability.astype(np.float32)

    def _fuse_reliability_weighted(
        self,
        results: List[AlgorithmResult],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Fuse using spatially-varying reliability weights.

        Adjusts weights based on local agreement between algorithms.
        """
        shape = results[0].confidence.shape

        # Calculate local reliability for each algorithm
        reliability = np.zeros((len(results),) + shape, dtype=np.float64)

        if self.config.use_spatial_reliability:
            reliability = self._calculate_spatial_reliability(results)
        else:
            for i in range(len(results)):
                reliability[i] = 1.0

        # Combine base weights with spatial reliability
        spatial_weights = np.zeros((len(results),) + shape, dtype=np.float64)
        for i in range(len(results)):
            spatial_weights[i] = weights[i] * reliability[i]

        # Normalize weights at each pixel
        weight_sum = np.sum(spatial_weights, axis=0)
        weight_sum = np.maximum(weight_sum, 1e-10)  # Avoid division by zero
        spatial_weights = spatial_weights / weight_sum[np.newaxis, :, :]

        # Weighted combination
        probability = np.zeros(shape, dtype=np.float64)
        for i, result in enumerate(results):
            prob = np.where(result.flood_extent, result.confidence, 1 - result.confidence)
            probability += spatial_weights[i] * prob

        return probability.astype(np.float32)

    def _calculate_spatial_reliability(
        self,
        results: List[AlgorithmResult]
    ) -> np.ndarray:
        """
        Calculate spatially-varying reliability for each algorithm.

        Reliability is higher where an algorithm agrees with others.
        """
        from scipy import ndimage

        n_algs = len(results)
        shape = results[0].flood_extent.shape
        reliability = np.ones((n_algs,) + shape, dtype=np.float64)

        # Stack all flood extents
        extents = np.stack([r.flood_extent.astype(float) for r in results], axis=0)

        # Mean prediction (consensus)
        consensus = np.mean(extents, axis=0)

        # Window for local agreement
        window_size = self.config.reliability_window_size
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)

        for i in range(n_algs):
            # Agreement with consensus
            agreement = 1 - np.abs(extents[i] - consensus)

            # Smooth to get local reliability
            local_reliability = ndimage.convolve(agreement, kernel, mode='reflect')

            reliability[i] = local_reliability

        return reliability

    def _calculate_agreement(
        self,
        results: List[AlgorithmResult],
        fused_extent: np.ndarray
    ) -> np.ndarray:
        """
        Calculate per-pixel agreement between algorithms.

        Returns the fraction of algorithms that agree with the fused result.
        """
        n_algs = len(results)
        agreement = np.zeros(results[0].flood_extent.shape, dtype=np.float32)

        for result in results:
            agreement += (result.flood_extent == fused_extent).astype(float)

        return agreement / n_algs

    def _calculate_uncertainty(
        self,
        results: List[AlgorithmResult],
        probability: np.ndarray
    ) -> np.ndarray:
        """
        Calculate uncertainty based on algorithm disagreement.

        Higher variance in confidence = higher uncertainty.
        """
        if len(results) < 2:
            return np.zeros_like(probability)

        # Stack confidences
        confidences = np.stack([
            np.where(r.flood_extent, r.confidence, 1 - r.confidence)
            for r in results
        ], axis=0)

        # Variance across algorithms
        variance = np.var(confidences, axis=0)

        # Scale to [0, 1] - max variance is 0.25 (for binary between 0 and 1)
        uncertainty = np.clip(variance / 0.25, 0, 1)

        return uncertainty.astype(np.float32)

    def _calculate_confidence(
        self,
        probability: np.ndarray,
        agreement_map: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate confidence in the fusion result.

        Confidence is based on:
        1. Distance from probability threshold (clearer decisions = higher confidence)
        2. Algorithm agreement (more agreement = higher confidence)
        """
        # Distance from threshold
        threshold = self.config.confidence_threshold
        distance = np.abs(probability - threshold)
        max_distance = max(threshold, 1 - threshold)
        if max_distance < 1e-10:
            # Edge case: threshold is exactly 0.0 or 1.0
            max_distance = 1.0
        prob_confidence = distance / max_distance

        if agreement_map is not None:
            # Combine probability and agreement confidence
            confidence = (prob_confidence + agreement_map) / 2
        else:
            confidence = prob_confidence

        return np.clip(confidence, 0, 1).astype(np.float32)

    def _calculate_statistics(
        self,
        flood_extent: np.ndarray,
        probability: np.ndarray,
        confidence: np.ndarray,
        agreement_map: Optional[np.ndarray],
        valid_mask: np.ndarray,
        pixel_size_m: float,
        n_algorithms: int
    ) -> Dict[str, float]:
        """Calculate summary statistics."""
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0
        flood_pixels = np.sum(flood_extent)
        valid_pixels = np.sum(valid_mask)

        stats = {
            "total_pixels": int(flood_extent.size),
            "valid_pixels": int(valid_pixels),
            "flood_pixels": int(flood_pixels),
            "flood_area_ha": float(flood_pixels * pixel_area_ha),
            "flood_percent": float(100.0 * flood_pixels / valid_pixels) if valid_pixels > 0 else 0.0,
            "mean_probability": float(np.mean(probability[valid_mask])) if valid_pixels > 0 else 0.0,
            "mean_confidence": float(np.mean(confidence[valid_mask])) if valid_pixels > 0 else 0.0,
            "algorithm_count": n_algorithms
        }

        if agreement_map is not None:
            stats["mean_agreement"] = float(np.mean(agreement_map[valid_mask])) if valid_pixels > 0 else 0.0
            stats["high_agreement_percent"] = float(
                100.0 * np.sum((agreement_map >= 0.8) & valid_mask) / valid_pixels
            ) if valid_pixels > 0 else 0.0
            stats["full_agreement_percent"] = float(
                100.0 * np.sum((agreement_map >= 1.0) & valid_mask) / valid_pixels
            ) if valid_pixels > 0 else 0.0
        else:
            stats["mean_agreement"] = 1.0

        return stats

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return EnsembleFusionAlgorithm.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'EnsembleFusionAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        # Handle enums
        if "fusion_method" in params and isinstance(params["fusion_method"], str):
            params["fusion_method"] = FusionMethod(params["fusion_method"])
        if "disagreement_handling" in params and isinstance(params["disagreement_handling"], str):
            params["disagreement_handling"] = DisagreementHandling(params["disagreement_handling"])

        # Handle algorithm weights
        if "algorithm_weights" in params:
            weights = []
            for w in params["algorithm_weights"]:
                if isinstance(w, dict):
                    weights.append(AlgorithmWeight(**w))
                else:
                    weights.append(w)
            params["algorithm_weights"] = weights

        config = EnsembleFusionConfig(**params)
        return EnsembleFusionAlgorithm(config=config)


def create_ensemble_from_algorithms(
    algorithm_classes: List[type],
    imagery_data: Dict[str, np.ndarray],
    fusion_config: Optional[EnsembleFusionConfig] = None,
    **kwargs
) -> EnsembleFusionResult:
    """
    Convenience function to run multiple algorithms and fuse results.

    Args:
        algorithm_classes: List of algorithm classes to run
        imagery_data: Dictionary of input data
        fusion_config: Configuration for fusion
        **kwargs: Additional arguments passed to execute()

    Returns:
        EnsembleFusionResult
    """
    results = []

    for alg_class in algorithm_classes:
        try:
            alg = alg_class()
            result = alg.execute(**imagery_data)

            # Convert to AlgorithmResult
            alg_result = AlgorithmResult(
                algorithm_id=alg.METADATA["id"],
                flood_extent=result.flood_extent,
                confidence=result.confidence_raster,
                metadata=result.metadata
            )
            results.append(alg_result)

        except Exception as e:
            logger.warning(f"Algorithm {alg_class.__name__} failed: {e}")
            continue

    if len(results) < 2:
        raise ValueError(f"Need at least 2 successful algorithms, got {len(results)}")

    fusion = EnsembleFusionAlgorithm(config=fusion_config)
    return fusion.execute(results, **kwargs)
