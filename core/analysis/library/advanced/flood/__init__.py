"""
Advanced Flood Detection Algorithms

This module contains advanced algorithms for flood extent detection
including deep learning and ensemble methods.

Algorithms:
    - unet_segmentation: U-Net semantic segmentation for flood detection
    - ensemble_fusion: Multi-algorithm ensemble fusion
"""

from .unet_segmentation import (
    UNetSegmentationAlgorithm,
    UNetConfig,
    UNetResult,
    ModelBackend
)

from .ensemble_fusion import (
    EnsembleFusionAlgorithm,
    EnsembleFusionConfig,
    EnsembleFusionResult,
    AlgorithmResult,
    AlgorithmWeight,
    FusionMethod,
    DisagreementHandling,
    create_ensemble_from_algorithms
)

__all__ = [
    # U-Net Segmentation
    "UNetSegmentationAlgorithm",
    "UNetConfig",
    "UNetResult",
    "ModelBackend",
    # Ensemble Fusion
    "EnsembleFusionAlgorithm",
    "EnsembleFusionConfig",
    "EnsembleFusionResult",
    "AlgorithmResult",
    "AlgorithmWeight",
    "FusionMethod",
    "DisagreementHandling",
    "create_ensemble_from_algorithms",
]

# Algorithm registry for easy lookup
ADVANCED_FLOOD_ALGORITHMS = {
    "flood.advanced.unet_segmentation": UNetSegmentationAlgorithm,
    "flood.advanced.ensemble_fusion": EnsembleFusionAlgorithm,
}


def get_algorithm(algorithm_id: str):
    """
    Get algorithm class by ID.

    Args:
        algorithm_id: Algorithm identifier (e.g., "flood.advanced.unet_segmentation")

    Returns:
        Algorithm class

    Raises:
        KeyError: If algorithm_id is not found
    """
    if algorithm_id not in ADVANCED_FLOOD_ALGORITHMS:
        available = ", ".join(ADVANCED_FLOOD_ALGORITHMS.keys())
        raise KeyError(f"Unknown algorithm: {algorithm_id}. Available: {available}")

    return ADVANCED_FLOOD_ALGORITHMS[algorithm_id]


def list_algorithms():
    """
    List all available advanced flood algorithms.

    Returns:
        List of (algorithm_id, algorithm_class) tuples
    """
    return list(ADVANCED_FLOOD_ALGORITHMS.items())
