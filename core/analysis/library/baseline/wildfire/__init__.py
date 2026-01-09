"""
Baseline Wildfire Detection and Mapping Algorithms

This module contains well-validated, interpretable baseline algorithms
for wildfire detection, burn severity mapping, and burned area classification.

Algorithms:
    - nbr_differenced: Differenced Normalized Burn Ratio burn severity mapping
    - thermal_anomaly: Thermal infrared active fire detection
    - ba_classifier: Machine learning burned area classification
"""

from .nbr_differenced import (
    DifferencedNBRAlgorithm,
    DifferencedNBRConfig,
    DifferencedNBRResult
)

from .thermal_anomaly import (
    ThermalAnomalyAlgorithm,
    ThermalAnomalyConfig,
    ThermalAnomalyResult
)

from .ba_classifier import (
    BurnedAreaClassifierAlgorithm,
    BurnedAreaClassifierConfig,
    BurnedAreaClassifierResult
)

__all__ = [
    # Differenced NBR
    "DifferencedNBRAlgorithm",
    "DifferencedNBRConfig",
    "DifferencedNBRResult",
    # Thermal Anomaly
    "ThermalAnomalyAlgorithm",
    "ThermalAnomalyConfig",
    "ThermalAnomalyResult",
    # Burned Area Classifier
    "BurnedAreaClassifierAlgorithm",
    "BurnedAreaClassifierConfig",
    "BurnedAreaClassifierResult",
]

# Algorithm registry for easy lookup
WILDFIRE_ALGORITHMS = {
    "wildfire.baseline.nbr_differenced": DifferencedNBRAlgorithm,
    "wildfire.baseline.thermal_anomaly": ThermalAnomalyAlgorithm,
    "wildfire.baseline.ba_classifier": BurnedAreaClassifierAlgorithm,
}


def get_algorithm(algorithm_id: str):
    """
    Get algorithm class by ID.

    Args:
        algorithm_id: Algorithm identifier (e.g., "wildfire.baseline.nbr_differenced")

    Returns:
        Algorithm class

    Raises:
        KeyError: If algorithm_id is not found
    """
    if algorithm_id not in WILDFIRE_ALGORITHMS:
        available = ", ".join(WILDFIRE_ALGORITHMS.keys())
        raise KeyError(f"Unknown algorithm: {algorithm_id}. Available: {available}")

    return WILDFIRE_ALGORITHMS[algorithm_id]


def list_algorithms():
    """
    List all available wildfire algorithms.

    Returns:
        List of (algorithm_id, algorithm_class) tuples
    """
    return list(WILDFIRE_ALGORITHMS.items())
