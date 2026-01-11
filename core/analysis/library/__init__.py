"""
Algorithm library for event intelligence analysis.

Contains baseline, advanced, and experimental algorithms for:
- Flood detection and mapping
- Wildfire detection and burn area mapping
- Storm damage assessment
- And other hazard types

Registry provides centralized algorithm discovery and metadata management.

Modules:
    - baseline: Well-validated, interpretable baseline algorithms
    - advanced: ML-driven and ensemble algorithms
    - registry: Algorithm metadata and discovery
"""

from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType,
    ResourceRequirements,
    ValidationMetrics,
    get_global_registry,
    load_default_algorithms
)

# Import algorithm modules for easy access
from . import baseline
from . import advanced

__all__ = [
    'AlgorithmRegistry',
    'AlgorithmMetadata',
    'AlgorithmCategory',
    'DataType',
    'ResourceRequirements',
    'ValidationMetrics',
    'get_global_registry',
    'load_default_algorithms',
    'baseline',
    'advanced',
]
