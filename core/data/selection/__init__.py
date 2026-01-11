"""
Data selection components.

Intelligent selection of sensors and data sources based on:
- Atmospheric conditions
- Sensor capabilities
- Observable requirements
- Quality thresholds
"""

from .atmospheric import (
    AtmosphericCondition,
    AtmosphericAssessment,
    AtmosphericEvaluator,
    assess_cloud_cover,
    recommend_sensors_for_event,
)
from .strategy import (
    SensorType,
    Observable,
    ConfidenceLevel,
    SensorCapability,
    SensorSelection,
    SensorSelectionStrategy,
    get_observables_for_event_class,
)

__all__ = [
    # Atmospheric assessment
    "AtmosphericCondition",
    "AtmosphericAssessment",
    "AtmosphericEvaluator",
    "assess_cloud_cover",
    "recommend_sensors_for_event",
    # Sensor selection strategy
    "SensorType",
    "Observable",
    "ConfidenceLevel",
    "SensorCapability",
    "SensorSelection",
    "SensorSelectionStrategy",
    "get_observables_for_event_class",
]
