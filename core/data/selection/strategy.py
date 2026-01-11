"""
Sensor selection strategy.

Determines optimal sensor combinations for observing specific phenomena,
with support for degraded modes when ideal sensors are unavailable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Sensor modality types."""
    OPTICAL = "optical"
    SAR = "sar"
    THERMAL = "thermal"
    DEM = "dem"
    WEATHER = "weather"
    ANCILLARY = "ancillary"


class Observable(Enum):
    """Physical phenomena that can be observed."""
    WATER_EXTENT = "water_extent"
    FLOOD_DEPTH = "flood_depth"
    BURN_SEVERITY = "burn_severity"
    ACTIVE_FIRE = "active_fire"
    VEGETATION_DAMAGE = "vegetation_damage"
    STRUCTURAL_DAMAGE = "structural_damage"
    WIND_DAMAGE = "wind_damage"
    CLOUD_COVER = "cloud_cover"
    PRECIPITATION = "precipitation"
    TERRAIN_HEIGHT = "terrain_height"
    LAND_COVER = "land_cover"


class ConfidenceLevel(Enum):
    """Confidence levels for observations."""
    HIGH = "high"          # Ideal sensor, good conditions
    MEDIUM = "medium"      # Acceptable sensor or suboptimal conditions
    LOW = "low"            # Degraded sensor or poor conditions
    UNRELIABLE = "unreliable"  # Should not be used


@dataclass
class SensorCapability:
    """
    Defines sensor capability for observing a phenomenon.

    Attributes:
        sensor_type: Type of sensor
        observable: What it can observe
        confidence: Base confidence level under ideal conditions
        requires: Other observables required for processing
        degrades_with: Conditions that reduce reliability
        alternatives: Alternative sensors if this is unavailable
    """
    sensor_type: SensorType
    observable: Observable
    confidence: ConfidenceLevel
    requires: List[Observable] = field(default_factory=list)
    degrades_with: Dict[str, str] = field(default_factory=dict)
    alternatives: List[SensorType] = field(default_factory=list)

    def evaluate_confidence(
        self,
        conditions: Dict[str, Any]
    ) -> ConfidenceLevel:
        """
        Evaluate actual confidence given atmospheric/scene conditions.

        Args:
            conditions: Current conditions (cloud_cover, precipitation, etc.)

        Returns:
            Adjusted confidence level
        """
        current_confidence = self.confidence

        # Check degradation factors
        for factor, impact in self.degrades_with.items():
            if factor in conditions:
                value = conditions[factor]

                # Cloud cover degradation
                if factor == "cloud_cover" and isinstance(value, (int, float)):
                    if value > 80:
                        return ConfidenceLevel.UNRELIABLE
                    elif value > 60:
                        # 60-80%: downgrade to LOW
                        if current_confidence == ConfidenceLevel.HIGH:
                            current_confidence = ConfidenceLevel.LOW
                        elif current_confidence == ConfidenceLevel.MEDIUM:
                            current_confidence = ConfidenceLevel.UNRELIABLE
                    elif value > 30:
                        # 30-60%: downgrade by 1 step
                        current_confidence = self._downgrade_confidence(current_confidence, 1)

                # Precipitation degradation
                elif factor == "precipitation" and value:
                    current_confidence = self._downgrade_confidence(current_confidence, 1)

                # Darkness for optical
                elif factor == "darkness" and value:
                    if self.sensor_type == SensorType.OPTICAL:
                        return ConfidenceLevel.UNRELIABLE

        return current_confidence

    @staticmethod
    def _downgrade_confidence(level: ConfidenceLevel, steps: float) -> ConfidenceLevel:
        """Downgrade confidence by specified steps."""
        levels = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM,
                  ConfidenceLevel.LOW, ConfidenceLevel.UNRELIABLE]

        current_idx = levels.index(level)
        new_idx = min(len(levels) - 1, current_idx + int(steps))

        return levels[new_idx]


@dataclass
class SensorSelection:
    """
    Selected sensor combination for observing phenomena.

    Attributes:
        observable: What is being observed
        primary_sensor: Main sensor for this observable
        supporting_sensors: Additional sensors for context/validation
        confidence: Overall confidence in observation
        degraded_mode: Whether operating in degraded mode
        rationale: Explanation of selection decision
        alternatives_rejected: Alternative sensors considered but rejected
    """
    observable: Observable
    primary_sensor: SensorType
    supporting_sensors: List[SensorType] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    degraded_mode: bool = False
    rationale: str = ""
    alternatives_rejected: Dict[SensorType, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "observable": self.observable.value,
            "primary_sensor": self.primary_sensor.value,
            "supporting_sensors": [s.value for s in self.supporting_sensors],
            "confidence": self.confidence.value,
            "degraded_mode": self.degraded_mode,
            "rationale": self.rationale,
            "alternatives_rejected": {
                k.value: v for k, v in self.alternatives_rejected.items()
            }
        }


class SensorSelectionStrategy:
    """
    Intelligent sensor selection with degraded mode handling.

    Determines optimal sensor combinations for observing specific phenomena,
    considering:
    - Sensor availability
    - Atmospheric conditions
    - Observable requirements
    - Degraded mode thresholds
    """

    def __init__(self):
        """Initialize sensor selection strategy."""
        self.capabilities = self._initialize_capabilities()

    def _initialize_capabilities(self) -> Dict[Observable, List[SensorCapability]]:
        """
        Initialize sensor capability matrix.

        Returns:
            Mapping from observable to list of capable sensors (best first)
        """
        return {
            # Water extent observation
            Observable.WATER_EXTENT: [
                SensorCapability(
                    sensor_type=SensorType.SAR,
                    observable=Observable.WATER_EXTENT,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={"wind": "high_backscatter"},
                    alternatives=[SensorType.OPTICAL]
                ),
                SensorCapability(
                    sensor_type=SensorType.OPTICAL,
                    observable=Observable.WATER_EXTENT,
                    confidence=ConfidenceLevel.MEDIUM,
                    requires=[],
                    degrades_with={"cloud_cover": "obscures", "darkness": "unusable"},
                    alternatives=[SensorType.SAR]
                ),
            ],

            # Flood depth requires DEM + water extent
            Observable.FLOOD_DEPTH: [
                SensorCapability(
                    sensor_type=SensorType.DEM,
                    observable=Observable.FLOOD_DEPTH,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[Observable.WATER_EXTENT],
                    degrades_with={},
                    alternatives=[]
                ),
            ],

            # Burn severity
            Observable.BURN_SEVERITY: [
                SensorCapability(
                    sensor_type=SensorType.OPTICAL,
                    observable=Observable.BURN_SEVERITY,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={"cloud_cover": "obscures", "smoke": "obscures"},
                    alternatives=[SensorType.SAR]
                ),
                SensorCapability(
                    sensor_type=SensorType.SAR,
                    observable=Observable.BURN_SEVERITY,
                    confidence=ConfidenceLevel.MEDIUM,
                    requires=[],
                    degrades_with={},
                    alternatives=[SensorType.OPTICAL]
                ),
            ],

            # Active fire detection
            Observable.ACTIVE_FIRE: [
                SensorCapability(
                    sensor_type=SensorType.THERMAL,
                    observable=Observable.ACTIVE_FIRE,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={"cloud_cover": "obscures"},
                    alternatives=[SensorType.OPTICAL]
                ),
                SensorCapability(
                    sensor_type=SensorType.OPTICAL,
                    observable=Observable.ACTIVE_FIRE,
                    confidence=ConfidenceLevel.MEDIUM,
                    requires=[],
                    degrades_with={"cloud_cover": "obscures", "darkness": "helps_detect"},
                    alternatives=[SensorType.THERMAL]
                ),
            ],

            # Vegetation damage
            Observable.VEGETATION_DAMAGE: [
                SensorCapability(
                    sensor_type=SensorType.OPTICAL,
                    observable=Observable.VEGETATION_DAMAGE,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={"cloud_cover": "obscures"},
                    alternatives=[SensorType.SAR]
                ),
                SensorCapability(
                    sensor_type=SensorType.SAR,
                    observable=Observable.VEGETATION_DAMAGE,
                    confidence=ConfidenceLevel.MEDIUM,
                    requires=[],
                    degrades_with={},
                    alternatives=[SensorType.OPTICAL]
                ),
            ],

            # Structural damage
            Observable.STRUCTURAL_DAMAGE: [
                SensorCapability(
                    sensor_type=SensorType.SAR,
                    observable=Observable.STRUCTURAL_DAMAGE,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[Observable.LAND_COVER],
                    degrades_with={},
                    alternatives=[SensorType.OPTICAL]
                ),
                SensorCapability(
                    sensor_type=SensorType.OPTICAL,
                    observable=Observable.STRUCTURAL_DAMAGE,
                    confidence=ConfidenceLevel.MEDIUM,
                    requires=[Observable.LAND_COVER],
                    degrades_with={"cloud_cover": "obscures"},
                    alternatives=[SensorType.SAR]
                ),
            ],

            # Terrain height (DEM)
            Observable.TERRAIN_HEIGHT: [
                SensorCapability(
                    sensor_type=SensorType.DEM,
                    observable=Observable.TERRAIN_HEIGHT,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={},
                    alternatives=[]
                ),
            ],

            # Land cover (ancillary)
            Observable.LAND_COVER: [
                SensorCapability(
                    sensor_type=SensorType.ANCILLARY,
                    observable=Observable.LAND_COVER,
                    confidence=ConfidenceLevel.HIGH,
                    requires=[],
                    degrades_with={},
                    alternatives=[SensorType.OPTICAL]
                ),
            ],
        }

    def select_sensors(
        self,
        observables: List[Observable],
        available_sensors: Set[SensorType],
        conditions: Optional[Dict[str, Any]] = None,
        allow_degraded: bool = True
    ) -> Dict[Observable, SensorSelection]:
        """
        Select optimal sensor combination for observables.

        Args:
            observables: List of phenomena to observe
            available_sensors: Set of available sensor types
            conditions: Current atmospheric/scene conditions
            allow_degraded: Whether to allow degraded mode selections

        Returns:
            Mapping from observable to sensor selection
        """
        conditions = conditions or {}
        selections: Dict[Observable, SensorSelection] = {}

        # Track which observables are satisfied
        satisfied: Set[Observable] = set()

        # Multiple passes to resolve dependencies
        max_passes = 5
        for pass_num in range(max_passes):
            made_progress = False

            for observable in observables:
                if observable in satisfied:
                    continue

                # Get capable sensors for this observable
                capabilities = self.capabilities.get(observable, [])
                if not capabilities:
                    logger.warning(f"No sensors registered for observable: {observable}")
                    continue

                # Check if requirements are satisfied
                requirements_met = True
                for cap in capabilities:
                    if cap.requires:
                        if not all(req in satisfied for req in cap.requires):
                            requirements_met = False
                            break

                if not requirements_met:
                    continue  # Try again in next pass

                # Find best available sensor
                selection = self._select_best_sensor(
                    observable,
                    capabilities,
                    available_sensors,
                    conditions,
                    allow_degraded
                )

                if selection:
                    selections[observable] = selection
                    satisfied.add(observable)
                    made_progress = True

                    logger.info(
                        f"Selected {selection.primary_sensor.value} for "
                        f"{observable.value} (confidence: {selection.confidence.value})"
                    )

            if not made_progress:
                break

        # Check for unsatisfied observables
        unsatisfied = set(observables) - satisfied
        if unsatisfied:
            logger.warning(
                f"Could not satisfy observables: "
                f"{[o.value for o in unsatisfied]}"
            )

        return selections

    def _select_best_sensor(
        self,
        observable: Observable,
        capabilities: List[SensorCapability],
        available_sensors: Set[SensorType],
        conditions: Dict[str, Any],
        allow_degraded: bool
    ) -> Optional[SensorSelection]:
        """
        Select best sensor from capabilities.

        Args:
            observable: What to observe
            capabilities: Sensor capabilities (ordered by preference)
            available_sensors: Available sensor types
            conditions: Current conditions
            allow_degraded: Whether to allow degraded mode

        Returns:
            SensorSelection if viable sensor found, None otherwise
        """
        best_selection: Optional[SensorSelection] = None
        alternatives_rejected: Dict[SensorType, str] = {}

        for capability in capabilities:
            sensor = capability.sensor_type

            # Check availability
            if sensor not in available_sensors:
                alternatives_rejected[sensor] = "not_available"
                continue

            # Evaluate confidence under current conditions
            confidence = capability.evaluate_confidence(conditions)

            # Check if unreliable
            if confidence == ConfidenceLevel.UNRELIABLE:
                alternatives_rejected[sensor] = "unreliable_conditions"
                continue

            # Check degraded mode threshold
            # Only LOW confidence is considered degraded mode
            # MEDIUM is acceptable, HIGH is optimal
            degraded_mode = confidence == ConfidenceLevel.LOW
            if degraded_mode and not allow_degraded:
                alternatives_rejected[sensor] = "degraded_not_allowed"
                continue

            # Found viable sensor
            rationale = self._build_rationale(
                sensor, observable, confidence, degraded_mode, conditions
            )

            # Add supporting sensors if available and viable
            supporting = []
            for alt_sensor in capability.alternatives:
                if alt_sensor in available_sensors and alt_sensor != sensor:
                    # Check if alternative sensor is viable under current conditions
                    alt_cap = self._find_capability(observable, alt_sensor)
                    if alt_cap:
                        alt_confidence = alt_cap.evaluate_confidence(conditions)
                        if alt_confidence != ConfidenceLevel.UNRELIABLE:
                            supporting.append(alt_sensor)

            best_selection = SensorSelection(
                observable=observable,
                primary_sensor=sensor,
                supporting_sensors=supporting,
                confidence=confidence,
                degraded_mode=degraded_mode,
                rationale=rationale,
                alternatives_rejected=alternatives_rejected.copy()
            )

            # Use first viable sensor (capabilities are pre-sorted by preference)
            break

        return best_selection

    def _find_capability(
        self,
        observable: Observable,
        sensor_type: SensorType
    ) -> Optional[SensorCapability]:
        """
        Find capability for specific observable and sensor type.

        Args:
            observable: Observable to look for
            sensor_type: Sensor type to find

        Returns:
            SensorCapability if found, None otherwise
        """
        capabilities = self.capabilities.get(observable, [])
        for cap in capabilities:
            if cap.sensor_type == sensor_type:
                return cap
        return None

    def _build_rationale(
        self,
        sensor: SensorType,
        observable: Observable,
        confidence: ConfidenceLevel,
        degraded_mode: bool,
        conditions: Dict[str, Any]
    ) -> str:
        """Build human-readable rationale for selection."""
        parts = [
            f"{sensor.value} selected for {observable.value}",
            f"confidence={confidence.value}"
        ]

        if degraded_mode:
            parts.append("(degraded mode)")

        # Add relevant condition info
        if "cloud_cover" in conditions:
            parts.append(f"cloud_cover={conditions['cloud_cover']}%")

        return "; ".join(parts)

    def get_required_data_types(
        self,
        event_class: str,
        observables: List[Observable]
    ) -> Set[str]:
        """
        Get required data types for event class and observables.

        Args:
            event_class: Event classification (e.g., "flood.coastal")
            observables: List of observables needed

        Returns:
            Set of required data type names
        """
        required_types: Set[str] = set()

        for observable in observables:
            capabilities = self.capabilities.get(observable, [])
            if capabilities:
                # Add primary sensor type
                primary_cap = capabilities[0]
                required_types.add(primary_cap.sensor_type.value)

                # Add required observables' sensor types
                for req in primary_cap.requires:
                    req_caps = self.capabilities.get(req, [])
                    if req_caps:
                        required_types.add(req_caps[0].sensor_type.value)

        return required_types


def get_observables_for_event_class(event_class: str) -> List[Observable]:
    """
    Map event class to required observables.

    Args:
        event_class: Event classification (e.g., "flood.coastal")

    Returns:
        List of observables needed for this event type
    """
    # Flood events
    if event_class.startswith("flood"):
        observables = [Observable.WATER_EXTENT, Observable.TERRAIN_HEIGHT]

        # Add flood depth for riverine/coastal
        if any(x in event_class for x in ["riverine", "coastal"]):
            observables.append(Observable.FLOOD_DEPTH)

        return observables

    # Wildfire events
    elif event_class.startswith("wildfire"):
        observables = [Observable.BURN_SEVERITY]

        # Add active fire detection for ongoing fires
        if "active" in event_class or "ongoing" in event_class:
            observables.append(Observable.ACTIVE_FIRE)

        return observables

    # Storm events
    elif event_class.startswith("storm"):
        observables = [Observable.VEGETATION_DAMAGE, Observable.LAND_COVER]

        # Add structural damage for urban areas
        if any(x in event_class for x in ["urban", "tornado", "hurricane"]):
            observables.append(Observable.STRUCTURAL_DAMAGE)

        return observables

    else:
        logger.warning(f"Unknown event class: {event_class}")
        return []
