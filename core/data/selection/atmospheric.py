"""
Atmospheric assessment for sensor selection.

Evaluates weather and atmospheric conditions to recommend optimal sensor types
and identify degraded observing conditions.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# Cloud cover thresholds (percentage)
CLOUD_COVER_EXCELLENT = 5.0
CLOUD_COVER_GOOD = 10.0
CLOUD_COVER_FAIR = 50.0
CLOUD_COVER_POOR = 80.0

# Default thresholds for sensor suitability
DEFAULT_OPTICAL_CLOUD_THRESHOLD = 20.0
DEFAULT_THERMAL_CLOUD_THRESHOLD = 50.0
DEFAULT_DEGRADED_MODE_THRESHOLD = 80.0

# Visibility threshold (km)
MIN_VISIBILITY_KM = 5.0


class SensorType(Enum):
    """Sensor types with different atmospheric sensitivities."""
    OPTICAL = "optical"
    SAR = "sar"
    THERMAL = "thermal"
    LIDAR = "lidar"


class AtmosphericCondition(Enum):
    """Atmospheric condition classifications."""
    EXCELLENT = "excellent"  # Cloud free, high visibility
    GOOD = "good"           # <10% clouds, good visibility
    FAIR = "fair"           # 10-50% clouds, moderate visibility
    POOR = "poor"           # 50-80% clouds, reduced visibility
    DEGRADED = "degraded"   # >80% clouds or severe weather


@dataclass
class AtmosphericAssessment:
    """
    Assessment of atmospheric conditions for sensor selection.

    Evaluates cloud cover, weather conditions, and makes sensor
    suitability recommendations.
    """

    # Input conditions
    cloud_cover_percent: Optional[float]
    precipitation: Optional[bool]
    severe_weather: Optional[bool]
    visibility_km: Optional[float]
    smoke_aerosols: Optional[bool]

    # Derived assessments
    condition: AtmosphericCondition
    optical_suitable: bool
    sar_suitable: bool
    thermal_suitable: bool
    confidence: float

    # Recommendations
    recommended_sensors: List[SensorType]
    degraded_mode: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cloud_cover_percent": self.cloud_cover_percent,
            "precipitation": self.precipitation,
            "severe_weather": self.severe_weather,
            "visibility_km": self.visibility_km,
            "smoke_aerosols": self.smoke_aerosols,
            "condition": self.condition.value,
            "optical_suitable": self.optical_suitable,
            "sar_suitable": self.sar_suitable,
            "thermal_suitable": self.thermal_suitable,
            "confidence": self.confidence,
            "recommended_sensors": [s.value for s in self.recommended_sensors],
            "degraded_mode": self.degraded_mode,
            "reason": self.reason
        }


class AtmosphericEvaluator:
    """
    Evaluates atmospheric conditions and makes sensor recommendations.

    Uses weather data, cloud cover, and other atmospheric indicators to
    determine which sensor types are suitable for the current conditions.
    """

    def __init__(
        self,
        cloud_cover_threshold_optical: float = DEFAULT_OPTICAL_CLOUD_THRESHOLD,
        cloud_cover_threshold_thermal: float = DEFAULT_THERMAL_CLOUD_THRESHOLD,
        degraded_mode_threshold: float = DEFAULT_DEGRADED_MODE_THRESHOLD
    ):
        """
        Initialize atmospheric evaluator.

        Args:
            cloud_cover_threshold_optical: Max cloud cover % for optical (default 20%)
            cloud_cover_threshold_thermal: Max cloud cover % for thermal (default 50%)
            degraded_mode_threshold: Cloud cover % to trigger degraded mode (default 80%)
        """
        self.cloud_cover_threshold_optical = cloud_cover_threshold_optical
        self.cloud_cover_threshold_thermal = cloud_cover_threshold_thermal
        self.degraded_mode_threshold = degraded_mode_threshold

    def assess(
        self,
        cloud_cover_percent: Optional[float] = None,
        precipitation: Optional[bool] = None,
        severe_weather: Optional[bool] = None,
        visibility_km: Optional[float] = None,
        smoke_aerosols: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AtmosphericAssessment:
        """
        Assess atmospheric conditions for sensor selection.

        Args:
            cloud_cover_percent: Cloud cover percentage (0-100)
            precipitation: Whether precipitation is occurring
            severe_weather: Whether severe weather is present
            visibility_km: Horizontal visibility in kilometers
            smoke_aerosols: Whether smoke or heavy aerosols present
            metadata: Additional atmospheric metadata

        Returns:
            AtmosphericAssessment with recommendations
        """
        # Determine overall condition
        condition = self._classify_condition(
            cloud_cover_percent,
            precipitation,
            severe_weather,
            visibility_km,
            smoke_aerosols
        )

        # Evaluate sensor suitability
        optical_suitable = self._evaluate_optical_suitability(
            cloud_cover_percent,
            smoke_aerosols,
            visibility_km
        )

        sar_suitable = self._evaluate_sar_suitability(
            severe_weather,
            precipitation
        )

        thermal_suitable = self._evaluate_thermal_suitability(
            cloud_cover_percent,
            severe_weather
        )

        # Determine recommended sensors
        recommended_sensors = self._recommend_sensors(
            optical_suitable,
            sar_suitable,
            thermal_suitable,
            condition
        )

        # Check for degraded mode
        degraded_mode = self._check_degraded_mode(
            cloud_cover_percent,
            severe_weather,
            recommended_sensors
        )

        # Generate explanation
        reason = self._generate_reason(
            condition,
            cloud_cover_percent,
            precipitation,
            severe_weather,
            smoke_aerosols,
            recommended_sensors
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(
            cloud_cover_percent,
            precipitation,
            severe_weather,
            visibility_km,
            smoke_aerosols
        )

        return AtmosphericAssessment(
            cloud_cover_percent=cloud_cover_percent,
            precipitation=precipitation,
            severe_weather=severe_weather,
            visibility_km=visibility_km,
            smoke_aerosols=smoke_aerosols,
            condition=condition,
            optical_suitable=optical_suitable,
            sar_suitable=sar_suitable,
            thermal_suitable=thermal_suitable,
            confidence=confidence,
            recommended_sensors=recommended_sensors,
            degraded_mode=degraded_mode,
            reason=reason
        )

    def assess_from_weather_data(
        self,
        weather_data: Dict[str, Any]
    ) -> AtmosphericAssessment:
        """
        Assess conditions from weather data dictionary.

        Args:
            weather_data: Dictionary containing weather parameters

        Returns:
            AtmosphericAssessment
        """
        return self.assess(
            cloud_cover_percent=weather_data.get("cloud_cover"),
            precipitation=weather_data.get("precipitation"),
            severe_weather=weather_data.get("severe_weather"),
            visibility_km=weather_data.get("visibility_km"),
            smoke_aerosols=weather_data.get("smoke_aerosols"),
            metadata=weather_data.get("metadata")
        )

    def _classify_condition(
        self,
        cloud_cover: Optional[float],
        precipitation: Optional[bool],
        severe_weather: Optional[bool],
        visibility: Optional[float],
        smoke: Optional[bool]
    ) -> AtmosphericCondition:
        """Classify overall atmospheric condition."""

        # Severe weather always results in degraded
        if severe_weather:
            return AtmosphericCondition.DEGRADED

        # Use cloud cover as primary indicator
        if cloud_cover is not None:
            if cloud_cover < CLOUD_COVER_EXCELLENT:
                return AtmosphericCondition.EXCELLENT
            elif cloud_cover < CLOUD_COVER_GOOD:
                return AtmosphericCondition.GOOD
            elif cloud_cover < CLOUD_COVER_FAIR:
                return AtmosphericCondition.FAIR
            elif cloud_cover < CLOUD_COVER_POOR:
                return AtmosphericCondition.POOR
            else:
                return AtmosphericCondition.DEGRADED

        # Fall back to precipitation/smoke if no cloud cover
        if precipitation or smoke:
            return AtmosphericCondition.POOR

        # Default to good if no negative indicators
        return AtmosphericCondition.GOOD

    def _evaluate_optical_suitability(
        self,
        cloud_cover: Optional[float],
        smoke: Optional[bool],
        visibility: Optional[float]
    ) -> bool:
        """Evaluate if optical sensors are suitable."""

        # Cloud cover check
        if cloud_cover is not None and cloud_cover > self.cloud_cover_threshold_optical:
            return False

        # Smoke/aerosol check
        if smoke:
            return False

        # Visibility check (if available)
        if visibility is not None and visibility < MIN_VISIBILITY_KM:
            return False

        return True

    def _evaluate_sar_suitability(
        self,
        severe_weather: Optional[bool],
        precipitation: Optional[bool]
    ) -> bool:
        """
        Evaluate if SAR sensors are suitable.

        SAR is generally all-weather, but extreme conditions can degrade performance.
        """

        # SAR works in most conditions
        # Only severe weather with heavy precipitation may degrade
        if severe_weather and precipitation:
            # Still suitable, but with reduced confidence
            return True

        return True

    def _evaluate_thermal_suitability(
        self,
        cloud_cover: Optional[float],
        severe_weather: Optional[bool]
    ) -> bool:
        """Evaluate if thermal/IR sensors are suitable."""

        # Thermal can see through some clouds but not heavy cover
        if cloud_cover is not None and cloud_cover > self.cloud_cover_threshold_thermal:
            return False

        # Severe weather may limit thermal
        if severe_weather:
            return False

        return True

    def _recommend_sensors(
        self,
        optical_suitable: bool,
        sar_suitable: bool,
        thermal_suitable: bool,
        condition: AtmosphericCondition
    ) -> List[SensorType]:
        """Determine recommended sensor types in priority order."""

        recommended = []

        # In good conditions, prefer optical for clarity
        if condition in [AtmosphericCondition.EXCELLENT, AtmosphericCondition.GOOD]:
            if optical_suitable:
                recommended.append(SensorType.OPTICAL)
            if thermal_suitable:
                recommended.append(SensorType.THERMAL)
            if sar_suitable:
                recommended.append(SensorType.SAR)

        # In fair/poor conditions, prioritize all-weather sensors
        elif condition in [AtmosphericCondition.FAIR, AtmosphericCondition.POOR]:
            if sar_suitable:
                recommended.append(SensorType.SAR)
            if thermal_suitable:
                recommended.append(SensorType.THERMAL)
            if optical_suitable:
                recommended.append(SensorType.OPTICAL)

        # In degraded conditions, rely on SAR
        else:
            if sar_suitable:
                recommended.append(SensorType.SAR)
            if thermal_suitable:
                recommended.append(SensorType.THERMAL)

        # If nothing recommended, fall back to SAR
        if not recommended:
            recommended.append(SensorType.SAR)

        return recommended

    def _check_degraded_mode(
        self,
        cloud_cover: Optional[float],
        severe_weather: Optional[bool],
        recommended_sensors: List[SensorType]
    ) -> bool:
        """Check if system should enter degraded mode."""

        # Degraded mode if cloud cover exceeds threshold
        if cloud_cover is not None and cloud_cover > self.degraded_mode_threshold:
            return True

        # Degraded mode if severe weather
        if severe_weather:
            return True

        # Degraded mode if no optical sensors available
        if SensorType.OPTICAL not in recommended_sensors:
            # Only degraded if SAR is the only option
            if len(recommended_sensors) == 1 and recommended_sensors[0] == SensorType.SAR:
                return True

        return False

    def _generate_reason(
        self,
        condition: AtmosphericCondition,
        cloud_cover: Optional[float],
        precipitation: Optional[bool],
        severe_weather: Optional[bool],
        smoke: Optional[bool],
        recommended_sensors: List[SensorType]
    ) -> str:
        """Generate human-readable explanation of assessment."""

        reasons = []

        # Condition description
        if condition == AtmosphericCondition.EXCELLENT:
            reasons.append("excellent conditions with clear skies")
        elif condition == AtmosphericCondition.GOOD:
            reasons.append("good conditions with minimal clouds")
        elif condition == AtmosphericCondition.FAIR:
            reasons.append("fair conditions with moderate cloud cover")
        elif condition == AtmosphericCondition.POOR:
            reasons.append("poor conditions with significant cloud cover")
        else:
            reasons.append("degraded conditions with heavy clouds or severe weather")

        # Specific factors
        if cloud_cover is not None:
            reasons.append(f"cloud cover at {cloud_cover:.1f}%")

        if precipitation:
            reasons.append("precipitation present")

        if severe_weather:
            reasons.append("severe weather detected")

        if smoke:
            reasons.append("smoke or aerosols present")

        # Sensor recommendation
        sensor_names = [s.value for s in recommended_sensors]
        reasons.append(f"recommended sensors: {', '.join(sensor_names)}")

        return "; ".join(reasons)

    def _calculate_confidence(
        self,
        cloud_cover: Optional[float],
        precipitation: Optional[bool],
        severe_weather: Optional[bool],
        visibility: Optional[float],
        smoke: Optional[bool]
    ) -> float:
        """
        Calculate confidence based on data availability.

        More data available = higher confidence in assessment.
        """

        # Count available parameters
        parameters = [cloud_cover, precipitation, severe_weather, visibility, smoke]
        available_count = sum(1 for param in parameters if param is not None)
        total_count = len(parameters)

        return available_count / total_count


def assess_cloud_cover(
    cloud_cover_percent: float,
    sensor_type: str
) -> Dict[str, Any]:
    """
    Quick assessment of cloud cover impact on a specific sensor.

    Args:
        cloud_cover_percent: Cloud cover percentage (0-100)
        sensor_type: "optical", "sar", or "thermal"

    Returns:
        Dictionary with suitability assessment
    """

    if sensor_type == "optical":
        suitable = cloud_cover_percent < DEFAULT_OPTICAL_CLOUD_THRESHOLD
        impact = "high" if cloud_cover_percent > 50 else "medium" if cloud_cover_percent > 10 else "low"
    elif sensor_type == "thermal":
        suitable = cloud_cover_percent < DEFAULT_THERMAL_CLOUD_THRESHOLD
        impact = "high" if cloud_cover_percent > 70 else "medium" if cloud_cover_percent > 30 else "low"
    elif sensor_type == "sar":
        suitable = True
        impact = "low"
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    return {
        "sensor_type": sensor_type,
        "cloud_cover_percent": cloud_cover_percent,
        "suitable": suitable,
        "impact": impact,
        "confidence": 0.9  # High confidence when cloud_cover data available
    }


def recommend_sensors_for_event(
    event_class: str,
    atmospheric_assessment: AtmosphericAssessment
) -> List[Dict[str, Any]]:
    """
    Recommend specific sensors based on event type and atmospheric conditions.

    Args:
        event_class: Event classification (e.g., "flood.coastal")
        atmospheric_assessment: Current atmospheric assessment

    Returns:
        List of sensor recommendations with rationale
    """

    recommendations = []

    # Base recommendations on atmospheric conditions
    for sensor in atmospheric_assessment.recommended_sensors:

        # Event-specific suitability
        if "flood" in event_class:
            if sensor == SensorType.SAR:
                priority = 1
                rationale = "SAR excellent for water detection in all weather"
            elif sensor == SensorType.OPTICAL:
                priority = 2
                rationale = "Optical provides high-resolution flood mapping when clear"
            else:
                priority = 3
                rationale = "Thermal can detect water temperature differences"

        elif "wildfire" in event_class or "fire" in event_class:
            if sensor == SensorType.THERMAL:
                priority = 1
                rationale = "Thermal ideal for active fire detection"
            elif sensor == SensorType.OPTICAL:
                priority = 2
                rationale = "Optical for burn scar mapping and smoke detection"
            elif sensor == SensorType.SAR:
                priority = 3
                rationale = "SAR can penetrate smoke for structural assessment"
            else:
                priority = 4
                rationale = "Additional sensor for cross-validation"

        elif "storm" in event_class:
            if sensor == SensorType.SAR:
                priority = 1
                rationale = "SAR for all-weather damage assessment"
            elif sensor == SensorType.OPTICAL:
                priority = 2
                rationale = "Optical for detailed damage mapping when clear"
            else:
                priority = 3
                rationale = "Thermal for temperature anomalies"

        else:
            # Default priority
            priority = atmospheric_assessment.recommended_sensors.index(sensor) + 1
            rationale = f"{sensor.value} recommended based on atmospheric conditions"

        recommendations.append({
            "sensor_type": sensor.value,
            "priority": priority,
            "rationale": rationale,
            "atmospheric_suitability": getattr(atmospheric_assessment, f"{sensor.value}_suitable")
        })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return recommendations
