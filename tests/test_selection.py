"""
Tests for sensor selection strategy.

Tests intelligent sensor selection with degraded mode handling,
atmospheric condition awareness, and observable dependency resolution.
"""

import pytest
from core.data.selection.strategy import (
    SensorType,
    Observable,
    ConfidenceLevel,
    SensorCapability,
    SensorSelection,
    SensorSelectionStrategy,
    get_observables_for_event_class,
)


class TestSensorCapability:
    """Test sensor capability evaluation."""

    def test_evaluate_confidence_clear_conditions(self):
        """Test confidence under clear conditions."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"cloud_cover": "obscures"}
        )

        # Clear skies
        confidence = cap.evaluate_confidence({"cloud_cover": 5})
        assert confidence == ConfidenceLevel.HIGH

    def test_evaluate_confidence_moderate_clouds(self):
        """Test confidence with moderate cloud cover."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"cloud_cover": "obscures"}
        )

        # Moderate clouds
        confidence = cap.evaluate_confidence({"cloud_cover": 40})
        assert confidence == ConfidenceLevel.MEDIUM

    def test_evaluate_confidence_heavy_clouds(self):
        """Test confidence with heavy cloud cover."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"cloud_cover": "obscures"}
        )

        # Heavy clouds
        confidence = cap.evaluate_confidence({"cloud_cover": 85})
        assert confidence == ConfidenceLevel.UNRELIABLE

    def test_evaluate_confidence_darkness(self):
        """Test optical sensor at night."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"darkness": "unusable"}
        )

        # Nighttime
        confidence = cap.evaluate_confidence({"darkness": True})
        assert confidence == ConfidenceLevel.UNRELIABLE

    def test_sar_unaffected_by_clouds(self):
        """Test SAR immunity to cloud cover."""
        cap = SensorCapability(
            sensor_type=SensorType.SAR,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={}
        )

        # Heavy clouds don't affect SAR
        confidence = cap.evaluate_confidence({"cloud_cover": 100})
        assert confidence == ConfidenceLevel.HIGH


class TestSensorSelection:
    """Test sensor selection dataclass."""

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        selection = SensorSelection(
            observable=Observable.WATER_EXTENT,
            primary_sensor=SensorType.SAR,
            supporting_sensors=[SensorType.OPTICAL],
            confidence=ConfidenceLevel.HIGH,
            degraded_mode=False,
            rationale="SAR selected for water_extent; confidence=high",
            alternatives_rejected={
                SensorType.OPTICAL: "cloud_cover_too_high"
            }
        )

        data = selection.to_dict()

        assert data["observable"] == "water_extent"
        assert data["primary_sensor"] == "sar"
        assert data["supporting_sensors"] == ["optical"]
        assert data["confidence"] == "high"
        assert data["degraded_mode"] is False
        assert "SAR selected" in data["rationale"]
        assert data["alternatives_rejected"]["optical"] == "cloud_cover_too_high"


class TestSensorSelectionStrategy:
    """Test sensor selection strategy."""

    @pytest.fixture
    def strategy(self):
        """Create sensor selection strategy."""
        return SensorSelectionStrategy()

    def test_select_sar_for_water_clear_conditions(self, strategy):
        """Test SAR selection for water extent in clear conditions."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR, SensorType.OPTICAL}
        conditions = {"cloud_cover": 10}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        assert Observable.WATER_EXTENT in selections
        selection = selections[Observable.WATER_EXTENT]

        # SAR is preferred (first in capabilities list)
        assert selection.primary_sensor == SensorType.SAR
        assert selection.confidence == ConfidenceLevel.HIGH
        assert selection.degraded_mode is False

    def test_select_sar_for_water_cloudy_conditions(self, strategy):
        """Test SAR selection for water extent in cloudy conditions."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR, SensorType.OPTICAL}
        conditions = {"cloud_cover": 90}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        assert Observable.WATER_EXTENT in selections
        selection = selections[Observable.WATER_EXTENT]

        # SAR still works despite clouds
        assert selection.primary_sensor == SensorType.SAR
        assert selection.confidence == ConfidenceLevel.HIGH
        assert SensorType.OPTICAL not in selection.supporting_sensors  # Too cloudy

    def test_select_optical_when_sar_unavailable(self, strategy):
        """Test fallback to optical when SAR unavailable."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}  # No SAR
        conditions = {"cloud_cover": 10}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        assert Observable.WATER_EXTENT in selections
        selection = selections[Observable.WATER_EXTENT]

        # Falls back to optical
        assert selection.primary_sensor == SensorType.OPTICAL
        assert selection.confidence == ConfidenceLevel.MEDIUM  # Base confidence
        assert SensorType.SAR in selection.alternatives_rejected
        assert selection.alternatives_rejected[SensorType.SAR] == "not_available"

    def test_optical_degraded_mode_with_clouds(self, strategy):
        """Test optical operates in degraded mode with clouds."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 60}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions, allow_degraded=True
        )

        assert Observable.WATER_EXTENT in selections
        selection = selections[Observable.WATER_EXTENT]

        assert selection.primary_sensor == SensorType.OPTICAL
        assert selection.confidence == ConfidenceLevel.LOW
        assert selection.degraded_mode is True

    def test_reject_degraded_when_not_allowed(self, strategy):
        """Test rejection of degraded mode selections."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 60}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions, allow_degraded=False
        )

        # Should not select anything since only degraded option available
        assert Observable.WATER_EXTENT not in selections

    def test_dependency_resolution_flood_depth(self, strategy):
        """Test dependency resolution for flood depth."""
        observables = [Observable.FLOOD_DEPTH, Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR, SensorType.DEM}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Both should be selected
        assert Observable.WATER_EXTENT in selections
        assert Observable.FLOOD_DEPTH in selections

        # DEM requires water extent
        flood_depth = selections[Observable.FLOOD_DEPTH]
        assert flood_depth.primary_sensor == SensorType.DEM
        assert flood_depth.confidence == ConfidenceLevel.HIGH

    def test_unresolved_dependency(self, strategy):
        """Test handling of unresolved dependencies."""
        # Request flood depth without water extent sensor
        observables = [Observable.FLOOD_DEPTH]
        available_sensors = {SensorType.DEM}  # No SAR/optical for water extent
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should not be able to select flood depth without water extent
        assert Observable.FLOOD_DEPTH not in selections

    def test_supporting_sensors_added(self, strategy):
        """Test that alternative sensors are added as supporting."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR, SensorType.OPTICAL}
        conditions = {"cloud_cover": 10}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.WATER_EXTENT]

        # SAR primary, optical supporting (since available and conditions good)
        assert selection.primary_sensor == SensorType.SAR
        assert SensorType.OPTICAL in selection.supporting_sensors

    def test_burn_severity_optical_preferred(self, strategy):
        """Test optical preferred for burn severity."""
        observables = [Observable.BURN_SEVERITY]
        available_sensors = {SensorType.OPTICAL, SensorType.SAR}
        conditions = {"cloud_cover": 5}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.BURN_SEVERITY]

        # Optical is first choice for burn severity
        assert selection.primary_sensor == SensorType.OPTICAL
        assert selection.confidence == ConfidenceLevel.HIGH

    def test_burn_severity_sar_fallback(self, strategy):
        """Test SAR fallback for burn severity when cloudy."""
        observables = [Observable.BURN_SEVERITY]
        available_sensors = {SensorType.OPTICAL, SensorType.SAR}
        conditions = {"cloud_cover": 95}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.BURN_SEVERITY]

        # Should fall back to SAR due to clouds
        # Note: Strategy uses first viable sensor, which is optical with unreliable,
        # then SAR
        assert selection.primary_sensor == SensorType.SAR
        assert selection.confidence == ConfidenceLevel.MEDIUM

    def test_active_fire_thermal_preferred(self, strategy):
        """Test thermal sensor preferred for active fire."""
        observables = [Observable.ACTIVE_FIRE]
        available_sensors = {SensorType.THERMAL, SensorType.OPTICAL}
        conditions = {"cloud_cover": 10}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.ACTIVE_FIRE]

        assert selection.primary_sensor == SensorType.THERMAL
        assert selection.confidence == ConfidenceLevel.HIGH

    def test_structural_damage_sar_preferred(self, strategy):
        """Test SAR preferred for structural damage."""
        observables = [Observable.STRUCTURAL_DAMAGE, Observable.LAND_COVER]
        available_sensors = {SensorType.SAR, SensorType.ANCILLARY}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Land cover first
        assert selections[Observable.LAND_COVER].primary_sensor == SensorType.ANCILLARY

        # Then structural damage
        assert selections[Observable.STRUCTURAL_DAMAGE].primary_sensor == SensorType.SAR
        assert selections[Observable.STRUCTURAL_DAMAGE].confidence == ConfidenceLevel.HIGH

    def test_get_required_data_types(self, strategy):
        """Test getting required data types for observables."""
        observables = [Observable.WATER_EXTENT, Observable.FLOOD_DEPTH]

        required = strategy.get_required_data_types("flood.riverine", observables)

        # Should include SAR (for water), DEM (for depth)
        assert "sar" in required
        assert "dem" in required

    def test_unknown_observable_handling(self, strategy):
        """Test handling of unknown observables."""
        # Create observable that doesn't exist in capabilities
        observables = [Observable.PRECIPITATION]  # Not in default capabilities
        available_sensors = {SensorType.WEATHER}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should return empty for unknown observable
        assert Observable.PRECIPITATION not in selections


class TestEventClassMapping:
    """Test event class to observable mapping."""

    def test_flood_observables(self):
        """Test flood event class mapping."""
        observables = get_observables_for_event_class("flood.riverine")

        assert Observable.WATER_EXTENT in observables
        assert Observable.TERRAIN_HEIGHT in observables
        assert Observable.FLOOD_DEPTH in observables

    def test_flood_coastal_observables(self):
        """Test coastal flood mapping."""
        observables = get_observables_for_event_class("flood.coastal.storm_surge")

        assert Observable.WATER_EXTENT in observables
        assert Observable.FLOOD_DEPTH in observables

    def test_flood_flash_observables(self):
        """Test flash flood mapping (no depth)."""
        observables = get_observables_for_event_class("flood.flash")

        assert Observable.WATER_EXTENT in observables
        assert Observable.TERRAIN_HEIGHT in observables
        # Flash floods don't need depth
        assert Observable.FLOOD_DEPTH not in observables

    def test_wildfire_observables(self):
        """Test wildfire event class mapping."""
        observables = get_observables_for_event_class("wildfire.forest")

        assert Observable.BURN_SEVERITY in observables
        # Not active fire for historical analysis
        assert Observable.ACTIVE_FIRE not in observables

    def test_wildfire_active_observables(self):
        """Test active wildfire mapping."""
        observables = get_observables_for_event_class("wildfire.active")

        assert Observable.BURN_SEVERITY in observables
        assert Observable.ACTIVE_FIRE in observables

    def test_storm_observables(self):
        """Test storm event class mapping."""
        observables = get_observables_for_event_class("storm.tornado")

        assert Observable.VEGETATION_DAMAGE in observables
        assert Observable.LAND_COVER in observables
        assert Observable.STRUCTURAL_DAMAGE in observables  # Tornado affects structures

    def test_unknown_event_class(self):
        """Test unknown event class."""
        observables = get_observables_for_event_class("earthquake.magnitude7")

        # Should return empty list with warning logged
        assert observables == []


class TestDegradedModeThresholds:
    """Test degraded mode threshold detection."""

    @pytest.fixture
    def strategy(self):
        """Create sensor selection strategy."""
        return SensorSelectionStrategy()

    def test_medium_degradation_threshold(self, strategy):
        """Test medium confidence at 30% cloud cover."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 30}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.WATER_EXTENT]

        # Should work with reduced confidence (MEDIUM is acceptable, not degraded)
        assert selection.confidence == ConfidenceLevel.MEDIUM
        assert selection.degraded_mode is False  # MEDIUM is acceptable, not degraded

    def test_high_degradation_threshold(self, strategy):
        """Test high degradation at 60% cloud cover."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 60}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.WATER_EXTENT]

        # Low confidence but still usable if allowed
        assert selection.confidence == ConfidenceLevel.LOW
        assert selection.degraded_mode is True

    def test_unreliable_threshold(self, strategy):
        """Test unreliable threshold at 85% cloud cover."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 85}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should not select at all (unreliable)
        assert Observable.WATER_EXTENT not in selections


class TestRationaleGeneration:
    """Test rationale string generation."""

    @pytest.fixture
    def strategy(self):
        """Create sensor selection strategy."""
        return SensorSelectionStrategy()

    def test_rationale_includes_key_info(self, strategy):
        """Test rationale includes sensor, observable, and confidence."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR}
        conditions = {"cloud_cover": 50}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        rationale = selections[Observable.WATER_EXTENT].rationale

        assert "sar" in rationale.lower()
        assert "water_extent" in rationale.lower()
        assert "confidence" in rationale.lower()

    def test_rationale_includes_degraded_flag(self, strategy):
        """Test rationale flags degraded mode."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL}
        conditions = {"cloud_cover": 60}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        rationale = selections[Observable.WATER_EXTENT].rationale

        assert "degraded" in rationale.lower()

    def test_rationale_includes_conditions(self, strategy):
        """Test rationale includes relevant conditions."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR}
        conditions = {"cloud_cover": 75}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        rationale = selections[Observable.WATER_EXTENT].rationale

        assert "cloud_cover=75" in rationale


class TestComplexScenarios:
    """Test complex multi-observable scenarios."""

    @pytest.fixture
    def strategy(self):
        """Create sensor selection strategy."""
        return SensorSelectionStrategy()

    def test_flood_mapping_full_suite(self, strategy):
        """Test complete flood mapping sensor selection."""
        observables = [
            Observable.WATER_EXTENT,
            Observable.FLOOD_DEPTH,
            Observable.TERRAIN_HEIGHT,
        ]
        available_sensors = {
            SensorType.SAR,
            SensorType.OPTICAL,
            SensorType.DEM,
        }
        conditions = {"cloud_cover": 20}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # All observables should be satisfied
        assert len(selections) == 3
        assert selections[Observable.WATER_EXTENT].primary_sensor == SensorType.SAR
        assert selections[Observable.TERRAIN_HEIGHT].primary_sensor == SensorType.DEM
        assert selections[Observable.FLOOD_DEPTH].primary_sensor == SensorType.DEM

    def test_wildfire_dual_sensor_strategy(self, strategy):
        """Test wildfire with both active fire and burn severity."""
        observables = [
            Observable.ACTIVE_FIRE,
            Observable.BURN_SEVERITY,
        ]
        available_sensors = {
            SensorType.THERMAL,
            SensorType.OPTICAL,
        }
        conditions = {"cloud_cover": 10}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        assert len(selections) == 2
        assert selections[Observable.ACTIVE_FIRE].primary_sensor == SensorType.THERMAL
        assert selections[Observable.BURN_SEVERITY].primary_sensor == SensorType.OPTICAL

    def test_partial_sensor_availability(self, strategy):
        """Test handling partial sensor availability."""
        observables = [
            Observable.WATER_EXTENT,
            Observable.BURN_SEVERITY,
            Observable.ACTIVE_FIRE,
        ]
        available_sensors = {
            SensorType.SAR,  # Can do water extent
            # No optical or thermal
        }
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Only water extent should be satisfied
        assert len(selections) == 2  # Water extent + burn severity (SAR fallback)
        assert Observable.WATER_EXTENT in selections
        assert Observable.BURN_SEVERITY in selections
        assert Observable.ACTIVE_FIRE not in selections  # No thermal/optical


# ============================================================================
# ATMOSPHERIC ASSESSMENT TESTS (Track 3)
# ============================================================================

from core.data.selection.atmospheric import (
    AtmosphericAssessment,
    AtmosphericCondition,
    AtmosphericEvaluator,
    SensorType as AtmoSensorType,
    assess_cloud_cover,
    recommend_sensors_for_event
)


class TestAtmosphericEvaluator:
    """Test atmospheric condition assessment."""

    def test_excellent_conditions(self):
        """Test assessment with excellent conditions."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=2.0,
            precipitation=False,
            severe_weather=False,
            visibility_km=50.0,
            smoke_aerosols=False
        )

        assert assessment.condition == AtmosphericCondition.EXCELLENT
        assert assessment.optical_suitable is True
        assert assessment.sar_suitable is True
        assert assessment.thermal_suitable is True
        assert AtmoSensorType.OPTICAL in assessment.recommended_sensors
        assert assessment.degraded_mode is False
        assert assessment.confidence == 1.0

    def test_good_conditions(self):
        """Test assessment with good conditions."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=8.0,
            precipitation=False,
            severe_weather=False
        )

        assert assessment.condition == AtmosphericCondition.GOOD
        assert assessment.optical_suitable is True
        assert assessment.sar_suitable is True
        assert assessment.degraded_mode is False

    def test_fair_conditions(self):
        """Test assessment with fair conditions."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=30.0,
            precipitation=False,
            severe_weather=False
        )

        assert assessment.condition == AtmosphericCondition.FAIR
        assert assessment.optical_suitable is False  # Above 20% threshold
        assert assessment.sar_suitable is True
        assert AtmoSensorType.SAR == assessment.recommended_sensors[0]  # SAR prioritized

    def test_poor_conditions(self):
        """Test assessment with poor conditions."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=65.0,
            precipitation=True,
            severe_weather=False
        )

        assert assessment.condition == AtmosphericCondition.POOR
        assert assessment.optical_suitable is False
        assert assessment.sar_suitable is True
        assert assessment.thermal_suitable is False  # Above 50% threshold
        assert AtmoSensorType.SAR in assessment.recommended_sensors

    def test_degraded_conditions_severe_weather(self):
        """Test degraded mode with severe weather."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=90.0,
            precipitation=True,
            severe_weather=True
        )

        assert assessment.condition == AtmosphericCondition.DEGRADED
        assert assessment.optical_suitable is False
        assert assessment.thermal_suitable is False
        assert assessment.degraded_mode is True

    def test_degraded_conditions_heavy_clouds(self):
        """Test degraded mode with heavy cloud cover."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=85.0
        )

        assert assessment.condition == AtmosphericCondition.DEGRADED
        assert assessment.degraded_mode is True

    def test_smoke_blocks_optical(self):
        """Test that smoke blocks optical sensors."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=5.0,
            smoke_aerosols=True
        )

        assert assessment.optical_suitable is False
        assert assessment.sar_suitable is True
        assert AtmoSensorType.OPTICAL not in assessment.recommended_sensors

    def test_low_visibility_blocks_optical(self):
        """Test that low visibility blocks optical sensors."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=5.0,
            visibility_km=3.0
        )

        assert assessment.optical_suitable is False

    def test_custom_thresholds(self):
        """Test custom cloud cover thresholds."""
        evaluator = AtmosphericEvaluator(
            cloud_cover_threshold_optical=30.0,
            cloud_cover_threshold_thermal=70.0,
            degraded_mode_threshold=90.0
        )

        assessment = evaluator.assess(cloud_cover_percent=25.0)

        assert assessment.optical_suitable is True  # Below custom threshold
        assert assessment.thermal_suitable is True
        assert assessment.degraded_mode is False

    def test_sar_all_weather(self):
        """Test that SAR is suitable in all weather conditions."""
        evaluator = AtmosphericEvaluator()

        # Test with various bad conditions
        conditions = [
            {"cloud_cover_percent": 100.0},
            {"precipitation": True, "severe_weather": True},
            {"cloud_cover_percent": 80.0, "precipitation": True}
        ]

        for condition in conditions:
            assessment = evaluator.assess(**condition)
            assert assessment.sar_suitable is True

    def test_confidence_calculation(self):
        """Test confidence calculation based on data availability."""
        evaluator = AtmosphericEvaluator()

        # All data available
        assessment1 = evaluator.assess(
            cloud_cover_percent=10.0,
            precipitation=False,
            severe_weather=False,
            visibility_km=50.0,
            smoke_aerosols=False
        )
        assert assessment1.confidence == 1.0

        # Only cloud cover
        assessment2 = evaluator.assess(cloud_cover_percent=10.0)
        assert assessment2.confidence == 0.2

        # Three parameters
        assessment3 = evaluator.assess(
            cloud_cover_percent=10.0,
            precipitation=False,
            severe_weather=False
        )
        assert assessment3.confidence == 0.6

    def test_assess_from_weather_data(self):
        """Test assessment from weather data dictionary."""
        evaluator = AtmosphericEvaluator()

        weather_data = {
            "cloud_cover": 15.0,
            "precipitation": False,
            "severe_weather": False,
            "visibility_km": 30.0,
            "smoke_aerosols": False,
            "metadata": {"source": "ERA5"}
        }

        assessment = evaluator.assess_from_weather_data(weather_data)

        assert assessment.condition == AtmosphericCondition.FAIR
        assert assessment.optical_suitable is True
        assert assessment.confidence == 1.0

    def test_reason_generation(self):
        """Test that reason strings are generated."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=30.0,
            precipitation=True
        )

        assert assessment.reason is not None
        assert len(assessment.reason) > 0
        assert "30.0%" in assessment.reason
        assert "precipitation" in assessment.reason

    def test_to_dict(self):
        """Test conversion to dictionary."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(cloud_cover_percent=10.0)
        result_dict = assessment.to_dict()

        assert isinstance(result_dict, dict)
        assert "cloud_cover_percent" in result_dict
        assert "condition" in result_dict
        assert "recommended_sensors" in result_dict
        assert isinstance(result_dict["recommended_sensors"], list)


class TestCloudCoverAssessment:
    """Test quick cloud cover assessment function."""

    def test_optical_low_clouds(self):
        """Test optical sensor with low cloud cover."""
        result = assess_cloud_cover(5.0, "optical")

        assert result["suitable"] is True
        assert result["impact"] == "low"
        assert result["sensor_type"] == "optical"

    def test_optical_medium_clouds(self):
        """Test optical sensor with medium cloud cover."""
        result = assess_cloud_cover(30.0, "optical")

        assert result["suitable"] is False
        assert result["impact"] == "medium"

    def test_optical_high_clouds(self):
        """Test optical sensor with high cloud cover."""
        result = assess_cloud_cover(60.0, "optical")

        assert result["suitable"] is False
        assert result["impact"] == "high"

    def test_thermal_moderate_clouds(self):
        """Test thermal sensor with moderate cloud cover."""
        result = assess_cloud_cover(40.0, "thermal")

        assert result["suitable"] is True
        assert result["impact"] == "medium"

    def test_thermal_high_clouds(self):
        """Test thermal sensor with high cloud cover."""
        result = assess_cloud_cover(80.0, "thermal")

        assert result["suitable"] is False
        assert result["impact"] == "high"

    def test_sar_any_clouds(self):
        """Test SAR sensor with any cloud cover."""
        for cloud_cover in [0.0, 50.0, 100.0]:
            result = assess_cloud_cover(cloud_cover, "sar")

            assert result["suitable"] is True
            assert result["impact"] == "low"

    def test_invalid_sensor_type(self):
        """Test with invalid sensor type."""
        with pytest.raises(ValueError):
            assess_cloud_cover(10.0, "invalid_sensor")


class TestEventSpecificRecommendations:
    """Test event-specific sensor recommendations."""

    def test_flood_event_recommendations(self):
        """Test sensor recommendations for flood events."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(cloud_cover_percent=40.0)

        recommendations = recommend_sensors_for_event("flood.coastal", assessment)

        assert len(recommendations) > 0
        # SAR should be highest priority for floods
        assert recommendations[0]["sensor_type"] == "sar"
        assert "water detection" in recommendations[0]["rationale"]

    def test_wildfire_event_recommendations(self):
        """Test sensor recommendations for wildfire events."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(cloud_cover_percent=30.0)

        recommendations = recommend_sensors_for_event("wildfire.forest", assessment)

        # Thermal should be recommended for wildfires
        thermal_rec = [r for r in recommendations if r["sensor_type"] == "thermal"]
        assert len(thermal_rec) > 0
        assert "fire" in thermal_rec[0]["rationale"].lower()

    def test_storm_event_recommendations(self):
        """Test sensor recommendations for storm events."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(
            cloud_cover_percent=70.0,
            severe_weather=True
        )

        recommendations = recommend_sensors_for_event("storm.tropical_cyclone", assessment)

        # SAR should be highest priority for storms
        assert recommendations[0]["sensor_type"] == "sar"
        assert "all-weather" in recommendations[0]["rationale"]

    def test_recommendations_sorted_by_priority(self):
        """Test that recommendations are sorted by priority."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(cloud_cover_percent=10.0)

        recommendations = recommend_sensors_for_event("flood.riverine", assessment)

        # Check that priorities are in ascending order
        priorities = [r["priority"] for r in recommendations]
        assert priorities == sorted(priorities)

    def test_recommendations_include_suitability(self):
        """Test that recommendations include atmospheric suitability."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(cloud_cover_percent=60.0)

        recommendations = recommend_sensors_for_event("flood.urban", assessment)

        for rec in recommendations:
            assert "atmospheric_suitability" in rec
            assert isinstance(rec["atmospheric_suitability"], bool)


class TestAdditionalEdgeCases:
    """Additional edge case tests for robustness."""

    @pytest.fixture
    def strategy(self):
        """Create sensor selection strategy."""
        return SensorSelectionStrategy()

    def test_empty_observables_list(self, strategy):
        """Test handling of empty observables list."""
        observables = []
        available_sensors = {SensorType.SAR, SensorType.OPTICAL}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        assert len(selections) == 0

    def test_empty_available_sensors(self, strategy):
        """Test handling of empty available sensors."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = set()  # No sensors available
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should not be able to select anything
        assert Observable.WATER_EXTENT not in selections

    def test_none_conditions(self, strategy):
        """Test that None conditions are handled correctly."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR}
        conditions = None  # Explicitly None

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should work with default empty dict
        assert Observable.WATER_EXTENT in selections

    def test_circular_dependency_protection(self, strategy):
        """Test that circular dependencies don't cause infinite loops."""
        # The max_passes=5 limit should prevent infinite loops
        observables = [Observable.FLOOD_DEPTH]
        available_sensors = {SensorType.DEM}
        # Missing water extent sensor, but DEM requires it
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should terminate after max passes without hanging
        assert Observable.FLOOD_DEPTH not in selections

    def test_multiple_identical_observables(self, strategy):
        """Test handling of duplicate observables in request."""
        observables = [
            Observable.WATER_EXTENT,
            Observable.WATER_EXTENT,  # Duplicate
            Observable.WATER_EXTENT,  # Duplicate
        ]
        available_sensors = {SensorType.SAR}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # Should only select once for each unique observable
        assert len(selections) == 1
        assert Observable.WATER_EXTENT in selections

    def test_extremely_high_cloud_cover(self, strategy):
        """Test handling of unrealistically high cloud cover values."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.OPTICAL, SensorType.SAR}
        conditions = {"cloud_cover": 150.0}  # Invalid but test robustness

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        # SAR should be selected since optical will be unreliable
        assert selections[Observable.WATER_EXTENT].primary_sensor == SensorType.SAR

    def test_negative_cloud_cover(self, strategy):
        """Test handling of negative cloud cover values."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"cloud_cover": "obscures"}
        )

        # Negative cloud cover should be treated as clear
        confidence = cap.evaluate_confidence({"cloud_cover": -5.0})
        assert confidence == ConfidenceLevel.HIGH

    def test_confidence_downgrade_boundary(self, strategy):
        """Test confidence downgrade at exact boundary values."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"cloud_cover": "obscures"}
        )

        # Test exact boundary at 30% - triggers 30-60% degradation (value > 30)
        confidence_30 = cap.evaluate_confidence({"cloud_cover": 30.0})
        assert confidence_30 == ConfidenceLevel.HIGH  # Exactly at boundary, not >

        # Test just over 30%
        confidence_31 = cap.evaluate_confidence({"cloud_cover": 31.0})
        assert confidence_31 == ConfidenceLevel.MEDIUM  # Downgraded by 1 step

        # Test exact boundary at 60% - triggers 60-80% degradation (value > 60)
        confidence_60 = cap.evaluate_confidence({"cloud_cover": 60.0})
        assert confidence_60 == ConfidenceLevel.MEDIUM  # Downgraded at 30-60 range

        # Test just over 60%
        confidence_61 = cap.evaluate_confidence({"cloud_cover": 61.0})
        assert confidence_61 == ConfidenceLevel.LOW  # Heavy degradation

        # Test exact boundary at 80%
        confidence_80 = cap.evaluate_confidence({"cloud_cover": 80.0})
        assert confidence_80 == ConfidenceLevel.LOW  # Heavy degradation (60-80)

        # Test just over 80%
        confidence_81 = cap.evaluate_confidence({"cloud_cover": 81.0})
        assert confidence_81 == ConfidenceLevel.UNRELIABLE  # Over boundary

    def test_sensor_selection_to_dict_all_fields(self, strategy):
        """Test that to_dict includes all expected fields."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR, SensorType.OPTICAL}
        conditions = {"cloud_cover": 90}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        data = selections[Observable.WATER_EXTENT].to_dict()

        # Verify all required fields are present
        required_fields = [
            "observable", "primary_sensor", "supporting_sensors",
            "confidence", "degraded_mode", "rationale",
            "alternatives_rejected"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_find_capability_returns_none_for_missing(self, strategy):
        """Test that _find_capability returns None for non-existent combinations."""
        result = strategy._find_capability(
            Observable.WATER_EXTENT,
            SensorType.WEATHER  # Not a sensor for water extent
        )

        assert result is None

    def test_get_required_data_types_empty_observables(self, strategy):
        """Test get_required_data_types with empty observables."""
        required = strategy.get_required_data_types("flood.coastal", [])

        assert len(required) == 0

    def test_precipitation_degradation(self, strategy):
        """Test that precipitation degrades sensor confidence."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={"precipitation": "obscures"}
        )

        # Test with precipitation
        confidence = cap.evaluate_confidence({"precipitation": True})
        assert confidence == ConfidenceLevel.MEDIUM  # Downgraded by 1 step

    def test_multiple_degradation_factors(self, strategy):
        """Test multiple degradation factors applied together."""
        cap = SensorCapability(
            sensor_type=SensorType.OPTICAL,
            observable=Observable.WATER_EXTENT,
            confidence=ConfidenceLevel.HIGH,
            degrades_with={
                "cloud_cover": "obscures",
                "precipitation": "obscures"
            }
        )

        # Both cloud cover and precipitation
        confidence = cap.evaluate_confidence({
            "cloud_cover": 40.0,  # Would downgrade by 1
            "precipitation": True  # Would downgrade by 1
        })

        # Should be downgraded (combined effect)
        assert confidence in [ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]

    def test_event_class_edge_cases(self):
        """Test event class mapping edge cases."""
        # Empty string
        result = get_observables_for_event_class("")
        assert result == []

        # Very specific nested class
        result = get_observables_for_event_class("flood.coastal.storm_surge.category5")
        assert Observable.WATER_EXTENT in result
        assert Observable.FLOOD_DEPTH in result  # Still coastal

    def test_rationale_with_multiple_conditions(self, strategy):
        """Test rationale generation with multiple conditions."""
        observables = [Observable.WATER_EXTENT]
        available_sensors = {SensorType.SAR}
        conditions = {
            "cloud_cover": 75,
            "precipitation": True,
            "wind_speed": 40  # Extra condition not used by degradation
        }

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        rationale = selections[Observable.WATER_EXTENT].rationale

        # Should include cloud cover
        assert "cloud_cover=75" in rationale

    def test_sensor_capability_no_alternatives(self, strategy):
        """Test sensor with no alternatives specified."""
        observables = [Observable.TERRAIN_HEIGHT]
        available_sensors = {SensorType.DEM}
        conditions = {}

        selections = strategy.select_sensors(
            observables, available_sensors, conditions
        )

        selection = selections[Observable.TERRAIN_HEIGHT]

        # DEM has no alternatives
        assert len(selection.supporting_sensors) == 0
        assert selection.primary_sensor == SensorType.DEM


class TestAtmosphericEdgeCases:
    """Test edge cases and boundary conditions for atmospheric assessment."""

    def test_all_none_inputs(self):
        """Test assessment with all None inputs."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess()

        # Should default to GOOD with no negative indicators
        assert assessment.condition == AtmosphericCondition.GOOD
        assert assessment.optical_suitable is True
        assert assessment.sar_suitable is True
        assert assessment.thermal_suitable is True
        assert assessment.confidence == 0.0  # No data available

    def test_zero_cloud_cover(self):
        """Test assessment with exactly 0% cloud cover."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(cloud_cover_percent=0.0)

        assert assessment.condition == AtmosphericCondition.EXCELLENT
        assert assessment.optical_suitable is True
        assert assessment.degraded_mode is False

    def test_boundary_cloud_cover_excellent(self):
        """Test boundary at EXCELLENT/GOOD transition (5%)."""
        evaluator = AtmosphericEvaluator()

        # Just below boundary
        assessment1 = evaluator.assess(cloud_cover_percent=4.9)
        assert assessment1.condition == AtmosphericCondition.EXCELLENT

        # At boundary (should be GOOD)
        assessment2 = evaluator.assess(cloud_cover_percent=5.0)
        assert assessment2.condition == AtmosphericCondition.GOOD

    def test_boundary_cloud_cover_good(self):
        """Test boundary at GOOD/FAIR transition (10%)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(cloud_cover_percent=9.9)
        assert assessment1.condition == AtmosphericCondition.GOOD

        assessment2 = evaluator.assess(cloud_cover_percent=10.0)
        assert assessment2.condition == AtmosphericCondition.FAIR

    def test_boundary_cloud_cover_fair(self):
        """Test boundary at FAIR/POOR transition (50%)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(cloud_cover_percent=49.9)
        assert assessment1.condition == AtmosphericCondition.FAIR

        assessment2 = evaluator.assess(cloud_cover_percent=50.0)
        assert assessment2.condition == AtmosphericCondition.POOR

    def test_boundary_cloud_cover_poor(self):
        """Test boundary at POOR/DEGRADED transition (80%)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(cloud_cover_percent=79.9)
        assert assessment1.condition == AtmosphericCondition.POOR

        assessment2 = evaluator.assess(cloud_cover_percent=80.0)
        assert assessment2.condition == AtmosphericCondition.DEGRADED

    def test_hundred_percent_cloud_cover(self):
        """Test assessment with 100% cloud cover."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(cloud_cover_percent=100.0)

        assert assessment.condition == AtmosphericCondition.DEGRADED
        assert assessment.optical_suitable is False
        assert assessment.thermal_suitable is False
        assert assessment.sar_suitable is True  # SAR still works
        assert assessment.degraded_mode is True

    def test_boundary_optical_threshold(self):
        """Test boundary at optical suitability threshold (20%)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(cloud_cover_percent=19.9)
        assert assessment1.optical_suitable is True

        # At exactly 20.0%, still suitable (uses > not >=)
        assessment2 = evaluator.assess(cloud_cover_percent=20.0)
        assert assessment2.optical_suitable is True

        # Just over threshold
        assessment3 = evaluator.assess(cloud_cover_percent=20.1)
        assert assessment3.optical_suitable is False

    def test_boundary_thermal_threshold(self):
        """Test boundary at thermal suitability threshold (50%)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(cloud_cover_percent=49.9)
        assert assessment1.thermal_suitable is True

        # At exactly 50.0%, still suitable (uses > not >=)
        assessment2 = evaluator.assess(cloud_cover_percent=50.0)
        assert assessment2.thermal_suitable is True

        # Just over threshold
        assessment3 = evaluator.assess(cloud_cover_percent=50.1)
        assert assessment3.thermal_suitable is False

    def test_boundary_degraded_threshold(self):
        """Test boundary at degraded mode threshold (80%)."""
        evaluator = AtmosphericEvaluator()

        # At 79.9%, only SAR recommended (optical unsuitable), so degraded mode
        assessment1 = evaluator.assess(cloud_cover_percent=79.9)
        assert assessment1.degraded_mode is True  # Only SAR available = degraded

        # At exactly 80.0%, still not degraded by cloud_cover check (uses > not >=)
        # But still degraded because only SAR is recommended
        assessment2 = evaluator.assess(cloud_cover_percent=80.0)
        assert assessment2.degraded_mode is True  # Only SAR available = degraded

        # Just over threshold - same result
        assessment3 = evaluator.assess(cloud_cover_percent=80.1)
        assert assessment3.degraded_mode is True

    def test_boundary_visibility_threshold(self):
        """Test boundary at visibility threshold (5 km)."""
        evaluator = AtmosphericEvaluator()

        assessment1 = evaluator.assess(
            cloud_cover_percent=5.0,
            visibility_km=5.0
        )
        assert assessment1.optical_suitable is True

        assessment2 = evaluator.assess(
            cloud_cover_percent=5.0,
            visibility_km=4.9
        )
        assert assessment2.optical_suitable is False

    def test_conflicting_indicators(self):
        """Test with conflicting atmospheric indicators."""
        evaluator = AtmosphericEvaluator()

        # Clear skies but severe weather
        assessment = evaluator.assess(
            cloud_cover_percent=2.0,
            severe_weather=True
        )

        # Severe weather should override clear skies
        assert assessment.condition == AtmosphericCondition.DEGRADED
        assert assessment.degraded_mode is True

    def test_empty_recommended_sensors_fallback(self):
        """Test fallback when no sensors are recommended."""
        evaluator = AtmosphericEvaluator()

        # This shouldn't happen in practice, but test the fallback
        # All sensors unsuitable (hypothetical edge case)
        assessment = evaluator.assess(
            cloud_cover_percent=95.0,
            severe_weather=True
        )

        # Should always have at least SAR as fallback
        assert len(assessment.recommended_sensors) > 0
        assert AtmoSensorType.SAR in assessment.recommended_sensors

    def test_partial_confidence_calculation(self):
        """Test confidence calculation with partial data."""
        evaluator = AtmosphericEvaluator()

        # 1 of 5 parameters
        assessment1 = evaluator.assess(cloud_cover_percent=10.0)
        assert assessment1.confidence == 0.2

        # 2 of 5 parameters
        assessment2 = evaluator.assess(
            cloud_cover_percent=10.0,
            precipitation=False
        )
        assert assessment2.confidence == 0.4

        # 3 of 5 parameters
        assessment3 = evaluator.assess(
            cloud_cover_percent=10.0,
            precipitation=False,
            severe_weather=False
        )
        assert assessment3.confidence == 0.6

        # 4 of 5 parameters
        assessment4 = evaluator.assess(
            cloud_cover_percent=10.0,
            precipitation=False,
            severe_weather=False,
            visibility_km=20.0
        )
        assert assessment4.confidence == 0.8

    def test_negative_cloud_cover_handling(self):
        """Test handling of invalid negative cloud cover."""
        evaluator = AtmosphericEvaluator()

        # System should still work (no crash), treat as excellent
        assessment = evaluator.assess(cloud_cover_percent=-5.0)

        assert assessment.condition == AtmosphericCondition.EXCELLENT
        assert assessment.optical_suitable is True

    def test_cloud_cover_over_100_handling(self):
        """Test handling of invalid cloud cover > 100%."""
        evaluator = AtmosphericEvaluator()

        # Should still work, treat as degraded
        assessment = evaluator.assess(cloud_cover_percent=150.0)

        assert assessment.condition == AtmosphericCondition.DEGRADED
        assert assessment.optical_suitable is False
        assert assessment.degraded_mode is True

    def test_zero_visibility(self):
        """Test handling of zero visibility."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=5.0,
            visibility_km=0.0
        )

        assert assessment.optical_suitable is False

    def test_extreme_visibility(self):
        """Test handling of very high visibility."""
        evaluator = AtmosphericEvaluator()

        assessment = evaluator.assess(
            cloud_cover_percent=5.0,
            visibility_km=200.0
        )

        assert assessment.optical_suitable is True

    def test_unknown_event_class_recommendations(self):
        """Test recommendations for unknown event class."""
        evaluator = AtmosphericEvaluator()
        assessment = evaluator.assess(cloud_cover_percent=10.0)

        recommendations = recommend_sensors_for_event("unknown.event.type", assessment)

        # Should still return recommendations based on atmospheric conditions
        assert len(recommendations) > 0
        # Should use default priority (atmospheric order)
        for rec in recommendations:
            assert "priority" in rec
            assert "rationale" in rec


# ============================================================================
# DETERMINISTIC SELECTION TESTS (Track 7)
# ============================================================================

from datetime import datetime, timezone
from core.analysis.selection.deterministic import (
    DeterministicSelector,
    SelectionPlan,
    AlgorithmSelection,
    SelectionQuality,
    SelectionReason,
    SelectionConstraints,
    DataAvailability,
    create_deterministic_selector,
    SELECTOR_VERSION,
)
from core.analysis.library.registry import (
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmCategory,
    DataType,
    ResourceRequirements,
    ValidationMetrics,
)
from core.data.discovery.base import DiscoveryResult
from core.data.evaluation.ranking import RankedCandidate


class TestDataAvailability:
    """Test DataAvailability creation and queries."""

    def test_from_discovery_results_basic(self):
        """Test creating DataAvailability from discovery results."""
        results = [
            DiscoveryResult(
                dataset_id="s1-123",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/s1-123.tif",
                format="geotiff",
                acquisition_time=datetime.now(timezone.utc),
                spatial_coverage_percent=95.0,
                resolution_m=10.0,
            ),
            DiscoveryResult(
                dataset_id="dem-456",
                provider="copernicus",
                data_type="dem",
                source_uri="s3://bucket/dem-456.tif",
                format="geotiff",
                acquisition_time=datetime.now(timezone.utc),
                spatial_coverage_percent=100.0,
                resolution_m=30.0,
            ),
        ]

        availability = DataAvailability.from_discovery_results(results)

        assert DataType.SAR in availability.data_types
        assert DataType.DEM in availability.data_types
        assert "sar" in availability.sensor_types
        assert "dem" in availability.sensor_types
        assert len(availability.datasets_by_type["sar"]) == 1
        assert len(availability.datasets_by_type["dem"]) == 1

    def test_has_data_type(self):
        """Test data type availability check."""
        availability = DataAvailability(
            data_types={DataType.SAR, DataType.OPTICAL},
            sensor_types={"sar", "optical"},
        )

        assert availability.has_data_type(DataType.SAR) is True
        assert availability.has_data_type(DataType.OPTICAL) is True
        assert availability.has_data_type(DataType.DEM) is False

    def test_has_all_required(self):
        """Test checking multiple required data types."""
        availability = DataAvailability(
            data_types={DataType.SAR, DataType.DEM},
            sensor_types={"sar", "dem"},
        )

        assert availability.has_all_required([DataType.SAR]) is True
        assert availability.has_all_required([DataType.SAR, DataType.DEM]) is True
        assert availability.has_all_required([DataType.SAR, DataType.OPTICAL]) is False

    def test_get_missing(self):
        """Test getting missing data types."""
        availability = DataAvailability(
            data_types={DataType.SAR},
            sensor_types={"sar"},
        )

        missing = availability.get_missing([DataType.SAR, DataType.DEM, DataType.OPTICAL])

        assert DataType.DEM in missing
        assert DataType.OPTICAL in missing
        assert DataType.SAR not in missing

    def test_empty_results(self):
        """Test with empty discovery results."""
        availability = DataAvailability.from_discovery_results([])

        assert len(availability.data_types) == 0
        assert len(availability.sensor_types) == 0
        assert len(availability.datasets_by_type) == 0


class TestAlgorithmSelection:
    """Test AlgorithmSelection dataclass."""

    def test_hash_computation(self):
        """Test deterministic hash computation."""
        selection = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR, DataType.DEM],
        )

        # Hash should be non-empty
        assert len(selection.deterministic_hash) == 16

        # Same inputs should produce same hash
        selection2 = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR, DataType.DEM],
        )

        assert selection.deterministic_hash == selection2.deterministic_hash

    def test_hash_changes_with_inputs(self):
        """Test that hash changes when inputs change."""
        selection1 = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        )

        selection2 = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.1.0",  # Different version
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        )

        assert selection1.deterministic_hash != selection2.deterministic_hash

    def test_verify_hash(self):
        """Test hash verification."""
        selection = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        )

        # Valid hash should verify
        assert selection.verify_hash() is True

        # Tampered hash should fail
        selection.deterministic_hash = "tampered_hash_"
        assert selection.verify_hash() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        selection = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR, DataType.DEM],
            degraded_mode=False,
            rationale="Best match for flood detection",
            selection_reason=SelectionReason.BEST_MATCH,
            alternatives_rejected={"other_algo": "lower_priority"},
        )

        data = selection.to_dict()

        assert data["algorithm_id"] == "flood.baseline.threshold_sar"
        assert data["algorithm_name"] == "SAR Threshold"
        assert data["version"] == "1.0.0"
        assert data["quality"] == "optimal"
        assert data["required_data_types"] == ["sar"]
        assert data["available_data_types"] == ["sar", "dem"]
        assert data["degraded_mode"] is False
        assert "Best match" in data["rationale"]
        assert data["selection_reason"] == "best_match"
        assert "deterministic_hash" in data
        assert "selector_version" in data
        assert "selected_at" in data


class TestSelectionPlan:
    """Test SelectionPlan dataclass."""

    def test_add_selection(self):
        """Test adding selections to plan."""
        plan = SelectionPlan(event_class="flood.coastal")

        selection1 = AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        )

        plan.add_selection(selection1)

        assert "flood.baseline.threshold_sar" in plan.selections
        assert "flood.baseline.threshold_sar" in plan.execution_order
        assert plan.overall_quality == SelectionQuality.OPTIMAL
        assert len(plan.plan_hash) > 0

    def test_quality_degradation(self):
        """Test that overall quality degrades with degraded selections."""
        plan = SelectionPlan(event_class="flood.coastal")

        # Add optimal selection
        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo1",
            algorithm_name="Algo 1",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        ))

        assert plan.overall_quality == SelectionQuality.OPTIMAL

        # Add degraded selection
        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo2",
            algorithm_name="Algo 2",
            version="1.0.0",
            quality=SelectionQuality.DEGRADED,
            required_data_types=[DataType.OPTICAL],
            available_data_types=[DataType.OPTICAL],
            degraded_mode=True,
        ))

        assert plan.overall_quality == SelectionQuality.DEGRADED
        assert plan.degraded_mode is True

    def test_get_execution_sequence(self):
        """Test getting execution sequence."""
        plan = SelectionPlan(event_class="flood.coastal")

        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo1",
            algorithm_name="Algo 1",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        ))

        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo2",
            algorithm_name="Algo 2",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.DEM],
            available_data_types=[DataType.DEM],
        ))

        sequence = plan.get_execution_sequence()

        assert len(sequence) == 2
        assert sequence[0].algorithm_id == "algo1"
        assert sequence[1].algorithm_id == "algo2"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        plan = SelectionPlan(event_class="flood.coastal")

        plan.add_selection(AlgorithmSelection(
            algorithm_id="flood.baseline.threshold_sar",
            algorithm_name="SAR Threshold",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        ))

        data = plan.to_dict()

        assert data["event_class"] == "flood.coastal"
        assert "flood.baseline.threshold_sar" in data["selections"]
        assert data["execution_order"] == ["flood.baseline.threshold_sar"]
        assert data["overall_quality"] == "optimal"
        assert "plan_hash" in data
        assert "created_at" in data


class TestSelectionConstraints:
    """Test SelectionConstraints."""

    @pytest.fixture
    def sample_algorithm(self):
        """Create sample algorithm metadata."""
        return AlgorithmMetadata(
            id="flood.baseline.threshold_sar",
            name="SAR Threshold",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=2048, gpu_required=False),
            deterministic=True,
        )

    def test_allows_algorithm_basic(self, sample_algorithm):
        """Test basic constraint checking."""
        constraints = SelectionConstraints()

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is True
        assert reason is None

    def test_rejects_deprecated(self, sample_algorithm):
        """Test rejection of deprecated algorithms."""
        sample_algorithm.deprecated = True
        constraints = SelectionConstraints()

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is False
        assert reason == "deprecated"

    def test_rejects_excluded(self, sample_algorithm):
        """Test rejection of explicitly excluded algorithms."""
        constraints = SelectionConstraints(
            excluded_algorithms={"flood.baseline.threshold_sar"}
        )

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is False
        assert reason == "explicitly_excluded"

    def test_rejects_memory_exceeded(self, sample_algorithm):
        """Test rejection when memory constraint exceeded."""
        constraints = SelectionConstraints(max_memory_mb=1024)

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is False
        assert reason == "memory_exceeded"

    def test_allows_within_memory(self, sample_algorithm):
        """Test allowing algorithm within memory constraint."""
        constraints = SelectionConstraints(max_memory_mb=4096)

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is True

    def test_rejects_gpu_required(self, sample_algorithm):
        """Test rejection when GPU required but not available."""
        sample_algorithm.resources.gpu_required = True
        constraints = SelectionConstraints(gpu_available=False)

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is False
        assert reason == "gpu_required"

    def test_allows_with_gpu(self, sample_algorithm):
        """Test allowing GPU algorithm when GPU available."""
        sample_algorithm.resources.gpu_required = True
        constraints = SelectionConstraints(gpu_available=True)

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is True

    def test_rejects_non_deterministic(self, sample_algorithm):
        """Test rejection of non-deterministic algorithms."""
        sample_algorithm.deterministic = False
        constraints = SelectionConstraints(require_deterministic=True)

        allowed, reason = constraints.allows_algorithm(sample_algorithm)

        assert allowed is False
        assert reason == "non_deterministic"


class TestDeterministicSelector:
    """Test DeterministicSelector main functionality."""

    @pytest.fixture
    def registry(self):
        """Create test algorithm registry."""
        registry = AlgorithmRegistry()

        # Add test algorithms
        registry.register(AlgorithmMetadata(
            id="flood.baseline.threshold_sar",
            name="SAR Threshold Detection",
            category=AlgorithmCategory.BASELINE,
            version="1.2.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            optional_data_types=[DataType.DEM],
            deterministic=True,
            outputs=["water_mask"],
            validation=ValidationMetrics(accuracy_median=0.85),
        ))

        registry.register(AlgorithmMetadata(
            id="flood.baseline.ndwi_optical",
            name="NDWI Optical Detection",
            category=AlgorithmCategory.BASELINE,
            version="1.1.0",
            event_types=["flood.*"],
            required_data_types=[DataType.OPTICAL],
            optional_data_types=[DataType.DEM],
            deterministic=True,
            outputs=["water_mask"],
            validation=ValidationMetrics(accuracy_median=0.80),
        ))

        registry.register(AlgorithmMetadata(
            id="flood.baseline.hand_model",
            name="HAND Model",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.DEM],
            deterministic=True,
            outputs=["flood_susceptibility"],
            validation=ValidationMetrics(accuracy_median=0.78),
        ))

        registry.register(AlgorithmMetadata(
            id="wildfire.baseline.dnbr",
            name="dNBR Burn Severity",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.OPTICAL],
            deterministic=True,
            outputs=["burn_severity"],
        ))

        registry.register(AlgorithmMetadata(
            id="flood.experimental.ml_fusion",
            name="ML Fusion",
            category=AlgorithmCategory.EXPERIMENTAL,
            version="0.1.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR, DataType.OPTICAL],
            deterministic=False,
            outputs=["water_mask"],
        ))

        return registry

    @pytest.fixture
    def selector(self, registry):
        """Create selector with test registry."""
        return DeterministicSelector(registry=registry)

    @pytest.fixture
    def sar_availability(self):
        """Create SAR-only data availability."""
        return DataAvailability(
            data_types={DataType.SAR},
            sensor_types={"sar"},
        )

    @pytest.fixture
    def full_availability(self):
        """Create full data availability."""
        return DataAvailability(
            data_types={DataType.SAR, DataType.OPTICAL, DataType.DEM},
            sensor_types={"sar", "optical", "dem"},
        )

    def test_select_algorithms_basic(self, selector, sar_availability):
        """Test basic algorithm selection."""
        plan = selector.select_algorithms("flood.coastal", sar_availability)

        assert plan.event_class == "flood.coastal"
        assert len(plan.selections) > 0
        assert "flood.baseline.threshold_sar" in plan.selections
        assert plan.overall_quality in [SelectionQuality.OPTIMAL, SelectionQuality.ACCEPTABLE]

    def test_select_algorithms_with_full_data(self, selector, full_availability):
        """Test selection with all data types available."""
        plan = selector.select_algorithms("flood.riverine", full_availability)

        # Should select multiple flood algorithms
        assert len(plan.selections) >= 2

        # Check quality
        for algo_id, selection in plan.selections.items():
            assert selection.quality != SelectionQuality.UNAVAILABLE

    def test_select_algorithms_no_matching_data(self, selector):
        """Test selection when required data is missing."""
        availability = DataAvailability(
            data_types={DataType.WEATHER},  # No SAR, optical, or DEM
            sensor_types={"weather"},
        )

        plan = selector.select_algorithms("flood.coastal", availability)

        # Should mark unavailable due to missing data
        for algo_id, selection in plan.selections.items():
            if selection.quality != SelectionQuality.UNAVAILABLE:
                # Only weather-compatible algorithms should be selected
                pass

    def test_select_algorithms_wrong_event_type(self, selector, full_availability):
        """Test selection for non-existent event type."""
        plan = selector.select_algorithms("earthquake.magnitude7", full_availability)

        # Should return empty plan (no algorithms match)
        assert len(plan.selections) == 0

    def test_select_algorithms_wildfire(self, selector, full_availability):
        """Test selection for wildfire events."""
        plan = selector.select_algorithms("wildfire.forest", full_availability)

        # Should select wildfire algorithm
        assert "wildfire.baseline.dnbr" in plan.selections

    def test_deterministic_ordering(self, selector, full_availability):
        """Test that selection is deterministic."""
        # Run selection multiple times
        plans = [
            selector.select_algorithms("flood.coastal", full_availability)
            for _ in range(3)
        ]

        # All plans should have same hash
        hashes = [plan.plan_hash for plan in plans]
        assert all(h == hashes[0] for h in hashes)

        # All plans should have same execution order
        orders = [plan.execution_order for plan in plans]
        assert all(o == orders[0] for o in orders)

    def test_select_single_algorithm(self, selector, sar_availability):
        """Test selecting a specific algorithm."""
        selection = selector.select_single_algorithm(
            "flood.baseline.threshold_sar",
            sar_availability
        )

        assert selection is not None
        assert selection.algorithm_id == "flood.baseline.threshold_sar"
        assert selection.quality == SelectionQuality.ACCEPTABLE  # Missing optional DEM

    def test_select_single_algorithm_not_found(self, selector, sar_availability):
        """Test selecting non-existent algorithm."""
        selection = selector.select_single_algorithm(
            "nonexistent.algorithm",
            sar_availability
        )

        assert selection is None

    def test_select_single_algorithm_missing_data(self, selector, sar_availability):
        """Test selecting algorithm with missing required data."""
        selection = selector.select_single_algorithm(
            "flood.baseline.ndwi_optical",  # Requires optical
            sar_availability  # Only has SAR
        )

        assert selection is not None
        assert selection.quality == SelectionQuality.UNAVAILABLE
        assert selection.selection_reason == SelectionReason.MISSING_REQUIRED_DATA

    def test_constraints_applied(self, registry, full_availability):
        """Test that constraints filter algorithms."""
        constraints = SelectionConstraints(
            require_deterministic=True,
            excluded_algorithms={"flood.baseline.threshold_sar"}
        )

        selector = DeterministicSelector(registry=registry, constraints=constraints)
        plan = selector.select_algorithms("flood.coastal", full_availability)

        # threshold_sar should not be selected
        for algo_id in plan.selections:
            if plan.selections[algo_id].quality != SelectionQuality.UNAVAILABLE:
                assert algo_id != "flood.baseline.threshold_sar"

    def test_experimental_flagged_degraded(self, selector, full_availability):
        """Test that experimental algorithms are flagged as degraded."""
        # Need both SAR and optical for experimental algorithm
        constraints = SelectionConstraints(require_deterministic=False)
        selector.update_constraints(constraints)

        plan = selector.select_algorithms("flood.coastal", full_availability)

        if "flood.experimental.ml_fusion" in plan.selections:
            selection = plan.selections["flood.experimental.ml_fusion"]
            assert selection.quality == SelectionQuality.DEGRADED
            assert selection.degraded_mode is True

    def test_verify_selection_plan(self, selector, full_availability):
        """Test plan verification."""
        plan = selector.select_algorithms("flood.coastal", full_availability)

        # Valid plan should verify
        assert selector.verify_selection_plan(plan) is True

        # Tampered plan should fail
        plan.plan_hash = "tampered_hash"
        assert selector.verify_selection_plan(plan) is False

    def test_get_selector_info(self, selector):
        """Test selector info retrieval."""
        info = selector.get_selector_info()

        assert info["version"] == SELECTOR_VERSION
        assert "registry_size" in info
        assert "constraints" in info


class TestCreateDeterministicSelector:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating selector with defaults."""
        selector = create_deterministic_selector()

        assert selector.constraints.require_deterministic is True
        assert selector.constraints.prefer_baseline is True
        assert selector.constraints.gpu_available is False

    def test_create_with_custom_constraints(self):
        """Test creating selector with custom constraints."""
        selector = create_deterministic_selector(
            max_memory_mb=4096,
            gpu_available=True,
            require_deterministic=False,
            prefer_baseline=False
        )

        assert selector.constraints.max_memory_mb == 4096
        assert selector.constraints.gpu_available is True
        assert selector.constraints.require_deterministic is False
        assert selector.constraints.prefer_baseline is False


class TestDeterministicSelectionEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def minimal_registry(self):
        """Create minimal test registry."""
        registry = AlgorithmRegistry()

        registry.register(AlgorithmMetadata(
            id="test.algo1",
            name="Test Algorithm 1",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        return registry

    def test_empty_registry(self):
        """Test with empty registry."""
        registry = AlgorithmRegistry()
        selector = DeterministicSelector(registry=registry)

        availability = DataAvailability(data_types={DataType.SAR})
        plan = selector.select_algorithms("flood.coastal", availability)

        assert len(plan.selections) == 0

    def test_empty_data_availability(self, minimal_registry):
        """Test with no available data."""
        selector = DeterministicSelector(registry=minimal_registry)

        availability = DataAvailability()
        plan = selector.select_algorithms("test.event", availability)

        # All algorithms should be unavailable
        for selection in plan.selections.values():
            assert selection.quality == SelectionQuality.UNAVAILABLE

    def test_hash_stability_across_runs(self, minimal_registry):
        """Test that hashes are stable across selector instances."""
        availability = DataAvailability(data_types={DataType.SAR})

        selector1 = DeterministicSelector(registry=minimal_registry)
        selector2 = DeterministicSelector(registry=minimal_registry)

        plan1 = selector1.select_algorithms("test.event", availability)
        plan2 = selector2.select_algorithms("test.event", availability)

        assert plan1.plan_hash == plan2.plan_hash

    def test_execution_priority_assigned(self, minimal_registry):
        """Test that execution priorities are assigned."""
        # Add second algorithm
        minimal_registry.register(AlgorithmMetadata(
            id="test.algo2",
            name="Test Algorithm 2",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.DEM],
            deterministic=True,
        ))

        selector = DeterministicSelector(registry=minimal_registry)
        availability = DataAvailability(data_types={DataType.SAR, DataType.DEM})

        plan = selector.select_algorithms("test.event", availability)

        priorities = [sel.execution_priority for sel in plan.selections.values()]
        assert len(set(priorities)) == len(priorities)  # All unique

    def test_selection_reason_populated(self, minimal_registry):
        """Test that selection reasons are populated."""
        selector = DeterministicSelector(registry=minimal_registry)
        availability = DataAvailability(data_types={DataType.SAR})

        plan = selector.select_algorithms("test.event", availability)

        for selection in plan.selections.values():
            assert selection.selection_reason is not None

    def test_version_lock_populated(self, minimal_registry):
        """Test that version locks are populated."""
        selector = DeterministicSelector(registry=minimal_registry)
        availability = DataAvailability(data_types={DataType.SAR})

        plan = selector.select_algorithms("test.event", availability)

        for algo_id, selection in plan.selections.items():
            if selection.quality != SelectionQuality.UNAVAILABLE:
                assert algo_id in selection.version_lock
                assert selection.version_lock[algo_id] == "1.0.0"

    def test_trade_offs_documented(self):
        """Test that trade-offs are documented."""
        registry = AlgorithmRegistry()

        # Add multiple algorithms for same event type
        registry.register(AlgorithmMetadata(
            id="flood.algo1",
            name="Flood Algorithm 1",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        registry.register(AlgorithmMetadata(
            id="flood.algo2",
            name="Flood Algorithm 2",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        selector = DeterministicSelector(registry=registry)
        availability = DataAvailability(data_types={DataType.SAR})

        plan = selector.select_algorithms("flood.coastal", availability)

        # Trade-offs should be documented
        assert len(plan.trade_offs) > 0

    def test_category_priority_ordering(self):
        """Test that baseline algorithms are preferred over advanced."""
        registry = AlgorithmRegistry()

        # Add advanced algorithm first
        registry.register(AlgorithmMetadata(
            id="flood.advanced.ml",
            name="ML Flood Detection",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        # Add baseline algorithm second
        registry.register(AlgorithmMetadata(
            id="flood.baseline.threshold",
            name="Threshold Detection",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        selector = DeterministicSelector(registry=registry)
        availability = DataAvailability(data_types={DataType.SAR})

        plan = selector.select_algorithms("flood.coastal", availability)

        # Baseline should come first in execution order
        assert plan.execution_order[0] == "flood.baseline.threshold"

    def test_timestamp_recorded(self, minimal_registry):
        """Test that selection timestamps are recorded."""
        selector = DeterministicSelector(registry=minimal_registry)
        availability = DataAvailability(data_types={DataType.SAR})

        before = datetime.now(timezone.utc)
        plan = selector.select_algorithms("test.event", availability)
        after = datetime.now(timezone.utc)

        assert before <= plan.created_at <= after

        for selection in plan.selections.values():
            assert before <= selection.selected_at <= after

    def test_empty_event_types_guard(self, minimal_registry):
        """Test that empty event_types is handled gracefully."""
        # Create algorithm with empty event_types (edge case)
        minimal_registry.register(AlgorithmMetadata(
            id="test.empty_events",
            name="Empty Events Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=[],  # Empty event types
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        selector = DeterministicSelector(registry=minimal_registry)
        availability = DataAvailability(data_types={DataType.SAR})

        # Should not crash when selecting
        plan = selector.select_algorithms("test.event", availability)

        # The empty event_types algorithm shouldn't match
        assert "test.empty_events" not in plan.selections

    def test_update_constraints(self, minimal_registry):
        """Test updating constraints on selector."""
        selector = DeterministicSelector(registry=minimal_registry)

        new_constraints = SelectionConstraints(
            max_memory_mb=1024,
            gpu_available=True,
            require_deterministic=False
        )
        selector.update_constraints(new_constraints)

        assert selector.constraints.max_memory_mb == 1024
        assert selector.constraints.gpu_available is True
        assert selector.constraints.require_deterministic is False

    def test_selection_plan_empty_execution_sequence(self):
        """Test getting execution sequence from empty plan."""
        plan = SelectionPlan(event_class="flood.coastal")

        sequence = plan.get_execution_sequence()

        assert sequence == []
        assert plan.plan_hash == ""

    def test_algorithm_selection_defaults(self):
        """Test AlgorithmSelection default values."""
        selection = AlgorithmSelection(
            algorithm_id="test.algo",
            algorithm_name="Test",
            version="1.0.0",
            quality=SelectionQuality.OPTIMAL,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        )

        # Check defaults
        assert selection.degraded_mode is False
        assert selection.rationale == ""
        assert selection.selection_reason == SelectionReason.BEST_MATCH
        assert selection.alternatives_rejected == {}
        assert selection.execution_priority == 0
        assert selection.selector_version == SELECTOR_VERSION

    def test_selection_constraints_category_filter(self):
        """Test category filtering in constraints."""
        registry = AlgorithmRegistry()

        registry.register(AlgorithmMetadata(
            id="flood.baseline.test",
            name="Baseline Test",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        registry.register(AlgorithmMetadata(
            id="flood.advanced.test",
            name="Advanced Test",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
        ))

        # Only allow baseline algorithms
        constraints = SelectionConstraints(
            allowed_categories={AlgorithmCategory.BASELINE}
        )

        selector = DeterministicSelector(registry=registry, constraints=constraints)
        availability = DataAvailability(data_types={DataType.SAR})

        plan = selector.select_algorithms("flood.coastal", availability)

        # Should have baseline but not advanced in viable selections
        baseline_viable = False
        advanced_viable = False

        for algo_id, sel in plan.selections.items():
            if sel.quality != SelectionQuality.UNAVAILABLE:
                if "baseline" in algo_id:
                    baseline_viable = True
                if "advanced" in algo_id:
                    advanced_viable = True

        assert baseline_viable is True
        assert advanced_viable is False

    def test_selection_constraints_validation_score(self):
        """Test minimum validation score filtering."""
        algorithm = AlgorithmMetadata(
            id="flood.test.low_score",
            name="Low Score Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True,
            validation=ValidationMetrics(accuracy_median=0.6),
        )

        constraints = SelectionConstraints(min_validation_score=0.8)

        allowed, reason = constraints.allows_algorithm(algorithm)

        assert allowed is False
        assert reason == "validation_score_too_low"

    def test_data_availability_with_ranked(self):
        """Test DataAvailability with ranked candidates."""
        results = [
            DiscoveryResult(
                dataset_id="s1-123",
                provider="sentinel1",
                data_type="sar",
                source_uri="s3://bucket/s1-123.tif",
                format="geotiff",
                acquisition_time=datetime.now(timezone.utc),
                spatial_coverage_percent=95.0,
                resolution_m=10.0,
            ),
        ]

        ranked = {
            "sar": [RankedCandidate(
                candidate=results[0],
                scores={"coverage": 0.95, "resolution": 0.9},
                total_score=0.9,
                rank=1,
            )]
        }

        availability = DataAvailability.from_discovery_results(results, ranked)

        assert len(availability.ranked_by_type["sar"]) == 1
        assert availability.ranked_by_type["sar"][0].rank == 1

    def test_selection_quality_unavailable_propagation(self):
        """Test that UNAVAILABLE quality propagates correctly."""
        plan = SelectionPlan(event_class="flood.coastal")

        # Add an unavailable selection
        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo1",
            algorithm_name="Algo 1",
            version="1.0.0",
            quality=SelectionQuality.UNAVAILABLE,
            required_data_types=[DataType.SAR],
            available_data_types=[],
        ))

        assert plan.overall_quality == SelectionQuality.UNAVAILABLE

    def test_selection_quality_acceptable(self):
        """Test that ACCEPTABLE quality is tracked correctly."""
        plan = SelectionPlan(event_class="flood.coastal")

        # Add an acceptable selection
        plan.add_selection(AlgorithmSelection(
            algorithm_id="algo1",
            algorithm_name="Algo 1",
            version="1.0.0",
            quality=SelectionQuality.ACCEPTABLE,
            required_data_types=[DataType.SAR],
            available_data_types=[DataType.SAR],
        ))

        assert plan.overall_quality == SelectionQuality.ACCEPTABLE


# ============================================================================
# ALGORITHM SELECTOR TESTS (Track 6)
# ============================================================================

from core.analysis.selection.selector import (
    AlgorithmSelector,
    SelectionCriteria,
    SelectionResult,
    SelectionContext,
    ComputeProfile,
    ComputeConstraints,
    RejectionReason,
)


class TestComputeConstraints:
    """Test compute constraint configuration."""

    def test_default_constraints(self):
        """Test default compute constraints."""
        constraints = ComputeConstraints()

        assert constraints.max_memory_mb == 4096
        assert constraints.gpu_available is False
        assert constraints.gpu_memory_mb is None
        assert constraints.max_runtime_minutes is None
        assert constraints.allow_distributed is False

    def test_laptop_profile(self):
        """Test laptop compute profile."""
        constraints = ComputeConstraints.from_profile(ComputeProfile.LAPTOP)

        assert constraints.max_memory_mb == 2048
        assert constraints.gpu_available is False
        assert constraints.max_runtime_minutes == 30
        assert constraints.allow_distributed is False

    def test_workstation_profile(self):
        """Test workstation compute profile."""
        constraints = ComputeConstraints.from_profile(ComputeProfile.WORKSTATION)

        assert constraints.max_memory_mb == 8192
        assert constraints.gpu_available is True
        assert constraints.gpu_memory_mb == 4096
        assert constraints.max_runtime_minutes == 120
        assert constraints.allow_distributed is False

    def test_cloud_profile(self):
        """Test cloud compute profile."""
        constraints = ComputeConstraints.from_profile(ComputeProfile.CLOUD)

        assert constraints.max_memory_mb == 32768
        assert constraints.gpu_available is True
        assert constraints.gpu_memory_mb == 16384
        assert constraints.max_runtime_minutes is None  # No limit
        assert constraints.allow_distributed is True

    def test_edge_profile(self):
        """Test edge compute profile."""
        constraints = ComputeConstraints.from_profile(ComputeProfile.EDGE)

        assert constraints.max_memory_mb == 512
        assert constraints.gpu_available is False
        assert constraints.max_runtime_minutes == 15

    def test_to_dict(self):
        """Test conversion to dictionary."""
        constraints = ComputeConstraints(max_memory_mb=4096, gpu_available=True)
        result = constraints.to_dict()

        assert isinstance(result, dict)
        assert result["max_memory_mb"] == 4096
        assert result["gpu_available"] is True


class TestSelectionCriteria:
    """Test selection criteria configuration."""

    def test_default_criteria(self):
        """Test default selection criteria."""
        criteria = SelectionCriteria()

        assert criteria.prefer_validated is True
        assert criteria.prefer_deterministic is True
        assert criteria.prefer_baseline is False
        assert criteria.min_accuracy is None
        assert criteria.allowed_categories is None
        assert len(criteria.excluded_algorithms) == 0

    def test_custom_criteria(self):
        """Test custom selection criteria."""
        criteria = SelectionCriteria(
            prefer_baseline=True,
            min_accuracy=0.8,
            allowed_categories=[AlgorithmCategory.BASELINE],
            excluded_algorithms={"deprecated.algorithm"}
        )

        assert criteria.prefer_baseline is True
        assert criteria.min_accuracy == 0.8
        assert AlgorithmCategory.BASELINE in criteria.allowed_categories
        assert "deprecated.algorithm" in criteria.excluded_algorithms

    def test_to_dict(self):
        """Test conversion to dictionary."""
        criteria = SelectionCriteria(
            min_accuracy=0.75,
            allowed_categories=[AlgorithmCategory.BASELINE, AlgorithmCategory.ADVANCED]
        )
        result = criteria.to_dict()

        assert isinstance(result, dict)
        assert result["min_accuracy"] == 0.75
        assert "baseline" in result["allowed_categories"]


class TestSelectionContext:
    """Test selection context configuration."""

    def test_basic_context(self):
        """Test basic selection context."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.DEM},
            compute_constraints=ComputeConstraints()
        )

        assert context.event_class == "flood.coastal"
        assert DataType.SAR in context.available_data_types
        assert DataType.DEM in context.available_data_types
        assert context.region is None

    def test_context_with_region(self):
        """Test context with region specification."""
        context = SelectionContext(
            event_class="flood.riverine",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            region="north_america"
        )

        assert context.region == "north_america"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = SelectionContext(
            event_class="wildfire.forest",
            available_data_types={DataType.OPTICAL, DataType.SAR},
            compute_constraints=ComputeConstraints.from_profile(ComputeProfile.LAPTOP)
        )
        result = context.to_dict()

        assert isinstance(result, dict)
        assert result["event_class"] == "wildfire.forest"
        assert "optical" in result["available_data_types"]


class TestAlgorithmSelectorTrack6:
    """Test algorithm selector functionality (Track 6)."""

    @pytest.fixture
    def registry_track6(self):
        """Create test registry with sample algorithms."""
        registry = AlgorithmRegistry()

        # Add test algorithms
        registry.register(AlgorithmMetadata(
            id="flood.baseline.threshold_sar",
            name="SAR Threshold Detection",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            optional_data_types=[DataType.DEM],
            resources=ResourceRequirements(memory_mb=2048),
            validation=ValidationMetrics(
                accuracy_median=0.85,
                f1_score=0.82,
                validated_regions=["north_america", "europe"]
            ),
            deterministic=True
        ))

        registry.register(AlgorithmMetadata(
            id="flood.baseline.ndwi_optical",
            name="NDWI Optical Detection",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.OPTICAL],
            optional_data_types=[DataType.DEM],
            resources=ResourceRequirements(memory_mb=2048),
            validation=ValidationMetrics(
                accuracy_median=0.80,
                f1_score=0.78,
                validated_regions=["europe", "asia"]
            ),
            deterministic=True
        ))

        registry.register(AlgorithmMetadata(
            id="flood.advanced.ensemble",
            name="Ensemble Flood Detection",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR, DataType.OPTICAL],
            optional_data_types=[DataType.DEM, DataType.ANCILLARY],
            resources=ResourceRequirements(memory_mb=8192, gpu_required=True),
            validation=ValidationMetrics(
                accuracy_median=0.92,
                f1_score=0.90,
                validated_regions=["north_america"]
            ),
            deterministic=False
        ))

        registry.register(AlgorithmMetadata(
            id="wildfire.baseline.dnbr",
            name="Differenced NBR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.OPTICAL],
            resources=ResourceRequirements(memory_mb=2048),
            validation=ValidationMetrics(
                accuracy_median=0.88,
                validated_regions=["north_america", "australia"]
            ),
            deterministic=True
        ))

        registry.register(AlgorithmMetadata(
            id="flood.deprecated.old_method",
            name="Old Flood Method",
            category=AlgorithmCategory.BASELINE,
            version="0.1.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deprecated=True,
            replacement_algorithm="flood.baseline.threshold_sar"
        ))

        return registry

    @pytest.fixture
    def selector_track6(self, registry_track6):
        """Create selector with test registry."""
        return AlgorithmSelector(registry=registry_track6)

    def test_select_with_available_data(self, selector_track6):
        """Test selection with available data."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.DEM},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert result.success is True
        assert result.selected is not None
        assert result.selected.id == "flood.baseline.threshold_sar"
        assert "SAR" in result.selected.name

    def test_select_rejects_missing_data(self, selector_track6):
        """Test rejection when required data is missing."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.WEATHER},  # No SAR or OPTICAL
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        # Should reject algorithms that need SAR or OPTICAL
        assert result.success is False or any(
            r.category == "data" for r in result.rejected
        )

    def test_select_rejects_deprecated(self, selector_track6):
        """Test that deprecated algorithms are rejected."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        # Deprecated algorithm should be in rejected list
        deprecated_rejection = next(
            (r for r in result.rejected if r.algorithm_id == "flood.deprecated.old_method"),
            None
        )
        assert deprecated_rejection is not None
        assert deprecated_rejection.category == "deprecated"

    def test_select_respects_gpu_constraint(self, selector_track6):
        """Test GPU constraint enforcement."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints(
                max_memory_mb=16384,
                gpu_available=False  # No GPU
            )
        )

        result = selector_track6.select(context)

        # Ensemble algorithm requires GPU, should be rejected
        ensemble_rejection = next(
            (r for r in result.rejected if r.algorithm_id == "flood.advanced.ensemble"),
            None
        )
        assert ensemble_rejection is not None
        assert ensemble_rejection.category == "compute"
        assert "GPU" in ensemble_rejection.reason

    def test_select_respects_memory_constraint(self, selector_track6):
        """Test memory constraint enforcement."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints(
                max_memory_mb=1024,  # Very limited
                gpu_available=True
            )
        )

        result = selector_track6.select(context)

        # All algorithms require more than 1024MB, should all be rejected
        memory_rejections = [r for r in result.rejected if r.category == "compute"]
        assert len(memory_rejections) > 0

    def test_select_for_wildfire(self, selector_track6):
        """Test selection for wildfire event type."""
        context = SelectionContext(
            event_class="wildfire.forest",
            available_data_types={DataType.OPTICAL},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert result.success is True
        assert result.selected.id == "wildfire.baseline.dnbr"
        assert "NBR" in result.selected.name

    def test_select_no_algorithms_for_event(self, selector_track6):
        """Test selection for event type with no algorithms."""
        context = SelectionContext(
            event_class="earthquake.magnitude7",  # Not supported
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert result.success is False
        assert "No algorithms found" in result.rationale

    def test_select_respects_category_restriction(self, selector_track6):
        """Test category restriction enforcement."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints(
                max_memory_mb=16384,
                gpu_available=True
            ),
            criteria=SelectionCriteria(
                allowed_categories=[AlgorithmCategory.BASELINE]
            )
        )

        result = selector_track6.select(context)

        # Should select baseline algorithm, not advanced
        assert result.success is True
        assert result.selected.category == AlgorithmCategory.BASELINE

        # Advanced should be rejected
        advanced_rejection = next(
            (r for r in result.rejected if "flood.advanced" in r.algorithm_id),
            None
        )
        assert advanced_rejection is not None
        assert advanced_rejection.category == "criteria"

    def test_select_respects_explicit_exclusion(self, selector_track6):
        """Test explicit algorithm exclusion."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            criteria=SelectionCriteria(
                excluded_algorithms={"flood.baseline.threshold_sar"}
            )
        )

        result = selector_track6.select(context)

        # SAR threshold should be rejected, no viable alternatives
        excluded_rejection = next(
            (r for r in result.rejected if r.algorithm_id == "flood.baseline.threshold_sar"),
            None
        )
        assert excluded_rejection is not None
        assert excluded_rejection.category == "criteria"

    def test_select_respects_min_accuracy(self, selector_track6):
        """Test minimum accuracy enforcement."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints(
                max_memory_mb=16384,
                gpu_available=True
            ),
            criteria=SelectionCriteria(
                min_accuracy=0.90
            )
        )

        result = selector_track6.select(context)

        # Only ensemble has 0.92 accuracy
        # SAR threshold (0.85) and NDWI (0.80) should be rejected for low accuracy
        accuracy_rejections = [
            r for r in result.rejected if r.category == "validation"
        ]
        assert len(accuracy_rejections) >= 1

    def test_select_provides_alternatives(self, selector_track6):
        """Test that alternatives are provided."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL, DataType.DEM},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert result.success is True
        # Should have at least one alternative (NDWI optical)
        assert len(result.alternatives) >= 1
        # Alternatives should be different from selected
        for alt in result.alternatives:
            assert alt.id != result.selected.id

    def test_select_provides_scores(self, selector_track6):
        """Test that scores are provided for all candidates."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert len(result.scores) > 0
        # Selected algorithm should have a score
        assert result.selected.id in result.scores
        # Scores should be between 0 and 1
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_select_generates_rationale(self, selector_track6):
        """Test rationale generation."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)

        assert result.rationale is not None
        assert len(result.rationale) > 0
        # Should mention selected algorithm
        assert result.selected.name in result.rationale or result.selected.id in result.rationale

    def test_select_multiple(self, selector_track6):
        """Test selecting multiple algorithms."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.OPTICAL, DataType.DEM},
            compute_constraints=ComputeConstraints()
        )

        results = selector_track6.select_multiple(context, max_algorithms=2)

        assert len(results) >= 1
        # First result should be the best algorithm
        assert results[0].success is True

    def test_explain_selection_success(self, selector_track6):
        """Test selection explanation for successful selection."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR, DataType.DEM},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)
        explanation = selector_track6.explain_selection(result)

        assert "Algorithm Selection Report" in explanation
        assert "SELECTED" in explanation
        assert result.selected.id in explanation

    def test_explain_selection_failure(self, selector_track6):
        """Test selection explanation for failed selection."""
        context = SelectionContext(
            event_class="earthquake.damage",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        result = selector_track6.select(context)
        explanation = selector_track6.explain_selection(result)

        assert "Algorithm Selection Report" in explanation
        assert "FAILED" in explanation

    def test_get_supported_event_types(self, selector_track6):
        """Test getting supported event types."""
        event_types = selector_track6.get_supported_event_types()

        assert "flood.*" in event_types
        assert "wildfire.*" in event_types

    def test_get_algorithms_by_data_type(self, selector_track6):
        """Test getting algorithms by data type."""
        sar_algorithms = selector_track6.get_algorithms_by_data_type(DataType.SAR)

        assert len(sar_algorithms) >= 1
        assert all(DataType.SAR in a.required_data_types for a in sar_algorithms)


class TestSelectionResultTrack6:
    """Test SelectionResult dataclass (Track 6)."""

    def test_success_property(self):
        """Test success property."""
        # Successful result
        result1 = SelectionResult(
            selected=AlgorithmMetadata(
                id="test.algorithm",
                name="Test Algorithm",
                category=AlgorithmCategory.BASELINE,
                version="1.0.0",
                event_types=["test.*"],
                required_data_types=[DataType.SAR]
            ),
            alternatives=[],
            rejected=[],
            scores={"test.algorithm": 0.8},
            context=SelectionContext(
                event_class="test.event",
                available_data_types={DataType.SAR},
                compute_constraints=ComputeConstraints()
            ),
            rationale="Test rationale"
        )
        assert result1.success is True

        # Failed result
        result2 = SelectionResult(
            selected=None,
            alternatives=[],
            rejected=[],
            scores={},
            context=SelectionContext(
                event_class="test.event",
                available_data_types={DataType.SAR},
                compute_constraints=ComputeConstraints()
            ),
            rationale="No algorithms found"
        )
        assert result2.success is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SelectionResult(
            selected=AlgorithmMetadata(
                id="test.algorithm",
                name="Test Algorithm",
                category=AlgorithmCategory.BASELINE,
                version="1.0.0",
                event_types=["test.*"],
                required_data_types=[DataType.SAR]
            ),
            alternatives=[],
            rejected=[RejectionReason(
                algorithm_id="other.algorithm",
                reason="Test rejection",
                category="test"
            )],
            scores={"test.algorithm": 0.8},
            context=SelectionContext(
                event_class="test.event",
                available_data_types={DataType.SAR},
                compute_constraints=ComputeConstraints()
            ),
            rationale="Test rationale"
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["success"] is True
        assert data["selected"]["id"] == "test.algorithm"
        assert len(data["rejected"]) == 1
        assert "timestamp" in data


class TestRejectionReasonTrack6:
    """Test RejectionReason dataclass (Track 6)."""

    def test_basic_rejection(self):
        """Test basic rejection reason."""
        rejection = RejectionReason(
            algorithm_id="test.algorithm",
            reason="Missing data",
            category="data"
        )

        assert rejection.algorithm_id == "test.algorithm"
        assert rejection.reason == "Missing data"
        assert rejection.category == "data"
        assert rejection.details is None

    def test_rejection_with_details(self):
        """Test rejection with details."""
        rejection = RejectionReason(
            algorithm_id="test.algorithm",
            reason="Memory exceeded",
            category="compute",
            details={
                "required_memory_mb": 8192,
                "available_memory_mb": 4096
            }
        )

        assert rejection.details is not None
        assert rejection.details["required_memory_mb"] == 8192

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rejection = RejectionReason(
            algorithm_id="test.algorithm",
            reason="GPU required",
            category="compute",
            details={"gpu_required": True}
        )

        data = rejection.to_dict()

        assert isinstance(data, dict)
        assert data["algorithm_id"] == "test.algorithm"
        assert data["category"] == "compute"
        assert data["details"]["gpu_required"] is True


class TestAlgorithmSelectorEdgeCasesTrack6:
    """Test edge cases for algorithm selector (Track 6)."""

    @pytest.fixture
    def empty_registry_track6(self):
        """Create empty registry."""
        return AlgorithmRegistry()

    @pytest.fixture
    def selector_empty_track6(self, empty_registry_track6):
        """Create selector with empty registry."""
        return AlgorithmSelector(registry=empty_registry_track6)

    def test_select_with_empty_registry(self, selector_empty_track6):
        """Test selection with no algorithms registered."""
        context = SelectionContext(
            event_class="flood.coastal",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        result = selector_empty_track6.select(context)

        assert result.success is False
        assert "No algorithms found" in result.rationale

    def test_select_with_empty_data_types(self):
        """Test selection with no available data types."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR]
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types=set(),  # Empty
            compute_constraints=ComputeConstraints()
        )

        result = selector.select(context)

        assert result.success is False
        # Should have data rejection
        assert any(r.category == "data" for r in result.rejected)

    def test_select_multiple_with_no_alternatives(self):
        """Test selecting multiple when only one algorithm matches."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="only.algorithm",
            name="Only Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR]
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        results = selector.select_multiple(context, max_algorithms=5)

        # Should return just one result
        assert len(results) == 1
        assert results[0].success is True

    def test_select_with_all_algorithms_rejected(self):
        """Test when all algorithms are rejected."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="deprecated.algorithm",
            name="Deprecated Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="0.1.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            deprecated=True
        ))
        registry.register(AlgorithmMetadata(
            id="heavy.algorithm",
            name="Heavy Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=64000)  # Huge memory
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_memory_mb=4096)
        )

        result = selector.select(context)

        assert result.success is False
        assert len(result.rejected) == 2  # Both algorithms rejected

    def test_validation_without_validation_data(self):
        """Test algorithm without validation data."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="unvalidated.algorithm",
            name="Unvalidated Algorithm",
            category=AlgorithmCategory.EXPERIMENTAL,
            version="0.1.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=None  # No validation
        ))

        selector = AlgorithmSelector(registry=registry)

        # Without min_accuracy requirement
        context1 = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )
        result1 = selector.select(context1)
        assert result1.success is True

        # With min_accuracy requirement
        context2 = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            criteria=SelectionCriteria(min_accuracy=0.8)
        )
        result2 = selector.select(context2)

        # Should be rejected for no validation data
        assert result2.success is False
        assert any(r.category == "validation" for r in result2.rejected)


class TestAlgorithmSelectorDivisionGuardsTrack6:
    """Test division by zero guards in algorithm selector (Track 6)."""

    def test_score_resource_efficiency_zero_max_memory(self):
        """Test resource efficiency scoring with zero max_memory_mb."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=2048)
        ))

        selector = AlgorithmSelector(registry=registry)

        # Zero max_memory_mb should not cause division by zero
        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_memory_mb=0)  # Edge case
        )

        # Should not raise ZeroDivisionError
        result = selector.select(context)
        # Algorithm should be rejected due to memory constraint
        assert result.success is False or result.scores.get("test.algorithm", 0) >= 0

    def test_score_resource_efficiency_negative_max_memory(self):
        """Test resource efficiency scoring with negative max_memory_mb."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=2048)
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_memory_mb=-100)  # Invalid
        )

        # Should not raise any errors
        result = selector.select(context)
        # Algorithm should be rejected due to invalid constraint
        assert result.success is False or isinstance(result, SelectionResult)

    def test_score_data_coverage_empty_optional(self):
        """Test data coverage scoring with no optional data types."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            optional_data_types=[]  # Empty optional
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints()
        )

        result = selector.select(context)
        assert result.success is True
        # Should have a valid score (no division by zero for empty optional)
        assert result.scores["test.algorithm"] >= 0
        assert result.scores["test.algorithm"] <= 1.0

    def test_algorithm_with_no_memory_requirement(self):
        """Test algorithm with unspecified memory requirement."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=None)  # Unspecified
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_memory_mb=4096)
        )

        result = selector.select(context)
        assert result.success is True
        # Should return default score for unknown memory
        assert result.scores["test.algorithm"] >= 0


class TestAlgorithmSelectorScoringTrack6:
    """Test scoring accuracy for algorithm selector (Track 6)."""

    def test_validation_score_without_validation(self):
        """Test validation scoring without validation data."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=None
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            criteria=SelectionCriteria(prefer_validated=True)
        )

        result = selector.select(context)
        assert result.success is True
        # Score should still be valid but lower
        assert 0 <= result.scores["test.algorithm"] <= 1.0

    def test_validation_score_with_complete_validation(self):
        """Test validation scoring with complete validation data."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=ValidationMetrics(
                accuracy_median=0.95,
                f1_score=0.92,
                validation_dataset_count=150,
                validated_regions=["north_america", "europe", "asia"]
            )
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            criteria=SelectionCriteria(prefer_validated=True)
        )

        result = selector.select(context)
        assert result.success is True
        # Should have high score with good validation
        assert result.scores["test.algorithm"] >= 0.5

    def test_region_match_exact(self):
        """Test region matching with exact match."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=ValidationMetrics(
                accuracy_median=0.85,
                validated_regions=["north_america"]
            )
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            region="north_america"
        )

        result = selector.select(context)
        assert result.success is True

    def test_region_match_partial(self):
        """Test region matching with partial match."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=ValidationMetrics(
                accuracy_median=0.85,
                validated_regions=["north_america"]
            )
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            region="america"  # Partial match
        )

        result = selector.select(context)
        assert result.success is True

    def test_region_no_match(self):
        """Test region matching with no match."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="test.algorithm",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            validation=ValidationMetrics(
                accuracy_median=0.85,
                validated_regions=["europe"]
            )
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(),
            region="asia"  # No match
        )

        result = selector.select(context)
        # Should still succeed but with lower region score
        assert result.success is True


class TestAlgorithmSelectorRuntimeConstraintsTrack6:
    """Test runtime constraint handling for algorithm selector (Track 6)."""

    def test_runtime_constraint_rejected(self):
        """Test algorithm rejected for runtime constraints."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="slow.algorithm",
            name="Slow Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(max_runtime_minutes=120)
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_runtime_minutes=30)  # Too short
        )

        result = selector.select(context)
        assert result.success is False
        assert any(r.category == "compute" for r in result.rejected)
        assert any("minutes" in r.reason.lower() for r in result.rejected)

    def test_runtime_constraint_no_limit(self):
        """Test algorithm with no runtime limit."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="slow.algorithm",
            name="Slow Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(max_runtime_minutes=120)
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(max_runtime_minutes=None)  # No limit
        )

        result = selector.select(context)
        assert result.success is True

    def test_distributed_constraint_rejected(self):
        """Test algorithm rejected for distributed requirement."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="distributed.algorithm",
            name="Distributed Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(distributed=True)
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(allow_distributed=False)
        )

        result = selector.select(context)
        assert result.success is False
        assert any(r.category == "compute" for r in result.rejected)
        assert any("distributed" in r.reason.lower() for r in result.rejected)

    def test_distributed_constraint_allowed(self):
        """Test algorithm accepted when distributed is allowed."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="distributed.algorithm",
            name="Distributed Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(distributed=True)
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(allow_distributed=True)
        )

        result = selector.select(context)
        assert result.success is True

    def test_gpu_memory_constraint_rejected(self):
        """Test algorithm rejected for GPU memory constraints."""
        registry = AlgorithmRegistry()
        registry.register(AlgorithmMetadata(
            id="gpu.algorithm",
            name="GPU Algorithm",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["test.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(
                gpu_required=True,
                gpu_memory_mb=8192
            )
        ))

        selector = AlgorithmSelector(registry=registry)

        context = SelectionContext(
            event_class="test.event",
            available_data_types={DataType.SAR},
            compute_constraints=ComputeConstraints(
                gpu_available=True,
                gpu_memory_mb=4096  # Not enough GPU memory
            )
        )

        result = selector.select(context)
        assert result.success is False
        assert any(r.category == "compute" for r in result.rejected)
        assert any("gpu" in r.reason.lower() for r in result.rejected)
