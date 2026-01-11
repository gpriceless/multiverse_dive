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
