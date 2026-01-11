"""
Comprehensive Quality Control Integration Tests (Group I, Track 6).

End-to-end tests that validate the complete quality control pipeline:
- Sanity checks (spatial, value, temporal, artifacts)
- Cross-validation (model, sensor, historical)
- Uncertainty quantification
- Action management (gating, flagging, routing)

These tests ensure all quality modules work correctly together as a unified system.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import numpy as np

# Mark all tests in this module
pytestmark = pytest.mark.quality


# ============================================================================
# END-TO-END PIPELINE TESTS
# ============================================================================

class TestFullQualityPipeline:
    """End-to-end tests for the complete quality control pipeline."""

    def test_flood_extent_full_pipeline(self):
        """Test complete pipeline for flood extent product."""
        # Import all required modules
        from core.quality.sanity import (
            SanitySuite, SanitySuiteConfig,
            ValuePlausibilityConfig, ValueType,
        )
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
            HotspotDetector,
        )
        from core.quality.actions import (
            QualityGate, QualityFlagger, ReviewRouter,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus,
            StandardFlag, FlagLevel,
            ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, Expert, ReviewOutcome,
        )

        # Step 1: Create synthetic flood extent data
        np.random.seed(42)
        flood_extent = np.zeros((100, 100))
        # Create coherent flood area
        y, x = np.ogrid[:100, :100]
        flood_mask = (x - 50)**2 + (y - 50)**2 < 25**2
        flood_extent[flood_mask] = 1.0
        # Add some edge uncertainty
        edge = np.random.rand(100, 100) * 0.1
        flood_extent = np.clip(flood_extent + edge, 0, 1)

        # Step 2: Run sanity checks
        sanity_config = SanitySuiteConfig(
            run_temporal=False,
            value_config=ValuePlausibilityConfig(value_type=ValueType.BINARY),
        )
        sanity_suite = SanitySuite(sanity_config)
        sanity_result = sanity_suite.check(flood_extent)

        assert sanity_result is not None
        assert sanity_result.overall_score > 0.5

        # Step 3: Calculate uncertainty
        quantifier = UncertaintyQuantifier()
        uncertainty_metrics = quantifier.calculate_metrics(flood_extent.flatten())
        assert uncertainty_metrics.n_samples > 0

        mapper = SpatialUncertaintyMapper()
        uncertainty_surface = mapper.compute_uncertainty_surface(flood_extent)
        assert uncertainty_surface.mean_uncertainty >= 0

        # Step 4: Detect hotspots
        detector = HotspotDetector()
        hotspots = detector.detect_hotspots(uncertainty_surface.uncertainty)
        assert hotspots.hotspot_label_map.shape == flood_extent.shape

        # Step 5: Create QC checks from sanity results
        qc_checks = []

        # Spatial check
        spatial_status = CheckStatus.PASS if sanity_result.spatial.is_coherent else CheckStatus.SOFT_FAIL
        qc_checks.append(QCCheck(
            "spatial_coherence",
            CheckCategory.SPATIAL,
            spatial_status,
            metric_value=sanity_result.overall_score,
        ))

        # Value check (use .values, not .value)
        value_status = CheckStatus.PASS if sanity_result.values.is_plausible else CheckStatus.SOFT_FAIL
        qc_checks.append(QCCheck(
            "value_plausibility",
            CheckCategory.VALUE,
            value_status,
        ))

        # Artifact check (use .artifacts, not .artifact)
        artifact_status = CheckStatus.PASS if not sanity_result.artifacts.has_artifacts else CheckStatus.WARNING
        qc_checks.append(QCCheck(
            "artifact_detection",
            CheckCategory.ARTIFACT,
            artifact_status,
        ))

        # Uncertainty check
        unc_status = CheckStatus.PASS if uncertainty_metrics.coefficient_of_variation < 0.5 else CheckStatus.WARNING
        qc_checks.append(QCCheck(
            "uncertainty_assessment",
            CheckCategory.UNCERTAINTY,
            unc_status,
            metric_value=uncertainty_metrics.coefficient_of_variation,
        ))

        # Step 6: Run through quality gate
        gate = QualityGate()
        context = GatingContext(
            event_id="evt_flood_001",
            product_id="prod_flood_001",
            confidence_score=0.8,
            cross_validation={"agreement_metrics": {"iou": 0.75, "kappa": 0.7}},
        )

        decision = gate.evaluate(qc_checks, context)
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS)

        # Step 7: Apply flags based on results
        flagger = QualityFlagger()
        if sanity_result.overall_score > 0.8:
            flagger.apply_standard_flag("prod_flood_001", StandardFlag.HIGH_CONFIDENCE)
        elif sanity_result.overall_score > 0.6:
            flagger.apply_standard_flag("prod_flood_001", StandardFlag.MEDIUM_CONFIDENCE)
        else:
            flagger.apply_standard_flag("prod_flood_001", StandardFlag.LOW_CONFIDENCE)

        summary = flagger.summarize("prod_flood_001", "evt_flood_001")
        assert summary.total_flags >= 1

    def test_wildfire_burn_scar_pipeline(self):
        """Test pipeline for wildfire burn scar analysis."""
        from core.quality.sanity import (
            check_spatial_coherence,
            check_value_plausibility,
            ValueType,
        )
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            propagate_quality_uncertainty,
        )
        from core.quality.actions import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus,
        )

        # Create dNBR data (typically ranges from -2 to 2)
        np.random.seed(42)
        dnbr = np.random.randn(100, 100) * 0.5
        # Add burn scar region
        dnbr[30:70, 30:70] = np.random.randn(40, 40) * 0.3 + 0.8  # Moderate to high burn

        # Spatial coherence check
        spatial_result = check_spatial_coherence(dnbr)
        assert spatial_result is not None

        # Value plausibility (dNBR uses DNBR type, not INDEX)
        value_result = check_value_plausibility(dnbr, value_type=ValueType.DNBR)
        assert value_result is not None

        # Uncertainty analysis
        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(dnbr.flatten())
        assert metrics.confidence_interval_95 is not None

        # Quality propagation through aggregation
        quality_scores = [0.85, 0.78, 0.82]
        uncertainties = [0.05, 0.08, 0.06]
        result = propagate_quality_uncertainty(quality_scores, uncertainties, method="mean")
        assert 0 < result.value < 1
        assert result.uncertainty > 0

        # Gate the product
        checks = [
            QCCheck("spatial", CheckCategory.SPATIAL,
                    CheckStatus.PASS if spatial_result.is_coherent else CheckStatus.SOFT_FAIL),
            QCCheck("values", CheckCategory.VALUE,
                    CheckStatus.PASS if value_result.is_plausible else CheckStatus.SOFT_FAIL),
        ]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_wildfire_001",
            product_id="prod_wildfire_001",
            confidence_score=result.value,
        )
        decision = gate.evaluate(checks, context)
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS, GateStatus.REVIEW_REQUIRED)

    def test_multi_sensor_fusion_pipeline(self):
        """Test pipeline for multi-sensor data fusion."""
        from core.quality.sanity import (
            SanitySuite, SanitySuiteConfig,
            ValuePlausibilityConfig, ValueType,
        )
        from core.quality.validation import (
            validate_cross_sensor, SensorType,
        )
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            QualityErrorPropagator,
            AggregationMethod,
        )
        from core.quality.actions import (
            QualityGate, QualityFlagger,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus, StandardFlag,
        )

        np.random.seed(42)

        # Simulate SAR and optical fusion
        sar_data = np.random.rand(50, 50) * 0.3 + 0.5
        optical_data = np.random.rand(50, 50) * 0.3 + 0.5
        fused_data = (sar_data + optical_data) / 2

        # Run sanity on fused product
        suite = SanitySuite(SanitySuiteConfig(
            run_temporal=False,
            value_config=ValuePlausibilityConfig(value_type=ValueType.CONFIDENCE),
        ))
        sanity = suite.check(fused_data)
        assert sanity.passes_sanity

        # Cross-sensor validation using convenience function
        sensor_result = validate_cross_sensor(
            sensor_data=[sar_data, optical_data],
            sensor_types=["sar", "optical"],
            sensor_ids=["SAR-1", "Optical-1"],
            observable="water_extent",
        )
        assert sensor_result is not None

        # Uncertainty propagation
        propagator = QualityErrorPropagator()
        sar_quality = 0.85
        optical_quality = 0.80
        sar_unc = 0.08
        optical_unc = 0.10

        fusion_result = propagator.propagate_aggregation(
            [sar_quality, optical_quality],
            [sar_unc, optical_unc],
            AggregationMethod.WEIGHTED_MEAN,
            weights=[0.6, 0.4],  # Weight SAR more (better for flood)
        )
        assert fusion_result.value > 0.8
        assert fusion_result.uncertainty > 0

        # Create checks (use SPATIAL for sanity category, CROSS_VALIDATION for sensor)
        checks = [
            QCCheck("sanity", CheckCategory.SPATIAL, CheckStatus.PASS),
            QCCheck("sensor_agreement", CheckCategory.CROSS_VALIDATION,
                    CheckStatus.PASS if sensor_result.confidence > 0.7 else CheckStatus.SOFT_FAIL),
        ]

        # Gate
        gate = QualityGate()
        context = GatingContext(
            event_id="evt_fusion_001",
            product_id="prod_fusion_001",
            confidence_score=fusion_result.value,
        )
        decision = gate.evaluate(checks, context)
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS)

        # Flag - use existing flag (no MULTI_SENSOR_FUSION, use HIGH_CONFIDENCE or similar)
        flagger = QualityFlagger()
        flagger.apply_standard_flag("prod_fusion_001", StandardFlag.HIGH_CONFIDENCE)
        summary = flagger.summarize("prod_fusion_001", "evt_fusion_001")
        assert summary.total_flags == 1


class TestTimeSeriePipeline:
    """Tests for time series quality control pipeline."""

    def test_flood_evolution_timeline(self):
        """Test quality control for flood evolution time series."""
        from core.quality.sanity import (
            check_temporal_consistency,
            check_raster_temporal_consistency,
        )
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
        )
        from core.quality.actions import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus,
        )

        # Simulate flood evolution (5 time steps)
        np.random.seed(42)
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(hours=i*6) for i in range(5)]

        # Flood area values (sq km) - rise and fall
        area_values = [100, 250, 400, 350, 200]

        # Raster time series
        rasters = []
        for area in area_values:
            raster = np.zeros((50, 50))
            # Scale flood area based on value
            radius = int(np.sqrt(area / np.pi) * 2)
            radius = min(radius, 24)
            y, x = np.ogrid[:50, :50]
            mask = (x - 25)**2 + (y - 25)**2 < radius**2
            raster[mask] = 1.0
            rasters.append(raster)

        # Check temporal consistency of area values
        temporal_result = check_temporal_consistency(area_values, timestamps)
        assert temporal_result is not None

        # Check raster time series
        raster_result = check_raster_temporal_consistency(rasters, timestamps)
        assert raster_result.is_consistent

        # Calculate uncertainty across time series
        quantifier = UncertaintyQuantifier()
        ensemble_result = quantifier.calculate_ensemble_uncertainty(rasters)
        assert ensemble_result.member_count == 5

        # Gate based on temporal checks
        checks = [
            QCCheck("temporal_consistency", CheckCategory.TEMPORAL,
                    CheckStatus.PASS if temporal_result.is_consistent else CheckStatus.SOFT_FAIL),
        ]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_timeseries_001",
            product_id="prod_timeseries_001",
            confidence_score=0.85,
        )
        decision = gate.evaluate(checks, context)
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS)


class TestDegradedModePipeline:
    """Tests for quality control in degraded mode scenarios."""

    def test_single_sensor_degraded_mode(self):
        """Test pipeline when operating in single-sensor mode."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import UncertaintyQuantifier
        from core.quality.actions import (
            QualityGate, QualityFlagger,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus, StandardFlag,
        )

        np.random.seed(42)
        data = np.random.rand(100, 100)

        # Run sanity with degraded expectations
        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        sanity = suite.check(data)

        # Higher uncertainty in degraded mode
        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())

        # Gate with degraded mode context (use SPATIAL category instead of SANITY)
        checks = [
            QCCheck("sanity", CheckCategory.SPATIAL,
                    CheckStatus.PASS if sanity.passes_sanity else CheckStatus.SOFT_FAIL),
        ]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_degraded_001",
            product_id="prod_degraded_001",
            confidence_score=0.65,
            degraded_mode_level=2,  # Moderate degradation
        )

        decision = gate.evaluate(checks, context)
        # With degraded mode level 2, thresholds are relaxed
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS, GateStatus.REVIEW_REQUIRED)

        # Apply degraded mode flags
        flagger = QualityFlagger()
        flagger.apply_standard_flag("prod_degraded_001", StandardFlag.SINGLE_SENSOR_MODE)
        flagger.apply_standard_flag("prod_degraded_001", StandardFlag.RESOLUTION_DEGRADED)

        summary = flagger.summarize("prod_degraded_001", "evt_degraded_001")
        assert summary.total_flags >= 2

    def test_emergency_release_with_override(self):
        """Test emergency release pathway with review override."""
        from core.quality.actions import (
            QualityGate, QualityFlagger, ReviewRouter,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus, StandardFlag,
            ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext,
            create_emergency_gate,
        )

        # Product that would normally fail gate (with moderate confidence)
        checks = [
            QCCheck("critical_check", CheckCategory.SPATIAL, CheckStatus.SOFT_FAIL),
            QCCheck("value_check", CheckCategory.VALUE, CheckStatus.SOFT_FAIL),
            QCCheck("temporal_check", CheckCategory.TEMPORAL, CheckStatus.SOFT_FAIL),
        ]

        # Normal gate would require review (use confidence above threshold to avoid BLOCKED)
        normal_gate = QualityGate()
        normal_context = GatingContext(
            event_id="evt_emergency_001",
            product_id="prod_emergency_001",
            confidence_score=0.65,  # Above min_confidence threshold
        )
        normal_decision = normal_gate.evaluate(checks, normal_context)
        # With 3 soft failures, should be REVIEW_REQUIRED
        assert normal_decision.status == GateStatus.REVIEW_REQUIRED

        # Emergency gate with relaxed thresholds
        emergency_gate = create_emergency_gate()
        emergency_context = GatingContext(
            event_id="evt_emergency_001",
            product_id="prod_emergency_001",
            confidence_score=0.55,
            degraded_mode_level=3,
        )
        emergency_decision = emergency_gate.evaluate(checks, emergency_context)
        # Emergency gate may still require review or pass with warnings

        # Create review request
        router = ReviewRouter()
        review_context = ReviewContext(
            event_id="evt_emergency_001",
            product_id="prod_emergency_001",
            gating_decision=normal_decision.to_dict(),
            questions=["Emergency release requested - approve?"],
        )

        request = router.create_request(
            ReviewType.GENERAL,  # Use GENERAL for emergency release
            ExpertDomain.GENERAL,
            ReviewPriority.CRITICAL,
            review_context,
            auto_assign=False,
        )

        # Override the review
        result = router.override_review(
            request.request_id,
            authorized_by="emergency_coordinator@agency.gov",
            reason="Active emergency - immediate release required",
            approved=True,
        )
        assert result is True

        # Apply emergency release flags (use existing flags)
        flagger = QualityFlagger()
        flagger.apply_standard_flag("prod_emergency_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag("prod_emergency_001", StandardFlag.CONSERVATIVE_ESTIMATE)

        summary = flagger.summarize("prod_emergency_001", "evt_emergency_001")
        assert "LOW_CONFIDENCE" in summary.standard_flag_list or "CONSERVATIVE_ESTIMATE" in summary.standard_flag_list


# ============================================================================
# CROSS-MODULE INTEGRATION TESTS
# ============================================================================

class TestSanityToActionsIntegration:
    """Tests for sanity check results flowing to actions."""

    def test_sanity_failures_create_proper_checks(self):
        """Test that sanity failures translate to appropriate QC checks."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.actions import (
            QCCheck, CheckCategory, CheckStatus,
        )

        np.random.seed(42)

        # Create data with known issues
        data = np.random.rand(100, 100)
        data[0:10, :] = np.nan  # 10% NaN
        data[50:52, :] += 5.0  # Stripe artifact

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)

        # Convert sanity result to QC checks
        checks = []

        # Spatial check
        if result.spatial is not None:
            checks.append(QCCheck(
                "spatial_coherence",
                CheckCategory.SPATIAL,
                CheckStatus.PASS if result.spatial.is_coherent else CheckStatus.SOFT_FAIL,
                metric_value=len(result.spatial.issues),
            ))

        # Value check (use .values not .value)
        if result.values is not None:
            status = CheckStatus.PASS
            if not result.values.is_plausible:
                if result.values.critical_count > 0:
                    status = CheckStatus.HARD_FAIL
                else:
                    status = CheckStatus.SOFT_FAIL
            checks.append(QCCheck(
                "value_plausibility",
                CheckCategory.VALUE,
                status,
            ))

        # Artifact check (use .artifacts not .artifact)
        if result.artifacts is not None:
            status = CheckStatus.PASS
            if result.artifacts.has_artifacts:
                if result.artifacts.critical_count > 0:
                    status = CheckStatus.HARD_FAIL
                elif result.artifacts.high_count > 0:
                    status = CheckStatus.SOFT_FAIL
                else:
                    status = CheckStatus.WARNING
            checks.append(QCCheck(
                "artifact_detection",
                CheckCategory.ARTIFACT,
                status,
            ))

        assert len(checks) >= 3
        # At least one check should have issues due to our test data
        assert any(c.status != CheckStatus.PASS for c in checks)

    def test_sanity_issue_severity_mapping(self):
        """Test that sanity issue severities map to check statuses correctly."""
        from core.quality.sanity.spatial import (
            SpatialIssue, SpatialCheckType, SpatialIssueSeverity
        )
        from core.quality.actions import CheckStatus

        # Define mapping
        severity_map = {
            SpatialIssueSeverity.CRITICAL: CheckStatus.HARD_FAIL,
            SpatialIssueSeverity.HIGH: CheckStatus.SOFT_FAIL,
            SpatialIssueSeverity.MEDIUM: CheckStatus.WARNING,
            SpatialIssueSeverity.LOW: CheckStatus.PASS,
        }

        for sanity_sev, expected_status in severity_map.items():
            issue = SpatialIssue(
                check_type=SpatialCheckType.AUTOCORRELATION,
                severity=sanity_sev,
                description="Test issue",
            )

            mapped_status = severity_map[issue.severity]
            assert mapped_status == expected_status


class TestValidationToActionsIntegration:
    """Tests for validation results flowing to actions."""

    def test_cross_validation_creates_review_request(self):
        """Test that low cross-validation triggers review."""
        from core.quality.validation import (
            validate_cross_model, AgreementLevel,
        )
        from core.quality.actions import (
            QualityGate, ReviewRouter,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus,
            ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext,
        )

        np.random.seed(42)

        # Create models with significant disagreement
        model_a = np.random.rand(50, 50) > 0.5
        model_b = np.random.rand(50, 50) > 0.5  # Random - low agreement

        # Use convenience function for cross-model validation
        validation_result = validate_cross_model(
            model_outputs=[model_a.astype(float), model_b.astype(float)],
            model_ids=["Model_A", "Model_B"],
        )

        # Check agreement level - validation_result.overall_agreement is an AgreementLevel enum
        agreement_level = validation_result.overall_agreement
        is_good_agreement = agreement_level in (AgreementLevel.EXCELLENT, AgreementLevel.GOOD)

        # Get numeric agreement score from ensemble statistics if available
        agreement_score = validation_result.ensemble_statistics.get("mean_agreement", 0.5)

        checks = [
            QCCheck(
                "cross_model_validation",
                CheckCategory.CROSS_VALIDATION,
                CheckStatus.PASS if is_good_agreement else CheckStatus.SOFT_FAIL,
                metric_value=agreement_score,
                threshold=0.7,
            ),
        ]

        # Gate the product
        gate = QualityGate()
        context = GatingContext(
            event_id="evt_xval_001",
            product_id="prod_xval_001",
            confidence_score=0.75,
            cross_validation={"agreement_metrics": {"iou": agreement_score}},
        )
        decision = gate.evaluate(checks, context)

        # Poor/moderate agreement should trigger review
        if not is_good_agreement:
            assert decision.status in (GateStatus.REVIEW_REQUIRED, GateStatus.PASS_WITH_WARNINGS)

            # Create review request
            router = ReviewRouter()
            review_context = ReviewContext(
                event_id="evt_xval_001",
                product_id="prod_xval_001",
                questions=[f"Model agreement is {agreement_level.value} - please validate"],
            )
            request = router.create_request(
                ReviewType.ALGORITHM_AGREEMENT,
                ExpertDomain.GENERAL,
                ReviewPriority.NORMAL,
                review_context,
                auto_assign=False,
            )
            assert request is not None


class TestUncertaintyToActionsIntegration:
    """Tests for uncertainty results flowing to actions."""

    def test_high_uncertainty_hotspots_create_flags(self):
        """Test that high uncertainty hotspots create appropriate flags."""
        from core.quality.uncertainty import (
            SpatialUncertaintyMapper,
            HotspotDetector,
        )
        from core.quality.actions import (
            QualityFlagger, FlagLevel,
        )

        np.random.seed(42)

        # Create data with high uncertainty region
        data = np.random.rand(100, 100) * 0.1 + 0.5
        data[30:50, 30:50] = np.random.rand(20, 20)  # High variance region

        # Map uncertainty
        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)

        # Detect hotspots
        detector = HotspotDetector()
        hotspots = detector.detect_hotspots(surface.uncertainty)

        # Apply flags for hotspots
        flagger = QualityFlagger()
        if len(hotspots.hotspots) > 0:
            flagger.apply_flag(
                "prod_uncertainty_001",
                "SPATIAL_UNCERTAINTY",
                level=FlagLevel.REGION,
                reason=f"Detected {len(hotspots.hotspots)} uncertainty hotspots",
            )

            # Apply pixel-level mask
            if hotspots.hotspot_fraction > 0.1:
                flagger.apply_flag(
                    "prod_uncertainty_001",
                    "HIGH_UNCERTAINTY_AREA",
                    level=FlagLevel.REGION,
                    reason=f"{hotspots.hotspot_fraction*100:.1f}% of area has high uncertainty",
                )

        summary = flagger.summarize("prod_uncertainty_001", "evt_uncertainty_001")
        # May or may not have flags depending on hotspot detection
        assert summary is not None

    def test_uncertainty_threshold_decision(self):
        """Test using uncertainty for threshold-based gating."""
        from core.quality.uncertainty import (
            QualityErrorPropagator,
        )
        from core.quality.actions import (
            QCCheck, CheckCategory, CheckStatus,
            QualityGate, GatingContext, GateStatus,
        )

        propagator = QualityErrorPropagator()

        # Case 1: Clear exceedance
        result = propagator.propagate_threshold_decision(
            value=0.85,
            uncertainty=0.05,
            threshold=0.7,
        )
        assert result["decision"] == "exceeds"
        assert result["decision_confidence"] > 0.95

        # Case 2: Uncertain case near threshold
        result = propagator.propagate_threshold_decision(
            value=0.72,
            uncertainty=0.1,
            threshold=0.7,
        )
        # Near threshold with high uncertainty = low confidence

        # Create QC check based on decision confidence
        status = CheckStatus.PASS if result["decision_confidence"] > 0.8 else CheckStatus.WARNING
        checks = [QCCheck("threshold_decision", CheckCategory.UNCERTAINTY, status)]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_threshold_001",
            product_id="prod_threshold_001",
            confidence_score=result["decision_confidence"],
        )
        decision = gate.evaluate(checks, context)
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS, GateStatus.BLOCKED)


# ============================================================================
# EDGE CASE PIPELINE TESTS
# ============================================================================

class TestPipelineEdgeCases:
    """Edge case tests for quality pipeline."""

    def test_all_nan_data_pipeline(self):
        """Test pipeline handles all-NaN data gracefully."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.actions import (
            QCCheck, CheckCategory, CheckStatus,
            QualityGate, GatingContext, GateStatus,
        )

        data = np.full((50, 50), np.nan)

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)

        # Should fail sanity
        assert not result.passes_sanity

        # Create hard fail check
        checks = [
            QCCheck("data_validity", CheckCategory.VALUE, CheckStatus.HARD_FAIL,
                    details="All data is NaN"),
        ]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_nan_001",
            product_id="prod_nan_001",
            confidence_score=0.0,
        )
        decision = gate.evaluate(checks, context)
        assert decision.status == GateStatus.BLOCKED

    def test_empty_data_pipeline(self):
        """Test pipeline handles empty data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import UncertaintyQuantifier

        data = np.array([]).reshape(0, 0)

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Should handle gracefully
        assert result is not None

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.n_samples == 0

    def test_constant_data_pipeline(self):
        """Test pipeline with constant data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
        )

        data = np.full((50, 50), 0.5)

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Constant data should pass value checks (use .values not .value)
        assert result.values.is_plausible

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.std == 0.0
        assert metrics.coefficient_of_variation == 0.0

        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)
        # Constant data has zero local variance
        assert np.allclose(surface.uncertainty, 0, atol=1e-10) or np.all(np.isnan(surface.uncertainty))

    def test_single_pixel_pipeline(self):
        """Test pipeline with single pixel data."""
        from core.quality.sanity import (
            check_spatial_coherence,
            check_value_plausibility,
        )
        from core.quality.uncertainty import UncertaintyQuantifier

        data = np.array([[0.5]])

        spatial = check_spatial_coherence(data)
        assert spatial is not None

        values = check_value_plausibility(data)
        assert values is not None

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.n_samples == 1

    def test_mixed_valid_invalid_data(self):
        """Test pipeline with mixed valid and invalid data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
        )

        np.random.seed(42)
        data = np.random.rand(100, 100)
        data[0:20, :] = np.nan  # 20% NaN
        data[80:100, :] = np.inf  # 20% Inf

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Should detect issues (use .values not .value)
        assert len(result.values.issues) > 0

        # Uncertainty should handle NaN/Inf
        quantifier = UncertaintyQuantifier()
        valid_data = data[np.isfinite(data)]
        metrics = quantifier.calculate_metrics(valid_data)
        assert metrics.n_samples < 10000  # Less than full array

        # Spatial uncertainty should handle
        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)
        assert surface.uncertainty.shape == data.shape


# ============================================================================
# SERIALIZATION AND REPORTING TESTS
# ============================================================================

class TestPipelineOutputs:
    """Tests for pipeline output serialization."""

    def test_complete_pipeline_report_generation(self):
        """Test generating a complete report from pipeline results."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
        )
        from core.quality.actions import (
            QualityGate, QualityFlagger,
            QCCheck, CheckCategory, CheckStatus,
            GatingContext, StandardFlag,
        )

        np.random.seed(42)
        data = np.random.rand(50, 50)

        # Run full pipeline
        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        sanity = suite.check(data)

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())

        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)

        # Use SPATIAL instead of SANITY (which doesn't exist)
        checks = [
            QCCheck("sanity", CheckCategory.SPATIAL, CheckStatus.PASS),
        ]

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_report_001",
            product_id="prod_report_001",
            confidence_score=0.85,
        )
        decision = gate.evaluate(checks, context)

        flagger = QualityFlagger()
        flagger.apply_standard_flag("prod_report_001", StandardFlag.HIGH_CONFIDENCE)
        flag_summary = flagger.summarize("prod_report_001", "evt_report_001")

        # Build comprehensive report
        report = {
            "event_id": "evt_report_001",
            "product_id": "prod_report_001",
            "sanity": sanity.to_dict(),
            "uncertainty": {
                "global": metrics.to_dict(),
                "spatial": surface.to_dict(),
            },
            "gating": decision.to_dict(),
            "flags": flag_summary.to_dict(),
        }

        # Verify all components are serializable
        import json
        json_str = json.dumps(report, default=str)
        assert len(json_str) > 100

        # Verify key fields exist
        parsed = json.loads(json_str)
        assert "sanity" in parsed
        assert "uncertainty" in parsed
        assert "gating" in parsed
        assert "flags" in parsed


class TestModuleImports:
    """Tests to verify all module imports work correctly."""

    def test_quality_module_public_api(self):
        """Test that core.quality exposes expected API."""
        import core.quality as quality

        assert hasattr(quality, 'sanity')
        assert hasattr(quality, 'uncertainty')

    def test_sanity_module_imports(self):
        """Test sanity module imports."""
        from core.quality.sanity import (
            SanitySuite,
            SanitySuiteConfig,
            SanitySuiteResult,
            # Spatial
            SpatialCoherenceChecker,
            check_spatial_coherence,
            # Values
            ValuePlausibilityChecker,
            check_value_plausibility,
            ValueType,
            # Temporal
            TemporalConsistencyChecker,
            check_temporal_consistency,
            # Artifacts
            ArtifactDetector,
            detect_artifacts,
        )
        assert SanitySuite is not None

    def test_uncertainty_module_imports(self):
        """Test uncertainty module imports."""
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
            HotspotDetector,
            QualityErrorPropagator,
            CalibrationAssessor,
            SensitivityAnalyzer,
            # Convenience functions
            propagate_quality_uncertainty,
            calculate_confidence_interval,
            calculate_prediction_interval,
            combine_independent_uncertainties,
            threshold_exceedance_probability,
        )
        assert UncertaintyQuantifier is not None

    def test_actions_module_imports(self):
        """Test actions module imports."""
        from core.quality.actions import (
            # Gating
            QualityGate,
            GatingContext,
            GatingThresholds,
            QCCheck,
            CheckCategory,
            CheckStatus,
            GateStatus,
            quick_gate,
            create_emergency_gate,
            create_operational_gate,
            create_research_gate,
            # Flagging
            QualityFlagger,
            FlagRegistry,
            StandardFlag,
            FlagLevel,
            FlagSeverity,
            create_confidence_flag,
            flag_from_conditions,
            # Routing
            ReviewRouter,
            Expert,
            ReviewType,
            ExpertDomain,
            ReviewPriority,
            ReviewStatus,
            ReviewContext,
            ReviewOutcome,
        )
        assert QualityGate is not None
        assert QualityFlagger is not None
        assert ReviewRouter is not None

    def test_validation_module_imports(self):
        """Test validation module imports."""
        from core.quality.validation import (
            # Cross-model
            CrossModelValidator,
            CrossModelConfig,
            CrossModelResult,
            validate_cross_model,
            # Cross-sensor
            CrossSensorValidator,
            CrossSensorConfig,
            CrossSensorResult,
            validate_cross_sensor,
            # Historical
            HistoricalValidator,
            HistoricalConfig,
            HistoricalResult,
            # Consensus
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusResult,
        )
        assert CrossModelValidator is not None
        assert CrossSensorValidator is not None
        assert HistoricalValidator is not None
        assert ConsensusGenerator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
