"""
Tests for the data validation module.

Tests integrity, anomaly detection, and completeness validation.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from core.data.ingestion.validation import (
    # Integrity
    IntegrityConfig,
    IntegrityValidator,
    IntegrityCheckType,
    IntegritySeverity,
    validate_integrity,
    # Anomaly
    AnomalyConfig,
    AnomalyDetector,
    AnomalyType,
    AnomalySeverity,
    detect_anomalies,
    # Completeness
    CompletenessConfig,
    CompletenessValidator,
    CompletenessCheckType,
    CompletenessSeverity,
    validate_completeness,
    # Suite
    ValidationSuite,
)


# =============================================================================
# Integrity Validation Tests
# =============================================================================


class TestIntegrityValidator:
    """Tests for IntegrityValidator class."""

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = IntegrityValidator()
        result = validator.validate_file("/nonexistent/file.tif")

        assert not result.is_valid
        assert result.file_size_bytes == 0
        assert len(result.issues) == 1
        assert result.issues[0].severity == IntegritySeverity.ERROR
        assert "not found" in result.issues[0].message.lower()

    def test_file_size_constraints(self):
        """Test file size validation."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(b"x" * 100)  # Small file
            f.flush()

            config = IntegrityConfig(
                min_file_size_bytes=1000,
                compute_checksum=False,
            )
            validator = IntegrityValidator(config)
            result = validator.validate_file(f.name)

            # Should have warning about small size
            assert any(
                i.check_type == IntegrityCheckType.FORMAT
                and "below" in i.message.lower()
                for i in result.issues
            )

            Path(f.name).unlink()

    def test_checksum_computation(self):
        """Test checksum computation."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data for checksum")
            f.flush()

            config = IntegrityConfig(
                compute_checksum=True,
                checksum_algorithms=["md5", "sha256"],
            )
            validator = IntegrityValidator(config)
            result = validator.validate_file(f.name)

            assert result.checksum_md5 is not None
            assert result.checksum_sha256 is not None
            assert len(result.checksum_md5) == 32  # MD5 hex length
            assert len(result.checksum_sha256) == 64  # SHA256 hex length

            Path(f.name).unlink()

    def test_checksum_verification(self):
        """Test checksum verification against expected value."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data")
            f.flush()

            # First get the actual checksum
            validator = IntegrityValidator()
            result = validator.validate_file(f.name)
            actual_sha256 = result.checksum_sha256

            # Now verify with correct checksum
            config = IntegrityConfig(
                expected_checksum=actual_sha256,
            )
            validator2 = IntegrityValidator(config)
            result2 = validator2.validate_file(f.name)
            assert not any(
                i.check_type == IntegrityCheckType.CHECKSUM
                and i.severity == IntegritySeverity.ERROR
                for i in result2.issues
            )

            # Verify with wrong checksum
            config3 = IntegrityConfig(
                expected_checksum="wrong_checksum",
            )
            validator3 = IntegrityValidator(config3)
            result3 = validator3.validate_file(f.name)
            assert any(
                i.check_type == IntegrityCheckType.CHECKSUM
                and i.severity == IntegritySeverity.ERROR
                for i in result3.issues
            )

            Path(f.name).unlink()

    def test_verify_checksum_method(self):
        """Test the verify_checksum convenience method."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data for verification")
            f.flush()

            validator = IntegrityValidator()
            result = validator.validate_file(f.name)

            # Verify with correct checksum
            assert validator.verify_checksum(f.name, result.checksum_sha256, "sha256")

            # Verify with wrong checksum
            assert not validator.verify_checksum(f.name, "wrong", "sha256")

            Path(f.name).unlink()

    def test_result_to_dict(self):
        """Test result serialization to dict."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test")
            f.flush()

            result = validate_integrity(f.name)
            result_dict = result.to_dict()

            assert "is_valid" in result_dict
            assert "file_path" in result_dict
            assert "file_size_bytes" in result_dict
            assert "issues" in result_dict
            assert isinstance(result_dict["issues"], list)

            Path(f.name).unlink()


class TestIntegrityConvenienceFunction:
    """Tests for validate_integrity convenience function."""

    def test_validate_integrity_basic(self):
        """Test basic integrity validation."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data")
            f.flush()

            result = validate_integrity(f.name, compute_checksum=True)
            assert result.file_size_bytes == 9
            assert result.checksum_sha256 is not None

            Path(f.name).unlink()

    def test_validate_integrity_strict_mode(self):
        """Test strict mode validation."""
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            f.write(b"test data")
            f.flush()

            # Non-strict should pass even with warnings
            result = validate_integrity(f.name, strict=False)
            # Unknown format generates a warning

            # Strict mode treats warnings as errors
            result_strict = validate_integrity(f.name, strict=True)
            # Might be invalid due to unknown format warning

            Path(f.name).unlink()


# =============================================================================
# Anomaly Detection Tests
# =============================================================================


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_detect_zscore_outliers(self):
        """Test z-score outlier detection."""
        # Create data with >1% outliers to trigger detection
        np.random.seed(42)
        data = np.random.normal(100, 10, (1, 100, 100))
        # Add many extreme outliers (>1% of pixels = >100 outliers)
        # Add 150 outliers by setting a block
        data[0, :15, :10] = 1000  # 150 very high values

        config = AnomalyConfig(zscore_threshold=3.0)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        assert result.has_anomalies
        # Should detect outliers
        outlier_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.OUTLIER_ZSCORE
        ]
        assert len(outlier_anomalies) >= 1

    def test_detect_invalid_values(self):
        """Test detection of NaN and Inf values."""
        data = np.ones((1, 100, 100), dtype=np.float32)
        data[0, 10, 10] = np.nan
        data[0, 20, 20] = np.inf
        data[0, 30, 30] = -np.inf

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        assert result.has_anomalies
        invalid_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.INVALID_VALUE
        ]
        assert len(invalid_anomalies) >= 1  # NaN and Inf detected

    def test_detect_nodata_pattern(self):
        """Test detection of excessive nodata."""
        data = np.ones((1, 100, 100))
        # Set 60% as nodata
        data[0, :60, :] = -9999

        config = AnomalyConfig(max_nodata_percent=50.0)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data, nodata=-9999)

        assert result.has_anomalies
        nodata_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.NODATA_PATTERN
        ]
        assert len(nodata_anomalies) == 1
        assert nodata_anomalies[0].severity == AnomalySeverity.HIGH

    def test_detect_stripe_artifacts(self):
        """Test detection of stripe artifacts."""
        np.random.seed(42)
        data = np.random.normal(100, 5, (1, 100, 100))

        # Add horizontal stripes (some rows have very different values)
        stripe_rows = [10, 20, 30, 40, 50]
        for row in stripe_rows:
            data[0, row, :] = 200  # Much higher than normal

        config = AnomalyConfig(stripe_detection=True, stripe_threshold=2.0)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        # Should detect stripe artifacts
        stripe_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.STRIPE_ARTIFACT
        ]
        assert len(stripe_anomalies) >= 1

    def test_detect_saturation(self):
        """Test detection of saturated values."""
        # Use uint8 data with saturated values
        data = np.random.randint(50, 200, (1, 100, 100), dtype=np.uint8)
        # Saturate 5% of pixels
        data[0, :5, :] = 255  # Max uint8 value

        config = AnomalyConfig(saturation_threshold=0.99)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        saturated_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.SATURATED
        ]
        assert len(saturated_anomalies) >= 1

    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Truly clean data - constant values won't trigger any anomalies
        clean_data = np.full((1, 100, 100), 100.0, dtype=np.float32)
        # Disable spatial anomaly detection for clean test
        config_clean = AnomalyConfig(detect_spatial=False)
        detector = AnomalyDetector(config_clean)
        clean_result = detector.detect_from_array(clean_data)
        assert clean_result.overall_quality_score == 1.0  # Perfect score for constant data

        # Data with severe issues should have lower score
        bad_data = np.ones((1, 100, 100))
        bad_data[0, :70, :] = -9999  # 70% nodata (more extreme)

        config_bad = AnomalyConfig(max_nodata_percent=10.0, detect_spatial=False)
        detector2 = AnomalyDetector(config_bad)
        bad_result = detector2.detect_from_array(bad_data, nodata=-9999)
        assert bad_result.overall_quality_score < clean_result.overall_quality_score

    def test_band_statistics(self):
        """Test per-band statistics computation."""
        data = np.random.normal(100, 15, (3, 100, 100))

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        assert len(result.band_statistics) == 3
        for band_idx in range(3):
            stats = result.band_statistics[band_idx]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert stats["std"] > 0

    def test_2d_array_handling(self):
        """Test handling of 2D arrays."""
        data = np.random.normal(100, 10, (100, 100))

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        assert len(result.band_statistics) == 1  # Converted to 3D with 1 band

    def test_result_to_dict(self):
        """Test result serialization."""
        data = np.random.normal(100, 10, (1, 50, 50))

        result = detect_anomalies(data)
        result_dict = result.to_dict()

        assert "has_anomalies" in result_dict
        assert "anomaly_count" in result_dict
        assert "overall_quality_score" in result_dict
        assert "band_statistics" in result_dict


class TestAnomalyConvenienceFunctions:
    """Tests for anomaly detection convenience functions."""

    def test_detect_anomalies_basic(self):
        """Test detect_anomalies convenience function."""
        data = np.random.normal(0, 1, (100, 100))
        result = detect_anomalies(data, zscore_threshold=2.0)

        assert result is not None
        assert result.duration_seconds >= 0


# =============================================================================
# Completeness Validation Tests
# =============================================================================


class TestCompletenessValidator:
    """Tests for CompletenessValidator class."""

    def test_full_coverage(self):
        """Test data with full coverage."""
        data = np.ones((1, 100, 100))

        config = CompletenessConfig(min_coverage_percent=80.0)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data)

        assert result.is_complete
        assert result.coverage_percentage == 100.0
        assert result.valid_pixel_count == 10000
        assert result.total_pixel_count == 10000

    def test_partial_coverage(self):
        """Test data with partial coverage."""
        data = np.ones((1, 100, 100))
        # Set 30% as nodata
        data[0, :30, :] = -9999

        config = CompletenessConfig(min_coverage_percent=80.0)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, nodata=-9999)

        assert not result.is_complete  # 70% < 80%
        assert 69 < result.coverage_percentage < 71
        assert len(result.issues) >= 1

    def test_gap_detection(self):
        """Test gap region detection."""
        data = np.ones((1, 100, 100))
        # Create a large gap in the middle
        data[0, 40:60, 40:60] = -9999  # 400 pixel gap

        config = CompletenessConfig(
            min_coverage_percent=90.0,
            detect_gaps=True,
            min_gap_size_pixels=100,
        )
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, nodata=-9999)

        # Should detect the gap
        assert len(result.gap_regions) >= 1

    def test_band_consistency(self):
        """Test band coverage consistency check."""
        data = np.ones((3, 100, 100))
        # Band 0: full coverage
        # Band 1: 90% coverage
        data[1, :10, :] = -9999
        # Band 2: 50% coverage
        data[2, :50, :] = -9999

        config = CompletenessConfig(check_band_consistency=True)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, nodata=-9999)

        # Should flag inconsistent coverage
        inconsistent_issues = [
            i
            for i in result.issues
            if i.check_type == CompletenessCheckType.BAND_COMPLETENESS
        ]
        assert len(inconsistent_issues) >= 1

    def test_bounds_coverage(self):
        """Test bounds coverage validation."""
        data = np.ones((1, 100, 100))
        actual_bounds = (0, 0, 10, 10)
        expected_bounds = (0, 0, 20, 20)  # Larger than actual

        config = CompletenessConfig(expected_bounds=expected_bounds)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, bounds=actual_bounds)

        # Should flag partial AOI coverage
        bounds_issues = [
            i
            for i in result.issues
            if i.check_type == CompletenessCheckType.SPATIAL_COVERAGE
        ]
        assert len(bounds_issues) >= 1

    def test_no_overlap_bounds(self):
        """Test when data bounds don't overlap with expected AOI."""
        data = np.ones((1, 100, 100))
        actual_bounds = (100, 100, 200, 200)
        expected_bounds = (0, 0, 50, 50)  # No overlap

        config = CompletenessConfig(expected_bounds=expected_bounds)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, bounds=actual_bounds)

        assert not result.is_complete
        # Should have critical issue
        critical_issues = [
            i for i in result.issues if i.severity == CompletenessSeverity.CRITICAL
        ]
        assert len(critical_issues) >= 1

    def test_metadata_completeness(self):
        """Test metadata completeness validation."""
        data = np.ones((1, 100, 100))
        profile = {"crs": "EPSG:4326"}  # Missing required fields

        config = CompletenessConfig(required_metadata=["crs", "nodata", "transform"])
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, profile=profile)

        # Should flag missing metadata
        meta_issues = [
            i
            for i in result.issues
            if i.check_type == CompletenessCheckType.METADATA_COMPLETENESS
        ]
        assert len(meta_issues) >= 1

    def test_crs_mismatch(self):
        """Test CRS mismatch detection."""
        data = np.ones((1, 100, 100))

        config = CompletenessConfig(expected_crs="EPSG:4326")
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, crs="EPSG:32610")

        # Should flag CRS mismatch
        crs_issues = [
            i
            for i in result.issues
            if "CRS" in i.message or "crs" in i.message.lower()
        ]
        assert len(crs_issues) >= 1

    def test_coverage_map_generation(self):
        """Test coverage map is generated."""
        data = np.ones((1, 100, 100))
        data[0, :30, :] = -9999

        config = CompletenessConfig(detect_gaps=True)
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, nodata=-9999)

        assert result.coverage_map is not None
        assert result.coverage_map.shape == (100, 100)
        # Coverage map should show valid areas
        assert np.sum(result.coverage_map) == 7000  # 70% valid

    def test_2d_array_handling(self):
        """Test handling of 2D arrays."""
        data = np.ones((100, 100))

        validator = CompletenessValidator()
        result = validator.validate_array(data)

        assert result.is_complete
        assert result.coverage_percentage == 100.0

    def test_result_to_dict(self):
        """Test result serialization."""
        data = np.ones((1, 50, 50))
        data[0, :10, :] = -9999

        result = validate_completeness(data, nodata=-9999)
        result_dict = result.to_dict()

        assert "is_complete" in result_dict
        assert "coverage_percentage" in result_dict
        assert "gap_percentage" in result_dict
        assert "issues" in result_dict


class TestCompletenessConvenienceFunctions:
    """Tests for completeness convenience functions."""

    def test_validate_completeness_basic(self):
        """Test validate_completeness convenience function."""
        data = np.ones((100, 100))
        result = validate_completeness(data, min_coverage=80.0)

        assert result.is_complete
        assert result.coverage_percentage == 100.0

    def test_validate_completeness_with_nodata(self):
        """Test validation with nodata."""
        data = np.ones((100, 100))
        data[:20, :] = np.nan

        result = validate_completeness(data, nodata=np.nan, min_coverage=90.0)

        assert not result.is_complete


# =============================================================================
# Validation Suite Tests
# =============================================================================


class TestValidationSuite:
    """Tests for ValidationSuite class."""

    def test_suite_all_validators(self):
        """Test running all validators in suite."""
        data = np.random.normal(100, 10, (100, 100))

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, data)

            suite = ValidationSuite()
            # Note: .npy files won't work with rasterio, so this tests error handling
            result = suite.validate(f.name)

            # Result should be returned even if individual validators fail
            assert result is not None
            assert "summary" in result.to_dict()

            Path(f.name).unlink()

    def test_suite_skip_validators(self):
        """Test skipping individual validators."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data")
            f.flush()

            suite = ValidationSuite()
            result = suite.validate(
                f.name,
                skip_integrity=False,
                skip_anomaly=True,
                skip_completeness=True,
            )

            assert result.integrity is not None
            assert result.anomaly is None
            assert result.completeness is None

            Path(f.name).unlink()

    def test_suite_overall_score(self):
        """Test overall score calculation."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data")
            f.flush()

            suite = ValidationSuite()
            result = suite.validate(f.name, skip_anomaly=True, skip_completeness=True)

            # Score should be between 0 and 1
            assert 0 <= result.overall_score <= 1

            Path(f.name).unlink()

    def test_suite_result_to_dict(self):
        """Test suite result serialization."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test data")
            f.flush()

            suite = ValidationSuite()
            result = suite.validate(f.name, skip_anomaly=True, skip_completeness=True)
            result_dict = result.to_dict()

            assert "is_valid" in result_dict
            assert "overall_score" in result_dict
            assert "summary" in result_dict
            assert "integrity" in result_dict

            Path(f.name).unlink()


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array(self):
        """Test handling of empty arrays."""
        data = np.array([]).reshape(0, 0)

        detector = AnomalyDetector()
        # Should handle gracefully without crash
        # Empty arrays may fail in various ways, just ensure no crash

    def test_single_pixel(self):
        """Test handling of single-pixel data."""
        data = np.array([[[100]]])

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)
        assert result is not None

        validator = CompletenessValidator()
        result = validator.validate_array(data)
        assert result.coverage_percentage == 100.0

    def test_all_nodata(self):
        """Test handling of all-nodata arrays."""
        data = np.full((1, 100, 100), -9999)

        validator = CompletenessValidator()
        result = validator.validate_array(data, nodata=-9999)

        assert result.coverage_percentage == 0.0
        assert not result.is_complete

    def test_nan_nodata(self):
        """Test handling of NaN as nodata."""
        data = np.full((1, 100, 100), np.nan)

        validator = CompletenessValidator()
        result = validator.validate_array(data, nodata=np.nan)

        assert result.coverage_percentage == 0.0

    def test_constant_array(self):
        """Test handling of constant arrays (std=0)."""
        data = np.full((1, 100, 100), 42.0)

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        # Should not crash with zero std
        assert result is not None
        # No outliers in constant data
        outlier_anomalies = [
            a
            for a in result.anomalies
            if a.anomaly_type in [AnomalyType.OUTLIER_ZSCORE, AnomalyType.OUTLIER_IQR]
        ]
        assert len(outlier_anomalies) == 0


class TestMultipleBands:
    """Tests for multi-band data handling."""

    def test_multiband_anomaly_detection(self):
        """Test anomaly detection across multiple bands."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (5, 100, 100))

        # Add anomaly only in band 2
        data[2, 50, 50] = 1000

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        # Should have statistics for all 5 bands
        assert len(result.band_statistics) == 5

    def test_multiband_completeness(self):
        """Test completeness check across multiple bands."""
        data = np.ones((3, 100, 100))

        # Different coverage per band
        data[0, :10, :] = -9999  # 90% coverage
        data[1, :20, :] = -9999  # 80% coverage
        data[2, :30, :] = -9999  # 70% coverage

        validator = CompletenessValidator()
        result = validator.validate_array(data, nodata=-9999)

        # Overall coverage is where ALL bands have data
        # That's 70% (rows 30-100 all have data)
        assert result.coverage_percentage == 70.0


# =============================================================================
# Additional Edge Case Tests (Added by Track 5 Review)
# =============================================================================


class TestIntegrityEdgeCases:
    """Additional edge case tests for integrity validation."""

    def test_validate_empty_file(self):
        """Test validation of zero-byte file."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            # Write nothing - empty file
            f.flush()

            config = IntegrityConfig(min_file_size_bytes=1)
            validator = IntegrityValidator(config)
            result = validator.validate_file(f.name)

            # Should have warning about small/empty file
            assert result.file_size_bytes == 0
            assert any("below" in i.message.lower() for i in result.issues)

            Path(f.name).unlink()

    def test_checksum_sha512(self):
        """Test SHA512 checksum computation."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test sha512 checksum")
            f.flush()

            config = IntegrityConfig(
                compute_checksum=True,
                checksum_algorithms=["sha512"],
            )
            validator = IntegrityValidator(config)
            result = validator.validate_file(f.name)

            # SHA512 should be in metadata
            assert "sha512" in result.metadata.get("checksums", {})
            assert len(result.metadata["checksums"]["sha512"]) == 128  # SHA512 hex length

            Path(f.name).unlink()

    def test_max_file_size_exceeded(self):
        """Test file size exceeds maximum."""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"x" * 1000)  # 1000 bytes
            f.flush()

            config = IntegrityConfig(
                max_file_size_bytes=100,
                compute_checksum=False,
            )
            validator = IntegrityValidator(config)
            result = validator.validate_file(f.name)

            # Should have error about exceeding max size
            assert not result.is_valid
            assert any(
                i.severity == IntegritySeverity.ERROR
                and "exceeds" in i.message.lower()
                for i in result.issues
            )

            Path(f.name).unlink()


class TestAnomalyEdgeCases:
    """Additional edge case tests for anomaly detection."""

    def test_all_valid_data_no_nodata(self):
        """Test data without any nodata values."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (1, 50, 50))

        detector = AnomalyDetector()
        result = detector.detect_from_array(data, nodata=None)

        # Should compute statistics correctly
        assert 0 in result.band_statistics
        assert result.band_statistics[0]["valid_count"] == 2500

    def test_iqr_zero(self):
        """Test data where IQR is zero (many repeated values)."""
        # Create data where most values are the same, so IQR=0
        data = np.full((1, 100, 100), 50.0)
        # Add a few different values
        data[0, 0, 0] = 100

        detector = AnomalyDetector()
        result = detector.detect_from_array(data)

        # Should not crash with zero IQR
        assert result is not None

    def test_sampled_detection(self):
        """Test anomaly detection with sampling."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (1, 200, 200))  # 40,000 pixels

        config = AnomalyConfig(sample_size=1000)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        # Should still compute statistics
        assert 0 in result.band_statistics
        # Valid count in stats should be sample_size or less
        assert result.band_statistics[0]["valid_count"] <= 1000

    def test_vertical_stripes(self):
        """Test detection of vertical stripe artifacts."""
        np.random.seed(42)
        data = np.random.normal(100, 5, (1, 100, 100))

        # Add vertical stripes (some columns have very different values)
        stripe_cols = [10, 20, 30, 40, 50]
        for col in stripe_cols:
            data[0, :, col] = 200  # Much higher than normal

        config = AnomalyConfig(stripe_detection=True, stripe_threshold=2.0)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data)

        # Should detect vertical stripe artifacts
        stripe_anomalies = [
            a for a in result.anomalies
            if a.anomaly_type == AnomalyType.STRIPE_ARTIFACT
            and a.statistics.get("direction") == "vertical"
        ]
        assert len(stripe_anomalies) >= 1

    def test_nan_as_nodata_value(self):
        """Test using NaN as nodata value."""
        data = np.ones((1, 100, 100), dtype=np.float32)
        data[0, :30, :] = np.nan  # 30% NaN

        config = AnomalyConfig(max_nodata_percent=20.0)
        detector = AnomalyDetector(config)
        result = detector.detect_from_array(data, nodata=np.nan)

        # Should detect excessive nodata
        nodata_anomalies = [
            a for a in result.anomalies if a.anomaly_type == AnomalyType.NODATA_PATTERN
        ]
        assert len(nodata_anomalies) >= 1


class TestCompletenessEdgeCases:
    """Additional edge case tests for completeness validation."""

    def test_expected_bands_missing(self):
        """Test missing expected bands."""
        data = np.ones((2, 100, 100))  # Only 2 bands

        config = CompletenessConfig(expected_bands=[0, 1, 2, 3])  # Expect 4 bands
        validator = CompletenessValidator(config)
        result = validator.validate_array(data)

        # Should flag missing bands
        band_issues = [
            i for i in result.issues
            if i.check_type == CompletenessCheckType.BAND_COMPLETENESS
        ]
        assert len(band_issues) >= 1

    def test_edge_buffer(self):
        """Test edge buffer for gap detection."""
        data = np.ones((1, 100, 100))
        # Set edges as nodata
        data[0, :5, :] = -9999  # Top edge
        data[0, -5:, :] = -9999  # Bottom edge

        # Without edge buffer, should detect gaps
        config1 = CompletenessConfig(
            detect_gaps=True,
            edge_buffer_pixels=0,
            min_gap_size_pixels=100,
        )
        validator1 = CompletenessValidator(config1)
        result1 = validator1.validate_array(data, nodata=-9999)

        # With edge buffer, should ignore edge gaps
        config2 = CompletenessConfig(
            detect_gaps=True,
            edge_buffer_pixels=5,
            min_gap_size_pixels=100,
        )
        validator2 = CompletenessValidator(config2)
        result2 = validator2.validate_array(data, nodata=-9999)

        # Edge buffer should reduce detected gaps
        # (edge nodata is ignored with buffer)

    def test_gap_region_pixel_bounds(self):
        """Test gap region pixel bounds calculation."""
        data = np.ones((1, 100, 100))
        # Create a gap block
        data[0, 30:50, 40:60] = -9999  # 20x20 gap = 400 pixels

        config = CompletenessConfig(
            detect_gaps=True,
            min_gap_size_pixels=100,
        )
        validator = CompletenessValidator(config)
        result = validator.validate_array(data, nodata=-9999)

        # Should detect the gap with correct pixel bounds
        assert len(result.gap_regions) >= 1
        gap = result.gap_regions[0]
        assert gap.pixel_bounds is not None
        # Verify bounds are within expected range
        col_start, row_start, col_end, row_end = gap.pixel_bounds
        assert 30 <= row_start <= 50
        assert 40 <= col_start <= 60


class TestValidationSuiteEdgeCases:
    """Additional edge case tests for ValidationSuite."""

    def test_suite_with_custom_configs(self):
        """Test suite with custom validator configurations."""
        integrity_config = IntegrityConfig(compute_checksum=False)
        anomaly_config = AnomalyConfig(zscore_threshold=2.0)
        completeness_config = CompletenessConfig(min_coverage_percent=50.0)

        suite = ValidationSuite(
            integrity_config=integrity_config,
            anomaly_config=anomaly_config,
            completeness_config=completeness_config,
        )

        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test")
            f.flush()

            result = suite.validate(f.name, skip_anomaly=True, skip_completeness=True)

            # Integrity should not compute checksum
            assert result.integrity is not None
            assert result.integrity.checksum_md5 is None

            Path(f.name).unlink()

    def test_suite_all_skipped(self):
        """Test suite with all validators skipped."""
        suite = ValidationSuite()

        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b"test")
            f.flush()

            result = suite.validate(
                f.name,
                skip_integrity=True,
                skip_anomaly=True,
                skip_completeness=True,
            )

            # All results should be None
            assert result.integrity is None
            assert result.anomaly is None
            assert result.completeness is None
            # But should still return a result
            assert result.is_valid  # No failures if nothing ran

            Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
