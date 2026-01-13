"""
Validation Module for Ingestion Pipeline.

Provides comprehensive validation for ingested geospatial data:
- Integrity: File format, checksum, header, and structure validation
- Anomaly: Statistical outliers, spatial artifacts, and pattern detection
- Completeness: Coverage analysis, gap detection, and metadata validation
- Image: Band presence, content validation, CRS/bounds verification (NEW)

Each validator follows a consistent pattern:
- Config dataclass for validation options
- Result dataclass with validation details and issues
- Validator class with validate_* methods
- Convenience function for simple usage

Example:
    from core.data.ingestion.validation import (
        validate_integrity,
        detect_anomalies_from_file,
        validate_completeness_from_file,
        ImageValidator,
        validate_image,
    )

    # Quick validation
    integrity = validate_integrity("data.tif")
    anomalies = detect_anomalies_from_file("data.tif")
    completeness = validate_completeness_from_file("data.tif")

    # Image validation (for production workflows)
    validator = ImageValidator()
    result = validator.validate("satellite_image.tif")
    if not result.is_valid:
        print(f"Validation failed: {result.errors}")

    # Combined validation
    from core.data.ingestion.validation import ValidationSuite
    suite = ValidationSuite()
    result = suite.validate("data.tif")
"""

from core.data.ingestion.validation.integrity import (
    IntegrityCheckType,
    IntegrityConfig,
    IntegrityIssue,
    IntegrityResult,
    IntegritySeverity,
    IntegrityValidator,
    validate_integrity,
)
from core.data.ingestion.validation.anomaly import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyLocation,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    DetectedAnomaly,
    detect_anomalies,
    detect_anomalies_from_file,
)
from core.data.ingestion.validation.completeness import (
    CompletenessCheckType,
    CompletenessConfig,
    CompletenessIssue,
    CompletenessResult,
    CompletenessSeverity,
    CompletenessValidator,
    CoverageRegion,
    validate_completeness,
    validate_completeness_from_file,
)

# Image validation (production workflow)
from core.data.ingestion.validation.exceptions import (
    BlankBandError,
    BoundsError,
    DimensionMismatchError,
    ImageValidationError,
    InvalidCRSError,
    LoadError,
    MissingBandError,
    ResolutionError,
    SARValidationError,
    ValidationTimeoutError,
)
from core.data.ingestion.validation.config import (
    ActionConfig,
    AlertConfig,
    OpticalThresholds,
    PerformanceConfig,
    SARThresholds,
    ScreenshotConfig,
    ValidationConfig,
    load_config,
)
from core.data.ingestion.validation.image_validator import (
    BandStatistics,
    BandValidationResult,
    ImageMetadata,
    ImageValidationResult,
    ImageValidator,
    validate_image,
)
from core.data.ingestion.validation.band_validator import BandValidator
from core.data.ingestion.validation.sar_validator import SARValidator, SARValidationResult
from core.data.ingestion.validation.screenshot_generator import ScreenshotGenerator

__all__ = [
    # Integrity
    "IntegrityCheckType",
    "IntegrityConfig",
    "IntegrityIssue",
    "IntegrityResult",
    "IntegritySeverity",
    "IntegrityValidator",
    "validate_integrity",
    # Anomaly
    "AnomalyConfig",
    "AnomalyDetector",
    "AnomalyLocation",
    "AnomalyResult",
    "AnomalySeverity",
    "AnomalyType",
    "DetectedAnomaly",
    "detect_anomalies",
    "detect_anomalies_from_file",
    # Completeness
    "CompletenessCheckType",
    "CompletenessConfig",
    "CompletenessIssue",
    "CompletenessResult",
    "CompletenessSeverity",
    "CompletenessValidator",
    "CoverageRegion",
    "validate_completeness",
    "validate_completeness_from_file",
    # Combined
    "ValidationSuite",
    "ValidationSuiteResult",
    # Image Validation (production workflow)
    "ImageValidationError",
    "MissingBandError",
    "BlankBandError",
    "InvalidCRSError",
    "LoadError",
    "BoundsError",
    "ResolutionError",
    "SARValidationError",
    "DimensionMismatchError",
    "ValidationTimeoutError",
    # Config
    "ValidationConfig",
    "OpticalThresholds",
    "SARThresholds",
    "ScreenshotConfig",
    "PerformanceConfig",
    "ActionConfig",
    "AlertConfig",
    "load_config",
    # Image Validator
    "ImageValidator",
    "ImageValidationResult",
    "ImageMetadata",
    "BandValidationResult",
    "BandStatistics",
    "validate_image",
    # Band/SAR Validators
    "BandValidator",
    "SARValidator",
    "SARValidationResult",
    # Screenshot
    "ScreenshotGenerator",
]


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class ValidationSuiteResult:
    """
    Combined result from all validators.

    Attributes:
        is_valid: Whether all validations passed
        integrity: Integrity validation result
        anomaly: Anomaly detection result
        completeness: Completeness validation result
        overall_score: Combined quality score (0-1)
        summary: Human-readable summary
    """

    is_valid: bool
    integrity: Optional[IntegrityResult] = None
    anomaly: Optional[AnomalyResult] = None
    completeness: Optional[CompletenessResult] = None
    overall_score: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "integrity": self.integrity.to_dict() if self.integrity else None,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "completeness": self.completeness.to_dict() if self.completeness else None,
        }


class ValidationSuite:
    """
    Combined validation suite running all validators.

    Runs integrity, anomaly, and completeness checks and provides
    a unified result with overall quality assessment.

    Example:
        suite = ValidationSuite()
        result = suite.validate("data.tif")
        print(f"Valid: {result.is_valid}, Score: {result.overall_score}")
    """

    def __init__(
        self,
        integrity_config: Optional[IntegrityConfig] = None,
        anomaly_config: Optional[AnomalyConfig] = None,
        completeness_config: Optional[CompletenessConfig] = None,
    ):
        """
        Initialize validation suite.

        Args:
            integrity_config: Config for integrity validation
            anomaly_config: Config for anomaly detection
            completeness_config: Config for completeness validation
        """
        self.integrity_validator = IntegrityValidator(integrity_config)
        self.anomaly_detector = AnomalyDetector(anomaly_config)
        self.completeness_validator = CompletenessValidator(completeness_config)

    def validate(
        self,
        path: Union[str, Path],
        skip_integrity: bool = False,
        skip_anomaly: bool = False,
        skip_completeness: bool = False,
    ) -> ValidationSuiteResult:
        """
        Run all validations on a file.

        Args:
            path: Path to file to validate
            skip_integrity: Skip integrity validation
            skip_anomaly: Skip anomaly detection
            skip_completeness: Skip completeness validation

        Returns:
            ValidationSuiteResult with all validation results
        """
        import logging

        logger = logging.getLogger(__name__)
        path = Path(path)

        integrity_result = None
        anomaly_result = None
        completeness_result = None

        # Run integrity validation
        if not skip_integrity:
            logger.info(f"Running integrity validation on {path}")
            integrity_result = self.integrity_validator.validate_file(path)

        # Run anomaly detection
        if not skip_anomaly:
            logger.info(f"Running anomaly detection on {path}")
            anomaly_result = self.anomaly_detector.detect_from_file(path)

        # Run completeness validation
        if not skip_completeness:
            logger.info(f"Running completeness validation on {path}")
            completeness_result = self.completeness_validator.validate_file(path)

        # Calculate overall validity
        is_valid = True
        if integrity_result and not integrity_result.is_valid:
            is_valid = False
        if completeness_result and not completeness_result.is_complete:
            is_valid = False
        if anomaly_result and anomaly_result.critical_count > 0:
            is_valid = False

        # Calculate overall score
        scores = []
        if integrity_result:
            # Integrity is binary with error weight
            integrity_score = 1.0 if integrity_result.is_valid else max(0.0, 1.0 - integrity_result.error_count * 0.2)
            scores.append(integrity_score)
        if anomaly_result:
            scores.append(anomaly_result.overall_quality_score)
        if completeness_result:
            scores.append(completeness_result.coverage_percentage / 100.0)

        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Generate summary
        summary_parts = []
        if integrity_result:
            status = "OK" if integrity_result.is_valid else f"{integrity_result.error_count} errors"
            summary_parts.append(f"Integrity: {status}")
        if anomaly_result:
            status = f"{len(anomaly_result.anomalies)} anomalies" if anomaly_result.has_anomalies else "OK"
            summary_parts.append(f"Anomaly: {status}")
        if completeness_result:
            status = f"{completeness_result.coverage_percentage:.1f}% coverage"
            summary_parts.append(f"Completeness: {status}")

        summary = "; ".join(summary_parts)

        logger.info(f"Validation complete: valid={is_valid}, score={overall_score:.2f}")

        return ValidationSuiteResult(
            is_valid=is_valid,
            integrity=integrity_result,
            anomaly=anomaly_result,
            completeness=completeness_result,
            overall_score=round(overall_score, 3),
            summary=summary,
        )
