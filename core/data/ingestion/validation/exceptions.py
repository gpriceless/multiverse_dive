"""
Custom Exceptions for Image Validation.

Provides a hierarchy of validation exceptions for different failure modes
in the image validation pipeline.
"""


class ImageValidationError(Exception):
    """
    Base exception for image validation failures.

    All validation-specific exceptions inherit from this class,
    allowing for broad exception handling when needed.

    Attributes:
        message: Human-readable error description
        details: Additional context about the failure
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class MissingBandError(ImageValidationError):
    """
    Required band is missing from the image.

    Raised when a band that is required for processing (e.g., NIR for NDWI)
    is not present in the downloaded image.

    Attributes:
        band_name: Name of the missing band
        expected_bands: List of all expected bands
        found_bands: List of bands that were found
    """

    def __init__(
        self,
        band_name: str,
        expected_bands: list = None,
        found_bands: list = None,
    ):
        message = f"Required band '{band_name}' is missing"
        details = {
            "band_name": band_name,
            "expected_bands": expected_bands or [],
            "found_bands": found_bands or [],
        }
        super().__init__(message, details)
        self.band_name = band_name
        self.expected_bands = expected_bands or []
        self.found_bands = found_bands or []


class BlankBandError(ImageValidationError):
    """
    Band contains no valid data (all zeros, NoData, or constant values).

    Raised when a band fails the "not blank" validation check,
    indicating the band does not contain meaningful data.

    Attributes:
        band_name: Name of the blank band
        statistics: Statistics that triggered the failure (std_dev, non_zero_ratio, etc.)
    """

    def __init__(
        self,
        band_name: str,
        statistics: dict = None,
    ):
        stats = statistics or {}
        message = f"Band '{band_name}' is blank or contains invalid data"
        details = {
            "band_name": band_name,
            "std_dev": stats.get("std_dev"),
            "non_zero_ratio": stats.get("non_zero_ratio"),
            "nodata_ratio": stats.get("nodata_ratio"),
        }
        super().__init__(message, details)
        self.band_name = band_name
        self.statistics = statistics or {}


class InvalidCRSError(ImageValidationError):
    """
    Coordinate Reference System is missing or invalid.

    Raised when the image has no CRS defined, an unrecognized CRS,
    or an invalid geotransform.

    Attributes:
        crs_value: The invalid or missing CRS value
        reason: Explanation of why the CRS is invalid
    """

    def __init__(self, crs_value: str = None, reason: str = None):
        message = "CRS is missing or invalid"
        if reason:
            message = f"CRS is invalid: {reason}"
        details = {
            "crs_value": crs_value,
            "reason": reason,
        }
        super().__init__(message, details)
        self.crs_value = crs_value
        self.reason = reason


class LoadError(ImageValidationError):
    """
    Failed to load raster file.

    Raised when the raster file cannot be opened or read,
    typically due to file corruption or format issues.

    Attributes:
        file_path: Path to the file that failed to load
        original_error: The underlying exception that caused the failure
    """

    def __init__(self, file_path: str, original_error: Exception = None):
        message = f"Failed to load raster file: {file_path}"
        if original_error:
            message = f"{message} - {str(original_error)}"
        details = {
            "file_path": file_path,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, details)
        self.file_path = file_path
        self.original_error = original_error


class BoundsError(ImageValidationError):
    """
    Image bounds do not match expected bounds.

    Raised when the image does not intersect the requested AOI
    or has invalid geographic bounds.

    Attributes:
        image_bounds: Bounds of the image (minx, miny, maxx, maxy)
        expected_bounds: Expected/requested bounds
        reason: Explanation of the bounds mismatch
    """

    def __init__(
        self,
        image_bounds: tuple = None,
        expected_bounds: tuple = None,
        reason: str = None,
    ):
        message = "Image bounds are invalid or do not match expected bounds"
        if reason:
            message = f"Bounds error: {reason}"
        details = {
            "image_bounds": image_bounds,
            "expected_bounds": expected_bounds,
            "reason": reason,
        }
        super().__init__(message, details)
        self.image_bounds = image_bounds
        self.expected_bounds = expected_bounds
        self.reason = reason


class ResolutionError(ImageValidationError):
    """
    Image resolution does not match expected resolution.

    Raised when the pixel resolution is outside acceptable tolerance
    from the expected resolution.

    Attributes:
        actual_resolution: Actual pixel resolution (x, y)
        expected_resolution: Expected pixel resolution (x, y)
        tolerance: Tolerance percentage that was exceeded
    """

    def __init__(
        self,
        actual_resolution: tuple = None,
        expected_resolution: tuple = None,
        tolerance: float = None,
    ):
        message = "Image resolution does not match expected resolution"
        details = {
            "actual_resolution": actual_resolution,
            "expected_resolution": expected_resolution,
            "tolerance_percent": tolerance,
        }
        super().__init__(message, details)
        self.actual_resolution = actual_resolution
        self.expected_resolution = expected_resolution
        self.tolerance = tolerance


class SARValidationError(ImageValidationError):
    """
    SAR-specific validation failure.

    Raised for SAR imagery validation issues such as missing
    polarization bands or invalid backscatter values.

    Attributes:
        reason: Specific reason for SAR validation failure
        polarization: Polarization that failed validation (if applicable)
    """

    def __init__(self, reason: str, polarization: str = None):
        message = f"SAR validation failed: {reason}"
        details = {
            "reason": reason,
            "polarization": polarization,
        }
        super().__init__(message, details)
        self.reason = reason
        self.polarization = polarization


class DimensionMismatchError(ImageValidationError):
    """
    Band dimensions are inconsistent within the image.

    Raised when bands in a multi-band image have different dimensions,
    which can cause processing errors.

    Attributes:
        band_dimensions: Dictionary mapping band names to their dimensions
    """

    def __init__(self, band_dimensions: dict = None):
        message = "Band dimensions are inconsistent within the image"
        details = {"band_dimensions": band_dimensions or {}}
        super().__init__(message, details)
        self.band_dimensions = band_dimensions or {}


class ValidationTimeoutError(ImageValidationError):
    """
    Validation timed out.

    Raised when validation exceeds the configured timeout,
    typically for very large images.

    Attributes:
        timeout_seconds: The timeout that was exceeded
        elapsed_seconds: How long validation ran before timeout
    """

    def __init__(self, timeout_seconds: float, elapsed_seconds: float = None):
        message = f"Validation timed out after {timeout_seconds} seconds"
        details = {
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds,
        }
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
