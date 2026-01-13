"""
Main Image Validator for Production Workflows.

Provides the primary validation interface for downloaded raster images,
integrating band validation, metadata checks, and optional screenshot capture.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.data.ingestion.validation.config import (
    ValidationConfig,
    load_config,
)
from core.data.ingestion.validation.exceptions import (
    BlankBandError,
    BoundsError,
    DimensionMismatchError,
    ImageValidationError,
    InvalidCRSError,
    LoadError,
    MissingBandError,
    ValidationTimeoutError,
)

logger = logging.getLogger(__name__)

# Optional rasterio import
try:
    import rasterio
    from rasterio.crs import CRS

    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    CRS = None
    HAS_RASTERIO = False


@dataclass
class BandStatistics:
    """
    Statistical measurements for a single band.

    Attributes:
        mean: Mean value (excluding NoData)
        std_dev: Standard deviation
        min_val: Minimum value
        max_val: Maximum value
        non_zero_ratio: Ratio of non-zero pixels (0.0 to 1.0)
        nodata_ratio: Ratio of NoData pixels (0.0 to 1.0)
        valid_pixel_count: Number of valid (non-NoData) pixels
        total_pixel_count: Total number of pixels
    """

    mean: float = 0.0
    std_dev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    non_zero_ratio: float = 0.0
    nodata_ratio: float = 0.0
    valid_pixel_count: int = 0
    total_pixel_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": round(self.mean, 4) if not np.isnan(self.mean) else None,
            "std_dev": round(self.std_dev, 4) if not np.isnan(self.std_dev) else None,
            "min": round(self.min_val, 4) if not np.isnan(self.min_val) else None,
            "max": round(self.max_val, 4) if not np.isnan(self.max_val) else None,
            "non_zero_ratio": round(self.non_zero_ratio, 4),
            "nodata_ratio": round(self.nodata_ratio, 4),
            "valid_pixel_count": self.valid_pixel_count,
            "total_pixel_count": self.total_pixel_count,
        }


@dataclass
class BandValidationResult:
    """
    Result of validating a single band.

    Attributes:
        band_name: Name or index of the band
        band_index: 1-indexed band number
        is_valid: Whether the band passed validation
        is_required: Whether this band is required
        statistics: Band statistics
        warnings: List of warning messages
        errors: List of error messages
    """

    band_name: str
    band_index: int
    is_valid: bool
    is_required: bool = True
    statistics: BandStatistics = field(default_factory=BandStatistics)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "band_name": self.band_name,
            "band_index": self.band_index,
            "is_valid": self.is_valid,
            "is_required": self.is_required,
            "statistics": self.statistics.to_dict(),
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ImageMetadata:
    """
    Metadata extracted from the image.

    Attributes:
        crs: Coordinate Reference System (EPSG string)
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        resolution: Pixel resolution (x_res, y_res)
        width: Image width in pixels
        height: Image height in pixels
        band_count: Number of bands
        dtype: Data type of the raster
        nodata_value: NoData value (if defined)
        driver: GDAL driver name
    """

    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[float, float]] = None
    width: int = 0
    height: int = 0
    band_count: int = 0
    dtype: str = ""
    nodata_value: Optional[float] = None
    driver: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crs": self.crs,
            "bounds": list(self.bounds) if self.bounds else None,
            "resolution": list(self.resolution) if self.resolution else None,
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "nodata_value": self.nodata_value,
            "driver": self.driver,
        }


@dataclass
class ImageValidationResult:
    """
    Complete result of image validation.

    Attributes:
        dataset_id: Identifier for the dataset
        file_path: Path to the validated file
        is_valid: Whether the image passed all required validations
        band_results: Per-band validation results
        metadata: Image metadata
        warnings: List of warning messages
        errors: List of error messages
        screenshot_path: Path to screenshot (if captured)
        validation_duration_seconds: Time taken for validation
    """

    dataset_id: str
    file_path: str
    is_valid: bool
    band_results: Dict[str, BandValidationResult] = field(default_factory=dict)
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    validation_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "file_path": self.file_path,
            "is_valid": self.is_valid,
            "band_results": {k: v.to_dict() for k, v in self.band_results.items()},
            "metadata": self.metadata.to_dict(),
            "warnings": self.warnings,
            "errors": self.errors,
            "screenshot_path": self.screenshot_path,
            "validation_duration_seconds": round(self.validation_duration_seconds, 3),
        }


class ImageValidator:
    """
    Main validator for downloaded raster images.

    Validates images for:
    - Band presence (required vs optional)
    - Band content (not blank, valid data)
    - Metadata (CRS, bounds, resolution)
    - Format and loading

    Example:
        validator = ImageValidator()
        result = validator.validate(
            raster_path=Path("/data/image.tif"),
            data_source_spec={"sensor": "Sentinel-2"},
        )
        if result.is_valid:
            print("Image passed validation")
        else:
            print(f"Validation failed: {result.errors}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the image validator.

        Args:
            config: Validation configuration (uses defaults if not provided)
        """
        self.config = config or load_config()

        # Import band validator and SAR validator lazily to avoid circular imports
        self._band_validator = None
        self._sar_validator = None
        self._screenshot_generator = None

    @property
    def band_validator(self):
        """Lazy-load band validator."""
        if self._band_validator is None:
            from core.data.ingestion.validation.band_validator import BandValidator

            self._band_validator = BandValidator(self.config)
        return self._band_validator

    @property
    def sar_validator(self):
        """Lazy-load SAR validator."""
        if self._sar_validator is None:
            from core.data.ingestion.validation.sar_validator import SARValidator

            self._sar_validator = SARValidator(self.config)
        return self._sar_validator

    @property
    def screenshot_generator(self):
        """Lazy-load screenshot generator."""
        if self._screenshot_generator is None and self.config.screenshots.enabled:
            from core.data.ingestion.validation.screenshot_generator import (
                ScreenshotGenerator,
            )

            self._screenshot_generator = ScreenshotGenerator(self.config)
        return self._screenshot_generator

    def validate(
        self,
        raster_path: Union[str, Path],
        data_source_spec: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
        expected_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> ImageValidationResult:
        """
        Validate a downloaded raster file.

        Args:
            raster_path: Path to the raster file
            data_source_spec: Data source specification with band mappings
            dataset_id: Optional dataset identifier
            expected_bounds: Expected geographic bounds for intersection check

        Returns:
            ImageValidationResult with validation status and details

        Raises:
            ImageValidationError: If critical validation fails and actions require rejection
        """
        if not self.config.enabled:
            # Validation disabled - return success
            return ImageValidationResult(
                dataset_id=dataset_id or str(raster_path),
                file_path=str(raster_path),
                is_valid=True,
                warnings=["Validation is disabled"],
            )

        start_time = time.time()
        raster_path = Path(raster_path)
        dataset_id = dataset_id or raster_path.stem

        # Initialize result
        result = ImageValidationResult(
            dataset_id=dataset_id,
            file_path=str(raster_path),
            is_valid=True,
        )

        try:
            # Step 1: Load and validate file can be opened
            dataset, metadata = self._load_and_extract_metadata(raster_path, result)
            result.metadata = metadata

            if dataset is None:
                result.is_valid = False
                return result

            # Step 2: Validate metadata (CRS, bounds)
            self._validate_metadata(metadata, expected_bounds, result)

            # Step 3: Determine if SAR or optical
            is_sar = self._is_sar_image(data_source_spec, metadata)

            # Step 4: Validate bands
            if is_sar:
                self._validate_sar_bands(dataset, data_source_spec, result)
            else:
                self._validate_optical_bands(dataset, data_source_spec, result)

            # Step 5: Generate screenshot if enabled
            if self.config.screenshots.enabled:
                if not self.config.screenshots.on_failure_only or not result.is_valid:
                    self._capture_screenshot(raster_path, result)

            dataset.close()

        except ValidationTimeoutError as e:
            result.is_valid = False
            result.errors.append(str(e))
            logger.error(f"Validation timed out for {raster_path}: {e}")

        except ImageValidationError as e:
            result.is_valid = False
            result.errors.append(str(e))
            logger.error(f"Validation error for {raster_path}: {e}")

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected error: {str(e)}")
            logger.exception(f"Unexpected validation error for {raster_path}")

        finally:
            result.validation_duration_seconds = time.time() - start_time

            # Check for timeout alert
            if result.validation_duration_seconds > self.config.alerts.hung_process_timeout_seconds:
                self._alert_hung_process(result)

            # Log completion
            if result.is_valid:
                logger.info(
                    f"Image validation passed for {dataset_id} "
                    f"({result.validation_duration_seconds:.2f}s)"
                )
            else:
                logger.warning(
                    f"Image validation failed for {dataset_id}: {result.errors}"
                )

        return result

    def _load_and_extract_metadata(
        self,
        raster_path: Path,
        result: ImageValidationResult,
    ) -> Tuple[Optional[Any], ImageMetadata]:
        """Load raster file and extract metadata."""
        if not HAS_RASTERIO:
            raise LoadError(str(raster_path), RuntimeError("rasterio not installed"))

        if not raster_path.exists():
            result.is_valid = False
            result.errors.append(f"File does not exist: {raster_path}")
            return None, ImageMetadata()

        retries = 0
        last_error = None

        while retries < self.config.actions.max_load_retries:
            try:
                dataset = rasterio.open(raster_path)

                # Extract metadata
                metadata = ImageMetadata(
                    crs=dataset.crs.to_string() if dataset.crs else None,
                    bounds=(
                        dataset.bounds.left,
                        dataset.bounds.bottom,
                        dataset.bounds.right,
                        dataset.bounds.top,
                    )
                    if dataset.bounds
                    else None,
                    resolution=(
                        abs(dataset.transform.a),
                        abs(dataset.transform.e),
                    )
                    if dataset.transform
                    else None,
                    width=dataset.width,
                    height=dataset.height,
                    band_count=dataset.count,
                    dtype=str(dataset.dtypes[0]) if dataset.dtypes else "",
                    nodata_value=dataset.nodata,
                    driver=dataset.driver,
                )

                return dataset, metadata

            except Exception as e:
                retries += 1
                last_error = e
                logger.warning(f"Failed to load {raster_path} (attempt {retries}): {e}")
                time.sleep(0.5 * retries)

        # All retries failed
        result.is_valid = False
        result.errors.append(f"Failed to load file after {retries} attempts: {last_error}")
        raise LoadError(str(raster_path), last_error)

    def _validate_metadata(
        self,
        metadata: ImageMetadata,
        expected_bounds: Optional[Tuple[float, float, float, float]],
        result: ImageValidationResult,
    ) -> None:
        """Validate image metadata."""
        # Check CRS
        if metadata.crs is None:
            if self.config.actions.reject_on_invalid_crs:
                result.is_valid = False
                result.errors.append("CRS is missing")
            else:
                result.warnings.append("CRS is missing")
        else:
            # Validate CRS is a recognized format
            try:
                if HAS_RASTERIO:
                    crs = CRS.from_string(metadata.crs)
                    if not crs.is_valid:
                        result.warnings.append(f"CRS may not be valid: {metadata.crs}")
            except Exception as e:
                result.warnings.append(f"Could not parse CRS: {e}")

        # Check bounds
        if metadata.bounds is None:
            result.warnings.append("Image bounds could not be determined")
        elif expected_bounds is not None:
            # Check intersection
            if not self._bounds_intersect(metadata.bounds, expected_bounds):
                result.is_valid = False
                result.errors.append(
                    f"Image bounds {metadata.bounds} do not intersect "
                    f"expected bounds {expected_bounds}"
                )

        # Check resolution
        if metadata.resolution is None:
            result.warnings.append("Image resolution could not be determined")

    def _bounds_intersect(
        self,
        bounds1: Tuple[float, float, float, float],
        bounds2: Tuple[float, float, float, float],
    ) -> bool:
        """Check if two bounding boxes intersect."""
        minx1, miny1, maxx1, maxy1 = bounds1
        minx2, miny2, maxx2, maxy2 = bounds2

        return not (
            maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1
        )

    def _is_sar_image(
        self,
        data_source_spec: Optional[Dict[str, Any]],
        metadata: ImageMetadata,
    ) -> bool:
        """Determine if image is SAR based on spec or heuristics."""
        if data_source_spec:
            sensor = data_source_spec.get("sensor", "").lower()
            if "sentinel-1" in sensor or "sar" in sensor or "radar" in sensor:
                return True
            if "radarsat" in sensor:
                return True

            # Check for polarization bands in spec
            bands = data_source_spec.get("bands", {})
            if any(pol in str(bands).upper() for pol in ["VV", "VH", "HH", "HV"]):
                return True

        return False

    def _validate_optical_bands(
        self,
        dataset: Any,
        data_source_spec: Optional[Dict[str, Any]],
        result: ImageValidationResult,
    ) -> None:
        """Validate optical imagery bands."""
        # Determine expected bands from spec or defaults
        expected_bands = self._get_expected_optical_bands(data_source_spec)

        # Validate using band validator
        band_results = self.band_validator.validate_bands(
            dataset=dataset,
            expected_bands=expected_bands,
            required_bands=self.config.required_optical_bands,
        )

        for band_name, band_result in band_results.items():
            result.band_results[band_name] = band_result

            # Update overall validity
            if not band_result.is_valid:
                if band_result.is_required and self.config.actions.reject_on_blank_band:
                    result.is_valid = False
                    result.errors.extend(band_result.errors)
                else:
                    result.warnings.extend(band_result.errors)

            result.warnings.extend(band_result.warnings)

    def _validate_sar_bands(
        self,
        dataset: Any,
        data_source_spec: Optional[Dict[str, Any]],
        result: ImageValidationResult,
    ) -> None:
        """Validate SAR imagery bands."""
        # Use SAR validator
        sar_result = self.sar_validator.validate(
            dataset=dataset,
            data_source_spec=data_source_spec,
        )

        # Merge results
        for band_name, band_result in sar_result.band_results.items():
            result.band_results[band_name] = band_result

            if not band_result.is_valid:
                if band_result.is_required:
                    result.is_valid = False
                    result.errors.extend(band_result.errors)
                else:
                    result.warnings.extend(band_result.errors)

            result.warnings.extend(band_result.warnings)

        result.errors.extend(sar_result.errors)
        result.warnings.extend(sar_result.warnings)

    def _get_expected_optical_bands(
        self,
        data_source_spec: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Get expected band names from data source spec."""
        if data_source_spec and "bands" in data_source_spec:
            return data_source_spec["bands"]

        # Default band mapping for common sensors
        return {
            "blue": ["B02", "B2"],
            "green": ["B03", "B3"],
            "red": ["B04", "B4"],
            "nir": ["B08", "B8", "B8A", "B5"],
            "swir1": ["B11", "B6"],
            "swir2": ["B12", "B7"],
        }

    def _capture_screenshot(
        self,
        raster_path: Path,
        result: ImageValidationResult,
    ) -> None:
        """Capture screenshot of the image."""
        if self.screenshot_generator is None:
            return

        try:
            screenshot_path = self.screenshot_generator.generate(
                raster_path=raster_path,
                dataset_id=result.dataset_id,
                validation_result=result,
            )
            result.screenshot_path = str(screenshot_path) if screenshot_path else None

            # Handle temporary retention
            if self.config.screenshots.retention == "temporary" and result.is_valid:
                # Schedule for deletion (handled elsewhere or immediately)
                self._schedule_screenshot_deletion(screenshot_path)

        except Exception as e:
            logger.warning(f"Failed to capture screenshot: {e}")
            result.warnings.append(f"Screenshot capture failed: {e}")

    def _schedule_screenshot_deletion(self, screenshot_path: Optional[Path]) -> None:
        """Schedule screenshot for deletion (temporary retention)."""
        if screenshot_path and screenshot_path.exists():
            try:
                screenshot_path.unlink()
                logger.debug(f"Deleted temporary screenshot: {screenshot_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary screenshot: {e}")

    def _alert_hung_process(self, result: ImageValidationResult) -> None:
        """Alert about hung/slow validation process."""
        if self.config.alerts.enabled:
            message = (
                f"Validation took {result.validation_duration_seconds:.1f}s "
                f"(threshold: {self.config.alerts.hung_process_timeout_seconds}s) "
                f"for {result.dataset_id}"
            )
            logger.warning(f"ALERT: {message}")

            if self.config.alerts.alert_callback:
                try:
                    self.config.alerts.alert_callback(
                        "hung_validation", message, result.to_dict()
                    )
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")


def validate_image(
    raster_path: Union[str, Path],
    data_source_spec: Optional[Dict[str, Any]] = None,
    config: Optional[ValidationConfig] = None,
) -> ImageValidationResult:
    """
    Convenience function to validate an image.

    Args:
        raster_path: Path to raster file
        data_source_spec: Optional data source specification
        config: Optional validation configuration

    Returns:
        ImageValidationResult with validation status
    """
    validator = ImageValidator(config)
    return validator.validate(raster_path, data_source_spec)
