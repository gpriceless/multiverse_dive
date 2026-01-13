# Production Image Validation Requirements

## Overview

This document specifies functional requirements for **production workflow** image validation in the multiverse_dive platform. These requirements define where and how downloaded satellite images are validated during the actual data ingestion and processing pipeline—NOT in E2E tests.

**Key Distinction**: This document specifies validation for the **production system workflow**, while `tests/e2e/IMAGE_VALIDATION_REQUIREMENTS.md` specifies validation for **testing workflows**.

## Scope

Production image validation applies to:
- **All satellite imagery** ingested through the data broker
- **Optical imagery**: Sentinel-2, Landsat 8/9, PlanetScope
- **SAR imagery**: Sentinel-1, RADARSAT
- **Any raster data** that will be processed by analysis algorithms

Validations occur **after download** but **before** data enters algorithm processing or tiling operations.

---

## 1. Workflow Integration Points

### 1.1 Data Ingestion Pipeline Location

**WORKFLOW-001**: Image validation MUST occur in the data ingestion pipeline at the following location:

```
Data Flow:
1. Data Discovery (core/data/discovery/)
   → identifies datasets
2. Data Download (core/data/ingestion/streaming.py)
   → downloads raster files
3. → **[VALIDATION CHECKPOINT]** ← NEW REQUIREMENT
4. Format Conversion (core/data/ingestion/formats/)
   → converts to COG if needed
5. Tiling (core/execution/tiling.py)
   → tiles for memory-efficient processing
6. Algorithm Execution (core/analysis/execution/runner.py)
   → processes data
```

**WORKFLOW-002**: The validation checkpoint MUST be implemented as a new module:
- **Location**: `core/data/ingestion/validation/image_validator.py`
- **Purpose**: Validate downloaded rasters before they enter the processing pipeline
- **Integration**: Called by streaming ingester after download completes

**WORKFLOW-003**: Validation MUST occur in the following functions:
- `StreamingIngester.ingest()` in `core/data/ingestion/streaming.py` (after line 1048)
- Before any call to tiling operations in `TiledAlgorithmRunner` (before line 509 in `core/analysis/execution/tiled_runner.py`)
- Before pipeline execution in `PipelineRunner._execute_task()` (before line 1047 in `core/analysis/execution/runner.py`)

### 1.2 Validation Timing

**WORKFLOW-004**: Validation MUST happen:
- **After**: Raster file is fully downloaded and written to disk
- **Before**: File is loaded into memory for processing
- **Before**: File is tiled or split for distributed processing
- **Before**: File is passed to any analysis algorithm

**WORKFLOW-005**: Validation MUST NOT block concurrent downloads of other datasets. Each downloaded file should be validated independently.

---

## 2. Screenshot Capture in Production

### 2.1 Screenshot Requirements

**FR-SCREENSHOT-PROD-001**: Production workflows MUST optionally capture screenshots of downloaded raster images for debugging and audit trails.

**FR-SCREENSHOT-PROD-002**: Screenshot capture MUST be:
- **Optional**: Controlled by a configuration flag (default: disabled in production, enabled in staging/dev)
- **Non-blocking**: Failures to capture screenshots MUST NOT fail the ingestion workflow
- **Asynchronous**: Screenshot generation should not delay pipeline execution

**FR-SCREENSHOT-PROD-003**: Screenshots MUST be saved to a configurable directory:
- **Default**: `~/.multiverse_dive/screenshots/`
- **Configurable via**: Environment variable `MULTIVERSE_SCREENSHOT_DIR` or config file
- **Structure**: `{screenshot_dir}/{execution_id}/{dataset_id}_{timestamp}.png`

**FR-SCREENSHOT-PROD-004**: Each screenshot MUST capture:
- **Individual bands**: Grayscale visualization of each band
- **RGB composite**: False-color composite for optical imagery (NIR-Red-Green)
- **Metadata overlay**: Dataset ID, acquisition date, bounds, resolution
- **Validation status**: Pass/fail indicator for band validation

**FR-SCREENSHOT-PROD-005**: Screenshots MUST be generated using a lightweight rendering method:
- **Preferred**: Use `matplotlib` or `PIL` for direct image rendering (no browser needed)
- **Alternative**: Use Playwright only if web-based rendering is required (e.g., for interactive maps)
- **Rationale**: Production workflows should not depend on browser automation

### 2.2 Screenshot Configuration

**CONF-SCREENSHOT-001**: Screenshot behavior MUST be configurable via:

```yaml
# config/ingestion.yaml (NEW FILE)
validation:
  screenshots:
    enabled: false  # true for staging/dev, false for production
    output_dir: "~/.multiverse_dive/screenshots"
    format: "png"
    resolution: [1200, 800]  # width, height
    bands_to_render: ["rgb_composite", "nir", "swir1"]  # configurable bands
    metadata_overlay: true
    on_failure_only: false  # if true, only capture screenshots for failed validations
```

**CONF-SCREENSHOT-002**: Screenshot capture MUST log:
- Timestamp of capture
- File path where screenshot was saved
- Dataset ID and execution ID
- Capture duration (for performance monitoring)

---

## 3. Band Validation for Optical/Raster Images

### 3.1 Band Presence Validation

**FR-BAND-PROD-001**: Production workflow MUST verify that all expected bands are present based on data source specifications loaded from `openspec/definitions/datasources/*.yaml`.

**FR-BAND-PROD-002**: For Sentinel-2 imagery, validation MUST check for:
- Required bands for algorithms: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
- Optional bands: B01, B05-B07, B8A, B09, B10

**FR-BAND-PROD-003**: For Landsat imagery, validation MUST check for:
- Required bands: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR1), B7 (SWIR2)
- Optional bands: B1, B9, B10, B11

**FR-BAND-PROD-004**: If required bands are missing, validation MUST:
- **Action**: Reject the dataset
- **Logging**: Log warning with missing band names
- **Provenance**: Record validation failure in lineage tracking
- **Recovery**: Do NOT proceed to algorithm execution

**FR-BAND-PROD-005**: If optional bands are missing, validation MUST:
- **Action**: Log warning but continue processing
- **Metadata**: Mark dataset as "partial" with list of missing bands
- **Algorithm Impact**: Algorithms requiring missing bands should skip or use fallback strategies

### 3.2 Band Non-Blank Validation

**FR-BAND-PROD-006**: Each band MUST be validated to ensure it contains valid data (not all zeros, not all NoData, not constant values).

**FR-BAND-PROD-007**: Validation MUST calculate per-band statistics:
- **Standard deviation**: Bands with std dev < 1.0 flagged as potentially blank
- **Non-zero pixel ratio**: Percentage of pixels with non-zero values (threshold: >5%)
- **NoData pixel ratio**: Percentage of NoData pixels (warning if >95%)
- **Value range**: Min/max values (must span reasonable range)

**FR-BAND-PROD-008**: Validation MUST account for NoData values:
- NoData pixels MUST be excluded from statistical calculations
- NoData value MUST be read from raster metadata (GDAL NoData field)
- If NoData is undefined but zeros dominate, validation SHOULD warn

**FR-BAND-PROD-009**: If a required band is blank, validation MUST:
- **Action**: Reject the dataset
- **Logging**: Log error with band name and statistics (std dev, non-zero ratio)
- **Recovery**: Do NOT proceed to algorithm execution

**FR-BAND-PROD-010**: If an optional band is blank, validation MUST:
- **Action**: Log warning but continue processing
- **Metadata**: Mark band as "invalid" in lineage tracking

### 3.3 Band Loading and Format Validation

**FR-BAND-PROD-011**: Validation MUST verify that raster files load correctly using GDAL/rasterio:
- **Test**: Attempt to open file with `rasterio.open()`
- **On Failure**: Log error and reject dataset
- **On Success**: Extract metadata (CRS, bounds, resolution, band count)

**FR-BAND-PROD-012**: Validation MUST check band dimensions:
- **Consistency**: All bands in a multi-band image must have consistent dimensions (unless multi-resolution sensor like Sentinel-2)
- **Bounds Check**: Image bounds must intersect the requested AOI
- **Resolution Check**: Resolution must match expected values (±10% tolerance)

**FR-BAND-PROD-013**: Validation MUST check band data types:
- **UInt16**: For Sentinel-2 L1C, Landsat reflectance products
- **Float32**: For Sentinel-2 L2A, processed reflectance products
- **Action**: Log warning if data type is unexpected but do not reject

**FR-BAND-PROD-014**: Validation MUST check coordinate reference system (CRS):
- **Requirement**: CRS MUST be defined and valid
- **Projection**: CRS MUST be a recognized EPSG code
- **Transform**: Geotransform MUST provide valid pixel-to-coordinate mapping
- **On Failure**: Reject dataset with error log

### 3.4 Pre-Mosaic Validation

**FR-BAND-PROD-015**: If multiple raster files will be mosaicked, validation MUST occur BEFORE mosaicking:
- **Per-Image Validation**: Each input image validated independently
- **Compatibility Check**: After individual validation, check CRS/resolution compatibility
- **Overlap Check**: Verify that overlapping regions have consistent radiometry (coefficient of variation <20%)

**FR-BAND-PROD-016**: Pre-mosaic compatibility validation MUST check:
- All input images have the same CRS (or can be reprojected)
- All input images have compatible resolutions (within 10% tolerance)
- NoData values are consistently defined across images

---

## 4. SAR Image Handling

### 4.1 SAR-Specific Band Validation

**FR-SAR-PROD-001**: SAR images MUST be validated for polarization band presence:
- **Required**: At least one of VV or VH
- **Optional**: HH, HV (if available)
- **Action**: Reject if no polarization bands found

**FR-SAR-PROD-002**: Validation MUST verify SAR backscatter value ranges:
- **Expected range**: -30 dB to +10 dB (typical land/water values)
- **Warning range**: -50 dB to +20 dB (extended range for unusual targets)
- **Action**: Log warning if >10% of pixels outside expected range

**FR-SAR-PROD-003**: SAR metadata validation MUST check:
- **Acquisition mode**: IW (Interferometric Wide), EW (Extra-Wide), SM (StripMap)
- **Orbit direction**: Ascending or Descending
- **Processing level**: GRD (Ground Range Detected) or SLC (Single Look Complex)
- **Action**: Log metadata; reject only if metadata is missing or corrupted

### 4.2 SAR Non-Blank Validation

**FR-SAR-PROD-004**: "Not blank" validation for SAR MUST account for speckle noise:
- **Threshold**: Standard deviation > 2.0 dB (speckle creates inherent variability)
- **Rationale**: SAR images are noisier than optical; lower std dev threshold would trigger false positives

**FR-SAR-PROD-005**: Validation MUST distinguish between:
- **Valid low-backscatter areas**: Water bodies (-15 to -25 dB is normal)
- **Data dropout**: Entire image at constant NoData or zero value

**FR-SAR-PROD-006**: If SAR image fails validation:
- **Action**: Reject dataset with error log
- **Logging**: Include backscatter statistics (min, max, mean, std dev)
- **Recovery**: Do NOT proceed to SAR flood detection algorithms

---

## 5. Implementation Architecture

### 5.1 Module Structure

**ARCH-001**: Create a new validation module at `core/data/ingestion/validation/`:

```
core/data/ingestion/validation/
├── __init__.py
├── image_validator.py       # Main validator class
├── band_validator.py         # Band presence/content validation
├── sar_validator.py          # SAR-specific validation
├── screenshot_generator.py   # Optional screenshot capture
├── config.py                 # Validation configuration
└── exceptions.py             # Custom validation exceptions
```

**ARCH-002**: Define validation exceptions:

```python
# core/data/ingestion/validation/exceptions.py

class ImageValidationError(Exception):
    """Base exception for image validation failures."""
    pass

class MissingBandError(ImageValidationError):
    """Required band is missing."""
    pass

class BlankBandError(ImageValidationError):
    """Band contains no valid data."""
    pass

class InvalidCRSError(ImageValidationError):
    """CRS is missing or invalid."""
    pass

class LoadError(ImageValidationError):
    """Failed to load raster file."""
    pass
```

**ARCH-003**: Define validation result dataclass:

```python
# core/data/ingestion/validation/image_validator.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class BandValidationResult:
    """Result of validating a single band."""
    band_name: str
    is_valid: bool
    statistics: Dict[str, float]  # mean, std, min, max, non_zero_ratio
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class ImageValidationResult:
    """Result of validating an entire image."""
    dataset_id: str
    is_valid: bool
    band_results: Dict[str, BandValidationResult] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)  # CRS, bounds, resolution
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    validation_duration_seconds: float = 0.0
```

### 5.2 Validator Class Interface

**ARCH-004**: Define the main validator interface:

```python
# core/data/ingestion/validation/image_validator.py

import rasterio
from pathlib import Path
from typing import Optional

class ImageValidator:
    """Validates downloaded raster images before processing."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Validation configuration (thresholds, screenshot settings)
        """
        self.config = config or self._load_default_config()
        self.screenshot_generator = ScreenshotGenerator(self.config)

    def validate(self, raster_path: Path, data_source_spec: dict) -> ImageValidationResult:
        """
        Validate a downloaded raster file.

        Args:
            raster_path: Path to raster file
            data_source_spec: Data source specification from openspec/definitions/datasources/

        Returns:
            ImageValidationResult with validation status and details

        Raises:
            ImageValidationError: If critical validation fails
        """
        pass

    def validate_band_presence(self, dataset: rasterio.DatasetReader, expected_bands: List[str]) -> List[str]:
        """Check if all expected bands are present."""
        pass

    def validate_band_content(self, dataset: rasterio.DatasetReader, band_index: int) -> BandValidationResult:
        """Validate that band contains valid data (not blank)."""
        pass

    def validate_metadata(self, dataset: rasterio.DatasetReader) -> Dict[str, any]:
        """Validate CRS, bounds, resolution, and other metadata."""
        pass
```

### 5.3 Integration with Streaming Ingester

**ARCH-005**: Modify `StreamingIngester.ingest()` to call validator:

```python
# core/data/ingestion/streaming.py (after line 1048)

from core.data.ingestion.validation import ImageValidator, ImageValidationError

class StreamingIngester:
    def __init__(self, ...):
        # ... existing code ...
        self.validator = ImageValidator()

    def ingest(self, source, output_path, ...):
        # ... existing download code ...

        try:
            # NEW: Validate downloaded image
            validation_result = self.validator.validate(
                raster_path=source_path,
                data_source_spec=self._get_data_source_spec(source)
            )

            if not validation_result.is_valid:
                result["status"] = "failed"
                result["errors"].append(f"Image validation failed: {validation_result.errors}")
                logger.error(f"Image validation failed for {source}: {validation_result.errors}")
                return result

            if validation_result.warnings:
                logger.warning(f"Image validation warnings for {source}: {validation_result.warnings}")
                result["validation_warnings"] = validation_result.warnings

            # Store validation result in lineage
            result["validation"] = {
                "is_valid": validation_result.is_valid,
                "band_results": {k: v.__dict__ for k, v in validation_result.band_results.items()},
                "screenshot_path": validation_result.screenshot_path
            }

        except ImageValidationError as e:
            result["status"] = "failed"
            result["errors"].append(f"Image validation error: {str(e)}")
            logger.error(f"Image validation error for {source}: {e}")
            return result

        # ... continue with existing processing code ...
```

### 5.4 Integration with Tiled Runner

**ARCH-006**: Modify `TiledAlgorithmRunner.process()` to validate before tiling:

```python
# core/analysis/execution/tiled_runner.py (before line 509)

from core.data.ingestion.validation import ImageValidator, ImageValidationError

class TiledAlgorithmRunner:
    def __init__(self, ...):
        # ... existing code ...
        self.validator = ImageValidator()

    def process(self, data, bounds, resolution, ...):
        # NEW: If data is a file path (not already loaded), validate it
        if isinstance(data, (str, Path)):
            raster_path = Path(data)

            try:
                validation_result = self.validator.validate(
                    raster_path=raster_path,
                    data_source_spec={}  # TODO: pass from context
                )

                if not validation_result.is_valid:
                    raise ImageValidationError(f"Image validation failed: {validation_result.errors}")

                # Load validated raster into memory
                with rasterio.open(raster_path) as src:
                    data = src.read()

            except ImageValidationError as e:
                logger.error(f"Validation failed before tiling: {e}")
                raise

        # ... continue with existing tiling code ...
```

---

## 6. Validation Criteria and Thresholds

### 6.1 Optical Imagery Thresholds

**THRESHOLD-OPTICAL-001**: Use the following default thresholds (configurable via `config/ingestion.yaml`):

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Band standard deviation** | < 1.0 | Reject as blank |
| **Non-zero pixel ratio** | < 5% | Reject as blank |
| **NoData pixel ratio** | > 95% | Warning (continue processing) |
| **Value range (UInt16)** | Outside [0, 10000] | Warning (log outliers) |
| **Value range (Float32)** | Outside [0.0, 1.0] | Warning (log outliers) |
| **Dimension mismatch** | Bands have different shapes | Reject |
| **CRS missing** | No CRS defined | Reject |
| **Bounds mismatch** | Bounds do not intersect AOI | Reject |

### 6.2 SAR Imagery Thresholds

**THRESHOLD-SAR-001**: Use the following default thresholds:

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Band standard deviation** | < 2.0 dB | Reject as blank |
| **Backscatter range** | Majority outside [-50, +20] dB | Warning |
| **Extreme values** | >50% pixels <-30 or >+10 dB | Warning |
| **NoData pixel ratio** | > 95% | Warning (continue processing) |
| **Polarization missing** | No VV or VH band | Reject |
| **CRS missing** | No CRS defined | Reject |

### 6.3 Configuration Example

**CONF-THRESHOLD-001**: Thresholds MUST be configurable via YAML:

```yaml
# config/ingestion.yaml
validation:
  optical:
    required_bands: ["blue", "green", "red", "nir"]  # Generic names mapped to sensor bands
    optional_bands: ["swir1", "swir2"]
    thresholds:
      std_dev_min: 1.0
      non_zero_ratio_min: 0.05
      nodata_ratio_max: 0.95
      value_range_uint16: [0, 10000]
      value_range_float32: [0.0, 1.0]

  sar:
    required_polarizations: ["VV"]  # or ["VH"]
    optional_polarizations: ["HH", "HV"]
    thresholds:
      std_dev_min_db: 2.0
      backscatter_range_db: [-50, 20]
      extreme_value_threshold_db: [-30, 10]
      extreme_value_ratio_max: 0.5

  actions:
    reject_on_blank_band: true
    reject_on_missing_required_band: true
    reject_on_invalid_crs: true
    warn_on_high_nodata: true
    warn_on_missing_optional_band: true
```

---

## 7. Logging and Provenance

### 7.1 Logging Requirements

**LOG-001**: All validation operations MUST log to the standard logging system with appropriate levels:

```python
import logging
logger = logging.getLogger(__name__)

# Validation start
logger.info(f"Starting image validation for dataset {dataset_id}")

# Validation success
logger.info(f"Image validation passed for dataset {dataset_id} ({duration:.2f}s)")

# Validation warnings
logger.warning(f"Image validation warning for dataset {dataset_id}: {warning_message}")

# Validation errors
logger.error(f"Image validation failed for dataset {dataset_id}: {error_message}")
```

**LOG-002**: Validation logs MUST include:
- Dataset ID or file path
- Validation duration (seconds)
- Band-specific statistics (for failed bands)
- Screenshot path (if captured)
- Provenance tracking ID (for lineage system)

### 7.2 Provenance Integration

**PROV-001**: Validation results MUST be recorded in the lineage tracking system:

```python
# core/data/ingestion/validation/image_validator.py

from core.data.ingestion.persistence.lineage import LineageTracker

class ImageValidator:
    def validate(self, raster_path, data_source_spec):
        # ... validation logic ...

        # Record in lineage
        lineage_tracker = LineageTracker()
        lineage_tracker.record_validation(
            dataset_id=dataset_id,
            validation_result=validation_result,
            timestamp=datetime.now(timezone.utc)
        )

        return validation_result
```

**PROV-002**: Lineage records MUST include:
- Dataset ID
- Validation timestamp
- Validation status (pass/fail)
- Per-band validation results
- Screenshot path (if captured)
- Validation configuration used (thresholds, rules)

---

## 8. Performance Requirements

### 8.1 Performance Targets

**PERF-001**: Band validation MUST complete within:
- **Small images** (<1000x1000 pixels): <1 second
- **Medium images** (1000-5000 pixels): <5 seconds
- **Large images** (5000-10,000 pixels): <10 seconds
- **Very large images** (>10,000 pixels): <30 seconds

**PERF-002**: Screenshot capture (if enabled) MUST complete within:
- **Per image**: <5 seconds
- **Multiple bands**: <10 seconds total

**PERF-003**: Validation MUST NOT load the entire raster into memory:
- Use windowed reads via `rasterio.open().read(window=...)` for statistics
- Read small sample windows (e.g., 512x512 tiles) to estimate statistics
- For large images, sample 10% of pixels to calculate statistics (configurable)

### 8.2 Resource Constraints

**PERF-004**: Validation MUST operate within memory constraints:
- **Target**: <500 MB memory per validation (for images up to 10,000x10,000)
- **Strategy**: Use streaming reads and incremental statistics calculation
- **Fallback**: If image is too large, sample random windows instead of full read

**PERF-005**: Validation MUST be parallelizable:
- Multiple images can be validated concurrently
- Each image validation is independent
- No shared state between validations (thread-safe)

---

## 9. Error Handling and Recovery

### 9.1 Graceful Degradation

**ERROR-001**: If validation fails, the system MUST:
- **Reject the dataset**: Do not proceed to algorithm execution
- **Log detailed error**: Include file path, error type, and error message
- **Record in provenance**: Mark dataset as "validation_failed" in lineage
- **Notify user**: If running via API, return validation error in response
- **Cleanup**: Delete invalid raster files if configured (`cleanup_invalid: true`)

**ERROR-002**: If screenshot capture fails, the system MUST:
- **Continue processing**: Screenshot failure should not block ingestion
- **Log warning**: Indicate screenshot capture failed
- **Do NOT reject dataset**: Only log the failure

### 9.2 Retry Logic

**ERROR-003**: Validation MUST NOT retry automatically:
- If validation fails, it is a data quality issue, not a transient error
- Retrying will produce the same result
- Exception: If validation fails due to file I/O error, retry once

**ERROR-004**: If file loading fails (e.g., corrupt file), validation MUST:
- Attempt to read file up to 3 times (in case of transient I/O error)
- Log each retry attempt
- After 3 failures, reject dataset with LoadError exception

---

## 10. Testing Strategy

### 10.1 Unit Tests

**TEST-UNIT-001**: Unit tests MUST be created for:
- `ImageValidator.validate()` - main validation entry point
- `BandValidator.validate_band_content()` - per-band validation
- `SARValidator.validate_sar_specific()` - SAR-specific validation
- `ScreenshotGenerator.generate()` - screenshot capture (optional)

**TEST-UNIT-002**: Unit tests MUST use mock raster data:
- Create synthetic rasters with `rasterio.MemoryFile()`
- Test valid images (all bands present, valid data)
- Test invalid images (missing bands, blank bands, corrupt data)
- Test edge cases (high NoData ratio, single-band images, multi-resolution)

### 10.2 Integration Tests

**TEST-INTEGRATION-001**: Integration tests MUST validate end-to-end workflows:
- Download real Sentinel-2 image → validate → tile → process
- Download real Sentinel-1 SAR image → validate → process
- Test with real STAC catalog data

**TEST-INTEGRATION-002**: Integration tests MUST be in `tests/integration/test_ingestion_validation.py`:

```python
# tests/integration/test_ingestion_validation.py

import pytest
from core.data.ingestion.streaming import StreamingIngester
from core.data.ingestion.validation import ImageValidator

@pytest.mark.integration
def test_sentinel2_validation_workflow():
    """Test full workflow: discover → download → validate → tile."""
    ingester = StreamingIngester()

    # Download Sentinel-2 image
    result = ingester.ingest(
        source="s3://sentinel-s2-l2a/...",
        output_path="./test_output.tif"
    )

    assert result["status"] == "completed"
    assert "validation" in result
    assert result["validation"]["is_valid"] == True

    # Check that validation results are recorded
    assert "band_results" in result["validation"]
    assert len(result["validation"]["band_results"]) > 0

@pytest.mark.integration
def test_blank_band_rejection():
    """Test that images with blank bands are rejected."""
    # Create test image with one blank band
    # ... setup code ...

    validator = ImageValidator()
    result = validator.validate(test_image_path, data_source_spec)

    assert result.is_valid == False
    assert "blank band" in str(result.errors).lower()
```

### 10.3 E2E Tests

**TEST-E2E-001**: E2E tests in `tests/e2e/test_image_validation.py` MUST test:
- Full ingestion workflow with screenshot capture enabled
- Validation of real downloaded imagery
- Provenance tracking of validation results

---

## 11. Configuration Management

### 11.1 Configuration Files

**CONFIG-001**: Validation configuration MUST be stored in `config/ingestion.yaml`:

```yaml
# config/ingestion.yaml

validation:
  enabled: true  # Set to false to disable all validation

  optical:
    required_bands: ["blue", "green", "red", "nir"]
    optional_bands: ["swir1", "swir2", "coastal", "red_edge"]
    thresholds:
      std_dev_min: 1.0
      non_zero_ratio_min: 0.05
      nodata_ratio_max: 0.95

  sar:
    required_polarizations: ["VV"]
    thresholds:
      std_dev_min_db: 2.0

  screenshots:
    enabled: false  # true for dev/staging
    output_dir: "~/.multiverse_dive/screenshots"
    format: "png"
    resolution: [1200, 800]
    on_failure_only: false

  performance:
    max_validation_time_seconds: 30
    sample_ratio: 1.0  # 1.0 = full image, 0.1 = sample 10% of pixels
    parallel_bands: true  # validate bands in parallel

  actions:
    reject_on_blank_band: true
    reject_on_missing_required_band: true
    reject_on_invalid_crs: true
    cleanup_invalid_files: false  # if true, delete invalid raster files

  logging:
    level: "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_statistics: true  # log per-band statistics
```

**CONFIG-002**: Configuration MUST be overridable via environment variables:

```bash
export MULTIVERSE_VALIDATION_ENABLED=true
export MULTIVERSE_VALIDATION_SCREENSHOT_DIR=/mnt/artifacts/screenshots
export MULTIVERSE_VALIDATION_SAMPLE_RATIO=0.1  # For large images
```

### 11.2 Band Mapping Configuration

**CONFIG-003**: Generic band names MUST map to sensor-specific bands via configuration:

```yaml
# openspec/definitions/datasources/sentinel2.yaml

sensor: "Sentinel-2"
bands:
  blue: ["B02"]
  green: ["B03"]
  red: ["B04"]
  nir: ["B08", "B8A"]  # Multiple options: use B08 or B8A
  swir1: ["B11"]
  swir2: ["B12"]
  red_edge: ["B05", "B06", "B07"]
  coastal: ["B01"]
  water_vapor: ["B09"]
  cirrus: ["B10"]

validation:
  required_bands: ["blue", "green", "red", "nir"]  # Generic names
  optional_bands: ["swir1", "swir2", "red_edge"]
```

**CONFIG-004**: Data source specifications MUST be loaded from `openspec/definitions/datasources/` to determine expected bands.

---

## 12. Deployment Considerations

### 12.1 Production Deployment

**DEPLOY-001**: In production environments:
- Screenshot capture MUST be **disabled** by default
- Validation MUST be **enabled** by default
- Validation logs MUST go to centralized logging (CloudWatch, Elasticsearch)
- Provenance records MUST be stored in persistent database

**DEPLOY-002**: Validation performance MUST be monitored:
- Track validation duration per image
- Alert if validation takes >30 seconds for typical images
- Track validation pass/fail rates
- Alert if validation failure rate >10%

### 12.2 Staging/Development Deployment

**DEPLOY-003**: In staging/dev environments:
- Screenshot capture MUST be **enabled** by default
- Validation MUST be **enabled** by default
- Screenshots MUST be accessible for manual inspection
- Validation failures MUST trigger notifications (Slack, email)

### 12.3 Edge Deployment

**DEPLOY-004**: For edge deployments (Raspberry Pi, field devices):
- Screenshot capture SHOULD be **disabled** (limited resources)
- Validation MUST be **enabled** but with reduced sampling (`sample_ratio: 0.1`)
- Validation timeout MUST be lower (10 seconds max)

---

## 13. Documentation Requirements

### 13.1 User Documentation

**DOC-USER-001**: User-facing documentation MUST be created at `docs/user/image_validation.md`:
- Explain what validation does
- List validation criteria and thresholds
- Provide examples of validation failures and resolutions
- Explain how to configure validation behavior

**DOC-USER-002**: API documentation MUST include validation results in response schemas:

```yaml
# Example API response with validation results
{
  "execution_id": "abc123",
  "status": "completed",
  "datasets": [
    {
      "dataset_id": "S2A_MSIL2A_20250110_T10TGK",
      "download_status": "success",
      "validation": {
        "is_valid": true,
        "duration_seconds": 2.3,
        "bands": {
          "B02": {"is_valid": true, "std_dev": 234.5, "non_zero_ratio": 0.87},
          "B03": {"is_valid": true, "std_dev": 312.1, "non_zero_ratio": 0.89},
          ...
        },
        "warnings": [],
        "screenshot_url": null
      }
    }
  ]
}
```

### 13.2 Developer Documentation

**DOC-DEV-001**: Developer documentation MUST be created at `docs/developer/validation_architecture.md`:
- Describe validation module architecture
- Explain integration points in ingestion pipeline
- Provide code examples for extending validation rules
- Document validation exception hierarchy

**DOC-DEV-002**: Inline code documentation MUST include:
- Docstrings for all public classes and methods
- Type hints for all function signatures
- Examples in docstrings

---

## 14. Acceptance Criteria

### 14.1 Functional Acceptance

**AC-FUNC-001**: The production validation system MUST:
- Validate all downloaded images before they reach algorithms
- Reject images with missing required bands
- Reject images with blank bands (std dev below threshold)
- Reject images with invalid CRS or metadata
- Log all validation events with appropriate detail
- Record validation results in provenance system

**AC-FUNC-002**: The validation system MUST integrate seamlessly with:
- `StreamingIngester.ingest()` in streaming.py
- `TiledAlgorithmRunner.process()` in tiled_runner.py
- `PipelineRunner` in runner.py
- Lineage tracking system

### 14.2 Performance Acceptance

**AC-PERF-001**: Validation performance MUST meet targets:
- <10 seconds for images up to 10,000x10,000 pixels
- <500 MB memory usage per validation
- No impact on concurrent download operations

### 14.3 Quality Acceptance

**AC-QUAL-001**: Test coverage MUST be:
- >90% code coverage for validation module
- Unit tests for all validation functions
- Integration tests for end-to-end workflows
- E2E tests with real satellite imagery

### 14.4 Documentation Acceptance

**AC-DOC-001**: Documentation MUST be complete:
- User guide for validation behavior
- Developer guide for extending validation
- API documentation for validation results
- Configuration examples

---

## 15. Implementation Phases

### 15.1 Phase 1: Core Validation (Week 1-2)

**PHASE-1-001**: Implement core validation module:
- Create `core/data/ingestion/validation/` directory structure
- Implement `ImageValidator` class with band presence/content validation
- Implement `BandValidationResult` and `ImageValidationResult` dataclasses
- Write unit tests for core validation functions

**PHASE-1-002**: Integrate with StreamingIngester:
- Modify `StreamingIngester.ingest()` to call validator after download
- Handle validation exceptions and update result status
- Add validation results to lineage tracking

### 15.2 Phase 2: SAR and Screenshot Support (Week 3)

**PHASE-2-001**: Implement SAR-specific validation:
- Create `SARValidator` class
- Implement polarization band validation
- Implement speckle-aware blank detection
- Write unit tests for SAR validation

**PHASE-2-002**: Implement screenshot capture (optional):
- Create `ScreenshotGenerator` class
- Implement matplotlib-based band rendering
- Add configuration for screenshot behavior
- Write unit tests for screenshot generation

### 15.3 Phase 3: Integration and Testing (Week 4)

**PHASE-3-001**: Integrate with algorithm execution:
- Modify `TiledAlgorithmRunner` to validate before tiling
- Modify `PipelineRunner` to validate input data
- Add validation checks to algorithm entry points

**PHASE-3-002**: Comprehensive testing:
- Write integration tests with real Sentinel-2/Sentinel-1 data
- Write E2E tests with full ingestion workflow
- Performance testing with large images
- Load testing with concurrent validations

### 15.4 Phase 4: Documentation and Deployment (Week 5)

**PHASE-4-001**: Complete documentation:
- User guide for validation
- Developer guide for extending validation
- API documentation updates
- Configuration examples

**PHASE-4-002**: Deployment preparation:
- Configure production validation settings
- Set up monitoring and alerting
- Deploy to staging environment for testing
- Production rollout

---

## 16. Future Enhancements

### 16.1 Planned Features (Not in Scope for Initial Implementation)

**FUTURE-001**: Advanced validation capabilities:
- **Radiometric calibration validation**: Detect sensor artifacts, striping, banding
- **Cloud mask validation**: Verify cloud masks align with optical imagery
- **Cross-sensor consistency**: Compare radiometry between Sentinel-2 and Landsat
- **ML-based anomaly detection**: Use trained models to detect subtle quality issues
- **Temporal consistency**: Validate that time-series imagery has consistent radiometry

**FUTURE-002**: Real-time validation:
- Streaming validation for large datasets (validate tiles as they download)
- Progressive validation (start algorithm execution before full validation completes)

**FUTURE-003**: Enhanced provenance:
- Store per-pixel quality flags (not just per-band)
- Track validation failures over time (dataset quality trends)
- Generate validation quality reports (PDF, HTML)

---

## 17. Comparison with E2E Test Requirements

This document is **complementary** to `tests/e2e/IMAGE_VALIDATION_REQUIREMENTS.md`:

| Aspect | E2E Test Requirements | Production Workflow Requirements (This Doc) |
|--------|----------------------|-------------------------------------------|
| **Purpose** | Validate test infrastructure and downloaded test images | Validate production data during ingestion pipeline |
| **Location** | `tests/e2e/test_*.py` | `core/data/ingestion/validation/` |
| **Execution** | During test runs (pytest) | During production ingestion workflow |
| **Screenshot Capture** | Always enabled (for debugging tests) | Optional, configurable (default: disabled in prod) |
| **Screenshot Tool** | Playwright (browser-based) | matplotlib/PIL (lightweight, no browser) |
| **Validation Strictness** | Strict (all bands must be valid) | Configurable (optional bands may be missing) |
| **Error Handling** | Test fails if validation fails | Workflow rejects invalid data, continues with warnings |
| **Provenance** | Not tracked (test artifacts only) | Recorded in lineage system |

---

## 18. References

### 18.1 Related Documents

- **E2E Test Requirements**: `tests/e2e/IMAGE_VALIDATION_REQUIREMENTS.md`
- **OpenSpec Architecture**: `OPENSPEC.md`
- **Roadmap**: `ROADMAP.md` (current platform status)
- **Data Source Specifications**: `openspec/definitions/datasources/`

### 18.2 Codebase References

- **Streaming Ingestion**: `core/data/ingestion/streaming.py` (line 974-1161)
- **Tiling System**: `core/execution/tiling.py` (entire module)
- **Tiled Algorithm Runner**: `core/analysis/execution/tiled_runner.py` (line 363-802)
- **Pipeline Execution**: `core/analysis/execution/runner.py` (line 673-1196)
- **Lineage Tracking**: `core/data/ingestion/persistence/lineage.py`
- **Data Discovery**: `core/data/discovery/base.py` (discovery result format)

### 18.3 Algorithm Requirements

- **NDWI Optical Flood Detection**: `core/analysis/library/baseline/flood/ndwi_optical.py` (requires green, NIR bands)
- **SAR Threshold Flood Detection**: `core/analysis/library/baseline/flood/threshold_sar.py` (requires VV/VH polarization)
- **dNBR Wildfire Detection**: `core/analysis/library/baseline/wildfire/dnbr.py` (requires NIR, SWIR bands)

---

## 19. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-12 | Claude Opus 4.5 | Initial production workflow requirements document |

---

## 20. Glossary

- **AOI**: Area of Interest (geographic region for analysis)
- **COG**: Cloud-Optimized GeoTIFF
- **CRS**: Coordinate Reference System (e.g., EPSG:4326)
- **dNBR**: Differenced Normalized Burn Ratio (wildfire severity index)
- **GDAL**: Geospatial Data Abstraction Library
- **GRD**: Ground Range Detected (SAR processing level)
- **HAND**: Height Above Nearest Drainage (flood model)
- **NDWI**: Normalized Difference Water Index (flood detection)
- **NIR**: Near-Infrared band
- **NoData**: Raster pixel value indicating missing/invalid data
- **SAR**: Synthetic Aperture Radar
- **STAC**: SpatioTemporal Asset Catalog
- **SWIR**: Shortwave Infrared band
- **VH**: SAR polarization (Vertical transmit, Horizontal receive)
- **VV**: SAR polarization (Vertical transmit, Vertical receive)

