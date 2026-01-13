# Image Validation Implementation Summary

## Overview

This document provides a high-level summary of the image validation requirements for the multiverse_dive platform. It serves as a quick reference for developers implementing the validation system.

## Key Documents

1. **Production Workflow Requirements**: `/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`
   - **Purpose**: Specifies how validation integrates into the actual data ingestion pipeline
   - **Scope**: Production system workflow (not tests)
   - **Audience**: Platform developers implementing core validation logic

2. **E2E Test Requirements**: `/tests/e2e/IMAGE_VALIDATION_REQUIREMENTS.md`
   - **Purpose**: Specifies validation for E2E testing infrastructure
   - **Scope**: Test workflows only
   - **Audience**: Test developers

## Critical Distinction

| Aspect | Production Workflow | E2E Tests |
|--------|-------------------|-----------|
| **When** | During actual data ingestion | During test execution |
| **Where** | `core/data/ingestion/validation/` | `tests/e2e/test_*.py` |
| **Screenshot Tool** | matplotlib/PIL (lightweight) | Playwright (browser-based) |
| **Screenshot Default** | Disabled in production | Always enabled for tests |
| **Validation Target** | Real satellite data from users | Test fixtures and sample data |
| **Error Handling** | Reject invalid data, log warnings | Fail test if validation fails |

## Production Workflow Integration Points

### 1. Data Ingestion Pipeline

**Location**: `core/data/ingestion/streaming.py`

**Integration Point**: After download, before processing (line ~1048)

```python
# core/data/ingestion/streaming.py

from core.data.ingestion.validation import ImageValidator

class StreamingIngester:
    def ingest(self, source, output_path, ...):
        # ... download code ...

        # NEW: Validate downloaded image
        validator = ImageValidator()
        validation_result = validator.validate(source_path, data_source_spec)

        if not validation_result.is_valid:
            # Reject dataset
            result["status"] = "failed"
            result["errors"] = validation_result.errors
            return result

        # ... continue processing ...
```

### 2. Tiled Processing

**Location**: `core/analysis/execution/tiled_runner.py`

**Integration Point**: Before tiling (line ~509)

```python
# core/analysis/execution/tiled_runner.py

from core.data.ingestion.validation import ImageValidator

class TiledAlgorithmRunner:
    def process(self, data, bounds, resolution, ...):
        # NEW: Validate if data is a file path
        if isinstance(data, (str, Path)):
            validator = ImageValidator()
            validation_result = validator.validate(Path(data), data_source_spec)

            if not validation_result.is_valid:
                raise ImageValidationError(f"Validation failed: {validation_result.errors}")

            # Load validated raster
            with rasterio.open(data) as src:
                data = src.read()

        # ... continue tiling ...
```

### 3. Pipeline Execution

**Location**: `core/analysis/execution/runner.py`

**Integration Point**: Before algorithm execution (line ~1047)

```python
# core/analysis/execution/runner.py

from core.data.ingestion.validation import ImageValidator

class PipelineRunner:
    def _execute_task(self, task, execution_id, input_data, shared_state):
        # NEW: Validate input data if it contains raster paths
        if "raster_path" in input_data:
            validator = ImageValidator()
            validation_result = validator.validate(
                input_data["raster_path"],
                data_source_spec={}
            )

            if not validation_result.is_valid:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Input validation failed: {validation_result.errors}"
                )

        # ... continue task execution ...
```

## Validation Requirements

### Screenshot Capture

**For Production Workflows**:
- **Default**: Disabled (configurable)
- **Purpose**: Debugging and audit trails
- **Tool**: matplotlib or PIL (lightweight, no browser)
- **Output**: `~/.multiverse_dive/screenshots/`
- **Configuration**: `config/ingestion.yaml`

**For E2E Tests**:
- **Default**: Always enabled
- **Purpose**: Visual test validation
- **Tool**: Playwright (browser-based)
- **Output**: `tests/e2e/artifacts/screenshots/`

### Band Validation (Optical/Raster)

**Required Checks**:
1. **Band Presence**: All expected bands exist (based on data source spec)
2. **Band Content**: Bands are not blank (std dev > threshold)
3. **Band Loading**: Raster loads correctly with GDAL/rasterio
4. **Metadata**: CRS, bounds, resolution are valid
5. **Pre-Mosaic**: Images are compatible before merging

**Thresholds** (configurable):
- Standard deviation: > 1.0 (optical), > 2.0 dB (SAR)
- Non-zero pixel ratio: > 5%
- NoData pixel ratio: < 95% (warning threshold)

### SAR Image Handling

**Special Considerations**:
- **Speckle noise**: SAR images have higher inherent variability
- **Polarization bands**: Must have VV or VH (or both)
- **Backscatter range**: Typical -30 to +10 dB, extended -50 to +20 dB
- **Std dev threshold**: > 2.0 dB (higher than optical due to speckle)

## Implementation Architecture

### New Module Structure

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

### Key Classes

**ImageValidator**:
```python
class ImageValidator:
    def validate(self, raster_path: Path, data_source_spec: dict) -> ImageValidationResult:
        """Validate a downloaded raster file."""
        pass
```

**ImageValidationResult**:
```python
@dataclass
class ImageValidationResult:
    dataset_id: str
    is_valid: bool
    band_results: Dict[str, BandValidationResult]
    metadata: Dict[str, any]
    warnings: List[str]
    errors: List[str]
    screenshot_path: Optional[str]
    validation_duration_seconds: float
```

## Configuration

### Production Configuration

```yaml
# config/ingestion.yaml

validation:
  enabled: true

  optical:
    required_bands: ["blue", "green", "red", "nir"]
    optional_bands: ["swir1", "swir2"]
    thresholds:
      std_dev_min: 1.0
      non_zero_ratio_min: 0.05

  sar:
    required_polarizations: ["VV"]
    thresholds:
      std_dev_min_db: 2.0

  screenshots:
    enabled: false  # true for staging/dev
    output_dir: "~/.multiverse_dive/screenshots"
    format: "png"
    on_failure_only: false

  actions:
    reject_on_blank_band: true
    reject_on_missing_required_band: true
    reject_on_invalid_crs: true
```

### Band Mapping

```yaml
# openspec/definitions/datasources/sentinel2.yaml

sensor: "Sentinel-2"
bands:
  blue: ["B02"]
  green: ["B03"]
  red: ["B04"]
  nir: ["B08", "B8A"]
  swir1: ["B11"]
  swir2: ["B12"]

validation:
  required_bands: ["blue", "green", "red", "nir"]
  optional_bands: ["swir1", "swir2"]
```

## Implementation Phases

### Phase 1: Core Validation (Week 1-2)
- Create validation module structure
- Implement `ImageValidator` class
- Implement band presence/content validation
- Write unit tests
- Integrate with `StreamingIngester`

### Phase 2: SAR and Screenshot Support (Week 3)
- Implement `SARValidator` for SAR-specific validation
- Implement `ScreenshotGenerator` for optional screenshots
- Add configuration loading
- Write SAR-specific tests

### Phase 3: Integration and Testing (Week 4)
- Integrate with `TiledAlgorithmRunner`
- Integrate with `PipelineRunner`
- Write integration tests with real data
- Performance testing

### Phase 4: Documentation and Deployment (Week 5)
- Complete user documentation
- Complete developer documentation
- Deploy to staging
- Production rollout

## Acceptance Criteria

### Functional
- ✅ All downloaded images are validated before algorithm execution
- ✅ Images with missing required bands are rejected
- ✅ Images with blank bands are rejected
- ✅ Validation results are recorded in provenance system

### Performance
- ✅ Validation completes in <10 seconds for typical images (up to 10K x 10K pixels)
- ✅ Memory usage <500 MB per validation
- ✅ No impact on concurrent downloads

### Quality
- ✅ >90% code coverage for validation module
- ✅ Unit tests for all validation functions
- ✅ Integration tests with real Sentinel-2 and Sentinel-1 data
- ✅ E2E tests with full ingestion workflow

## Next Steps

1. **Review Requirements**:
   - Read `/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md` in detail
   - Review existing code at integration points
   - Clarify any ambiguities or questions

2. **Create Module Structure**:
   - Create `core/data/ingestion/validation/` directory
   - Create skeleton files with class definitions
   - Set up unit test structure

3. **Implement Core Validation**:
   - Implement `ImageValidator.validate()`
   - Implement band presence validation
   - Implement band content validation (blank detection)
   - Write unit tests

4. **Integrate with Ingestion**:
   - Modify `StreamingIngester.ingest()` to call validator
   - Handle validation exceptions
   - Add validation results to lineage tracking

5. **Test with Real Data**:
   - Test with real Sentinel-2 imagery
   - Test with real Sentinel-1 SAR imagery
   - Test edge cases (blank bands, missing bands, corrupt files)

## Key References

### Codebase
- **Streaming Ingestion**: `core/data/ingestion/streaming.py` (line 974-1161)
- **Tiling System**: `core/execution/tiling.py` (entire module)
- **Tiled Runner**: `core/analysis/execution/tiled_runner.py` (line 363-802)
- **Pipeline Runner**: `core/analysis/execution/runner.py` (line 673-1196)
- **Lineage Tracking**: `core/data/ingestion/persistence/lineage.py`

### Algorithms
- **NDWI Flood Detection**: Requires green, NIR bands
- **SAR Flood Detection**: Requires VV/VH polarization
- **dNBR Wildfire**: Requires NIR, SWIR bands

### Data Sources
- **Sentinel-2**: 13 spectral bands (B01-B12, B8A)
- **Landsat 8/9**: 11 bands (B1-B11)
- **Sentinel-1**: SAR with VV/VH polarizations

## Questions?

For detailed requirements, see:
- **Production Workflow**: `/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`
- **E2E Test Requirements**: `/tests/e2e/IMAGE_VALIDATION_REQUIREMENTS.md`
- **Platform Architecture**: `OPENSPEC.md`
- **Current Status**: `ROADMAP.md`

