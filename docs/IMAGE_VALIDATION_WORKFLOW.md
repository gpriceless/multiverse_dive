# Image Validation Workflow Diagram

## Production Data Ingestion Flow

This diagram illustrates where image validation integrates into the production data ingestion and processing pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

   USER REQUEST
   (area, time, event type)
        │
        ▼
┌─────────────────┐
│ 1. DATA         │  core/data/discovery/
│    DISCOVERY    │  Identify datasets from STAC catalogs
│                 │  (Sentinel-2, Sentinel-1, Landsat, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. DATA         │  core/data/ingestion/streaming.py
│    DOWNLOAD     │  StreamingIngester.ingest()
│                 │  Download raster files (COG, GeoTIFF)
└────────┬────────┘
         │
         │
         ▼
    ┌───────────────────────────────────────────────────────────┐
    │                                                             │
    │  ★★★ NEW: VALIDATION CHECKPOINT ★★★                       │
    │                                                             │
    │  Location: core/data/ingestion/validation/                │
    │  Called by: StreamingIngester.ingest() (after line 1048) │
    │                                                             │
    │  Checks:                                                    │
    │  1. Band Presence: Are all expected bands present?        │
    │  2. Band Content: Are bands valid (not blank)?            │
    │  3. Band Loading: Does raster load correctly?             │
    │  4. Metadata: Is CRS/bounds/resolution valid?             │
    │  5. Screenshot (optional): Capture debug screenshots      │
    │                                                             │
    │  Result:                                                    │
    │  • PASS → Continue to next step                            │
    │  • FAIL → Reject dataset, log error, record in provenance │
    │                                                             │
    └───────────────────────────────────────────────────────────┘
         │
         │ (validation passed)
         ▼
┌─────────────────┐
│ 3. FORMAT       │  core/data/ingestion/formats/
│    CONVERSION   │  Convert to COG (Cloud-Optimized GeoTIFF)
│                 │  Normalize format for processing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. TILING       │  core/execution/tiling.py
│    (Optional)   │  TiledAlgorithmRunner.process()
│                 │  Split large images into tiles for memory efficiency
│                 │
│                 │  ★★★ SECOND VALIDATION CHECKPOINT ★★★
│                 │  Before tiling, validate if data is a file path
│                 │  (line ~509 in tiled_runner.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. ALGORITHM    │  core/analysis/execution/runner.py
│    EXECUTION    │  PipelineRunner._execute_task()
│                 │  Run analysis algorithms (NDWI, SAR, dNBR, etc.)
│                 │
│                 │  ★★★ THIRD VALIDATION CHECKPOINT ★★★
│                 │  Before execution, validate input data
│                 │  (line ~1047 in runner.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. RESULT       │  core/analysis/execution/
│    GENERATION   │  Generate flood extent, burn severity, etc.
│                 │  Produce decision products
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. PROVENANCE   │  core/data/ingestion/persistence/lineage.py
│    TRACKING     │  Record validation results, algorithm outputs
│                 │  Maintain full lineage of data processing
└─────────────────┘
```

---

## Validation Checkpoint Detail

### What Happens at the Validation Checkpoint

```
┌──────────────────────────────────────────────────────────────┐
│  VALIDATION CHECKPOINT                                        │
│  core/data/ingestion/validation/image_validator.py          │
└──────────────────────────────────────────────────────────────┘

INPUT: Downloaded raster file path
       Data source specification (from openspec/definitions/datasources/)

   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: LOAD RASTER METADATA                                │
│ • Open file with rasterio/GDAL                              │
│ • Extract: CRS, bounds, resolution, band count, data type   │
│ • Check: File loads without errors                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: VALIDATE METADATA                                   │
│ • CRS: Must be defined and valid                            │
│ • Bounds: Must intersect requested AOI                      │
│ • Resolution: Must match expected values (±10% tolerance)   │
│ • Data Type: Check for expected type (UInt16, Float32)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: VALIDATE BAND PRESENCE                              │
│ • Load expected bands from data source spec                 │
│ • Check: All required bands exist (e.g., blue, green, NIR)  │
│ • Check: Optional bands (flag if missing, continue)         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: VALIDATE BAND CONTENT (Per Band)                    │
│ • Read band data (windowed or sampled for large images)     │
│ • Calculate statistics:                                      │
│   - Standard deviation (> 1.0 for optical, > 2.0 dB for SAR)│
│   - Non-zero pixel ratio (> 5%)                             │
│   - NoData pixel ratio (< 95%)                              │
│   - Value range (within expected bounds)                    │
│ • Determine: Is band blank or valid?                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: SCREENSHOT CAPTURE (Optional)                       │
│ • If enabled: Render bands with matplotlib/PIL              │
│ • Generate: RGB composite, individual bands, metadata       │
│ • Save to: ~/.multiverse_dive/screenshots/                  │
│ • Log: Screenshot path for debugging                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: GENERATE VALIDATION RESULT                          │
│ • Aggregate: All band results, metadata checks              │
│ • Determine: Overall validation status (pass/fail)          │
│ • Record: Warnings (optional bands missing, high NoData)    │
│ • Record: Errors (required bands missing, blank bands)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
OUTPUT: ImageValidationResult
        {
          is_valid: true/false,
          band_results: {...},
          warnings: [...],
          errors: [...],
          screenshot_path: "...",
          validation_duration_seconds: 2.3
        }
```

---

## Decision Flow

### What Happens Based on Validation Result?

```
┌──────────────────────────────────────┐
│  ImageValidationResult               │
│  is_valid: true/false                │
└───────────────┬──────────────────────┘
                │
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
  [PASS]            [FAIL]
       │                 │
       │                 │
       ▼                 ▼
┌──────────────┐    ┌──────────────────────────────────────┐
│ CONTINUE     │    │ REJECT DATASET                        │
│ PROCESSING   │    │                                       │
│              │    │ Actions:                              │
│ • Proceed to │    │ • Do NOT proceed to next stage        │
│   format     │    │ • Log error with details              │
│   conversion │    │ • Record failure in provenance system │
│              │    │ • Return error to user/API            │
│ • Proceed to │    │ • Optionally: Delete invalid file     │
│   tiling     │    │                                       │
│              │    │ Error Response:                        │
│ • Proceed to │    │ {                                     │
│   algorithm  │    │   "status": "failed",                 │
│   execution  │    │   "error": "validation_failed",       │
│              │    │   "details": {                        │
│ • Record     │    │     "missing_bands": ["B02", "B03"],  │
│   validation │    │     "blank_bands": ["B08"],           │
│   success in │    │     "invalid_crs": false              │
│   provenance │    │   }                                   │
│              │    │ }                                     │
└──────────────┘    └──────────────────────────────────────┘
```

---

## Example Workflows

### Example 1: Sentinel-2 Flood Detection

```
1. User Request:
   • Area: Miami, FL (25.7617° N, 80.1918° W)
   • Time: 2025-01-10 to 2025-01-11
   • Event: Flood

2. Data Discovery:
   • Found: Sentinel-2 L2A scene S2A_MSIL2A_20250110_T17RNJ

3. Data Download:
   • Download: All bands (B01-B12, B8A) as COG
   • Size: 1.2 GB

4. ★★★ VALIDATION CHECKPOINT ★★★
   • Load metadata: CRS=EPSG:32617, Bounds=[...], Resolution=10m
   • Check bands: B02, B03, B04, B08 (required for NDWI) ✅ Present
   • Check content: B02 (std=234.5) ✅, B03 (std=312.1) ✅, B04 (std=289.3) ✅, B08 (std=401.2) ✅
   • Result: ✅ PASS

5. Format Conversion:
   • Already COG → Skip

6. Tiling:
   • Tile size: 512x512 pixels
   • Number of tiles: 144

7. Algorithm Execution:
   • Run NDWI (Normalized Difference Water Index)
   • Formula: (Green - NIR) / (Green + NIR)
   • Generate flood extent map

8. Result:
   • Flood extent: 12.3 km²
   • Confidence: 0.87
   • Screenshot: /screenshots/abc123/flood_extent.png
```

### Example 2: Sentinel-1 SAR Flood Detection (Validation Failure)

```
1. User Request:
   • Area: Houston, TX
   • Time: 2025-01-08
   • Event: Flood

2. Data Discovery:
   • Found: Sentinel-1 GRD scene S1A_IW_GRDH_20250108_T123456

3. Data Download:
   • Download: VV and VH polarization bands
   • Size: 800 MB

4. ★★★ VALIDATION CHECKPOINT ★★★
   • Load metadata: CRS=EPSG:32615, Bounds=[...], Resolution=10m ✅
   • Check bands: VV ✅ Present, VH ✅ Present
   • Check content VV: std_dev=0.8 dB ❌ BELOW THRESHOLD (2.0 dB)
   • Result: ❌ FAIL - VV band appears blank (low std dev)

5. Dataset Rejected:
   • Error: "VV band validation failed: std_dev=0.8 dB (threshold: 2.0 dB)"
   • Action: Do NOT proceed to algorithm execution
   • Provenance: Record validation failure
   • User Response:
     {
       "status": "failed",
       "error": "Image validation failed: VV band is blank or corrupted",
       "details": {
         "band": "VV",
         "std_dev": 0.8,
         "threshold": 2.0,
         "reason": "Band appears to contain no valid backscatter data"
       }
     }

6-8. NOT EXECUTED
```

### Example 3: Landsat 8 Wildfire Detection (Partial Validation)

```
1. User Request:
   • Area: Northern California
   • Time: 2025-01-05 to 2025-01-06
   • Event: Wildfire

2. Data Discovery:
   • Found: Landsat 8 scene LC08_L1TP_20250105_T044033

3. Data Download:
   • Download: All bands (B1-B11)
   • Size: 950 MB

4. ★★★ VALIDATION CHECKPOINT ★★★
   • Load metadata: CRS=EPSG:32610, Bounds=[...] ✅
   • Check required bands: B3 (Green) ✅, B4 (Red) ✅, B5 (NIR) ✅, B6 (SWIR1) ✅, B7 (SWIR2) ✅
   • Check optional bands: B9 (Cirrus) ❌ MISSING
   • Check content: All required bands valid (std > 1.0) ✅
   • Result: ✅ PASS (with warning: B9 missing)
   • Warning: "Optional band B9 (Cirrus) is missing"

5. Format Conversion:
   • Convert to COG with compression

6. Tiling:
   • Tile size: 1024x1024 pixels
   • Number of tiles: 64

7. Algorithm Execution:
   • Run dNBR (Differenced Normalized Burn Ratio)
   • Formula: (Pre-NIR - Pre-SWIR) / (Pre-NIR + Pre-SWIR) - (Post-NIR - Post-SWIR) / (Post-NIR + Post-SWIR)
   • Generate burn severity map

8. Result:
   • Burned area: 234.5 km²
   • Burn severity: High (dNBR > 0.66)
   • Warning: "Cirrus band not used for cloud screening"
```

---

## Configuration Impact

### Screenshot Capture Configuration

```yaml
# config/ingestion.yaml

validation:
  screenshots:
    enabled: true  # Set to false in production
    output_dir: "~/.multiverse_dive/screenshots"
    on_failure_only: true  # Only capture screenshots for failed validations
```

**Impact**:
- `enabled: false` → No screenshots, faster validation
- `enabled: true` → Screenshots saved for all images
- `on_failure_only: true` → Screenshots only for validation failures (debugging)

### Band Validation Configuration

```yaml
# config/ingestion.yaml

validation:
  optical:
    thresholds:
      std_dev_min: 1.0  # Lower = more strict (reject more images)
      non_zero_ratio_min: 0.05  # Higher = more strict
      nodata_ratio_max: 0.95  # Lower = more strict

  sar:
    thresholds:
      std_dev_min_db: 2.0  # Lower = more strict
```

**Impact**:
- Lower thresholds → More images rejected (stricter validation)
- Higher thresholds → More images pass (lenient validation)

### Action Configuration

```yaml
# config/ingestion.yaml

validation:
  actions:
    reject_on_blank_band: true  # Reject if band is blank
    reject_on_missing_required_band: true  # Reject if required band missing
    reject_on_invalid_crs: true  # Reject if CRS is invalid
    warn_on_high_nodata: true  # Warn if NoData ratio > threshold (but continue)
    warn_on_missing_optional_band: true  # Warn if optional band missing (but continue)
```

**Impact**:
- `reject_on_*: true` → Strict validation (reject invalid data)
- `reject_on_*: false` → Lenient validation (log warnings but continue)

---

## Performance Considerations

### Validation Duration by Image Size

| Image Size | Validation Time | Notes |
|------------|----------------|--------|
| 1K x 1K pixels | <1 second | Full read, all bands |
| 5K x 5K pixels | 2-5 seconds | Full read, all bands |
| 10K x 10K pixels | 5-10 seconds | Windowed read or sampling |
| 20K x 20K pixels | 10-30 seconds | Sampling recommended (`sample_ratio: 0.1`) |

### Memory Usage

| Image Size | Bands | Memory Usage |
|------------|-------|--------------|
| 1K x 1K | 3 bands (RGB) | ~12 MB (Float32) |
| 5K x 5K | 10 bands (Sentinel-2) | ~1 GB |
| 10K x 10K | 10 bands | ~4 GB |
| 20K x 20K | 10 bands | ~16 GB (requires sampling) |

### Optimization Strategies

**For Large Images**:
```yaml
# config/ingestion.yaml

validation:
  performance:
    sample_ratio: 0.1  # Validate 10% of pixels (faster for large images)
    parallel_bands: true  # Validate bands in parallel
    max_validation_time_seconds: 30  # Timeout validation after 30 seconds
```

**Impact**:
- `sample_ratio: 0.1` → 10x faster validation (but less accurate statistics)
- `parallel_bands: true` → N bands validated in parallel (N = CPU cores)

---

## Summary

### Key Integration Points

1. **StreamingIngester** (`core/data/ingestion/streaming.py`, line ~1048)
   - Validate after download, before format conversion

2. **TiledAlgorithmRunner** (`core/analysis/execution/tiled_runner.py`, line ~509)
   - Validate before tiling if data is a file path

3. **PipelineRunner** (`core/analysis/execution/runner.py`, line ~1047)
   - Validate input data before task execution

### Key Files to Create

- `core/data/ingestion/validation/image_validator.py` (main validator)
- `core/data/ingestion/validation/band_validator.py` (band validation logic)
- `core/data/ingestion/validation/sar_validator.py` (SAR-specific validation)
- `core/data/ingestion/validation/screenshot_generator.py` (optional screenshots)
- `core/data/ingestion/validation/config.py` (configuration loading)
- `core/data/ingestion/validation/exceptions.py` (custom exceptions)
- `config/ingestion.yaml` (validation configuration)

### Key Tests to Create

- `tests/unit/test_image_validator.py` (unit tests for validator)
- `tests/integration/test_ingestion_validation.py` (integration tests)
- `tests/e2e/test_image_validation.py` (E2E tests with real data)

