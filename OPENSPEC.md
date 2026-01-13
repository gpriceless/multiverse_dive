# OpenSpec: Active Specifications

**Status:** In-Progress Specifications Only
**Last Updated:** 2026-01-13
**Archive:** See `docs/OPENSPEC_ARCHIVE.md` for completed specifications

---

## Completed Work (Archived)

The following have been fully implemented and moved to the archive:

- Core Schemas (event, intent, datasource, pipeline, provenance, quality)
- Data Broker Architecture (13 provider integrations)
- Data Ingestion & Normalization Pipeline
- Analysis & Modeling Layer (8 baseline algorithms)
- Multi-Sensor Fusion Engine
- Forecast & Scenario Integration
- Quality Control & Validation
- Agent Architecture (orchestrator, discovery, pipeline, quality, reporting)
- API & Deployment (FastAPI, Docker, Kubernetes, cloud configs)

**Total:** 170K+ lines, 518+ passing tests

---

## Active Specifications

### 1. Distributed Raster Processing

**Status:** COMPLETE (2026-01-13)

#### Problem Statement

Earth observation datasets are massive:
- Single Sentinel-2 scene: 500MB-5GB (100km x 100km at 10m resolution)
- Continental analysis: 100,000km² = 1,000+ scenes = 500GB-5TB
- Current serial tiled processing: 20-30 minutes for 100km² on laptop
- Memory constraints: Existing tiling works but doesn't parallelize across cores
- Download bottleneck: Must download entire scenes before processing begins

#### Goals

1. **Laptop-Scale Parallelization:** Leverage all CPU cores for 4-8x speedup
2. **Cloud-Scale Distribution:** Process continental areas on Spark/Flink clusters
3. **Streaming Ingestion:** Never download full scenes - stream only needed tiles
4. **Transparent Scaling:** Same API works on laptop or 100-node cluster

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event Specification                          │
│         (Area: 1000km², Event: flood.coastal)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Discovery (STAC)                        │
│  Finds: 25 Sentinel-1 scenes, 30 Sentinel-2 scenes, DEM        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Virtual Raster Index (NEW)                      │
│  - Build GDAL VRT from STAC query results                      │
│  - No download - just index tile locations                      │
│  - Track which tiles needed for AOI                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Router (NEW)                         │
│  ├─ Small (<100 tiles): Serial execution                       │
│  ├─ Medium (100-1000): Dask local cluster                      │
│  └─ Large (1000+): Sedona on Spark                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Distributed Tile Processing                       │
│  - Stream tiles via HTTP range requests                        │
│  - Process in parallel across cores/workers                    │
│  - Memory-mapped intermediate results                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamed Results                             │
│  - Mosaic tiles on-the-fly                                     │
│  - Output as COG with overviews                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Technology Stack

| Technology | Use Case | Status |
|-----------|----------|--------|
| **Dask + Rasterio** | Local parallelization (laptop/workstation) | COMPLETE |
| **Apache Sedona 1.5+** | Cloud-scale Spark-based processing | Planned |
| **GDAL Virtual Rasters** | Lightweight tile indexing | COMPLETE |

#### New Files Required

```
core/data/ingestion/
└── virtual_index.py          # GDAL VRT from STAC results

core/analysis/execution/
├── dask_tiled.py             # Dask-based parallel tile processing
├── router.py                 # Execution environment router
└── sedona_backend.py         # Apache Sedona integration (future)
```

#### Success Metrics

- [x] Laptop: 1000km² analysis in <10 minutes (currently 30+ min) - IMPLEMENTED
- [ ] Cloud: 100,000km² analysis in <1 hour (requires Sedona)
- [x] Memory: Peak <4GB on laptop regardless of AOI size - IMPLEMENTED
- [x] Streaming: Zero full scene downloads (tile streaming only) - IMPLEMENTED
- [x] Parallelization: 80%+ CPU utilization on multi-core laptops - IMPLEMENTED

#### Implementation Summary (2026-01-13)

**Files Created:**
- `core/data/ingestion/virtual_index.py` - VirtualRasterIndex, STACVRTBuilder (DP-1)
- `core/analysis/execution/dask_tiled.py` - DaskTileProcessor, parallel engine (DP-2)
- `core/analysis/execution/dask_adapters.py` - Algorithm adapters (DP-3)
- `core/analysis/execution/router.py` - ExecutionRouter, auto selection (DP-4)
- `tests/test_dask_execution.py` - Integration tests (DP-5)

**Key Classes:**
- `VirtualRasterIndex` - Lazy tile access via VRT
- `DaskTileProcessor` - Parallel tile processing
- `ExecutionRouter` - Automatic backend selection
- `DaskAlgorithmAdapter` - Wrap any algorithm for Dask

---

### 2. Production Image Validation

**Status:** Requirements defined, implementation pending

#### Overview

Add production-grade image validation to the ingestion workflow. When satellite imagery is downloaded for algorithm processing, validate band integrity and optionally capture screenshots BEFORE images are processed or merged.

#### Requirements

- Validate band presence and content for optical imagery (Sentinel-2, Landsat)
- Ensure images are not blank before merging into mosaics
- Handle SAR imagery differently (speckle-aware validation)
- Optional screenshot capture for debugging/audit trails

#### Validation Thresholds

```yaml
optical_validation:
  min_std_dev: 1.0          # Band not blank if std dev > 1.0
  max_nodata_percent: 50    # Fail if >50% nodata
  expected_value_range:
    sentinel2_l1c: [0, 10000]
    sentinel2_l2a: [0, 10000]
    landsat8_l1: [0, 65535]
    landsat8_l2: [0, 10000]

sar_validation:
  min_std_dev_db: 2.0       # Higher threshold due to speckle
  backscatter_range_db: [-30, 10]
  required_polarizations: ["VV", "VH"]
```

#### New Files Required

```
core/data/ingestion/validation/
├── __init__.py
├── exceptions.py             # ValidationError, BlankBandError, etc.
├── config.py                 # Validation configuration schema
├── image_validator.py        # Base ImageValidator class
├── band_validator.py         # OpticalBandValidator
├── sar_validator.py          # SARValidator (speckle-aware)
└── screenshot_generator.py   # Optional screenshot capture

config/
└── ingestion.yaml            # Validation settings
```

#### Integration Points

1. `core/data/ingestion/streaming.py` (~line 1048) - Post-download validation
2. `core/analysis/execution/tiled_runner.py` (~line 509) - Pre-merge validation
3. `core/analysis/execution/runner.py` (~line 1047) - Pipeline integration

#### Success Criteria

- [ ] All optical bands validated before processing
- [ ] Blank band detection catches >95% of corrupt/missing data
- [ ] SAR validation correctly handles speckle noise
- [ ] Screenshots captured when enabled (configurable)
- [ ] Zero false positives on valid imagery
- [ ] <500ms validation overhead per image

**Full Requirements:** See `docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`

---

## Architecture Decisions Pending

### Distributed Processing

- [ ] Tile caching strategy: S3 vs local SSD vs memory
- [ ] Spark session lifecycle: per-job vs persistent pool
- [ ] Failure handling: retry tiles vs fail entire job
- [ ] Progress reporting: polling vs push notifications
- [ ] Tile size optimization: 256x256 vs 512x512 pixels (TBD via profiling)

### Image Validation

- [ ] Screenshot storage location and retention policy
- [ ] Integration with existing QC pipeline
- [ ] Alerting strategy for validation failures

---

## Reference

### Related Documentation

- `docs/OPENSPEC_ARCHIVE.md` - Complete specifications for implemented features
- `docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md` - Detailed validation requirements
- `ROADMAP.md` - Implementation roadmap and task breakdown
- `FIXES.md` - Bug tracking (P0 bugs must be fixed before new features)

### Test Commands

```bash
# Run all tests
./run_tests.py

# Run specific categories
./run_tests.py flood
./run_tests.py wildfire
./run_tests.py schemas

# Run with specific algorithm
./run_tests.py --algorithm sar
./run_tests.py --algorithm ndwi
```
