# OpenSpec Archive: Completed Specifications

**Status:** IMPLEMENTED - All specifications in this document have been fully implemented and tested.
**Archive Date:** 2026-01-13
**See:** `OPENSPEC.md` for active/in-progress specifications only.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Common Schema Definitions](#common-schema-definitions)
3. [Intent Schema](#intent-schema)
4. [Event Schema](#event-schema)
5. [Event Class Registry](#event-class-registry)
6. [Data Source Schema](#data-source-schema)
7. [Pipeline Schema](#pipeline-schema)
8. [Provenance Schema](#provenance-schema)
9. [Data Broker Architecture](#data-broker-architecture)
10. [Intelligent Sensor Selection](#intelligent-sensor-selection)
11. [Data Ingestion & Normalization](#data-ingestion--normalization)
12. [Analysis & Modeling Layer](#analysis--modeling-layer)
13. [Multi-Sensor Fusion Engine](#multi-sensor-fusion-engine)
14. [Forecast & Scenario Integration](#forecast--scenario-integration)
15. [Quality Control & Validation](#quality-control--validation)
16. [Agent Architecture](#agent-architecture)
17. [Implementation Phases](#implementation-phases)

---

## Architecture Overview

Design: Cloud-native, agent-orchestrated platform that transforms (area, time window, event type) into reproducible decision products. Core principle: situation-agnostic specifications enable the same agents/pipelines to handle floods, wildfires, storms, and other hazards.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Spec Format | JSON Schema + YAML | Validation + human readability |
| Stack | Python + FastAPI | Geospatial ecosystem (GDAL, rasterio, xarray) |
| Data Sources | Extensible framework | Start open, design for any provider |
| Agent Model | Hierarchical orchestrator | Clear delegation, easier debugging |
| Deployment | Serverless-first | Burst scaling for time-critical response |

### Project Structure

```
multiverse_dive/
├── openspec/                    # Core specification schemas
│   ├── schemas/
│   │   ├── event.schema.json
│   │   ├── intent.schema.json
│   │   ├── datasource.schema.json
│   │   ├── pipeline.schema.json
│   │   ├── product.schema.json
│   │   └── provenance.schema.json
│   ├── definitions/
│   │   ├── event_classes/
│   │   ├── datasources/
│   │   └── pipelines/
│   └── validator.py
├── agents/                      # Agent implementations
│   ├── orchestrator/
│   ├── discovery/
│   ├── pipeline/
│   ├── quality/
│   └── reporting/
├── core/                        # Core platform services
│   ├── intent/
│   ├── data/
│   ├── execution/
│   └── provenance/
├── api/                         # FastAPI application
├── tests/
└── deploy/
```

---

## Common Schema Definitions

Shared type definitions referenced by all schemas (`common.schema.json`):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/common.schema.json",

  "$defs": {
    "geometry": {
      "type": "object",
      "description": "GeoJSON geometry object",
      "required": ["type", "coordinates"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"]
        },
        "coordinates": {
          "type": "array"
        }
      }
    },

    "bbox": {
      "type": "array",
      "description": "Bounding box [west, south, east, north]",
      "items": {"type": "number"},
      "minItems": 4,
      "maxItems": 4
    },

    "temporal_extent": {
      "type": "object",
      "description": "Time window with optional reference point",
      "required": ["start", "end"],
      "properties": {
        "start": {"type": "string", "format": "date-time"},
        "end": {"type": "string", "format": "date-time"},
        "reference_time": {
          "type": "string",
          "format": "date-time",
          "description": "Event peak or reference timestamp"
        }
      }
    },

    "confidence_score": {
      "type": "number",
      "description": "Confidence as ratio 0-1 (NOT percentage)",
      "minimum": 0,
      "maximum": 1
    },

    "crs": {
      "type": "string",
      "description": "Coordinate reference system as EPSG code",
      "pattern": "^EPSG:[0-9]+$",
      "default": "EPSG:4326"
    },

    "uri": {
      "type": "string",
      "format": "uri",
      "description": "Resource identifier (URL, S3 path, etc.)"
    },

    "checksum": {
      "type": "object",
      "description": "Data integrity checksum",
      "required": ["algorithm", "value"],
      "properties": {
        "algorithm": {"type": "string", "enum": ["md5", "sha256"]},
        "value": {"type": "string"}
      }
    },

    "data_type_category": {
      "type": "string",
      "description": "Semantic data category",
      "enum": ["optical", "sar", "dem", "weather", "ancillary"]
    },

    "data_format": {
      "type": "string",
      "description": "File/data format",
      "enum": ["geotiff", "cog", "netcdf", "zarr", "jp2", "hdf5", "grib", "geojson", "geoparquet"]
    },

    "quality_flag": {
      "type": "string",
      "description": "Standardized quality/status flags",
      "enum": [
        "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "INSUFFICIENT_CONFIDENCE",
        "RESOLUTION_DEGRADED", "SINGLE_SENSOR_MODE", "TEMPORALLY_INTERPOLATED",
        "HISTORICAL_PROXY", "MISSING_OBSERVABLE", "FORECAST_DISCREPANCY",
        "SPATIAL_UNCERTAINTY", "MAGNITUDE_CONFLICT", "CONSERVATIVE_ESTIMATE"
      ]
    },

    "band_mapping": {
      "type": "object",
      "description": "Maps generic band names to sensor-specific bands",
      "properties": {
        "blue": {"type": "array", "items": {"type": "string"}},
        "green": {"type": "array", "items": {"type": "string"}},
        "red": {"type": "array", "items": {"type": "string"}},
        "nir": {"type": "array", "items": {"type": "string"}},
        "swir1": {"type": "array", "items": {"type": "string"}},
        "swir2": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

---

## Intent Schema

Handles event typing requirements with predefined classes, NLP inference, user override capability, and machine-interpretable output.

```yaml
# Example intent resolution flow
input:
  natural_language: "flooding in coastal areas after hurricane"
  explicit_class: null

resolution:
  inferred_class: "flood.coastal.storm_surge"
  confidence: 0.87
  alternatives:
    - class: "flood.riverine"
      confidence: 0.12

output:
  resolved_class: "flood.coastal.storm_surge"
  source: "inferred"
  parameters:
    flood_type: "storm_surge"
    causation: "tropical_cyclone"
```

**JSON Schema:** `openspec/schemas/intent.schema.json`

---

## Event Schema

Core specification combining area, time, and resolved intent:

```yaml
event:
  id: "evt_2024_atlantic_flood_001"

  intent:
    class: "flood.coastal.storm_surge"
    source: "inferred"
    original_input: "flooding after hurricane in Miami area"

  spatial:
    type: "Polygon"
    coordinates: [...]
    crs: "EPSG:4326"

  temporal:
    start: "2024-09-15T00:00:00Z"
    end: "2024-09-20T23:59:59Z"
    reference_time: "2024-09-17T12:00:00Z"

  constraints:
    max_cloud_cover: 0.3
    min_resolution_m: 10
    required_bands: ["nir", "swir"]
```

**JSON Schema:** `openspec/schemas/event.schema.json`

---

## Event Class Registry

Hierarchical taxonomy:

```
hazard/
├── flood/
│   ├── riverine
│   ├── coastal/
│   │   ├── storm_surge
│   │   └── tidal
│   ├── flash
│   └── urban
├── wildfire/
│   ├── forest
│   ├── grassland
│   └── interface
├── storm/
│   ├── tropical_cyclone
│   ├── severe_convective
│   └── winter
└── other/
    ├── earthquake
    ├── landslide
    └── volcanic
```

Each class defines: required data types, applicable pipelines, output product templates, and validation thresholds.

---

## Data Source Schema

Extensible provider framework (`datasource.schema.json`):

```yaml
datasource:
  id: "sentinel2_l2a"
  provider: "copernicus"
  type: "optical"

  capabilities:
    bands: ["B02", "B03", "B04", "B08", "B11", "B12"]
    resolution_m: 10
    revisit_days: 5

  access:
    protocol: "stac"
    endpoint: "https://earth-search.aws.element84.com/v1"
    authentication: "none"

  applicability:
    event_classes: ["flood.*", "wildfire.*"]
    constraints:
      requires_clear_sky: true
```

---

## Pipeline Schema

Composable analytical workflows (`pipeline.schema.json`):

```yaml
pipeline:
  id: "flood_extent_sar"
  name: "SAR-based Flood Extent Mapping"
  applicable_classes: ["flood.*"]

  inputs:
    - name: "pre_event_sar"
      type: "raster"
      source: "sentinel1_grd"
    - name: "post_event_sar"
      type: "raster"
      source: "sentinel1_grd"

  steps:
    - id: "calibrate"
      processor: "sar.radiometric_calibration"
      inputs: ["pre_event_sar", "post_event_sar"]

    - id: "change_detect"
      processor: "flood.change_detection"
      inputs: ["calibrate.pre", "calibrate.post"]
      parameters:
        threshold_db: -3.0

    - id: "validate"
      processor: "qa.plausibility_check"
      inputs: ["change_detect.extent"]

  outputs:
    - name: "flood_extent"
      type: "vector"
      format: "geojson"
```

---

## Provenance Schema

Full lineage tracking (`provenance.schema.json`):

```yaml
provenance:
  product_id: "prod_flood_001"

  lineage:
    - step: "data_acquisition"
      timestamp: "2024-09-18T10:00:00Z"
      inputs: []
      outputs: ["scene_S1A_20240917"]
      agent: "discovery_agent"

    - step: "flood_detection"
      timestamp: "2024-09-18T10:15:00Z"
      inputs: ["scene_S1A_20240917", "scene_S1A_20240910"]
      outputs: ["flood_extent_v1"]
      agent: "pipeline_agent"
      processor: "flood.change_detection"
      parameters: {threshold_db: -3.0}

  quality:
    uncertainty_percent: 12.5
    plausibility_score: 0.91
    validation_method: "cross_sensor"
```

---

## Data Broker Architecture

Multi-source data broker for discovering, evaluating, and acquiring data across heterogeneous sources.

### Directory Structure

```
core/data/
├── broker.py
├── discovery/
│   ├── base.py
│   ├── stac.py
│   ├── wms_wcs.py
│   └── provider_api.py
├── providers/
│   ├── registry.py
│   ├── optical/
│   ├── sar/
│   ├── dem/
│   ├── weather/
│   └── ancillary/
├── evaluation/
│   ├── constraints.py
│   ├── ranking.py
│   └── tradeoffs.py
├── cache/
│   ├── manager.py
│   ├── index.py
│   └── storage.py
└── acquisition/
    ├── fetcher.py
    └── preprocessor.py
```

### Data Type Categories

| Category | Sources | Use Cases |
|----------|---------|-----------|
| **Optical** | Sentinel-2, Landsat, MODIS | Flood extent, burn scars, vegetation damage |
| **SAR** | Sentinel-1, ALOS-2 | All-weather flood mapping, change detection |
| **DEM** | Copernicus DEM, SRTM, FABDEM | Flow modeling, terrain analysis |
| **Weather** | ERA5, GFS, ECMWF | Event context, forecast integration |
| **Ancillary** | OSM, WorldCover, WSF | Population exposure, land use context |

### Provider Preference System

```yaml
provider_registry:
  preference_order:
    - tier: "open"
      providers: ["copernicus", "usgs", "nasa", "noaa"]
      cost: 0
      preference_weight: 1.0

    - tier: "open_restricted"
      providers: ["jaxa", "esa_tpm"]
      cost: 0
      preference_weight: 0.8

    - tier: "commercial"
      providers: ["planet", "maxar", "iceye", "capella"]
      cost: "variable"
      preference_weight: 0.5

  fallback_policy:
    attempt_open_first: true
    escalate_to_commercial:
      condition: "no_viable_open_source"
      requires_approval: true
```

---

## Intelligent Sensor Selection

Atmospheric-aware selection and multi-sensor fusion.

### Atmospheric Selection

```yaml
atmospheric_selection:
  conditions:
    clear:
      cloud_cover_max: 0.2
      prefer: ["optical", "sar"]
    partly_cloudy:
      cloud_cover_range: [0.2, 0.5]
      prefer: ["sar", "optical_partial"]
    cloudy:
      cloud_cover_min: 0.5
      prefer: ["sar"]
      exclude: ["optical"]
    severe_weather:
      indicators: ["heavy_precipitation", "high_winds"]
      prefer: ["sar_post_event"]
```

### Degraded Mode Handling

```yaml
degraded_modes:
  modes:
    full_capability:
      level: 0
      confidence_modifier: 1.0

    reduced_resolution:
      level: 1
      fallback: "use_lower_resolution_alternative"
      confidence_modifier: 0.9
      flag: "RESOLUTION_DEGRADED"

    single_sensor:
      level: 2
      fallback: "proceed_with_available_sensor"
      confidence_modifier: 0.8
      flag: "SINGLE_SENSOR_MODE"

    temporal_interpolation:
      level: 2
      fallback: "interpolate_from_adjacent_acquisitions"
      confidence_modifier: 0.7
      flag: "TEMPORALLY_INTERPOLATED"

    historical_proxy:
      level: 3
      fallback: "use_historical_baseline_with_model"
      confidence_modifier: 0.5
      flag: "HISTORICAL_PROXY"

    insufficient_data:
      level: 4
      action: "abort_with_explanation"
      flag: "INSUFFICIENT_DATA"
```

---

## Data Ingestion & Normalization

Transforms raw heterogeneous inputs into analysis-ready, cloud-native formats.

### Directory Structure

```
core/data/ingestion/
├── pipeline.py
├── formats/
│   ├── cog.py
│   ├── zarr.py
│   ├── parquet.py
│   └── stac_item.py
├── normalization/
│   ├── projection.py
│   ├── tiling.py
│   ├── temporal.py
│   └── resolution.py
├── enrichment/
│   ├── overviews.py
│   ├── statistics.py
│   └── quality.py
├── validation/
│   ├── integrity.py
│   ├── anomaly.py
│   └── completeness.py
└── persistence/
    ├── storage.py
    ├── intermediate.py
    └── lineage.py
```

### Format Targets

```yaml
format_targets:
  raster:
    format: "cog"
    compression: "deflate"
    predictor: 2
    blocksize: 512
    overviews: [2, 4, 8, 16, 32]

  multidimensional:
    format: "zarr"
    chunks: {time: 1, y: 512, x: 512}
    compression: "zstd"

  vector:
    format: "geoparquet"
    compression: "snappy"
```

---

## Analysis & Modeling Layer

### Algorithm Library

```
core/analysis/
├── library/
│   ├── registry.py
│   ├── baseline/
│   │   ├── flood/
│   │   │   ├── threshold_sar.py
│   │   │   ├── ndwi_optical.py
│   │   │   ├── change_detection.py
│   │   │   └── hand_model.py
│   │   ├── wildfire/
│   │   │   ├── nbr_differenced.py
│   │   │   ├── thermal_anomaly.py
│   │   │   └── ba_classifier.py
│   │   └── storm/
│   │       ├── wind_damage.py
│   │       └── structural_damage.py
│   ├── advanced/
│   │   ├── flood/
│   │   │   ├── unet_segmentation.py
│   │   │   ├── ensemble_fusion.py
│   │   │   └── physics_informed.py
│   │   └── ...
│   └── experimental/
├── selection/
│   ├── selector.py
│   ├── constraints.py
│   ├── deterministic.py
│   └── hybrid.py
├── assembly/
│   ├── assembler.py
│   ├── graph.py
│   ├── validator.py
│   └── optimizer.py
└── execution/
    ├── runner.py
    ├── distributed.py
    └── checkpoint.py
```

### Algorithm Selection Engine

Rule-based and ML-hybrid selection with deterministic mode for reproducibility.

---

## Multi-Sensor Fusion Engine

Handles spatial/temporal alignment, terrain corrections, and conflict resolution with uncertainty propagation.

---

## Forecast & Scenario Integration

Integrates weather forecasts (GFS, ECMWF, ERA5), hazard-specific models (NWM, HWRF), and scenario-based analysis (baseline, forecast-based, worst-case, best-case, historical analog).

---

## Quality Control & Validation

### QC Architecture

```
core/quality/
├── sanity/
│   ├── spatial.py
│   ├── values.py
│   ├── temporal.py
│   └── artifacts.py
├── validation/
│   ├── cross_model.py
│   ├── cross_sensor.py
│   ├── historical.py
│   └── consensus.py
├── uncertainty/
│   ├── quantification.py
│   ├── spatial_uncertainty.py
│   └── propagation.py
├── actions/
│   ├── gating.py
│   ├── flagging.py
│   └── routing.py
└── reporting/
    ├── qa_report.py
    └── diagnostics.py
```

### Sanity Checks

- Spatial coherence (connectivity, boundary smoothness, topographic consistency)
- Value plausibility (physical bounds, statistical outliers, rate of change)
- Historical baseline comparison
- Artifact detection (striping, tile seams, temporal jumps)

### Gating Actions

- **PASS**: Proceed to output
- **DOWNGRADE**: Proceed with caveats (confidence_modifier: 0.7)
- **REVIEW**: Route to expert review
- **BLOCK**: Block output, require human override

---

## Agent Architecture

### Agent Responsibilities

| Agent | Primary Responsibility | Inputs | Outputs |
|-------|----------------------|--------|---------|
| **Orchestrator** | Workflow coordination | Event spec | Final products |
| **Discovery** | Data acquisition | Broker query | Broker response |
| **Pipeline** | Analysis execution | Pipeline spec + data | Analysis results |
| **Quality** | Validation & QC | Results + config | QA report |
| **Reporting** | Product generation | Validated results | Products + reports |

### Agent Communication Protocol

JSON Schema for inter-agent messaging with correlation IDs, priority levels, and execution context.

### Agent State Management

- Execution state persistence (PostgreSQL/DynamoDB)
- Checkpoint-based recovery
- Retry policies with exponential/linear backoff
- Partial failure handling

---

## Implementation Phases

All phases below are **COMPLETE**:

### Phase 1: Foundation & Core Schemas
- Project setup, JSON Schema files, validator, event class taxonomy

### Phase 2: Intent Resolution System
- Event class registry, NLP classifier, resolution logic

### Phase 3: Data Broker & Ingestion
- Multi-source broker, STAC client, constraint evaluation, sensor selection, cache system, ingestion pipeline

### Phase 4: Analysis & Pipeline Engine
- Algorithm library, selection engine, pipeline assembler, fusion engine, forecast integration, distributed execution

### Phase 5: Quality Control & Validation
- Sanity checks, cross-validation, consensus generation, uncertainty quantification, gating

### Phase 6: Agent Orchestration
- Base agent class, orchestrator, specialized agents, message passing, state persistence

### Phase 7: API & Deployment
- FastAPI application, routes, serverless deployment

---

## Verification

```bash
# Schema validation
pytest tests/test_schemas.py -v

# Intent resolution
pytest tests/test_intent.py -v

# Data broker
pytest tests/test_broker.py -v

# Pipeline execution
pytest tests/test_pipeline.py -v

# Full test suite
./run_tests.py
```

---

**Archive Note:** This document preserves the complete design specifications for the implemented Multiverse Dive platform. For active development work, see `OPENSPEC.md`.
