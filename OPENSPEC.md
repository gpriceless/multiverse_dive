# OpenSpec: Geospatial Event Intelligence Platform

## Overview

Design a cloud-native, agent-orchestrated platform that transforms (area, time window, event type) into reproducible decision products. Core principle: situation-agnostic specifications enable the same agents/pipelines to handle floods, wildfires, storms, and other hazards.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Spec Format | JSON Schema + YAML | Validation + human readability |
| Stack | Python + FastAPI | Geospatial ecosystem (GDAL, rasterio, xarray) |
| Data Sources | Extensible framework | Start open, design for any provider |
| Agent Model | Hierarchical orchestrator | Clear delegation, easier debugging |
| Deployment | Serverless-first | Burst scaling for time-critical response |

## Project Structure

```
multiverse_dive/
├── openspec/                    # Core specification schemas
│   ├── schemas/
│   │   ├── event.schema.json    # Event specification schema
│   │   ├── intent.schema.json   # Intent/event-type schema
│   │   ├── datasource.schema.json
│   │   ├── pipeline.schema.json
│   │   ├── product.schema.json
│   │   └── provenance.schema.json
│   ├── definitions/             # YAML spec instances
│   │   ├── event_classes/       # Predefined event types
│   │   ├── datasources/         # Data source configs
│   │   └── pipelines/           # Pipeline templates
│   └── validator.py             # Schema validation utilities
├── agents/                      # Agent implementations
│   ├── orchestrator/            # Main orchestrator agent
│   ├── discovery/               # Data discovery agent
│   ├── pipeline/                # Pipeline assembly agent
│   ├── quality/                 # Plausibility/uncertainty agent
│   └── reporting/               # Product generation agent
├── core/                        # Core platform services
│   ├── intent/                  # Intent resolution system
│   │   ├── classifier.py        # NLP event type inference
│   │   ├── resolver.py          # Intent resolution logic
│   │   └── registry.py          # Predefined event class registry
│   ├── data/                    # Data access layer
│   ├── execution/               # Pipeline execution engine
│   └── provenance/              # Lineage tracking
├── api/                         # FastAPI application
│   ├── main.py
│   ├── routes/
│   └── models/
├── tests/
└── deploy/                      # Serverless deployment configs
```

## OpenSpec Schema Design

### Common Definitions (`common.schema.json`)

Shared type definitions referenced by all schemas:

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
      },
      "examples": [
        {"nir": ["B08", "B8A"], "swir1": ["B11"], "swir2": ["B12"]}
      ]
    }
  }
}
```

### 1. Intent Schema (`intent.schema.json`)

Handles the event typing requirements:
- Predefined event classes with hierarchical taxonomy
- NLP inference from natural language
- User override capability
- Machine-interpretable structured output

```yaml
# Example intent resolution flow
input:
  natural_language: "flooding in coastal areas after hurricane"
  explicit_class: null  # Optional override

resolution:
  inferred_class: "flood.coastal.storm_surge"
  confidence: 0.87
  alternatives:
    - class: "flood.riverine"
      confidence: 0.12

output:
  resolved_class: "flood.coastal.storm_surge"
  source: "inferred"  # or "explicit" if overridden
  parameters:
    flood_type: "storm_surge"
    causation: "tropical_cyclone"
```

**JSON Schema Definition:**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/intent.schema.json",

  "type": "object",
  "required": ["input", "output"],
  "properties": {
    "input": {
      "type": "object",
      "description": "Intent resolution input",
      "properties": {
        "natural_language": {
          "type": "string",
          "description": "Free-text event description for NLP inference"
        },
        "explicit_class": {
          "type": ["string", "null"],
          "description": "User-specified event class override",
          "pattern": "^[a-z]+\\.[a-z_]+(\\.[a-z_]+)*$"
        }
      },
      "anyOf": [
        {"required": ["natural_language"]},
        {"required": ["explicit_class"]}
      ]
    },

    "resolution": {
      "type": "object",
      "description": "NLP inference results (populated by classifier)",
      "properties": {
        "inferred_class": {
          "type": "string",
          "pattern": "^[a-z]+\\.[a-z_]+(\\.[a-z_]+)*$"
        },
        "confidence": {"$ref": "common.schema.json#/$defs/confidence_score"},
        "alternatives": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["class", "confidence"],
            "properties": {
              "class": {"type": "string"},
              "confidence": {"$ref": "common.schema.json#/$defs/confidence_score"}
            }
          }
        }
      }
    },

    "output": {
      "type": "object",
      "description": "Final resolved intent",
      "required": ["resolved_class", "source"],
      "properties": {
        "resolved_class": {
          "type": "string",
          "description": "Final event class (explicit override or inferred)",
          "pattern": "^[a-z]+\\.[a-z_]+(\\.[a-z_]+)*$"
        },
        "source": {
          "type": "string",
          "enum": ["explicit", "inferred"],
          "description": "Whether class was user-specified or NLP-inferred"
        },
        "confidence": {"$ref": "common.schema.json#/$defs/confidence_score"},
        "parameters": {
          "type": "object",
          "description": "Event-class-specific parameters extracted from input",
          "additionalProperties": {"type": "string"}
        }
      }
    }
  }
}
```

### 2. Event Schema (`event.schema.json`)

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
    reference_time: "2024-09-17T12:00:00Z"  # Event peak

  constraints:
    max_cloud_cover: 0.3
    min_resolution_m: 10
    required_bands: ["nir", "swir"]
```

**JSON Schema Definition:**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/event.schema.json",

  "type": "object",
  "required": ["id", "intent", "spatial", "temporal"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique event identifier",
      "pattern": "^evt_[a-z0-9_]+$"
    },

    "intent": {
      "type": "object",
      "description": "Resolved event intent (from intent.schema.json output)",
      "required": ["class", "source"],
      "properties": {
        "class": {
          "type": "string",
          "description": "Resolved event class",
          "pattern": "^[a-z]+\\.[a-z_]+(\\.[a-z_]+)*$"
        },
        "source": {
          "type": "string",
          "enum": ["explicit", "inferred"]
        },
        "original_input": {
          "type": "string",
          "description": "Original natural language input if inferred"
        },
        "confidence": {"$ref": "common.schema.json#/$defs/confidence_score"}
      }
    },

    "spatial": {
      "type": "object",
      "description": "Area of interest",
      "required": ["type", "coordinates"],
      "properties": {
        "type": {"type": "string", "enum": ["Polygon", "MultiPolygon"]},
        "coordinates": {"type": "array"},
        "crs": {"$ref": "common.schema.json#/$defs/crs"},
        "bbox": {"$ref": "common.schema.json#/$defs/bbox"}
      }
    },

    "temporal": {"$ref": "common.schema.json#/$defs/temporal_extent"},

    "constraints": {
      "type": "object",
      "description": "Data acquisition constraints",
      "properties": {
        "max_cloud_cover": {"$ref": "common.schema.json#/$defs/confidence_score"},
        "min_resolution_m": {"type": "number", "minimum": 0},
        "max_resolution_m": {"type": "number", "minimum": 0},
        "required_bands": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Generic band names (nir, swir, etc.)"
        },
        "required_data_types": {
          "type": "array",
          "items": {"$ref": "common.schema.json#/$defs/data_type_category"}
        },
        "polarization": {
          "type": "array",
          "items": {"type": "string", "enum": ["VV", "VH", "HH", "HV"]}
        }
      }
    },

    "priority": {
      "type": "string",
      "enum": ["critical", "high", "normal", "low"],
      "default": "normal",
      "description": "Processing priority for time-critical events"
    },

    "metadata": {
      "type": "object",
      "description": "Additional event metadata",
      "properties": {
        "created_at": {"type": "string", "format": "date-time"},
        "created_by": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

### 3. Event Class Registry

Predefined hierarchical taxonomy:

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
│   └── interface  # WUI fires
├── storm/
│   ├── tropical_cyclone
│   ├── severe_convective
│   └── winter
└── other/
    ├── earthquake
    ├── landslide
    └── volcanic
```

Each class defines:
- Required data types (optical, SAR, weather, etc.)
- Applicable analysis pipelines
- Output product templates
- Validation thresholds

### 4. Data Source Schema (`datasource.schema.json`)

Extensible provider framework:

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
    authentication: "none"  # or "api_key", "oauth2"

  applicability:
    event_classes: ["flood.*", "wildfire.*"]
    constraints:
      requires_clear_sky: true
```

### 5. Pipeline Schema (`pipeline.schema.json`)

Composable analytical workflows:

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

**JSON Schema Definition:**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/pipeline.schema.json",

  "type": "object",
  "required": ["id", "name", "applicable_classes", "inputs", "steps", "outputs"],
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9_]*$"
    },
    "name": {"type": "string"},
    "description": {"type": "string"},
    "version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},

    "applicable_classes": {
      "type": "array",
      "description": "Event classes this pipeline applies to (supports wildcards)",
      "items": {"type": "string"}
    },

    "inputs": {
      "type": "array",
      "description": "Required input data",
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": {"type": "string", "description": "Logical input name for reference in steps"},
          "type": {"type": "string", "enum": ["raster", "vector", "table", "scalar"]},
          "source": {"type": "string", "description": "Data source ID (e.g., sentinel1_grd)"},
          "temporal_role": {
            "type": "string",
            "enum": ["pre_event", "post_event", "reference", "any"],
            "description": "Temporal relationship to event"
          },
          "required": {"type": "boolean", "default": true}
        }
      }
    },

    "steps": {
      "type": "array",
      "description": "Processing steps (executed as DAG)",
      "items": {
        "type": "object",
        "required": ["id", "processor"],
        "properties": {
          "id": {"type": "string", "description": "Unique step identifier"},
          "processor": {
            "type": "string",
            "description": "Algorithm ID from registry (e.g., flood.change_detection)"
          },
          "inputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Input references: 'input_name' or 'step_id.output_name'"
          },
          "parameters": {
            "type": "object",
            "description": "Algorithm parameters (override defaults)",
            "additionalProperties": true
          },
          "outputs": {
            "type": "object",
            "description": "Named outputs from this step",
            "additionalProperties": {"type": "string"}
          },
          "qc_gate": {
            "type": "object",
            "description": "QC check after this step",
            "properties": {
              "enabled": {"type": "boolean", "default": false},
              "checks": {"type": "array", "items": {"type": "string"}},
              "on_fail": {"type": "string", "enum": ["continue", "warn", "abort"]}
            }
          }
        }
      }
    },

    "outputs": {
      "type": "array",
      "description": "Final pipeline outputs",
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string", "enum": ["raster", "vector", "table", "report"]},
          "format": {"$ref": "common.schema.json#/$defs/data_format"},
          "source_step": {"type": "string", "description": "Step ID that produces this output"}
        }
      }
    },

    "metadata": {
      "type": "object",
      "properties": {
        "author": {"type": "string"},
        "created": {"type": "string", "format": "date-time"},
        "validated_regions": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

**Step Input Reference Resolution:**

Pipeline steps reference inputs using a path syntax:
- `"input_name"` - References a pipeline-level input
- `"step_id.output_name"` - References output from a previous step
- `"$event.field"` - References event specification field
- `"$broker.dataset_id"` - References broker-selected dataset

### 6. Provenance Schema (`provenance.schema.json`)

Full lineage tracking:

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

## Data Broker Architecture

### Multi-Source Data Broker

Central component for discovering, evaluating, and acquiring data across heterogeneous sources.

```
core/data/
├── broker.py              # Main data broker orchestration
├── discovery/
│   ├── base.py            # Abstract discovery interface
│   ├── stac.py            # STAC catalog discovery
│   ├── wms_wcs.py         # OGC services discovery
│   └── provider_api.py    # Custom provider APIs
├── providers/
│   ├── registry.py        # Provider registry with preferences
│   ├── optical/
│   │   ├── sentinel2.py   # Copernicus Sentinel-2
│   │   ├── landsat.py     # USGS Landsat
│   │   └── modis.py       # NASA MODIS/VIIRS
│   ├── sar/
│   │   ├── sentinel1.py   # Copernicus Sentinel-1
│   │   └── alos.py        # JAXA ALOS-2
│   ├── dem/
│   │   ├── copernicus.py  # Copernicus DEM
│   │   ├── srtm.py        # NASA SRTM
│   │   └── fabdem.py      # Forest-adjusted DEM
│   ├── weather/
│   │   ├── era5.py        # ECMWF ERA5 reanalysis
│   │   ├── gfs.py         # NOAA GFS forecasts
│   │   └── ecmwf.py       # ECMWF operational
│   └── ancillary/
│       ├── osm.py         # OpenStreetMap
│       ├── wsf.py         # World Settlement Footprint
│       └── landcover.py   # ESA WorldCover, etc.
├── evaluation/
│   ├── constraints.py     # Constraint evaluation engine
│   ├── ranking.py         # Multi-criteria ranking
│   └── tradeoffs.py       # Trade-off documentation
├── cache/
│   ├── manager.py         # Cache lifecycle management
│   ├── index.py           # Spatial-temporal index
│   └── storage.py         # Storage backends (S3, local)
└── acquisition/
    ├── fetcher.py         # Data download/streaming
    └── preprocessor.py    # Standardization layer
```

### Data Type Categories

| Category | Sources | Use Cases |
|----------|---------|-----------|
| **Optical** | Sentinel-2, Landsat, MODIS | Flood extent, burn scars, vegetation damage |
| **SAR** | Sentinel-1, ALOS-2 | All-weather flood mapping, change detection |
| **DEM** | Copernicus DEM, SRTM, FABDEM | Flow modeling, terrain analysis |
| **Weather** | ERA5, GFS, ECMWF | Event context, forecast integration |
| **Ancillary** | OSM, WorldCover, WSF | Population exposure, land use context |

### Discovery Flow

```yaml
discovery_request:
  event_id: "evt_001"
  spatial: {bbox: [-80.5, 25.5, -80.0, 26.0]}
  temporal: {start: "2024-09-15", end: "2024-09-20"}
  intent_class: "flood.coastal.storm_surge"

  requirements:
    optical:
      max_cloud_cover: 0.3
      min_resolution_m: 10
      required: false  # Nice to have
    sar:
      polarization: ["VV", "VH"]
      required: true   # Essential for flood
    dem:
      min_resolution_m: 30
      required: true
    weather:
      variables: ["precipitation", "wind_speed"]
      required: true
```

### Constraint Evaluation Engine

Evaluates each candidate dataset against:

1. **Hard constraints** (must satisfy):
   - Spatial coverage (intersects AOI)
   - Temporal coverage (within window)
   - Data availability (accessible)

2. **Soft constraints** (scored):
   - Cloud cover percentage
   - Spatial resolution
   - Temporal proximity to event
   - Data quality flags

```python
# Example constraint evaluation
class ConstraintEvaluator:
    def evaluate(self, candidate: Dataset, requirements: Requirements) -> Evaluation:
        hard_pass = self._check_hard_constraints(candidate, requirements)
        if not hard_pass.satisfied:
            return Evaluation(viable=False, reason=hard_pass.failure_reason)

        soft_scores = self._score_soft_constraints(candidate, requirements)
        return Evaluation(
            viable=True,
            score=soft_scores.weighted_total,
            breakdown=soft_scores.by_criterion
        )
```

### Multi-Criteria Ranking

Ranks viable candidates using weighted scoring:

```yaml
ranking_criteria:
  resolution_score:
    weight: 0.25
    prefer: "higher"  # Lower meter value = higher score

  temporal_proximity:
    weight: 0.30
    prefer: "closer"  # Nearer to event peak

  cloud_cover:
    weight: 0.20
    prefer: "lower"

  provider_preference:
    weight: 0.15
    order: ["open", "commercial"]  # Prefer open data

  acquisition_cost:
    weight: 0.10
    prefer: "lower"
```

### Trade-off Documentation

Every selection decision is documented:

```yaml
selection_record:
  dataset_id: "S1A_IW_GRDH_20240917"
  selected: true
  rank: 1

  scores:
    resolution: 0.95
    temporal_proximity: 0.88
    cloud_cover: 1.0  # N/A for SAR
    provider_preference: 1.0  # Open data
    total_weighted: 0.93

  trade_offs:
    - "Selected over S1B scene (rank 2) due to 6hr closer temporal proximity"
    - "Commercial ICEYE available with 3m resolution but open Sentinel-1 sufficient for flood extent at 10m"

  alternatives_considered:
    - id: "S1B_IW_GRDH_20240917"
      rank: 2
      total_score: 0.87
      reason_not_selected: "6hr later acquisition"
```

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
      note: "Requires registration or terms acceptance"

    - tier: "commercial"
      providers: ["planet", "maxar", "iceye", "capella"]
      cost: "variable"
      preference_weight: 0.5
      note: "Use only when open sources insufficient"

  fallback_policy:
    attempt_open_first: true
    escalate_to_commercial:
      condition: "no_viable_open_source"
      requires_approval: true
```

### Intelligent Sensor Selection

Automatically selects optimal sensor combinations based on conditions and requirements.

```
core/data/selection/
├── strategy.py            # Selection strategy orchestration
├── atmospheric.py         # Atmosphere condition assessment
├── fusion.py              # Multi-sensor blending logic
├── degraded.py            # Degraded mode handling
└── confidence.py          # Observable confidence tracking
```

#### Atmospheric-Aware Selection

```yaml
atmospheric_selection:
  conditions:
    clear:
      cloud_cover_max: 0.2
      prefer: ["optical", "sar"]
      rationale: "Optical preferred for spectral richness"

    partly_cloudy:
      cloud_cover_range: [0.2, 0.5]
      prefer: ["sar", "optical_partial"]
      rationale: "SAR reliable, optical usable in clear patches"

    cloudy:
      cloud_cover_min: 0.5
      prefer: ["sar"]
      exclude: ["optical"]
      rationale: "Optical unusable, SAR unaffected by clouds"

    severe_weather:
      indicators: ["heavy_precipitation", "high_winds"]
      prefer: ["sar_post_event"]
      timing: "wait_for_clearance"
      rationale: "Even SAR quality degrades in extreme precipitation"

  assessment_sources:
    - weather_forecast: "gfs"
    - cloud_mask: "sentinel2_scl"
    - nowcast: "mrms"  # Multi-Radar Multi-Sensor
```

#### Multi-Sensor Fusion

```yaml
sensor_fusion:
  strategies:
    complementary:
      description: "Combine sensors with complementary strengths"
      examples:
        - optical_sar_flood:
            optical: "water_spectral_signature"
            sar: "surface_roughness_change"
            fusion: "weighted_ensemble"
            weights: {optical: 0.4, sar: 0.6}
            condition: "partial_cloud"

        - multi_resolution:
            coarse: "modis_daily"
            fine: "sentinel2_sparse"
            fusion: "spatiotemporal_interpolation"
            use_case: "gap_filling"

    redundant:
      description: "Use multiple sensors for validation"
      examples:
        - cross_validation:
            primary: "sentinel1"
            secondary: "sentinel2"
            method: "independent_detection"
            agreement_threshold: 0.8

    temporal_densification:
      description: "Combine revisit rates"
      sources: ["sentinel1a", "sentinel1b", "sentinel2a", "sentinel2b"]
      effective_revisit: "1-3 days"

  fusion_algorithms:
    - weighted_average:
        weights_from: ["resolution", "temporal_proximity", "quality_score"]
    - bayesian_combination:
        priors_from: "historical_accuracy"
    - machine_learning:
        model: "random_forest_fusion"
        features: ["sensor_type", "atmospheric_conditions", "event_type"]
```

#### Degraded Mode Handling

```yaml
degraded_modes:
  triggers:
    - no_optimal_sensor_available
    - partial_spatial_coverage
    - temporal_gap_exceeds_threshold
    - quality_below_minimum

  modes:
    full_capability:
      level: 0
      description: "All required sensors available at desired quality"
      confidence_modifier: 1.0

    reduced_resolution:
      level: 1
      trigger: "high_res_unavailable"
      fallback: "use_lower_resolution_alternative"
      example: "Landsat-30m instead of Sentinel2-10m"
      confidence_modifier: 0.9
      flag: "RESOLUTION_DEGRADED"

    single_sensor:
      level: 2
      trigger: "fusion_sources_unavailable"
      fallback: "proceed_with_available_sensor"
      confidence_modifier: 0.8
      flag: "SINGLE_SENSOR_MODE"

    temporal_interpolation:
      level: 2
      trigger: "no_direct_observation_in_window"
      fallback: "interpolate_from_adjacent_acquisitions"
      max_gap_days: 5
      confidence_modifier: 0.7
      flag: "TEMPORALLY_INTERPOLATED"

    historical_proxy:
      level: 3
      trigger: "no_observation_possible"
      fallback: "use_historical_baseline_with_model"
      confidence_modifier: 0.5
      flag: "HISTORICAL_PROXY"

    insufficient_data:
      level: 4
      trigger: "minimum_requirements_not_met"
      action: "abort_with_explanation"
      flag: "INSUFFICIENT_DATA"

  escalation:
    notify_on_level: 2
    require_approval_on_level: 3
    abort_on_level: 4
```

#### Observable Confidence Tracking

```yaml
confidence_tracking:
  per_observable:
    flood_extent:
      confidence_factors:
        - sensor_suitability: 0.3
        - atmospheric_conditions: 0.2
        - temporal_proximity: 0.25
        - spatial_coverage: 0.15
        - algorithm_accuracy: 0.1

      confidence_levels:
        high: {min: 0.8, flag: null}
        medium: {min: 0.6, flag: "MEDIUM_CONFIDENCE"}
        low: {min: 0.4, flag: "LOW_CONFIDENCE"}
        insufficient: {max: 0.4, flag: "INSUFFICIENT_CONFIDENCE"}

  missing_observable_handling:
    strategy: "explicit_null"
    metadata:
      reason: "string"           # Why observable is missing
      attempted_sources: "list"  # What was tried
      fallback_attempted: "bool" # Was degraded mode tried
      next_availability: "datetime"  # When data might be available

  output_format:
    observables:
      flood_extent:
        value: "geometry"
        confidence: 0.85
        flags: []
        sources: ["S1A_20240917", "S1B_20240918"]

      flood_depth:
        value: null
        confidence: 0.0
        flags: ["MISSING_OBSERVABLE", "NO_DEM_AVAILABLE"]
        reason: "DEM resolution insufficient for depth estimation"
        sources: []

      affected_population:
        value: 15420
        confidence: 0.72
        flags: ["MEDIUM_CONFIDENCE", "SINGLE_SENSOR_MODE"]
        sources: ["WSF_2019", "flood_extent_derived"]
```

#### Selection Decision Schema

```json
{
  "sensor_selection_decision": {
    "type": "object",
    "properties": {
      "event_id": {"type": "string"},
      "atmospheric_assessment": {
        "type": "object",
        "properties": {
          "condition": {"enum": ["clear", "partly_cloudy", "cloudy", "severe_weather"]},
          "cloud_cover": {"type": "number"},
          "source": {"type": "string"}
        }
      },
      "selected_sensors": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "sensor": {"type": "string"},
            "role": {"enum": ["primary", "secondary", "validation"]},
            "rationale": {"type": "string"}
          }
        }
      },
      "fusion_strategy": {
        "type": "string",
        "enum": ["none", "complementary", "redundant", "temporal_densification"]
      },
      "degraded_mode": {
        "type": "object",
        "properties": {
          "active": {"type": "boolean"},
          "level": {"type": "integer"},
          "flags": {"type": "array", "items": {"type": "string"}},
          "confidence_modifier": {"type": "number"}
        }
      }
    }
  }
}
```

### Cache System

```yaml
cache_config:
  storage:
    backend: "s3"  # or "local", "gcs"
    bucket: "geoint-cache"
    retention_days: 90

  index:
    type: "spatiotemporal"
    spatial_index: "rtree"
    temporal_resolution: "1h"

  lookup_strategy:
    check_cache_first: true
    cache_hit_criteria:
      spatial_overlap_min: 0.95
      temporal_tolerance_hours: 24
      same_processing_level: true

  reuse_policy:
    dem: "always"           # DEMs rarely change
    ancillary: "weekly"     # Refresh weekly
    weather: "never"        # Always fetch current
    optical: "by_scene_id"  # Exact scene match
    sar: "by_scene_id"
```

## Data Ingestion & Normalization

### Ingestion Pipeline

Transforms raw heterogeneous inputs into analysis-ready, cloud-native formats.

```
core/data/ingestion/
├── pipeline.py            # Main ingestion orchestration
├── formats/
│   ├── cog.py             # Cloud-Optimized GeoTIFF conversion
│   ├── zarr.py            # Zarr array conversion (multidimensional)
│   ├── parquet.py         # GeoParquet for vectors
│   └── stac_item.py       # STAC metadata generation
├── normalization/
│   ├── projection.py      # CRS normalization
│   ├── tiling.py          # Tile scheme management
│   ├── temporal.py        # Temporal alignment/indexing
│   └── resolution.py      # Spatial resampling
├── enrichment/
│   ├── overviews.py       # Pyramid/overview generation
│   ├── statistics.py      # Band statistics calculation
│   └── quality.py         # Quality summary generation
├── validation/
│   ├── integrity.py       # Corruption detection
│   ├── anomaly.py         # Anomalous value detection
│   └── completeness.py    # Coverage validation
└── persistence/
    ├── storage.py         # Storage backend abstraction
    ├── intermediate.py    # Intermediate product management
    └── lineage.py         # Lineage tracking integration
```

### Cloud-Native Format Conversion

```yaml
format_targets:
  raster:
    format: "cog"  # Cloud-Optimized GeoTIFF
    compression: "deflate"
    predictor: 2
    blocksize: 512
    overviews: [2, 4, 8, 16, 32]
    overview_resampling: "average"

  multidimensional:
    format: "zarr"
    chunks:
      time: 1
      y: 512
      x: 512
    compression: "zstd"

  vector:
    format: "geoparquet"
    compression: "snappy"
    row_group_size: 100000

  metadata:
    format: "stac"
    version: "1.0.0"
    extensions: ["proj", "eo", "sar", "processing"]
```

### Projection Normalization

```yaml
projection_config:
  target_crs:
    default: "EPSG:4326"        # WGS84 for storage
    analysis: "utm_auto"        # Auto-select UTM zone for analysis
    global: "EPSG:3857"         # Web Mercator for visualization

  utm_auto_selection:
    method: "centroid"          # Based on AOI centroid
    hemisphere_handling: "split" # Handle cross-hemisphere AOIs

  resampling:
    continuous: "bilinear"      # For continuous data (reflectance, elevation)
    categorical: "nearest"      # For classified data (land cover)
    quality_flags: "nearest"    # For bit-packed QA bands
```

### Tiling Scheme

```yaml
tiling_config:
  scheme: "xyz"                 # XYZ tile pyramid
  tile_size: 512
  max_zoom: 18

  partitioning:
    spatial:
      method: "quadkey"         # Efficient spatial partitioning
      level: 12                 # ~10km tiles at equator
    temporal:
      method: "year_month"      # YYYY/MM directory structure

  naming:
    pattern: "{source}/{date}/{quadkey}/{layer}.{ext}"
    example: "sentinel2/2024/09/0231032/B04.tif"
```

### Temporal Indexing

```yaml
temporal_index:
  resolution: "1h"              # Minimum temporal granularity
  timezone: "UTC"

  alignment:
    snap_to: "hour"             # Round acquisition times
    tolerance_minutes: 30

  dimensions:
    acquisition_time: "datetime64[ns]"
    valid_time: "datetime64[ns]"     # For forecasts
    reference_time: "datetime64[ns]" # Event reference

  indexing:
    backend: "duckdb"           # Fast analytical queries
    partitions: ["year", "month", "source"]
```

### Overview & Statistics Generation

```yaml
enrichment:
  overviews:
    factors: [2, 4, 8, 16, 32, 64]
    resampling: "average"
    format: "internal"          # Embedded in COG

  statistics:
    compute:
      - min
      - max
      - mean
      - stddev
      - percentiles: [2, 25, 50, 75, 98]
      - histogram:
          bins: 256
          range: "data"         # or explicit [min, max]
    per_band: true
    store_in: "stac_metadata"

  quality_summary:
    metrics:
      - valid_pixel_percent
      - nodata_percent
      - cloud_cover_percent     # If applicable
      - shadow_percent
    spatial_distribution: true  # Per-tile breakdown
```

### Input Validation & Anomaly Detection

```yaml
validation:
  integrity_checks:
    - checksum_verification     # MD5/SHA256 against source
    - file_structure            # Valid GeoTIFF/NetCDF structure
    - georeferencing            # Valid CRS and transform
    - band_count                # Expected number of bands
    - dtype_match               # Expected data types

  anomaly_detection:
    statistical:
      - z_score_threshold: 5    # Flag extreme outliers
      - iqr_multiplier: 3       # Interquartile range check
    physical:
      - reflectance_range: [0, 1]        # Valid reflectance bounds
      - temperature_range: [200, 350]    # Kelvin bounds
      - elevation_range: [-500, 9000]    # Meters
    spatial:
      - striping_detection      # Sensor artifacts
      - tile_boundary_artifacts # Processing seams

  completeness:
    - spatial_coverage_min: 0.8  # Minimum AOI coverage
    - temporal_gaps_max: 3       # Maximum missing timesteps
    - required_bands_present: true

  on_failure:
    corrupted: "reject_with_log"
    anomalous: "flag_and_continue"
    incomplete: "warn_and_continue"
```

### Intermediate Product Persistence

```yaml
intermediate_products:
  storage:
    backend: "s3"
    bucket: "geoint-intermediate"
    retention:
      raw_normalized: "30d"
      analysis_ready: "90d"
      derived_products: "365d"

  catalog:
    type: "stac"
    api: "stac-fastapi"
    database: "postgresql"

  lineage_tracking:
    enabled: true
    record:
      - source_datasets
      - processing_chain
      - parameters_used
      - software_versions
      - timestamps
      - checksums

  product_types:
    - id: "normalized_optical"
      description: "Surface reflectance, COG format, UTM projection"
      retention: "90d"

    - id: "normalized_sar"
      description: "Calibrated backscatter, COG format, UTM projection"
      retention: "90d"

    - id: "analysis_stack"
      description: "Multi-temporal stack ready for analysis"
      retention: "30d"
```

### Ingestion Schema (`ingestion.schema.json`)

```json
{
  "ingestion_job": {
    "type": "object",
    "required": ["source", "target_format", "normalization"],
    "properties": {
      "source": {
        "type": "object",
        "properties": {
          "uri": {"type": "string", "format": "uri"},
          "format": {"enum": ["geotiff", "netcdf", "jp2", "hdf5", "grib"]},
          "checksum": {"type": "string"}
        }
      },
      "target_format": {
        "enum": ["cog", "zarr", "geoparquet"]
      },
      "normalization": {
        "type": "object",
        "properties": {
          "target_crs": {"type": "string"},
          "resolution_m": {"type": "number"},
          "resampling": {"enum": ["nearest", "bilinear", "cubic"]}
        }
      },
      "validation": {
        "type": "object",
        "properties": {
          "integrity_checks": {"type": "boolean", "default": true},
          "anomaly_detection": {"type": "boolean", "default": true},
          "fail_on_anomaly": {"type": "boolean", "default": false}
        }
      },
      "enrichment": {
        "type": "object",
        "properties": {
          "generate_overviews": {"type": "boolean", "default": true},
          "compute_statistics": {"type": "boolean", "default": true},
          "quality_summary": {"type": "boolean", "default": true}
        }
      },
      "lineage": {
        "type": "object",
        "properties": {
          "parent_products": {"type": "array", "items": {"type": "string"}},
          "processing_step": {"type": "string"},
          "parameters": {"type": "object"}
        }
      }
    }
  }
}
```

### Data Broker Schema (`datasource.schema.json` extension)

**Broker Query** (input to discovery):

```json
{
  "broker_query": {
    "type": "object",
    "required": ["event_id", "spatial", "temporal", "intent_class"],
    "properties": {
      "event_id": {"type": "string"},
      "spatial": {"$ref": "common.schema.json#/$defs/geometry"},
      "temporal": {"$ref": "common.schema.json#/$defs/temporal_extent"},
      "intent_class": {"type": "string"},
      "data_types": {
        "type": "array",
        "items": {"$ref": "common.schema.json#/$defs/data_type_category"}
      },
      "constraints": {
        "type": "object",
        "additionalProperties": {
          "$ref": "#/$defs/data_type_constraints"
        }
      },
      "ranking_weights": {"$ref": "#/$defs/ranking_criteria"},
      "cache_policy": {
        "type": "string",
        "enum": ["prefer_cache", "prefer_fresh", "cache_only", "no_cache"],
        "default": "prefer_cache"
      }
    }
  }
}
```

**Broker Response** (output from discovery → input to ingestion):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/broker_response.schema.json",

  "type": "object",
  "required": ["event_id", "query_timestamp", "selected_datasets", "selection_summary"],
  "properties": {
    "event_id": {"type": "string"},
    "query_timestamp": {"type": "string", "format": "date-time"},

    "selected_datasets": {
      "type": "array",
      "description": "Datasets selected for acquisition, ordered by priority",
      "items": {
        "type": "object",
        "required": ["dataset_id", "data_type", "source_uri", "provider"],
        "properties": {
          "dataset_id": {"type": "string"},
          "data_type": {"$ref": "common.schema.json#/$defs/data_type_category"},
          "provider": {"type": "string"},
          "source_uri": {"$ref": "common.schema.json#/$defs/uri"},
          "source_format": {"$ref": "common.schema.json#/$defs/data_format"},
          "checksum": {"$ref": "common.schema.json#/$defs/checksum"},

          "acquisition_time": {"type": "string", "format": "date-time"},
          "spatial_coverage_percent": {"type": "number", "minimum": 0, "maximum": 100},
          "cloud_cover_percent": {"type": "number", "minimum": 0, "maximum": 100},
          "resolution_m": {"type": "number"},

          "role": {
            "type": "string",
            "enum": ["primary", "secondary", "pre_event", "post_event", "reference"],
            "description": "Role in analysis pipeline"
          },

          "selection_score": {"$ref": "common.schema.json#/$defs/confidence_score"},
          "selection_rationale": {"type": "string"}
        }
      }
    },

    "selection_summary": {
      "type": "object",
      "properties": {
        "total_candidates_evaluated": {"type": "integer"},
        "datasets_selected": {"type": "integer"},
        "data_types_covered": {
          "type": "array",
          "items": {"$ref": "common.schema.json#/$defs/data_type_category"}
        },
        "data_types_missing": {
          "type": "array",
          "items": {"$ref": "common.schema.json#/$defs/data_type_category"}
        },
        "cache_hits": {"type": "integer"},
        "estimated_acquisition_size_gb": {"type": "number"}
      }
    },

    "trade_off_record": {
      "type": "array",
      "description": "Documented trade-offs for provenance",
      "items": {
        "type": "object",
        "properties": {
          "decision": {"type": "string"},
          "selected": {"type": "string"},
          "alternatives": {"type": "array", "items": {"type": "string"}},
          "rationale": {"type": "string"}
        }
      }
    },

    "degraded_mode": {
      "type": "object",
      "properties": {
        "active": {"type": "boolean"},
        "level": {"type": "integer", "minimum": 0, "maximum": 4},
        "flags": {
          "type": "array",
          "items": {"$ref": "common.schema.json#/$defs/quality_flag"}
        },
        "confidence_modifier": {"$ref": "common.schema.json#/$defs/confidence_score"}
      }
    }
  }
}
```

### Constraint Mapper

Maps event constraints to data-type-specific discovery requirements. This bridges the gap between the event specification and the data broker query.

```
core/intent/
├── constraint_mapper.py     # Event constraints → broker query requirements
```

**Mapping Logic:**

```yaml
constraint_mapping:
  # Event class determines required data types
  class_to_data_types:
    "flood.*":
      required: ["sar", "dem"]
      optional: ["optical", "weather"]
    "wildfire.*":
      required: ["optical"]
      optional: ["sar", "weather", "dem"]
    "storm.*":
      required: ["weather", "optical"]
      optional: ["sar"]

  # Event constraints map to data-type-specific constraints
  constraint_translation:
    max_cloud_cover:
      applies_to: ["optical"]
      maps_to: "cloud_cover_max"

    min_resolution_m:
      applies_to: ["optical", "sar", "dem"]
      maps_to: "resolution_max_m"

    required_bands:
      applies_to: ["optical"]
      maps_to: "bands"
      transform: "resolve_band_aliases"  # nir → B08, etc.

    polarization:
      applies_to: ["sar"]
      maps_to: "polarization"

  # Output: broker_query.constraints structured by data type
  example_output:
    optical:
      cloud_cover_max: 0.3
      resolution_max_m: 10
      bands: ["B08", "B11", "B12"]
      required: false
    sar:
      polarization: ["VV", "VH"]
      resolution_max_m: 10
      required: true
    dem:
      resolution_max_m: 30
      required: true
```

## Analysis & Modeling Layer

### Algorithm Library Architecture

```
core/analysis/
├── library/
│   ├── registry.py            # Algorithm registry and metadata
│   ├── baseline/              # Established, well-validated algorithms
│   │   ├── flood/
│   │   │   ├── threshold_sar.py      # Simple SAR backscatter threshold
│   │   │   ├── ndwi_optical.py       # NDWI water detection
│   │   │   ├── change_detection.py   # Pre/post change detection
│   │   │   └── hand_model.py         # Height Above Nearest Drainage
│   │   ├── wildfire/
│   │   │   ├── nbr_differenced.py    # dNBR burn severity
│   │   │   ├── thermal_anomaly.py    # MODIS/VIIRS hotspots
│   │   │   └── ba_classifier.py      # Burned area classification
│   │   └── storm/
│   │       ├── wind_damage.py        # Vegetation damage detection
│   │       └── structural_damage.py  # Building damage assessment
│   ├── advanced/              # ML-driven and experimental algorithms
│   │   ├── flood/
│   │   │   ├── unet_segmentation.py  # Deep learning flood mapping
│   │   │   ├── ensemble_fusion.py    # Multi-model ensemble
│   │   │   └── physics_informed.py   # Physics-constrained ML
│   │   ├── wildfire/
│   │   │   ├── spread_prediction.py  # Fire spread modeling
│   │   │   └── severity_regression.py
│   │   └── common/
│   │       ├── anomaly_detection.py  # Unsupervised anomaly detection
│   │       └── change_forest.py      # Random forest change detection
│   └── experimental/          # Under validation
│       └── README.md
├── selection/
│   ├── selector.py            # Algorithm selection engine
│   ├── constraints.py         # Constraint evaluation
│   ├── deterministic.py       # Reproducible selection logic
│   └── hybrid.py              # Rule + ML hybrid selector
├── assembly/
│   ├── assembler.py           # Pipeline assembly engine
│   ├── graph.py               # DAG representation
│   ├── validator.py           # Pipeline validation
│   └── optimizer.py           # Execution optimization
└── execution/
    ├── runner.py              # Pipeline execution engine
    ├── distributed.py         # Distributed execution (Dask/Ray)
    └── checkpoint.py          # State checkpointing
```

### Algorithm Registry

```yaml
algorithm_registry:
  flood.baseline.threshold_sar:
    name: "SAR Backscatter Threshold"
    category: "baseline"
    event_types: ["flood.*"]

    description: |
      Simple thresholding of SAR backscatter coefficient to detect
      standing water. Reliable, interpretable, well-validated.

    requirements:
      data:
        - type: "sar"
          polarization: ["VV", "VH"]
          temporal: "post_event"
      optional:
        - type: "sar"
          temporal: "pre_event"
          benefit: "enables_change_detection"
      compute:
        memory_gb: 4
        gpu: false

    parameters:
      threshold_db:
        type: "float"
        default: -15.0
        range: [-20.0, -10.0]
        description: "Backscatter threshold for water classification"

      min_area_ha:
        type: "float"
        default: 0.5
        description: "Minimum flood polygon area"

    outputs:
      - name: "flood_extent"
        type: "vector"
        geometry: "polygon"
      - name: "confidence_raster"
        type: "raster"
        dtype: "float32"

    validation:
      accuracy_range: [0.75, 0.90]
      validated_regions: ["north_america", "europe", "southeast_asia"]
      citations: ["doi:10.1016/j.rse.2019.111489"]

    reproducibility:
      deterministic: true
      seed_required: false
      version: "1.2.0"

  flood.advanced.unet_segmentation:
    name: "U-Net Flood Segmentation"
    category: "advanced"
    event_types: ["flood.*"]

    description: |
      Deep learning semantic segmentation using U-Net architecture.
      Higher accuracy but requires more compute and less interpretable.

    requirements:
      data:
        - type: "sar"
          polarization: ["VV", "VH"]
        - type: "dem"
          optional: true
          benefit: "improved_accuracy"
      compute:
        memory_gb: 16
        gpu: true
        gpu_memory_gb: 8

    parameters:
      model_checkpoint:
        type: "string"
        default: "flood_unet_v3.pt"

      confidence_threshold:
        type: "float"
        default: 0.5
        range: [0.3, 0.9]

    validation:
      accuracy_range: [0.85, 0.95]
      validated_regions: ["global"]
      citations: ["doi:10.1109/TGRS.2021.3084632"]

    reproducibility:
      deterministic: true
      seed_required: true
      version: "3.1.0"
```

### Algorithm Selection Engine

```yaml
selection_engine:
  # Rule-based selection criteria
  rules:
    - name: "data_availability"
      priority: 1
      logic: |
        Filter algorithms where ALL required data types are available.
        Score higher if optional data also available.

    - name: "compute_constraints"
      priority: 2
      logic: |
        Filter algorithms that exceed available compute resources.
        Consider GPU availability, memory limits.

    - name: "event_type_match"
      priority: 3
      logic: |
        Select algorithms designed for the event type.
        Support wildcards (flood.* matches flood.coastal).

    - name: "accuracy_preference"
      priority: 4
      logic: |
        When multiple algorithms qualify, prefer higher accuracy.
        Weight by regional validation if available.

    - name: "reproducibility_requirement"
      priority: 5
      condition: "reproducibility_required == true"
      logic: |
        Only select deterministic algorithms with version pinning.

  # Hybrid ML-assisted selection
  ml_selection:
    enabled: true
    model: "algorithm_selector_rf.pkl"

    features:
      - event_type
      - data_availability_vector
      - aoi_size_km2
      - cloud_cover_percent
      - time_criticality
      - historical_accuracy_for_region

    output: "algorithm_ranking"

    override_rules:
      - "ml_suggestion_must_pass_rule_filters"
      - "human_override_always_respected"

  # Deterministic selection for reproducibility
  deterministic_mode:
    enabled_by: "reproducibility_required"

    strategy: |
      1. Apply all rule filters deterministically
      2. Sort qualified algorithms by (accuracy_desc, name_asc)
      3. Select first algorithm
      4. Pin exact version in provenance

    hash_inputs:
      - event_type
      - available_data_types
      - algorithm_versions
      - parameter_values

    output: "selection_hash for reproducibility verification"
```

### Dynamic Pipeline Assembly

```yaml
pipeline_assembly:
  # Pipeline as a Directed Acyclic Graph (DAG)
  dag_structure:
    nodes:
      - type: "data_input"
        outputs: ["raw_data"]

      - type: "preprocessing"
        inputs: ["raw_data"]
        outputs: ["normalized_data"]
        algorithms: ["calibration", "coregistration", "reprojection"]

      - type: "analysis"
        inputs: ["normalized_data"]
        outputs: ["raw_results"]
        algorithms: ["primary_detection"]

      - type: "postprocessing"
        inputs: ["raw_results"]
        outputs: ["refined_results"]
        algorithms: ["morphological_cleanup", "area_filtering"]

      - type: "validation"
        inputs: ["refined_results", "ancillary_data"]
        outputs: ["validated_results"]
        algorithms: ["plausibility_check", "cross_validation"]

      - type: "output"
        inputs: ["validated_results"]
        outputs: ["final_product"]

  # Assembly rules
  assembly_rules:
    - rule: "preprocessing_required"
      condition: "input_data.format != 'analysis_ready'"
      insert: "preprocessing_step"

    - rule: "multi_sensor_fusion"
      condition: "len(input_sensors) > 1"
      insert: "fusion_step"
      position: "before:analysis"

    - rule: "uncertainty_propagation"
      condition: "uncertainty_required == true"
      insert: "uncertainty_step"
      position: "parallel:analysis"

    - rule: "validation_required"
      condition: "always"
      insert: "validation_step"

  # Example assembled pipeline
  example_flood_pipeline:
    id: "flood_sar_standard_v1"
    assembled_for:
      event_type: "flood.coastal"
      available_data: ["sentinel1_grd", "copernicus_dem"]

    steps:
      - id: "ingest"
        algorithm: "ingestion.sar_grd"
        inputs: {scene: "$input.sentinel1"}
        outputs: {calibrated: "sar_calibrated"}

      - id: "terrain_correct"
        algorithm: "preprocessing.terrain_correction"
        inputs: {sar: "sar_calibrated", dem: "$input.dem"}
        outputs: {corrected: "sar_tc"}

      - id: "detect"
        algorithm: "flood.baseline.threshold_sar"
        inputs: {sar: "sar_tc"}
        parameters: {threshold_db: -15.0}
        outputs: {extent: "flood_raw"}

      - id: "cleanup"
        algorithm: "postprocessing.morphological"
        inputs: {polygons: "flood_raw"}
        parameters: {min_area_ha: 0.5, smoothing: true}
        outputs: {extent: "flood_clean"}

      - id: "validate"
        algorithm: "validation.plausibility"
        inputs: {extent: "flood_clean", dem: "$input.dem"}
        outputs: {extent: "flood_validated", qa: "qa_report"}

      - id: "output"
        algorithm: "output.geojson"
        inputs: {extent: "flood_validated"}
        outputs: {product: "$output.flood_extent"}
```

### Hybrid Rule-Based and ML Approaches

```yaml
hybrid_approaches:
  # Strategy selection
  strategy_matrix:
    high_confidence_scenario:
      conditions:
        - well_validated_event_type
        - sufficient_training_data
        - non_critical_application
      approach: "ml_primary"
      fallback: "rule_based"

    critical_scenario:
      conditions:
        - emergency_response
        - legal_implications
        - limited_validation_data
      approach: "rule_based_primary"
      augment_with: "ml_confidence"

    novel_scenario:
      conditions:
        - new_event_type
        - unusual_conditions
        - no_historical_precedent
      approach: "ensemble"
      components: ["rule_based", "ml_anomaly_detection"]

  # Hybrid execution patterns
  patterns:
    ml_with_rule_guardrails:
      description: |
        ML model provides primary prediction, rule-based checks
        validate physical plausibility and flag anomalies.

      execution:
        - step: "ml_prediction"
          output: "raw_prediction"
        - step: "rule_validation"
          input: "raw_prediction"
          checks:
            - "flood_below_dem_elevation"
            - "extent_within_watershed"
            - "area_physically_plausible"
          output: "validated_prediction"

    rule_based_with_ml_refinement:
      description: |
        Rule-based algorithm provides initial detection,
        ML model refines boundaries and estimates uncertainty.

      execution:
        - step: "rule_detection"
          output: "initial_extent"
        - step: "ml_refinement"
          input: "initial_extent"
          tasks:
            - "boundary_refinement"
            - "uncertainty_estimation"
          output: "refined_extent"

    ensemble_voting:
      description: |
        Multiple approaches vote, disagreement triggers
        additional validation or human review.

      execution:
        - step: "parallel_execution"
          algorithms:
            - "rule_based_threshold"
            - "ml_segmentation"
            - "change_detection"
          outputs: ["result_1", "result_2", "result_3"]
        - step: "ensemble_fusion"
          method: "majority_voting"
          agreement_threshold: 0.7
          on_disagreement: "flag_for_review"
          output: "ensemble_result"
```

### Multi-Sensor Fusion Engine

```yaml
fusion_engine:
  # Spatial and temporal alignment
  alignment:
    spatial:
      reference_grid:
        method: "highest_resolution_input"  # or explicit grid
        fallback_resolution_m: 10

      coregistration:
        method: "phase_correlation"  # For SAR-SAR
        optical_method: "feature_matching"
        tolerance_pixels: 0.5

      resampling:
        upscale: "bilinear"
        downscale: "average"
        categorical: "nearest"

    temporal:
      reference_time:
        method: "event_peak"  # or "latest_acquisition"

      interpolation:
        method: "linear"
        max_gap_hours: 48
        flag_interpolated: true

      compositing:
        method: "median"  # For multi-date stacks
        outlier_rejection: "iqr"

  # Terrain and ancillary corrections
  corrections:
    terrain:
      - name: "sar_terrain_correction"
        applies_to: "sar"
        method: "range_doppler"
        dem_source: "copernicus_dem"

      - name: "topographic_normalization"
        applies_to: "optical"
        method: "c_correction"
        dem_source: "copernicus_dem"

      - name: "slope_aspect_masking"
        applies_to: ["sar", "optical"]
        threshold_degrees: 45
        action: "mask_and_flag"

    atmospheric:
      - name: "atmospheric_correction"
        applies_to: "optical"
        method: "sen2cor"  # or "6s", "dos"

    ancillary:
      - name: "water_mask"
        source: "permanent_water_layer"
        action: "distinguish_permanent_vs_flood"

      - name: "urban_mask"
        source: "wsf"
        action: "adjust_thresholds"

  # Conflict resolution
  conflict_resolution:
    strategy: "weighted_reasoning"

    weights:
      by_sensor_quality:
        sentinel2_clear: 1.0
        sentinel2_hazy: 0.6
        sentinel1: 0.9
        landsat: 0.85
        modis: 0.5

      by_temporal_proximity:
        same_day: 1.0
        1_day: 0.9
        3_days: 0.7
        7_days: 0.4

      by_spatial_resolution:
        10m: 1.0
        30m: 0.8
        250m: 0.4

    conflict_types:
      binary_disagreement:
        condition: "sensor_a.detected != sensor_b.detected"
        resolution: "weighted_vote"
        threshold: 0.6
        on_tie: "conservative"  # Assume no detection

      magnitude_disagreement:
        condition: "abs(sensor_a.value - sensor_b.value) > threshold"
        resolution: "weighted_average"
        flag: "MAGNITUDE_CONFLICT"

      spatial_disagreement:
        condition: "intersection(extent_a, extent_b) / union(...) < 0.8"
        resolution: "union_with_confidence_gradient"
        flag: "SPATIAL_UNCERTAINTY"

    output:
      primary_result: "fused_best_estimate"
      confidence_layer: "pixel_wise_confidence"
      conflict_layer: "disagreement_locations"
      contribution_layer: "per_sensor_weight_applied"

  # Uncertainty propagation
  uncertainty_propagation:
    method: "monte_carlo"  # or "analytical", "ensemble"

    input_uncertainties:
      radiometric: "sensor_specifications"
      geometric: "coregistration_rmse"
      algorithmic: "validation_accuracy"

    propagation_rules:
      addition: "sqrt(sum_of_squares)"
      multiplication: "relative_error_sum"
      thresholding: "sigmoid_uncertainty_near_threshold"

    output:
      uncertainty_raster: true
      confidence_intervals: [0.68, 0.95]
      sensitivity_analysis: true

  # Diagnostic outputs
  diagnostic_layers:
    primary:
      - "fused_result"
      - "confidence"
      - "uncertainty_bounds"

    diagnostic:
      - "per_sensor_contribution"
      - "conflict_locations"
      - "interpolation_mask"
      - "correction_applied"
      - "quality_flags"

    metadata:
      - "fusion_parameters"
      - "input_sources"
      - "processing_chain"
```

### Forecast & Scenario Integration

```yaml
forecast_integration:
  # Weather and hazard forecast ingestion
  forecast_sources:
    weather:
      - id: "gfs"
        provider: "noaa"
        variables: ["precipitation", "wind_speed", "temperature"]
        horizons: ["6h", "12h", "24h", "48h", "72h"]
        resolution_km: 25

      - id: "ecmwf_hres"
        provider: "ecmwf"
        variables: ["precipitation", "wind_speed", "temperature", "cape"]
        horizons: ["6h", "12h", "24h", "48h", "72h", "120h", "240h"]
        resolution_km: 9

      - id: "era5"
        provider: "ecmwf"
        type: "reanalysis"
        use_for: "historical_context"

    hazard_specific:
      - id: "nwm"
        provider: "noaa"
        type: "streamflow_forecast"
        horizons: ["short_range", "medium_range"]

      - id: "hwrf"
        provider: "noaa"
        type: "hurricane_track"
        horizons: ["5_day"]

      - id: "wrf_fire"
        provider: "custom"
        type: "fire_spread"
        horizons: ["6h", "12h", "24h"]

  # Forecast-observation comparison
  forecast_validation:
    compare_metrics:
      - name: "precipitation_total"
        observed: "era5_analysis"
        forecast: "gfs_forecast"
        metrics: ["mae", "rmse", "bias", "correlation"]

      - name: "flood_extent"
        observed: "satellite_derived"
        forecast: "nwm_inundation"
        metrics: ["iou", "precision", "recall", "f1"]

    discrepancy_handling:
      threshold_significant: 0.3  # 30% difference
      actions:
        - flag: "FORECAST_DISCREPANCY"
        - annotate: "observed_vs_forecast_delta"
        - trigger: "scenario_reanalysis"

  # Scenario-based analysis
  scenario_analysis:
    types:
      baseline:
        description: "Current observed conditions"
        source: "satellite_observation"

      forecast_based:
        description: "Projected conditions from forecast"
        source: "weather_hazard_forecast"
        horizons: ["24h", "48h", "72h"]

      worst_case:
        description: "Upper bound of forecast ensemble"
        method: "ensemble_percentile_95"

      best_case:
        description: "Lower bound of forecast ensemble"
        method: "ensemble_percentile_5"

      historical_analog:
        description: "Similar historical event for comparison"
        method: "analog_matching"
        features: ["event_type", "magnitude", "location"]

    execution:
      run_parallel: true
      compare_outputs: true
      generate_delta_maps: true

    output:
      scenario_comparison:
        - "baseline_vs_24h_forecast"
        - "best_case_vs_worst_case"
        - "observed_vs_historical_analog"

  # Impact projections
  impact_projection:
    horizons:
      short_term:
        range: "0-24h"
        confidence: "high"
        use_cases: ["immediate_response", "evacuation"]

      medium_term:
        range: "24-72h"
        confidence: "medium"
        use_cases: ["resource_positioning", "damage_estimation"]

      extended:
        range: "72h-7d"
        confidence: "low"
        use_cases: ["planning", "trend_analysis"]

    impact_types:
      flood:
        - affected_area_km2
        - affected_population
        - affected_infrastructure
        - estimated_damage_usd

      wildfire:
        - burned_area_km2
        - structures_threatened
        - containment_projection

      storm:
        - affected_area_km2
        - power_outage_customers
        - structural_damage_count

    projection_method:
      deterministic: "best_estimate_from_forecast"
      probabilistic: "ensemble_distribution"

      uncertainty_sources:
        - forecast_uncertainty
        - model_uncertainty
        - exposure_data_uncertainty

  # Forecast confidence annotation
  confidence_annotation:
    levels:
      high:
        condition: "horizon < 24h AND ensemble_spread < 0.2"
        label: "HIGH_CONFIDENCE"
        action: "use_for_decisions"

      medium:
        condition: "horizon < 72h AND ensemble_spread < 0.4"
        label: "MEDIUM_CONFIDENCE"
        action: "use_with_caution"

      low:
        condition: "horizon >= 72h OR ensemble_spread >= 0.4"
        label: "LOW_CONFIDENCE"
        action: "advisory_only"

    limitations_recorded:
      - "forecast_model_biases"
      - "resolution_limitations"
      - "known_failure_modes"
      - "data_latency"

    output_format:
      projections:
        flood_extent_24h:
          value: "geometry"
          confidence: "HIGH"
          uncertainty_range: [950, 1050]  # km2
          limitations:
            - "Based on GFS precipitation forecast"
            - "DEM resolution limits accuracy in urban areas"
            - "Does not account for emergency drainage measures"
          forecast_sources:
            - {id: "gfs_20240917_12z", horizon: "24h"}
            - {id: "nwm_short_range", horizon: "18h"}
```

## Quality Control & Plausibility Validation

### QC Architecture

```
core/quality/
├── sanity/
│   ├── spatial.py             # Spatial coherence checks
│   ├── values.py              # Value plausibility checks
│   ├── temporal.py            # Temporal consistency checks
│   └── artifacts.py           # Artifact detection
├── validation/
│   ├── cross_model.py         # Cross-model comparison
│   ├── cross_sensor.py        # Cross-sensor validation
│   ├── historical.py          # Historical baseline comparison
│   └── consensus.py           # Consensus generation
├── uncertainty/
│   ├── quantification.py      # Uncertainty metrics
│   ├── spatial_uncertainty.py # Spatial uncertainty mapping
│   └── propagation.py         # Uncertainty propagation
├── actions/
│   ├── gating.py              # Pass/fail/downgrade logic
│   ├── flagging.py            # Quality flag management
│   └── routing.py             # Expert review routing
└── reporting/
    ├── qa_report.py           # QA report generation
    └── diagnostics.py         # Diagnostic outputs
```

### Automated Sanity Checks

```yaml
sanity_checks:
  # Spatial coherence and continuity
  spatial_coherence:
    checks:
      - name: "connectivity"
        description: "Flood extents should be spatially connected or explainable"
        method: "connected_components_analysis"
        parameters:
          max_isolated_components: 10
          min_component_area_ha: 1.0
        on_fail: "flag_fragmented"

      - name: "boundary_smoothness"
        description: "Boundaries should not have unnatural jaggedness"
        method: "fractal_dimension"
        parameters:
          max_fractal_dimension: 1.4
          window_size_m: 100
        on_fail: "flag_artifacts"

      - name: "topographic_consistency"
        description: "Flood should respect topographic constraints"
        method: "dem_consistency_check"
        parameters:
          max_uphill_flood_m: 2.0  # Allow for DEM error
          watershed_containment: true
        on_fail: "block_critical"

      - name: "edge_continuity"
        description: "No artificial breaks at tile/scene boundaries"
        method: "boundary_discontinuity_detection"
        parameters:
          max_discontinuity_percent: 5
        on_fail: "flag_processing_artifact"

  # Impossible or implausible values
  value_plausibility:
    checks:
      - name: "physical_bounds"
        description: "Values within physically possible range"
        rules:
          flood_depth_m: {min: 0, max: 50}
          burn_severity_dnbr: {min: -500, max: 1300}
          wind_damage_percent: {min: 0, max: 100}
          water_fraction: {min: 0, max: 1}
        on_fail: "clip_and_flag"

      - name: "statistical_outliers"
        description: "Detect statistically improbable values"
        method: "modified_z_score"
        parameters:
          threshold: 3.5
          spatial_window: true
        on_fail: "flag_outlier"

      - name: "rate_of_change"
        description: "Changes should be physically plausible"
        rules:
          flood_expansion_km2_per_hour: {max: 100}
          fire_spread_km_per_hour: {max: 10}
        on_fail: "flag_implausible_dynamics"

      - name: "area_reasonableness"
        description: "Total affected area within reasonable bounds"
        method: "historical_percentile_check"
        parameters:
          max_percentile: 99.9
          context: ["event_type", "region", "season"]
        on_fail: "flag_extreme"

  # Historical baseline comparison
  historical_comparison:
    checks:
      - name: "baseline_deviation"
        description: "Compare against normal conditions"
        baselines:
          permanent_water: "jrc_global_surface_water"
          land_cover: "esa_worldcover"
          historical_floods: "gfd_archive"
        parameters:
          max_deviation_from_normal: 3.0  # std devs
        on_fail: "flag_anomalous"

      - name: "seasonal_consistency"
        description: "Results consistent with seasonal patterns"
        method: "seasonal_envelope_check"
        parameters:
          envelope_source: "historical_climatology"
          allow_exceedance_if: "extreme_event_confirmed"
        on_fail: "flag_seasonal_anomaly"

      - name: "similar_event_comparison"
        description: "Compare with analogous historical events"
        method: "analog_matching"
        parameters:
          similarity_features: ["magnitude", "location", "meteorology"]
          min_analogs: 3
        output: "analog_comparison_report"

  # Discontinuity and artifact detection
  artifact_detection:
    checks:
      - name: "striping"
        description: "Detect sensor striping artifacts"
        method: "directional_frequency_analysis"
        parameters:
          orientations: [0, 90]  # Along/across track
          threshold_power: 2.0
        on_fail: "flag_sensor_artifact"

      - name: "tile_seams"
        description: "Detect processing tile boundaries"
        method: "gradient_discontinuity"
        parameters:
          kernel_size: 5
          threshold_gradient: 0.3
        on_fail: "flag_processing_artifact"

      - name: "temporal_jumps"
        description: "Detect sudden unexplained changes"
        method: "change_point_detection"
        parameters:
          method: "pelt"
          min_segment_length: 2
        on_fail: "flag_temporal_discontinuity"

      - name: "classification_artifacts"
        description: "Detect salt-and-pepper noise in classifications"
        method: "local_homogeneity"
        parameters:
          window_size: 3
          min_homogeneity: 0.6
        on_fail: "flag_classification_noise"

  # Gating actions
  gating:
    levels:
      pass:
        condition: "all_critical_pass AND warning_count < 3"
        action: "proceed_to_output"
        confidence_modifier: 1.0

      downgrade:
        condition: "all_critical_pass AND warning_count >= 3"
        action: "proceed_with_caveats"
        confidence_modifier: 0.7
        required_annotations:
          - "quality_degraded"
          - "specific_warnings"

      review:
        condition: "critical_soft_fail"
        action: "route_to_expert_review"
        confidence_modifier: 0.5
        timeout_hours: 24
        fallback: "conservative_output"

      block:
        condition: "critical_hard_fail"
        action: "block_output"
        confidence_modifier: 0.0
        required: "human_override_to_release"
```

### Cross-Model and Cross-Sensor Validation

```yaml
cross_validation:
  # Cross-model comparison
  cross_model:
    description: "Compare results from independent analytical methods"

    comparison_pairs:
      flood_detection:
        - method_a: "threshold_sar"
          method_b: "ml_segmentation"
          comparison: "spatial_overlap"

        - method_a: "optical_ndwi"
          method_b: "sar_change_detection"
          comparison: "spatial_overlap"

      burn_severity:
        - method_a: "dnbr_threshold"
          method_b: "random_forest_severity"
          comparison: "categorical_agreement"

    metrics:
      spatial_overlap:
        - iou: "intersection_over_union"
        - dice: "dice_coefficient"
        - precision: "method_a_as_reference"
        - recall: "method_b_as_reference"

      categorical_agreement:
        - kappa: "cohen_kappa"
        - accuracy: "overall_accuracy"
        - per_class_f1: "f1_per_category"

      continuous_agreement:
        - correlation: "pearson_r"
        - rmse: "root_mean_square_error"
        - bias: "mean_error"

    thresholds:
      high_agreement: {iou: 0.8, kappa: 0.8}
      acceptable_agreement: {iou: 0.6, kappa: 0.6}
      low_agreement: {iou: 0.4, kappa: 0.4}

  # Cross-sensor comparison
  cross_sensor:
    description: "Compare results derived from different sensors"

    comparison_scenarios:
      - scenario: "optical_vs_sar_flood"
        sensor_a: "sentinel2"
        sensor_b: "sentinel1"
        condition: "both_available_within_24h"

      - scenario: "high_vs_medium_resolution"
        sensor_a: "sentinel2_10m"
        sensor_b: "landsat_30m"
        condition: "both_available_same_day"

      - scenario: "sar_constellation"
        sensor_a: "sentinel1a"
        sensor_b: "sentinel1b"
        condition: "acquisitions_within_6h"

    analysis:
      spatial_agreement_map:
        output: "agreement_raster"
        values:
          3: "both_detect"
          2: "sensor_a_only"
          1: "sensor_b_only"
          0: "neither_detect"

      disagreement_analysis:
        compute:
          - "disagreement_area_km2"
          - "disagreement_percent"
          - "spatial_pattern"  # Edge vs interior disagreement
        attribute_to:
          - "sensor_characteristics"
          - "atmospheric_conditions"
          - "temporal_offset"

  # Spatial uncertainty mapping
  uncertainty_mapping:
    methods:
      ensemble_variance:
        description: "Variance across multiple methods/sensors"
        output: "uncertainty_raster"
        interpretation: "higher_variance = higher_uncertainty"

      agreement_based:
        description: "Uncertainty from inter-method disagreement"
        calculation: |
          uncertainty = 1 - (agreement_count / total_methods)
        output: "confidence_raster"

      distance_to_boundary:
        description: "Higher uncertainty near classification boundaries"
        method: "distance_transform"
        parameters:
          decay_function: "exponential"
          characteristic_length_m: 30

    output_layers:
      - name: "uncertainty_continuous"
        type: "raster"
        values: [0, 1]
        description: "Pixel-wise uncertainty estimate"

      - name: "high_uncertainty_mask"
        type: "vector"
        description: "Polygons where uncertainty exceeds threshold"
        threshold: 0.5

      - name: "uncertainty_summary_by_zone"
        type: "table"
        description: "Aggregated uncertainty statistics per admin zone"
```

### Consensus Generation

```yaml
consensus_generation:
  # Consensus strategies
  strategies:
    majority_vote:
      description: "Pixel classified by majority of methods"
      applicable_to: "binary_classification"
      tie_breaker: "conservative"  # No detection on tie

    weighted_consensus:
      description: "Weighted combination based on method confidence"
      weights_from:
        - "historical_accuracy"
        - "sensor_suitability"
        - "atmospheric_conditions"
      output: "probability_surface"

    conservative_consensus:
      description: "Only include pixels detected by all methods"
      method: "intersection"
      use_when:
        - "high_stakes_decision"
        - "legal_implications"
        - "low_uncertainty_required"

    liberal_consensus:
      description: "Include pixels detected by any method"
      method: "union"
      use_when:
        - "screening_application"
        - "risk_averse_planning"
      annotation: "includes_uncertain_areas"

    hierarchical_consensus:
      description: "Trust higher-quality sources, fill gaps with others"
      hierarchy:
        1: "high_res_clear_optical"
        2: "sar_any_condition"
        3: "medium_res_optical"
        4: "coarse_res_modis"

  # Conservative output selection
  conservative_mode:
    triggers:
      - "cross_validation_disagreement > 0.3"
      - "uncertainty > 0.5"
      - "novel_conditions_detected"
      - "critical_application_flag"

    behavior:
      primary_output: "conservative_consensus"
      confidence_label: "CONSERVATIVE"
      annotations:
        - "conservative_estimate_used"
        - "see_alternatives_for_full_range"

    alternative_preservation:
      store: true
      formats: ["geojson", "cog"]
      retention_days: 90
      access: "expert_review_portal"

  # Alternative output preservation
  alternative_outputs:
    purpose: "Preserve non-consensus results for expert review"

    storage:
      location: "s3://geoint-alternatives/{event_id}/"
      structure:
        - "consensus/"
        - "method_a/"
        - "method_b/"
        - "method_c/"
        - "comparison/"

    metadata:
      per_alternative:
        - "method_name"
        - "method_version"
        - "input_data"
        - "parameters"
        - "confidence_score"
        - "agreement_with_consensus"

    expert_review_interface:
      features:
        - "side_by_side_comparison"
        - "swipe_tool"
        - "difference_highlighting"
        - "annotation_capability"
        - "override_workflow"

      outputs:
        - "expert_selected_output"
        - "expert_annotations"
        - "override_justification"

  # Output selection decision tree
  decision_tree:
    step_1:
      condition: "all_methods_agree (IoU > 0.8)"
      action: "use_weighted_consensus"
      confidence: "HIGH"

    step_2:
      condition: "majority_agree (IoU > 0.6)"
      action: "use_majority_vote"
      confidence: "MEDIUM"
      preserve_alternatives: true

    step_3:
      condition: "significant_disagreement (IoU < 0.6)"
      action: "use_conservative_consensus"
      confidence: "LOW"
      preserve_alternatives: true
      route_to_review: true

    step_4:
      condition: "methods_contradictory (IoU < 0.3)"
      action: "flag_for_expert_review"
      confidence: "INSUFFICIENT"
      block_automated_release: true
```

### QC Output Schema

```json
{
  "qa_report": {
    "type": "object",
    "required": ["event_id", "timestamp", "overall_status", "checks"],
    "properties": {
      "event_id": {"type": "string"},
      "timestamp": {"type": "string", "format": "date-time"},

      "overall_status": {
        "enum": ["PASS", "PASS_WITH_WARNINGS", "REVIEW_REQUIRED", "BLOCKED"]
      },

      "confidence_score": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      },

      "checks": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "check_name": {"type": "string"},
            "category": {"enum": ["spatial", "value", "historical", "artifact", "cross_validation"]},
            "status": {"enum": ["pass", "warning", "soft_fail", "hard_fail"]},
            "details": {"type": "string"},
            "metric_value": {"type": "number"},
            "threshold": {"type": "number"},
            "spatial_extent": {"$ref": "#/definitions/geometry"}
          }
        }
      },

      "cross_validation": {
        "type": "object",
        "properties": {
          "methods_compared": {"type": "array", "items": {"type": "string"}},
          "agreement_metrics": {
            "type": "object",
            "properties": {
              "iou": {"type": "number"},
              "kappa": {"type": "number"},
              "disagreement_area_km2": {"type": "number"}
            }
          },
          "consensus_method_used": {"type": "string"},
          "alternatives_preserved": {"type": "boolean"}
        }
      },

      "uncertainty_summary": {
        "type": "object",
        "properties": {
          "mean_uncertainty": {"type": "number"},
          "max_uncertainty": {"type": "number"},
          "high_uncertainty_area_km2": {"type": "number"},
          "high_uncertainty_percent": {"type": "number"}
        }
      },

      "actions_taken": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "action": {"enum": ["flagged", "clipped", "downgraded", "blocked", "routed_to_review"]},
            "reason": {"type": "string"},
            "affected_area": {"$ref": "#/definitions/geometry"}
          }
        }
      },

      "expert_review": {
        "type": "object",
        "properties": {
          "required": {"type": "boolean"},
          "reason": {"type": "string"},
          "deadline": {"type": "string", "format": "date-time"},
          "reviewer_assigned": {"type": "string"},
          "status": {"enum": ["pending", "in_progress", "completed", "overridden"]}
        }
      }
    }
  }
}
```

### Pipeline Validation

```yaml
pipeline_validation:
  pre_execution:
    - check: "dag_acyclicity"
      description: "Verify no circular dependencies"

    - check: "input_output_compatibility"
      description: "Verify data types match between steps"

    - check: "resource_availability"
      description: "Verify compute resources available"

    - check: "algorithm_versions"
      description: "Verify all algorithms available at specified versions"

  runtime:
    - check: "intermediate_quality"
      at: "after_each_step"
      metrics: ["null_percent", "value_range", "coverage"]

    - check: "execution_time"
      at: "each_step"
      alert_if: "exceeds_estimate_by_50_percent"

  post_execution:
    - check: "output_completeness"
      description: "All expected outputs generated"

    - check: "provenance_completeness"
      description: "Full lineage recorded"

    - check: "reproducibility_verification"
      description: "Hash matches expected for deterministic pipelines"
```

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

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/agent_message.schema.json",

  "type": "object",
  "required": ["message_id", "timestamp", "from_agent", "to_agent", "message_type", "payload"],
  "properties": {
    "message_id": {"type": "string", "format": "uuid"},
    "correlation_id": {
      "type": "string",
      "description": "Links related messages (e.g., request-response pairs)"
    },
    "timestamp": {"type": "string", "format": "date-time"},

    "from_agent": {
      "type": "string",
      "enum": ["orchestrator", "discovery", "pipeline", "quality", "reporting"]
    },
    "to_agent": {
      "type": "string",
      "enum": ["orchestrator", "discovery", "pipeline", "quality", "reporting"]
    },

    "message_type": {
      "type": "string",
      "enum": ["request", "response", "event", "error", "status_update"]
    },

    "priority": {
      "type": "string",
      "enum": ["critical", "high", "normal", "low"],
      "default": "normal"
    },

    "payload": {
      "type": "object",
      "description": "Message-type-specific content"
    },

    "context": {
      "type": "object",
      "description": "Shared execution context",
      "properties": {
        "event_id": {"type": "string"},
        "execution_id": {"type": "string"},
        "degraded_mode": {"type": "object"},
        "cumulative_confidence": {"$ref": "common.schema.json#/$defs/confidence_score"}
      }
    }
  }
}
```

**Message Flow Example:**

```yaml
# 1. Orchestrator → Discovery: Request data
- message_type: "request"
  from_agent: "orchestrator"
  to_agent: "discovery"
  payload:
    action: "discover_data"
    broker_query: {$ref: "broker_query"}

# 2. Discovery → Orchestrator: Response with datasets
- message_type: "response"
  from_agent: "discovery"
  to_agent: "orchestrator"
  payload:
    status: "success"
    broker_response: {$ref: "broker_response"}

# 3. Orchestrator → Pipeline: Execute analysis
- message_type: "request"
  from_agent: "orchestrator"
  to_agent: "pipeline"
  payload:
    action: "execute_pipeline"
    pipeline_id: "flood_sar_standard_v1"
    inputs: {$ref: "resolved_inputs"}

# 4. Pipeline → Quality: Validate results (in-band)
- message_type: "request"
  from_agent: "pipeline"
  to_agent: "quality"
  payload:
    action: "validate_step"
    step_id: "detect"
    result: {$ref: "step_output"}

# 5. Quality → Pipeline: QC gate decision
- message_type: "response"
  from_agent: "quality"
  to_agent: "pipeline"
  payload:
    status: "pass_with_warnings"
    qa_report: {$ref: "qa_report_fragment"}
    action: "continue"  # or "abort", "downgrade"
```

### Agent State Management

```yaml
agent_state:
  schema:
    execution_id: "string"
    event_id: "string"
    agent_id: "string"
    status: "enum[pending, running, paused, completed, failed, cancelled]"

    started_at: "datetime"
    updated_at: "datetime"
    completed_at: "datetime"

    current_step: "string"
    progress_percent: "number"

    inputs_received: "object"
    outputs_produced: "object"

    error: "object | null"
    retry_count: "integer"

    checkpoints: "array"  # For resumability

  persistence:
    backend: "postgresql"  # or "dynamodb" for serverless
    retention: "90d"

  recovery:
    on_restart: "resume_from_checkpoint"
    checkpoint_frequency: "after_each_step"
```

### Failure Handling & Retry Policies

```yaml
failure_handling:
  # Retry configuration per agent
  retry_policies:
    discovery:
      max_retries: 3
      backoff: "exponential"
      base_delay_seconds: 5
      max_delay_seconds: 300
      retryable_errors:
        - "timeout"
        - "rate_limit"
        - "transient_network"
      non_retryable_errors:
        - "invalid_query"
        - "authentication_failed"

    pipeline:
      max_retries: 2
      backoff: "linear"
      base_delay_seconds: 10
      retryable_errors:
        - "resource_exhausted"
        - "transient_io"
      checkpoint_before_retry: true

    quality:
      max_retries: 1
      retryable_errors:
        - "timeout"

  # Escalation paths
  escalation:
    on_max_retries_exceeded:
      action: "escalate_to_orchestrator"
      orchestrator_options:
        - "try_alternative_algorithm"
        - "proceed_with_degraded_mode"
        - "abort_with_partial_results"
        - "abort_completely"

    on_critical_failure:
      action: "immediate_abort"
      notify: ["operator", "webhook"]
      preserve_state: true

  # Partial failure handling
  partial_failure:
    pipeline:
      on_step_failure:
        preserve_successful_outputs: true
        mark_failed_outputs: "FAILED"
        continue_independent_branches: true

    discovery:
      on_provider_failure:
        continue_with_available: true
        flag: "PARTIAL_DATA_COVERAGE"

  # Timeout configuration
  timeouts:
    discovery:
      query_timeout_seconds: 120
      total_timeout_seconds: 600
    pipeline:
      step_timeout_seconds: 1800
      total_timeout_seconds: 14400  # 4 hours
    quality:
      check_timeout_seconds: 300
      total_timeout_seconds: 900
```

### QC Gating Integration with Pipeline Execution

```yaml
qc_pipeline_integration:
  # When QC runs during pipeline execution
  gate_points:
    - point: "after_each_step"
      enabled_by: "step.qc_gate.enabled"
      checks: "step.qc_gate.checks"

    - point: "before_final_output"
      enabled: true
      checks: "full_qa_suite"

    - point: "on_demand"
      trigger: "anomaly_detected_in_intermediate"

  # QC gate actions and pipeline response
  gate_actions:
    pass:
      pipeline_action: "continue"
      confidence_modifier: 1.0

    pass_with_warnings:
      pipeline_action: "continue"
      confidence_modifier: 0.9
      annotations: ["warnings_attached"]

    soft_fail:
      pipeline_action: "pause_and_decide"
      decision_options:
        - action: "retry_step"
          condition: "retry_count < max_retries"
        - action: "try_alternative_algorithm"
          condition: "alternatives_available"
        - action: "continue_with_degradation"
          confidence_modifier: 0.7
          flags: ["QC_SOFT_FAIL"]
        - action: "abort_pipeline"
          preserve_partial: true

    hard_fail:
      pipeline_action: "abort"
      preserve_partial: true
      require_human_override: true

    review_required:
      pipeline_action: "pause"
      timeout_hours: 24
      on_timeout: "apply_conservative_output"
      on_approval: "continue"
      on_rejection: "abort"

  # Confidence propagation through pipeline
  confidence_propagation:
    method: "multiplicative"

    sources:
      - name: "data_selection"
        from: "broker_response.degraded_mode.confidence_modifier"

      - name: "algorithm"
        from: "algorithm_registry.validation.accuracy_range.midpoint"

      - name: "qc_gate"
        from: "qa_report.confidence_score"

    composition: |
      final_confidence = data_confidence × algorithm_confidence × qc_confidence

    output:
      field: "product.confidence"
      flags_from: "all_sources"
```

### Unified Confidence Composition Model

```yaml
confidence_composition:
  # All confidence scores normalized to 0-1 range
  normalization:
    percentage_to_ratio: "value / 100"
    accuracy_range_to_confidence: "(min + max) / 2"

  # Composition formula
  formula: |
    final_confidence = (
      data_confidence^w_data ×
      algorithm_confidence^w_algo ×
      qc_confidence^w_qc
    )^(1/(w_data + w_algo + w_qc))

    # Geometric mean with weights

  weights:
    w_data: 0.3      # Data availability/quality
    w_algo: 0.3      # Algorithm validation accuracy
    w_qc: 0.4        # QC check results

  # Confidence sources
  sources:
    data_confidence:
      base: 1.0
      modifiers:
        - condition: "degraded_mode.level >= 1"
          multiply_by: "degraded_mode.confidence_modifier"
        - condition: "single_sensor_mode"
          multiply_by: 0.9
        - condition: "temporal_interpolation"
          multiply_by: 0.8

    algorithm_confidence:
      base: "algorithm_registry[selected].validation.accuracy_range.midpoint"
      modifiers:
        - condition: "region not in validated_regions"
          multiply_by: 0.9
        - condition: "novel_conditions"
          multiply_by: 0.8

    qc_confidence:
      base: "qa_report.confidence_score"
      modifiers:
        - condition: "cross_validation_disagreement > 0.3"
          multiply_by: 0.8
        - condition: "consensus_method == 'conservative'"
          multiply_by: 0.9

  # Output classification
  output_levels:
    high:
      range: [0.8, 1.0]
      label: "HIGH_CONFIDENCE"
      usable_for: ["operational_decisions", "official_reports"]

    medium:
      range: [0.6, 0.8)
      label: "MEDIUM_CONFIDENCE"
      usable_for: ["situational_awareness", "preliminary_reports"]

    low:
      range: [0.4, 0.6)
      label: "LOW_CONFIDENCE"
      usable_for: ["screening", "with_caveats"]

    insufficient:
      range: [0.0, 0.4)
      label: "INSUFFICIENT_CONFIDENCE"
      usable_for: ["expert_review_only"]
      action: "flag_for_review"
```

### Product Schema (`product.schema.json`)

Final output product specification:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/product.schema.json",

  "type": "object",
  "required": ["product_id", "event_id", "product_type", "created_at", "confidence", "data", "provenance"],
  "properties": {
    "product_id": {"type": "string"},
    "event_id": {"type": "string"},
    "product_type": {
      "type": "string",
      "enum": ["flood_extent", "burn_severity", "damage_assessment", "impact_estimate"]
    },
    "version": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},

    "confidence": {
      "type": "object",
      "required": ["overall", "level"],
      "properties": {
        "overall": {"$ref": "common.schema.json#/$defs/confidence_score"},
        "level": {
          "type": "string",
          "enum": ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "INSUFFICIENT_CONFIDENCE"]
        },
        "components": {
          "type": "object",
          "properties": {
            "data": {"$ref": "common.schema.json#/$defs/confidence_score"},
            "algorithm": {"$ref": "common.schema.json#/$defs/confidence_score"},
            "qc": {"$ref": "common.schema.json#/$defs/confidence_score"}
          }
        }
      }
    },

    "quality_flags": {
      "type": "array",
      "items": {"$ref": "common.schema.json#/$defs/quality_flag"}
    },

    "data": {
      "type": "object",
      "description": "Product-specific data payload",
      "properties": {
        "geometry": {"$ref": "common.schema.json#/$defs/geometry"},
        "raster_uri": {"$ref": "common.schema.json#/$defs/uri"},
        "vector_uri": {"$ref": "common.schema.json#/$defs/uri"},
        "statistics": {"type": "object"},
        "observables": {
          "type": "object",
          "description": "Named observable values with per-observable confidence",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "value": {},
              "unit": {"type": "string"},
              "confidence": {"$ref": "common.schema.json#/$defs/confidence_score"},
              "flags": {"type": "array", "items": {"$ref": "common.schema.json#/$defs/quality_flag"}}
            }
          }
        }
      }
    },

    "provenance": {
      "type": "object",
      "description": "Full lineage reference",
      "properties": {
        "provenance_id": {"type": "string"},
        "source_datasets": {"type": "array", "items": {"type": "string"}},
        "pipeline_id": {"type": "string"},
        "algorithm_versions": {"type": "object"},
        "qa_report_id": {"type": "string"}
      }
    },

    "limitations": {
      "type": "array",
      "description": "Known limitations and caveats",
      "items": {"type": "string"}
    },

    "alternatives_available": {
      "type": "boolean",
      "description": "Whether non-consensus alternatives are preserved"
    }
  }
}
```

### Extended Provenance for Decision Rationale

Provenance now captures not just data lineage but decision rationale:

```yaml
provenance_extensions:
  decision_records:
    # Sensor selection rationale
    - decision_type: "sensor_selection"
      step_id: "discovery"
      timestamp: "2024-09-18T10:00:00Z"

      selected: "sentinel1_grd"
      alternatives_considered:
        - id: "sentinel2_l2a"
          score: 0.65
          rejected_reason: "cloud_cover_exceeded_threshold"
        - id: "landsat9"
          score: 0.72
          rejected_reason: "lower_resolution"

      selection_criteria:
        atmospheric_condition: "cloudy"
        strategy: "sar_preferred"

      rationale: "SAR selected due to 75% cloud cover; optical infeasible"

    # Algorithm selection rationale
    - decision_type: "algorithm_selection"
      step_id: "pipeline_assembly"
      timestamp: "2024-09-18T10:05:00Z"

      selected: "flood.baseline.threshold_sar"
      alternatives_considered:
        - id: "flood.advanced.unet_segmentation"
          score: 0.92
          rejected_reason: "gpu_unavailable"

      selection_features:
        event_type: "flood.coastal"
        data_available: ["sar", "dem"]
        compute_constraints: {gpu: false, memory_gb: 8}

      rationale: "Baseline threshold selected; ML model requires GPU not available"

    # QC consensus rationale
    - decision_type: "consensus_selection"
      step_id: "quality_validation"
      timestamp: "2024-09-18T10:30:00Z"

      selected: "conservative_consensus"
      agreement_metrics:
        iou: 0.58
        methods_compared: ["threshold_sar", "change_detection"]

      rationale: "Conservative consensus due to IoU < 0.6 between methods"
      alternatives_preserved: true
      alternative_location: "s3://geoint-alternatives/evt_001/"
```

## Implementation Phases

### Phase 1: Foundation & Core Schemas
1. Project setup with `pyproject.toml` and dependencies
2. JSON Schema files for all specification types
3. Schema validator with helpful error messages
4. Event class taxonomy (flood, wildfire, storm)
5. Example YAML specifications

**Files:**
- `pyproject.toml`
- `openspec/schemas/*.schema.json` (intent, event, datasource, pipeline, ingestion, quality, provenance)
- `openspec/validator.py`
- `openspec/definitions/event_classes/*.yaml`

### Phase 2: Intent Resolution System
1. Event class registry with hierarchical taxonomy
2. NLP classifier for natural language inference
3. Resolution logic with confidence scoring
4. Override handling and structured output

**Files:**
- `core/intent/registry.py`
- `core/intent/classifier.py`
- `core/intent/resolver.py`

### Phase 3: Data Broker & Ingestion
1. Multi-source data broker with provider registry
2. STAC client for open data catalogs
3. Constraint evaluation and ranking engine
4. Intelligent sensor selection (atmospheric-aware)
5. Cache system with spatial-temporal indexing
6. Ingestion pipeline with normalization

**Files:**
- `core/data/broker.py`
- `core/data/providers/*.py`
- `core/data/evaluation/*.py`
- `core/data/selection/*.py`
- `core/data/cache/*.py`
- `core/data/ingestion/*.py`

### Phase 4: Analysis & Pipeline Engine
1. Algorithm library with registry
2. Algorithm selection engine (rule + ML hybrid)
3. Dynamic pipeline assembler (DAG-based)
4. Multi-sensor fusion engine
5. Forecast integration and scenario analysis
6. Distributed execution (Dask/Ray)

**Files:**
- `core/analysis/library/*.py`
- `core/analysis/selection/*.py`
- `core/analysis/assembly/*.py`
- `core/analysis/fusion/*.py`
- `core/analysis/forecast/*.py`
- `core/analysis/execution/*.py`

### Phase 5: Quality Control & Validation
1. Automated sanity checks (spatial, value, temporal)
2. Cross-model and cross-sensor validation
3. Consensus generation with alternatives
4. Uncertainty quantification and mapping
5. Gating and expert review routing

**Files:**
- `core/quality/sanity/*.py`
- `core/quality/validation/*.py`
- `core/quality/uncertainty/*.py`
- `core/quality/actions/*.py`

### Phase 6: Agent Orchestration
1. Base agent class with lifecycle
2. Orchestrator agent with delegation
3. Specialized agents (discovery, pipeline, quality, reporting)
4. Message passing and state persistence

**Files:**
- `agents/base.py`
- `agents/orchestrator/*.py`
- `agents/discovery/*.py`
- `agents/pipeline/*.py`
- `agents/quality/*.py`
- `agents/reporting/*.py`

### Phase 7: API & Deployment
1. FastAPI application with routes
2. Authentication and rate limiting
3. Webhook notifications
4. Serverless deployment configuration

**Files:**
- `api/main.py`
- `api/routes/*.py`
- `api/models/*.py`
- `deploy/serverless.yml`

## Parallelized Execution Plan

The implementation phases can be reorganized for parallel execution:

```
Timeline (Work Streams)
═══════════════════════════════════════════════════════════════════════════

Stream A          Stream B          Stream C          Integration
────────          ────────          ────────          ───────────

┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Foundation                          │
│            (All schemas, validator, event classes)              │
│                    [SEQUENTIAL - Required First]                │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   PHASE 2:    │   │    PHASE 3:     │   │   PHASE 4a:     │
│    Intent     │   │   Data Broker   │   │ Algorithm Lib   │
│  Resolution   │   │   & Ingestion   │   │   (Registry)    │
│               │   │                 │   │                 │
│ - Registry    │   │ - Broker core   │   │ - Library arch  │
│ - Classifier  │   │ - Providers     │   │ - Baseline algs │
│ - Resolver    │   │ - Evaluation    │   │ - Selection eng │
│               │   │ - Selection     │   │                 │
│               │   │ - Cache         │   │                 │
│               │   │ - Ingestion     │   │                 │
└───────┬───────┘   └────────┬────────┘   └────────┬────────┘
        │                    │                     │
        │                    ▼                     │
        │           ┌─────────────────┐            │
        │           │   PHASE 4b:     │◄───────────┘
        │           │ Pipeline Engine │
        │           │                 │
        │           │ - Assembler     │
        │           │ - Fusion engine │
        │           │ - Forecast intg │
        │           │ - Execution     │
        │           └────────┬────────┘
        │                    │
        │                    ▼
        │           ┌─────────────────┐   ┌─────────────────┐
        │           │    PHASE 5:     │   │   PHASE 5a:     │
        │           │  QC & Validation│◄──│  QC Framework   │
        │           │                 │   │  (can start     │
        │           │ - Sanity checks │   │   with Phase 1) │
        │           │ - Cross-valid   │   └─────────────────┘
        │           │ - Consensus     │
        │           │ - Uncertainty   │
        │           │ - Gating        │
        │           └────────┬────────┘
        │                    │
        └────────────────────┼────────────────────────────────┐
                             ▼                                │
                    ┌─────────────────────────────────────────┴──┐
                    │              PHASE 6: Agent Orchestration  │
                    │                                            │
                    │ - Base agent class                         │
                    │ - Orchestrator (integrates all components) │
                    │ - Discovery agent (wraps Phase 3)          │
                    │ - Pipeline agent (wraps Phase 4)           │
                    │ - Quality agent (wraps Phase 5)            │
                    │ - Reporting agent                          │
                    │                                            │
                    │         [INTEGRATION POINT]                │
                    └────────────────────┬───────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────────┐
                    │              PHASE 7: API & Deployment     │
                    │                                            │
                    │ - FastAPI application                      │
                    │ - Routes & models                          │
                    │ - Serverless config                        │
                    │                                            │
                    │         [FINAL INTEGRATION]                │
                    └────────────────────────────────────────────┘
```

### Parallel Execution Summary

| Work Stream | Components | Dependencies | Team Skills |
|-------------|-----------|--------------|-------------|
| **Stream A** | Intent Resolution | Phase 1 schemas | NLP, Python |
| **Stream B** | Data Broker & Ingestion | Phase 1 schemas | Geospatial, STAC, APIs |
| **Stream C** | Algorithm Library | Phase 1 schemas | Remote Sensing, ML |
| **Integration** | Agents, API | All streams complete | Architecture, DevOps |

### Critical Path

The critical path runs through:
1. Phase 1 (Foundation) →
2. Phase 3 (Data Broker) →
3. Phase 4b (Pipeline Engine) →
4. Phase 5 (QC) →
5. Phase 6 (Agents) →
6. Phase 7 (API)

**Optimization opportunity**: Phase 2 (Intent) and Phase 4a (Algorithm Library) can be developed entirely off the critical path.

### Sub-Phase Parallelization

Within each phase, further parallelization is possible:

**Phase 3 internal parallelization:**
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Provider plugins │  │ Constraint eval  │  │  Cache system    │
│ (sentinel2.py)   │  │ (ranking.py)     │  │  (manager.py)    │
│ (sentinel1.py)   │  │ (tradeoffs.py)   │  │  (index.py)      │
│ (landsat.py)     │  │                  │  │                  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         └──────────────────────┼──────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │    broker.py         │
                    │  (integration layer) │
                    └──────────────────────┘
```

**Phase 5 internal parallelization:**
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Sanity checks   │  │ Cross-validation │  │   Uncertainty    │
│  (spatial.py)    │  │ (cross_model.py) │  │ (quantification) │
│  (values.py)     │  │ (cross_sensor.py)│  │ (propagation.py) │
│  (temporal.py)   │  │                  │  │                  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         └──────────────────────┼──────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │   consensus.py       │
                    │   gating.py          │
                    └──────────────────────┘
```

### Recommended Team Allocation (3-Team Setup)

| Team | Primary Focus | Secondary Focus |
|------|---------------|-----------------|
| **Team 1** | Phase 1 (lead), Phase 2 | Phase 6 (integration) |
| **Team 2** | Phase 3, Phase 4b | Phase 7 |
| **Team 3** | Phase 4a, Phase 5 | Testing & validation |

### Handoff Points

| From | To | Interface | Artifact |
|------|-----|-----------|----------|
| Phase 1 | Phases 2,3,4a | JSON Schemas | `openspec/schemas/*.json` |
| Phase 3 | Phase 4b | Broker Response | `broker_response.schema.json` |
| Phase 4b | Phase 5 | Pipeline Output | Analysis results + metadata |
| Phase 5 | Phase 6 | QA Report | `qa_report.schema.json` |
| All | Phase 6 | Agent Interfaces | `agent_message.schema.json` |

## Verification

1. **Schema validation**:
   ```bash
   pytest tests/test_schemas.py -v
   ```

2. **Intent resolution**:
   ```bash
   pytest tests/test_intent.py -v
   ```

3. **Data broker**:
   ```bash
   pytest tests/test_broker.py -v
   ```

4. **Pipeline execution**:
   ```bash
   pytest tests/test_pipeline.py -v
   ```

5. **End-to-end**:
   ```bash
   # Start API server
   uvicorn api.main:app --reload &

   # Submit event specification
   curl -X POST http://localhost:8000/events \
     -H "Content-Type: application/yaml" \
     -d @examples/flood_event.yaml

   # Check job status
   curl http://localhost:8000/events/{event_id}/status
   ```

6. **Serverless deployment**:
   ```bash
   cd deploy && serverless deploy --stage dev
   ```

## First Implementation: Phase 1 Files

```
multiverse_dive/
├── pyproject.toml
├── openspec/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── intent.schema.json
│   │   ├── event.schema.json
│   │   ├── datasource.schema.json
│   │   ├── pipeline.schema.json
│   │   ├── ingestion.schema.json
│   │   ├── quality.schema.json
│   │   └── provenance.schema.json
│   ├── definitions/
│   │   └── event_classes/
│   │       ├── flood.yaml
│   │       ├── wildfire.yaml
│   │       └── storm.yaml
│   └── validator.py
├── core/
│   └── intent/
│       ├── __init__.py
│       ├── registry.py
│       └── resolver.py
├── examples/
│   └── flood_event.yaml
└── tests/
    ├── __init__.py
    ├── test_schemas.py
    └── test_intent.py
```

---

## Distributed Raster Processing Architecture

**Last Updated:** 2026-01-11
**Status:** Specification for next implementation phase

### Challenge: Large-Scale Earth Observation Processing

#### Problem Statement

Earth observation datasets are massive and growing:
- Single Sentinel-2 scene: 500MB-5GB (100km x 100km at 10m resolution)
- Continental analysis: 100,000km² = 1,000+ scenes = 500GB-5TB
- Current serial tiled processing: 20-30 minutes for 100km² on laptop
- Memory constraints: Existing tiling works but doesn't parallelize across cores
- Download bottleneck: Must download entire scenes before processing begins

#### Goals

1. **Laptop-Scale Parallelization:** Leverage all CPU cores for 4-8x speedup
2. **Cloud-Scale Distribution:** Process continental areas on Spark/Flink clusters  
3. **Streaming Ingestion:** Never download full scenes—stream only needed tiles
4. **Transparent Scaling:** Same API works on laptop or 100-node cluster

### Architecture Overview

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
│               Virtual Raster Index Builder                      │◄── NEW
│  Creates: GDAL VRT referencing COG tiles via HTTP               │
│  NO DOWNLOAD: Index metadata only (~100KB per scene)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Router                             │◄── NEW
│  Decides: Local Dask vs Distributed Dask vs Apache Sedona       │
│  Based on: AOI size, available resources, cluster availability  │
└───┬─────────────┬───────────────────────┬──────────────────────┘
    │             │                       │
    ▼             ▼                       ▼
┌─────────┐ ┌──────────────┐    ┌─────────────────────┐
│ Serial  │ │ Dask Local   │    │ Dask Distributed /  │
│ (<100   │ │ (100-1000    │    │ Apache Sedona       │
│  tiles) │ │  tiles)      │    │ (1000+ tiles)       │
└─────────┘ └──────────────┘    └─────────────────────┘
    │             │                       │
    └─────────────┴───────────────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │ Streaming Tile      │◄── NEW
         │ Processing          │
         │ (HTTP Range Reads)  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Result Stitching    │
         │ & Mosaic Generation │
         └─────────────────────┘
```

### Component 1: Virtual Raster Index

**Purpose:** Build lightweight index of available raster data without downloading

**Location:** `core/data/ingestion/virtual_index.py`

#### Virtual Index Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/virtual_index.schema.json",
  
  "type": "object",
  "required": ["index_id", "source_type", "tiles", "spatial_index"],
  "properties": {
    "index_id": {
      "type": "string",
      "description": "Unique index identifier"
    },
    
    "source_type": {
      "type": "string",
      "enum": ["vrt", "stac_virtual", "mosaic_json"],
      "description": "Type of virtual index"
    },
    
    "tiles": {
      "type": "array",
      "description": "Tile metadata for spatial index",
      "items": {
        "type": "object",
        "required": ["tile_id", "url", "bbox", "resolution_m"],
        "properties": {
          "tile_id": {"type": "string"},
          "url": {
            "type": "string",
            "format": "uri",
            "description": "HTTP(S) URL to COG tile with byte range support"
          },
          "bbox": {"$ref": "common.schema.json#/$defs/bbox"},
          "resolution_m": {"type": "number", "minimum": 0},
          "bands": {
            "type": "array",
            "items": {"type": "string"}
          },
          "temporal": {
            "type": "string",
            "format": "date-time"
          },
          "crs": {"$ref": "common.schema.json#/$defs/crs"},
          "size_bytes": {
            "type": "integer",
            "description": "Uncompressed tile size for memory planning"
          }
        }
      }
    },
    
    "spatial_index": {
      "type": "object",
      "description": "R-tree spatial index for fast AOI queries",
      "properties": {
        "type": {"type": "string", "enum": ["rtree", "quadtree"]},
        "index_file": {
          "type": "string",
          "description": "Path to serialized spatial index"
        }
      }
    },
    
    "vrt_path": {
      "type": "string",
      "description": "Path to GDAL VRT file if source_type=vrt"
    },
    
    "statistics": {
      "type": "object",
      "properties": {
        "total_tiles": {"type": "integer"},
        "total_coverage_km2": {"type": "number"},
        "index_size_mb": {"type": "number"},
        "estimated_data_size_gb": {"type": "number"}
      }
    }
  }
}
```

#### Virtual Index Implementation

```python
# core/data/ingestion/virtual_index.py

from typing import List, Dict
from dataclasses import dataclass
import rasterio
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
from rtree import index as rtree_index

@dataclass
class TileReference:
    """Metadata for a single raster tile without loading data."""
    tile_id: str
    url: str  # HTTP(S) URL to COG
    bbox: tuple  # (minx, miny, maxx, maxy)
    resolution_m: float
    bands: List[str]
    temporal: str  # ISO timestamp
    crs: str
    size_bytes: int

class VirtualRasterIndex:
    """
    Build and query virtual raster index from STAC catalog results.
    Never downloads actual raster data—only metadata.
    """
    
    def __init__(self):
        self.tiles: List[TileReference] = []
        self.spatial_index = rtree_index.Index()
        
    def add_from_stac_items(self, stac_items: List[Dict]) -> None:
        """
        Parse STAC items and build tile references.
        Uses STAC asset URLs with /vsicurl/ for remote access.
        """
        for item in stac_items:
            for asset_key, asset in item['assets'].items():
                if asset['type'] in ['image/tiff', 'image/vnd.stac.geotiff', 'application/x-geotiff']:
                    tile = self._parse_stac_asset(item, asset_key, asset)
                    self.tiles.append(tile)
                    
                    # Add to R-tree spatial index
                    idx = len(self.tiles) - 1
                    self.spatial_index.insert(idx, tile.bbox)
    
    def query_aoi(self, aoi_bbox: tuple) -> List[TileReference]:
        """
        Fast spatial query using R-tree index.
        Returns only tiles intersecting AOI.
        """
        intersecting_indices = list(self.spatial_index.intersection(aoi_bbox))
        return [self.tiles[i] for i in intersecting_indices]
    
    def build_vrt(self, output_path: str, aoi_bbox: tuple = None) -> str:
        """
        Create GDAL VRT file from tile references.
        VRT uses /vsicurl/ for HTTP range request streaming.
        """
        tiles_to_include = self.query_aoi(aoi_bbox) if aoi_bbox else self.tiles
        
        vrt_options = {
            'resolution': 'highest',
            'resampling': rasterio.enums.Resampling.bilinear
        }
        
        # Build VRT XML referencing remote COGs
        # GDAL will stream only needed blocks via HTTP range requests
        # ...implementation details...
        
        return output_path
    
    def estimate_memory_requirements(self, aoi_bbox: tuple, tile_size_px: int = 256) -> Dict:
        """
        Estimate memory needed for processing AOI with given tile size.
        """
        tiles = self.query_aoi(aoi_bbox)
        total_pixels = sum(self._calculate_pixels(t, aoi_bbox) for t in tiles)
        
        # Assume float32 per band, 3 bands typical
        bytes_per_pixel = 4 * 3
        peak_memory_gb = (tile_size_px ** 2) * bytes_per_pixel / (1024**3)
        
        return {
            'tiles_needed': len(tiles),
            'total_pixels_million': total_pixels / 1e6,
            'peak_memory_gb': peak_memory_gb,
            'recommended_workers': min(len(tiles), os.cpu_count())
        }
```

### Component 2: Execution Router

**Purpose:** Intelligently route processing to appropriate execution backend

**Location:** `core/analysis/execution/router.py`

#### Execution Profile Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://openspec.io/schemas/execution_profile.schema.json",
  
  "type": "object",
  "required": ["profile_id", "resource_limits", "backend"],
  "properties": {
    "profile_id": {
      "type": "string",
      "enum": ["laptop", "workstation", "edge", "cloud_dask", "cloud_sedona"]
    },
    
    "resource_limits": {
      "type": "object",
      "properties": {
        "max_memory_gb": {"type": "number", "minimum": 0},
        "max_workers": {"type": "integer", "minimum": 1},
        "max_tiles_parallel": {"type": "integer", "minimum": 1},
        "tile_size_px": {"type": "integer", "enum": [256, 512, 1024]},
        "enable_distributed": {"type": "boolean"}
      }
    },
    
    "backend": {
      "type": "object",
      "required": ["type"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["serial", "dask_local", "dask_distributed", "sedona_spark"]
        },
        
        "dask_config": {
          "type": "object",
          "properties": {
            "scheduler_address": {"type": "string"},
            "n_workers": {"type": "integer"},
            "threads_per_worker": {"type": "integer"},
            "memory_limit": {"type": "string"}
          }
        },
        
        "sedona_config": {
          "type": "object",
          "properties": {
            "spark_master": {"type": "string"},
            "executor_memory": {"type": "string"},
            "executor_instances": {"type": "integer"},
            "spatial_partitioning": {
              "type": "string",
              "enum": ["quadtree", "kdbtree", "voronoi"]
            }
          }
        }
      }
    },
    
    "routing_rules": {
      "type": "object",
      "description": "Rules for auto-selecting backend based on job characteristics",
      "properties": {
        "tile_count_thresholds": {
          "type": "object",
          "properties": {
            "serial_max": {"type": "integer", "default": 100},
            "dask_local_max": {"type": "integer", "default": 1000},
            "dask_distributed_max": {"type": "integer", "default": 10000}
          }
        },
        
        "memory_thresholds": {
          "type": "object",
          "properties": {
            "peak_memory_trigger_gb": {
              "type": "number",
              "description": "Switch to distributed if peak > this"
            }
          }
        }
      }
    }
  }
}
```

#### Execution Router Implementation

```python
# core/analysis/execution/router.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class ExecutionBackend(Enum):
    SERIAL = "serial"
    DASK_LOCAL = "dask_local"
    DASK_DISTRIBUTED = "dask_distributed"
    SEDONA_SPARK = "sedona_spark"

@dataclass
class ExecutionContext:
    """Context for execution routing decision."""
    tile_count: int
    estimated_memory_gb: float
    aoi_size_km2: float
    algorithm_complexity: str  # "simple", "medium", "complex"
    available_resources: Dict[str, Any]
    cluster_available: bool

class ExecutionRouter:
    """
    Routes execution to appropriate backend based on job characteristics.
    Provides transparent scaling from laptop to cloud.
    """
    
    def __init__(self, default_profile: str = "laptop"):
        self.profiles = self._load_profiles()
        self.default_profile = default_profile
        
    def route(self, context: ExecutionContext, force_backend: Optional[ExecutionBackend] = None) -> ExecutionBackend:
        """
        Decide which execution backend to use.
        
        Decision logic:
        1. If force_backend specified, use it (with validation)
        2. Apply routing rules based on:
           - Tile count (serial < 100, dask_local < 1000, distributed >= 1000)
           - Memory requirements (if > available, need distributed)
           - Cluster availability (if Spark cluster accessible, prefer for large jobs)
           - Algorithm complexity (ML models benefit from distributed)
        """
        if force_backend:
            if self._validate_backend_available(force_backend, context):
                return force_backend
            else:
                raise ValueError(f"Backend {force_backend} not available or unsuitable")
        
        # Auto-routing logic
        if context.tile_count < 100:
            return ExecutionBackend.SERIAL
        
        elif context.tile_count < 1000:
            if context.estimated_memory_gb <= context.available_resources.get('memory_gb', 4):
                return ExecutionBackend.DASK_LOCAL
            else:
                return ExecutionBackend.DASK_DISTRIBUTED if context.cluster_available else ExecutionBackend.DASK_LOCAL
        
        else:  # >= 1000 tiles
            if context.cluster_available:
                # Prefer Sedona for very large raster jobs on Spark
                if context.tile_count > 5000:
                    return ExecutionBackend.SEDONA_SPARK
                else:
                    return ExecutionBackend.DASK_DISTRIBUTED
            else:
                # Fallback to dask local with chunking
                return ExecutionBackend.DASK_LOCAL
    
    def get_executor(self, backend: ExecutionBackend) -> 'BaseExecutor':
        """Factory method to instantiate appropriate executor."""
        if backend == ExecutionBackend.SERIAL:
            from core.analysis.execution.runner import SerialExecutor
            return SerialExecutor()
        
        elif backend == ExecutionBackend.DASK_LOCAL:
            from core.analysis.execution.dask_tiled import DaskLocalExecutor
            return DaskLocalExecutor()
        
        elif backend == ExecutionBackend.DASK_DISTRIBUTED:
            from core.analysis.execution.dask_tiled import DaskDistributedExecutor
            return DaskDistributedExecutor()
        
        elif backend == ExecutionBackend.SEDONA_SPARK:
            from core.analysis.execution.sedona_backend import SedonaExecutor
            return SedonaExecutor()
```

### Component 3: Dask Parallelization

**Purpose:** Multi-core parallelization on laptop/workstation using Dask

**Location:** `core/analysis/execution/dask_tiled.py`

#### Dask Tiled Executor

```python
# core/analysis/execution/dask_tiled.py

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import rasterio
from rasterio.windows import Window
import numpy as np

class DaskLocalExecutor:
    """
    Execute algorithms using Dask on local machine.
    Parallelizes tile processing across CPU cores.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or os.cpu_count()
        self.client = None
        
    def execute_algorithm(self, algorithm, virtual_index: VirtualRasterIndex, aoi_bbox: tuple, **params):
        """
        Execute algorithm on AOI using Dask parallelization.
        
        Workflow:
        1. Query virtual index for tiles intersecting AOI
        2. Create Dask delayed tasks for each tile
        3. Execute in parallel across workers
        4. Stitch results into final mosaic
        """
        
        # Get tiles for AOI
        tiles = virtual_index.query_aoi(aoi_bbox)
        
        # Create delayed tasks
        delayed_results = []
        for tile in tiles:
            # Create delayed task that reads tile via HTTP range request
            delayed_task = dask.delayed(self._process_single_tile)(
                algorithm=algorithm,
                tile_url=tile.url,
                tile_bbox=tile.bbox,
                aoi_bbox=aoi_bbox,
                params=params
            )
            delayed_results.append(delayed_task)
        
        # Execute in parallel
        with LocalCluster(n_workers=self.n_workers, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                results = dask.compute(*delayed_results)
        
        # Stitch results
        final_result = self._stitch_results(results, aoi_bbox)
        return final_result
    
    @staticmethod
    def _process_single_tile(algorithm, tile_url: str, tile_bbox: tuple, aoi_bbox: tuple, params: Dict):
        """
        Process a single tile with HTTP range request streaming.
        
        Uses rasterio's /vsicurl/ to read only needed pixels via HTTP range requests.
        Never downloads full tile.
        """
        # Calculate window for AOI intersection
        window = calculate_intersection_window(tile_bbox, aoi_bbox)
        
        # Stream tile data via HTTP range request
        with rasterio.open(f'/vsicurl/{tile_url}') as src:
            # Read only the window we need
            data = src.read(window=window)
            
            # Execute algorithm on this tile
            result = algorithm.process_tile(data, **params)
            
        return {
            'bbox': window_to_bbox(window, tile_bbox),
            'data': result,
            'tile_id': tile_url
        }
    
    def _stitch_results(self, results: List[Dict], aoi_bbox: tuple) -> np.ndarray:
        """Combine tile results into final mosaic."""
        # Use Dask for efficient stitching
        # Handle overlaps, blending, etc.
        pass
```

### Component 4: Apache Sedona Integration

**Purpose:** Spark-based distributed processing for continental-scale analyses

**Location:** `core/analysis/execution/sedona_backend.py`

#### Sedona Executor

```python
# core/analysis/execution/sedona_backend.py

from pyspark.sql import SparkSession
from sedona.spark import *
from sedona.core.spatialOperator import RangeQuery
from sedona.core.enums import GridType, IndexType
import geopandas as gpd

class SedonaExecutor:
    """
    Execute algorithms using Apache Sedona on Spark cluster.
    Handles massive-scale raster processing (100,000+ km²).
    """
    
    def __init__(self, spark_master: str = "local[*]"):
        self.spark = self._init_spark_session(spark_master)
        SedonaRegistrator.registerAll(self.spark)
        
    def _init_spark_session(self, master: str) -> SparkSession:
        """Initialize Spark session with Sedona extensions."""
        return SparkSession.builder \
            .master(master) \
            .appName("MultiverseDive-Sedona") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .getOrCreate()
    
    def execute_algorithm(self, algorithm, stac_items: List[Dict], aoi_wkt: str, **params):
        """
        Execute algorithm using Sedona's distributed raster processing.
        
        Workflow:
        1. Load STAC items as Sedona RasterRDD
        2. Apply spatial partitioning (quadtree)
        3. Distribute algorithm execution across Spark workers
        4. Use Sedona's map algebra for raster operations
        5. Collect results
        """
        
        # Create Sedona RasterRDD from STAC items
        raster_rdd = self._create_raster_rdd(stac_items)
        
        # Apply spatial partitioning for efficient distributed processing
        raster_rdd.spatialPartitioning(GridType.QUADTREE)
        raster_rdd.buildIndex(IndexType.RTREE, True)
        
        # Perform spatial query to filter to AOI
        from shapely.wkt import loads as wkt_loads
        aoi_geom = wkt_loads(aoi_wkt)
        filtered_rdd = RangeQuery.SpatialRangeQuery(raster_rdd, aoi_geom, True, True)
        
        # Apply algorithm using Sedona's map algebra
        if algorithm.name == 'ndwi_optical':
            result_rdd = self._compute_ndwi_sedona(filtered_rdd, params)
        elif algorithm.name == 'threshold_sar':
            result_rdd = self._compute_sar_threshold_sedona(filtered_rdd, params)
        else:
            # Generic algorithm execution
            result_rdd = filtered_rdd.map(lambda tile: algorithm.process_tile(tile.raster_data, **params))
        
        # Collect and mosaic results
        final_result = self._mosaic_raster_rdd(result_rdd)
        return final_result
    
    def _compute_ndwi_sedona(self, raster_rdd, params: Dict):
        """
        NDWI calculation using Sedona's map algebra.
        
        NDWI = (Green - NIR) / (Green + NIR)
        """
        from sedona.raster import MapAlgebra
        
        # Extract green and NIR bands
        green_rdd = raster_rdd.map(lambda r: r.getBand(params.get('green_band', 3)))
        nir_rdd = raster_rdd.map(lambda r: r.getBand(params.get('nir_band', 8)))
        
        # Compute NDWI using Sedona map algebra
        ndwi_rdd = MapAlgebra.apply_algebra(
            green_rdd, 
            nir_rdd,
            lambda g, n: (g - n) / (g + n + 1e-10)
        )
        
        return ndwi_rdd
```

### Component 5: Algorithm Adapters

**Purpose:** Add distributed processing support to existing baseline algorithms

**Location:** Update existing algorithm files

#### Algorithm Interface Extension

```python
# core/analysis/library/baseline/flood/ndwi_optical.py

class NDWIOpticalFloodDetection:
    """
    NDWI-based flood detection with distributed processing support.
    """
    
    # Existing serial implementation
    def process(self, green: np.ndarray, nir: np.ndarray, **params) -> Dict:
        """Original serial implementation - unchanged."""
        pass
    
    # NEW: Dask-aware tile processing
    def process_tile(self, data: np.ndarray, **params) -> np.ndarray:
        """
        Process single tile for Dask parallelization.
        
        Args:
            data: 3D array (bands, height, width) for single tile
            
        Returns:
            2D array: NDWI values for this tile
        """
        green_idx = params.get('green_band_idx', 1)
        nir_idx = params.get('nir_band_idx', 3)
        
        green = data[green_idx, :, :]
        nir = data[nir_idx, :, :]
        
        # NDWI calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir + 1e-10)
        
        return ndwi
    
    # NEW: Sedona map algebra support
    @staticmethod
    def sedona_map_algebra(green_band, nir_band):
        """
        Sedona-compatible map algebra expression for NDWI.
        
        Used by SedonaExecutor for distributed raster processing.
        """
        return "(B1 - B2) / (B1 + B2 + 0.0000000001)"
```

### Deployment Configuration

#### Execution Profiles

```yaml
# config/execution_profiles.yaml

profiles:
  laptop:
    resource_limits:
      max_memory_gb: 4
      max_workers: 4
      max_tiles_parallel: 50
      tile_size_px: 256
    backend:
      type: "dask_local"
      dask_config:
        n_workers: 4
        threads_per_worker: 1
        memory_limit: "1GB"
    routing_rules:
      tile_count_thresholds:
        serial_max: 50
        dask_local_max: 500
  
  workstation:
    resource_limits:
      max_memory_gb: 16
      max_workers: 8
      max_tiles_parallel: 200
      tile_size_px: 512
    backend:
      type: "dask_local"
      dask_config:
        n_workers: 8
        threads_per_worker: 2
        memory_limit: "2GB"
    routing_rules:
      tile_count_thresholds:
        serial_max: 100
        dask_local_max: 1000
  
  cloud_dask:
    resource_limits:
      max_memory_gb: 128
      max_workers: 32
      enable_distributed: true
    backend:
      type: "dask_distributed"
      dask_config:
        scheduler_address: "tcp://dask-scheduler:8786"
        n_workers: 32
        threads_per_worker: 2
        memory_limit: "4GB"
  
  cloud_sedona:
    resource_limits:
      max_memory_gb: 512
      enable_distributed: true
    backend:
      type: "sedona_spark"
      sedona_config:
        spark_master: "spark://spark-master:7077"
        executor_memory: "8g"
        executor_instances: 64
        spatial_partitioning: "quadtree"
        raster_partitions: 256
```

### Testing Strategy

#### Unit Tests

```python
# tests/test_distributed_execution.py

import pytest
from core.analysis.execution.router import ExecutionRouter, ExecutionContext, ExecutionBackend
from core.data.ingestion.virtual_index import VirtualRasterIndex

def test_router_selects_serial_for_small_job():
    """Router should select serial execution for < 100 tiles."""
    router = ExecutionRouter()
    context = ExecutionContext(
        tile_count=50,
        estimated_memory_gb=2.0,
        aoi_size_km2=100,
        algorithm_complexity="simple",
        available_resources={'memory_gb': 4, 'cpu_cores': 4},
        cluster_available=False
    )
    
    backend = router.route(context)
    assert backend == ExecutionBackend.SERIAL

def test_router_selects_dask_local_for_medium_job():
    """Router should select Dask local for 100-1000 tiles."""
    router = ExecutionRouter()
    context = ExecutionContext(
        tile_count=500,
        estimated_memory_gb=3.5,
        aoi_size_km2=1000,
        algorithm_complexity="medium",
        available_resources={'memory_gb': 16, 'cpu_cores': 8},
        cluster_available=False
    )
    
    backend = router.route(context)
    assert backend == ExecutionBackend.DASK_LOCAL

def test_virtual_index_builds_from_stac():
    """Virtual index should parse STAC items without downloading."""
    index = VirtualRasterIndex()
    
    stac_items = load_fixture('stac_sentinel2_items.json')
    index.add_from_stac_items(stac_items)
    
    assert len(index.tiles) == 30
    assert index.spatial_index.get_bounds() is not None

def test_virtual_index_aoi_query():
    """Virtual index should quickly query tiles for AOI."""
    index = VirtualRasterIndex()
    index.add_from_stac_items(load_fixture('stac_items.json'))
    
    aoi_bbox = (-80.5, 25.5, -80.0, 26.0)  # Miami
    tiles = index.query_aoi(aoi_bbox)
    
    assert len(tiles) > 0
    assert all(bbox_intersects(t.bbox, aoi_bbox) for t in tiles)

def test_dask_executor_parallelizes():
    """Dask executor should use multiple workers."""
    from core.analysis.execution.dask_tiled import DaskLocalExecutor
    
    executor = DaskLocalExecutor(n_workers=4)
    # Test execution with mock algorithm
    # Verify worker utilization
    pass
```

#### Integration Tests

```python
# tests/test_distributed_integration.py

@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_dask_execution():
    """
    Full end-to-end test: STAC discovery → VRT → Dask execution → Result.
    """
    # 1. Discover data via STAC
    from core.data.broker import DataBroker
    broker = DataBroker()
    discovery_results = broker.discover(
        spatial_bbox=(-80.5, 25.5, -80.0, 26.0),
        temporal=('2024-09-15', '2024-09-20'),
        event_class='flood.coastal'
    )
    
    # 2. Build virtual index
    from core.data.ingestion.virtual_index import VirtualRasterIndex
    vindex = VirtualRasterIndex()
    vindex.add_from_discovery_results(discovery_results)
    
    # 3. Route execution
    from core.analysis.execution.router import ExecutionRouter, ExecutionContext
    router = ExecutionRouter(default_profile='workstation')
    context = ExecutionContext(
        tile_count=len(vindex.tiles),
        estimated_memory_gb=vindex.estimate_memory_requirements(aoi_bbox)['peak_memory_gb'],
        aoi_size_km2=100,
        algorithm_complexity='simple',
        available_resources={'memory_gb': 16, 'cpu_cores': 8},
        cluster_available=False
    )
    backend = router.route(context)
    executor = router.get_executor(backend)
    
    # 4. Execute algorithm
    from core.analysis.library.baseline.flood.ndwi_optical import NDWIOpticalFloodDetection
    algorithm = NDWIOpticalFloodDetection()
    result = executor.execute_algorithm(
        algorithm=algorithm,
        virtual_index=vindex,
        aoi_bbox=(-80.5, 25.5, -80.0, 26.0),
        threshold=0.3
    )
    
    # 5. Verify result
    assert result is not None
    assert result.shape[0] > 0
    assert 0 <= result.max() <= 1.0  # NDWI range check

@pytest.mark.integration
@pytest.mark.requires_spark
def test_sedona_execution():
    """Test Apache Sedona execution on Spark cluster."""
    # Requires Spark cluster running
    pytest.skip("Requires Spark cluster")
    # Full Sedona test implementation
    pass
```

### Performance Targets

| Metric | Current (Serial) | Target (Dask Local) | Target (Sedona) |
|--------|------------------|---------------------|-----------------|
| 100km² Sentinel-2 | 20-30 min | <5 min | <2 min |
| 1000km² Sentinel-2 | 3-5 hours | 30-45 min | <10 min |
| 10,000km² (state) | Not feasible | 5-8 hours | <1 hour |
| 100,000km² (region) | Not feasible | Not feasible | <4 hours |
| Memory (laptop) | 2-6GB peak | <4GB peak | N/A |
| CPU utilization | 12-25% | 80-95% | N/A |
| Download size (100km²) | 2-5GB | <100MB | <100MB |

### Migration Path

**Phase 1: Virtual Index & Dask Local** (Weeks 1-3)
- Implement VirtualRasterIndex
- Implement ExecutionRouter
- Implement DaskLocalExecutor
- Add process_tile() to baseline algorithms
- Integration tests

**Phase 2: Apache Sedona** (Weeks 4-7)  
- Implement SedonaExecutor
- Port algorithms to Sedona map algebra
- Spark cluster deployment configs
- Performance benchmarking

**Phase 3: Optimization** (Weeks 8-9)
- Profile and optimize tile sizes
- Implement adaptive chunking
- Add caching layer for frequently accessed tiles
- Cost optimization for cloud execution

---

**Last Updated:** 2026-01-11
**Implementation Status:** Specification phase
**Next Steps:** Begin Phase 1 implementation after P0 bug fixes complete
