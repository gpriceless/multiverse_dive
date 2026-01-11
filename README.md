# Multiverse Dive

**Geospatial event intelligence platform that converts (area, time window, event type) into decision products.**

## What This Does

When a flood, wildfire, or storm happens, decision-makers need answers fast: *Where exactly is affected? How severe? What's the uncertainty?* Getting those answers typically requires specialists to manually find satellite data, run analyses, validate results, and produce reports—a process that takes days.

Multiverse Dive automates this entire pipeline. You provide:
- A geographic area (polygon or bounding box)
- A time window (start/end dates)
- An event type ("coastal flood", "forest fire", "hurricane damage")

The system autonomously:
1. **Discovers** the best available satellite imagery (Sentinel-1 SAR, Sentinel-2 optical, Landsat, MODIS, DEMs, weather data)
2. **Selects** appropriate algorithms based on what data is available and cloud conditions
3. **Processes** multi-sensor fusion with proper atmospheric and terrain corrections
4. **Validates** results through cross-sensor comparison and plausibility checks
5. **Produces** GeoTIFF maps, GeoJSON vectors, uncertainty layers, and human-readable reports

All with complete provenance—every output traces back to source data and processing steps.

## Why This Matters

**Speed**: Hours instead of days for initial situational awareness.

**Consistency**: Same methodology applied regardless of who requests the analysis.

**Transparency**: Full audit trail from raw data to final product.

**Scalability**: Handle multiple concurrent events without bottlenecks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Event Request                           │
│              (area + time window + event type)                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Intent Resolution                          │
│         NLP + taxonomy lookup → structured event class          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Data Discovery                            │
│    Query STAC catalogs, OGC services, provider APIs             │
│    Rank by: availability, cloud cover, resolution, cost         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Algorithm Selection                         │
│    Match algorithms to available data + event type              │
│    SAR threshold, NDWI, change detection, HAND model, etc.      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Execution                           │
│    Ingest → normalize → fuse → analyze → validate               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Quality Control                            │
│    Sanity checks, cross-validation, uncertainty quantification  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Products                                 │
│    GeoTIFF, GeoJSON, PDF report, provenance record              │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
multiverse_dive/
├── core/                    # Core processing logic
│   ├── intent/              # Event type classification and resolution
│   ├── data/                # Data discovery, providers, and ingestion
│   └── analysis/            # Algorithm library and pipeline assembly
├── openspec/                # Specification layer
│   ├── schemas/             # JSON Schema definitions
│   └── definitions/         # YAML event classes, algorithms, data sources
├── agents/                  # Autonomous agent implementations
├── api/                     # FastAPI REST interface
├── tests/                   # Test suites
└── examples/                # Example event specifications
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run tests
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Example: process a flood event (API not yet implemented)
# curl -X POST http://localhost:8000/events \
#   -H "Content-Type: application/yaml" \
#   --data-binary @examples/flood_event.yaml
```

## Current Status

| Group | Status | Description |
|-------|--------|-------------|
| **A** | ✅ Complete | Foundation schemas, project structure |
| **B** | ✅ Complete | Schema validation, examples |
| **C** | ✅ Complete | Intent resolution, NLP classification |
| **D** | ✅ Complete | Data discovery, provider implementations |
| **E** | ✅ Complete | Algorithm library (flood, wildfire, storm) |
| **F** | ✅ Complete | Intelligent selection (constraints, ranking, sensor strategy) |
| **G** | ✅ Complete | Ingestion & normalization pipeline |
| **H** | ✅ Complete | Fusion & analysis engine |
| **I** | ✅ Complete | Quality control (sanity, validation, uncertainty, reporting) |
| **J** | 🔜 Next | Agent orchestration |
| **K** | 📋 Planned | API gateway & deployment |
| **L** | 📋 Planned | Lightweight CLI execution |
| **M-N** | 📋 Planned | Resilience, containerization |

**Test Coverage**: 2,385 tests passing (44 skipped for optional deps, 4 minor failures in new execution code)

See [ROADMAP.md](ROADMAP.md) for detailed implementation progress.

## Key Documents

| Document | Purpose |
|----------|---------|
| [OPENSPEC.md](OPENSPEC.md) | Complete system design specification |
| [ROADMAP.md](ROADMAP.md) | Implementation roadmap with parallel work groups |
| [FIXES.md](FIXES.md) | Known bugs with exact code fixes |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI agents working on this codebase |

## Supported Hazard Types

| Hazard | Algorithms | Status |
|--------|------------|--------|
| **Flood** | SAR threshold, NDWI optical, change detection, HAND model | Implemented |
| **Wildfire** | dNBR burn severity, thermal anomaly, burned area classifier | Implemented |
| **Storm** | Wind damage detection, structural damage assessment | Implemented |

## Data Sources

The platform queries multiple data sources with preference for open/free data:

- **Optical**: Sentinel-2, Landsat-8/9, MODIS
- **SAR**: Sentinel-1 (cloud-penetrating radar)
- **DEM**: Copernicus DEM, SRTM, FABDEM
- **Weather**: ERA5, GFS, ECMWF
- **Ancillary**: OpenStreetMap, World Settlement Footprint, land cover

## License

TBD
