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

### Installation

```bash
# Clone and install
git clone https://github.com/gpriceless/multiverse_dive.git
cd multiverse_dive
pip install -e .

# Install optional dependencies for full functionality
pip install rasterio geopandas xarray pyproj shapely

# Run tests
./run_tests.py                    # All tests
./run_tests.py flood              # Flood tests only
./run_tests.py --list             # Show all categories
```

### Command Line Interface (mdive)

The `mdive` CLI provides full access to all platform capabilities:

```bash
# Get help
mdive --help
mdive info                        # Show system info and configuration

# Discover available satellite data
mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood
mdive discover --bbox -80.5,25.5,-80.0,26.0 --event wildfire --format json

# Download and prepare data
mdive ingest --area miami.geojson --source sentinel1 --output ./data/
mdive ingest --area california.geojson --source sentinel2,landsat8 --output ./data/

# Run analysis with specific algorithm
mdive analyze --input ./data/ --algorithm sar_threshold --output ./results/
mdive analyze --input ./data/ --algorithm ndwi --confidence 0.8 --output ./results/

# Validate results
mdive validate --input ./results/ --checks sanity,cross_validation
mdive validate --input ./results/ --reference ground_truth.geojson

# Export products
mdive export --input ./results/ --format geotiff,geojson,pdf --output ./products/

# Full end-to-end pipeline
mdive run --area miami.geojson --event flood --profile laptop --output ./products/
mdive run --area california.geojson --event wildfire --profile workstation --output ./products/

# Monitor and resume
mdive status --workdir ./products/
mdive resume --workdir ./products/
```

**Execution Profiles** adapt to your hardware:

| Profile | Memory | Workers | Tile Size | Use Case |
|---------|--------|---------|-----------|----------|
| `edge` | 1 GB | 1 | 128px | Raspberry Pi, embedded |
| `laptop` | 2 GB | 2 | 256px | Local development |
| `workstation` | 8 GB | 4 | 512px | Desktop processing |
| `cloud` | 32 GB | 16 | 1024px | Server deployment |

### REST API

Start the API server for programmatic access:

```bash
# Development server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API documentation available at:
# http://localhost:8000/api/docs     (Swagger UI)
# http://localhost:8000/api/redoc    (ReDoc)
```

**API Endpoints:**

```bash
# Submit an event for processing
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d @examples/flood_event.yaml

# Check processing status
curl http://localhost:8000/api/v1/events/{event_id}/status

# Download products
curl http://localhost:8000/api/v1/events/{event_id}/products/flood_extent.geojson

# Browse data catalog
curl http://localhost:8000/api/v1/catalog/algorithms
curl http://localhost:8000/api/v1/catalog/providers

# Health check
curl http://localhost:8000/api/v1/health
```

### Using Modules Independently

Each core module can be used standalone in your own Python code:

```python
# Intent Resolution - classify event types from natural language
from core.intent import IntentResolver

resolver = IntentResolver()
result = resolver.resolve("flooding in coastal Miami after Hurricane Milton")
print(result.resolved_class)  # "flood.coastal"
print(result.confidence)      # 0.92

# Data Discovery - find available satellite imagery
from core.data.broker import DataBroker

broker = DataBroker()
datasets = await broker.discover(
    spatial={"type": "Polygon", "coordinates": [...]},
    temporal={"start": "2024-09-15", "end": "2024-09-20"},
    event_class="flood.coastal"
)

# Algorithm Execution - run specific detection algorithms
from core.analysis.library.baseline.flood import ThresholdSARAlgorithm

algorithm = ThresholdSARAlgorithm()
result = algorithm.execute(sar_data, pixel_size_m=10.0)
print(f"Flood area: {result.statistics['flood_area_ha']} hectares")

# Quality Control - validate results
from core.quality import SanityChecker, CrossValidator

checker = SanityChecker()
issues = checker.check(result.flood_extent, event_class="flood.coastal")

validator = CrossValidator()
consensus = validator.validate([result1, result2, result3])
```

### Docker Deployment

```bash
# Build images
docker build -f docker/Dockerfile.api -t mdive-api .
docker build -f docker/Dockerfile.cli -t mdive-cli .

# Run with Docker Compose
docker-compose up -d

# Or run CLI in container
docker run -v $(pwd)/data:/data mdive-cli run \
  --area /data/area.geojson --event flood --output /data/output/
```

### Example Event Specifications

See the `examples/` directory for sample event configurations:

- `flood_event.yaml` - Coastal flooding analysis
- `wildfire_event.yaml` - Burn severity mapping
- `storm_event.yaml` - Hurricane damage assessment
- `datasource_sentinel1.yaml` - SAR data source configuration
- `pipeline_flood_sar.yaml` - Custom pipeline definition


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
