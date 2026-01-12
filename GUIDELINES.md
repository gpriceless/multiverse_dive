# Multiverse Dive - Complete Guidelines

This document covers every process available in the Multiverse Dive platform: from local development to cloud deployment, from algorithm selection to end-to-end workflows.

## Table of Contents

1. [Running Locally](#1-running-locally)
2. [Running on a Server](#2-running-on-a-server)
3. [Deployment Options](#3-deployment-options)
4. [Available Models & Algorithms](#4-available-models--algorithms)
5. [Processing Tracks & Pipelines](#5-processing-tracks--pipelines)
6. [Module Interconnections](#6-module-interconnections)
7. [Generating Visualizations](#7-generating-visualizations)
8. [Creating Analytics](#8-creating-analytics)
9. [End-to-End Workflow Examples](#9-end-to-end-workflow-examples)
10. [Quick Reference](#10-quick-reference)

---

## 1. Running Locally

### 1.1 Installation

```bash
# Clone the repository
git clone https://github.com/gpriceless/multiverse_dive.git
cd multiverse_dive

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .

# Install geospatial dependencies
pip install rasterio geopandas xarray pyproj shapely
```

### 1.2 Running Tests

```bash
# All tests
./run_tests.py

# By hazard type
./run_tests.py flood           # Flood detection tests
./run_tests.py wildfire        # Wildfire tests
./run_tests.py storm           # Storm damage tests

# By component
./run_tests.py schemas         # Schema validation
./run_tests.py intent          # Intent resolution
./run_tests.py providers       # Data providers
./run_tests.py algorithms      # All algorithm tests

# By algorithm
./run_tests.py --algorithm sar      # SAR threshold
./run_tests.py --algorithm ndwi     # NDWI optical
./run_tests.py --algorithm hand     # HAND model
./run_tests.py --algorithm dnbr     # Burn severity
./run_tests.py --algorithm thermal  # Thermal anomaly
./run_tests.py --algorithm wind     # Wind damage

# Test types
./run_tests.py --quick         # Fast tests only
./run_tests.py slow            # Slow/comprehensive tests
./run_tests.py integration     # Integration tests

# List all categories
./run_tests.py --list
```

### 1.3 CLI Usage (mdive)

The `mdive` command provides access to all platform capabilities:

```bash
# Get help
mdive --help
mdive info                    # Show system info

# Data discovery
mdive discover --area area.geojson --start 2024-09-15 --end 2024-09-20 --event flood
mdive discover --bbox -80.5,25.5,-80.0,26.0 --event wildfire --format json

# Data ingestion
mdive ingest --area area.geojson --source sentinel1 --output ./data/
mdive ingest --area area.geojson --source sentinel2,landsat8 --output ./data/

# Analysis
mdive analyze --input ./data/ --algorithm sar_threshold --output ./results/
mdive analyze --input ./data/ --algorithm ndwi --confidence 0.8 --output ./results/

# Validation
mdive validate --input ./results/ --checks sanity,cross_validation
mdive validate --input ./results/ --reference ground_truth.geojson

# Export
mdive export --input ./results/ --format geotiff,geojson,pdf --output ./products/

# Full pipeline
mdive run --area area.geojson --event flood --profile laptop --output ./products/
mdive run --area area.geojson --event wildfire --profile workstation --output ./products/

# Monitoring
mdive status --workdir ./products/
mdive resume --workdir ./products/
```

### 1.4 Execution Profiles

Profiles adapt processing to your hardware:

| Profile | Memory | Workers | Tile Size | Use Case |
|---------|--------|---------|-----------|----------|
| `edge` | 1 GB | 1 | 128px | Raspberry Pi, embedded devices |
| `laptop` | 2 GB | 2 | 256px | Local development |
| `workstation` | 8 GB | 4 | 512px | Desktop processing |
| `cloud` | 32 GB | 16 | 1024px | Server deployment |

Usage:
```bash
mdive run --area miami.geojson --event flood --profile laptop
mdive run --area california.geojson --event wildfire --profile cloud
```

### 1.5 Configuration Files

Create a `.mdive.yaml` in your project or home directory:

```yaml
# .mdive.yaml - User configuration
profile: laptop
cache_dir: ~/.mdive/cache
log_level: info

providers:
  sentinel1:
    api_key: ${COPERNICUS_API_KEY}
  sentinel2:
    api_key: ${COPERNICUS_API_KEY}
  landsat:
    use_usgs: true

output:
  formats:
    - geotiff
    - geojson
  include_provenance: true
```

---

## 2. Running on a Server

### 2.1 REST API Server

```bash
# Development server with hot reload
python -m api.main

# Production with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn (recommended for production)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

API documentation available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### 2.2 API Endpoints

```bash
# Submit event for processing
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {"class": "flood.coastal"},
    "spatial": {"type": "Polygon", "coordinates": [[[-80.3, 25.7], [-80.1, 25.7], [-80.1, 25.9], [-80.3, 25.9], [-80.3, 25.7]]]},
    "temporal": {"start": "2024-09-15T00:00:00Z", "end": "2024-09-20T23:59:59Z"}
  }'

# Check status
curl http://localhost:8000/api/v1/events/{event_id}/status

# Get products
curl http://localhost:8000/api/v1/events/{event_id}/products
curl http://localhost:8000/api/v1/events/{event_id}/products/flood_extent.geojson --output flood.geojson

# Browse catalog
curl http://localhost:8000/api/v1/catalog/algorithms
curl http://localhost:8000/api/v1/catalog/providers
curl http://localhost:8000/api/v1/catalog/event-classes

# Health check
curl http://localhost:8000/api/v1/health
```

### 2.3 Docker Compose (Full Stack)

```bash
# Start full stack (API + Workers + Redis + PostgreSQL)
docker compose up -d

# Start specific services
docker compose up -d api worker

# View logs
docker compose logs -f api
docker compose logs -f worker

# Stop all services
docker compose down

# With custom configuration
POSTGRES_PASSWORD=secure_password docker compose up -d
```

### 2.4 Docker Compose (Minimal/CLI)

For CLI-only operations without external services:

```bash
# Run CLI commands
docker compose -f docker-compose.minimal.yml run --rm cli mdive --help
docker compose -f docker-compose.minimal.yml run --rm cli mdive analyze --help

# Run full pipeline
docker compose -f docker-compose.minimal.yml run --rm cli mdive run \
  --area /app/examples/flood_event.yaml \
  --output /app/output/
```

---

## 3. Deployment Options

### 3.1 Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f deploy/kubernetes/namespace.yaml

# Apply configuration
kubectl apply -f deploy/kubernetes/configmaps/
kubectl apply -f deploy/kubernetes/secrets/

# Deploy storage
kubectl apply -f deploy/kubernetes/persistentvolumes/

# Deploy services
kubectl apply -f deploy/kubernetes/deployments/api.yaml
kubectl apply -f deploy/kubernetes/deployments/worker.yaml
kubectl apply -f deploy/kubernetes/services/

# Configure ingress
kubectl apply -f deploy/kubernetes/ingress.yaml

# Enable autoscaling
kubectl apply -f deploy/kubernetes/hpa.yaml
```

Check deployment:
```bash
kubectl get pods -n multiverse-dive
kubectl get services -n multiverse-dive
kubectl logs -f deployment/api -n multiverse-dive
```

### 3.2 AWS Deployment

**ECS (Elastic Container Service)**:
```bash
# Deploy using AWS CLI
aws ecs create-cluster --cluster-name mdive-cluster
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs/task-definition.json
aws ecs create-service --cli-input-json file://deploy/aws/ecs/service.json
```

**Lambda (Serverless)**:
```bash
# Deploy Lambda function
cd deploy/aws/lambda
sam build
sam deploy --guided
```

**AWS Batch (Large-scale processing)**:
```bash
# Submit batch job
aws batch submit-job \
  --job-name flood-analysis \
  --job-queue mdive-queue \
  --job-definition mdive-analysis \
  --container-overrides '{
    "command": ["mdive", "run", "--area", "s3://bucket/area.geojson", "--event", "flood"]
  }'
```

### 3.3 Azure Deployment

**Azure Kubernetes Service (AKS)**:
```bash
# Deploy to AKS
kubectl apply -f deploy/azure/aks/deployment.yaml
```

**Azure Container Instances**:
```bash
az container create \
  --resource-group mdive-rg \
  --name mdive-api \
  --image multiverse-dive-api:latest \
  --ports 8000 \
  --environment-variables LOG_LEVEL=info
```

### 3.4 Google Cloud Deployment

**Cloud Run**:
```bash
gcloud run deploy mdive-api \
  --image gcr.io/your-project/multiverse-dive-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**GKE (Google Kubernetes Engine)**:
```bash
kubectl apply -f deploy/gcp/kubernetes/deployment.yaml
```

### 3.5 On-Premises Deployment

Using Ansible:
```bash
# Standalone deployment
ansible-playbook -i inventory deploy/on-prem/standalone.yaml

# Cluster deployment
ansible-playbook -i inventory deploy/on-prem/cluster.yaml
```

### 3.6 Edge Deployment

For resource-constrained devices (Raspberry Pi, NVIDIA Jetson):
```bash
# Build edge image
docker build -f deploy/edge/Dockerfile.arm64 -t mdive-edge .

# Run on edge device
docker run -v /data:/data mdive-edge mdive run \
  --area /data/area.geojson \
  --event flood \
  --profile edge
```

---

## 4. Available Models & Algorithms

### 4.1 Flood Detection Algorithms

| Algorithm | ID | Data Required | Accuracy | Best For |
|-----------|----|--------------|---------|---------|
| **SAR Threshold** | `flood.baseline.threshold_sar` | Sentinel-1 SAR | 75-90% | Cloud cover, all-weather |
| **NDWI Optical** | `flood.baseline.ndwi_optical` | Sentinel-2, Landsat | 80-92% | Clear conditions |
| **Change Detection** | `flood.baseline.change_detection` | Pre/post imagery | 78-88% | Event comparison |
| **HAND Model** | `flood.baseline.hand_model` | DEM + flood extent | 70-85% | Flood depth estimation |
| **UNet Segmentation** | `flood.advanced.unet_segmentation` | Multi-sensor | 85-95% | GPU available (experimental) |
| **Ensemble Fusion** | `flood.advanced.ensemble_fusion` | Multiple outputs | 88-95% | High confidence needed |

**SAR Threshold** (`core/analysis/library/baseline/flood/threshold_sar.py`):
```python
from core.analysis.library.baseline.flood import ThresholdSARAlgorithm

algo = ThresholdSARAlgorithm()
result = algo.execute(
    sar_data,
    pixel_size_m=10.0,
    threshold_db=-15.0,  # Default threshold
    min_area_pixels=25
)
print(f"Flood area: {result.statistics['flood_area_ha']} hectares")
```

**NDWI Optical** (`core/analysis/library/baseline/flood/ndwi_optical.py`):
```python
from core.analysis.library.baseline.flood import NDWIOpticalAlgorithm

algo = NDWIOpticalAlgorithm()
result = algo.execute(
    optical_data,
    green_band="B03",
    nir_band="B08",
    threshold=0.3
)
```

### 4.2 Wildfire Algorithms

| Algorithm | ID | Data Required | Use Case |
|-----------|----|--------------| ---------|
| **Thermal Anomaly** | `wildfire.baseline.thermal_anomaly` | MODIS thermal | Active fire detection |
| **dNBR Burn Severity** | `wildfire.baseline.nbr_differenced` | Pre/post optical | Post-fire severity |
| **Burned Area Classifier** | `wildfire.baseline.ba_classifier` | Multi-temporal optical | Burned/unburned mapping |

**dNBR Burn Severity** (`core/analysis/library/baseline/wildfire/nbr_differenced.py`):
```python
from core.analysis.library.baseline.wildfire import NBRDifferencedAlgorithm

algo = NBRDifferencedAlgorithm()
result = algo.execute(
    pre_image=pre_fire_data,
    post_image=post_fire_data,
    nir_band="B08",
    swir_band="B12"
)
# Severity classes: unburned, low, moderate-low, moderate-high, high
```

### 4.3 Storm Damage Algorithms

| Algorithm | ID | Data Required | Use Case |
|-----------|----|--------------| ---------|
| **Wind Damage** | `storm.baseline.wind_damage` | Optical/SAR + wind data | Tree/vegetation damage |
| **Structural Damage** | `storm.baseline.structural_damage` | High-res optical | Building damage assessment |

### 4.4 Algorithm Selection

The system automatically selects algorithms based on available data:

```python
from core.analysis.selection import DeterministicSelector

selector = DeterministicSelector()
algorithms = selector.select(
    event_class="flood.coastal",
    available_data=["sentinel1_sar", "copernicus_dem"],
    constraints={"max_cloud_cover": 0.4}
)
# Returns: ["flood.baseline.threshold_sar", "flood.baseline.hand_model"]
```

---

## 5. Processing Tracks & Pipelines

### 5.1 Event Types (Intent Classes)

The system supports hierarchical event classification:

**Flood Events:**
- `flood.riverine` - River overflow flooding
- `flood.coastal.storm_surge` - Coastal storm surge
- `flood.coastal.tsunami` - Tsunami inundation
- `flood.pluvial` - Urban/rainfall flooding
- `flood.flash` - Flash flooding

**Wildfire Events:**
- `wildfire.forest` - Forest fires
- `wildfire.grassland` - Grassland fires
- `wildfire.urban` - Urban interface fires

**Storm Events:**
- `storm.tropical` - Hurricane/typhoon damage
- `storm.severe_convective` - Tornado damage
- `storm.winter` - Blizzard/ice damage

### 5.2 Custom Pipeline Definition

Create custom pipelines in YAML:

```yaml
# my_flood_pipeline.yaml
id: custom_flood_pipeline
name: Custom SAR Flood Mapping
version: "1.0.0"

applicable_classes:
  - flood.riverine
  - flood.coastal

inputs:
  - name: sar_pre_event
    type: raster
    source: sentinel1_grd
    temporal_role: pre_event
    required: true

  - name: sar_post_event
    type: raster
    source: sentinel1_grd
    temporal_role: post_event
    required: true

  - name: dem
    type: raster
    source: copernicus_dem
    required: true

steps:
  - id: normalize_sar_pre
    processor: ingestion.normalize_sar
    inputs: [sar_pre_event]
    parameters:
      calibration: sigma0
      output_db: true

  - id: normalize_sar_post
    processor: ingestion.normalize_sar
    inputs: [sar_post_event]
    parameters:
      calibration: sigma0
      output_db: true

  - id: speckle_filter
    processor: sar.speckle_filter
    inputs: [normalize_sar_post.output]
    parameters:
      method: lee
      window_size: 7

  - id: change_detection
    processor: flood.sar_change_detection
    inputs:
      - normalize_sar_pre.output
      - speckle_filter.output
    parameters:
      threshold_db: -3.0

  - id: terrain_mask
    processor: flood.terrain_mask
    inputs:
      - change_detection.output
      - dem
    parameters:
      slope_threshold_degrees: 15

outputs:
  - name: flood_extent
    type: raster
    format: cog
    step: terrain_mask

quality_checks:
  - type: spatial_coherence
  - type: temporal_consistency
```

Run custom pipeline:
```bash
mdive run --pipeline my_flood_pipeline.yaml --area area.geojson --output ./products/
```

### 5.3 Data Sources

| Source | Type | Provider | Availability |
|--------|------|----------|--------------|
| Sentinel-1 | SAR | Copernicus | Free, global |
| Sentinel-2 | Optical | Copernicus | Free, global |
| Landsat 8/9 | Optical | USGS | Free, global |
| MODIS | Optical/Thermal | NASA | Free, global |
| Copernicus DEM | Elevation | Copernicus | Free, global |
| SRTM | Elevation | NASA | Free, 60°N-56°S |
| FABDEM | Elevation | Bristol/CEDA | Free, academic |
| ERA5 | Weather | ECMWF | Free, global |
| GFS | Weather | NOAA | Free, global |
| OpenStreetMap | Vector | OSM | Free, global |

---

## 6. Module Interconnections

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                                 │
├─────────────┬─────────────┬─────────────────────────────────────────┤
│   CLI       │   REST API  │   Python SDK                            │
│  (mdive)    │  (FastAPI)  │   (import core.*)                       │
└──────┬──────┴──────┬──────┴─────────────────────────────────────────┘
       │             │
       ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATION                             │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│ Orchestrator│  Discovery  │  Pipeline   │   Quality   │  Reporting  │
│   Agent     │   Agent     │   Agent     │   Agent     │   Agent     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CORE PROCESSING LAYER                           │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   Intent    │    Data     │  Analysis   │   Quality   │  Resilience │
│  Resolution │  Discovery  │  Execution  │   Control   │  & Fallback │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │
│ │Resolver │ │ │ STAC    │ │ │Algorithm│ │ │ Sanity  │ │ │ Degrade │ │
│ │Classifier│ │ │ WMS/WCS │ │ │ Library │ │ │ Checks  │ │ │ Mode    │ │
│ │Registry │ │ │Providers│ │ │ Selector│ │ │CrossVal │ │ │Fallbacks│ │
│ └─────────┘ │ └─────────┘ │ │ Assembly│ │ │Uncertain│ │ └─────────┘ │
│             │             │ │ Runner  │ │ └─────────┘ │             │
│             │             │ └─────────┘ │             │             │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                      │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│  Ingestion  │   Fusion    │   Cache     │  Storage    │ Provenance  │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │
│ │Normalize│ │ │Alignment│ │ │ R-tree  │ │ │  Local  │ │ │ Lineage │ │
│ │Validate │ │ │Correct  │ │ │ Index   │ │ │  Cloud  │ │ │Tracking │ │
│ │ Enrich  │ │ │ Merge   │ │ │  TTL    │ │ │   COG   │ │ └─────────┘ │
│ │ Persist │ │ └─────────┘ │ └─────────┘ │ │  Zarr   │ │             │
│ └─────────┘ │             │             │ └─────────┘ │             │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### 6.2 Using Modules Independently

**Intent Resolution:**
```python
from core.intent import IntentResolver

resolver = IntentResolver()
result = resolver.resolve("flooding in coastal Miami after Hurricane Milton")
print(result.resolved_class)  # "flood.coastal.storm_surge"
print(result.confidence)       # 0.92
print(result.reasoning)        # Explanation of classification
```

**Data Discovery:**
```python
from core.data.broker import DataBroker

broker = DataBroker()
datasets = await broker.discover(
    spatial={"type": "Polygon", "coordinates": [...]},
    temporal={"start": "2024-09-15", "end": "2024-09-20"},
    event_class="flood.coastal",
    constraints={"max_cloud_cover": 0.4}
)

for ds in datasets:
    print(f"{ds.provider}: {ds.id}, resolution: {ds.resolution_m}m")
```

**Data Ingestion:**
```python
from core.data.ingestion import DataIngester

ingester = DataIngester()
normalized_data = await ingester.ingest(
    source="sentinel1",
    dataset_id="S1A_IW_GRDH_1SDV_20240917...",
    target_crs="EPSG:32617",
    target_resolution=10.0
)
```

**Algorithm Execution:**
```python
from core.analysis.library.baseline.flood import ThresholdSARAlgorithm

algo = ThresholdSARAlgorithm()
result = algo.execute(sar_data, pixel_size_m=10.0)

# Access results
flood_mask = result.data              # numpy array
stats = result.statistics             # dict with area, etc.
quality = result.quality_metrics      # confidence scores
```

**Quality Control:**
```python
from core.quality import SanityChecker, CrossValidator, UncertaintyEstimator

# Sanity checks
checker = SanityChecker()
issues = checker.check(result, event_class="flood.coastal")

# Cross-validation
validator = CrossValidator()
consensus = validator.validate([result1, result2, result3])

# Uncertainty
estimator = UncertaintyEstimator()
uncertainty_map = estimator.estimate(result)
```

**Multi-Sensor Fusion:**
```python
from core.analysis.fusion import MultiSensorFusion

fusion = MultiSensorFusion()
fused_result = fusion.fuse(
    results=[sar_result, optical_result],
    weights=[0.6, 0.4],
    method="weighted_consensus"
)
```

### 6.3 Agent Communication

```python
from agents.orchestrator import OrchestratorAgent
from agents.discovery import DiscoveryAgent
from agents.pipeline import PipelineAgent

# Create agents
orchestrator = OrchestratorAgent()
discovery = DiscoveryAgent()
pipeline = PipelineAgent()

# Submit event
event = orchestrator.submit(event_spec)

# Agents communicate via message bus
# Orchestrator delegates to Discovery → Pipeline → Quality → Reporting
result = await orchestrator.wait_for_completion(event.id)
```

---

## 7. Generating Visualizations

### 7.1 Export Formats

```bash
# Export to multiple formats
mdive export --input ./results/ --format geotiff,geojson,pdf,png --output ./products/

# Format-specific options
mdive export --input ./results/ --format geotiff --cog --compression lzw
mdive export --input ./results/ --format geojson --simplify 10m
mdive export --input ./results/ --format pdf --include-maps --include-statistics
```

### 7.2 Programmatic Visualization

```python
from core.visualization import MapGenerator, ReportGenerator

# Generate flood extent map
map_gen = MapGenerator()
map_gen.create_flood_map(
    flood_extent=result.data,
    background="satellite",
    output_path="flood_map.png",
    title="Miami Flood Extent - Sept 17, 2024",
    legend=True,
    scale_bar=True,
    north_arrow=True
)

# Generate uncertainty visualization
map_gen.create_uncertainty_map(
    uncertainty=uncertainty_layer,
    output_path="uncertainty_map.png",
    colormap="RdYlGn_r"
)

# Generate PDF report
report_gen = ReportGenerator()
report_gen.generate(
    event=event_spec,
    results=analysis_results,
    quality_metrics=quality_report,
    output_path="flood_report.pdf",
    template="emergency_response"
)
```

### 7.3 Available Visualization Types

| Type | Description | Output |
|------|-------------|--------|
| Flood extent map | Binary flood/no-flood overlay | PNG, GeoTIFF |
| Flood depth map | Continuous depth estimation | PNG, GeoTIFF |
| Burn severity map | Severity classification | PNG, GeoTIFF |
| Change magnitude | Pre/post difference | PNG, GeoTIFF |
| Uncertainty map | Confidence visualization | PNG, GeoTIFF |
| Time series | Temporal progression | PNG, GIF |
| Vector overlay | GeoJSON on basemap | HTML, PNG |
| PDF report | Complete analysis report | PDF |

---

## 8. Creating Analytics

### 8.1 Built-in Statistics

Every algorithm output includes statistics:

```python
result = algo.execute(data)

# Access statistics
stats = result.statistics
print(f"Flood area: {stats['flood_area_ha']} hectares")
print(f"Flood area: {stats['flood_area_km2']} km²")
print(f"Affected pixels: {stats['affected_pixel_count']}")
print(f"Total pixels: {stats['total_pixel_count']}")
print(f"Flood percentage: {stats['flood_percentage']}%")
```

### 8.2 Custom Analytics

```python
from core.analytics import AreaCalculator, ZonalStatistics, ImpactEstimator

# Calculate area by administrative zone
zonal = ZonalStatistics()
stats_by_zone = zonal.calculate(
    raster=flood_extent,
    zones=admin_boundaries,  # GeoDataFrame
    statistics=["area", "percentage", "max_depth"]
)

# Impact estimation
impact = ImpactEstimator()
impact_report = impact.estimate(
    flood_extent=flood_extent,
    population_data=population_raster,
    building_footprints=buildings_geojson,
    infrastructure=roads_and_utilities
)

print(f"Affected population: {impact_report['affected_population']}")
print(f"Affected buildings: {impact_report['affected_buildings']}")
print(f"Road km flooded: {impact_report['flooded_road_km']}")
```

### 8.3 Time Series Analysis

```python
from core.analytics import TimeSeriesAnalyzer

# Analyze flood progression
ts_analyzer = TimeSeriesAnalyzer()
progression = ts_analyzer.analyze(
    images=[day1_data, day2_data, day3_data, day4_data],
    timestamps=["2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18"],
    algorithm="flood.baseline.threshold_sar"
)

# Plot flood area over time
progression.plot_area_time_series()

# Get peak flood timing
print(f"Peak flood date: {progression.peak_date}")
print(f"Peak flood area: {progression.peak_area_km2} km²")
```

### 8.4 Quality Metrics

```python
from core.quality.reporting import QualityReporter

reporter = QualityReporter()
quality_report = reporter.generate(
    result=analysis_result,
    validation_data=ground_truth,  # Optional
    cross_validation_results=cv_results
)

print(f"Overall confidence: {quality_report['overall_confidence']}")
print(f"Spatial coherence: {quality_report['spatial_coherence_score']}")
print(f"Algorithm agreement: {quality_report['algorithm_agreement']}")
print(f"Issues found: {quality_report['issues']}")
```

---

## 9. End-to-End Workflow Examples

### 9.1 Example 1: Flood Analysis (CLI)

Scenario: Hurricane made landfall in Miami, need flood extent mapping.

```bash
# Step 1: Create event specification
cat > miami_flood.yaml << 'EOF'
id: evt_miami_flood_2024
intent:
  class: flood.coastal.storm_surge
spatial:
  type: Polygon
  coordinates:
    - [[-80.3, 25.7], [-80.1, 25.7], [-80.1, 25.9], [-80.3, 25.9], [-80.3, 25.7]]
temporal:
  start: "2024-09-15T00:00:00Z"
  end: "2024-09-20T23:59:59Z"
  reference_time: "2024-09-17T12:00:00Z"
constraints:
  max_cloud_cover: 0.4
  required_data_types: [sar, dem]
priority: critical
EOF

# Step 2: Discover available data
mdive discover --event miami_flood.yaml --format json > available_data.json

# Step 3: Run full analysis pipeline
mdive run --event miami_flood.yaml --profile workstation --output ./miami_products/

# Step 4: Check status (if running async)
mdive status --workdir ./miami_products/

# Step 5: Validate results
mdive validate --input ./miami_products/ --checks sanity,cross_validation

# Step 6: Export final products
mdive export --input ./miami_products/ --format geotiff,geojson,pdf --output ./miami_final/
```

Output files:
- `miami_final/flood_extent.tif` - GeoTIFF flood mask
- `miami_final/flood_extent.geojson` - Vector polygons
- `miami_final/flood_depth.tif` - Depth estimation (if HAND model ran)
- `miami_final/uncertainty.tif` - Confidence layer
- `miami_final/report.pdf` - Analysis report
- `miami_final/provenance.json` - Full lineage record

### 9.2 Example 2: Wildfire Burn Severity (Python SDK)

Scenario: Forest fire in California, need burn severity mapping.

```python
import asyncio
from core.intent import IntentResolver
from core.data.broker import DataBroker
from core.data.ingestion import DataIngester
from core.analysis.selection import DeterministicSelector
from core.analysis.library.baseline.wildfire import NBRDifferencedAlgorithm
from core.quality import SanityChecker, QualityReporter
from core.visualization import MapGenerator, ReportGenerator

async def analyze_wildfire():
    # Define event parameters
    event = {
        "intent": {"class": "wildfire.forest"},
        "spatial": {
            "type": "Polygon",
            "coordinates": [[[-121.5, 38.8], [-121.2, 38.8], [-121.2, 39.1],
                            [-121.5, 39.1], [-121.5, 38.8]]]
        },
        "temporal": {
            "start": "2024-08-10T00:00:00Z",
            "end": "2024-08-25T23:59:59Z",
            "reference_time": "2024-08-15T18:00:00Z"
        }
    }

    # Step 1: Discover data
    broker = DataBroker()
    datasets = await broker.discover(
        spatial=event["spatial"],
        temporal=event["temporal"],
        event_class=event["intent"]["class"],
        constraints={"max_cloud_cover": 0.3}
    )

    print(f"Found {len(datasets)} datasets")

    # Step 2: Select best pre/post images
    pre_image = next(d for d in datasets if d.temporal_role == "pre_event")
    post_image = next(d for d in datasets if d.temporal_role == "post_event")

    # Step 3: Ingest and normalize
    ingester = DataIngester()
    pre_data = await ingester.ingest(pre_image, target_resolution=10.0)
    post_data = await ingester.ingest(post_image, target_resolution=10.0)

    # Step 4: Run burn severity algorithm
    algo = NBRDifferencedAlgorithm()
    result = algo.execute(
        pre_image=pre_data,
        post_image=post_data,
        nir_band="B08",
        swir_band="B12"
    )

    print(f"Burned area: {result.statistics['burned_area_ha']} hectares")
    print(f"High severity: {result.statistics['high_severity_ha']} hectares")

    # Step 5: Quality control
    checker = SanityChecker()
    issues = checker.check(result, event_class="wildfire.forest")

    if issues:
        print(f"Quality issues: {issues}")

    # Step 6: Generate outputs
    map_gen = MapGenerator()
    map_gen.create_burn_severity_map(
        severity=result.data,
        output_path="burn_severity.png",
        title="California Fire Burn Severity"
    )

    reporter = QualityReporter()
    quality_report = reporter.generate(result)

    report_gen = ReportGenerator()
    report_gen.generate(
        event=event,
        results=result,
        quality_metrics=quality_report,
        output_path="wildfire_report.pdf"
    )

    return result

# Run analysis
result = asyncio.run(analyze_wildfire())
```

### 9.3 Example 3: Storm Damage Assessment (API)

Scenario: Tornado struck Oklahoma, need structural damage assessment.

```bash
# Step 1: Submit event via API
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "class": "storm.severe_convective",
      "original_input": "tornado damage assessment"
    },
    "spatial": {
      "type": "Polygon",
      "coordinates": [[[-97.6, 35.4], [-97.3, 35.4], [-97.3, 35.6], [-97.6, 35.6], [-97.6, 35.4]]]
    },
    "temporal": {
      "start": "2024-05-18T00:00:00Z",
      "end": "2024-05-22T23:59:59Z",
      "reference_time": "2024-05-19T20:30:00Z"
    },
    "constraints": {
      "max_cloud_cover": 0.2,
      "min_resolution_m": 3
    },
    "priority": "high"
  }'

# Response: {"event_id": "evt_123456", "status": "queued"}

# Step 2: Poll for status
curl http://localhost:8000/api/v1/events/evt_123456/status
# Response: {"status": "processing", "progress": 45, "current_step": "algorithm_execution"}

# Step 3: Wait for completion, then get products
curl http://localhost:8000/api/v1/events/evt_123456/products
# Response: {
#   "products": [
#     {"name": "damage_extent.geojson", "size_bytes": 524288},
#     {"name": "damage_severity.tif", "size_bytes": 10485760},
#     {"name": "report.pdf", "size_bytes": 2097152}
#   ]
# }

# Step 4: Download products
curl http://localhost:8000/api/v1/events/evt_123456/products/damage_extent.geojson \
  --output damage_extent.geojson

curl http://localhost:8000/api/v1/events/evt_123456/products/report.pdf \
  --output storm_report.pdf
```

### 9.4 Example 4: Multi-Event Parallel Processing (Docker)

Process multiple events simultaneously:

```bash
# Start full stack
docker compose up -d

# Submit multiple events (they process in parallel)
for event in flood_miami.yaml wildfire_california.yaml storm_oklahoma.yaml; do
  curl -X POST http://localhost:8000/api/v1/events \
    -H "Content-Type: application/json" \
    -d @examples/$event &
done

# Monitor all events
watch -n 5 'curl -s http://localhost:8000/api/v1/events | jq ".events[] | {id, status, progress}"'
```

### 9.5 Example 5: Resumable Long-Running Analysis

```bash
# Start analysis (may take hours for large areas)
mdive run --area large_area.geojson --event flood --profile cloud --output ./analysis/

# If interrupted (Ctrl+C, system restart, etc.), resume:
mdive resume --workdir ./analysis/

# The system tracks checkpoints and resumes from last completed step
```

---

## 10. Quick Reference

### Command Cheat Sheet

```bash
# Discovery
mdive discover --area X.geojson --event flood         # Find available data
mdive discover --bbox -80.5,25.5,-80.0,26.0 --event wildfire

# Analysis
mdive run --event X.yaml --profile laptop             # Full pipeline
mdive analyze --input ./data/ --algorithm sar_threshold

# Validation
mdive validate --input ./results/ --checks sanity
mdive validate --input ./results/ --reference truth.geojson

# Export
mdive export --input ./results/ --format geotiff,geojson,pdf

# Status
mdive status --workdir ./products/
mdive resume --workdir ./products/
```

### Python Import Reference

```python
# Intent
from core.intent import IntentResolver

# Data
from core.data.broker import DataBroker
from core.data.ingestion import DataIngester

# Algorithms
from core.analysis.library.baseline.flood import (
    ThresholdSARAlgorithm,
    NDWIOpticalAlgorithm,
    ChangeDetectionAlgorithm,
    HANDModelAlgorithm
)
from core.analysis.library.baseline.wildfire import (
    ThermalAnomalyAlgorithm,
    NBRDifferencedAlgorithm,
    BurnedAreaClassifier
)
from core.analysis.library.baseline.storm import (
    WindDamageAlgorithm,
    StructuralDamageAlgorithm
)

# Selection
from core.analysis.selection import DeterministicSelector

# Quality
from core.quality import SanityChecker, CrossValidator, UncertaintyEstimator
from core.quality.reporting import QualityReporter

# Visualization
from core.visualization import MapGenerator, ReportGenerator
```

### Event Specification Template

```yaml
id: evt_unique_id

intent:
  class: flood.coastal | wildfire.forest | storm.severe_convective
  source: explicit | inferred
  confidence: 0.0-1.0

spatial:
  type: Polygon
  coordinates: [[[lon1, lat1], [lon2, lat2], ...]]
  crs: EPSG:4326
  bbox: [min_lon, min_lat, max_lon, max_lat]

temporal:
  start: "YYYY-MM-DDTHH:MM:SSZ"
  end: "YYYY-MM-DDTHH:MM:SSZ"
  reference_time: "YYYY-MM-DDTHH:MM:SSZ"

constraints:
  max_cloud_cover: 0.0-1.0
  min_resolution_m: number
  required_data_types: [sar, optical, dem, weather]
  optional_data_types: [...]

priority: critical | high | normal | low

metadata:
  created_at: "YYYY-MM-DDTHH:MM:SSZ"
  created_by: "string"
  tags: [tag1, tag2]
```

### Docker Commands

```bash
# Full stack
docker compose up -d
docker compose down
docker compose logs -f api

# CLI only
docker compose -f docker-compose.minimal.yml run --rm cli mdive run ...

# Build images
docker build -f docker/api/Dockerfile -t mdive-api .
docker build -f docker/cli/Dockerfile -t mdive-cli .
```

### Test Commands

```bash
./run_tests.py                      # All tests
./run_tests.py flood                # Flood tests
./run_tests.py wildfire             # Wildfire tests
./run_tests.py storm                # Storm tests
./run_tests.py schemas              # Schema validation
./run_tests.py --algorithm sar      # SAR algorithm
./run_tests.py --quick              # Fast tests only
./run_tests.py --list               # List categories
```

---

## Troubleshooting

### Common Issues

**"No data found" error:**
- Check date range - satellite revisit is typically 6-12 days
- Expand temporal window
- Try different data sources

**"Algorithm not applicable" error:**
- Check if required data types are available
- Some algorithms need specific bands (e.g., dNBR needs NIR+SWIR)
- Use `mdive discover` to see what's available

**Memory errors:**
- Use appropriate profile for your hardware
- Use `--profile laptop` for machines with <8GB RAM
- Large areas automatically use tiling

**Docker connection refused:**
- Ensure services are running: `docker compose ps`
- Check logs: `docker compose logs api`
- Verify ports aren't in use: `lsof -i :8000`

### Getting Help

```bash
mdive --help                        # General help
mdive run --help                    # Command-specific help
mdive info                          # System information

# Test your setup
./run_tests.py schemas              # Verify schemas
./run_tests.py --quick              # Quick sanity check
```

---

*Last updated: 2024*
