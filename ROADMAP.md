# Multiverse Dive: Journey Through the Event Intelligence Cosmos

## Executive Summary

Welcome, cosmic explorer! This roadmap charts our voyage through the multiverse of geospatial event intelligence. We're building a platform that transforms raw observations of floods, wildfires, and storms into actionable decision products—situation-agnostic, reproducible, and powered by intelligent agents.

Our journey is organized into parallel exploration groups, starting with quick wins that establish our foundation and progressively building toward full autonomous orchestration. Each group represents a constellation of tasks that can be tackled simultaneously, with clear dependencies on previous explorations.

Think of this as a tech tree in a cosmic strategy game: early groups unlock fundamental capabilities, middle groups add intelligence and sophistication, and later groups bring it all together into a living, breathing system. No time estimates—we focus on *what* needs exploring, not *when*. Each milestone brings new superpowers.

Let's dive into the multiverse!

---

## Dependency Flow Diagram

```
GROUP A: Foundation Firmament
    ├─→ pyproject.toml
    ├─→ 7 JSON Schemas (parallel)
    └─→ Event Class Definitions (3 parallel)
         │
         ↓
GROUP B: Schema Validation & Examples
    ├─→ Validator
    ├─→ Example Specs
    └─→ Basic Tests
         │
         ↓
GROUP C: Intent Resolution Core
    ├─→ Registry
    ├─→ NLP Classifier
    └─→ Resolver Logic
         │
         ↓
┌────────┴────────┐
│                 │
GROUP D:          GROUP E:
Data Discovery    Algorithm Foundation
    │                 │
    ↓                 ├─────────────────────────────┐
GROUP F: Intelligent Selection                      │
(combines D + E outputs)                            │
    │                                               │
    ↓                                               ↓
┌────────┴────────┐                         GROUP L: Lightweight
│                 │                         Execution & CLI
GROUP G:          GROUP H:                  (parallel with F-K)
Ingestion         Fusion & Analysis             │
Pipeline          Engine                        │
    │                 │                         │
    └────────┬────────┘                         │
             ↓                                  │
    GROUP I: Quality Control ◄──────────────────┤
             │                                  │
             ↓                                  │
    GROUP J: Agent Orchestration                │
             │                                  │
             ↓                                  │
    GROUP K: API & Deployment ◄─────────────────┘
             │
             ↓
    ┌────────┴────────┐
    │                 │
GROUP M:          GROUP N:
Resilience &      Containerization
Fallbacks         & Deployment
    │                 │
    └────────┬────────┘
             ↓
    [Production Ready: Cloud + Local + Edge]
```

---

## Detailed Groups

### **Group A: Foundation Firmament** ⭐ **[DONE]**
*"In the beginning, there were schemas..."*

**Prerequisites:** None—this is where it all starts!

**Parallel Tracks:**

1. **Track 1: Project Configuration**
   - Create `pyproject.toml` with dependencies (GDAL, rasterio, xarray, FastAPI, etc.)
   - Set up directory structure

2. **Track 2: Schema Constellation** (All 7 schemas can be written in parallel!)
   - `intent.schema.json` - Handles event typing and NLP inference
   - `event.schema.json` - Core specification (area + time + intent)
   - `datasource.schema.json` - Provider framework schema
   - `pipeline.schema.json` - Workflow composition schema
   - `ingestion.schema.json` - Normalization and conversion schema
   - `quality.schema.json` - QA/QC schema
   - `provenance.schema.json` - Lineage tracking schema

3. **Track 3: Event Class Taxonomy** (All 3 definitions in parallel!)
   - `openspec/definitions/event_classes/flood.yaml`
   - `openspec/definitions/event_classes/wildfire.yaml`
   - `openspec/definitions/event_classes/storm.yaml`

**Deliverables:**
- `pyproject.toml`
- 7 JSON schema files in `openspec/schemas/`
- 3 YAML event class definitions
- Directory structure created

**Success Criteria:**
- All schema files are valid JSON
- Event class YAML files parse correctly
- Directory structure matches spec
- Dependencies installable via `pip install -e .`

**Celebration Checkpoint:** 🎆
You now have the universal laws of your multiverse defined! Every piece of data that flows through the system will speak this common language. Schemas are your physics engine.

---

### **Group B: Schema Validation & Examples** 🔍 **[DONE]**
*"With great schemas comes great validation responsibility"*

**Prerequisites:** Group A complete

**Parallel Tracks:**

1. **Track 1: Validator Engine**
   - `openspec/validator.py` with helpful error messages
   - Schema loading and caching
   - Validation utilities for each schema type

2. **Track 2: Example Gallery** (All examples in parallel!)
   - `examples/flood_event.yaml` - Coastal flood example
   - `examples/wildfire_event.yaml` - Forest fire example
   - `examples/storm_event.yaml` - Hurricane example
   - Example data source configurations
   - Example pipeline definitions

3. **Track 3: Foundation Tests**
   - `tests/test_schemas.py` - Validate all schemas
   - `tests/test_validator.py` - Test validation logic
   - `tests/test_examples.py` - Ensure examples validate

**Deliverables:**
- `openspec/validator.py`
- 3+ example YAML files in `examples/`
- Test suite covering schemas and validation

**Success Criteria:**
- `pytest tests/test_schemas.py -v` passes
- All example files validate successfully
- Validator provides clear, actionable error messages
- 100% schema coverage in tests

**Celebration Checkpoint:** 🎨
Your universe now has guardrails! Invalid specifications can't sneak through. The example gallery serves as both documentation and validation that your schemas actually work. New contributors can learn by example.

---

### **Group C: Intent Resolution Core** 🧠 **[DONE]**
*"What did the human actually mean by 'flooding after hurricane'?"*

**Prerequisites:** Group B complete (needs schemas + validator)

**Parallel Tracks:**

1. **Track 1: Registry Foundation**
   - `core/intent/registry.py`
   - Hierarchical taxonomy loading from YAML
   - Wildcard matching (e.g., `flood.*` matches `flood.coastal`)
   - Event class metadata lookup

2. **Track 2: NLP Classifier**
   - `core/intent/classifier.py`
   - Natural language → event class inference
   - Confidence scoring
   - Alternative suggestions
   - Simple rule-based approach initially (can enhance with ML later)

3. **Track 3: Resolution Orchestration**
   - `core/intent/resolver.py`
   - Combine registry + classifier + user overrides
   - Structured output generation
   - Resolution logging and provenance

4. **Track 4: Intent Tests**
   - `tests/test_intent.py`
   - Test registry loading
   - Test NLP inference accuracy
   - Test override handling
   - Test edge cases ("what if input is nonsensical?")

**Deliverables:**
- `core/intent/registry.py`
- `core/intent/classifier.py`
- `core/intent/resolver.py`
- Comprehensive test suite

**Success Criteria:**
- Registry loads all event class definitions
- NLP classifier achieves reasonable accuracy on test cases
- User overrides always take precedence
- Confidence scores are calibrated (don't claim 0.99 without evidence!)
- `pytest tests/test_intent.py -v` passes

**Celebration Checkpoint:** 🎯
Your system can now understand intent! Whether users speak in precise taxonomy terms or natural language, the platform knows what they're asking for. This is the bridge between human urgency and machine precision.

---

### **Group D: Data Discovery Expedition** 🔭 **[DONE]**
*"Where is the data hiding in this vast cosmos?"*

**Prerequisites:** Group C complete (needs intent resolution to know *what* data to seek)

**Parallel Tracks:**

1. **Track 1: Broker Core**
   - `core/data/broker.py` - Main orchestration
   - Discovery request handling
   - Selection decision output

2. **Track 2: Discovery Adapters** (All in parallel!)
   - `core/data/discovery/base.py` - Abstract interface
   - `core/data/discovery/stac.py` - STAC catalog queries
   - `core/data/discovery/wms_wcs.py` - OGC services
   - `core/data/discovery/provider_api.py` - Custom APIs

3. **Track 3: Provider Implementations** (Groups can be parallel!)
   - **Optical** (all parallel):
     - `core/data/providers/optical/sentinel2.py`
     - `core/data/providers/optical/landsat.py`
     - `core/data/providers/optical/modis.py`
   - **SAR** (both parallel):
     - `core/data/providers/sar/sentinel1.py`
     - `core/data/providers/sar/alos.py`
   - **DEM** (all parallel):
     - `core/data/providers/dem/copernicus.py`
     - `core/data/providers/dem/srtm.py`
     - `core/data/providers/dem/fabdem.py`
   - **Weather** (all parallel):
     - `core/data/providers/weather/era5.py`
     - `core/data/providers/weather/gfs.py`
     - `core/data/providers/weather/ecmwf.py`
   - **Ancillary** (all parallel):
     - `core/data/providers/ancillary/osm.py`
     - `core/data/providers/ancillary/wsf.py`
     - `core/data/providers/ancillary/landcover.py`

4. **Track 4: Provider Registry**
   - `core/data/providers/registry.py`
   - Provider preference ordering (open → restricted → commercial)
   - Capability metadata
   - Fallback policies

5. **Track 5: Discovery Tests**
   - `tests/test_broker.py`
   - Mock provider responses
   - Test spatial/temporal filtering
   - Test multi-source discovery

**Deliverables:**
- Data broker architecture
- 13 provider implementations
- Provider registry with preference system
- Test suite with mocked responses

**Success Criteria:**
- Broker can query multiple catalogs in parallel
- STAC queries work against Element84 Earth Search
- Provider registry correctly prioritizes open data
- Tests don't require actual data downloads (use mocks)
- `pytest tests/test_broker.py -v` passes

**Celebration Checkpoint:** 🛰️
Your platform can now see across the data universe! Sentinel, Landsat, weather models, DEMs—they're all discoverable. The broker speaks the language of each provider and knows where to look for the perfect dataset.

---

### **Group E: Algorithm Foundation** ⚗️ **[DONE]**
*"Assembling our toolkit of analytical sorcery"*

**Prerequisites:** Group C complete (need event classes), but can develop in parallel with Group D

**Parallel Tracks:**

1. **Track 1: Registry Infrastructure** ✅ **[DONE]**
   - `core/analysis/library/registry.py` - Complete algorithm registry system (546 lines)
   - Algorithm metadata schema with version tracking
   - Event type wildcard matching (e.g., `flood.*` matches `flood.coastal`)
   - Data availability and resource constraint filtering
   - YAML loading from `openspec/definitions/algorithms/`
   - 9 baseline algorithm definitions loaded successfully
   - 38 comprehensive tests passing

2. **Track 2: Baseline Flood Algorithms** (all parallel) ✅ **[DONE]**
   - `core/analysis/library/baseline/flood/threshold_sar.py` - SAR backscatter threshold detection (v1.2.0)
   - `core/analysis/library/baseline/flood/ndwi_optical.py` - NDWI optical flood detection (v1.1.0)
   - `core/analysis/library/baseline/flood/change_detection.py` - Pre/post temporal change detection (v1.0.0)
   - `core/analysis/library/baseline/flood/hand_model.py` - Height Above Nearest Drainage susceptibility (v1.0.0)
   - `core/analysis/library/baseline/flood/__init__.py` - Module exports and algorithm registry
   - `tests/test_flood_algorithms.py` - Comprehensive test suite (818 lines, all tests passing)

3. **Track 3: Baseline Wildfire Algorithms** (all parallel) ✅ **[DONE]**
   - `core/analysis/library/baseline/wildfire/nbr_differenced.py` - Differenced NBR burn severity mapping (v1.0.0, 425 lines)
   - `core/analysis/library/baseline/wildfire/thermal_anomaly.py` - Thermal anomaly active fire detection (v1.0.0, 511 lines)
   - `core/analysis/library/baseline/wildfire/ba_classifier.py` - ML-based burned area classification (v1.0.0, 620 lines)
   - `core/analysis/library/baseline/wildfire/__init__.py` - Module exports and algorithm registry (82 lines)
   - Tests integrated into `tests/test_algorithms.py` - 37 wildfire-specific tests passing

4. **Track 4: Baseline Storm Algorithms** (both parallel) ✅ **[DONE]**
   - `core/analysis/library/baseline/storm/wind_damage.py` - Wind damage vegetation detection (v1.0.0)
   - `core/analysis/library/baseline/storm/structural_damage.py` - Structural damage assessment (v1.0.0)
   - `core/analysis/library/baseline/storm/__init__.py` - Module exports and algorithm registry

5. **Track 5: Algorithm Tests** ✅ **[DONE]**
   - `tests/test_algorithms.py` - Comprehensive test suite (1899 lines, 104 tests passing)
   - Test each algorithm with synthetic data (SyntheticDataGenerator class)
   - Validate reproducibility (deterministic algorithms) - TestAlgorithmReproducibility
   - Test parameter ranges - TestParameterRanges
   - Registry integration tests - TestRegistryIntegration
   - Edge case handling - TestEdgeCases

**Deliverables:**
- Algorithm registry system
- 9 baseline algorithms across 3 hazard types
- Comprehensive test suite

**Success Criteria:**
- Each algorithm has clear metadata (requirements, parameters, outputs)
- Algorithms are reproducible with same inputs
- Registry can look up algorithms by event type
- Tests cover happy path + edge cases
- `pytest tests/test_algorithms.py -v` passes

**Celebration Checkpoint:** 🔬
Your analytical arsenal is ready! These baseline algorithms are battle-tested workhorses—simple, interpretable, and reliable. They form the foundation for more sophisticated approaches later.

---

### **Group F: Intelligent Selection Systems** 🎲 **[DONE]**
*"Choosing wisely from the buffet of data and algorithms"*

**Prerequisites:** Groups D + E complete (needs both data discovery and algorithms)

**Parallel Tracks:**

1. **Track 1: Constraint Evaluation Engine** ✅ **[DONE]**
   - `core/data/evaluation/constraints.py` (674 lines, comprehensive implementation)
   - Hard constraint checking (spatial/temporal/availability)
   - Soft constraint scoring (cloud cover, resolution, proximity)
   - Additional soft constraints: SAR noise quality, geometric accuracy, AOI proximity, view/incidence angle
   - Evaluation result schema with full diagnostics
   - 99 comprehensive tests passing (test_constraints.py)

2. **Track 2: Multi-Criteria Ranking** ✅ **[DONE]**
   - `core/data/evaluation/ranking.py`
   - Weighted scoring across criteria
   - Provider preference integration
   - Trade-off documentation

3. **Track 3: Atmospheric Assessment** ✅ **[DONE]**
   - `core/data/selection/atmospheric.py` (573 lines, comprehensive implementation)
   - Cloud cover assessment with configurable thresholds
   - Weather condition evaluation (precipitation, severe weather, visibility, smoke/aerosols)
   - Sensor suitability recommendations for optical, SAR, thermal, and LIDAR
   - 33 comprehensive tests passing (including atmospheric edge cases)
   - Event-specific sensor recommendations integrated

4. **Track 4: Sensor Selection Strategy** ✅ **[DONE]**
   - `core/data/selection/strategy.py` (606 lines, comprehensive implementation)
   - Optimal sensor combination logic with dependency resolution
   - Degraded mode handling with confidence thresholds
   - Confidence tracking per observable
   - 78 comprehensive tests passing (includes atmospheric assessment tests from Track 3)
   - Bug fixed: degraded mode threshold corrected (MEDIUM is acceptable, only LOW is degraded)

5. **Track 5: Fusion Strategy** ✅ **[DONE]**
   - `core/data/selection/fusion.py` (870 lines, comprehensive implementation)
   - Multi-sensor blending rules with 6 blending methods (weighted_average, quality_mosaic, temporal_composite, consensus, priority_stack, kalman_filter)
   - Complementary vs redundant sensor strategies with automatic detection
   - Temporal densification with gap-fill support
   - Pre-built configurations for flood, wildfire, storm, and general use
   - FusionStrategyEngine with event-aware strategy selection
   - 54 comprehensive tests covering fusion edge cases (test_fusion.py)
   - Bug fixed: Division by zero guard in _calculate_fusion_confidence when configuration.sensors is empty

6. **Track 6: Algorithm Selector** ✅ **[DONE]**
   - `core/analysis/selection/selector.py` (835 lines, comprehensive implementation)
   - Rule-based algorithm filtering with event type matching
   - Data availability matching with missing data rejection
   - Compute constraint checking (memory, GPU, runtime, distributed)
   - Multi-criteria scoring with configurable weights
   - Validation-aware selection with min accuracy enforcement
   - Compute profiles (laptop, workstation, cloud, edge)
   - Comprehensive rejection tracking with detailed reasons
   - 40 comprehensive tests covering edge cases and constraints

7. **Track 7: Deterministic Selection** ✅ **[DONE]**
   - `core/analysis/selection/deterministic.py` (470 lines, comprehensive implementation)
   - `core/analysis/selection/__init__.py` - Module exports for selector components
   - Reproducible selection logic with category-aware ordering
   - Version pinning with full version lock tracking
   - Selection hash generation (SHA-256 based, 16-char truncated)
   - SelectionPlan with plan-level hash verification
   - Constraint evaluation (memory, GPU, determinism, excluded algorithms)
   - Trade-off documentation between selected algorithms
   - 25 comprehensive tests covering edge cases and determinism
   - Full integration with AlgorithmRegistry and DataAvailability

8. **Track 8: Selection Tests** ✅ **[DONE]**
   - `tests/test_selection.py` (209 tests, all passing)
   - Test constraint evaluation (TestSelectionConstraints, 8 tests)
   - Test ranking with various weights (TestAlgorithmSelectorScoringTrack6, 5 tests)
   - Test atmospheric-aware selection (TestAtmosphericEvaluator/EdgeCases, 38 tests)
   - Test algorithm selection determinism (TestDeterministicSelector/EdgeCases, 30 tests)
   - Additional edge cases: NaN/Inf handling for cloud cover and visibility

**Deliverables:**
- Constraint evaluation engine
- Multi-criteria ranking system
- Intelligent sensor selection with degraded modes
- Algorithm selection engine
- Comprehensive test coverage

**Success Criteria:**
- Constraint evaluator correctly filters viable datasets
- Ranking produces consistent, explainable orderings
- Atmospheric conditions influence sensor choice appropriately
- Degraded modes trigger at correct thresholds
- Algorithm selector only picks algorithms with available data
- Deterministic mode produces identical selections given same inputs
- `pytest tests/test_selection.py -v` passes

**Celebration Checkpoint:** 🧩
Your system can now make intelligent decisions! It knows when to use optical vs SAR, when to fall back to lower resolution, and which algorithm is best suited for the situation. Trade-offs are documented, and selections are reproducible.

---

### **Group G: Ingestion & Normalization Pipeline** 🌊 **[DONE]**
*"Taming the chaos into harmonious, analysis-ready data"*

**Prerequisites:** Group F complete (needs data selection to know what to ingest)

**Parallel Tracks:**

1. **Track 1: Pipeline Orchestration** ✅ **[DONE]**
   - `core/data/ingestion/pipeline.py` (760+ lines, comprehensive implementation)
   - IngestionJob dataclass with full lifecycle tracking
   - IngestionPipeline orchestrator with parallel execution
   - JobManager for concurrent job management and progress tracking
   - Error handling with configurable retry policies
   - Event-driven callbacks for progress notifications
   - Integration with validation, enrichment, and persistence stages

2. **Track 2: Format Converters** (all parallel) ✅ **[DONE]**
   - `core/data/ingestion/formats/cog.py` - Cloud-Optimized GeoTIFF (455 lines)
   - `core/data/ingestion/formats/zarr.py` - Zarr arrays (476 lines)
   - `core/data/ingestion/formats/parquet.py` - GeoParquet vectors (432 lines)
   - `core/data/ingestion/formats/stac_item.py` - STAC metadata (588 lines)
   - `core/data/ingestion/formats/__init__.py` - Module exports

3. **Track 3: Normalization Tools** (all parallel) ✅ **[DONE]**
   - `core/data/ingestion/normalization/projection.py` - CRS handling (903 lines)
     - CRSHandler with EPSG parsing, UTM zone detection, bounds transformation
     - CoordinateTransformer for point and array coordinate transformations
     - RasterReprojector for raster reprojection with configurable resampling
   - `core/data/ingestion/normalization/tiling.py` - Tile schemes (878 lines)
     - TileScheme definitions (Web Mercator, geographic, custom grids)
     - TileManager for grid generation, overlap handling, AOI coverage
     - Tile coordinate systems and indexing utilities
   - `core/data/ingestion/normalization/temporal.py` - Time alignment (921 lines)
     - TimeNormalizer for timezone handling and temporal binning
     - TemporalAligner for multi-source time series alignment
     - Gap detection, interpolation, and temporal resampling
   - `core/data/ingestion/normalization/resolution.py` - Resampling (936 lines)
     - ResolutionManager for resolution analysis and harmonization
     - Resampler with multiple methods (nearest, bilinear, cubic, lanczos)
     - Quality-aware resolution matching across sensor types
   - `core/data/ingestion/normalization/__init__.py` - Module exports (130 lines)
   - `tests/test_normalization.py` - Comprehensive test suite (72 tests, 56 passing, 16 skipped for optional deps)

4. **Track 4: Enrichment** (all parallel) ✅ **[DONE]**
   - `core/data/ingestion/enrichment/overviews.py` - Pyramid generation (602 lines)
   - `core/data/ingestion/enrichment/statistics.py` - Band statistics (669 lines)
   - `core/data/ingestion/enrichment/quality.py` - Quality summaries (896 lines)
   - `core/data/ingestion/enrichment/__init__.py` - Module exports (74 lines)
   - `tests/test_enrichment.py` - Comprehensive test suite (1348 lines, 100 tests)

5. **Track 5: Validation** (all parallel) ✅ **[DONE]**
   - `core/data/ingestion/validation/integrity.py` - File integrity checks (805 lines)
     - IntegrityValidator with checksum (MD5, SHA256), format, CRS validation
     - GeoTIFF, Zarr, GeoParquet, NetCDF format support
   - `core/data/ingestion/validation/anomaly.py` - Anomaly detection (765 lines)
     - Statistical outliers (z-score, IQR, MAD)
     - Spatial artifacts (stripes, saturation, dark/bright regions)
     - Per-band quality scoring
   - `core/data/ingestion/validation/completeness.py` - Coverage checks (830 lines)
     - Spatial/temporal coverage validation
     - Gap region detection with connected components
     - Band consistency and metadata validation
   - `core/data/ingestion/validation/__init__.py` - ValidationSuite (258 lines)
     - Combined validation runner with overall quality scoring
   - 55 comprehensive tests in tests/test_validation.py

6. **Track 6: Persistence** ✅ **[DONE]**
   - `core/data/ingestion/persistence/storage.py` - Storage backends (1212 lines)
     - LocalStorageBackend with checksum verification
     - S3StorageBackend with boto3 integration
     - Factory function and URI parsing utilities
   - `core/data/ingestion/persistence/intermediate.py` - Product management (1039 lines)
     - ProductManager with SQLite-backed tracking
     - Lifecycle management (creation, access, expiration, cleanup)
     - Content deduplication via checksum indexing
     - Dependency tracking between products
   - `core/data/ingestion/persistence/lineage.py` - Lineage tracking (1052 lines)
     - LineageTracker with provenance.schema.json compliance
     - TrackingContext and StepContext for pipeline tracking
     - Input dataset, algorithm, and quality summary tracking
     - Reproducibility hashes and environment capture
   - `core/data/ingestion/persistence/__init__.py` - Module exports (105 lines)
   - 108 comprehensive tests in `tests/test_persistence.py` (1430 lines)

7. **Track 7: Cache System** (can develop in parallel with above) ✅ **[DONE]**
   - `core/data/cache/manager.py` - Lifecycle management (1033 lines)
     - CacheManager with SQLite-backed metadata tracking
     - TTL-based expiration and LRU/LFU/FIFO eviction policies
     - Thread-safe operations with background cleanup
     - Access statistics and hit/miss tracking
   - `core/data/cache/index.py` - Spatiotemporal indexing (959 lines)
     - R-tree spatial indexing for bounding box queries
     - Temporal range queries with overlap detection
     - Combined spatiotemporal queries with relevance scoring
     - Support for in-memory and file-based databases
   - `core/data/cache/storage.py` - S3/local backends (1267 lines)
     - LocalCacheStorage with content-addressable storage
     - MemoryCacheStorage for testing and small datasets
     - S3CacheStorage with boto3 integration
     - Tiered storage with hot/warm/cold promotion/demotion
   - `core/data/cache/__init__.py` - Module exports (107 lines)
   - `tests/test_cache.py` - Comprehensive test suite (1433 lines, 78 tests)

8. **Track 8: Ingestion Tests** ✅ **[DONE]**
   - `tests/test_ingestion.py` (2213 lines, 143 tests)
   - Test format conversions (COG, Zarr, GeoParquet, STAC)
   - Test normalization accuracy (projection, tiling, temporal, resolution)
   - Test validation catches issues (integrity, anomaly, completeness)
   - Test edge cases and error handling
   - Test end-to-end pipeline integration
   - **Cache Manager Tests** (complete integration with CacheConfig, CacheEntry, CacheManager)
     - Cache entry lifecycle (put, get, invalidate, delete)
     - Cache statistics and hit rate tracking
     - Eviction policies (LRU, entry limits, size limits)
     - Cache cleanup and expiration
     - Thread safety for concurrent operations
   - 88 tests passing (55 skipped due to optional dependencies)

**Deliverables:**
- Full ingestion pipeline infrastructure
- Cloud-native format converters
- Normalization and enrichment tools
- Validation suite
- Cache system
- Comprehensive tests

**Success Criteria:**
- Raw data converts to COG/Zarr successfully
- Projections transform correctly with <1 pixel error
- Overviews render smoothly at multiple zoom levels
- Validation detects corrupted files
- Cache provides fast lookup by space/time
- Lineage tracking captures full provenance
- `pytest tests/test_ingestion.py -v` passes

**Celebration Checkpoint:** 🏗️
Your data factory is operational! Raw, messy, heterogeneous inputs now flow through and emerge as pristine, analysis-ready, cloud-native products. The cache makes repeated operations lightning-fast.

---

### **Group H: Fusion & Analysis Engine** ⚛️ **[DONE]**
*"Where multiple perspectives converge into singular truth"*

**Prerequisites:** Groups E + G complete (needs algorithms + ingested data)

**Parallel Tracks:**

1. **Track 1: Pipeline Assembly** ✅ **[DONE]**
   - `core/analysis/assembly/graph.py` - Pipeline graph representation (894 lines)
   - `core/analysis/assembly/assembler.py` - DAG construction (1029 lines)
   - `core/analysis/assembly/validator.py` - Pre-execution validation (808 lines)
   - `core/analysis/assembly/optimizer.py` - Execution optimization (828 lines)
   - `core/analysis/assembly/__init__.py` - Module exports (141 lines)
   - `tests/test_assembly.py` - Comprehensive test suite (64 tests passing)

2. **Track 2: Fusion Core** ✅ **[DONE]**
   - `core/analysis/fusion/alignment.py` - Spatial/temporal alignment (1268 lines)
   - `core/analysis/fusion/corrections.py` - Terrain/atmospheric corrections (1074 lines)
   - `core/analysis/fusion/conflict.py` - Conflict resolution (927 lines)
   - `core/analysis/fusion/uncertainty.py` - Uncertainty propagation (1006 lines)
   - `core/analysis/fusion/__init__.py` - Module exports (187 lines)
   - `tests/test_fusion_core.py` - Comprehensive test suite (82 tests passing)

3. **Track 3: Execution Engine** ✅ **[DONE]**
   - `core/analysis/execution/runner.py` - Pipeline executor (1196 lines)
   - `core/analysis/execution/distributed.py` - Dask/Ray integration (1045 lines)
   - `core/analysis/execution/checkpoint.py` - State persistence (935 lines)
   - `core/analysis/execution/__init__.py` - Module exports (171 lines)

4. **Track 4: Forecast Integration** ✅ **[DONE]**
   - `core/analysis/forecast/ingestion.py` - Forecast data handling (926 lines)
   - `core/analysis/forecast/validation.py` - Forecast vs observation (1017 lines)
   - `core/analysis/forecast/scenarios.py` - Scenario analysis (957 lines)
   - `core/analysis/forecast/projection.py` - Impact projections (904 lines)
   - `core/analysis/forecast/__init__.py` - Module exports (163 lines)
   - `tests/test_forecast.py` - Comprehensive test suite (57 tests passing)

5. **Track 5: Advanced Algorithms** ✅ **[DONE]**
   - `core/analysis/library/advanced/flood/unet_segmentation.py` (672 lines)
   - `core/analysis/library/advanced/flood/ensemble_fusion.py` (741 lines)
   - `core/analysis/library/advanced/flood/__init__.py` - Algorithm registry (81 lines)
   - `tests/test_advanced_algorithms.py` - Comprehensive test suite (62 tests passing)

6. **Track 6: Fusion Tests** ✅ **[DONE]**
   - `tests/test_fusion.py` (2514 lines, 127 tests passing)
   - Test multi-sensor alignment (alignment enums, reference grid, spatial/temporal alignment)
   - Test conflict resolution strategies (weighted mean, majority vote, median, priority order)
   - Test pipeline assembly and execution (fusion strategy engine, blending weights)
   - Test forecast integration
   - Test corrections (terrain, atmospheric, radiometric normalization)
   - Test uncertainty propagation and combination

**Deliverables:**
- Dynamic pipeline assembler
- Multi-sensor fusion engine
- Distributed execution system
- Forecast integration framework
- Optional advanced algorithms
- Test coverage

**Success Criteria:**
- Pipeline assembler creates valid DAGs
- Multi-sensor data aligns within sub-pixel accuracy
- Conflict resolution produces reasonable consensus
- Distributed execution scales across workers
- Forecast data integrates with observations
- `pytest tests/test_fusion.py -v` passes

**Celebration Checkpoint:** ⚡
Your analytical engine roars to life! Multiple sensors combine their perspectives, algorithms run in parallel across distributed workers, and forecasts blend with observations. This is where the magic happens.

---

### **Group I: Quality Control Citadel** 🛡️
*"Trust, but verify—rigorously"*

**Prerequisites:** Group H complete (needs analysis outputs to validate)

**Parallel Tracks:**

1. **Track 1: Sanity Checks** (all parallel)
   - `core/quality/sanity/spatial.py` - Spatial coherence
   - `core/quality/sanity/values.py` - Value plausibility
   - `core/quality/sanity/temporal.py` - Temporal consistency
   - `core/quality/sanity/artifacts.py` - Artifact detection

2. **Track 2: Cross-Validation** (all parallel)
   - `core/quality/validation/cross_model.py` - Model comparison
   - `core/quality/validation/cross_sensor.py` - Sensor validation
   - `core/quality/validation/historical.py` - Historical baselines
   - `core/quality/validation/consensus.py` - Consensus generation

3. **Track 3: Uncertainty Quantification**
   - `core/quality/uncertainty/quantification.py` - Metrics calculation
   - `core/quality/uncertainty/spatial_uncertainty.py` - Spatial mapping
   - `core/quality/uncertainty/propagation.py` - Error propagation

4. **Track 4: Action Management** (all parallel)
   - `core/quality/actions/gating.py` - Pass/fail/review logic
   - `core/quality/actions/flagging.py` - Quality flag system
   - `core/quality/actions/routing.py` - Expert review routing

5. **Track 5: Reporting**
   - `core/quality/reporting/qa_report.py` - QA report generation
   - `core/quality/reporting/diagnostics.py` - Diagnostic outputs

6. **Track 6: Quality Tests**
   - `tests/test_quality.py`
   - Test sanity check detection
   - Test cross-validation metrics
   - Test gating logic
   - Test uncertainty propagation

**Deliverables:**
- Comprehensive sanity check suite
- Cross-validation framework
- Uncertainty quantification
- Gating and routing system
- QA reporting
- Test coverage

**Success Criteria:**
- Sanity checks catch physically impossible results
- Cross-validation correctly identifies disagreements
- Uncertainty estimates are calibrated
- Gating system appropriately blocks bad outputs
- Expert review routes only when truly needed
- QA reports are clear and actionable
- `pytest tests/test_quality.py -v` passes

**Celebration Checkpoint:** 🏆
Your fortress of quality is complete! No bogus results escape. Cross-validation catches disagreements, uncertainty is quantified honestly, and expert review handles edge cases. Confidence is earned, not assumed.

---

### **Group J: Agent Orchestration Symphony** 🎼
*"Conducting the autonomous intelligence ensemble"*

**Prerequisites:** Groups C, D, F, H, I complete (agents need all core systems)

**Parallel Tracks:**

1. **Track 1: Agent Foundation**
   - `agents/base.py` - Base agent class
   - Lifecycle management
   - Message passing interfaces
   - State persistence

2. **Track 2: Orchestrator Agent**
   - `agents/orchestrator/main.py` - Main orchestrator
   - `agents/orchestrator/delegation.py` - Task delegation
   - `agents/orchestrator/state.py` - State tracking
   - `agents/orchestrator/assembly.py` - Product assembly

3. **Track 3: Discovery Agent**
   - `agents/discovery/main.py` - Discovery orchestration
   - `agents/discovery/catalog.py` - Catalog querying
   - `agents/discovery/selection.py` - Dataset selection
   - `agents/discovery/acquisition.py` - Data acquisition

4. **Track 4: Pipeline Agent**
   - `agents/pipeline/main.py` - Pipeline orchestration
   - `agents/pipeline/assembly.py` - Pipeline assembly
   - `agents/pipeline/execution.py` - Execution management
   - `agents/pipeline/monitoring.py` - Progress tracking

5. **Track 5: Quality Agent**
   - `agents/quality/main.py` - QA orchestration
   - `agents/quality/validation.py` - Validation execution
   - `agents/quality/review.py` - Review management
   - `agents/quality/reporting.py` - QA reporting

6. **Track 6: Reporting Agent**
   - `agents/reporting/main.py` - Report orchestration
   - `agents/reporting/products.py` - Product generation
   - `agents/reporting/formats.py` - Format handling
   - `agents/reporting/delivery.py` - Distribution

7. **Track 7: Agent Tests**
   - `tests/test_agents.py`
   - Test agent lifecycle
   - Test message passing
   - Test delegation and coordination
   - Test end-to-end workflow

**Deliverables:**
- Agent framework
- Orchestrator agent
- 4 specialized agents (discovery, pipeline, quality, reporting)
- Inter-agent communication
- Test suite

**Success Criteria:**
- Agents start, execute, and shut down cleanly
- Message passing is reliable
- Orchestrator correctly delegates tasks
- Specialized agents handle their domains
- State persists across restarts
- End-to-end event processing succeeds
- `pytest tests/test_agents.py -v` passes

**Celebration Checkpoint:** 🎭
The ensemble performs! Agents coordinate autonomously, each handling their specialty. The orchestrator conducts the symphony, and from event specification to final product, the entire flow runs with minimal human intervention.

---

### **Group K: API Gateway & Deployment Launchpad** 🚀
*"Opening the portal to the multiverse"*

**Prerequisites:** Group J complete (needs full agent system)

**Parallel Tracks:**

1. **Track 1: FastAPI Application**
   - `api/main.py` - Application entry point
   - `api/config.py` - Configuration management
   - `api/dependencies.py` - Dependency injection
   - `api/middleware.py` - Middleware stack

2. **Track 2: API Routes** (all parallel)
   - `api/routes/events.py` - Event submission and retrieval
   - `api/routes/status.py` - Job status and monitoring
   - `api/routes/products.py` - Product download
   - `api/routes/catalog.py` - Data catalog browsing
   - `api/routes/health.py` - Health checks

3. **Track 3: API Models**
   - `api/models/requests.py` - Request schemas
   - `api/models/responses.py` - Response schemas
   - `api/models/errors.py` - Error handling

4. **Track 4: Security** (all parallel)
   - `api/auth.py` - Authentication
   - `api/rate_limit.py` - Rate limiting
   - `api/cors.py` - CORS configuration

5. **Track 5: Notifications**
   - `api/webhooks.py` - Webhook system
   - `api/notifications.py` - Notification dispatch

6. **Track 6: Deployment Configurations** (all parallel)
   - `deploy/serverless.yml` - Serverless framework config
   - `deploy/docker-compose.yml` - Local development
   - `deploy/kubernetes/` - K8s manifests (if needed)
   - `deploy/terraform/` - Infrastructure as code (if needed)

7. **Track 7: API Tests**
   - `tests/test_api.py`
   - Test all endpoints
   - Test authentication
   - Test error handling
   - Test webhooks
   - Load testing

8. **Track 8: Documentation**
   - OpenAPI/Swagger documentation
   - API client examples (Python, cURL, JavaScript)
   - Deployment guides

**Deliverables:**
- Full FastAPI application
- All API endpoints
- Authentication and security
- Webhook system
- Deployment configurations
- API documentation
- Test suite

**Success Criteria:**
- API endpoints handle requests correctly
- Authentication prevents unauthorized access
- Rate limiting prevents abuse
- Webhooks deliver notifications reliably
- Serverless deployment works
- API docs are complete and accurate
- `pytest tests/test_api.py -v` passes
- Load tests show acceptable performance

**Celebration Checkpoint:** 🌟
The portal is open! External systems can now submit events, track progress, and retrieve products via clean REST APIs. Webhooks provide real-time updates. Serverless deployment means the platform scales effortlessly with demand.

---

### **Group L: Lightweight Execution & CLI** 💻
*"Run anywhere, from laptop to cloud"*

**Prerequisites:** Group E complete (needs algorithms); can develop in parallel with Groups F-K

**Motivation:** The platform should work on a low-power laptop processing one tile at a time, not just on cloud servers with unlimited RAM. Users should be able to run incremental workflows from the command line without deploying a full API.

**Parallel Tracks:**

1. **Track 1: Tiled Processing Infrastructure**
   - `core/execution/tiling.py` - Tile grid generation and management
   - Define tile schemes (256x256, 512x512, configurable)
   - Overlap handling for edge effects
   - Tile coordinate systems (pixel, geo-referenced)
   - Progress tracking per tile

2. **Track 2: Streaming Data Ingestion**
   - `core/data/ingestion/streaming.py` - Chunked download and processing
   - Window-based raster reading (rasterio windowed reads)
   - Memory-mapped file support for large datasets
   - Resume capability for interrupted downloads
   - Bandwidth throttling for constrained networks

3. **Track 3: Algorithm Tiling Adapters**
   - `core/analysis/execution/tiled_runner.py` - Run algorithms tile-by-tile
   - Automatic algorithm wrapping for tile processing
   - Edge artifact handling (overlap, blending)
   - Result stitching and mosaic generation
   - Per-tile statistics aggregation

4. **Track 4: Memory-Efficient Algorithms**
   - Update existing algorithms to support optional chunked mode
   - `threshold_sar.py` - Add `process_tile()` method
   - `ndwi_optical.py` - Add windowed NDWI computation
   - `change_detection.py` - Tile-aware change detection
   - `hand_model.py` - Chunked DEM processing (critical: currently needs 8GB)
   - Wildfire/storm algorithms - Add tile support

5. **Track 5: CLI Framework**
   - `cli/main.py` - Main entry point using Click or Typer
   - `cli/commands/` - Command modules
   - Subcommands: `discover`, `ingest`, `analyze`, `validate`, `export`
   - Progress bars and status output
   - JSON/YAML output modes for scripting

6. **Track 6: Incremental Workflow Commands**
   - `cli/commands/discover.py` - Find available data for area/time
   - `cli/commands/ingest.py` - Download and normalize data (resumable)
   - `cli/commands/analyze.py` - Run analysis (can specify tile range)
   - `cli/commands/validate.py` - Run quality checks
   - `cli/commands/export.py` - Generate outputs (GeoTIFF, GeoJSON, report)
   - `cli/commands/run.py` - Full pipeline in one command

7. **Track 7: Execution Profiles**
   - `core/execution/profiles.py` - Predefined resource configurations
   - `laptop` profile: 2GB RAM, sequential processing, small tiles
   - `workstation` profile: 8GB RAM, parallel tiles, medium tiles
   - `cloud` profile: 32GB+ RAM, distributed processing, large tiles
   - Auto-detection of available resources
   - User-configurable limits

8. **Track 8: State Persistence**
   - `core/execution/state.py` - Save/resume workflow state
   - Checkpoint after each processing stage
   - SQLite or JSON state files
   - Resume from last successful step
   - State inspection commands

9. **Track 9: Local Storage Backend**
   - `core/data/storage/local.py` - Filesystem-based storage
   - Organized directory structure for intermediate products
   - Cleanup utilities for temporary files
   - Disk space estimation before processing
   - Works offline after initial data download

10. **Track 10: CLI Tests**
    - `tests/test_cli.py` - Test all CLI commands
    - `tests/test_tiling.py` - Test tile processing
    - `tests/test_streaming.py` - Test chunked ingestion
    - Integration tests for full incremental workflow

**CLI Usage Examples:**

```bash
# Discover available data
mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood

# Download data incrementally (can Ctrl+C and resume)
mdive ingest --area miami.geojson --source sentinel1 --output ./data/

# Run analysis on specific tiles (for testing or partial processing)
mdive analyze --input ./data/ --algorithm sar_threshold --tiles 0-10 --output ./results/

# Run full pipeline with laptop profile
mdive run --area miami.geojson --event flood --profile laptop --output ./products/

# Check status of interrupted workflow
mdive status --workdir ./products/

# Resume interrupted workflow
mdive resume --workdir ./products/
```

**Deliverables:**
- Tiled processing infrastructure
- Memory-efficient algorithm adapters
- Full CLI with incremental workflow support
- Execution profiles for different hardware
- State persistence for resume capability
- Local storage backend

**Success Criteria:**
- Full flood analysis runs on 4GB RAM laptop (may be slow, but works)
- Processing can be interrupted and resumed
- CLI provides clear progress feedback
- Each stage can run independently
- Results match full-memory processing (within floating-point tolerance)
- `pytest tests/test_cli.py tests/test_tiling.py -v` passes

**Celebration Checkpoint:** 🖥️
The platform now runs anywhere! From a Raspberry Pi processing one tile at a time to a cloud cluster processing hundreds in parallel, the same codebase adapts. Users can work incrementally, pause and resume, and see exactly what's happening at each step.

---

**Components Requiring Updates for Lightweight Mode:**

The following existing components need modifications to support tiled/lightweight execution:

| Component | Current State | Required Changes |
|-----------|---------------|------------------|
| `threshold_sar.py` | Full array (4GB) | Add `process_tile()` method |
| `ndwi_optical.py` | Full array (4GB) | Add windowed computation |
| `change_detection.py` | Full array (4GB) | Tile-aware differencing |
| `hand_model.py` | Full array (8GB) | **Critical**: Chunked DEM, streaming flow accumulation |
| `thermal_anomaly.py` | Full array (0.5GB) | Add tile support with context window |
| `wind_damage.py` | Full array (2GB) | Add tile support |
| `structural_damage.py` | Full array (2GB) | Add tile support |
| `core/data/broker.py` | Returns full datasets | Add tile-aware queries |
| `core/analysis/library/registry.py` | No execution | Add tiled execution methods |

---

---

## Bug Fixes Required (Before Continuing Development)

> **Full details in [FIXES.md](./FIXES.md)** - Each fix includes exact code changes, context, and verification steps.

Before proceeding with Groups F-K, the following critical bugs must be fixed. These were discovered during code review and will cause runtime errors.

### P0: Critical Fixes (Will Crash at Runtime)

| Fix ID | File | Line | Issue | Parallel Group |
|--------|------|------|-------|----------------|
| **FIX-001** | `core/data/broker.py` | 455 | Calls `.get()` on Provider dataclass (not dict) | Can fix in parallel |
| **FIX-002** | `core/data/broker.py` | 152 | References non-existent `candidates` attribute | Can fix in parallel |
| **FIX-003** | `core/data/discovery/wms_wcs.py` | 379-380 | Duplicate dict key breaks WCS queries | Can fix in parallel |
| **FIX-004** | `core/analysis/.../hand_model.py` | 305 | `ndimage.grey_erosion()` doesn't exist | Can fix in parallel |
| **FIX-005** | `core/analysis/.../hand_model.py` | 378-382 | Wrong `distance_transform_edt` parameters | Fix with FIX-004 |
| **FIX-006** | `openspec/schemas/provenance.schema.json` | 112 | Broken `$ref` to non-existent definition | Can fix in parallel |

### P1: Medium Priority (Logic Bugs)

| Fix ID | File | Line | Issue | Parallel Group |
|--------|------|------|-------|----------------|
| **FIX-007** | `core/intent/classifier.py` | 206-208 | Classification bias toward deeper classes | Can fix in parallel |
| **FIX-008** | `core/intent/resolver.py` | 8 | Python 3.11+ only import (`datetime.UTC`) | Can fix in parallel |
| **FIX-009** | `core/data/broker.py` | 125 | Deprecated `datetime.utcnow()` | Fix with FIX-001/002 |
| **FIX-010** | `core/analysis/.../hand_model.py` | 310-342 | Stub D8 flow accumulation algorithm | Fix with FIX-004/005 |
| **FIX-011** | `core/data/discovery/provider_api.py` | 250 | Import statement inside class body | Can fix in parallel |

### P2: Low Priority (Style/Best Practice)

| Fix ID | File | Issue |
|--------|------|-------|
| **FIX-012** | `core/intent/resolver.py:15-16` | Library calls `logging.basicConfig()` |
| **FIX-013** | `core/intent/registry.py:108-109` | Hardcoded `parent.parent.parent` path |
| **FIX-014** | Multiple discovery files | Unnecessary `hasattr(provider, "cost")` checks |
| **FIX-015** | `core/data/providers/registry.py:68` | Empty `_load_default_providers()` stub |
| **FIX-016** | Multiple schema files | Inconsistent `confidence_score` definitions |

### Fix Execution Strategy

**Parallel Batch 1** (All P0 fixes - can run simultaneously):
```
FIX-001, FIX-002, FIX-003 → broker.py and wms_wcs.py fixes
FIX-004, FIX-005, FIX-010 → hand_model.py scipy fixes (do together)
FIX-006 → Add processing_level to common.schema.json
```

**Parallel Batch 2** (P1 fixes - after Batch 1):
```
FIX-007 → classifier.py depth bonus adjustment
FIX-008, FIX-009 → datetime compatibility fixes
FIX-011 → provider_api.py import cleanup
```

**Parallel Batch 3** (P2 fixes - when convenient):
```
FIX-012 through FIX-016 → Style and consistency fixes
```

### Verification After Fixes

```bash
# Run full test suite after each batch
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Expected: 277+ tests passing, 0 errors
```

---

## Tech Debt Backlog (Ongoing)

The following issues were identified during earlier reviews and should be addressed during ongoing development.

### Previously Identified Critical Issues

| Issue | Location | Description |
|-------|----------|-------------|
| Division by zero risk | `change_detection.py:293` | Ratio method doesn't properly guard against zero denominators |
| Shell injection vulnerability | `parallel_orchestrator.sh:119-141` | Unquoted variables with `--dangerously-skip-permissions` |
| Class name mismatches | `baseline_algorithms.yaml:59,132` | YAML `class_name` fields don't match actual Python class names |

### Previously Identified High Priority

| Issue | Location | Description |
|-------|----------|-------------|
| Thread-unsafe singleton | `registry.py:464-497` | Global registry needs `threading.Lock()` for parallel execution |
| Lock file race conditions | `parallel_orchestrator.sh:108-110` | Multiple agents could claim same work simultaneously |
| Unimplemented algorithms | `baseline_algorithms.yaml:257-540` | Wildfire/storm algorithms declared but code doesn't exist |
| No registry caching | `registry.py:231` | O(n) search on every algorithm lookup |

### Design Improvements

| Area | Recommendation |
|------|----------------|
| **Shell → Python** | Migrate orchestrators to Python for safety, testability, and proper process management |
| **Shared Utilities** | Create `core/analysis/utils/` for common functions (confidence scoring, masking, validation) |
| **Base Algorithm Class** | Add abstract base class for algorithms to enforce interface consistency |
| **Documentation** | Add algorithm complexity estimates, runtime requirements, and paper citations to YAML |

---

## Agent Code Review Checklist

All agents MUST complete this checklist during the Review phase before marking work as done.

### 1. Correctness & Safety
- [ ] Division operations guarded against zero
- [ ] Array indexing validated for bounds
- [ ] NaN/Inf handling with `np.isnan()`, `np.isinf()`
- [ ] Edge cases tested (empty arrays, single elements, all-same values)
- [ ] Shell variables quoted: `"$VAR"` not `$VAR`
- [ ] No hardcoded credentials or API keys

### 2. Consistency
- [ ] Names match across files (class names in YAML match Python)
- [ ] Default values in code match YAML/spec defaults
- [ ] Error handling patterns match rest of codebase
- [ ] Import paths in registry match actual file locations

### 3. Completeness
- [ ] All declared features implemented (no YAML entries for unwritten code)
- [ ] Every public function has at least one test
- [ ] Error paths tested, not just happy path
- [ ] Docstrings present on public classes/functions
- [ ] Type hints on function signatures

### 4. Robustness
- [ ] Specific exceptions caught (no bare `except:`)
- [ ] Resources cleaned up in finally blocks
- [ ] Thread safety considered for global state
- [ ] Atomic operations for locks (`fcntl.flock()` not pid files)
- [ ] Graceful degradation for partial failures

### 5. Performance
- [ ] No O(n²) loops on large data without justification
- [ ] Expensive computations cached if reused
- [ ] Large arrays processed in chunks if memory-constrained
- [ ] Lazy loading where appropriate

### 6. Security
- [ ] External input validated before use
- [ ] No dangerous flags (`--dangerously-skip-permissions` removed for production)
- [ ] Minimal permissions requested
- [ ] Secrets not logged in debug output

### 7. Maintainability
- [ ] Magic numbers extracted to named constants
- [ ] No code duplication (shared logic extracted)
- [ ] Single responsibility per function
- [ ] Clear naming (variables describe content, functions describe action)

---

## Documentation Task: Directory READMEs

Each major directory should have a brief, human-readable README explaining its purpose. These can be created incrementally by agents as they work on each area.

### Directories Needing READMEs

| Directory | Purpose | Priority |
|-----------|---------|----------|
| `core/` | Core processing logic overview | High |
| `core/intent/` | Event type classification and NLP resolution | High |
| `core/data/` | Data discovery, providers, and ingestion | High |
| `core/data/providers/` | Satellite and ancillary data provider implementations | Medium |
| `core/data/discovery/` | STAC, WMS/WCS, and API discovery adapters | Medium |
| `core/analysis/` | Algorithm library and pipeline assembly | High |
| `core/analysis/library/` | Algorithm registry and baseline implementations | High |
| `core/analysis/library/baseline/flood/` | Flood detection algorithms | Medium |
| `core/analysis/library/baseline/wildfire/` | Wildfire/burn area algorithms | Medium |
| `core/analysis/library/baseline/storm/` | Storm damage algorithms | Medium |
| `openspec/` | Specification layer overview | High |
| `openspec/schemas/` | JSON Schema definitions | Medium |
| `openspec/definitions/` | YAML event classes, algorithms, data sources | Medium |
| `agents/` | Autonomous agent implementations | Medium |
| `api/` | FastAPI REST interface | Medium |
| `tests/` | Test suite organization and conventions | Medium |
| `examples/` | Example event specifications | Low |

### README Template

Each README should be brief (50-150 words) and include:

```markdown
# [Directory Name]

**Purpose**: One sentence explaining what this directory contains.

## Contents

- `file1.py` - Brief description
- `file2.py` - Brief description
- `subdir/` - Brief description

## Key Concepts

2-3 bullet points explaining the main ideas or patterns used here.

## Usage

Brief example or pointer to where this code is used.
```

### When to Create

- **During implementation**: When you complete a track, add a README to directories you created
- **During review**: If reviewing code without a README, add one
- **Standalone task**: Can be done as a documentation pass between groups

---

## Advanced Enhancements (Post-Launch Upgrades)

Once the core multiverse is stable, consider these expansion packs:

### **Enhancement Alpha: Machine Learning Ascension**
- Replace rule-based NLP classifier with transformer models
- Add ML-based algorithm selector trained on historical performance
- Implement advanced algorithms (U-Net segmentation, ensemble fusion)
- Active learning for algorithm improvement

### **Enhancement Beta: Temporal Intelligence**
- Time-series analysis for change detection
- Predictive modeling for hazard evolution
- Seasonal pattern learning
- Anomaly detection in temporal sequences

### **Enhancement Gamma: User Experience Nexus**
- Web-based UI for event submission
- Interactive map viewer for products
- Expert review portal with annotation tools
- Dashboard for system health and job monitoring

### **Enhancement Delta: Commercial Sensor Integration**
- Planet Labs high-resolution optical
- ICEYE/Capella X-band SAR
- Maxar/Airbus very-high-resolution optical
- Cost optimization for commercial data

### **Enhancement Epsilon: Specialized Hazards**
- Earthquake damage assessment
- Landslide detection
- Volcanic activity monitoring
- Drought monitoring

### **Enhancement Zeta: Real-Time Streaming**
- WebSocket support for live updates
- Real-time sensor feeds
- Continuous processing pipelines
- Nowcasting integration

---

## Test Runner Guide

A modular test runner is available for quick, targeted testing during development.

### Quick Start

```bash
./run_tests.py --list             # Show all categories and options
./run_tests.py                    # Run all tests (518 total)
```

### By Hazard Type

```bash
./run_tests.py flood              # Flood detection tests (~52 tests)
./run_tests.py wildfire           # Wildfire/burn tests (~86 tests)
./run_tests.py storm              # Storm damage tests (~62 tests)
```

### By Component

```bash
./run_tests.py schemas            # JSON schema validation
./run_tests.py intent             # Intent resolution & classification
./run_tests.py providers          # Data provider implementations
./run_tests.py registry           # Algorithm registry
```

### By Specific Algorithm

```bash
./run_tests.py --algorithm sar        # SAR backscatter threshold
./run_tests.py --algorithm ndwi       # NDWI optical flood detection
./run_tests.py --algorithm hand       # Height Above Nearest Drainage
./run_tests.py --algorithm dnbr       # Differenced NBR burn severity
./run_tests.py --algorithm thermal    # Thermal anomaly detection
./run_tests.py --algorithm wind       # Wind damage assessment
./run_tests.py --algorithm structural # Structural damage assessment
```

### Development Workflow

When implementing a new feature:

1. **Start with tests**: `./run_tests.py <hazard> --quick` to verify baseline
2. **Develop iteratively**: Run specific algorithm tests as you code
3. **Verify no regressions**: `./run_tests.py` full suite before committing

### Adding New Tests

Tests are auto-marked based on file and test names:
- File named `test_wildfire_*.py` → automatically gets `@pytest.mark.wildfire`
- Test named `test_flood_*` → automatically gets `@pytest.mark.flood`

Common fixtures available in `tests/conftest.py`:
- `sample_dem` - 100x100 terrain with river valley
- `sample_sar_image` - SAR backscatter with water region
- `sample_optical_bands` - Green/NIR bands for NDWI
- `sample_event_spec` - Valid event specification dict

---

## Verification Checkpoints Throughout the Journey

As you progress through each group:

1. **After Group A:**
   ```bash
   python -m json.tool openspec/schemas/event.schema.json
   yamllint openspec/definitions/event_classes/
   ```

2. **After Group B:**
   ```bash
   ./run_tests.py schemas
   python -m openspec.validator examples/flood_event.yaml
   ```

3. **After Group C:**
   ```bash
   ./run_tests.py intent
   python -m core.intent.resolver "flooding after hurricane in Miami"
   ```

4. **After Group D:**
   ```bash
   ./run_tests.py providers
   python -m core.data.broker --event examples/flood_event.yaml
   ```

5. **After Group E:**
   ```bash
   ./run_tests.py flood wildfire storm    # All hazard algorithms
   ./run_tests.py registry                 # Algorithm registry
   ```

6. **After Group F:**
   ```bash
   ./run_tests.py --file selection
   ```

7. **After Group G:**
   ```bash
   ./run_tests.py --file ingestion
   gdalinfo output.tif  # Verify COG structure
   ```

8. **After Group H:**
   ```bash
   ./run_tests.py --file fusion
   ```

9. **After Group I:**
   ```bash
   ./run_tests.py --file quality
   ```

10. **After Group J:**
    ```bash
    ./run_tests.py --file agents
    python -m agents.orchestrator.main examples/flood_event.yaml
    ```

11. **After Group K (Full System!):**
    ```bash
    ./run_tests.py                        # All 518+ tests
    uvicorn api.main:app --reload &
    curl -X POST http://localhost:8000/events \
      -H "Content-Type: application/yaml" \
      --data-binary @examples/flood_event.yaml

    # Check status
    curl http://localhost:8000/events/{event_id}/status

    # Deploy
    cd deploy && serverless deploy --stage dev
    ```

---

## Principles for the Journey

1. **Test as You Build:** Don't wait until the end. Each group includes its tests.

2. **Parallelize Fearlessly:** Within each group, tracks can run simultaneously. Embrace concurrency!

3. **Celebrate Small Wins:** Each checkpoint unlocks new capabilities. Acknowledge progress.

4. **Documentation is Code:** Schemas, examples, and tests ARE documentation. Keep them pristine.

5. **Fail Fast, Learn Faster:** Quality checks catch issues early. Embrace failure as feedback.

6. **Reproducibility is Sacred:** Deterministic selections, version pinning, and provenance tracking are non-negotiable.

7. **User Empathy:** Remember that someone in an emergency will use this system. Clear errors, helpful defaults, and reliability matter.

8. **Future-Proof Extensibility:** New sensors, algorithms, and hazard types should slot in easily.

---

## What Success Looks Like

When you reach the end of this roadmap:

- A user submits: `"flooding in Miami after Hurricane XYZ, September 15-20"`
- The system:
  - ✅ Understands intent (coastal storm surge flood)
  - ✅ Discovers optimal datasets (Sentinel-1 SAR, weather forecasts, DEM)
  - ✅ **Detects 90% cloud cover → automatically switches to SAR-only mode**
  - ✅ Ingests and normalizes data into cloud-native formats
  - ✅ Selects appropriate algorithms (SAR threshold + change detection)
  - ✅ **No pre-event baseline found → falls back to static water mask + anomaly detection**
  - ✅ Fuses multi-sensor observations with quality checks
  - ✅ Generates validated flood extent with uncertainty
  - ✅ **Flags degraded mode in provenance: "partial optical coverage, fallback to SAR"**
  - ✅ Produces GeoJSON, COG, and PDF report
  - ✅ Sends webhook notification upon completion
  - ✅ Full provenance from input to output
  - ✅ **Deploys instantly: `docker-compose up` or `kubectl apply`**
  - ✅ **Runs anywhere: AWS Lambda, edge device, or laptop**

All of this happens autonomously, reproducibly, and transparently—with graceful degradation when data is imperfect.

---

## Final Words from Mission Control

This roadmap is your star map. Each group is a waypoint, each celebration a milestone. There are no deadlines—only destinations. The multiverse is vast, complex, and beautiful. Build with curiosity, test with rigor, and celebrate every working component.

The cosmos awaits your exploration. Let's dive! 🌌✨


### **Group M: Resilience & Fallback Systems** 🛟
*"When plan A fails, plans B through Z kick in automatically"*

**Prerequisites:** Groups F, H, I complete (needs selection, fusion, and quality systems)

**Philosophy:** In emergency response, "I don't know" is unacceptable. The system should degrade gracefully, trying every possible approach before admitting defeat. Document what was tried, why it failed, and what workaround was used.

**Parallel Tracks:**

1. **Track 1: Data Quality Assessment & Fallbacks**
   - `core/resilience/assessment/optical_quality.py` - Cloud cover, haze, shadows
   - `core/resilience/assessment/sar_quality.py` - Speckle noise, geometric distortion
   - `core/resilience/assessment/dem_quality.py` - Voids, artifacts, resolution
   - `core/resilience/assessment/temporal_quality.py` - Pre-event baseline availability
   - Quality scoring with confidence intervals
   - Automated quality-based fallback triggers

2. **Track 2: Sensor Fallback Chains**
   - `core/resilience/fallbacks/optical_fallback.py`
     - Primary: Sentinel-2 → Landsat-8 → MODIS
     - Cloud cover > 80% → switch to SAR
     - Cloud cover 40-80% → use cloud-free pixels only
   - `core/resilience/fallbacks/sar_fallback.py`
     - High noise → apply enhanced filtering (Lee, Frost, Gamma MAP)
     - Geometric distortion → switch to different orbit/mode
     - C-band → X-band or L-band depending on target
   - `core/resilience/fallbacks/dem_fallback.py`
     - Copernicus DEM → SRTM → ASTER → interpolated from contours
     - Void filling strategies
   - Fallback decision trees with provenance logging

3. **Track 3: Algorithm Fallback Strategies**
   - `core/resilience/fallbacks/algorithm_fallback.py`
     - No pre-event baseline → use static water masks + anomaly detection
     - Insufficient SAR → optical-only with cloud masking
     - No optical or SAR → use forecasts + terrain analysis
     - Algorithm failure → try alternative algorithm with different parameters
   - `core/resilience/fallbacks/parameter_tuning.py`
     - Adaptive threshold adjustment when results are suspect
     - Automatic parameter grid search if primary params fail
     - Per-region parameter optimization

4. **Track 4: Degraded Mode Operations**
   - `core/resilience/degraded_mode/mode_manager.py`
     - Define degraded mode levels (FULL → PARTIAL → MINIMAL → EMERGENCY)
     - Automatic mode switching based on data availability
     - User notification of degraded mode operation
   - `core/resilience/degraded_mode/partial_coverage.py`
     - Handle incomplete spatial coverage
     - Mosaic from multiple partial acquisitions
     - Extrapolate/interpolate missing areas with uncertainty
   - `core/resilience/degraded_mode/low_confidence.py`
     - Ensemble methods when single algorithm confidence is low
     - Multiple algorithm voting
     - Flag outputs for manual review when all methods disagree

5. **Track 5: Failure Documentation & Recovery**
   - `core/resilience/failure/failure_log.py`
     - Structured failure logging (what failed, why, fallback used)
     - Failure analysis for system improvement
   - `core/resilience/failure/recovery_strategies.py`
     - Retry logic with exponential backoff
     - Alternative data source discovery
     - Graceful degradation with user communication
   - `core/resilience/failure/provenance_tracking.py`
     - Full audit trail of fallback decisions
     - Confidence scoring based on fallback depth
     - User-facing explanations of limitations

6. **Track 6: Resilience Tests**
   - `tests/test_resilience.py`
   - Test all fallback chains with simulated failures
   - Test degraded mode operations
   - Test failure documentation
   - Verify provenance tracking through fallbacks

**Deliverables:**
- Quality assessment modules for all sensor types
- Comprehensive fallback chains
- Degraded mode operation system
- Failure logging and recovery
- Test suite covering all failure scenarios

**Success Criteria:**
- System never returns empty result without trying all fallbacks
- Each fallback decision is logged with rationale
- Confidence scores adjust based on fallback depth
- Users receive clear explanations of data limitations
- Tests simulate 20+ failure scenarios
- `pytest tests/test_resilience.py -v` passes

**Celebration Checkpoint:** 🏥
Your system is now antifragile! Cloud cover, noise, missing baselines, sensor failures—none of these stop the analysis. The system tries every trick in the book, documents what it tried, and delivers the best possible result given the circumstances. Emergency responders always get an answer, even if it's "here's what we know with these limitations."

---

### **Group N: Containerization & Multi-Environment Deployment** 🐳
*"Build once, run anywhere—from Raspberry Pi to AWS Lambda"*

**Prerequisites:** Groups F-K complete (needs full system), Group L recommended (benefits from CLI)

**Philosophy:** Deployment should be trivial. Whether you're running on a laptop, deploying to Kubernetes, or spinning up serverless functions, the same Docker images work everywhere. Modular containers mean you only deploy what you need.

**Parallel Tracks:**

1. **Track 1: Modular Dockerfile Architecture**
   - `docker/base/Dockerfile` - Base image (GDAL, rasterio, Python deps)
   - `docker/core/Dockerfile` - Core processing (intent, algorithms, analysis)
   - `docker/api/Dockerfile` - API service (FastAPI + agent orchestrator)
   - `docker/cli/Dockerfile` - CLI tools (lightweight, no API)
   - `docker/worker/Dockerfile` - Background worker (pipeline execution)
   - Multi-stage builds for minimal image sizes
   - ARM64 + x86_64 builds for cross-platform compatibility

2. **Track 2: Docker Compose Orchestration**
   - `docker-compose.yml` - Full stack (API + workers + Redis + PostgreSQL)
   - `docker-compose.dev.yml` - Development mode (hot reload, debug ports)
   - `docker-compose.minimal.yml` - Minimal stack (CLI only, no API)
   - Service networking and health checks
   - Volume mounts for data persistence
   - Environment variable configuration

3. **Track 3: Cloud Deployment Configurations**
   - **AWS**:
     - `deploy/aws/ecs/task-definition.json` - ECS task definitions
     - `deploy/aws/lambda/serverless.yml` - Lambda functions for API
     - `deploy/aws/batch/job-definition.json` - Batch processing for large jobs
   - **Google Cloud**:
     - `deploy/gcp/cloud-run/service.yaml` - Cloud Run services
     - `deploy/gcp/kubernetes/deployment.yaml` - GKE deployment
   - **Azure**:
     - `deploy/azure/container-instances/deployment.json` - ACI
     - `deploy/azure/aks/deployment.yaml` - AKS deployment

4. **Track 4: Kubernetes Manifests**
   - `deploy/kubernetes/namespace.yaml` - Namespace isolation
   - `deploy/kubernetes/deployments/api.yaml` - API deployment
   - `deploy/kubernetes/deployments/worker.yaml` - Worker deployment
   - `deploy/kubernetes/services/` - Service definitions
   - `deploy/kubernetes/ingress.yaml` - Ingress configuration
   - `deploy/kubernetes/configmaps/` - Configuration management
   - `deploy/kubernetes/secrets/` - Secret management
   - `deploy/kubernetes/hpa.yaml` - Horizontal pod autoscaling
   - `deploy/kubernetes/persistentvolumes/` - Storage configuration

5. **Track 5: On-Prem & Edge Deployment**
   - `deploy/on-prem/standalone/docker-compose.yml` - Single-server deployment
   - `deploy/on-prem/cluster/ansible-playbook.yml` - Multi-node cluster setup
   - `deploy/edge/arm64/Dockerfile.lightweight` - Raspberry Pi / edge devices
   - `deploy/edge/nvidia-jetson/Dockerfile.gpu` - NVIDIA Jetson for GPU acceleration
   - Resource-constrained configurations
   - Offline operation support
   - Local data caching strategies

6. **Track 6: CI/CD & Image Management**
   - `.github/workflows/docker-build.yml` - Automated image builds
   - `.github/workflows/docker-push.yml` - Push to registries (Docker Hub, ECR, GCR)
   - `.github/workflows/deploy-staging.yml` - Deploy to staging
   - `.github/workflows/deploy-production.yml` - Deploy to production
   - `scripts/build-images.sh` - Local build script
   - `scripts/push-images.sh` - Push to registry
   - Version tagging strategy (semantic versioning)
   - Multi-architecture manifest creation

7. **Track 7: Configuration Management**
   - `deploy/config/base.env` - Base configuration
   - `deploy/config/production.env` - Production overrides
   - `deploy/config/development.env` - Development overrides
   - Environment-specific configuration loading
   - Secrets management (Vault, AWS Secrets Manager, etc.)
   - Runtime configuration validation

8. **Track 8: Deployment Tests & Validation**
   - `tests/test_docker_builds.py` - Test all Dockerfiles build successfully
   - `tests/test_docker_compose.py` - Test compose stacks start correctly
   - `tests/test_k8s_manifests.py` - Validate Kubernetes YAML
   - `tests/test_deployment_smoke.py` - Smoke tests for deployed services
   - Health check endpoints
   - Readiness probes
   - Liveness probes

**Deliverables:**
- Modular Docker images for all components
- Docker Compose configurations for local/dev/prod
- Cloud deployment templates (AWS, GCP, Azure)
- Kubernetes manifests with autoscaling
- Edge/on-prem deployment guides
- CI/CD pipelines for automated deployment
- Configuration management system

**Success Criteria:**
- `docker-compose up` brings up full stack in < 2 minutes
- Images support both x86_64 and ARM64
- API container < 500MB, CLI container < 300MB
- Kubernetes deployment scales from 1 to 100+ pods
- Edge deployment runs on Raspberry Pi 4 (4GB RAM)
- CI/CD automatically builds and deploys on merge to main
- All deployment targets have smoke tests
- `pytest tests/test_docker*.py tests/test_deployment*.py -v` passes

**Celebration Checkpoint:** 🚢
Your platform is now truly portable! Docker images run identically everywhere. Deploy to AWS Lambda for serverless, Kubernetes for massive scale, or a Raspberry Pi at a remote field site. One codebase, one set of containers, infinite deployment targets. From cloud to edge, the multiverse is accessible to all.

---

**Refactoring Notes for Existing Groups:**

The following groups should be enhanced to support failure-first behavior:

| Group | Refactoring Needed | References Group M |
|-------|-------------------|-------------------|
| **Group F** | Add quality-based triggers for fallback chains | Track 1, Track 2 |
| **Group H** | Add algorithm fallback on fusion failure | Track 3 |
| **Group I** | Add degraded mode detection triggers | Track 4 |

---

