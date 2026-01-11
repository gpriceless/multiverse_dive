"""
Ingestion pipeline for transforming raw data into cloud-native formats.

This module provides:
- Pipeline orchestration with job management
- Format converters (COG, Zarr, GeoParquet)
- Normalization tools (projection, tiling, resolution)
- Data enrichment (overviews, statistics, quality)
- Validation (integrity, anomaly, completeness)
- Persistence with lineage tracking
"""

# Pipeline orchestration (Track 1 - may not be implemented yet)
try:
    from core.data.ingestion.pipeline import (
        IngestionJob,
        IngestionConfig,
        IngestionResult,
        IngestionPipeline,
        JobManager,
        JobStatus,
        IngestionStage,
    )
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

# Persistence layer (Track 6)
from core.data.ingestion.persistence import (
    # Storage
    ChecksumAlgorithm,
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    StorageConfig,
    StorageObject,
    StorageType,
    UploadResult,
    create_storage_backend,
    parse_storage_uri,
    # Product Management
    ProductInfo,
    ProductManager,
    ProductManagerConfig,
    ProductStatus,
    ProductType,
    # Lineage
    AlgorithmInfo,
    InputDataset,
    LineageEventType,
    LineageTracker,
    ProcessingStep,
    ProvenanceRecord,
    QualitySummary,
    StepContext,
    TrackingContext,
    compute_environment_hash,
    create_provenance_from_job,
    get_compute_resources,
    get_software_environment,
)

__all__ = [
    # Storage
    "ChecksumAlgorithm",
    "LocalStorageBackend",
    "S3StorageBackend",
    "StorageBackend",
    "StorageConfig",
    "StorageObject",
    "StorageType",
    "UploadResult",
    "create_storage_backend",
    "parse_storage_uri",
    # Product Management
    "ProductInfo",
    "ProductManager",
    "ProductManagerConfig",
    "ProductStatus",
    "ProductType",
    # Lineage
    "AlgorithmInfo",
    "InputDataset",
    "LineageEventType",
    "LineageTracker",
    "ProcessingStep",
    "ProvenanceRecord",
    "QualitySummary",
    "StepContext",
    "TrackingContext",
    "compute_environment_hash",
    "create_provenance_from_job",
    "get_compute_resources",
    "get_software_environment",
]

# Add pipeline exports if available
if _PIPELINE_AVAILABLE:
    __all__.extend([
        "IngestionJob",
        "IngestionConfig",
        "IngestionResult",
        "IngestionPipeline",
        "JobManager",
        "JobStatus",
        "IngestionStage",
    ])
