"""
Persistence Layer for Ingestion Pipeline.

Provides comprehensive persistence capabilities for ingested data:
- Storage Backends: Local filesystem and cloud storage (S3, GCS, Azure)
- Product Management: Intermediate product lifecycle management
- Lineage Tracking: Full provenance from inputs to outputs

Usage:
    from core.data.ingestion.persistence import (
        # Storage
        StorageBackend,
        LocalStorageBackend,
        S3StorageBackend,
        StorageConfig,
        StorageType,
        create_storage_backend,

        # Product Management
        ProductManager,
        ProductManagerConfig,
        ProductType,
        ProductStatus,
        ProductInfo,

        # Lineage
        LineageTracker,
        ProvenanceRecord,
        InputDataset,
        AlgorithmInfo,
        ProcessingStep,
        QualitySummary,
    )
"""

from core.data.ingestion.persistence.storage import (
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
)

from core.data.ingestion.persistence.intermediate import (
    ProductInfo,
    ProductManager,
    ProductManagerConfig,
    ProductStatus,
    ProductType,
)

from core.data.ingestion.persistence.lineage import (
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
