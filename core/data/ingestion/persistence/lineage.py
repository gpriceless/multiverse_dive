"""
Lineage Tracking for Ingestion Pipeline.

Provides comprehensive provenance and lineage tracking for data products:
- Full processing chain from inputs to outputs
- Algorithm and parameter tracking
- Software environment capture
- Resource usage monitoring
- Reproducibility support with hashes

Implements the provenance.schema.json specification for interoperability.
"""

import hashlib
import json
import logging
import platform
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class LineageEventType(Enum):
    """Types of lineage events."""

    INPUT_REGISTERED = "input_registered"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    OUTPUT_CREATED = "output_created"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    QUALITY_ASSESSED = "quality_assessed"


@dataclass
class InputDataset:
    """
    Input dataset metadata for lineage.

    Attributes:
        dataset_id: Unique dataset identifier
        provider: Data provider name
        data_type: Data type category (optical, sar, dem, etc.)
        acquisition_time: When data was acquired
        processing_level: Processing level (L1C, L2A, etc.)
        uri: Location URI
        checksum: Data checksum
        checksum_algorithm: Checksum algorithm
        spatial_extent: Bounding box [west, south, east, north]
        temporal_extent: Time range
        bands: List of band names
        resolution_m: Spatial resolution in meters
        metadata: Additional metadata
    """

    dataset_id: str
    provider: str
    data_type: str
    acquisition_time: Optional[datetime] = None
    processing_level: Optional[str] = None
    uri: Optional[str] = None
    checksum: Optional[str] = None
    checksum_algorithm: str = "sha256"
    spatial_extent: Optional[List[float]] = None
    temporal_extent: Optional[Dict[str, str]] = None
    bands: List[str] = field(default_factory=list)
    resolution_m: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "provider": self.provider,
            "data_type": self.data_type,
            "acquisition_time": self.acquisition_time.isoformat()
            if self.acquisition_time
            else None,
            "processing_level": self.processing_level,
            "uri": self.uri,
            "checksum": {
                "algorithm": self.checksum_algorithm,
                "value": self.checksum,
            }
            if self.checksum
            else None,
            "spatial_extent": self.spatial_extent,
            "temporal_extent": self.temporal_extent,
            "bands": self.bands,
            "resolution_m": self.resolution_m,
            "metadata": self.metadata,
        }


@dataclass
class AlgorithmInfo:
    """
    Algorithm information for lineage.

    Attributes:
        algorithm_id: Unique algorithm identifier
        name: Algorithm name
        version: Version string
        category: Category (baseline, advanced, experimental)
        purpose: Description of purpose
        parameters: Parameters used
        class_name: Python class name
        module_path: Python module path
    """

    algorithm_id: str
    name: str
    version: str
    category: str = "baseline"
    purpose: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    class_name: Optional[str] = None
    module_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "purpose": self.purpose,
            "parameters": self.parameters,
            "class_name": self.class_name,
            "module_path": self.module_path,
        }


@dataclass
class ProcessingStep:
    """
    A single processing step in the lineage chain.

    Attributes:
        step: Step name/identifier
        timestamp: When step was executed
        inputs: Input product/dataset IDs
        outputs: Output product IDs
        agent: Agent that executed this step
        processor: Algorithm/processor used
        processor_version: Version of processor
        parameters: Processing parameters
        software_environment: Software versions
        execution_time_seconds: Wall-clock time
        compute_resources: Resources used
        status: Step status
        error_message: Error if failed
    """

    step: str
    timestamp: datetime
    inputs: List[str]
    outputs: List[str]
    agent: Optional[str] = None
    processor: Optional[str] = None
    processor_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    software_environment: Dict[str, str] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    compute_resources: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "agent": self.agent,
            "processor": self.processor,
            "processor_version": self.processor_version,
            "parameters": self.parameters,
            "software_environment": self.software_environment,
            "execution_time_seconds": self.execution_time_seconds,
            "compute_resources": self.compute_resources,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class QualitySummary:
    """
    Quality assessment summary for lineage.

    Attributes:
        overall_confidence: Overall confidence score (0-1)
        uncertainty_percent: Uncertainty as percentage
        plausibility_score: Plausibility score (0-1)
        validation_method: Validation method used
        quality_flags: List of quality flags
    """

    overall_confidence: float = 1.0
    uncertainty_percent: float = 0.0
    plausibility_score: float = 1.0
    validation_method: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_confidence": self.overall_confidence,
            "uncertainty_percent": self.uncertainty_percent,
            "plausibility_score": self.plausibility_score,
            "validation_method": self.validation_method,
            "quality_flags": self.quality_flags,
        }


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a product.

    Follows the provenance.schema.json specification.

    Attributes:
        product_id: Unique product identifier
        event_id: Associated event ID
        lineage: Processing chain
        input_datasets: Primary input datasets
        algorithms_used: All algorithms used
        quality_summary: Quality assessment
        reproducibility: Reproducibility metadata
        created_at: When record was created
        created_by: Who/what created it
        total_processing_time_seconds: Total processing time
        cost_estimate_usd: Estimated processing cost
    """

    product_id: str
    event_id: str
    lineage: List[ProcessingStep] = field(default_factory=list)
    input_datasets: List[InputDataset] = field(default_factory=list)
    algorithms_used: List[AlgorithmInfo] = field(default_factory=list)
    quality_summary: Optional[QualitySummary] = None
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    total_processing_time_seconds: float = 0.0
    cost_estimate_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to provenance schema format."""
        return {
            "product_id": self.product_id,
            "event_id": self.event_id,
            "lineage": [step.to_dict() for step in self.lineage],
            "input_datasets": [ds.to_dict() for ds in self.input_datasets],
            "algorithms_used": [alg.to_dict() for alg in self.algorithms_used],
            "quality_summary": self.quality_summary.to_dict()
            if self.quality_summary
            else None,
            "reproducibility": self.reproducibility,
            "metadata": {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "created_by": self.created_by,
                "total_processing_time_seconds": self.total_processing_time_seconds,
                "cost_estimate_usd": self.cost_estimate_usd,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class LineageTracker:
    """
    Tracks lineage for processing pipelines.

    Provides methods to record inputs, processing steps, outputs,
    and generate complete provenance records.

    Example:
        tracker = LineageTracker()
        ctx = tracker.start_tracking("prod_123", "evt_456")

        ctx.add_input(InputDataset(
            dataset_id="S2A_...",
            provider="sentinel2",
            data_type="optical"
        ))

        with ctx.step("normalization") as step:
            step.set_processor("reprojection", "1.0.0")
            step.add_parameter("target_crs", "EPSG:4326")
            # ... do processing ...
            step.add_output("prod_123_normalized")

        record = ctx.finish()
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize LineageTracker.

        Args:
            db_path: Path to SQLite database (uses :memory: if None)
        """
        self._db_path = db_path or ":memory:"
        self._db: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._active_contexts: Dict[str, "TrackingContext"] = {}

        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS provenance_records (
                product_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS lineage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES provenance_records(product_id)
            )
        """)

        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_prov_event_id
            ON provenance_records(event_id)
        """)

        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_product_id
            ON lineage_events(product_id)
        """)

        self._db.commit()

    def start_tracking(
        self,
        product_id: str,
        event_id: str,
        agent: Optional[str] = None,
    ) -> "TrackingContext":
        """
        Start tracking lineage for a product.

        Args:
            product_id: Product being produced
            event_id: Associated event ID
            agent: Agent performing processing

        Returns:
            TrackingContext for recording lineage
        """
        ctx = TrackingContext(self, product_id, event_id, agent)
        self._active_contexts[product_id] = ctx
        return ctx

    def get_context(self, product_id: str) -> Optional["TrackingContext"]:
        """Get active tracking context."""
        return self._active_contexts.get(product_id)

    def _save_record(self, record: ProvenanceRecord):
        """Save provenance record to database."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO provenance_records
                (product_id, event_id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.product_id,
                    record.event_id,
                    record.to_json(),
                    record.created_at.isoformat() if record.created_at else now,
                    now,
                ),
            )
            self._db.commit()

    def _save_event(
        self,
        product_id: str,
        event_type: LineageEventType,
        data: Dict[str, Any],
    ):
        """Save a lineage event."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._db.execute(
                """
                INSERT INTO lineage_events
                (product_id, event_type, timestamp, data)
                VALUES (?, ?, ?, ?)
                """,
                (product_id, event_type.value, now, json.dumps(data)),
            )
            self._db.commit()

    def get_record(self, product_id: str) -> Optional[ProvenanceRecord]:
        """
        Get provenance record for a product.

        Args:
            product_id: Product ID

        Returns:
            ProvenanceRecord if exists
        """
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM provenance_records WHERE product_id = ?",
                (product_id,),
            ).fetchone()

        if not row:
            return None

        data = json.loads(row["data"])
        return self._record_from_dict(data)

    def _record_from_dict(self, data: Dict[str, Any]) -> ProvenanceRecord:
        """Convert dictionary to ProvenanceRecord."""
        lineage = []
        for step_data in data.get("lineage", []):
            lineage.append(
                ProcessingStep(
                    step=step_data["step"],
                    timestamp=datetime.fromisoformat(step_data["timestamp"]),
                    inputs=step_data.get("inputs", []),
                    outputs=step_data.get("outputs", []),
                    agent=step_data.get("agent"),
                    processor=step_data.get("processor"),
                    processor_version=step_data.get("processor_version"),
                    parameters=step_data.get("parameters", {}),
                    software_environment=step_data.get("software_environment", {}),
                    execution_time_seconds=step_data.get("execution_time_seconds", 0),
                    compute_resources=step_data.get("compute_resources", {}),
                    status=step_data.get("status", "completed"),
                    error_message=step_data.get("error_message"),
                )
            )

        input_datasets = []
        for ds_data in data.get("input_datasets", []):
            checksum_data = ds_data.get("checksum", {}) or {}
            input_datasets.append(
                InputDataset(
                    dataset_id=ds_data["dataset_id"],
                    provider=ds_data["provider"],
                    data_type=ds_data["data_type"],
                    acquisition_time=datetime.fromisoformat(ds_data["acquisition_time"])
                    if ds_data.get("acquisition_time")
                    else None,
                    processing_level=ds_data.get("processing_level"),
                    uri=ds_data.get("uri"),
                    checksum=checksum_data.get("value"),
                    checksum_algorithm=checksum_data.get("algorithm", "sha256"),
                    spatial_extent=ds_data.get("spatial_extent"),
                    temporal_extent=ds_data.get("temporal_extent"),
                    bands=ds_data.get("bands", []),
                    resolution_m=ds_data.get("resolution_m"),
                    metadata=ds_data.get("metadata", {}),
                )
            )

        algorithms_used = []
        for alg_data in data.get("algorithms_used", []):
            algorithms_used.append(
                AlgorithmInfo(
                    algorithm_id=alg_data["algorithm_id"],
                    name=alg_data["name"],
                    version=alg_data["version"],
                    category=alg_data.get("category", "baseline"),
                    purpose=alg_data.get("purpose"),
                    parameters=alg_data.get("parameters", {}),
                    class_name=alg_data.get("class_name"),
                    module_path=alg_data.get("module_path"),
                )
            )

        quality = None
        if data.get("quality_summary"):
            qs = data["quality_summary"]
            quality = QualitySummary(
                overall_confidence=qs.get("overall_confidence", 1.0),
                uncertainty_percent=qs.get("uncertainty_percent", 0.0),
                plausibility_score=qs.get("plausibility_score", 1.0),
                validation_method=qs.get("validation_method"),
                quality_flags=qs.get("quality_flags", []),
            )

        metadata = data.get("metadata", {})

        return ProvenanceRecord(
            product_id=data["product_id"],
            event_id=data["event_id"],
            lineage=lineage,
            input_datasets=input_datasets,
            algorithms_used=algorithms_used,
            quality_summary=quality,
            reproducibility=data.get("reproducibility", {}),
            created_at=datetime.fromisoformat(metadata["created_at"])
            if metadata.get("created_at")
            else None,
            created_by=metadata.get("created_by"),
            total_processing_time_seconds=metadata.get(
                "total_processing_time_seconds", 0
            ),
            cost_estimate_usd=metadata.get("cost_estimate_usd"),
        )

    def list_records(
        self,
        event_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProvenanceRecord]:
        """
        List provenance records.

        Args:
            event_id: Filter by event ID
            limit: Maximum records to return

        Returns:
            List of ProvenanceRecords
        """
        with self._lock:
            if event_id:
                rows = self._db.execute(
                    """
                    SELECT data FROM provenance_records
                    WHERE event_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (event_id, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    """
                    SELECT data FROM provenance_records
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [self._record_from_dict(json.loads(row["data"])) for row in rows]

    def get_events(
        self,
        product_id: str,
        event_type: Optional[LineageEventType] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get lineage events for a product.

        Args:
            product_id: Product ID
            event_type: Filter by event type

        Returns:
            List of event dictionaries
        """
        with self._lock:
            if event_type:
                rows = self._db.execute(
                    """
                    SELECT * FROM lineage_events
                    WHERE product_id = ? AND event_type = ?
                    ORDER BY timestamp
                    """,
                    (product_id, event_type.value),
                ).fetchall()
            else:
                rows = self._db.execute(
                    """
                    SELECT * FROM lineage_events
                    WHERE product_id = ?
                    ORDER BY timestamp
                    """,
                    (product_id,),
                ).fetchall()

        return [
            {
                "id": row["id"],
                "product_id": row["product_id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "data": json.loads(row["data"]),
            }
            for row in rows
        ]

    def export_to_json(self, product_id: str, path: Union[str, Path]) -> Path:
        """
        Export provenance record to JSON file.

        Args:
            product_id: Product ID
            path: Output file path

        Returns:
            Path to written file
        """
        record = self.get_record(product_id)
        if not record:
            raise ValueError(f"No record found for {product_id}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(record.to_json())

        return path

    def close(self):
        """Close tracker and release resources."""
        if self._db:
            self._db.close()
            self._db = None


class StepContext:
    """
    Context manager for recording a processing step.

    Used within TrackingContext to record step-level details.
    """

    def __init__(
        self,
        tracking_ctx: "TrackingContext",
        step_name: str,
    ):
        self._tracking_ctx = tracking_ctx
        self._step_name = step_name
        self._start_time: Optional[datetime] = None
        self._inputs: List[str] = []
        self._outputs: List[str] = []
        self._processor: Optional[str] = None
        self._processor_version: Optional[str] = None
        self._parameters: Dict[str, Any] = {}
        self._error: Optional[str] = None

    def __enter__(self) -> "StepContext":
        self._start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - self._start_time).total_seconds()

        status = "completed"
        if exc_type is not None:
            status = "failed"
            self._error = str(exc_val)

        step = ProcessingStep(
            step=self._step_name,
            timestamp=self._start_time,
            inputs=self._inputs,
            outputs=self._outputs,
            agent=self._tracking_ctx._agent,
            processor=self._processor,
            processor_version=self._processor_version,
            parameters=self._parameters,
            software_environment=get_software_environment(),
            execution_time_seconds=execution_time,
            compute_resources=get_compute_resources(),
            status=status,
            error_message=self._error,
        )

        self._tracking_ctx._steps.append(step)

        event_type = (
            LineageEventType.PROCESSING_FAILED
            if status == "failed"
            else LineageEventType.PROCESSING_COMPLETED
        )
        self._tracking_ctx._tracker._save_event(
            self._tracking_ctx._product_id,
            event_type,
            step.to_dict(),
        )

        # Don't suppress exceptions
        return False

    def add_input(self, input_id: str):
        """Add an input to this step."""
        self._inputs.append(input_id)

    def add_output(self, output_id: str):
        """Add an output from this step."""
        self._outputs.append(output_id)

    def set_processor(self, processor: str, version: Optional[str] = None):
        """Set the processor/algorithm used."""
        self._processor = processor
        self._processor_version = version

    def add_parameter(self, key: str, value: Any):
        """Add a processing parameter."""
        self._parameters[key] = value

    def set_parameters(self, params: Dict[str, Any]):
        """Set all parameters."""
        self._parameters = params


class TrackingContext:
    """
    Context for tracking lineage of a single product.

    Created by LineageTracker.start_tracking() and used to record
    inputs, processing steps, and outputs.
    """

    def __init__(
        self,
        tracker: LineageTracker,
        product_id: str,
        event_id: str,
        agent: Optional[str] = None,
    ):
        self._tracker = tracker
        self._product_id = product_id
        self._event_id = event_id
        self._agent = agent
        self._start_time = datetime.now(timezone.utc)

        self._inputs: List[InputDataset] = []
        self._algorithms: List[AlgorithmInfo] = []
        self._steps: List[ProcessingStep] = []
        self._quality: Optional[QualitySummary] = None
        self._finished = False

    def add_input(self, dataset: InputDataset):
        """
        Add an input dataset.

        Args:
            dataset: Input dataset information
        """
        self._inputs.append(dataset)
        self._tracker._save_event(
            self._product_id,
            LineageEventType.INPUT_REGISTERED,
            dataset.to_dict(),
        )

    def add_algorithm(self, algorithm: AlgorithmInfo):
        """
        Add an algorithm used.

        Args:
            algorithm: Algorithm information
        """
        self._algorithms.append(algorithm)

    def step(self, step_name: str) -> StepContext:
        """
        Create a step context for recording a processing step.

        Args:
            step_name: Name of the step

        Returns:
            StepContext for use with 'with' statement
        """
        self._tracker._save_event(
            self._product_id,
            LineageEventType.PROCESSING_STARTED,
            {"step": step_name},
        )
        return StepContext(self, step_name)

    def add_step(self, step: ProcessingStep):
        """
        Add a completed processing step directly.

        Args:
            step: Processing step
        """
        self._steps.append(step)

    def set_quality(self, quality: QualitySummary):
        """
        Set the quality summary.

        Args:
            quality: Quality summary
        """
        self._quality = quality
        self._tracker._save_event(
            self._product_id,
            LineageEventType.QUALITY_ASSESSED,
            quality.to_dict(),
        )

    def finish(self) -> ProvenanceRecord:
        """
        Finish tracking and generate provenance record.

        Returns:
            Complete ProvenanceRecord
        """
        if self._finished:
            raise RuntimeError("Tracking already finished")

        end_time = datetime.now(timezone.utc)
        total_time = (end_time - self._start_time).total_seconds()

        # Generate reproducibility hash
        repro_hash = self._compute_reproducibility_hash()

        record = ProvenanceRecord(
            product_id=self._product_id,
            event_id=self._event_id,
            lineage=self._steps,
            input_datasets=self._inputs,
            algorithms_used=self._algorithms,
            quality_summary=self._quality,
            reproducibility={
                "deterministic": self._check_determinism(),
                "selection_hash": repro_hash,
                "environment_hash": compute_environment_hash(),
            },
            created_at=self._start_time,
            created_by=self._agent or "system",
            total_processing_time_seconds=total_time,
        )

        self._tracker._save_record(record)
        self._finished = True

        # Remove from active contexts
        if self._product_id in self._tracker._active_contexts:
            del self._tracker._active_contexts[self._product_id]

        return record

    def _compute_reproducibility_hash(self) -> str:
        """Compute hash for reproducibility."""
        data = {
            "inputs": [ds.to_dict() for ds in self._inputs],
            "algorithms": [alg.to_dict() for alg in self._algorithms],
            "steps": [
                {"step": s.step, "processor": s.processor, "parameters": s.parameters}
                for s in self._steps
            ],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def _check_determinism(self) -> bool:
        """Check if all algorithms used are deterministic."""
        # Check parameters for random seeds
        for step in self._steps:
            params = step.parameters
            if "random_seed" in params or "shuffle" in params:
                return True  # Has seed, so reproducible
            if any(k.startswith("random") for k in params):
                return False  # Random without seed

        return True  # Assume deterministic by default


def get_software_environment() -> Dict[str, str]:
    """
    Get current software environment versions.

    Returns:
        Dictionary of library versions
    """
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # Try to get common library versions
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import rasterio

        env["rasterio"] = rasterio.__version__
    except ImportError:
        pass

    try:
        from osgeo import gdal

        env["gdal"] = gdal.__version__
    except ImportError:
        pass

    try:
        import scipy

        env["scipy"] = scipy.__version__
    except ImportError:
        pass

    return env


def get_compute_resources() -> Dict[str, Any]:
    """
    Get current compute resource usage.

    Returns:
        Dictionary with resource info
    """
    import os

    resources = {
        "cpu_cores": os.cpu_count() or 1,
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        resources["memory_gb"] = round(mem.total / (1024**3), 2)
        resources["memory_used_percent"] = mem.percent
    except ImportError:
        pass

    # Check for GPU
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            resources["gpu_used"] = True
            resources["gpu_name"] = result.stdout.strip().split("\n")[0]
        else:
            resources["gpu_used"] = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        resources["gpu_used"] = False

    return resources


def compute_environment_hash() -> str:
    """
    Compute hash of the software environment.

    Returns:
        16-character hex hash
    """
    env = get_software_environment()
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]


def create_provenance_from_job(
    product_id: str,
    event_id: str,
    job_data: Dict[str, Any],
) -> ProvenanceRecord:
    """
    Create provenance record from ingestion job data.

    Args:
        product_id: Output product ID
        event_id: Event ID
        job_data: Ingestion job dictionary

    Returns:
        ProvenanceRecord
    """
    inputs = []
    source = job_data.get("source", {})
    if source:
        inputs.append(
            InputDataset(
                dataset_id=source.get("uri", "unknown"),
                provider=source.get("provider", "unknown"),
                data_type=source.get("data_type", "unknown"),
                uri=source.get("uri"),
                checksum=source.get("checksum", {}).get("value")
                if source.get("checksum")
                else None,
            )
        )

    steps = []
    for step_name in ["ingestion", "normalization", "validation", "enrichment"]:
        if job_data.get(step_name):
            steps.append(
                ProcessingStep(
                    step=step_name,
                    timestamp=datetime.now(timezone.utc),
                    inputs=[product_id],
                    outputs=[product_id],
                    parameters=job_data.get(step_name, {}),
                )
            )

    return ProvenanceRecord(
        product_id=product_id,
        event_id=event_id,
        lineage=steps,
        input_datasets=inputs,
        created_at=datetime.now(timezone.utc),
    )
