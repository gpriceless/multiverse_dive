"""
Checkpoint Management for Pipeline Execution.

Provides state persistence and recovery for long-running pipelines:
- Checkpoint creation at configurable intervals
- Automatic recovery from last good state
- Versioned checkpoint storage
- Checkpoint cleanup and retention policies
- Thread-safe operations

Supports multiple storage backends:
- Local filesystem (default)
- SQLite database for metadata
- S3-compatible object storage (optional)
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class CheckpointStatus(Enum):
    """Status of a checkpoint."""

    PENDING = "pending"  # Being created
    VALID = "valid"  # Successfully created and verified
    CORRUPTED = "corrupted"  # Failed verification
    EXPIRED = "expired"  # Past retention period
    DELETED = "deleted"  # Marked for deletion


class CheckpointType(Enum):
    """Type of checkpoint."""

    AUTO = "auto"  # Automatic interval checkpoint
    MANUAL = "manual"  # Manually triggered
    RECOVERY = "recovery"  # Created before recovery attempt
    FINAL = "final"  # End of execution


@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        execution_id: Associated execution identifier
        version: Checkpoint version number
        checkpoint_type: Type of checkpoint
        status: Current status
        created_at: Creation timestamp
        completed_tasks: List of completed task IDs
        pending_tasks: List of pending task IDs
        task_count: Total number of tasks
        progress_percent: Progress percentage
        size_bytes: Checkpoint size in bytes
        checksum: Content checksum for verification
        storage_path: Path to checkpoint data
        metadata: Additional metadata
    """

    checkpoint_id: str
    execution_id: str
    version: int = 1
    checkpoint_type: CheckpointType = CheckpointType.AUTO
    status: CheckpointStatus = CheckpointStatus.PENDING
    created_at: Optional[datetime] = None
    completed_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    task_count: int = 0
    progress_percent: float = 0.0
    size_bytes: int = 0
    checksum: Optional[str] = None
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "version": self.version,
            "checkpoint_type": self.checkpoint_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_tasks": self.completed_tasks,
            "pending_tasks": self.pending_tasks,
            "task_count": self.task_count,
            "progress_percent": self.progress_percent,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "storage_path": self.storage_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            execution_id=data["execution_id"],
            version=data.get("version", 1),
            checkpoint_type=CheckpointType(data.get("checkpoint_type", "auto")),
            status=CheckpointStatus(data.get("status", "valid")),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            completed_tasks=data.get("completed_tasks", []),
            pending_tasks=data.get("pending_tasks", []),
            task_count=data.get("task_count", 0),
            progress_percent=data.get("progress_percent", 0.0),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum"),
            storage_path=data.get("storage_path"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointData:
    """
    Complete checkpoint data.

    Attributes:
        metadata: Checkpoint metadata
        task_results: Results from completed tasks
        shared_state: Shared state dictionary
        input_data: Original input data
        pipeline_state: Pipeline configuration state
    """

    metadata: CheckpointMetadata
    task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    pipeline_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "task_results": self.task_results,
            "shared_state": self.shared_state,
            "input_data": self.input_data,
            "pipeline_state": self.pipeline_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            task_results=data.get("task_results", {}),
            shared_state=data.get("shared_state", {}),
            input_data=data.get("input_data", {}),
            pipeline_state=data.get("pipeline_state", {}),
        )


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint management.

    Attributes:
        base_path: Base directory for checkpoint storage
        max_checkpoints: Maximum checkpoints to retain per execution
        retention_hours: Hours to retain checkpoints (0 = forever)
        auto_cleanup: Whether to automatically clean up old checkpoints
        compression: Whether to compress checkpoint data
        verify_on_load: Whether to verify checksum on load
        use_atomic_writes: Whether to use atomic file operations
        db_path: Path to SQLite database for metadata (None = in base_path)
    """

    base_path: Path = field(default_factory=lambda: Path("./checkpoints"))
    max_checkpoints: int = 10
    retention_hours: int = 168  # 7 days
    auto_cleanup: bool = True
    compression: bool = False
    verify_on_load: bool = True
    use_atomic_writes: bool = True
    db_path: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        if self.max_checkpoints < 1:
            raise ValueError(f"max_checkpoints must be >= 1, got {self.max_checkpoints}")
        if self.retention_hours < 0:
            raise ValueError(
                f"retention_hours must be >= 0, got {self.retention_hours}"
            )


class CheckpointStorageBackend(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    def save(
        self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata
    ) -> str:
        """
        Save checkpoint data.

        Args:
            checkpoint_id: Unique checkpoint identifier
            data: Serialized checkpoint data
            metadata: Checkpoint metadata

        Returns:
            Storage path/key for the checkpoint
        """
        pass

    @abstractmethod
    def load(self, checkpoint_id: str) -> Optional[bytes]:
        """
        Load checkpoint data.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Serialized checkpoint data or None if not found
        """
        pass

    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        pass

    @abstractmethod
    def list_checkpoints(self, execution_id: Optional[str] = None) -> List[str]:
        """List checkpoint IDs, optionally filtered by execution."""
        pass


class LocalStorageBackend(CheckpointStorageBackend):
    """
    Local filesystem storage backend.

    Stores checkpoints as files in a directory structure:
    base_path/
      execution_id/
        checkpoint_id.json  # Metadata
        checkpoint_id.pkl   # Data
    """

    def __init__(self, base_path: Path, use_atomic_writes: bool = True):
        """
        Initialize local storage backend.

        Args:
            base_path: Base directory for storage
            use_atomic_writes: Use atomic file operations
        """
        self.base_path = base_path
        self.use_atomic_writes = use_atomic_writes
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata
    ) -> str:
        """Save checkpoint to local filesystem."""
        # Create execution directory
        exec_dir = self.base_path / metadata.execution_id
        exec_dir.mkdir(parents=True, exist_ok=True)

        data_path = exec_dir / f"{checkpoint_id}.pkl"
        meta_path = exec_dir / f"{checkpoint_id}.json"

        if self.use_atomic_writes:
            # Write to temp file then rename (atomic on most filesystems)
            temp_data_path = data_path.with_suffix(".pkl.tmp")
            temp_meta_path = meta_path.with_suffix(".json.tmp")

            with open(temp_data_path, "wb") as f:
                f.write(data)
            with open(temp_meta_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            temp_data_path.rename(data_path)
            temp_meta_path.rename(meta_path)
        else:
            with open(data_path, "wb") as f:
                f.write(data)
            with open(meta_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        return str(data_path)

    def load(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint from local filesystem."""
        # Search for checkpoint in all execution directories
        for exec_dir in self.base_path.iterdir():
            if not exec_dir.is_dir():
                continue
            data_path = exec_dir / f"{checkpoint_id}.pkl"
            if data_path.exists():
                with open(data_path, "rb") as f:
                    return f.read()
        return None

    def load_with_metadata(
        self, checkpoint_id: str
    ) -> Optional[Tuple[bytes, CheckpointMetadata]]:
        """Load checkpoint data and metadata."""
        for exec_dir in self.base_path.iterdir():
            if not exec_dir.is_dir():
                continue
            data_path = exec_dir / f"{checkpoint_id}.pkl"
            meta_path = exec_dir / f"{checkpoint_id}.json"
            if data_path.exists() and meta_path.exists():
                with open(data_path, "rb") as f:
                    data = f.read()
                with open(meta_path, "r") as f:
                    metadata = CheckpointMetadata.from_dict(json.load(f))
                return data, metadata
        return None

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from filesystem."""
        deleted = False
        for exec_dir in self.base_path.iterdir():
            if not exec_dir.is_dir():
                continue
            data_path = exec_dir / f"{checkpoint_id}.pkl"
            meta_path = exec_dir / f"{checkpoint_id}.json"
            if data_path.exists():
                data_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
        return deleted

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        for exec_dir in self.base_path.iterdir():
            if not exec_dir.is_dir():
                continue
            if (exec_dir / f"{checkpoint_id}.pkl").exists():
                return True
        return False

    def list_checkpoints(self, execution_id: Optional[str] = None) -> List[str]:
        """List all checkpoint IDs."""
        checkpoints = []
        dirs_to_scan = []

        if execution_id:
            exec_dir = self.base_path / execution_id
            if exec_dir.exists():
                dirs_to_scan = [exec_dir]
        else:
            dirs_to_scan = [d for d in self.base_path.iterdir() if d.is_dir()]

        for exec_dir in dirs_to_scan:
            for path in exec_dir.glob("*.pkl"):
                checkpoints.append(path.stem)

        return checkpoints


class CheckpointManager:
    """
    Manages checkpoint lifecycle for pipeline executions.

    Features:
    - Automatic checkpoint creation at intervals
    - Version tracking and cleanup
    - Checkpoint verification and recovery
    - Thread-safe operations
    - SQLite-backed metadata for fast queries
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self.config = config or CheckpointConfig()
        self.storage = LocalStorageBackend(
            self.config.base_path,
            use_atomic_writes=self.config.use_atomic_writes,
        )
        self._db_path = self.config.db_path or (
            self.config.base_path / "checkpoints.db"
        )
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for metadata."""
        self.config.base_path.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    checkpoint_type TEXT DEFAULT 'auto',
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    task_count INTEGER DEFAULT 0,
                    completed_count INTEGER DEFAULT 0,
                    progress_percent REAL DEFAULT 0.0,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT,
                    storage_path TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_execution_id
                ON checkpoints(execution_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status
                ON checkpoints(status)
                """
            )
            conn.commit()

    def save(
        self,
        execution_id: str,
        data: Dict[str, Any],
        checkpoint_type: CheckpointType = CheckpointType.AUTO,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint.

        Args:
            execution_id: Execution identifier
            data: Checkpoint data dictionary
            checkpoint_type: Type of checkpoint

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        with self._lock:
            # Generate checkpoint ID
            timestamp = datetime.now(timezone.utc)
            version = self._get_next_version(execution_id)
            checkpoint_id = self._generate_checkpoint_id(execution_id, version)

            # Extract task info
            completed_tasks = list(data.get("completed", []))
            task_results = data.get("task_results", {})

            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                version=version,
                checkpoint_type=checkpoint_type,
                status=CheckpointStatus.PENDING,
                created_at=timestamp,
                completed_tasks=completed_tasks,
                pending_tasks=[],  # Will be filled by caller if needed
                task_count=len(task_results),
                progress_percent=len(completed_tasks) / max(len(task_results), 1) * 100,
            )

            # Serialize data
            checkpoint_data = CheckpointData(
                metadata=metadata,
                task_results=task_results,
                shared_state=data.get("shared_state", {}),
                input_data=data.get("input_data", {}),
                pipeline_state=data.get("pipeline_state", {}),
            )

            serialized = pickle.dumps(checkpoint_data.to_dict())
            checksum = hashlib.sha256(serialized).hexdigest()

            # Update metadata
            metadata.size_bytes = len(serialized)
            metadata.checksum = checksum

            # Save to storage
            storage_path = self.storage.save(checkpoint_id, serialized, metadata)
            metadata.storage_path = storage_path
            metadata.status = CheckpointStatus.VALID

            # Save metadata to database
            self._save_metadata_to_db(metadata)

            # Cleanup old checkpoints if needed
            if self.config.auto_cleanup:
                self._cleanup_old_checkpoints(execution_id)

            logger.info(
                f"Saved checkpoint {checkpoint_id} for execution {execution_id}"
            )
            return metadata

    def load_latest(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest valid checkpoint for an execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Checkpoint data dictionary or None if not found
        """
        metadata = self._get_latest_metadata(execution_id)
        if metadata is None:
            return None

        return self.load(metadata.checkpoint_id)

    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Checkpoint data dictionary or None if not found
        """
        with self._lock:
            data = self.storage.load(checkpoint_id)
            if data is None:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return None

            # Verify checksum if configured
            if self.config.verify_on_load:
                metadata = self._get_metadata(checkpoint_id)
                if metadata and metadata.checksum:
                    actual_checksum = hashlib.sha256(data).hexdigest()
                    if actual_checksum != metadata.checksum:
                        logger.error(
                            f"Checkpoint {checkpoint_id} checksum mismatch"
                        )
                        self._mark_corrupted(checkpoint_id)
                        return None

            # Deserialize
            try:
                checkpoint_dict = pickle.loads(data)
                logger.info(f"Loaded checkpoint {checkpoint_id}")
                return checkpoint_dict
            except Exception as e:
                logger.error(f"Failed to deserialize checkpoint {checkpoint_id}: {e}")
                self._mark_corrupted(checkpoint_id)
                return None

    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            # Delete from storage
            deleted = self.storage.delete(checkpoint_id)

            # Update database
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "UPDATE checkpoints SET status = ? WHERE checkpoint_id = ?",
                    (CheckpointStatus.DELETED.value, checkpoint_id),
                )
                conn.commit()

            if deleted:
                logger.info(f"Deleted checkpoint {checkpoint_id}")
            return deleted

    def list_checkpoints(
        self,
        execution_id: Optional[str] = None,
        status: Optional[CheckpointStatus] = None,
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints with optional filtering.

        Args:
            execution_id: Filter by execution
            status: Filter by status

        Returns:
            List of CheckpointMetadata objects
        """
        query = "SELECT * FROM checkpoints WHERE 1=1"
        params: List[Any] = []

        if execution_id:
            query += " AND execution_id = ?"
            params.append(execution_id)
        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC"

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_metadata(row) for row in rows]

    def get_execution_progress(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution progress from latest checkpoint.

        Args:
            execution_id: Execution identifier

        Returns:
            Progress information or None if no checkpoint exists
        """
        metadata = self._get_latest_metadata(execution_id)
        if metadata is None:
            return None

        return {
            "execution_id": execution_id,
            "checkpoint_id": metadata.checkpoint_id,
            "version": metadata.version,
            "progress_percent": metadata.progress_percent,
            "completed_tasks": len(metadata.completed_tasks),
            "total_tasks": metadata.task_count,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
        }

    def cleanup(
        self,
        execution_id: Optional[str] = None,
        older_than_hours: Optional[int] = None,
        keep_latest: int = 1,
    ) -> int:
        """
        Clean up old checkpoints.

        Args:
            execution_id: Limit cleanup to specific execution
            older_than_hours: Delete checkpoints older than this
            keep_latest: Number of latest checkpoints to keep per execution

        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0
        cutoff_time = None
        if older_than_hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        # Get executions to process
        if execution_id:
            executions = [execution_id]
        else:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT execution_id FROM checkpoints"
                )
                executions = [row[0] for row in cursor.fetchall()]

        for exec_id in executions:
            checkpoints = self.list_checkpoints(
                execution_id=exec_id, status=CheckpointStatus.VALID
            )

            # Skip the latest N checkpoints
            to_delete = checkpoints[keep_latest:]

            for metadata in to_delete:
                # Check age if cutoff specified
                if cutoff_time and metadata.created_at:
                    if metadata.created_at > cutoff_time:
                        continue

                if self.delete(metadata.checkpoint_id):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} checkpoints")
        return deleted_count

    def _generate_checkpoint_id(self, execution_id: str, version: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{execution_id[:8]}_{version:04d}_{timestamp}"

    def _get_next_version(self, execution_id: str) -> int:
        """Get next version number for execution."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "SELECT MAX(version) FROM checkpoints WHERE execution_id = ?",
                (execution_id,),
            )
            row = cursor.fetchone()
            current_max = row[0] if row[0] else 0
            return current_max + 1

    def _save_metadata_to_db(self, metadata: CheckpointMetadata) -> None:
        """Save metadata to SQLite database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, execution_id, version, checkpoint_type,
                    status, created_at, task_count, completed_count,
                    progress_percent, size_bytes, checksum, storage_path,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.checkpoint_id,
                    metadata.execution_id,
                    metadata.version,
                    metadata.checkpoint_type.value,
                    metadata.status.value,
                    metadata.created_at.isoformat() if metadata.created_at else None,
                    metadata.task_count,
                    len(metadata.completed_tasks),
                    metadata.progress_percent,
                    metadata.size_bytes,
                    metadata.checksum,
                    metadata.storage_path,
                    json.dumps(metadata.metadata),
                ),
            )
            conn.commit()

    def _get_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata from database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
        return None

    def _get_latest_metadata(
        self, execution_id: str
    ) -> Optional[CheckpointMetadata]:
        """Get latest valid checkpoint metadata for execution."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE execution_id = ? AND status = ?
                ORDER BY version DESC LIMIT 1
                """,
                (execution_id, CheckpointStatus.VALID.value),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
        return None

    def _row_to_metadata(self, row: sqlite3.Row) -> CheckpointMetadata:
        """Convert database row to CheckpointMetadata."""
        return CheckpointMetadata(
            checkpoint_id=row["checkpoint_id"],
            execution_id=row["execution_id"],
            version=row["version"],
            checkpoint_type=CheckpointType(row["checkpoint_type"]),
            status=CheckpointStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            completed_tasks=[],  # Not stored in DB, load from file if needed
            pending_tasks=[],
            task_count=row["task_count"],
            progress_percent=row["progress_percent"],
            size_bytes=row["size_bytes"],
            checksum=row["checksum"],
            storage_path=row["storage_path"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    def _mark_corrupted(self, checkpoint_id: str) -> None:
        """Mark a checkpoint as corrupted."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE checkpoints SET status = ? WHERE checkpoint_id = ?",
                (CheckpointStatus.CORRUPTED.value, checkpoint_id),
            )
            conn.commit()

    def _cleanup_old_checkpoints(self, execution_id: str) -> None:
        """Clean up old checkpoints for execution."""
        checkpoints = self.list_checkpoints(
            execution_id=execution_id, status=CheckpointStatus.VALID
        )

        if len(checkpoints) > self.config.max_checkpoints:
            # Delete oldest checkpoints beyond limit
            to_delete = checkpoints[self.config.max_checkpoints:]
            for metadata in to_delete:
                self.delete(metadata.checkpoint_id)

        # Delete expired checkpoints
        if self.config.retention_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=self.config.retention_hours
            )
            for metadata in checkpoints:
                if metadata.created_at and metadata.created_at < cutoff:
                    self.delete(metadata.checkpoint_id)


class AutoCheckpointer:
    """
    Automatic checkpoint creation at configurable intervals.

    Can be used as a context manager or standalone.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        execution_id: str,
        interval_seconds: float = 60.0,
        on_checkpoint: Optional[Callable[[CheckpointMetadata], None]] = None,
    ):
        """
        Initialize auto checkpointer.

        Args:
            checkpoint_manager: Checkpoint manager to use
            execution_id: Execution identifier
            interval_seconds: Seconds between checkpoints
            on_checkpoint: Callback when checkpoint is created
        """
        self.checkpoint_manager = checkpoint_manager
        self.execution_id = execution_id
        self.interval_seconds = interval_seconds
        self.on_checkpoint = on_checkpoint
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_data: Dict[str, Any] = {}
        self._data_lock = threading.Lock()

    def start(self) -> None:
        """Start automatic checkpointing."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Started auto checkpointing for {self.execution_id} "
            f"every {self.interval_seconds}s"
        )

    def stop(self) -> None:
        """Stop automatic checkpointing."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"Stopped auto checkpointing for {self.execution_id}")

    def update_data(self, data: Dict[str, Any]) -> None:
        """Update the data to be checkpointed."""
        with self._data_lock:
            self._current_data = data.copy()

    def _checkpoint_loop(self) -> None:
        """Background loop for creating checkpoints."""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                with self._data_lock:
                    data = self._current_data.copy()

                if data:
                    metadata = self.checkpoint_manager.save(
                        self.execution_id, data, CheckpointType.AUTO
                    )
                    if self.on_checkpoint:
                        self.on_checkpoint(metadata)
            except Exception as e:
                logger.error(f"Auto checkpoint failed: {e}")

    def __enter__(self) -> "AutoCheckpointer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
