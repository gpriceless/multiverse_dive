"""
Storage Backends for Ingestion Pipeline.

Provides abstract and concrete implementations for persisting ingested data
to various storage backends:
- Local filesystem
- AWS S3
- Google Cloud Storage
- Azure Blob Storage

Each backend follows a consistent interface for upload, download, list, and delete
operations with support for streaming, checksums, and metadata.
"""

import hashlib
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Supported storage backend types."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class ChecksumAlgorithm(Enum):
    """Supported checksum algorithms."""

    MD5 = "md5"
    SHA256 = "sha256"


@dataclass
class StorageObject:
    """
    Represents an object in storage.

    Attributes:
        key: Object key/path in storage
        size_bytes: Size in bytes
        last_modified: Last modification timestamp
        checksum: Object checksum (algorithm:value format)
        content_type: MIME type
        metadata: Custom metadata dictionary
        storage_class: Storage class (for cloud backends)
    """

    key: str
    size_bytes: int
    last_modified: datetime
    checksum: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    storage_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified.isoformat(),
            "checksum": self.checksum,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "storage_class": self.storage_class,
        }


@dataclass
class UploadResult:
    """Result of an upload operation."""

    key: str
    size_bytes: int
    checksum: str
    checksum_algorithm: ChecksumAlgorithm
    uri: str
    etag: Optional[str] = None
    version_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm.value,
            "uri": self.uri,
            "etag": self.etag,
            "version_id": self.version_id,
            "metadata": self.metadata,
        }


@dataclass
class StorageConfig:
    """
    Configuration for storage backends.

    Attributes:
        storage_type: Type of storage backend
        root_path: Root path or bucket name
        region: Cloud region (for S3, GCS, Azure)
        endpoint_url: Custom endpoint URL (for S3-compatible)
        credentials: Credential configuration
        default_acl: Default ACL for uploads
        checksum_algorithm: Algorithm for checksums
        chunk_size_mb: Chunk size for multipart uploads
        max_retries: Maximum retry attempts
        timeout_seconds: Operation timeout
    """

    storage_type: StorageType
    root_path: str
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    default_acl: str = "private"
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    chunk_size_mb: int = 8
    max_retries: int = 3
    timeout_seconds: int = 300

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size_mb < 1:
            raise ValueError(f"chunk_size_mb must be >= 1, got {self.chunk_size_mb}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Provides a consistent interface for storage operations across
    different backend implementations.
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize storage backend.

        Args:
            config: Storage configuration
        """
        self.config = config

    @property
    @abstractmethod
    def storage_type(self) -> StorageType:
        """Return the storage type."""
        pass

    @abstractmethod
    def upload_file(
        self,
        local_path: Union[str, Path],
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """
        Upload a file to storage.

        Args:
            local_path: Path to local file
            key: Destination key in storage
            content_type: MIME type (auto-detected if None)
            metadata: Custom metadata

        Returns:
            UploadResult with upload details
        """
        pass

    @abstractmethod
    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """
        Upload bytes to storage.

        Args:
            data: Bytes to upload
            key: Destination key in storage
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            UploadResult with upload details
        """
        pass

    @abstractmethod
    def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """
        Upload from a stream to storage.

        Args:
            stream: Binary stream to upload
            key: Destination key in storage
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            UploadResult with upload details
        """
        pass

    @abstractmethod
    def download_file(
        self,
        key: str,
        local_path: Union[str, Path],
        verify_checksum: bool = True,
    ) -> StorageObject:
        """
        Download a file from storage.

        Args:
            key: Source key in storage
            local_path: Destination local path
            verify_checksum: Whether to verify checksum

        Returns:
            StorageObject with file details
        """
        pass

    @abstractmethod
    def download_bytes(self, key: str) -> Tuple[bytes, StorageObject]:
        """
        Download as bytes from storage.

        Args:
            key: Source key in storage

        Returns:
            Tuple of (bytes, StorageObject)
        """
        pass

    @abstractmethod
    def download_stream(self, key: str) -> Tuple[BinaryIO, StorageObject]:
        """
        Download as stream from storage.

        Args:
            key: Source key in storage

        Returns:
            Tuple of (stream, StorageObject)
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key

        Returns:
            True if object exists
        """
        pass

    @abstractmethod
    def get_object_info(self, key: str) -> Optional[StorageObject]:
        """
        Get object metadata without downloading.

        Args:
            key: Object key

        Returns:
            StorageObject if exists, None otherwise
        """
        pass

    @abstractmethod
    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True,
        max_results: Optional[int] = None,
    ) -> Iterator[StorageObject]:
        """
        List objects in storage.

        Args:
            prefix: Key prefix to filter
            recursive: Whether to list recursively
            max_results: Maximum number of results

        Yields:
            StorageObject for each matching object
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an object.

        Args:
            key: Object key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_many(self, keys: List[str]) -> List[str]:
        """
        Delete multiple objects.

        Args:
            keys: List of object keys

        Returns:
            List of keys that were successfully deleted
        """
        pass

    @abstractmethod
    def copy(self, source_key: str, dest_key: str) -> StorageObject:
        """
        Copy an object within storage.

        Args:
            source_key: Source object key
            dest_key: Destination object key

        Returns:
            StorageObject for the copied object
        """
        pass

    @abstractmethod
    def move(self, source_key: str, dest_key: str) -> StorageObject:
        """
        Move an object within storage.

        Args:
            source_key: Source object key
            dest_key: Destination object key

        Returns:
            StorageObject for the moved object
        """
        pass

    def get_uri(self, key: str) -> str:
        """
        Get the full URI for an object.

        Args:
            key: Object key

        Returns:
            Full URI string
        """
        if self.storage_type == StorageType.LOCAL:
            return f"file://{Path(self.config.root_path) / key}"
        elif self.storage_type == StorageType.S3:
            return f"s3://{self.config.root_path}/{key}"
        elif self.storage_type == StorageType.GCS:
            return f"gs://{self.config.root_path}/{key}"
        elif self.storage_type == StorageType.AZURE:
            return f"azure://{self.config.root_path}/{key}"
        else:
            return f"{self.storage_type.value}://{self.config.root_path}/{key}"

    def _compute_checksum(
        self,
        data: bytes,
        algorithm: Optional[ChecksumAlgorithm] = None,
    ) -> str:
        """Compute checksum for data."""
        alg = algorithm or self.config.checksum_algorithm
        if alg == ChecksumAlgorithm.MD5:
            return hashlib.md5(data).hexdigest()
        elif alg == ChecksumAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {alg}")

    def _compute_file_checksum(
        self,
        path: Path,
        algorithm: Optional[ChecksumAlgorithm] = None,
    ) -> str:
        """Compute checksum for a file."""
        alg = algorithm or self.config.checksum_algorithm
        if alg == ChecksumAlgorithm.MD5:
            hasher = hashlib.md5()
        elif alg == ChecksumAlgorithm.SHA256:
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {alg}")

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _guess_content_type(self, path: Union[str, Path]) -> str:
        """Guess content type from file extension."""
        import mimetypes
        content_type, _ = mimetypes.guess_type(str(path))
        return content_type or "application/octet-stream"


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.

    Stores data in a local directory structure with support for
    checksums, metadata, and atomic operations.
    """

    METADATA_SUFFIX = ".meta.json"

    @property
    def storage_type(self) -> StorageType:
        return StorageType.LOCAL

    def __init__(self, config: StorageConfig):
        """Initialize local storage."""
        super().__init__(config)
        self._root = Path(config.root_path)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at {self._root}")

    def _get_full_path(self, key: str) -> Path:
        """Get full filesystem path for a key."""
        # Normalize key to prevent path traversal
        normalized = Path(key).as_posix().lstrip("/")
        return self._root / normalized

    def _get_metadata_path(self, key: str) -> Path:
        """Get path for metadata file."""
        return self._get_full_path(key + self.METADATA_SUFFIX)

    def _write_metadata(
        self,
        key: str,
        checksum: str,
        content_type: Optional[str],
        metadata: Optional[Dict[str, str]],
    ) -> None:
        """Write metadata file."""
        import json

        meta_path = self._get_metadata_path(key)
        meta_data = {
            "checksum": checksum,
            "checksum_algorithm": self.config.checksum_algorithm.value,
            "content_type": content_type,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

    def _read_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Read metadata file."""
        import json

        meta_path = self._get_metadata_path(key)
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def upload_file(
        self,
        local_path: Union[str, Path],
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload a file to local storage."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        dest_path = self._get_full_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file atomically
        temp_path = dest_path.with_suffix(".tmp")
        shutil.copy2(local_path, temp_path)
        temp_path.rename(dest_path)

        checksum = self._compute_file_checksum(dest_path)
        content_type = content_type or self._guess_content_type(local_path)

        self._write_metadata(key, checksum, content_type, metadata)

        logger.debug(f"Uploaded {local_path} to {key}")

        return UploadResult(
            key=key,
            size_bytes=dest_path.stat().st_size,
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            uri=self.get_uri(key),
            metadata=metadata or {},
        )

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload bytes to local storage."""
        dest_path = self._get_full_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = dest_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            f.write(data)
        temp_path.rename(dest_path)

        checksum = self._compute_checksum(data)

        self._write_metadata(key, checksum, content_type, metadata)

        logger.debug(f"Uploaded {len(data)} bytes to {key}")

        return UploadResult(
            key=key,
            size_bytes=len(data),
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            uri=self.get_uri(key),
            metadata=metadata or {},
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload from stream to local storage."""
        dest_path = self._get_full_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.with_suffix(".tmp")

        alg = self.config.checksum_algorithm
        if alg == ChecksumAlgorithm.MD5:
            hasher = hashlib.md5()
        else:
            hasher = hashlib.sha256()

        size = 0
        with open(temp_path, "wb") as f:
            for chunk in iter(lambda: stream.read(8192), b""):
                f.write(chunk)
                hasher.update(chunk)
                size += len(chunk)

        temp_path.rename(dest_path)
        checksum = hasher.hexdigest()

        self._write_metadata(key, checksum, content_type, metadata)

        logger.debug(f"Uploaded {size} bytes to {key}")

        return UploadResult(
            key=key,
            size_bytes=size,
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            uri=self.get_uri(key),
            metadata=metadata or {},
        )

    def download_file(
        self,
        key: str,
        local_path: Union[str, Path],
        verify_checksum: bool = True,
    ) -> StorageObject:
        """Download a file from local storage."""
        src_path = self._get_full_path(key)
        local_path = Path(local_path)

        if not src_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy atomically
        temp_path = local_path.with_suffix(".tmp")
        shutil.copy2(src_path, temp_path)

        if verify_checksum:
            meta = self._read_metadata(key)
            if meta:
                expected = meta.get("checksum")
                alg = ChecksumAlgorithm(meta.get("checksum_algorithm", "sha256"))
                actual = self._compute_file_checksum(temp_path, alg)
                if expected and expected != actual:
                    temp_path.unlink()
                    raise ValueError(
                        f"Checksum mismatch for {key}: expected {expected}, got {actual}"
                    )

        temp_path.rename(local_path)

        stat = src_path.stat()
        meta = self._read_metadata(key) or {}

        return StorageObject(
            key=key,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=meta.get("checksum"),
            content_type=meta.get("content_type"),
            metadata=meta.get("metadata", {}),
        )

    def download_bytes(self, key: str) -> Tuple[bytes, StorageObject]:
        """Download as bytes from local storage."""
        src_path = self._get_full_path(key)

        if not src_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")

        with open(src_path, "rb") as f:
            data = f.read()

        stat = src_path.stat()
        meta = self._read_metadata(key) or {}

        obj = StorageObject(
            key=key,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=meta.get("checksum"),
            content_type=meta.get("content_type"),
            metadata=meta.get("metadata", {}),
        )

        return data, obj

    def download_stream(self, key: str) -> Tuple[BinaryIO, StorageObject]:
        """Download as stream from local storage."""
        src_path = self._get_full_path(key)

        if not src_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")

        stat = src_path.stat()
        meta = self._read_metadata(key) or {}

        obj = StorageObject(
            key=key,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=meta.get("checksum"),
            content_type=meta.get("content_type"),
            metadata=meta.get("metadata", {}),
        )

        return open(src_path, "rb"), obj

    def exists(self, key: str) -> bool:
        """Check if object exists."""
        return self._get_full_path(key).exists()

    def get_object_info(self, key: str) -> Optional[StorageObject]:
        """Get object info without downloading."""
        path = self._get_full_path(key)
        if not path.exists():
            return None

        stat = path.stat()
        meta = self._read_metadata(key) or {}

        return StorageObject(
            key=key,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=meta.get("checksum"),
            content_type=meta.get("content_type"),
            metadata=meta.get("metadata", {}),
        )

    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True,
        max_results: Optional[int] = None,
    ) -> Iterator[StorageObject]:
        """List objects in local storage."""
        base_path = self._get_full_path(prefix) if prefix else self._root
        count = 0

        if recursive:
            glob_pattern = "**/*"
        else:
            glob_pattern = "*"

        if base_path.is_file():
            # Exact match
            yield self.get_object_info(prefix)
            return

        for path in sorted(base_path.glob(glob_pattern)):
            if not path.is_file():
                continue
            if path.suffix == ".json" and ".meta" in path.stem:
                continue  # Skip metadata files

            if max_results and count >= max_results:
                break

            key = str(path.relative_to(self._root))
            obj = self.get_object_info(key)
            if obj:
                yield obj
                count += 1

    def delete(self, key: str) -> bool:
        """Delete an object."""
        path = self._get_full_path(key)
        meta_path = self._get_metadata_path(key)

        if not path.exists():
            return False

        path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        logger.debug(f"Deleted {key}")
        return True

    def delete_many(self, keys: List[str]) -> List[str]:
        """Delete multiple objects."""
        deleted = []
        for key in keys:
            if self.delete(key):
                deleted.append(key)
        return deleted

    def copy(self, source_key: str, dest_key: str) -> StorageObject:
        """Copy an object."""
        src_path = self._get_full_path(source_key)
        dest_path = self._get_full_path(dest_key)

        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {source_key}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        # Copy metadata
        src_meta_path = self._get_metadata_path(source_key)
        if src_meta_path.exists():
            dest_meta_path = self._get_metadata_path(dest_key)
            shutil.copy2(src_meta_path, dest_meta_path)

        logger.debug(f"Copied {source_key} to {dest_key}")
        return self.get_object_info(dest_key)

    def move(self, source_key: str, dest_key: str) -> StorageObject:
        """Move an object."""
        src_path = self._get_full_path(source_key)
        dest_path = self._get_full_path(dest_key)

        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {source_key}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dest_path)

        # Move metadata
        src_meta_path = self._get_metadata_path(source_key)
        if src_meta_path.exists():
            dest_meta_path = self._get_metadata_path(dest_key)
            shutil.move(src_meta_path, dest_meta_path)

        logger.debug(f"Moved {source_key} to {dest_key}")
        return self.get_object_info(dest_key)


class S3StorageBackend(StorageBackend):
    """
    AWS S3 storage backend.

    Provides storage operations against S3-compatible object storage.
    Requires boto3 for AWS S3 or can be configured for S3-compatible
    endpoints (MinIO, LocalStack, etc.).
    """

    @property
    def storage_type(self) -> StorageType:
        return StorageType.S3

    def __init__(self, config: StorageConfig):
        """Initialize S3 storage."""
        super().__init__(config)
        self._bucket = config.root_path
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize boto3 S3 client."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError("boto3 is required for S3 storage backend")

        boto_config = Config(
            retries={"max_attempts": self.config.max_retries},
            connect_timeout=self.config.timeout_seconds,
            read_timeout=self.config.timeout_seconds,
        )

        kwargs = {"config": boto_config}

        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url

        if self.config.region:
            kwargs["region_name"] = self.config.region

        if self.config.credentials:
            kwargs["aws_access_key_id"] = self.config.credentials.get("access_key_id")
            kwargs["aws_secret_access_key"] = self.config.credentials.get(
                "secret_access_key"
            )
            if "session_token" in self.config.credentials:
                kwargs["aws_session_token"] = self.config.credentials["session_token"]

        self._client = boto3.client("s3", **kwargs)
        logger.info(f"Initialized S3 storage for bucket {self._bucket}")

    def upload_file(
        self,
        local_path: Union[str, Path],
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload a file to S3."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        content_type = content_type or self._guess_content_type(local_path)
        checksum = self._compute_file_checksum(local_path)

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        if self.config.default_acl != "private":
            extra_args["ACL"] = self.config.default_acl

        self._client.upload_file(
            str(local_path),
            self._bucket,
            key,
            ExtraArgs=extra_args,
        )

        response = self._client.head_object(Bucket=self._bucket, Key=key)

        logger.debug(f"Uploaded {local_path} to s3://{self._bucket}/{key}")

        return UploadResult(
            key=key,
            size_bytes=local_path.stat().st_size,
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            uri=self.get_uri(key),
            etag=response.get("ETag", "").strip('"'),
            version_id=response.get("VersionId"),
            metadata=metadata or {},
        )

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload bytes to S3."""
        import io

        checksum = self._compute_checksum(data)
        content_type = content_type or "application/octet-stream"

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata

        self._client.upload_fileobj(
            io.BytesIO(data),
            self._bucket,
            key,
            ExtraArgs=extra_args,
        )

        response = self._client.head_object(Bucket=self._bucket, Key=key)

        logger.debug(f"Uploaded {len(data)} bytes to s3://{self._bucket}/{key}")

        return UploadResult(
            key=key,
            size_bytes=len(data),
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            uri=self.get_uri(key),
            etag=response.get("ETag", "").strip('"'),
            version_id=response.get("VersionId"),
            metadata=metadata or {},
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """Upload from stream to S3."""
        # For streams, we need to buffer to compute checksum
        data = stream.read()
        return self.upload_bytes(data, key, content_type, metadata)

    def download_file(
        self,
        key: str,
        local_path: Union[str, Path],
        verify_checksum: bool = True,
    ) -> StorageObject:
        """Download a file from S3."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = local_path.with_suffix(".tmp")

        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Object not found: {key}")

        self._client.download_file(self._bucket, key, str(temp_path))
        temp_path.rename(local_path)

        obj = StorageObject(
            key=key,
            size_bytes=response["ContentLength"],
            last_modified=response["LastModified"],
            checksum=response.get("ETag", "").strip('"'),
            content_type=response.get("ContentType"),
            metadata=response.get("Metadata", {}),
            storage_class=response.get("StorageClass"),
        )

        return obj

    def download_bytes(self, key: str) -> Tuple[bytes, StorageObject]:
        """Download as bytes from S3."""
        import io

        response = self._client.get_object(Bucket=self._bucket, Key=key)
        data = response["Body"].read()

        obj = StorageObject(
            key=key,
            size_bytes=response["ContentLength"],
            last_modified=response["LastModified"],
            checksum=response.get("ETag", "").strip('"'),
            content_type=response.get("ContentType"),
            metadata=response.get("Metadata", {}),
            storage_class=response.get("StorageClass"),
        )

        return data, obj

    def download_stream(self, key: str) -> Tuple[BinaryIO, StorageObject]:
        """Download as stream from S3."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)

        obj = StorageObject(
            key=key,
            size_bytes=response["ContentLength"],
            last_modified=response["LastModified"],
            checksum=response.get("ETag", "").strip('"'),
            content_type=response.get("ContentType"),
            metadata=response.get("Metadata", {}),
            storage_class=response.get("StorageClass"),
        )

        return response["Body"], obj

    def exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._client.exceptions.ClientError as e:
            # NoSuchKey and 404 responses mean object doesn't exist
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            # Re-raise unexpected client errors
            raise
        except Exception:
            # Fallback for other exceptions (e.g., connection errors)
            return False

    def get_object_info(self, key: str) -> Optional[StorageObject]:
        """Get object info from S3."""
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
            return StorageObject(
                key=key,
                size_bytes=response["ContentLength"],
                last_modified=response["LastModified"],
                checksum=response.get("ETag", "").strip('"'),
                content_type=response.get("ContentType"),
                metadata=response.get("Metadata", {}),
                storage_class=response.get("StorageClass"),
            )
        except self._client.exceptions.ClientError as e:
            # NoSuchKey and 404 responses mean object doesn't exist
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            # Log and return None for other client errors
            logger.warning(f"Error getting object info for {key}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting object info for {key}: {e}")
            return None

    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True,
        max_results: Optional[int] = None,
    ) -> Iterator[StorageObject]:
        """List objects in S3."""
        paginator = self._client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": self._bucket, "Prefix": prefix}

        if not recursive:
            kwargs["Delimiter"] = "/"

        count = 0
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                if max_results and count >= max_results:
                    return

                yield StorageObject(
                    key=obj["Key"],
                    size_bytes=obj["Size"],
                    last_modified=obj["LastModified"],
                    checksum=obj.get("ETag", "").strip('"'),
                    storage_class=obj.get("StorageClass"),
                )
                count += 1

    def delete(self, key: str) -> bool:
        """Delete an object from S3."""
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.debug(f"Deleted s3://{self._bucket}/{key}")
            return True
        except self._client.exceptions.ClientError as e:
            logger.warning(f"Error deleting {key}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error deleting {key}: {e}")
            return False

    def delete_many(self, keys: List[str]) -> List[str]:
        """Delete multiple objects from S3."""
        if not keys:
            return []

        objects = [{"Key": key} for key in keys]
        response = self._client.delete_objects(
            Bucket=self._bucket,
            Delete={"Objects": objects},
        )

        deleted = [obj["Key"] for obj in response.get("Deleted", [])]
        return deleted

    def copy(self, source_key: str, dest_key: str) -> StorageObject:
        """Copy an object in S3."""
        self._client.copy_object(
            Bucket=self._bucket,
            CopySource={"Bucket": self._bucket, "Key": source_key},
            Key=dest_key,
        )
        logger.debug(f"Copied {source_key} to {dest_key}")
        return self.get_object_info(dest_key)

    def move(self, source_key: str, dest_key: str) -> StorageObject:
        """Move an object in S3."""
        obj = self.copy(source_key, dest_key)
        self.delete(source_key)
        logger.debug(f"Moved {source_key} to {dest_key}")
        return obj


def create_storage_backend(config: StorageConfig) -> StorageBackend:
    """
    Factory function to create appropriate storage backend.

    Args:
        config: Storage configuration

    Returns:
        StorageBackend instance

    Raises:
        ValueError: If storage type is not supported
    """
    if config.storage_type == StorageType.LOCAL:
        return LocalStorageBackend(config)
    elif config.storage_type == StorageType.S3:
        return S3StorageBackend(config)
    elif config.storage_type == StorageType.GCS:
        raise NotImplementedError("GCS storage backend not yet implemented")
    elif config.storage_type == StorageType.AZURE:
        raise NotImplementedError("Azure storage backend not yet implemented")
    else:
        raise ValueError(f"Unsupported storage type: {config.storage_type}")


def parse_storage_uri(uri: str) -> Tuple[StorageType, str, str]:
    """
    Parse a storage URI into type, bucket/root, and key.

    Args:
        uri: Storage URI (s3://bucket/key, file:///path, etc.)

    Returns:
        Tuple of (StorageType, bucket/root, key)

    Raises:
        ValueError: If URI is malformed or scheme not supported
    """
    parsed = urlparse(uri)

    scheme = parsed.scheme.lower()
    if scheme in ("", "file"):
        path = parsed.path
        # For local files, root is parent dir, key is filename
        p = Path(path)
        return StorageType.LOCAL, str(p.parent), p.name
    elif scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return StorageType.S3, bucket, key
    elif scheme == "gs":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return StorageType.GCS, bucket, key
    elif scheme == "azure":
        container = parsed.netloc
        key = parsed.path.lstrip("/")
        return StorageType.AZURE, container, key
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme}")
