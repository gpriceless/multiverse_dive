"""
Cache Storage Backends.

Provides storage implementations optimized for caching ingested geospatial data.
Supports local filesystem and S3-compatible object storage with:
- Content-addressable storage (CAS) using checksums
- Tiered storage (hot/warm/cold)
- Efficient partial reads for COG/Zarr access patterns
- Atomic operations for concurrent access safety

The cache storage layer sits between the ingestion pipeline and the analysis
layer, providing fast access to normalized, analysis-ready data products.
"""

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CacheStorageTier(Enum):
    """Storage tiers for tiered caching."""

    HOT = "hot"  # Fastest access, limited capacity (SSD/RAM)
    WARM = "warm"  # Moderate access, moderate capacity (HDD)
    COLD = "cold"  # Slow access, large capacity (archive/S3 IA)


class CacheStorageType(Enum):
    """Supported cache storage backend types."""

    LOCAL = "local"
    S3 = "s3"
    MEMORY = "memory"  # For testing and small datasets


@dataclass
class CachedObject:
    """
    Represents an object stored in the cache.

    Attributes:
        key: Cache key (content hash or structured key)
        size_bytes: Size in bytes
        checksum: Content checksum (SHA-256)
        tier: Current storage tier
        content_type: MIME type
        created_at: When the object was cached
        accessed_at: Last access timestamp
        access_count: Number of times accessed
        metadata: Additional metadata
    """

    key: str
    size_bytes: int
    checksum: str
    tier: CacheStorageTier = CacheStorageTier.WARM
    content_type: Optional[str] = None
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "tier": self.tier.value,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedObject":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            tier=CacheStorageTier(data.get("tier", "warm")),
            content_type=data.get("content_type"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            accessed_at=datetime.fromisoformat(data["accessed_at"])
            if data.get("accessed_at")
            else None,
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheStorageConfig:
    """
    Configuration for cache storage backends.

    Attributes:
        storage_type: Type of storage backend
        root_path: Root path or bucket name
        tier: Storage tier for this backend
        max_size_bytes: Maximum storage capacity
        region: Cloud region (for S3)
        endpoint_url: Custom endpoint URL (for S3-compatible)
        credentials: Credential configuration
    """

    storage_type: CacheStorageType
    root_path: str
    tier: CacheStorageTier = CacheStorageTier.WARM
    max_size_bytes: Optional[int] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.max_size_bytes is not None and self.max_size_bytes < 1024:
            raise ValueError(
                f"max_size_bytes must be >= 1024, got {self.max_size_bytes}"
            )


class CacheStorageBackend(ABC):
    """
    Abstract base class for cache storage backends.

    Provides a consistent interface for cache storage operations
    optimized for geospatial data caching patterns.
    """

    def __init__(self, config: CacheStorageConfig):
        """
        Initialize cache storage backend.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._total_size: int = 0

    @property
    @abstractmethod
    def storage_type(self) -> CacheStorageType:
        """Return the storage type."""
        pass

    @property
    def tier(self) -> CacheStorageTier:
        """Return the storage tier."""
        return self.config.tier

    @property
    def total_size(self) -> int:
        """Return total size of cached data in bytes."""
        return self._total_size

    @property
    def available_space(self) -> Optional[int]:
        """Return available space in bytes, or None if unlimited."""
        if self.config.max_size_bytes is None:
            return None
        return max(0, self.config.max_size_bytes - self._total_size)

    @abstractmethod
    def put(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """
        Store data in the cache.

        Args:
            key: Cache key
            data: Data bytes to store
            content_type: MIME type
            metadata: Additional metadata

        Returns:
            CachedObject with storage details
        """
        pass

    @abstractmethod
    def put_file(
        self,
        key: str,
        local_path: Union[str, Path],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """
        Store a file in the cache.

        Args:
            key: Cache key
            local_path: Path to local file
            content_type: MIME type
            metadata: Additional metadata

        Returns:
            CachedObject with storage details
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Tuple[bytes, CachedObject]]:
        """
        Retrieve data from the cache.

        Args:
            key: Cache key

        Returns:
            Tuple of (data bytes, CachedObject) or None if not found
        """
        pass

    @abstractmethod
    def get_file(
        self,
        key: str,
        local_path: Union[str, Path],
    ) -> Optional[CachedObject]:
        """
        Retrieve a file from the cache.

        Args:
            key: Cache key
            local_path: Destination path

        Returns:
            CachedObject if found, None otherwise
        """
        pass

    @abstractmethod
    def get_range(
        self,
        key: str,
        start: int,
        end: int,
    ) -> Optional[bytes]:
        """
        Retrieve a byte range from cached data.

        Optimized for COG/Zarr access patterns that read specific
        tile ranges rather than entire files.

        Args:
            key: Cache key
            start: Start byte offset
            end: End byte offset (exclusive)

        Returns:
            Bytes in range or None if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        pass

    @abstractmethod
    def info(self, key: str) -> Optional[CachedObject]:
        """
        Get metadata for a cached object without retrieving data.

        Args:
            key: Cache key

        Returns:
            CachedObject if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an object from the cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_many(self, keys: List[str]) -> List[str]:
        """
        Delete multiple objects from the cache.

        Args:
            keys: List of cache keys

        Returns:
            List of successfully deleted keys
        """
        pass

    @abstractmethod
    def list_keys(
        self,
        prefix: str = "",
        max_results: Optional[int] = None,
    ) -> Iterator[str]:
        """
        List cache keys.

        Args:
            prefix: Key prefix filter
            max_results: Maximum number of results

        Yields:
            Cache keys matching the prefix
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all objects from the cache.

        Returns:
            Number of objects deleted
        """
        pass

    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def compute_file_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class LocalCacheStorage(CacheStorageBackend):
    """
    Local filesystem cache storage.

    Stores cached data in a local directory with metadata sidecar files.
    Supports atomic writes and efficient partial reads.
    """

    METADATA_SUFFIX = ".cache.json"

    @property
    def storage_type(self) -> CacheStorageType:
        return CacheStorageType.LOCAL

    def __init__(self, config: CacheStorageConfig):
        """Initialize local cache storage."""
        super().__init__(config)
        self._root = Path(config.root_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._recalculate_size()
        logger.info(
            f"Initialized local cache storage at {self._root} "
            f"(tier={config.tier.value}, size={self._total_size} bytes)"
        )

    def _recalculate_size(self):
        """Recalculate total size of cached data."""
        total = 0
        for path in self._root.rglob("*"):
            if path.is_file() and not path.name.endswith(self.METADATA_SUFFIX):
                total += path.stat().st_size
        self._total_size = total

    def _get_path(self, key: str) -> Path:
        """Get filesystem path for a cache key."""
        # Use first 2 chars of key as subdirectory to avoid too many files
        # in one directory
        safe_key = key.replace("/", "_").replace("\\", "_")
        if len(safe_key) >= 2:
            subdir = safe_key[:2]
            return self._root / subdir / safe_key
        return self._root / safe_key

    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for a cache key."""
        return Path(str(self._get_path(key)) + self.METADATA_SUFFIX)

    def _write_metadata(self, key: str, obj: CachedObject):
        """Write metadata sidecar file."""
        meta_path = self._get_metadata_path(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(obj.to_dict(), f, indent=2)

    def _read_metadata(self, key: str) -> Optional[CachedObject]:
        """Read metadata sidecar file."""
        meta_path = self._get_metadata_path(key)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return CachedObject.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Invalid metadata for key {key}")
            return None

    def _update_access(self, key: str, obj: CachedObject) -> CachedObject:
        """Update access timestamp and count."""
        obj.accessed_at = datetime.now(timezone.utc)
        obj.access_count += 1
        self._write_metadata(key, obj)
        return obj

    def put(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store data in the cache."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check space
        if self.config.max_size_bytes is not None:
            available = self.config.max_size_bytes - self._total_size
            if len(data) > available:
                raise IOError(
                    f"Insufficient cache space: need {len(data)} bytes, "
                    f"available {available} bytes"
                )

        # Atomic write
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(data)
            temp_path.rename(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        checksum = self.compute_checksum(data)
        now = datetime.now(timezone.utc)

        obj = CachedObject(
            key=key,
            size_bytes=len(data),
            checksum=checksum,
            tier=self.tier,
            content_type=content_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
        )

        self._write_metadata(key, obj)
        self._total_size += len(data)

        logger.debug(f"Cached {len(data)} bytes at key {key}")
        return obj

    def put_file(
        self,
        key: str,
        local_path: Union[str, Path],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store a file in the cache."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        size = local_path.stat().st_size
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check space
        if self.config.max_size_bytes is not None:
            available = self.config.max_size_bytes - self._total_size
            if size > available:
                raise IOError(
                    f"Insufficient cache space: need {size} bytes, "
                    f"available {available} bytes"
                )

        # Atomic copy
        temp_path = path.with_suffix(".tmp")
        try:
            shutil.copy2(local_path, temp_path)
            temp_path.rename(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        checksum = self.compute_file_checksum(path)
        now = datetime.now(timezone.utc)

        obj = CachedObject(
            key=key,
            size_bytes=size,
            checksum=checksum,
            tier=self.tier,
            content_type=content_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
        )

        self._write_metadata(key, obj)
        self._total_size += size

        logger.debug(f"Cached file {local_path} at key {key} ({size} bytes)")
        return obj

    def get(self, key: str) -> Optional[Tuple[bytes, CachedObject]]:
        """Retrieve data from the cache."""
        path = self._get_path(key)
        if not path.exists():
            return None

        obj = self._read_metadata(key)
        if obj is None:
            # Metadata missing, create basic info
            stat = path.stat()
            obj = CachedObject(
                key=key,
                size_bytes=stat.st_size,
                checksum="",
                tier=self.tier,
            )

        with open(path, "rb") as f:
            data = f.read()

        obj = self._update_access(key, obj)
        return data, obj

    def get_file(
        self,
        key: str,
        local_path: Union[str, Path],
    ) -> Optional[CachedObject]:
        """Retrieve a file from the cache."""
        path = self._get_path(key)
        if not path.exists():
            return None

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic copy
        temp_path = local_path.with_suffix(".tmp")
        shutil.copy2(path, temp_path)
        temp_path.rename(local_path)

        obj = self._read_metadata(key)
        if obj is None:
            stat = path.stat()
            obj = CachedObject(
                key=key,
                size_bytes=stat.st_size,
                checksum="",
                tier=self.tier,
            )

        obj = self._update_access(key, obj)
        return obj

    def get_range(
        self,
        key: str,
        start: int,
        end: int,
    ) -> Optional[bytes]:
        """Retrieve a byte range from cached data."""
        path = self._get_path(key)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self._get_path(key).exists()

    def info(self, key: str) -> Optional[CachedObject]:
        """Get metadata for a cached object."""
        path = self._get_path(key)
        if not path.exists():
            return None

        obj = self._read_metadata(key)
        if obj is None:
            stat = path.stat()
            obj = CachedObject(
                key=key,
                size_bytes=stat.st_size,
                checksum="",
                tier=self.tier,
                accessed_at=datetime.fromtimestamp(stat.st_atime, tz=timezone.utc),
            )
        return obj

    def delete(self, key: str) -> bool:
        """Delete an object from the cache."""
        path = self._get_path(key)
        meta_path = self._get_metadata_path(key)

        if not path.exists():
            return False

        size = path.stat().st_size
        path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        self._total_size -= size
        logger.debug(f"Deleted cached object {key} ({size} bytes)")
        return True

    def delete_many(self, keys: List[str]) -> List[str]:
        """Delete multiple objects from the cache."""
        deleted = []
        for key in keys:
            if self.delete(key):
                deleted.append(key)
        return deleted

    def list_keys(
        self,
        prefix: str = "",
        max_results: Optional[int] = None,
    ) -> Iterator[str]:
        """List cache keys."""
        count = 0
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith(self.METADATA_SUFFIX):
                continue

            # Extract key from path
            try:
                rel_path = path.relative_to(self._root)
                parts = rel_path.parts
                if len(parts) == 2:
                    key = parts[1]
                else:
                    key = str(rel_path)

                if prefix and not key.startswith(prefix):
                    continue

                if max_results and count >= max_results:
                    return

                yield key
                count += 1
            except ValueError:
                continue

    def clear(self) -> int:
        """Clear all objects from the cache."""
        count = 0
        for key in list(self.list_keys()):
            if self.delete(key):
                count += 1
        self._total_size = 0
        logger.info(f"Cleared {count} objects from cache")
        return count


class MemoryCacheStorage(CacheStorageBackend):
    """
    In-memory cache storage for testing and small datasets.

    Stores data in a dictionary, suitable for testing or caching
    small frequently-accessed datasets.
    """

    @property
    def storage_type(self) -> CacheStorageType:
        return CacheStorageType.MEMORY

    def __init__(self, config: CacheStorageConfig):
        """Initialize memory cache storage."""
        super().__init__(config)
        self._data: Dict[str, bytes] = {}
        self._metadata: Dict[str, CachedObject] = {}
        logger.info(f"Initialized memory cache storage (tier={config.tier.value})")

    def put(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store data in the cache."""
        # Check space
        if self.config.max_size_bytes is not None:
            available = self.config.max_size_bytes - self._total_size
            if len(data) > available:
                raise IOError(
                    f"Insufficient cache space: need {len(data)} bytes, "
                    f"available {available} bytes"
                )

        checksum = self.compute_checksum(data)
        now = datetime.now(timezone.utc)

        obj = CachedObject(
            key=key,
            size_bytes=len(data),
            checksum=checksum,
            tier=self.tier,
            content_type=content_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
        )

        self._data[key] = data
        self._metadata[key] = obj
        self._total_size += len(data)

        return obj

    def put_file(
        self,
        key: str,
        local_path: Union[str, Path],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store a file in the cache."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        with open(local_path, "rb") as f:
            data = f.read()

        return self.put(key, data, content_type, metadata)

    def get(self, key: str) -> Optional[Tuple[bytes, CachedObject]]:
        """Retrieve data from the cache."""
        if key not in self._data:
            return None

        data = self._data[key]
        obj = self._metadata[key]
        obj.accessed_at = datetime.now(timezone.utc)
        obj.access_count += 1

        return data, obj

    def get_file(
        self,
        key: str,
        local_path: Union[str, Path],
    ) -> Optional[CachedObject]:
        """Retrieve a file from the cache."""
        result = self.get(key)
        if result is None:
            return None

        data, obj = result
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(data)

        return obj

    def get_range(
        self,
        key: str,
        start: int,
        end: int,
    ) -> Optional[bytes]:
        """Retrieve a byte range from cached data."""
        if key not in self._data:
            return None
        return self._data[key][start:end]

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._data

    def info(self, key: str) -> Optional[CachedObject]:
        """Get metadata for a cached object."""
        return self._metadata.get(key)

    def delete(self, key: str) -> bool:
        """Delete an object from the cache."""
        if key not in self._data:
            return False

        size = len(self._data[key])
        del self._data[key]
        del self._metadata[key]
        self._total_size -= size

        return True

    def delete_many(self, keys: List[str]) -> List[str]:
        """Delete multiple objects from the cache."""
        deleted = []
        for key in keys:
            if self.delete(key):
                deleted.append(key)
        return deleted

    def list_keys(
        self,
        prefix: str = "",
        max_results: Optional[int] = None,
    ) -> Iterator[str]:
        """List cache keys."""
        count = 0
        for key in sorted(self._data.keys()):
            if prefix and not key.startswith(prefix):
                continue

            if max_results and count >= max_results:
                return

            yield key
            count += 1

    def clear(self) -> int:
        """Clear all objects from the cache."""
        count = len(self._data)
        self._data.clear()
        self._metadata.clear()
        self._total_size = 0
        return count


class S3CacheStorage(CacheStorageBackend):
    """
    S3-compatible cache storage.

    Stores cached data in S3 or S3-compatible storage (MinIO, LocalStack).
    Optimized for cloud deployments with support for range requests.
    """

    @property
    def storage_type(self) -> CacheStorageType:
        return CacheStorageType.S3

    def __init__(self, config: CacheStorageConfig):
        """Initialize S3 cache storage."""
        super().__init__(config)
        self._bucket = config.root_path
        self._client = None
        self._init_client()
        logger.info(
            f"Initialized S3 cache storage at bucket {self._bucket} "
            f"(tier={config.tier.value})"
        )

    def _init_client(self):
        """Initialize boto3 S3 client."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError("boto3 is required for S3 cache storage")

        boto_config = BotoConfig(
            retries={"max_attempts": 3},
            connect_timeout=30,
            read_timeout=30,
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

        self._client = boto3.client("s3", **kwargs)

    def _get_key(self, cache_key: str) -> str:
        """Get S3 key for a cache key."""
        # Use prefix for organization
        safe_key = cache_key.replace("\\", "/")
        if len(safe_key) >= 2:
            prefix = safe_key[:2]
            return f"cache/{prefix}/{safe_key}"
        return f"cache/{safe_key}"

    def _get_meta_key(self, cache_key: str) -> str:
        """Get S3 key for metadata."""
        return self._get_key(cache_key) + ".meta.json"

    def put(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store data in the cache."""
        s3_key = self._get_key(key)
        checksum = self.compute_checksum(data)
        now = datetime.now(timezone.utc)

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=data,
            **extra_args,
        )

        obj = CachedObject(
            key=key,
            size_bytes=len(data),
            checksum=checksum,
            tier=self.tier,
            content_type=content_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
        )

        # Store metadata
        meta_key = self._get_meta_key(key)
        self._client.put_object(
            Bucket=self._bucket,
            Key=meta_key,
            Body=json.dumps(obj.to_dict()).encode(),
            ContentType="application/json",
        )

        self._total_size += len(data)
        logger.debug(f"Cached {len(data)} bytes at key {key} in S3")

        return obj

    def put_file(
        self,
        key: str,
        local_path: Union[str, Path],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CachedObject:
        """Store a file in the cache."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        s3_key = self._get_key(key)
        checksum = self.compute_file_checksum(local_path)
        size = local_path.stat().st_size
        now = datetime.now(timezone.utc)

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        self._client.upload_file(
            str(local_path),
            self._bucket,
            s3_key,
            ExtraArgs=extra_args if extra_args else None,
        )

        obj = CachedObject(
            key=key,
            size_bytes=size,
            checksum=checksum,
            tier=self.tier,
            content_type=content_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
        )

        # Store metadata
        meta_key = self._get_meta_key(key)
        self._client.put_object(
            Bucket=self._bucket,
            Key=meta_key,
            Body=json.dumps(obj.to_dict()).encode(),
            ContentType="application/json",
        )

        self._total_size += size
        return obj

    def get(self, key: str) -> Optional[Tuple[bytes, CachedObject]]:
        """Retrieve data from the cache."""
        s3_key = self._get_key(key)

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            data = response["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

        # Get metadata
        obj = self._read_metadata(key)
        if obj is None:
            obj = CachedObject(
                key=key,
                size_bytes=len(data),
                checksum=self.compute_checksum(data),
                tier=self.tier,
            )

        obj.accessed_at = datetime.now(timezone.utc)
        obj.access_count += 1

        return data, obj

    def _read_metadata(self, key: str) -> Optional[CachedObject]:
        """Read metadata from S3."""
        meta_key = self._get_meta_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=meta_key)
            data = json.loads(response["Body"].read().decode())
            return CachedObject.from_dict(data)
        except Exception:
            return None

    def get_file(
        self,
        key: str,
        local_path: Union[str, Path],
    ) -> Optional[CachedObject]:
        """Retrieve a file from the cache."""
        s3_key = self._get_key(key)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._client.download_file(self._bucket, s3_key, str(local_path))
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

        obj = self._read_metadata(key)
        if obj is None:
            stat = local_path.stat()
            obj = CachedObject(
                key=key,
                size_bytes=stat.st_size,
                checksum="",
                tier=self.tier,
            )

        obj.accessed_at = datetime.now(timezone.utc)
        obj.access_count += 1

        return obj

    def get_range(
        self,
        key: str,
        start: int,
        end: int,
    ) -> Optional[bytes]:
        """Retrieve a byte range from cached data."""
        s3_key = self._get_key(key)

        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=s3_key,
                Range=f"bytes={start}-{end - 1}",
            )
            return response["Body"].read()
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        s3_key = self._get_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except Exception:
            return False

    def info(self, key: str) -> Optional[CachedObject]:
        """Get metadata for a cached object."""
        if not self.exists(key):
            return None
        return self._read_metadata(key)

    def delete(self, key: str) -> bool:
        """Delete an object from the cache."""
        s3_key = self._get_key(key)
        meta_key = self._get_meta_key(key)

        try:
            # Get size before delete
            obj = self.info(key)
            size = obj.size_bytes if obj else 0

            self._client.delete_object(Bucket=self._bucket, Key=s3_key)
            self._client.delete_object(Bucket=self._bucket, Key=meta_key)

            self._total_size -= size
            return True
        except Exception:
            return False

    def delete_many(self, keys: List[str]) -> List[str]:
        """Delete multiple objects from the cache."""
        if not keys:
            return []

        objects = []
        for key in keys:
            objects.append({"Key": self._get_key(key)})
            objects.append({"Key": self._get_meta_key(key)})

        try:
            response = self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": objects},
            )
            # Extract base keys from deleted objects
            deleted = set()
            for obj in response.get("Deleted", []):
                s3_key = obj["Key"]
                # Extract original key from S3 key
                if s3_key.startswith("cache/") and not s3_key.endswith(".meta.json"):
                    parts = s3_key.split("/", 2)
                    if len(parts) == 3:
                        deleted.add(parts[2])
            return list(deleted)
        except Exception:
            return []

    def list_keys(
        self,
        prefix: str = "",
        max_results: Optional[int] = None,
    ) -> Iterator[str]:
        """List cache keys."""
        s3_prefix = f"cache/{prefix[:2]}/{prefix}" if len(prefix) >= 2 else "cache/"
        paginator = self._client.get_paginator("list_objects_v2")

        count = 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                if s3_key.endswith(".meta.json"):
                    continue

                # Extract original key
                if s3_key.startswith("cache/"):
                    parts = s3_key.split("/", 2)
                    if len(parts) == 3:
                        key = parts[2]
                        if prefix and not key.startswith(prefix):
                            continue

                        if max_results and count >= max_results:
                            return

                        yield key
                        count += 1

    def clear(self) -> int:
        """Clear all objects from the cache."""
        count = 0
        keys = list(self.list_keys())
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            deleted = self.delete_many(batch)
            count += len(deleted)
        self._total_size = 0
        return count


def create_cache_storage(config: CacheStorageConfig) -> CacheStorageBackend:
    """
    Factory function to create appropriate cache storage backend.

    Args:
        config: Cache storage configuration

    Returns:
        CacheStorageBackend instance

    Raises:
        ValueError: If storage type is not supported
    """
    if config.storage_type == CacheStorageType.LOCAL:
        return LocalCacheStorage(config)
    elif config.storage_type == CacheStorageType.MEMORY:
        return MemoryCacheStorage(config)
    elif config.storage_type == CacheStorageType.S3:
        return S3CacheStorage(config)
    else:
        raise ValueError(f"Unsupported cache storage type: {config.storage_type}")
