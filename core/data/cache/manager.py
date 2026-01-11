"""
Cache Manager for Geospatial Data Products.

Provides lifecycle management for cached data products including:
- Cache entry registration and tracking
- TTL-based expiration policies
- LRU eviction when size limits are exceeded
- Access pattern tracking for analytics
- Automatic cleanup of expired entries
- Thread-safe operations

The cache manager integrates with the storage backends to provide
efficient storage and retrieval of cached products.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class CacheEntryStatus(Enum):
    """Status of a cache entry."""

    PENDING = "pending"  # Entry registered but data not yet stored
    ACTIVE = "active"  # Entry active and available
    EXPIRED = "expired"  # Entry past TTL, pending cleanup
    EVICTED = "evicted"  # Entry evicted due to size limits
    INVALID = "invalid"  # Entry invalidated by user/system
    DELETED = "deleted"  # Entry deleted from cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live (expire oldest first)
    SIZE = "size"  # Largest entries first


@dataclass
class CacheConfig:
    """
    Configuration for cache manager.

    Attributes:
        max_size_bytes: Maximum cache size in bytes (0 = unlimited)
        max_entries: Maximum number of cache entries (0 = unlimited)
        default_ttl_seconds: Default TTL for new entries (0 = no expiration)
        eviction_policy: Policy for evicting entries when full
        cleanup_interval_seconds: Interval between cleanup runs
        db_path: Path to SQLite database for cache metadata
        enable_statistics: Whether to track access statistics
    """

    max_size_bytes: int = 0
    max_entries: int = 0
    default_ttl_seconds: int = 86400  # 24 hours
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    cleanup_interval_seconds: int = 3600  # 1 hour
    db_path: Optional[Path] = None
    enable_statistics: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_size_bytes < 0:
            raise ValueError(f"max_size_bytes must be >= 0, got {self.max_size_bytes}")
        if self.max_entries < 0:
            raise ValueError(f"max_entries must be >= 0, got {self.max_entries}")
        if self.default_ttl_seconds < 0:
            raise ValueError(
                f"default_ttl_seconds must be >= 0, got {self.default_ttl_seconds}"
            )
        if self.cleanup_interval_seconds < 60:
            raise ValueError(
                f"cleanup_interval_seconds must be >= 60, got {self.cleanup_interval_seconds}"
            )


@dataclass
class CacheEntry:
    """
    Represents an entry in the cache.

    Attributes:
        cache_key: Unique cache key
        storage_key: Key in storage backend
        data_type: Type of cached data (optical, sar, dem, etc.)
        size_bytes: Size in bytes
        checksum: Content checksum
        status: Current entry status
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        expires_at: Expiration timestamp
        access_count: Number of times accessed
        metadata: Additional metadata (bbox, temporal, etc.)
        tags: Searchable tags
    """

    cache_key: str
    storage_key: str
    data_type: str
    size_bytes: int
    checksum: str
    status: CacheEntryStatus = CacheEntryStatus.ACTIVE
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_key": self.cache_key,
            "storage_key": self.storage_key,
            "data_type": self.data_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            cache_key=data["cache_key"],
            storage_key=data["storage_key"],
            data_type=data["data_type"],
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            status=CacheEntryStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            accessed_at=datetime.fromisoformat(data["accessed_at"])
            if data.get("accessed_at")
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class CacheStatistics:
    """Statistics about cache usage."""

    total_entries: int = 0
    active_entries: int = 0
    expired_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "active_entries": self.active_entries,
            "expired_entries": self.expired_entries,
            "total_size_bytes": self.total_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
        }


class CacheManager:
    """
    Manages cache lifecycle for geospatial data products.

    Provides thread-safe operations for registering, accessing, and
    cleaning up cached data products. Uses SQLite for persistent
    metadata storage and integrates with storage backends.
    """

    def __init__(
        self,
        config: CacheConfig,
        storage_backend: Optional[Any] = None,
    ):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
            storage_backend: Storage backend for data (from persistence.storage)
        """
        self.config = config
        self.storage_backend = storage_backend
        self._lock = threading.RLock()
        self._statistics = CacheStatistics()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Initialize database
        self._db_path = config.db_path or Path.home() / ".multiverse_dive" / "cache.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        logger.info(
            f"CacheManager initialized with db={self._db_path}, "
            f"max_size={config.max_size_bytes}, max_entries={config.max_entries}"
        )

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    storage_key TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    tags TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_status
                ON cache_entries(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON cache_entries(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_accessed
                ON cache_entries(accessed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_data_type
                ON cache_entries(data_type)
            """)
            # Statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_statistics (
                    stat_name TEXT PRIMARY KEY,
                    stat_value INTEGER DEFAULT 0
                )
            """)
            # Initialize statistics if not present
            for stat in ["hits", "misses", "evictions", "expirations"]:
                conn.execute(
                    "INSERT OR IGNORE INTO cache_statistics (stat_name, stat_value) VALUES (?, 0)",
                    (stat,),
                )
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def generate_cache_key(
        self,
        provider: str,
        dataset_id: str,
        bbox: Optional[List[float]] = None,
        temporal: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a deterministic cache key.

        Args:
            provider: Data provider name
            dataset_id: Dataset identifier
            bbox: Bounding box [west, south, east, north]
            temporal: Temporal extent dict
            extra: Additional parameters

        Returns:
            Deterministic cache key (SHA256 hash prefix)
        """
        key_parts = [provider, dataset_id]

        if bbox:
            # Round to 4 decimal places for consistency
            key_parts.append(
                f"bbox:{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
            )

        if temporal:
            if "start" in temporal:
                key_parts.append(f"start:{temporal['start']}")
            if "end" in temporal:
                key_parts.append(f"end:{temporal['end']}")

        if extra:
            # Sort for determinism
            for k, v in sorted(extra.items()):
                key_parts.append(f"{k}:{v}")

        key_string = "|".join(key_parts)
        hash_digest = hashlib.sha256(key_string.encode()).hexdigest()
        return hash_digest[:32]

    def put(
        self,
        cache_key: str,
        storage_key: str,
        data_type: str,
        size_bytes: int,
        checksum: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> CacheEntry:
        """
        Register a cache entry.

        Args:
            cache_key: Unique cache key
            storage_key: Key in storage backend
            data_type: Type of data (optical, sar, dem, etc.)
            size_bytes: Size in bytes
            checksum: Content checksum
            ttl_seconds: Time to live (uses default if None)
            metadata: Additional metadata
            tags: Searchable tags

        Returns:
            Created CacheEntry

        Raises:
            CacheFullError: If cache is full and eviction fails
        """
        with self._lock:
            # Check if we need to evict
            self._ensure_capacity(size_bytes)

            now = datetime.now(timezone.utc)
            ttl = ttl_seconds if ttl_seconds is not None else self.config.default_ttl_seconds
            expires_at = now + timedelta(seconds=ttl) if ttl > 0 else None

            entry = CacheEntry(
                cache_key=cache_key,
                storage_key=storage_key,
                data_type=data_type,
                size_bytes=size_bytes,
                checksum=checksum,
                status=CacheEntryStatus.ACTIVE,
                created_at=now,
                accessed_at=now,
                expires_at=expires_at,
                access_count=0,
                metadata=metadata or {},
                tags=tags or set(),
            )

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key, storage_key, data_type, size_bytes, checksum,
                     status, created_at, accessed_at, expires_at, access_count,
                     metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.cache_key,
                        entry.storage_key,
                        entry.data_type,
                        entry.size_bytes,
                        entry.checksum,
                        entry.status.value,
                        entry.created_at.isoformat(),
                        entry.accessed_at.isoformat(),
                        entry.expires_at.isoformat() if entry.expires_at else None,
                        entry.access_count,
                        json.dumps(entry.metadata),
                        json.dumps(list(entry.tags)),
                    ),
                )
                conn.commit()

            logger.debug(f"Cached entry: {cache_key} -> {storage_key}")
            return entry

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        """
        Get a cache entry and update access statistics.

        Args:
            cache_key: Cache key to look up

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    self._record_miss()
                    return None

                entry = self._row_to_entry(row)

                # Check if expired
                if entry.is_expired:
                    self._mark_expired(cache_key)
                    self._record_miss()
                    return None

                # Check if still active
                if entry.status != CacheEntryStatus.ACTIVE:
                    self._record_miss()
                    return None

                # Update access statistics
                now = datetime.now(timezone.utc)
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                    """,
                    (now.isoformat(), cache_key),
                )
                conn.commit()
                entry.accessed_at = now
                entry.access_count += 1

                self._record_hit()
                return entry

    def contains(self, cache_key: str) -> bool:
        """
        Check if cache key exists and is valid.

        Args:
            cache_key: Cache key to check

        Returns:
            True if entry exists and is active
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT status, expires_at FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return False

                if row["status"] != "active":
                    return False

                if row["expires_at"]:
                    expires_at = datetime.fromisoformat(row["expires_at"])
                    if datetime.now(timezone.utc) > expires_at:
                        return False

                return True

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if entry was invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    UPDATE cache_entries
                    SET status = ?
                    WHERE cache_key = ? AND status = ?
                    """,
                    (CacheEntryStatus.INVALID.value, cache_key, CacheEntryStatus.ACTIVE.value),
                )
                conn.commit()
                invalidated = result.rowcount > 0

                if invalidated:
                    logger.debug(f"Invalidated cache entry: {cache_key}")

                return invalidated

    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: SQL LIKE pattern for cache_key

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    UPDATE cache_entries
                    SET status = ?
                    WHERE cache_key LIKE ? AND status = ?
                    """,
                    (CacheEntryStatus.INVALID.value, pattern, CacheEntryStatus.ACTIVE.value),
                )
                conn.commit()
                count = result.rowcount
                logger.debug(f"Invalidated {count} entries matching pattern: {pattern}")
                return count

    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate cache entries with a specific tag.

        Args:
            tag: Tag to match

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                # SQLite JSON search
                pattern = f'%"{tag}"%'
                result = conn.execute(
                    """
                    UPDATE cache_entries
                    SET status = ?
                    WHERE tags LIKE ? AND status = ?
                    """,
                    (CacheEntryStatus.INVALID.value, pattern, CacheEntryStatus.ACTIVE.value),
                )
                conn.commit()
                count = result.rowcount
                logger.debug(f"Invalidated {count} entries with tag: {tag}")
                return count

    def delete(self, cache_key: str, delete_storage: bool = True) -> bool:
        """
        Delete a cache entry and optionally its storage.

        Args:
            cache_key: Cache key to delete
            delete_storage: Whether to delete from storage backend

        Returns:
            True if entry was deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get storage key first
                row = conn.execute(
                    "SELECT storage_key FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return False

                storage_key = row["storage_key"]

                # Delete from storage if requested
                if delete_storage and self.storage_backend is not None:
                    try:
                        self.storage_backend.delete(storage_key)
                    except Exception as e:
                        logger.warning(f"Failed to delete storage for {cache_key}: {e}")

                # Delete from database
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()

                logger.debug(f"Deleted cache entry: {cache_key}")
                return True

    def list_entries(
        self,
        data_type: Optional[str] = None,
        status: Optional[CacheEntryStatus] = None,
        tag: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[CacheEntry]:
        """
        List cache entries with optional filters.

        Args:
            data_type: Filter by data type
            status: Filter by status
            tag: Filter by tag
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of matching CacheEntry objects
        """
        with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM cache_entries WHERE 1=1"
                params: List[Any] = []

                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type)

                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                if tag:
                    query += " AND tags LIKE ?"
                    params.append(f'%"{tag}"%')

                query += " ORDER BY accessed_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                rows = conn.execute(query, params).fetchall()
                return [self._row_to_entry(row) for row in rows]

    def get_statistics(self) -> CacheStatistics:
        """
        Get current cache statistics.

        Returns:
            CacheStatistics object
        """
        with self._lock:
            with self._get_connection() as conn:
                # Count entries by status
                active = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE status = ?",
                    (CacheEntryStatus.ACTIVE.value,),
                ).fetchone()[0]

                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE status = ?",
                    (CacheEntryStatus.EXPIRED.value,),
                ).fetchone()[0]

                total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

                # Sum size
                size_result = conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries WHERE status = ?",
                    (CacheEntryStatus.ACTIVE.value,),
                ).fetchone()[0]

                # Get hit/miss statistics
                stats = {}
                for row in conn.execute("SELECT stat_name, stat_value FROM cache_statistics"):
                    stats[row["stat_name"]] = row["stat_value"]

                return CacheStatistics(
                    total_entries=total,
                    active_entries=active,
                    expired_entries=expired,
                    total_size_bytes=size_result,
                    hits=stats.get("hits", 0),
                    misses=stats.get("misses", 0),
                    evictions=stats.get("evictions", 0),
                    expirations=stats.get("expirations", 0),
                )

    def cleanup_expired(self, delete_storage: bool = True) -> int:
        """
        Clean up expired cache entries.

        Args:
            delete_storage: Whether to delete from storage backend

        Returns:
            Number of entries cleaned up
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            cleaned = 0

            with self._get_connection() as conn:
                # Find expired entries
                rows = conn.execute(
                    """
                    SELECT cache_key, storage_key FROM cache_entries
                    WHERE (expires_at IS NOT NULL AND expires_at < ?)
                    OR status IN (?, ?, ?)
                    """,
                    (
                        now.isoformat(),
                        CacheEntryStatus.EXPIRED.value,
                        CacheEntryStatus.EVICTED.value,
                        CacheEntryStatus.INVALID.value,
                    ),
                ).fetchall()

                for row in rows:
                    cache_key = row["cache_key"]
                    storage_key = row["storage_key"]

                    # Delete from storage if requested
                    if delete_storage and self.storage_backend is not None:
                        try:
                            self.storage_backend.delete(storage_key)
                        except Exception as e:
                            logger.warning(f"Failed to delete storage for {cache_key}: {e}")

                    # Delete from database
                    conn.execute(
                        "DELETE FROM cache_entries WHERE cache_key = ?",
                        (cache_key,),
                    )
                    cleaned += 1

                # Update expiration counter
                if cleaned > 0:
                    conn.execute(
                        """
                        UPDATE cache_statistics
                        SET stat_value = stat_value + ?
                        WHERE stat_name = 'expirations'
                        """,
                        (cleaned,),
                    )

                conn.commit()

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired cache entries")

            return cleaned

    def clear(self, delete_storage: bool = True) -> int:
        """
        Clear all cache entries.

        Args:
            delete_storage: Whether to delete from storage backend

        Returns:
            Number of entries cleared
        """
        with self._lock:
            cleared = 0

            with self._get_connection() as conn:
                if delete_storage and self.storage_backend is not None:
                    # Get all storage keys first
                    rows = conn.execute(
                        "SELECT storage_key FROM cache_entries"
                    ).fetchall()

                    for row in rows:
                        try:
                            self.storage_backend.delete(row["storage_key"])
                        except Exception as e:
                            logger.warning(f"Failed to delete storage: {e}")

                # Delete all entries
                result = conn.execute("DELETE FROM cache_entries")
                cleared = result.rowcount

                # Reset statistics
                conn.execute("UPDATE cache_statistics SET stat_value = 0")
                conn.commit()

            logger.info(f"Cleared {cleared} cache entries")
            return cleared

    def start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="cache-cleanup",
        )
        self._cleanup_thread.start()
        logger.info("Started cache cleanup thread")

    def stop_cleanup_thread(self, timeout: float = 5.0) -> None:
        """
        Stop the background cleanup thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            return

        self._shutdown_event.set()
        self._cleanup_thread.join(timeout=timeout)
        logger.info("Stopped cache cleanup thread")

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

            # Wait for next cleanup or shutdown
            self._shutdown_event.wait(timeout=self.config.cleanup_interval_seconds)

    def _ensure_capacity(self, required_bytes: int) -> None:
        """
        Ensure cache has capacity for new entry.

        Args:
            required_bytes: Size of new entry

        Raises:
            CacheFullError: If unable to free enough space
        """
        # Check entry limit
        if self.config.max_entries > 0:
            with self._get_connection() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE status = ?",
                    (CacheEntryStatus.ACTIVE.value,),
                ).fetchone()[0]

                while count >= self.config.max_entries:
                    if not self._evict_one():
                        raise CacheFullError("Cannot evict entries to make room")
                    count -= 1

        # Check size limit
        if self.config.max_size_bytes > 0:
            with self._get_connection() as conn:
                total_size = conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries WHERE status = ?",
                    (CacheEntryStatus.ACTIVE.value,),
                ).fetchone()[0]

                while total_size + required_bytes > self.config.max_size_bytes:
                    if not self._evict_one():
                        raise CacheFullError("Cannot evict entries to make room")

                    total_size = conn.execute(
                        "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries WHERE status = ?",
                        (CacheEntryStatus.ACTIVE.value,),
                    ).fetchone()[0]

    def _evict_one(self) -> bool:
        """
        Evict one entry based on eviction policy.

        Returns:
            True if an entry was evicted
        """
        policy = self.config.eviction_policy

        with self._get_connection() as conn:
            if policy == EvictionPolicy.LRU:
                order_by = "accessed_at ASC"
            elif policy == EvictionPolicy.LFU:
                order_by = "access_count ASC"
            elif policy == EvictionPolicy.FIFO:
                order_by = "created_at ASC"
            elif policy == EvictionPolicy.TTL:
                order_by = "expires_at ASC NULLS LAST"
            elif policy == EvictionPolicy.SIZE:
                order_by = "size_bytes DESC"
            else:
                order_by = "accessed_at ASC"

            row = conn.execute(
                f"""
                SELECT cache_key, storage_key FROM cache_entries
                WHERE status = ?
                ORDER BY {order_by}
                LIMIT 1
                """,
                (CacheEntryStatus.ACTIVE.value,),
            ).fetchone()

            if row is None:
                return False

            cache_key = row["cache_key"]
            storage_key = row["storage_key"]

            # Delete from storage
            if self.storage_backend is not None:
                try:
                    self.storage_backend.delete(storage_key)
                except Exception as e:
                    logger.warning(f"Failed to delete storage during eviction: {e}")

            # Mark as evicted
            conn.execute(
                "UPDATE cache_entries SET status = ? WHERE cache_key = ?",
                (CacheEntryStatus.EVICTED.value, cache_key),
            )

            # Update eviction counter
            conn.execute(
                """
                UPDATE cache_statistics
                SET stat_value = stat_value + 1
                WHERE stat_name = 'evictions'
                """,
            )
            conn.commit()

            logger.debug(f"Evicted cache entry: {cache_key}")
            return True

    def _mark_expired(self, cache_key: str) -> None:
        """Mark an entry as expired."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE cache_entries SET status = ? WHERE cache_key = ?",
                (CacheEntryStatus.EXPIRED.value, cache_key),
            )
            conn.commit()

    def _record_hit(self) -> None:
        """Record a cache hit."""
        if not self.config.enable_statistics:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE cache_statistics
                SET stat_value = stat_value + 1
                WHERE stat_name = 'hits'
                """,
            )
            conn.commit()

    def _record_miss(self) -> None:
        """Record a cache miss."""
        if not self.config.enable_statistics:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE cache_statistics
                SET stat_value = stat_value + 1
                WHERE stat_name = 'misses'
                """,
            )
            conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        """Convert a database row to CacheEntry."""
        return CacheEntry(
            cache_key=row["cache_key"],
            storage_key=row["storage_key"],
            data_type=row["data_type"],
            size_bytes=row["size_bytes"],
            checksum=row["checksum"],
            status=CacheEntryStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            accessed_at=datetime.fromisoformat(row["accessed_at"])
            if row["accessed_at"]
            else None,
            expires_at=datetime.fromisoformat(row["expires_at"])
            if row["expires_at"]
            else None,
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=set(json.loads(row["tags"])) if row["tags"] else set(),
        )


class CacheFullError(Exception):
    """Raised when cache is full and cannot evict entries."""

    pass
