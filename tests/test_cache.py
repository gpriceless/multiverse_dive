"""
Tests for Cache System.

Tests cache storage backends, spatiotemporal indexing, and cache manager
functionality for the ingestion pipeline.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from core.data.cache.storage import (
    CachedObject,
    CacheStorageBackend,
    CacheStorageConfig,
    CacheStorageTier,
    CacheStorageType,
    LocalCacheStorage,
    MemoryCacheStorage,
    create_cache_storage,
)

from core.data.cache.index import (
    BoundingBox,
    IndexEntry,
    SpatialPredicate,
    SpatiotemporalIndex,
    TemporalPredicate,
    TimeRange,
)

from core.data.cache.manager import (
    CacheConfig,
    CacheEntry,
    CacheEntryStatus,
    CacheFullError,
    CacheManager,
    EvictionPolicy,
)


# ==============================================================================
# Storage Backend Tests
# ==============================================================================


class TestCacheStorageConfig:
    """Tests for CacheStorageConfig."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = CacheStorageConfig(
            storage_type=CacheStorageType.LOCAL,
            root_path="/tmp/cache",
            tier=CacheStorageTier.HOT,
            max_size_bytes=1024 * 1024,
        )
        assert config.storage_type == CacheStorageType.LOCAL
        assert config.tier == CacheStorageTier.HOT
        assert config.max_size_bytes == 1024 * 1024

    def test_invalid_max_size(self):
        """Test invalid max_size_bytes."""
        with pytest.raises(ValueError, match="max_size_bytes must be >= 1024"):
            CacheStorageConfig(
                storage_type=CacheStorageType.LOCAL,
                root_path="/tmp/cache",
                max_size_bytes=100,  # Too small
            )


class TestMemoryCacheStorage:
    """Tests for MemoryCacheStorage."""

    @pytest.fixture
    def storage(self):
        """Create memory cache storage."""
        config = CacheStorageConfig(
            storage_type=CacheStorageType.MEMORY,
            root_path="test",
            tier=CacheStorageTier.HOT,
        )
        return MemoryCacheStorage(config)

    def test_put_and_get(self, storage):
        """Test storing and retrieving data."""
        data = b"test data content"
        obj = storage.put("key1", data, content_type="text/plain")

        assert obj.key == "key1"
        assert obj.size_bytes == len(data)
        assert obj.checksum is not None
        assert obj.tier == CacheStorageTier.HOT

        result = storage.get("key1")
        assert result is not None
        retrieved_data, retrieved_obj = result
        assert retrieved_data == data
        assert retrieved_obj.access_count == 1

    def test_get_nonexistent(self, storage):
        """Test getting nonexistent key."""
        result = storage.get("nonexistent")
        assert result is None

    def test_exists(self, storage):
        """Test exists check."""
        assert not storage.exists("key1")
        storage.put("key1", b"data")
        assert storage.exists("key1")

    def test_delete(self, storage):
        """Test deleting data."""
        storage.put("key1", b"data")
        assert storage.exists("key1")

        result = storage.delete("key1")
        assert result is True
        assert not storage.exists("key1")

    def test_delete_nonexistent(self, storage):
        """Test deleting nonexistent key."""
        result = storage.delete("nonexistent")
        assert result is False

    def test_list_keys(self, storage):
        """Test listing keys."""
        storage.put("prefix_a", b"data1")
        storage.put("prefix_b", b"data2")
        storage.put("other", b"data3")

        all_keys = list(storage.list_keys())
        assert len(all_keys) == 3

        prefix_keys = list(storage.list_keys(prefix="prefix"))
        assert len(prefix_keys) == 2
        assert all(k.startswith("prefix") for k in prefix_keys)

    def test_list_keys_with_limit(self, storage):
        """Test listing keys with limit."""
        for i in range(10):
            storage.put(f"key_{i}", b"data")

        keys = list(storage.list_keys(max_results=5))
        assert len(keys) == 5

    def test_clear(self, storage):
        """Test clearing all data."""
        for i in range(5):
            storage.put(f"key_{i}", b"data")

        assert storage.total_size > 0
        count = storage.clear()
        assert count == 5
        assert storage.total_size == 0

    def test_get_range(self, storage):
        """Test getting byte range."""
        data = b"0123456789"
        storage.put("key1", data)

        result = storage.get_range("key1", 2, 7)
        assert result == b"23456"

    def test_max_size_enforcement(self):
        """Test max size enforcement."""
        config = CacheStorageConfig(
            storage_type=CacheStorageType.MEMORY,
            root_path="test",
            max_size_bytes=1024,
        )
        storage = MemoryCacheStorage(config)

        # Store data up to limit
        storage.put("key1", b"x" * 512)
        assert storage.total_size == 512

        # Attempt to store more than remaining space
        with pytest.raises(IOError, match="Insufficient cache space"):
            storage.put("key2", b"x" * 600)

    def test_info(self, storage):
        """Test getting info without data."""
        storage.put("key1", b"data", content_type="text/plain")

        info = storage.info("key1")
        assert info is not None
        assert info.key == "key1"
        assert info.content_type == "text/plain"

    def test_put_file(self, storage):
        """Test storing from file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"file content")
            temp_path = f.name

        try:
            obj = storage.put_file("key1", temp_path)
            assert obj.size_bytes == 12

            result = storage.get("key1")
            assert result is not None
            data, _ = result
            assert data == b"file content"
        finally:
            os.unlink(temp_path)

    def test_get_file(self, storage):
        """Test retrieving to file."""
        storage.put("key1", b"data content")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.bin"
            obj = storage.get_file("key1", output_path)

            assert obj is not None
            assert output_path.exists()
            assert output_path.read_bytes() == b"data content"


class TestLocalCacheStorage:
    """Tests for LocalCacheStorage."""

    @pytest.fixture
    def storage(self):
        """Create local cache storage in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheStorageConfig(
                storage_type=CacheStorageType.LOCAL,
                root_path=tmpdir,
                tier=CacheStorageTier.WARM,
            )
            yield LocalCacheStorage(config)

    def test_put_and_get(self, storage):
        """Test storing and retrieving data."""
        data = b"test data content"
        obj = storage.put("key1", data, content_type="text/plain")

        assert obj.key == "key1"
        assert obj.size_bytes == len(data)
        assert obj.checksum is not None

        result = storage.get("key1")
        assert result is not None
        retrieved_data, _ = result
        assert retrieved_data == data

    def test_atomic_writes(self, storage):
        """Test that writes are atomic."""
        data = b"x" * 10000
        storage.put("key1", data)

        # Verify no .tmp files left behind
        for path in Path(storage._root).rglob("*.tmp"):
            assert False, f"Temp file left behind: {path}"

    def test_metadata_persistence(self, storage):
        """Test metadata is persisted."""
        storage.put(
            "key1",
            b"data",
            content_type="application/octet-stream",
            metadata={"source": "test"},
        )

        obj = storage.info("key1")
        assert obj is not None
        assert obj.content_type == "application/octet-stream"
        assert obj.metadata.get("source") == "test"

    def test_checksum_computation(self, storage):
        """Test checksum is computed correctly."""
        data = b"test data"
        obj = storage.put("key1", data)

        expected_checksum = storage.compute_checksum(data)
        assert obj.checksum == expected_checksum

    def test_get_range(self, storage):
        """Test getting byte range from local storage."""
        data = b"0123456789ABCDEF"
        storage.put("key1", data)

        result = storage.get_range("key1", 5, 10)
        assert result == b"56789"

    def test_directory_structure(self, storage):
        """Test subdirectory organization."""
        # Keys with same 2-char prefix go in same subdir
        storage.put("aa_key1", b"data1")
        storage.put("aa_key2", b"data2")
        storage.put("bb_key1", b"data3")

        # Check directory structure
        subdirs = [d.name for d in Path(storage._root).iterdir() if d.is_dir()]
        assert "aa" in subdirs
        assert "bb" in subdirs


class TestCreateCacheStorage:
    """Tests for create_cache_storage factory."""

    def test_create_local(self):
        """Test creating local storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheStorageConfig(
                storage_type=CacheStorageType.LOCAL,
                root_path=tmpdir,
            )
            storage = create_cache_storage(config)
            assert isinstance(storage, LocalCacheStorage)

    def test_create_memory(self):
        """Test creating memory storage."""
        config = CacheStorageConfig(
            storage_type=CacheStorageType.MEMORY,
            root_path="test",
        )
        storage = create_cache_storage(config)
        assert isinstance(storage, MemoryCacheStorage)


# ==============================================================================
# Spatiotemporal Index Tests
# ==============================================================================


class TestBoundingBox:
    """Tests for BoundingBox."""

    def test_valid_bbox(self):
        """Test valid bounding box."""
        bbox = BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0)
        assert bbox.west == -80.0
        assert bbox.south == 25.0
        assert bbox.east == -79.0
        assert bbox.north == 26.0

    def test_invalid_bbox_south_north(self):
        """Test invalid south > north."""
        with pytest.raises(ValueError, match="south.*must be <= north"):
            BoundingBox(west=-80.0, south=26.0, east=-79.0, north=25.0)

    def test_bbox_area(self):
        """Test area calculation."""
        bbox = BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0)
        assert bbox.area == 1.0

    def test_bbox_width_height(self):
        """Test width and height calculation."""
        bbox = BoundingBox(west=-80.0, south=25.0, east=-78.0, north=27.0)
        assert bbox.width == 2.0
        assert bbox.height == 2.0

    def test_bbox_intersects(self):
        """Test intersection check."""
        bbox1 = BoundingBox(west=0, south=0, east=10, north=10)
        bbox2 = BoundingBox(west=5, south=5, east=15, north=15)
        bbox3 = BoundingBox(west=20, south=20, east=30, north=30)

        assert bbox1.intersects(bbox2)
        assert bbox2.intersects(bbox1)
        assert not bbox1.intersects(bbox3)

    def test_bbox_contains(self):
        """Test containment check."""
        outer = BoundingBox(west=0, south=0, east=10, north=10)
        inner = BoundingBox(west=2, south=2, east=8, north=8)
        partial = BoundingBox(west=5, south=5, east=15, north=15)

        assert outer.contains(inner)
        assert not inner.contains(outer)
        assert not outer.contains(partial)


class TestTimeRange:
    """Tests for TimeRange."""

    def test_valid_time_range(self):
        """Test valid time range."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        tr = TimeRange(start=start, end=end)

        assert tr.start == start
        assert tr.end == end

    def test_invalid_time_range(self):
        """Test invalid start > end."""
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="start.*must be <= end"):
            TimeRange(start=start, end=end)

    def test_time_range_duration(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        tr = TimeRange(start=start, end=end)

        assert tr.duration == timedelta(days=1)

    def test_time_range_overlaps(self):
        """Test overlap check."""
        tr1 = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        tr2 = TimeRange(
            start=datetime(2024, 1, 5, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        tr3 = TimeRange(
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 10, tzinfo=timezone.utc),
        )

        assert tr1.overlaps(tr2)
        assert tr2.overlaps(tr1)
        assert not tr1.overlaps(tr3)


class TestSpatiotemporalIndex:
    """Tests for SpatiotemporalIndex."""

    @pytest.fixture
    def index(self):
        """Create in-memory index."""
        # db_path=None creates in-memory database
        return SpatiotemporalIndex(db_path=None)

    def test_add_and_query(self, index):
        """Test adding and querying entries."""
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="sentinel-2",
            data_type="optical",
        )

        index.add(entry)

        # Query should find it
        results = index.query(
            bbox=BoundingBox(west=-81, south=24, east=-78, north=27),
        )
        assert len(results) >= 1
        # Results are QueryResult, get entry from them
        found_keys = {r.entry.cache_key for r in results}
        assert "test_key" in found_keys

    def test_spatial_query(self, index):
        """Test spatial querying."""
        # Add entries in different locations
        for i, lon in enumerate([-80, -70, -60]):
            entry = IndexEntry(
                cache_key=f"key_{i}",
                bbox=BoundingBox(west=lon, south=25.0, east=lon + 1, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                ),
                provider="test",
                data_type="raster",
            )
            index.add(entry)

        # Query overlapping first entry
        query_bbox = BoundingBox(west=-81, south=24, east=-79, north=27)
        results = index.query(bbox=query_bbox)

        assert len(results) == 1
        assert results[0].entry.cache_key == "key_0"

    def test_temporal_query(self, index):
        """Test temporal querying."""
        # Add entries at different times
        for i in range(3):
            entry = IndexEntry(
                cache_key=f"key_{i}",
                bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, i + 2, tzinfo=timezone.utc),
                ),
                provider="test",
                data_type="raster",
            )
            index.add(entry)

        # Query first two days
        query_range = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
        )
        results = index.query(time_range=query_range)

        assert len(results) == 2

    def test_spatiotemporal_query(self, index):
        """Test combined spatiotemporal querying."""
        # Add entries
        entry1 = IndexEntry(
            cache_key="miami_jan",
            bbox=BoundingBox(west=-80.5, south=25.5, east=-80.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            ),
            provider="sentinel-2",
            data_type="optical",
        )
        entry2 = IndexEntry(
            cache_key="miami_feb",
            bbox=BoundingBox(west=-80.5, south=25.5, east=-80.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 5, tzinfo=timezone.utc),
            ),
            provider="sentinel-2",
            data_type="optical",
        )
        entry3 = IndexEntry(
            cache_key="tampa_jan",
            bbox=BoundingBox(west=-82.5, south=27.5, east=-82.0, north=28.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            ),
            provider="sentinel-2",
            data_type="optical",
        )

        index.add(entry1)
        index.add(entry2)
        index.add(entry3)

        # Query Miami in January
        results = index.query(
            bbox=BoundingBox(west=-81, south=25, east=-79, north=27),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            ),
        )

        assert len(results) == 1
        assert results[0].entry.cache_key == "miami_jan"

    def test_remove(self, index):
        """Test removing entries."""
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
        )

        index.add(entry)

        # Query should find it
        results = index.query(bbox=entry.bbox)
        assert len(results) >= 1

        result = index.remove("test_key")
        assert result is True

        # Query should not find it
        results = index.query(bbox=entry.bbox)
        found_keys = {r.entry.cache_key for r in results}
        assert "test_key" not in found_keys

    def test_persistence(self):
        """Test index persistence to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.db"

            # Create index and add entry
            index1 = SpatiotemporalIndex(db_path=db_path)
            entry = IndexEntry(
                cache_key="persistent_key",
                bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                ),
                provider="test",
                data_type="raster",
            )
            index1.add(entry)
            # Close by letting the object go out of scope or explicit close if available

            # Reopen and verify
            index2 = SpatiotemporalIndex(db_path=db_path)
            results = index2.query(bbox=entry.bbox)
            found_keys = {r.entry.cache_key for r in results}
            assert "persistent_key" in found_keys


# ==============================================================================
# Cache Manager Tests
# ==============================================================================


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = CacheConfig(
            max_size_bytes=1024 * 1024 * 1024,
            max_entries=10000,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
        )
        assert config.max_size_bytes == 1024 * 1024 * 1024
        assert config.eviction_policy == EvictionPolicy.LRU

    def test_invalid_max_size(self):
        """Test invalid max_size_bytes."""
        with pytest.raises(ValueError, match="max_size_bytes must be >= 0"):
            CacheConfig(max_size_bytes=-1)

    def test_invalid_cleanup_interval(self):
        """Test invalid cleanup_interval_seconds."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be >= 60"):
            CacheConfig(cleanup_interval_seconds=30)


class TestCacheManager:
    """Tests for CacheManager."""

    @pytest.fixture
    def manager(self):
        """Create cache manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                max_size_bytes=10 * 1024 * 1024,  # 10 MB
                default_ttl_seconds=3600,
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)
            yield manager

    def test_put_and_get(self, manager):
        """Test storing and retrieving metadata entries."""
        # CacheManager.put() stores metadata about cached items
        # It requires cache_key, storage_key, data_type, size_bytes, checksum
        checksum = hashlib.sha256(b"test data").hexdigest()

        entry = manager.put(
            cache_key="test_key",
            storage_key="storage/test_key.tif",
            data_type="raster",
            size_bytes=1024,
            checksum=checksum,
        )

        assert entry.cache_key == "test_key"
        assert entry.status == CacheEntryStatus.ACTIVE

        # get() returns the CacheEntry, not data+entry
        retrieved = manager.get("test_key")
        assert retrieved is not None
        assert retrieved.cache_key == "test_key"
        assert retrieved.storage_key == "storage/test_key.tif"

    def test_get_nonexistent(self, manager):
        """Test getting nonexistent key."""
        result = manager.get("nonexistent")
        assert result is None

    def test_contains(self, manager):
        """Test contains check."""
        assert not manager.contains("test_key")

        checksum = hashlib.sha256(b"data").hexdigest()
        manager.put("test_key", "storage/key", "raster", 100, checksum)

        assert manager.contains("test_key")

    def test_generate_cache_key(self, manager):
        """Test cache key generation."""
        key1 = manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240101",
            bbox=[-80.0, 25.0, -79.0, 26.0],
        )

        key2 = manager.generate_cache_key(
            provider="sentinel-2",
            dataset_id="S2A_MSIL2A_20240101",
            bbox=[-80.0, 25.0, -79.0, 26.0],
        )

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different key
        key3 = manager.generate_cache_key(
            provider="landsat",
            dataset_id="S2A_MSIL2A_20240101",
            bbox=[-80.0, 25.0, -79.0, 26.0],
        )
        assert key1 != key3

    def test_metadata_and_tags(self, manager):
        """Test metadata and tags storage."""
        checksum = hashlib.sha256(b"data").hexdigest()

        manager.put(
            cache_key="key1",
            storage_key="storage/key1",
            data_type="raster",
            size_bytes=100,
            checksum=checksum,
            metadata={"source": "sentinel-2", "band": "B4"},
            tags={"optical", "red"},
        )

        entry = manager.get("key1")
        assert entry is not None
        assert entry.metadata.get("source") == "sentinel-2"
        assert "optical" in entry.tags


class TestCacheManagerConcurrency:
    """Tests for concurrent cache manager access."""

    def test_concurrent_puts(self):
        """Test concurrent put operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            errors = []

            def put_entries(thread_id):
                try:
                    for i in range(10):
                        checksum = hashlib.sha256(f"data_{thread_id}_{i}".encode()).hexdigest()
                        manager.put(
                            cache_key=f"key_{thread_id}_{i}",
                            storage_key=f"storage/{thread_id}/{i}",
                            data_type="raster",
                            size_bytes=100,
                            checksum=checksum,
                        )
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=put_entries, args=(i,)) for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCacheIntegration:
    """Integration tests for cache system."""

    def test_storage_and_manager_workflow(self):
        """Test using storage backend with cache manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create storage backend
            storage_config = CacheStorageConfig(
                storage_type=CacheStorageType.LOCAL,
                root_path=Path(tmpdir) / "storage",
            )
            storage = create_cache_storage(storage_config)

            # Create manager
            cache_config = CacheConfig(
                default_ttl_seconds=3600,
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config, storage_backend=storage)

            # Store data in storage
            data = b"raster data content"
            storage_obj = storage.put("tile_123", data, content_type="image/tiff")

            # Register in manager
            entry = manager.put(
                cache_key="sentinel2_tile_123",
                storage_key="tile_123",
                data_type="optical",
                size_bytes=storage_obj.size_bytes,
                checksum=storage_obj.checksum,
                metadata={"source": "sentinel-2", "tile_id": "T31TFM"},
            )

            # Lookup in manager
            cached_entry = manager.get("sentinel2_tile_123")
            assert cached_entry is not None
            assert cached_entry.storage_key == "tile_123"

            # Retrieve from storage
            result = storage.get("tile_123")
            assert result is not None
            retrieved_data, _ = result
            assert retrieved_data == data

    def test_index_with_storage_workflow(self):
        """Test spatiotemporal index with storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index
            index = SpatiotemporalIndex(db_path=Path(tmpdir) / "index.db")

            # Create storage
            storage_config = CacheStorageConfig(
                storage_type=CacheStorageType.LOCAL,
                root_path=Path(tmpdir) / "storage",
            )
            storage = create_cache_storage(storage_config)

            # Store data
            data = b"flood extent raster data"
            cache_key = "miami_flood_2024"
            storage.put(cache_key, data)

            # Add to spatial index
            index_entry = IndexEntry(
                cache_key=cache_key,
                bbox=BoundingBox(west=-80.5, south=25.5, east=-80.0, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 9, 15, tzinfo=timezone.utc),
                    end=datetime(2024, 9, 20, tzinfo=timezone.utc),
                ),
                provider="analysis",
                data_type="flood_extent",
            )
            index.add(index_entry)

            # Query by location
            results = index.query(
                bbox=BoundingBox(west=-81, south=25, east=-79, north=27)
            )
            assert len(results) == 1

            # Retrieve from storage using index result
            found_key = results[0].entry.cache_key
            result = storage.get(found_key)
            assert result is not None
            cached_data, _ = result
            assert cached_data == data


# ==============================================================================
# Edge Case Tests - Track 7 Review
# ==============================================================================


class TestBoundingBoxEdgeCases:
    """Edge case tests for BoundingBox."""

    def test_point_bbox(self):
        """Test bounding box for a single point."""
        bbox = BoundingBox(west=-80.0, south=25.0, east=-80.0, north=25.0)
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0
        assert bbox.center == (-80.0, 25.0)

    def test_antimeridian_crossing(self):
        """Test bounding box crossing antimeridian."""
        bbox = BoundingBox(west=170.0, south=0.0, east=-170.0, north=10.0)
        assert bbox.width == 20.0  # 170 to 180 + 180 to -170
        assert bbox.height == 10.0

    def test_antimeridian_intersection(self):
        """Test intersection with antimeridian-crossing bbox."""
        bbox1 = BoundingBox(west=170.0, south=0.0, east=-170.0, north=10.0)
        bbox2 = BoundingBox(west=175.0, south=5.0, east=-175.0, north=15.0)
        assert bbox1.intersects(bbox2)

    def test_extreme_coordinates(self):
        """Test extreme coordinate values."""
        bbox = BoundingBox(west=-180.0, south=-90.0, east=180.0, north=90.0)
        assert bbox.width == 360.0
        assert bbox.height == 180.0
        assert bbox.area == 64800.0

    def test_expand_clamping(self):
        """Test expand doesn't exceed coordinate bounds."""
        bbox = BoundingBox(west=-175.0, south=-85.0, east=175.0, north=85.0)
        expanded = bbox.expand(10.0)
        assert expanded.west == -180.0
        assert expanded.south == -90.0
        assert expanded.east == 180.0
        assert expanded.north == 90.0


class TestTimeRangeEdgeCases:
    """Edge case tests for TimeRange."""

    def test_same_start_end(self):
        """Test time range with same start and end (instant)."""
        instant = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        tr = TimeRange(start=instant, end=instant)
        assert tr.duration == timedelta(0)

    def test_naive_datetime_conversion(self):
        """Test naive datetimes are converted to UTC."""
        naive_start = datetime(2024, 1, 1, 0, 0, 0)
        naive_end = datetime(2024, 1, 2, 0, 0, 0)
        tr = TimeRange(start=naive_start, end=naive_end)
        assert tr.start.tzinfo == timezone.utc
        assert tr.end.tzinfo == timezone.utc

    def test_instant_overlap(self):
        """Test instant time range overlap."""
        instant = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        tr1 = TimeRange(start=instant, end=instant)
        tr2 = TimeRange(
            start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        )
        assert tr1.overlaps(tr2)

    def test_expand_time_range(self):
        """Test expanding time range."""
        tr = TimeRange(
            start=datetime(2024, 1, 10, tzinfo=timezone.utc),
            end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        )
        expanded = tr.expand(timedelta(days=5))
        assert expanded.start == datetime(2024, 1, 5, tzinfo=timezone.utc)
        assert expanded.end == datetime(2024, 1, 25, tzinfo=timezone.utc)


class TestCacheManagerEviction:
    """Tests for cache manager eviction policies."""

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                max_entries=3,
                eviction_policy=EvictionPolicy.LRU,
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            # Add 3 entries
            for i in range(3):
                checksum = hashlib.sha256(f"data_{i}".encode()).hexdigest()
                manager.put(f"key_{i}", f"storage/{i}", "raster", 100, checksum)

            # Access key_0 to make it recently used
            manager.get("key_0")
            time.sleep(0.01)  # Ensure timestamp difference

            # Add another entry - should evict least recently used (key_1)
            checksum = hashlib.sha256(b"data_3").hexdigest()
            manager.put("key_3", "storage/3", "raster", 100, checksum)

            # key_1 should be evicted (oldest accessed), key_0 should remain
            assert manager.contains("key_0")
            assert not manager.contains("key_1")
            assert manager.contains("key_2")
            assert manager.contains("key_3")

    def test_max_size_eviction(self):
        """Test max size enforcement with eviction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                max_size_bytes=300,  # Allow ~3 x 100 byte entries
                eviction_policy=EvictionPolicy.LRU,
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            # Add entries until we exceed limit
            for i in range(5):
                checksum = hashlib.sha256(f"data_{i}".encode()).hexdigest()
                manager.put(f"key_{i}", f"storage/{i}", "raster", 100, checksum)

            # Check that we have at most 3 active entries
            stats = manager.get_statistics()
            assert stats.active_entries <= 3


class TestCacheManagerTTL:
    """Tests for cache TTL expiration."""

    def test_entry_expiration(self):
        """Test entry expires after TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                default_ttl_seconds=1,  # 1 second TTL
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            checksum = hashlib.sha256(b"data").hexdigest()
            manager.put("key1", "storage/1", "raster", 100, checksum)

            # Immediately should be accessible
            assert manager.get("key1") is not None

            # Wait for expiration
            time.sleep(1.5)

            # Should now return None (expired)
            assert manager.get("key1") is None

    def test_custom_ttl_override(self):
        """Test custom TTL per entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                default_ttl_seconds=60,  # 60 second default
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            checksum = hashlib.sha256(b"data").hexdigest()
            # Entry with 1 second TTL
            manager.put("key1", "storage/1", "raster", 100, checksum, ttl_seconds=1)

            assert manager.get("key1") is not None
            time.sleep(1.5)
            assert manager.get("key1") is None


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    def test_hit_miss_tracking(self):
        """Test hit and miss tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            checksum = hashlib.sha256(b"data").hexdigest()
            manager.put("key1", "storage/1", "raster", 100, checksum)

            # Record some hits and misses
            manager.get("key1")  # hit
            manager.get("key1")  # hit
            manager.get("nonexistent")  # miss
            manager.get("missing")  # miss

            stats = manager.get_statistics()
            assert stats.hits == 2
            assert stats.misses == 2
            assert stats.hit_rate == 0.5


class TestSpatiotemporalIndexEdgeCases:
    """Edge case tests for SpatiotemporalIndex."""

    def test_overlapping_entries(self):
        """Test handling of overlapping entries."""
        index = SpatiotemporalIndex(db_path=None)

        # Add overlapping entries
        for i in range(3):
            entry = IndexEntry(
                cache_key=f"overlap_{i}",
                bbox=BoundingBox(west=-80.5 + i * 0.1, south=25.0, east=-79.5 + i * 0.1, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 10, tzinfo=timezone.utc),
                ),
                provider="test",
                data_type="raster",
            )
            index.add(entry)

        # Query should return all overlapping
        results = index.query(
            bbox=BoundingBox(west=-80.5, south=24.5, east=-79.5, north=26.5)
        )
        assert len(results) >= 1

    def test_query_with_all_filters(self):
        """Test query with all filter types combined."""
        index = SpatiotemporalIndex(db_path=None)

        # Add entries
        entry = IndexEntry(
            cache_key="filtered_entry",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            ),
            provider="sentinel-2",
            data_type="optical",
            resolution_m=10.0,
        )
        index.add(entry)

        # Query with all filters
        results = index.query(
            bbox=BoundingBox(west=-81.0, south=24.0, east=-78.0, north=27.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            ),
            data_type="optical",
            provider="sentinel-2",
            min_resolution_m=5.0,
            max_resolution_m=15.0,
        )
        assert len(results) == 1
        assert results[0].entry.cache_key == "filtered_entry"

    def test_temporal_predicate_during(self):
        """Test DURING temporal predicate."""
        index = SpatiotemporalIndex(db_path=None)

        # Entry fully within query range
        entry = IndexEntry(
            cache_key="during_entry",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 5, tzinfo=timezone.utc),
                end=datetime(2024, 1, 8, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
        )
        index.add(entry)

        results = index.query(
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            ),
            temporal_predicate=TemporalPredicate.DURING,
        )
        assert len(results) == 1

    def test_coverage_statistics(self):
        """Test coverage statistics calculation."""
        index = SpatiotemporalIndex(db_path=None)

        # Add entries from different providers
        for i, provider in enumerate(["sentinel-2", "landsat", "modis"]):
            entry = IndexEntry(
                cache_key=f"provider_{i}",
                bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
                time_range=TimeRange(
                    start=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2 + i, tzinfo=timezone.utc),
                ),
                provider=provider,
                data_type="optical",
            )
            index.add(entry)

        coverage = index.get_coverage(
            bbox=BoundingBox(west=-81.0, south=24.0, east=-78.0, north=27.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            ),
        )
        assert coverage["entry_count"] == 3
        assert len(coverage["providers"]) == 3
        assert "sentinel-2" in coverage["providers"]

    def test_empty_query_results(self):
        """Test query returning no results."""
        index = SpatiotemporalIndex(db_path=None)

        # Add entry
        entry = IndexEntry(
            cache_key="test_entry",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
        )
        index.add(entry)

        # Query non-overlapping area
        results = index.query(
            bbox=BoundingBox(west=0.0, south=0.0, east=1.0, north=1.0)
        )
        assert len(results) == 0

    def test_index_update_existing(self):
        """Test updating an existing index entry."""
        index = SpatiotemporalIndex(db_path=None)

        # Add initial entry
        entry1 = IndexEntry(
            cache_key="update_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
        )
        index.add(entry1)

        # Update with new bbox
        entry2 = IndexEntry(
            cache_key="update_key",  # Same key
            bbox=BoundingBox(west=-70.0, south=30.0, east=-69.0, north=31.0),  # Different location
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
        )
        index.add(entry2)

        # Should only find at new location
        old_results = index.query(
            bbox=BoundingBox(west=-81.0, south=24.0, east=-78.0, north=27.0)
        )
        new_results = index.query(
            bbox=BoundingBox(west=-71.0, south=29.0, east=-68.0, north=32.0)
        )

        assert len(old_results) == 0
        assert len(new_results) == 1
        assert new_results[0].entry.cache_key == "update_key"

    def test_index_rebuild(self):
        """Test rebuilding the spatial index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "index.db"
            index = SpatiotemporalIndex(db_path=db_path)

            # Add entries
            for i in range(5):
                entry = IndexEntry(
                    cache_key=f"key_{i}",
                    bbox=BoundingBox(west=-80.0 + i, south=25.0, east=-79.0 + i, north=26.0),
                    time_range=TimeRange(
                        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                    ),
                    provider="test",
                    data_type="raster",
                )
                index.add(entry)

            # Rebuild and verify
            index.rebuild()
            assert index.count() == 5


class TestIndexEntryValidation:
    """Tests for IndexEntry validation."""

    def test_nan_resolution_normalized(self):
        """Test that NaN resolution_m is normalized to 0.0."""
        import math
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
            resolution_m=float("nan"),
        )
        assert entry.resolution_m == 0.0
        assert not math.isnan(entry.resolution_m)

    def test_inf_resolution_normalized(self):
        """Test that Inf resolution_m is normalized to 0.0."""
        import math
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
            resolution_m=float("inf"),
        )
        assert entry.resolution_m == 0.0
        assert not math.isinf(entry.resolution_m)

    def test_negative_inf_resolution_normalized(self):
        """Test that -Inf resolution_m is normalized to 0.0."""
        import math
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
            resolution_m=float("-inf"),
        )
        assert entry.resolution_m == 0.0
        assert not math.isinf(entry.resolution_m)

    def test_negative_resolution_normalized(self):
        """Test that negative resolution_m is normalized to 0.0."""
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
            resolution_m=-10.0,
        )
        assert entry.resolution_m == 0.0

    def test_valid_resolution_preserved(self):
        """Test that valid resolution values are preserved."""
        entry = IndexEntry(
            cache_key="test_key",
            bbox=BoundingBox(west=-80.0, south=25.0, east=-79.0, north=26.0),
            time_range=TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            provider="test",
            data_type="raster",
            resolution_m=10.0,
        )
        assert entry.resolution_m == 10.0


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_by_pattern(self):
        """Test invalidating entries by pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            # Add entries with common prefix
            for i in range(5):
                checksum = hashlib.sha256(f"data_{i}".encode()).hexdigest()
                manager.put(f"prefix_key_{i}", f"storage/{i}", "raster", 100, checksum)

            # Add entries with different prefix
            for i in range(3):
                checksum = hashlib.sha256(f"other_{i}".encode()).hexdigest()
                manager.put(f"other_key_{i}", f"storage/other/{i}", "raster", 100, checksum)

            # Invalidate prefix entries
            count = manager.invalidate_by_pattern("prefix%")
            assert count == 5

            # Verify prefix entries are invalid
            for i in range(5):
                assert not manager.contains(f"prefix_key_{i}")

            # Verify other entries still valid
            for i in range(3):
                assert manager.contains(f"other_key_{i}")

    def test_invalidate_by_tag(self):
        """Test invalidating entries by tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = CacheConfig(
                db_path=Path(tmpdir) / "cache.db",
            )
            manager = CacheManager(config=cache_config)

            # Add entries with specific tags
            checksum = hashlib.sha256(b"data1").hexdigest()
            manager.put("key1", "storage/1", "raster", 100, checksum, tags={"optical", "sentinel"})

            checksum = hashlib.sha256(b"data2").hexdigest()
            manager.put("key2", "storage/2", "raster", 100, checksum, tags={"optical", "landsat"})

            checksum = hashlib.sha256(b"data3").hexdigest()
            manager.put("key3", "storage/3", "raster", 100, checksum, tags={"sar"})

            # Invalidate sentinel entries
            count = manager.invalidate_by_tag("sentinel")
            assert count == 1

            assert not manager.contains("key1")
            assert manager.contains("key2")
            assert manager.contains("key3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
