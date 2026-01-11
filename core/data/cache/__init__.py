"""
Cache System for Geospatial Data Products.

Provides comprehensive caching infrastructure for the ingestion pipeline,
enabling fast access to previously downloaded and processed data products.

Components:
- Storage backends (local, S3, memory) for cached data
- Spatiotemporal indexing for efficient lookups by area and time
- Lifecycle management with TTL, eviction policies, and cleanup

Example usage:
    from core.data.cache import (
        CacheManager,
        CacheConfig,
        CacheStorageConfig,
        CacheStorageType,
        SpatiotemporalIndex,
    )

    # Configure cache storage
    storage_config = CacheStorageConfig(
        storage_type=CacheStorageType.LOCAL,
        root_path="/tmp/geospatial_cache",
    )

    # Configure cache manager
    cache_config = CacheConfig(
        max_size_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
        default_ttl_seconds=86400,  # 24 hours
    )

    # Create cache manager
    manager = CacheManager(
        config=cache_config,
        storage_config=storage_config,
    )

    # Store data in cache
    entry = manager.put(
        cache_key="sentinel2_tile_123",
        data=image_bytes,
        data_type="raster",
    )

    # Retrieve from cache
    data, entry = manager.get("sentinel2_tile_123")
"""

from core.data.cache.storage import (
    CachedObject,
    CacheStorageBackend,
    CacheStorageConfig,
    CacheStorageTier,
    CacheStorageType,
    LocalCacheStorage,
    MemoryCacheStorage,
    S3CacheStorage,
    create_cache_storage,
)

from core.data.cache.index import (
    BoundingBox,
    IndexEntry,
    QueryResult,
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

__all__ = [
    # Storage
    "CachedObject",
    "CacheStorageBackend",
    "CacheStorageConfig",
    "CacheStorageTier",
    "CacheStorageType",
    "LocalCacheStorage",
    "MemoryCacheStorage",
    "S3CacheStorage",
    "create_cache_storage",
    # Index
    "BoundingBox",
    "IndexEntry",
    "QueryResult",
    "SpatialPredicate",
    "SpatiotemporalIndex",
    "TemporalPredicate",
    "TimeRange",
    # Manager
    "CacheConfig",
    "CacheEntry",
    "CacheEntryStatus",
    "CacheFullError",
    "CacheManager",
    "EvictionPolicy",
]
