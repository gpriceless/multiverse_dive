"""
Spatiotemporal Index for Cache Lookups.

Provides efficient spatial and temporal indexing for cached data products:
- R-tree spatial indexing for bounding box queries
- Temporal indexing for time range queries
- Combined spatiotemporal queries
- Support for various geospatial predicates (intersects, contains, within)
- Index persistence and recovery

The index enables fast cache lookups when searching for data
covering a specific area and time window.
"""

import json
import logging
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class SpatialPredicate(Enum):
    """Spatial query predicates."""

    INTERSECTS = "intersects"  # Any overlap
    CONTAINS = "contains"  # Query fully contains indexed item
    WITHIN = "within"  # Indexed item fully within query
    OVERLAPS = "overlaps"  # Partial overlap (not fully contains)


class TemporalPredicate(Enum):
    """Temporal query predicates."""

    OVERLAPS = "overlaps"  # Any temporal overlap
    DURING = "during"  # Indexed item during query period
    CONTAINS = "contains"  # Query contains indexed item's time
    BEFORE = "before"  # Indexed item before query start
    AFTER = "after"  # Indexed item after query end


@dataclass
class BoundingBox:
    """
    Geographic bounding box.

    Attributes:
        west: Western longitude (-180 to 180)
        south: Southern latitude (-90 to 90)
        east: Eastern longitude (-180 to 180)
        north: Northern latitude (-90 to 90)
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        """Validate coordinates."""
        if self.west > self.east:
            # Handle antimeridian crossing
            pass
        if self.south > self.north:
            raise ValueError(f"south ({self.south}) must be <= north ({self.north})")

    @property
    def width(self) -> float:
        """Get width in degrees."""
        if self.west <= self.east:
            return self.east - self.west
        else:
            # Antimeridian crossing
            return (180 - self.west) + (self.east + 180)

    @property
    def height(self) -> float:
        """Get height in degrees."""
        return self.north - self.south

    @property
    def area(self) -> float:
        """Get approximate area in square degrees."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (lon, lat)."""
        if self.west <= self.east:
            center_lon = (self.west + self.east) / 2
        else:
            # Antimeridian crossing
            center_lon = ((self.west + self.east) / 2 + 180) % 360 - 180
        center_lat = (self.south + self.north) / 2
        return (center_lon, center_lat)

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bbox intersects with another."""
        # Handle normal case (no antimeridian crossing)
        if self.west <= self.east and other.west <= other.east:
            return not (
                self.east < other.west
                or self.west > other.east
                or self.north < other.south
                or self.south > other.north
            )

        # Handle antimeridian crossing
        return self._intersects_antimeridian(other)

    def _intersects_antimeridian(self, other: "BoundingBox") -> bool:
        """Handle intersection with antimeridian crossing."""
        # Check latitude overlap first
        if self.north < other.south or self.south > other.north:
            return False

        # Handle longitude with potential antimeridian crossing
        if self.west <= self.east:
            # Self doesn't cross, check if other does
            if other.west <= other.east:
                return self.east >= other.west and self.west <= other.east
            else:
                # Other crosses antimeridian
                return self.east >= other.west or self.west <= other.east
        else:
            # Self crosses antimeridian
            if other.west <= other.east:
                return other.east >= self.west or other.west <= self.east
            else:
                # Both cross antimeridian - always overlap in longitude
                return True

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this bbox fully contains another."""
        # Latitude check
        if other.south < self.south or other.north > self.north:
            return False

        # Simple case: neither crosses antimeridian
        if self.west <= self.east and other.west <= other.east:
            return other.west >= self.west and other.east <= self.east

        # Self crosses antimeridian but other doesn't
        if self.west > self.east and other.west <= other.east:
            return (other.west >= self.west) or (other.east <= self.east)

        # Other cases are complex - skip for now
        return False

    def expand(self, degrees: float) -> "BoundingBox":
        """
        Expand bbox by given degrees in all directions.

        Args:
            degrees: Amount to expand

        Returns:
            Expanded BoundingBox
        """
        return BoundingBox(
            west=max(-180, self.west - degrees),
            south=max(-90, self.south - degrees),
            east=min(180, self.east + degrees),
            north=min(90, self.north + degrees),
        )

    def to_list(self) -> List[float]:
        """Convert to [west, south, east, north] list."""
        return [self.west, self.south, self.east, self.north]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        """Create from [west, south, east, north] list."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(west=coords[0], south=coords[1], east=coords[2], north=coords[3])


@dataclass
class TimeRange:
    """
    Temporal range.

    Attributes:
        start: Start timestamp
        end: End timestamp
    """

    start: datetime
    end: datetime

    def __post_init__(self):
        """Validate and normalize timestamps."""
        # Ensure timezone-aware
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=timezone.utc)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=timezone.utc)

        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    @property
    def duration(self) -> timedelta:
        """Get duration of time range."""
        return self.end - self.start

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and self.end >= other.start

    def contains(self, other: "TimeRange") -> bool:
        """Check if this range fully contains another."""
        return self.start <= other.start and self.end >= other.end

    def during(self, other: "TimeRange") -> bool:
        """Check if this range is during another (fully contained)."""
        return other.start <= self.start and other.end >= self.end

    def expand(self, delta: timedelta) -> "TimeRange":
        """
        Expand range by given timedelta in both directions.

        Args:
            delta: Amount to expand

        Returns:
            Expanded TimeRange
        """
        return TimeRange(start=self.start - delta, end=self.end + delta)


@dataclass
class IndexEntry:
    """
    An entry in the spatiotemporal index.

    Attributes:
        cache_key: Reference to cache entry
        bbox: Geographic bounding box
        time_range: Temporal range
        data_type: Type of data (optical, sar, etc.)
        provider: Data provider
        resolution_m: Spatial resolution in meters
        metadata: Additional metadata
    """

    cache_key: str
    bbox: BoundingBox
    time_range: TimeRange
    data_type: str
    provider: str
    resolution_m: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize resolution_m field."""
        import math
        if math.isnan(self.resolution_m) or math.isinf(self.resolution_m):
            self.resolution_m = 0.0
        if self.resolution_m < 0:
            self.resolution_m = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_key": self.cache_key,
            "bbox": self.bbox.to_list(),
            "time_start": self.time_range.start.isoformat(),
            "time_end": self.time_range.end.isoformat(),
            "data_type": self.data_type,
            "provider": self.provider,
            "resolution_m": self.resolution_m,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexEntry":
        """Create from dictionary."""
        return cls(
            cache_key=data["cache_key"],
            bbox=BoundingBox.from_list(data["bbox"]),
            time_range=TimeRange(
                start=datetime.fromisoformat(data["time_start"]),
                end=datetime.fromisoformat(data["time_end"]),
            ),
            data_type=data["data_type"],
            provider=data["provider"],
            resolution_m=data.get("resolution_m", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryResult:
    """
    Result from a spatiotemporal query.

    Attributes:
        entry: The matched index entry
        spatial_overlap: Fraction of query bbox covered (0-1)
        temporal_overlap: Fraction of query time covered (0-1)
        score: Combined relevance score (0-1)
    """

    entry: IndexEntry
    spatial_overlap: float = 1.0
    temporal_overlap: float = 1.0
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry": self.entry.to_dict(),
            "spatial_overlap": self.spatial_overlap,
            "temporal_overlap": self.temporal_overlap,
            "score": self.score,
        }


class SpatiotemporalIndex:
    """
    Spatiotemporal index for efficient cache lookups.

    Uses SQLite R-tree extension for spatial indexing and
    B-tree for temporal indexing. Provides combined queries
    for finding cached data covering a specific area and time.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize spatiotemporal index.

        Args:
            db_path: Path to SQLite database (uses memory if None)
        """
        self._db_path = db_path
        self._lock = threading.RLock()
        # For in-memory databases, we need to keep the connection alive
        # because each new ":memory:" connection creates a separate db
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path is None:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_database()

        logger.info(f"SpatiotemporalIndex initialized at {db_path or ':memory:'}")

    def _init_database(self) -> None:
        """Initialize SQLite database with R-tree."""
        with self._get_connection() as conn:
            # Main index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_entries (
                    cache_key TEXT PRIMARY KEY,
                    west REAL NOT NULL,
                    south REAL NOT NULL,
                    east REAL NOT NULL,
                    north REAL NOT NULL,
                    time_start TEXT NOT NULL,
                    time_end TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    resolution_m REAL DEFAULT 0,
                    metadata TEXT
                )
            """)

            # R-tree for spatial indexing
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS spatial_index
                USING rtree(
                    id,
                    west, east,
                    south, north
                )
            """)

            # Mapping from cache_key to R-tree id
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rtree_mapping (
                    cache_key TEXT PRIMARY KEY,
                    rtree_id INTEGER NOT NULL
                )
            """)

            # Temporal index
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_time_start
                ON index_entries(time_start)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_time_end
                ON index_entries(time_end)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_type
                ON index_entries(data_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider
                ON index_entries(provider)
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._persistent_conn is not None:
            # Use persistent connection for in-memory databases
            return self._persistent_conn
        # File-based database - create new connection
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_next_rtree_id(self, conn: sqlite3.Connection) -> int:
        """Get next available R-tree ID."""
        result = conn.execute("SELECT MAX(rtree_id) FROM rtree_mapping").fetchone()[0]
        return (result or 0) + 1

    def add(self, entry: IndexEntry) -> None:
        """
        Add an entry to the index.

        Args:
            entry: Index entry to add
        """
        with self._lock:
            with self._get_connection() as conn:
                # Insert main entry
                conn.execute(
                    """
                    INSERT OR REPLACE INTO index_entries
                    (cache_key, west, south, east, north, time_start, time_end,
                     data_type, provider, resolution_m, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.cache_key,
                        entry.bbox.west,
                        entry.bbox.south,
                        entry.bbox.east,
                        entry.bbox.north,
                        entry.time_range.start.isoformat(),
                        entry.time_range.end.isoformat(),
                        entry.data_type,
                        entry.provider,
                        entry.resolution_m,
                        json.dumps(entry.metadata),
                    ),
                )

                # Check if already in R-tree
                existing = conn.execute(
                    "SELECT rtree_id FROM rtree_mapping WHERE cache_key = ?",
                    (entry.cache_key,),
                ).fetchone()

                if existing:
                    rtree_id = existing["rtree_id"]
                    # Update R-tree entry
                    conn.execute(
                        """
                        UPDATE spatial_index
                        SET west = ?, east = ?, south = ?, north = ?
                        WHERE id = ?
                        """,
                        (
                            entry.bbox.west,
                            entry.bbox.east,
                            entry.bbox.south,
                            entry.bbox.north,
                            rtree_id,
                        ),
                    )
                else:
                    # Insert new R-tree entry
                    rtree_id = self._get_next_rtree_id(conn)
                    conn.execute(
                        """
                        INSERT INTO spatial_index (id, west, east, south, north)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            rtree_id,
                            entry.bbox.west,
                            entry.bbox.east,
                            entry.bbox.south,
                            entry.bbox.north,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO rtree_mapping (cache_key, rtree_id) VALUES (?, ?)",
                        (entry.cache_key, rtree_id),
                    )

                conn.commit()

        logger.debug(f"Added index entry: {entry.cache_key}")

    def remove(self, cache_key: str) -> bool:
        """
        Remove an entry from the index.

        Args:
            cache_key: Cache key to remove

        Returns:
            True if entry was removed
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get R-tree ID
                row = conn.execute(
                    "SELECT rtree_id FROM rtree_mapping WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return False

                rtree_id = row["rtree_id"]

                # Remove from R-tree
                conn.execute("DELETE FROM spatial_index WHERE id = ?", (rtree_id,))

                # Remove mapping
                conn.execute(
                    "DELETE FROM rtree_mapping WHERE cache_key = ?", (cache_key,)
                )

                # Remove main entry
                conn.execute(
                    "DELETE FROM index_entries WHERE cache_key = ?", (cache_key,)
                )

                conn.commit()

        logger.debug(f"Removed index entry: {cache_key}")
        return True

    def query(
        self,
        bbox: Optional[BoundingBox] = None,
        time_range: Optional[TimeRange] = None,
        data_type: Optional[str] = None,
        provider: Optional[str] = None,
        spatial_predicate: SpatialPredicate = SpatialPredicate.INTERSECTS,
        temporal_predicate: TemporalPredicate = TemporalPredicate.OVERLAPS,
        min_resolution_m: Optional[float] = None,
        max_resolution_m: Optional[float] = None,
        limit: int = 100,
    ) -> List[QueryResult]:
        """
        Query the index with spatial and temporal constraints.

        Args:
            bbox: Bounding box to query
            time_range: Time range to query
            data_type: Filter by data type
            provider: Filter by provider
            spatial_predicate: Spatial relationship to match
            temporal_predicate: Temporal relationship to match
            min_resolution_m: Minimum resolution filter
            max_resolution_m: Maximum resolution filter
            limit: Maximum results to return

        Returns:
            List of QueryResult objects sorted by relevance
        """
        with self._lock:
            with self._get_connection() as conn:
                # Start with R-tree spatial query if bbox provided
                if bbox is not None:
                    # Get cache keys that spatially intersect
                    spatial_keys = set()
                    for row in conn.execute(
                        """
                        SELECT m.cache_key
                        FROM spatial_index s
                        JOIN rtree_mapping m ON s.id = m.rtree_id
                        WHERE s.west <= ? AND s.east >= ?
                        AND s.south <= ? AND s.north >= ?
                        """,
                        (bbox.east, bbox.west, bbox.north, bbox.south),
                    ):
                        spatial_keys.add(row["cache_key"])
                else:
                    spatial_keys = None

                # Build main query
                query = "SELECT * FROM index_entries WHERE 1=1"
                params: List[Any] = []

                if spatial_keys is not None:
                    if not spatial_keys:
                        return []  # No spatial matches
                    placeholders = ",".join("?" * len(spatial_keys))
                    query += f" AND cache_key IN ({placeholders})"
                    params.extend(spatial_keys)

                if time_range is not None:
                    if temporal_predicate == TemporalPredicate.OVERLAPS:
                        query += " AND time_start <= ? AND time_end >= ?"
                        params.extend([
                            time_range.end.isoformat(),
                            time_range.start.isoformat(),
                        ])
                    elif temporal_predicate == TemporalPredicate.DURING:
                        query += " AND time_start >= ? AND time_end <= ?"
                        params.extend([
                            time_range.start.isoformat(),
                            time_range.end.isoformat(),
                        ])
                    elif temporal_predicate == TemporalPredicate.CONTAINS:
                        query += " AND time_start <= ? AND time_end >= ?"
                        params.extend([
                            time_range.start.isoformat(),
                            time_range.end.isoformat(),
                        ])
                    elif temporal_predicate == TemporalPredicate.BEFORE:
                        query += " AND time_end < ?"
                        params.append(time_range.start.isoformat())
                    elif temporal_predicate == TemporalPredicate.AFTER:
                        query += " AND time_start > ?"
                        params.append(time_range.end.isoformat())

                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type)

                if provider:
                    query += " AND provider = ?"
                    params.append(provider)

                if min_resolution_m is not None:
                    query += " AND resolution_m >= ?"
                    params.append(min_resolution_m)

                if max_resolution_m is not None:
                    query += " AND resolution_m <= ?"
                    params.append(max_resolution_m)

                query += " LIMIT ?"
                params.append(limit * 2)  # Fetch extra for filtering

                rows = conn.execute(query, params).fetchall()

        # Post-process results
        results: List[QueryResult] = []

        for row in rows:
            entry = self._row_to_entry(row)

            # Apply more precise spatial predicate if needed
            if bbox is not None and spatial_predicate != SpatialPredicate.INTERSECTS:
                entry_bbox = entry.bbox
                if spatial_predicate == SpatialPredicate.CONTAINS:
                    if not bbox.contains(entry_bbox):
                        continue
                elif spatial_predicate == SpatialPredicate.WITHIN:
                    if not entry_bbox.contains(bbox):
                        continue
                elif spatial_predicate == SpatialPredicate.OVERLAPS:
                    if not entry_bbox.intersects(bbox):
                        continue

            # Calculate overlap scores
            spatial_overlap = 1.0
            if bbox is not None:
                spatial_overlap = self._calculate_spatial_overlap(entry.bbox, bbox)

            temporal_overlap = 1.0
            if time_range is not None:
                temporal_overlap = self._calculate_temporal_overlap(
                    entry.time_range, time_range
                )

            # Combined score (weighted geometric mean)
            score = (spatial_overlap * temporal_overlap) ** 0.5

            results.append(
                QueryResult(
                    entry=entry,
                    spatial_overlap=spatial_overlap,
                    temporal_overlap=temporal_overlap,
                    score=score,
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    def query_by_point(
        self,
        lon: float,
        lat: float,
        time: Optional[datetime] = None,
        data_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[QueryResult]:
        """
        Query index for entries containing a specific point.

        Args:
            lon: Longitude
            lat: Latitude
            time: Optional time point
            data_type: Filter by data type
            limit: Maximum results

        Returns:
            List of QueryResult objects
        """
        # Create a point bbox
        point_bbox = BoundingBox(west=lon, south=lat, east=lon, north=lat)

        time_range = None
        if time is not None:
            time_range = TimeRange(start=time, end=time)

        return self.query(
            bbox=point_bbox,
            time_range=time_range,
            data_type=data_type,
            spatial_predicate=SpatialPredicate.INTERSECTS,
            temporal_predicate=TemporalPredicate.CONTAINS
            if time is not None
            else TemporalPredicate.OVERLAPS,
            limit=limit,
        )

    def get_coverage(
        self,
        bbox: BoundingBox,
        time_range: TimeRange,
        data_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate coverage statistics for an area and time.

        Args:
            bbox: Query bounding box
            time_range: Query time range
            data_type: Filter by data type

        Returns:
            Dictionary with coverage statistics
        """
        results = self.query(
            bbox=bbox,
            time_range=time_range,
            data_type=data_type,
            limit=1000,
        )

        if not results:
            return {
                "entry_count": 0,
                "max_spatial_coverage": 0.0,
                "max_temporal_coverage": 0.0,
                "avg_score": 0.0,
                "providers": [],
            }

        providers = set()
        max_spatial = 0.0
        max_temporal = 0.0
        total_score = 0.0

        for result in results:
            providers.add(result.entry.provider)
            max_spatial = max(max_spatial, result.spatial_overlap)
            max_temporal = max(max_temporal, result.temporal_overlap)
            total_score += result.score

        return {
            "entry_count": len(results),
            "max_spatial_coverage": max_spatial,
            "max_temporal_coverage": max_temporal,
            "avg_score": total_score / len(results),
            "providers": list(providers),
        }

    def get_entry(self, cache_key: str) -> Optional[IndexEntry]:
        """
        Get a specific index entry.

        Args:
            cache_key: Cache key to retrieve

        Returns:
            IndexEntry if found, None otherwise
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM index_entries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return None

                return self._row_to_entry(row)

    def count(self, data_type: Optional[str] = None) -> int:
        """
        Count index entries.

        Args:
            data_type: Filter by data type

        Returns:
            Number of entries
        """
        with self._lock:
            with self._get_connection() as conn:
                if data_type:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM index_entries WHERE data_type = ?",
                        (data_type,),
                    ).fetchone()[0]
                else:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM index_entries"
                    ).fetchone()[0]

                return result

    def clear(self) -> int:
        """
        Clear all index entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            with self._get_connection() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM index_entries"
                ).fetchone()[0]

                conn.execute("DELETE FROM index_entries")
                conn.execute("DELETE FROM spatial_index")
                conn.execute("DELETE FROM rtree_mapping")
                conn.commit()

        logger.info(f"Cleared {count} index entries")
        return count

    def rebuild(self) -> None:
        """Rebuild spatial index from main table."""
        with self._lock:
            with self._get_connection() as conn:
                # Clear spatial index
                conn.execute("DELETE FROM spatial_index")
                conn.execute("DELETE FROM rtree_mapping")

                # Rebuild from main entries
                rows = conn.execute("SELECT * FROM index_entries").fetchall()

                for idx, row in enumerate(rows, 1):
                    conn.execute(
                        """
                        INSERT INTO spatial_index (id, west, east, south, north)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (idx, row["west"], row["east"], row["south"], row["north"]),
                    )
                    conn.execute(
                        "INSERT INTO rtree_mapping (cache_key, rtree_id) VALUES (?, ?)",
                        (row["cache_key"], idx),
                    )

                conn.commit()

        logger.info(f"Rebuilt spatial index with {len(rows)} entries")

    def _row_to_entry(self, row: sqlite3.Row) -> IndexEntry:
        """Convert database row to IndexEntry."""
        return IndexEntry(
            cache_key=row["cache_key"],
            bbox=BoundingBox(
                west=row["west"],
                south=row["south"],
                east=row["east"],
                north=row["north"],
            ),
            time_range=TimeRange(
                start=datetime.fromisoformat(row["time_start"]),
                end=datetime.fromisoformat(row["time_end"]),
            ),
            data_type=row["data_type"],
            provider=row["provider"],
            resolution_m=row["resolution_m"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _calculate_spatial_overlap(
        self, entry_bbox: BoundingBox, query_bbox: BoundingBox
    ) -> float:
        """
        Calculate spatial overlap ratio.

        Returns fraction of query bbox covered by entry bbox.
        """
        # Calculate intersection
        if not entry_bbox.intersects(query_bbox):
            return 0.0

        int_west = max(entry_bbox.west, query_bbox.west)
        int_south = max(entry_bbox.south, query_bbox.south)
        int_east = min(entry_bbox.east, query_bbox.east)
        int_north = min(entry_bbox.north, query_bbox.north)

        if int_west >= int_east or int_south >= int_north:
            return 0.0

        int_area = (int_east - int_west) * (int_north - int_south)
        query_area = query_bbox.area

        if query_area == 0:
            return 0.0

        return min(1.0, int_area / query_area)

    def _calculate_temporal_overlap(
        self, entry_range: TimeRange, query_range: TimeRange
    ) -> float:
        """
        Calculate temporal overlap ratio.

        Returns fraction of query range covered by entry range.
        """
        if not entry_range.overlaps(query_range):
            return 0.0

        int_start = max(entry_range.start, query_range.start)
        int_end = min(entry_range.end, query_range.end)

        if int_start >= int_end:
            return 0.0

        int_duration = (int_end - int_start).total_seconds()
        query_duration = query_range.duration.total_seconds()

        if query_duration == 0:
            return 0.0

        return min(1.0, int_duration / query_duration)


class IndexError(Exception):
    """Error in spatiotemporal index operations."""

    pass
