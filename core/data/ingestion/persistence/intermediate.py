"""
Intermediate Product Management for Ingestion Pipeline.

Manages intermediate products during multi-stage processing pipelines:
- Product registration and tracking
- Lifecycle management (creation, access, expiration, cleanup)
- Dependency tracking between products
- Efficient storage with deduplication
- Automatic cleanup of expired products

Intermediate products are temporary artifacts created during processing
that may be needed by subsequent stages or for debugging/audit purposes.
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

from core.data.ingestion.persistence.storage import (
    ChecksumAlgorithm,
    StorageBackend,
    StorageConfig,
    StorageObject,
    StorageType,
    UploadResult,
    create_storage_backend,
)

logger = logging.getLogger(__name__)


class ProductType(Enum):
    """Types of intermediate products."""

    RAW = "raw"  # Original downloaded data
    NORMALIZED = "normalized"  # Reprojected/resampled data
    ENRICHED = "enriched"  # Data with added metadata/overviews
    VALIDATED = "validated"  # Validated data
    ANALYSIS = "analysis"  # Analysis outputs
    MASK = "mask"  # Derived masks (cloud, water, etc.)
    INDEX = "index"  # Derived indices (NDWI, NBR, etc.)
    MOSAIC = "mosaic"  # Mosaicked tiles
    FINAL = "final"  # Final output products
    METADATA = "metadata"  # Metadata files (STAC, etc.)


class ProductStatus(Enum):
    """Status of a product in the pipeline."""

    PENDING = "pending"  # Registered but not yet created
    CREATING = "creating"  # Currently being created
    READY = "ready"  # Successfully created and available
    EXPIRED = "expired"  # Past expiration, pending cleanup
    DELETED = "deleted"  # Cleaned up
    FAILED = "failed"  # Creation failed


@dataclass
class ProductInfo:
    """
    Information about an intermediate product.

    Attributes:
        product_id: Unique product identifier
        job_id: Parent ingestion job ID
        product_type: Type classification
        status: Current status
        storage_key: Key in storage backend
        size_bytes: Size in bytes
        checksum: Content checksum
        checksum_algorithm: Algorithm used for checksum
        content_type: MIME type
        format: Data format (cog, zarr, parquet, etc.)
        dependencies: IDs of products this depends on
        dependents: IDs of products depending on this
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        expires_at: Expiration timestamp
        metadata: Additional metadata
        tags: Searchable tags
    """

    product_id: str
    job_id: str
    product_type: ProductType
    status: ProductStatus = ProductStatus.PENDING
    storage_key: Optional[str] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    content_type: Optional[str] = None
    format: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "job_id": self.job_id,
            "product_type": self.product_type.value,
            "status": self.status.value,
            "storage_key": self.storage_key,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm.value,
            "content_type": self.content_type,
            "format": self.format,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductInfo":
        """Create from dictionary."""
        return cls(
            product_id=data["product_id"],
            job_id=data["job_id"],
            product_type=ProductType(data["product_type"]),
            status=ProductStatus(data.get("status", "pending")),
            storage_key=data.get("storage_key"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum"),
            checksum_algorithm=ChecksumAlgorithm(
                data.get("checksum_algorithm", "sha256")
            ),
            content_type=data.get("content_type"),
            format=data.get("format"),
            dependencies=data.get("dependencies", []),
            dependents=data.get("dependents", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            accessed_at=datetime.fromisoformat(data["accessed_at"])
            if data.get("accessed_at")
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )


@dataclass
class ProductManagerConfig:
    """
    Configuration for ProductManager.

    Attributes:
        storage_config: Storage backend configuration
        db_path: Path to SQLite database for tracking
        default_ttl_hours: Default time-to-live for products
        max_storage_bytes: Maximum total storage allowed
        cleanup_interval_seconds: Interval between cleanup runs
        enable_deduplication: Whether to deduplicate identical content
        preserve_final_products: Whether to skip cleanup of final products
    """

    storage_config: StorageConfig
    db_path: Optional[str] = None
    default_ttl_hours: int = 24
    max_storage_bytes: Optional[int] = None
    cleanup_interval_seconds: int = 3600
    enable_deduplication: bool = True
    preserve_final_products: bool = True


class ProductManager:
    """
    Manages intermediate products during ingestion.

    Provides registration, storage, retrieval, and cleanup of
    intermediate products with dependency tracking and lifecycle
    management.

    Example:
        config = ProductManagerConfig(
            storage_config=StorageConfig(
                storage_type=StorageType.LOCAL,
                root_path="/tmp/products"
            )
        )
        manager = ProductManager(config)

        # Register a product
        product = manager.register(
            job_id="ingest_123",
            product_type=ProductType.NORMALIZED,
            format="cog"
        )

        # Store content
        manager.store_file(product.product_id, "/path/to/file.tif")

        # Retrieve later
        path = manager.retrieve_file(product.product_id, "/tmp/output.tif")
    """

    def __init__(self, config: ProductManagerConfig):
        """
        Initialize ProductManager.

        Args:
            config: Manager configuration
        """
        self.config = config
        self._storage = create_storage_backend(config.storage_config)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None

        # Content hash -> product_id for deduplication
        self._content_index: Dict[str, str] = {}

        self._init_database()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

    def _init_database(self):
        """Initialize SQLite database for product tracking."""
        db_path = self.config.db_path
        if db_path is None:
            if self.config.storage_config.storage_type == StorageType.LOCAL:
                db_path = str(
                    Path(self.config.storage_config.root_path) / ".products.db"
                )
            else:
                db_path = ":memory:"

        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                product_type TEXT NOT NULL,
                status TEXT NOT NULL,
                storage_key TEXT,
                size_bytes INTEGER DEFAULT 0,
                checksum TEXT,
                checksum_algorithm TEXT DEFAULT 'sha256',
                content_type TEXT,
                format TEXT,
                dependencies TEXT DEFAULT '[]',
                dependents TEXT DEFAULT '[]',
                created_at TEXT,
                accessed_at TEXT,
                expires_at TEXT,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]'
            )
        """)

        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_id ON products(job_id)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON products(status)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON products(expires_at)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_checksum ON products(checksum)
        """)

        self._db.commit()
        logger.info(f"Initialized product database at {db_path}")

    def _generate_id(self, job_id: str, product_type: ProductType) -> str:
        """Generate a unique product ID."""
        import uuid

        short_uuid = str(uuid.uuid4())[:8]
        return f"prod_{job_id}_{product_type.value}_{short_uuid}"

    def _generate_storage_key(self, product_id: str, format: Optional[str]) -> str:
        """Generate storage key for a product."""
        ext = f".{format}" if format else ""
        return f"products/{product_id}{ext}"

    def _save_product(self, product: ProductInfo):
        """Save product info to database."""
        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO products
                (product_id, job_id, product_type, status, storage_key,
                 size_bytes, checksum, checksum_algorithm, content_type, format,
                 dependencies, dependents, created_at, accessed_at, expires_at,
                 metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product.product_id,
                    product.job_id,
                    product.product_type.value,
                    product.status.value,
                    product.storage_key,
                    product.size_bytes,
                    product.checksum,
                    product.checksum_algorithm.value,
                    product.content_type,
                    product.format,
                    json.dumps(product.dependencies),
                    json.dumps(product.dependents),
                    product.created_at.isoformat() if product.created_at else None,
                    product.accessed_at.isoformat() if product.accessed_at else None,
                    product.expires_at.isoformat() if product.expires_at else None,
                    json.dumps(product.metadata),
                    json.dumps(list(product.tags)),
                ),
            )
            self._db.commit()

    def _load_product(self, product_id: str) -> Optional[ProductInfo]:
        """Load product info from database."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM products WHERE product_id = ?", (product_id,)
            ).fetchone()

        if not row:
            return None

        return ProductInfo(
            product_id=row["product_id"],
            job_id=row["job_id"],
            product_type=ProductType(row["product_type"]),
            status=ProductStatus(row["status"]),
            storage_key=row["storage_key"],
            size_bytes=row["size_bytes"],
            checksum=row["checksum"],
            checksum_algorithm=ChecksumAlgorithm(
                row["checksum_algorithm"] or "sha256"
            ),
            content_type=row["content_type"],
            format=row["format"],
            dependencies=json.loads(row["dependencies"] or "[]"),
            dependents=json.loads(row["dependents"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            accessed_at=datetime.fromisoformat(row["accessed_at"])
            if row["accessed_at"]
            else None,
            expires_at=datetime.fromisoformat(row["expires_at"])
            if row["expires_at"]
            else None,
            metadata=json.loads(row["metadata"] or "{}"),
            tags=set(json.loads(row["tags"] or "[]")),
        )

    def register(
        self,
        job_id: str,
        product_type: ProductType,
        format: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        ttl_hours: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> ProductInfo:
        """
        Register a new intermediate product.

        Args:
            job_id: Parent job ID
            product_type: Type classification
            format: Data format
            dependencies: Product IDs this depends on
            ttl_hours: Time-to-live (uses default if None)
            metadata: Additional metadata
            tags: Searchable tags

        Returns:
            ProductInfo for the registered product
        """
        product_id = self._generate_id(job_id, product_type)
        storage_key = self._generate_storage_key(product_id, format)
        now = datetime.now(timezone.utc)
        ttl = ttl_hours or self.config.default_ttl_hours
        expires = now + timedelta(hours=ttl)

        product = ProductInfo(
            product_id=product_id,
            job_id=job_id,
            product_type=product_type,
            status=ProductStatus.PENDING,
            storage_key=storage_key,
            format=format,
            dependencies=dependencies or [],
            created_at=now,
            accessed_at=now,
            expires_at=expires,
            metadata=metadata or {},
            tags=tags or set(),
        )

        # Update dependents for dependencies
        for dep_id in product.dependencies:
            dep = self._load_product(dep_id)
            if dep:
                dep.dependents.append(product_id)
                self._save_product(dep)

        self._save_product(product)
        logger.debug(f"Registered product {product_id}")

        return product

    def store_file(
        self,
        product_id: str,
        local_path: Union[str, Path],
        content_type: Optional[str] = None,
    ) -> ProductInfo:
        """
        Store a file as product content.

        Args:
            product_id: Product ID
            local_path: Path to local file
            content_type: MIME type

        Returns:
            Updated ProductInfo
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        product.status = ProductStatus.CREATING
        self._save_product(product)

        try:
            result = self._storage.upload_file(
                local_path,
                product.storage_key,
                content_type=content_type,
                metadata={"product_id": product_id, "job_id": product.job_id},
            )

            product.status = ProductStatus.READY
            product.size_bytes = result.size_bytes
            product.checksum = result.checksum
            product.checksum_algorithm = result.checksum_algorithm
            product.content_type = content_type
            product.accessed_at = datetime.now(timezone.utc)

            # Index for deduplication
            if self.config.enable_deduplication and result.checksum:
                self._content_index[result.checksum] = product_id

            self._save_product(product)
            logger.debug(f"Stored file for product {product_id}")

            return product

        except Exception as e:
            product.status = ProductStatus.FAILED
            product.metadata["error"] = str(e)
            self._save_product(product)
            raise

    def store_bytes(
        self,
        product_id: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> ProductInfo:
        """
        Store bytes as product content.

        Args:
            product_id: Product ID
            data: Bytes to store
            content_type: MIME type

        Returns:
            Updated ProductInfo
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        product.status = ProductStatus.CREATING
        self._save_product(product)

        try:
            # Check for duplicate content
            if self.config.enable_deduplication:
                checksum = hashlib.sha256(data).hexdigest()
                if checksum in self._content_index:
                    existing_id = self._content_index[checksum]
                    existing = self._load_product(existing_id)
                    if existing and existing.status == ProductStatus.READY:
                        # Link to existing content
                        product.status = ProductStatus.READY
                        product.storage_key = existing.storage_key
                        product.size_bytes = existing.size_bytes
                        product.checksum = checksum
                        product.content_type = existing.content_type
                        product.accessed_at = datetime.now(timezone.utc)
                        product.metadata["deduplicated_from"] = existing_id
                        self._save_product(product)
                        logger.debug(
                            f"Deduplicated product {product_id} from {existing_id}"
                        )
                        return product

            result = self._storage.upload_bytes(
                data,
                product.storage_key,
                content_type=content_type,
                metadata={"product_id": product_id, "job_id": product.job_id},
            )

            product.status = ProductStatus.READY
            product.size_bytes = result.size_bytes
            product.checksum = result.checksum
            product.checksum_algorithm = result.checksum_algorithm
            product.content_type = content_type
            product.accessed_at = datetime.now(timezone.utc)

            if self.config.enable_deduplication and result.checksum:
                self._content_index[result.checksum] = product_id

            self._save_product(product)
            logger.debug(f"Stored bytes for product {product_id}")

            return product

        except Exception as e:
            product.status = ProductStatus.FAILED
            product.metadata["error"] = str(e)
            self._save_product(product)
            raise

    def retrieve_file(
        self,
        product_id: str,
        local_path: Union[str, Path],
        verify_checksum: bool = True,
    ) -> Path:
        """
        Retrieve product content to a local file.

        Args:
            product_id: Product ID
            local_path: Destination path
            verify_checksum: Whether to verify content checksum

        Returns:
            Path to downloaded file
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        if product.status != ProductStatus.READY:
            raise ValueError(f"Product not ready: {product_id} ({product.status})")

        self._storage.download_file(
            product.storage_key, local_path, verify_checksum=verify_checksum
        )

        product.accessed_at = datetime.now(timezone.utc)
        self._save_product(product)

        return Path(local_path)

    def retrieve_bytes(self, product_id: str) -> bytes:
        """
        Retrieve product content as bytes.

        Args:
            product_id: Product ID

        Returns:
            Content bytes
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        if product.status != ProductStatus.READY:
            raise ValueError(f"Product not ready: {product_id} ({product.status})")

        data, _ = self._storage.download_bytes(product.storage_key)

        product.accessed_at = datetime.now(timezone.utc)
        self._save_product(product)

        return data

    def get_uri(self, product_id: str) -> str:
        """
        Get storage URI for a product.

        Args:
            product_id: Product ID

        Returns:
            Full storage URI
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        return self._storage.get_uri(product.storage_key)

    def get(self, product_id: str) -> Optional[ProductInfo]:
        """
        Get product information.

        Args:
            product_id: Product ID

        Returns:
            ProductInfo if exists, None otherwise
        """
        return self._load_product(product_id)

    def list_by_job(
        self,
        job_id: str,
        product_type: Optional[ProductType] = None,
        status: Optional[ProductStatus] = None,
    ) -> List[ProductInfo]:
        """
        List products for a job.

        Args:
            job_id: Job ID
            product_type: Filter by type
            status: Filter by status

        Returns:
            List of matching products
        """
        with self._lock:
            query = "SELECT product_id FROM products WHERE job_id = ?"
            params = [job_id]

            if product_type:
                query += " AND product_type = ?"
                params.append(product_type.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            rows = self._db.execute(query, params).fetchall()

        return [self._load_product(row["product_id"]) for row in rows]

    def list_by_type(
        self,
        product_type: ProductType,
        status: Optional[ProductStatus] = None,
    ) -> List[ProductInfo]:
        """
        List products by type.

        Args:
            product_type: Product type
            status: Filter by status

        Returns:
            List of matching products
        """
        with self._lock:
            query = "SELECT product_id FROM products WHERE product_type = ?"
            params = [product_type.value]

            if status:
                query += " AND status = ?"
                params.append(status.value)

            rows = self._db.execute(query, params).fetchall()

        return [self._load_product(row["product_id"]) for row in rows]

    def search_by_tags(
        self,
        tags: Set[str],
        match_all: bool = True,
    ) -> List[ProductInfo]:
        """
        Search products by tags.

        Args:
            tags: Tags to search for
            match_all: Whether all tags must match

        Returns:
            List of matching products
        """
        with self._lock:
            rows = self._db.execute("SELECT product_id, tags FROM products").fetchall()

        results = []
        for row in rows:
            product_tags = set(json.loads(row["tags"] or "[]"))
            if match_all:
                if tags.issubset(product_tags):
                    results.append(self._load_product(row["product_id"]))
            else:
                if tags.intersection(product_tags):
                    results.append(self._load_product(row["product_id"]))

        return results

    def get_dependencies(self, product_id: str) -> List[ProductInfo]:
        """
        Get products this product depends on.

        Args:
            product_id: Product ID

        Returns:
            List of dependency products
        """
        product = self._load_product(product_id)
        if not product:
            return []

        return [
            self._load_product(dep_id)
            for dep_id in product.dependencies
            if self._load_product(dep_id)
        ]

    def get_dependents(self, product_id: str) -> List[ProductInfo]:
        """
        Get products that depend on this product.

        Args:
            product_id: Product ID

        Returns:
            List of dependent products
        """
        product = self._load_product(product_id)
        if not product:
            return []

        return [
            self._load_product(dep_id)
            for dep_id in product.dependents
            if self._load_product(dep_id)
        ]

    def mark_expired(self, product_id: str) -> ProductInfo:
        """
        Mark a product as expired.

        Args:
            product_id: Product ID

        Returns:
            Updated ProductInfo
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        product.status = ProductStatus.EXPIRED
        self._save_product(product)

        return product

    def extend_expiration(
        self,
        product_id: str,
        hours: int,
    ) -> ProductInfo:
        """
        Extend product expiration.

        Args:
            product_id: Product ID
            hours: Additional hours

        Returns:
            Updated ProductInfo
        """
        product = self._load_product(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        if product.expires_at:
            product.expires_at = product.expires_at + timedelta(hours=hours)
        else:
            product.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)

        self._save_product(product)

        return product

    def delete(self, product_id: str, cascade: bool = False) -> bool:
        """
        Delete a product.

        Args:
            product_id: Product ID
            cascade: Whether to delete dependents

        Returns:
            True if deleted
        """
        product = self._load_product(product_id)
        if not product:
            return False

        # Check for dependents
        if product.dependents and not cascade:
            raise ValueError(
                f"Product has dependents: {product.dependents}. Use cascade=True."
            )

        # Delete dependents first
        if cascade:
            for dep_id in product.dependents:
                self.delete(dep_id, cascade=True)

        # Remove from dependencies' dependents lists
        for dep_id in product.dependencies:
            dep = self._load_product(dep_id)
            if dep and product_id in dep.dependents:
                dep.dependents.remove(product_id)
                self._save_product(dep)

        # Delete storage content
        if product.storage_key:
            # Check if content is shared (deduplication)
            shared = False
            if product.checksum and product.checksum in self._content_index:
                if self._content_index[product.checksum] != product_id:
                    shared = True
                else:
                    # Check if any other product references this content
                    with self._lock:
                        rows = self._db.execute(
                            "SELECT COUNT(*) as cnt FROM products WHERE checksum = ? AND product_id != ?",
                            (product.checksum, product_id),
                        ).fetchone()
                    if rows["cnt"] > 0:
                        shared = True

            if not shared:
                self._storage.delete(product.storage_key)
                if product.checksum and product.checksum in self._content_index:
                    del self._content_index[product.checksum]

        # Delete from database
        with self._lock:
            self._db.execute(
                "DELETE FROM products WHERE product_id = ?", (product_id,)
            )
            self._db.commit()

        logger.debug(f"Deleted product {product_id}")
        return True

    def cleanup_expired(self) -> List[str]:
        """
        Clean up expired products.

        Returns:
            List of deleted product IDs
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            rows = self._db.execute(
                """
                SELECT product_id, product_type FROM products
                WHERE status != 'deleted'
                AND expires_at IS NOT NULL
                AND expires_at < ?
                """,
                (now.isoformat(),),
            ).fetchall()

        deleted = []
        for row in rows:
            product_id = row["product_id"]
            product_type = ProductType(row["product_type"])

            # Skip final products if configured
            if (
                self.config.preserve_final_products
                and product_type == ProductType.FINAL
            ):
                continue

            try:
                self.delete(product_id)
                deleted.append(product_id)
            except ValueError as e:
                logger.warning(f"Could not delete {product_id}: {e}")

        if deleted:
            logger.info(f"Cleaned up {len(deleted)} expired products")

        return deleted

    def cleanup_job(self, job_id: str, keep_final: bool = True) -> List[str]:
        """
        Clean up all products for a job.

        Args:
            job_id: Job ID
            keep_final: Whether to keep final products

        Returns:
            List of deleted product IDs
        """
        products = self.list_by_job(job_id)
        deleted = []

        for product in products:
            if keep_final and product.product_type == ProductType.FINAL:
                continue

            try:
                self.delete(product.product_id, cascade=True)
                deleted.append(product.product_id)
            except ValueError as e:
                logger.warning(f"Could not delete {product.product_id}: {e}")

        if deleted:
            logger.info(f"Cleaned up {len(deleted)} products for job {job_id}")

        return deleted

    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.

        Returns:
            Dictionary with storage stats
        """
        with self._lock:
            rows = self._db.execute(
                """
                SELECT
                    product_type,
                    status,
                    COUNT(*) as count,
                    SUM(size_bytes) as total_bytes
                FROM products
                GROUP BY product_type, status
                """
            ).fetchall()

        stats = {
            "total_products": 0,
            "total_bytes": 0,
            "by_type": {},
            "by_status": {},
        }

        for row in rows:
            pt = row["product_type"]
            st = row["status"]
            count = row["count"]
            size = row["total_bytes"] or 0

            stats["total_products"] += count
            stats["total_bytes"] += size

            if pt not in stats["by_type"]:
                stats["by_type"][pt] = {"count": 0, "bytes": 0}
            stats["by_type"][pt]["count"] += count
            stats["by_type"][pt]["bytes"] += size

            if st not in stats["by_status"]:
                stats["by_status"][st] = {"count": 0, "bytes": 0}
            stats["by_status"][st]["count"] += count
            stats["by_status"][st]["bytes"] += size

        return stats

    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._stop_cleanup.clear()

        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")

                self._stop_cleanup.wait(self.config.cleanup_interval_seconds)

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started cleanup thread")

    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Stopped cleanup thread")

    def close(self):
        """Close manager and release resources."""
        self.stop_cleanup_thread()
        if self._db:
            self._db.close()
            self._db = None
