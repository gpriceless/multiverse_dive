"""
Tests for the Persistence Layer (Group G, Track 6).

Tests cover:
- Storage backends (local and S3)
- Product management lifecycle
- Lineage tracking and provenance
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.data.ingestion.persistence import (
    # Storage
    ChecksumAlgorithm,
    LocalStorageBackend,
    StorageConfig,
    StorageObject,
    StorageType,
    UploadResult,
    create_storage_backend,
    parse_storage_uri,
    # Product Management
    ProductInfo,
    ProductManager,
    ProductManagerConfig,
    ProductStatus,
    ProductType,
    # Lineage
    AlgorithmInfo,
    InputDataset,
    LineageEventType,
    LineageTracker,
    ProcessingStep,
    ProvenanceRecord,
    QualitySummary,
    TrackingContext,
    compute_environment_hash,
    create_provenance_from_job,
    get_compute_resources,
    get_software_environment,
)


# =============================================================================
# Storage Backend Tests
# =============================================================================


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_local_config_basic(self):
        """Test basic local storage configuration."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path="/tmp/test_storage",
        )
        assert config.storage_type == StorageType.LOCAL
        assert config.root_path == "/tmp/test_storage"
        assert config.checksum_algorithm == ChecksumAlgorithm.SHA256

    def test_config_validation_chunk_size(self):
        """Test that invalid chunk_size_mb is rejected."""
        with pytest.raises(ValueError, match="chunk_size_mb must be >= 1"):
            StorageConfig(
                storage_type=StorageType.LOCAL,
                root_path="/tmp",
                chunk_size_mb=0,
            )

    def test_config_validation_max_retries(self):
        """Test that invalid max_retries is rejected."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            StorageConfig(
                storage_type=StorageType.LOCAL,
                root_path="/tmp",
                max_retries=-1,
            )

    def test_config_validation_timeout(self):
        """Test that invalid timeout_seconds is rejected."""
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            StorageConfig(
                storage_type=StorageType.LOCAL,
                root_path="/tmp",
                timeout_seconds=0,
            )


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a local storage backend in a temp directory."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        return LocalStorageBackend(config)

    def test_storage_type(self, storage):
        """Test storage type property."""
        assert storage.storage_type == StorageType.LOCAL

    def test_upload_bytes(self, storage):
        """Test uploading bytes."""
        data = b"Hello, World!"
        result = storage.upload_bytes(data, "test/hello.txt")

        assert result.key == "test/hello.txt"
        assert result.size_bytes == len(data)
        assert result.checksum is not None
        assert storage.exists("test/hello.txt")

    def test_upload_file(self, storage, tmp_path):
        """Test uploading a file."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Test content")

        result = storage.upload_file(test_file, "uploaded/file.txt")

        assert result.key == "uploaded/file.txt"
        assert result.size_bytes == 12  # "Test content"
        assert storage.exists("uploaded/file.txt")

    def test_upload_file_not_found(self, storage):
        """Test uploading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            storage.upload_file("/nonexistent/file.txt", "test.txt")

    def test_download_bytes(self, storage):
        """Test downloading as bytes."""
        original = b"Binary data content"
        storage.upload_bytes(original, "data.bin")

        data, obj = storage.download_bytes("data.bin")

        assert data == original
        assert obj.key == "data.bin"
        assert obj.size_bytes == len(original)

    def test_download_file(self, storage, tmp_path):
        """Test downloading to file."""
        storage.upload_bytes(b"File content", "source.txt")
        dest = tmp_path / "downloaded.txt"

        obj = storage.download_file("source.txt", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"File content"
        assert obj.key == "source.txt"

    def test_download_not_found(self, storage, tmp_path):
        """Test downloading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            storage.download_file("nonexistent.txt", tmp_path / "out.txt")

    def test_checksum_verification(self, storage, tmp_path):
        """Test checksum verification on download."""
        storage.upload_bytes(b"Original data", "verified.txt")

        # Corrupt the file
        internal_path = Path(storage.config.root_path) / "verified.txt"
        internal_path.write_bytes(b"Corrupted data")

        dest = tmp_path / "output.txt"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            storage.download_file("verified.txt", dest, verify_checksum=True)

    def test_exists(self, storage):
        """Test existence check."""
        assert not storage.exists("missing.txt")
        storage.upload_bytes(b"x", "present.txt")
        assert storage.exists("present.txt")

    def test_get_object_info(self, storage):
        """Test getting object info without downloading."""
        storage.upload_bytes(b"Info test", "info.txt", content_type="text/plain")

        obj = storage.get_object_info("info.txt")

        assert obj is not None
        assert obj.key == "info.txt"
        assert obj.size_bytes == 9
        assert obj.content_type == "text/plain"

    def test_get_object_info_not_found(self, storage):
        """Test getting info for non-existent object."""
        assert storage.get_object_info("missing.txt") is None

    def test_list_objects(self, storage):
        """Test listing objects."""
        storage.upload_bytes(b"1", "a/1.txt")
        storage.upload_bytes(b"2", "a/2.txt")
        storage.upload_bytes(b"3", "b/3.txt")

        # List all
        all_objs = list(storage.list_objects())
        assert len(all_objs) == 3

        # List with prefix
        a_objs = list(storage.list_objects(prefix="a/"))
        assert len(a_objs) == 2

    def test_list_objects_max_results(self, storage):
        """Test listing with max_results."""
        for i in range(10):
            storage.upload_bytes(f"{i}".encode(), f"file_{i}.txt")

        limited = list(storage.list_objects(max_results=3))
        assert len(limited) == 3

    def test_delete(self, storage):
        """Test deleting object."""
        storage.upload_bytes(b"x", "to_delete.txt")
        assert storage.exists("to_delete.txt")

        result = storage.delete("to_delete.txt")
        assert result is True
        assert not storage.exists("to_delete.txt")

    def test_delete_not_found(self, storage):
        """Test deleting non-existent object."""
        result = storage.delete("missing.txt")
        assert result is False

    def test_delete_many(self, storage):
        """Test bulk delete."""
        for i in range(5):
            storage.upload_bytes(f"{i}".encode(), f"bulk_{i}.txt")

        keys = [f"bulk_{i}.txt" for i in range(5)]
        deleted = storage.delete_many(keys)

        assert len(deleted) == 5
        for key in keys:
            assert not storage.exists(key)

    def test_copy(self, storage):
        """Test copying object."""
        storage.upload_bytes(b"Original", "original.txt")

        copied = storage.copy("original.txt", "copy.txt")

        assert storage.exists("original.txt")
        assert storage.exists("copy.txt")
        data, _ = storage.download_bytes("copy.txt")
        assert data == b"Original"

    def test_copy_not_found(self, storage):
        """Test copying non-existent object."""
        with pytest.raises(FileNotFoundError):
            storage.copy("missing.txt", "dest.txt")

    def test_move(self, storage):
        """Test moving object."""
        storage.upload_bytes(b"Moving", "source.txt")

        moved = storage.move("source.txt", "dest.txt")

        assert not storage.exists("source.txt")
        assert storage.exists("dest.txt")
        data, _ = storage.download_bytes("dest.txt")
        assert data == b"Moving"

    def test_get_uri(self, storage):
        """Test URI generation."""
        uri = storage.get_uri("path/to/file.txt")
        assert uri.startswith("file://")
        assert "path/to/file.txt" in uri

    def test_upload_with_metadata(self, storage):
        """Test uploading with custom metadata."""
        result = storage.upload_bytes(
            b"data",
            "meta.txt",
            metadata={"custom": "value", "purpose": "test"},
        )

        assert result.metadata == {"custom": "value", "purpose": "test"}

        obj = storage.get_object_info("meta.txt")
        assert obj.metadata.get("custom") == "value"


class TestStorageFactory:
    """Tests for storage factory function."""

    def test_create_local_backend(self, tmp_path):
        """Test creating local backend."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path),
        )
        backend = create_storage_backend(config)
        assert isinstance(backend, LocalStorageBackend)

    def test_unsupported_storage_type_gcs(self, tmp_path):
        """Test that GCS raises NotImplementedError."""
        config = StorageConfig(
            storage_type=StorageType.GCS,
            root_path="bucket",
        )
        with pytest.raises(NotImplementedError):
            create_storage_backend(config)

    def test_unsupported_storage_type_azure(self, tmp_path):
        """Test that Azure raises NotImplementedError."""
        config = StorageConfig(
            storage_type=StorageType.AZURE,
            root_path="container",
        )
        with pytest.raises(NotImplementedError):
            create_storage_backend(config)


class TestParseStorageUri:
    """Tests for parse_storage_uri function."""

    def test_parse_local_file_uri(self):
        """Test parsing file:// URI."""
        st, root, key = parse_storage_uri("file:///path/to/file.txt")
        assert st == StorageType.LOCAL
        assert key == "file.txt"

    def test_parse_s3_uri(self):
        """Test parsing s3:// URI."""
        st, bucket, key = parse_storage_uri("s3://mybucket/path/to/object")
        assert st == StorageType.S3
        assert bucket == "mybucket"
        assert key == "path/to/object"

    def test_parse_gs_uri(self):
        """Test parsing gs:// URI."""
        st, bucket, key = parse_storage_uri("gs://mybucket/path/to/object")
        assert st == StorageType.GCS
        assert bucket == "mybucket"
        assert key == "path/to/object"

    def test_parse_azure_uri(self):
        """Test parsing azure:// URI."""
        st, container, key = parse_storage_uri("azure://mycontainer/path/blob")
        assert st == StorageType.AZURE
        assert container == "mycontainer"
        assert key == "path/blob"

    def test_parse_unsupported_scheme(self):
        """Test parsing unsupported scheme raises error."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            parse_storage_uri("ftp://server/file")


# =============================================================================
# Product Manager Tests
# =============================================================================


class TestProductInfo:
    """Tests for ProductInfo dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        product = ProductInfo(
            product_id="prod_123",
            job_id="job_456",
            product_type=ProductType.NORMALIZED,
            status=ProductStatus.READY,
            size_bytes=1000,
            metadata={"key": "value"},
        )

        d = product.to_dict()

        assert d["product_id"] == "prod_123"
        assert d["job_id"] == "job_456"
        assert d["product_type"] == "normalized"
        assert d["status"] == "ready"
        assert d["size_bytes"] == 1000

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "product_id": "prod_123",
            "job_id": "job_456",
            "product_type": "raw",
            "status": "pending",
            "size_bytes": 500,
        }

        product = ProductInfo.from_dict(data)

        assert product.product_id == "prod_123"
        assert product.product_type == ProductType.RAW
        assert product.status == ProductStatus.PENDING


class TestProductManager:
    """Tests for ProductManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a product manager with temp storage."""
        storage_config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        config = ProductManagerConfig(
            storage_config=storage_config,
            db_path=str(tmp_path / "products.db"),
            default_ttl_hours=1,
        )
        mgr = ProductManager(config)
        yield mgr
        mgr.close()

    def test_register_product(self, manager):
        """Test registering a new product."""
        product = manager.register(
            job_id="job_123",
            product_type=ProductType.NORMALIZED,
            format="cog",
        )

        assert product.job_id == "job_123"
        assert product.product_type == ProductType.NORMALIZED
        assert product.status == ProductStatus.PENDING
        assert product.format == "cog"
        assert product.storage_key is not None

    def test_store_and_retrieve_file(self, manager, tmp_path):
        """Test storing and retrieving file content."""
        # Register
        product = manager.register(
            job_id="job_123",
            product_type=ProductType.RAW,
        )

        # Create test file
        input_file = tmp_path / "input.tif"
        input_file.write_bytes(b"GeoTIFF content")

        # Store
        updated = manager.store_file(product.product_id, input_file)
        assert updated.status == ProductStatus.READY
        assert updated.size_bytes == 15

        # Retrieve
        output_file = tmp_path / "output.tif"
        manager.retrieve_file(product.product_id, output_file)
        assert output_file.read_bytes() == b"GeoTIFF content"

    def test_store_and_retrieve_bytes(self, manager):
        """Test storing and retrieving bytes."""
        product = manager.register(
            job_id="job_456",
            product_type=ProductType.MASK,
        )

        data = b"Mask data bytes"
        manager.store_bytes(product.product_id, data)

        retrieved = manager.retrieve_bytes(product.product_id)
        assert retrieved == data

    def test_product_not_found(self, manager):
        """Test operations on non-existent product."""
        with pytest.raises(ValueError, match="Product not found"):
            manager.store_bytes("nonexistent", b"data")

    def test_retrieve_not_ready(self, manager):
        """Test retrieving product that isn't ready."""
        product = manager.register(
            job_id="job_123",
            product_type=ProductType.RAW,
        )

        with pytest.raises(ValueError, match="Product not ready"):
            manager.retrieve_bytes(product.product_id)

    def test_get_product(self, manager):
        """Test getting product info."""
        product = manager.register(
            job_id="job_123",
            product_type=ProductType.INDEX,
        )

        retrieved = manager.get(product.product_id)
        assert retrieved is not None
        assert retrieved.job_id == "job_123"

    def test_get_nonexistent(self, manager):
        """Test getting non-existent product."""
        assert manager.get("nonexistent") is None

    def test_list_by_job(self, manager):
        """Test listing products by job."""
        for i in range(3):
            manager.register(job_id="job_A", product_type=ProductType.RAW)
        for i in range(2):
            manager.register(job_id="job_B", product_type=ProductType.RAW)

        job_a_products = manager.list_by_job("job_A")
        assert len(job_a_products) == 3

        job_b_products = manager.list_by_job("job_B")
        assert len(job_b_products) == 2

    def test_list_by_type(self, manager):
        """Test listing products by type."""
        manager.register(job_id="job_1", product_type=ProductType.RAW)
        manager.register(job_id="job_2", product_type=ProductType.RAW)
        manager.register(job_id="job_3", product_type=ProductType.NORMALIZED)

        raw_products = manager.list_by_type(ProductType.RAW)
        assert len(raw_products) == 2

    def test_dependency_tracking(self, manager):
        """Test dependency tracking between products."""
        parent = manager.register(
            job_id="job_123",
            product_type=ProductType.RAW,
        )
        manager.store_bytes(parent.product_id, b"Parent data")

        child = manager.register(
            job_id="job_123",
            product_type=ProductType.NORMALIZED,
            dependencies=[parent.product_id],
        )

        # Check dependency is recorded
        assert parent.product_id in child.dependencies

        # Check dependent is updated on parent
        updated_parent = manager.get(parent.product_id)
        assert child.product_id in updated_parent.dependents

    def test_delete_product(self, manager):
        """Test deleting a product."""
        product = manager.register(
            job_id="job_123",
            product_type=ProductType.RAW,
        )
        manager.store_bytes(product.product_id, b"data")

        result = manager.delete(product.product_id)
        assert result is True
        assert manager.get(product.product_id) is None

    def test_delete_with_dependents_blocked(self, manager):
        """Test that deleting product with dependents is blocked."""
        parent = manager.register(job_id="job_1", product_type=ProductType.RAW)
        manager.store_bytes(parent.product_id, b"parent")

        child = manager.register(
            job_id="job_1",
            product_type=ProductType.NORMALIZED,
            dependencies=[parent.product_id],
        )

        with pytest.raises(ValueError, match="has dependents"):
            manager.delete(parent.product_id)

    def test_delete_cascade(self, manager):
        """Test cascade delete of product and dependents."""
        parent = manager.register(job_id="job_1", product_type=ProductType.RAW)
        manager.store_bytes(parent.product_id, b"parent")

        child = manager.register(
            job_id="job_1",
            product_type=ProductType.NORMALIZED,
            dependencies=[parent.product_id],
        )
        manager.store_bytes(child.product_id, b"child")

        result = manager.delete(parent.product_id, cascade=True)
        assert result is True
        assert manager.get(parent.product_id) is None
        assert manager.get(child.product_id) is None

    def test_cleanup_expired(self, manager):
        """Test cleanup of expired products."""
        # Create product with 0 TTL (immediate expiration)
        product = manager.register(
            job_id="job_1",
            product_type=ProductType.RAW,
            ttl_hours=0,  # Expired immediately
        )
        manager.store_bytes(product.product_id, b"data")

        # Force expiration
        p = manager._load_product(product.product_id)
        p.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        manager._save_product(p)

        deleted = manager.cleanup_expired()
        assert product.product_id in deleted

    def test_deduplication(self, manager):
        """Test content deduplication."""
        data = b"Identical content"

        prod1 = manager.register(job_id="job_1", product_type=ProductType.RAW)
        manager.store_bytes(prod1.product_id, data)

        prod2 = manager.register(job_id="job_2", product_type=ProductType.RAW)
        manager.store_bytes(prod2.product_id, data)

        # Check that prod2 was deduplicated
        p2 = manager.get(prod2.product_id)
        assert p2.metadata.get("deduplicated_from") == prod1.product_id

    def test_storage_usage(self, manager):
        """Test storage usage statistics."""
        manager.register(job_id="job_1", product_type=ProductType.RAW)
        manager.register(job_id="job_2", product_type=ProductType.NORMALIZED)

        stats = manager.get_storage_usage()
        assert stats["total_products"] >= 2
        assert "by_type" in stats
        assert "by_status" in stats

    def test_extend_expiration(self, manager):
        """Test extending product expiration."""
        product = manager.register(
            job_id="job_1",
            product_type=ProductType.RAW,
            ttl_hours=1,
        )

        original_expiry = product.expires_at
        updated = manager.extend_expiration(product.product_id, hours=24)

        assert updated.expires_at > original_expiry

    def test_tags(self, manager):
        """Test tag-based search."""
        prod1 = manager.register(
            job_id="job_1",
            product_type=ProductType.RAW,
            tags={"flood", "sentinel1"},
        )
        prod2 = manager.register(
            job_id="job_2",
            product_type=ProductType.RAW,
            tags={"wildfire", "sentinel2"},
        )

        flood_products = manager.search_by_tags({"flood"})
        assert len(flood_products) == 1
        assert flood_products[0].product_id == prod1.product_id


# =============================================================================
# Lineage Tracking Tests
# =============================================================================


class TestInputDataset:
    """Tests for InputDataset dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ds = InputDataset(
            dataset_id="S2A_20230101",
            provider="sentinel2",
            data_type="optical",
            checksum="abc123",
            bands=["B02", "B03", "B04"],
        )

        d = ds.to_dict()

        assert d["dataset_id"] == "S2A_20230101"
        assert d["provider"] == "sentinel2"
        assert d["checksum"]["value"] == "abc123"


class TestAlgorithmInfo:
    """Tests for AlgorithmInfo dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alg = AlgorithmInfo(
            algorithm_id="ndwi_v1",
            name="NDWI",
            version="1.0.0",
            parameters={"threshold": 0.3},
        )

        d = alg.to_dict()

        assert d["name"] == "NDWI"
        assert d["parameters"]["threshold"] == 0.3


class TestProcessingStep:
    """Tests for ProcessingStep dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        step = ProcessingStep(
            step="normalization",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            inputs=["input1"],
            outputs=["output1"],
            execution_time_seconds=10.5,
        )

        d = step.to_dict()

        assert d["step"] == "normalization"
        assert d["execution_time_seconds"] == 10.5


class TestLineageTracker:
    """Tests for LineageTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a lineage tracker with temp database."""
        db_path = str(tmp_path / "lineage.db")
        trk = LineageTracker(db_path)
        yield trk
        trk.close()

    def test_start_tracking(self, tracker):
        """Test starting tracking context."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        assert isinstance(ctx, TrackingContext)
        assert tracker.get_context("prod_123") is ctx

    def test_add_input(self, tracker):
        """Test adding input dataset."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        ctx.add_input(InputDataset(
            dataset_id="S2A_123",
            provider="sentinel2",
            data_type="optical",
        ))

        assert len(ctx._inputs) == 1

    def test_add_algorithm(self, tracker):
        """Test adding algorithm."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        ctx.add_algorithm(AlgorithmInfo(
            algorithm_id="ndwi_v1",
            name="NDWI",
            version="1.0.0",
        ))

        assert len(ctx._algorithms) == 1

    def test_step_context(self, tracker):
        """Test step context manager."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        with ctx.step("normalization") as step:
            step.set_processor("reprojection", "1.0")
            step.add_parameter("target_crs", "EPSG:4326")
            step.add_input("input_1")
            step.add_output("output_1")

        assert len(ctx._steps) == 1
        assert ctx._steps[0].step == "normalization"
        assert ctx._steps[0].processor == "reprojection"

    def test_step_context_timing(self, tracker):
        """Test that step context records execution time."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        with ctx.step("slow_step") as step:
            time.sleep(0.1)  # Small delay

        assert ctx._steps[0].execution_time_seconds >= 0.1

    def test_step_context_error(self, tracker):
        """Test step context with error."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        with pytest.raises(ValueError):
            with ctx.step("failing_step") as step:
                raise ValueError("Test error")

        assert ctx._steps[0].status == "failed"
        assert "Test error" in ctx._steps[0].error_message

    def test_finish_tracking(self, tracker):
        """Test finishing tracking and getting record."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        ctx.add_input(InputDataset(
            dataset_id="input_1",
            provider="test",
            data_type="test",
        ))

        with ctx.step("process") as step:
            step.set_processor("test_proc", "1.0")

        ctx.set_quality(QualitySummary(
            overall_confidence=0.95,
            uncertainty_percent=5.0,
        ))

        record = ctx.finish()

        assert isinstance(record, ProvenanceRecord)
        assert record.product_id == "prod_123"
        assert record.event_id == "evt_456"
        assert len(record.input_datasets) == 1
        assert len(record.lineage) == 1
        assert record.quality_summary.overall_confidence == 0.95

    def test_double_finish_error(self, tracker):
        """Test that double finish raises error."""
        ctx = tracker.start_tracking("prod_123", "evt_456")
        ctx.finish()

        with pytest.raises(RuntimeError, match="already finished"):
            ctx.finish()

    def test_get_record(self, tracker):
        """Test retrieving saved record."""
        ctx = tracker.start_tracking("prod_123", "evt_456")
        ctx.finish()

        record = tracker.get_record("prod_123")
        assert record is not None
        assert record.product_id == "prod_123"

    def test_list_records(self, tracker):
        """Test listing records."""
        for i in range(3):
            ctx = tracker.start_tracking(f"prod_{i}", "evt_1")
            ctx.finish()

        records = tracker.list_records()
        assert len(records) == 3

    def test_list_records_by_event(self, tracker):
        """Test listing records by event ID."""
        ctx1 = tracker.start_tracking("prod_1", "evt_A")
        ctx1.finish()
        ctx2 = tracker.start_tracking("prod_2", "evt_A")
        ctx2.finish()
        ctx3 = tracker.start_tracking("prod_3", "evt_B")
        ctx3.finish()

        evt_a_records = tracker.list_records(event_id="evt_A")
        assert len(evt_a_records) == 2

    def test_get_events(self, tracker):
        """Test getting lineage events."""
        ctx = tracker.start_tracking("prod_123", "evt_456")

        ctx.add_input(InputDataset(
            dataset_id="input_1",
            provider="test",
            data_type="test",
        ))

        ctx.finish()

        events = tracker.get_events("prod_123")
        assert len(events) >= 1  # At least input registered

    def test_export_to_json(self, tracker, tmp_path):
        """Test exporting record to JSON."""
        ctx = tracker.start_tracking("prod_123", "evt_456")
        ctx.finish()

        output_path = tmp_path / "provenance.json"
        result = tracker.export_to_json("prod_123", output_path)

        assert result.exists()
        data = json.loads(result.read_text())
        assert data["product_id"] == "prod_123"


class TestProvenanceRecord:
    """Tests for ProvenanceRecord."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ProvenanceRecord(
            product_id="prod_123",
            event_id="evt_456",
            lineage=[],
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        d = record.to_dict()

        assert d["product_id"] == "prod_123"
        assert d["event_id"] == "evt_456"

    def test_to_json(self):
        """Test JSON serialization."""
        record = ProvenanceRecord(
            product_id="prod_123",
            event_id="evt_456",
        )

        json_str = record.to_json()
        data = json.loads(json_str)

        assert data["product_id"] == "prod_123"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_software_environment(self):
        """Test getting software environment."""
        env = get_software_environment()

        assert "python" in env
        assert "platform" in env

    def test_get_compute_resources(self):
        """Test getting compute resources."""
        resources = get_compute_resources()

        assert "cpu_cores" in resources
        assert resources["cpu_cores"] >= 1

    def test_compute_environment_hash(self):
        """Test environment hash is deterministic."""
        hash1 = compute_environment_hash()
        hash2 = compute_environment_hash()

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_create_provenance_from_job(self):
        """Test creating provenance from job data."""
        job_data = {
            "source": {
                "uri": "s3://bucket/file.tif",
                "provider": "sentinel2",
                "data_type": "optical",
            },
            "normalization": {"target_crs": "EPSG:4326"},
        }

        record = create_provenance_from_job("prod_123", "evt_456", job_data)

        assert record.product_id == "prod_123"
        assert record.event_id == "evt_456"
        assert len(record.input_datasets) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestPersistenceIntegration:
    """Integration tests for the persistence layer."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up integrated product manager and lineage tracker."""
        storage_config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        pm_config = ProductManagerConfig(
            storage_config=storage_config,
            db_path=str(tmp_path / "products.db"),
        )
        manager = ProductManager(pm_config)
        tracker = LineageTracker(str(tmp_path / "lineage.db"))

        yield {"manager": manager, "tracker": tracker, "tmp_path": tmp_path}

        manager.close()
        tracker.close()

    def test_full_workflow(self, setup):
        """Test complete workflow: ingest, track, retrieve."""
        manager = setup["manager"]
        tracker = setup["tracker"]
        tmp_path = setup["tmp_path"]

        # Start lineage tracking
        ctx = tracker.start_tracking("product_final", "event_123", agent="test_agent")

        # Register input
        ctx.add_input(InputDataset(
            dataset_id="sentinel2_scene",
            provider="sentinel2",
            data_type="optical",
            acquisition_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))

        # Register and store raw product
        raw_product = manager.register(
            job_id="event_123",
            product_type=ProductType.RAW,
            format="tif",
        )

        raw_file = tmp_path / "raw.tif"
        raw_file.write_bytes(b"Raw satellite data")
        manager.store_file(raw_product.product_id, raw_file)

        # Process with lineage tracking
        with ctx.step("normalization") as step:
            step.add_input(raw_product.product_id)
            step.set_processor("reprojection", "1.0.0")
            step.add_parameter("target_crs", "EPSG:4326")

            # Create normalized product
            norm_product = manager.register(
                job_id="event_123",
                product_type=ProductType.NORMALIZED,
                format="cog",
                dependencies=[raw_product.product_id],
            )

            norm_data = b"Normalized COG data"
            manager.store_bytes(norm_product.product_id, norm_data)
            step.add_output(norm_product.product_id)

        ctx.add_algorithm(AlgorithmInfo(
            algorithm_id="reprojection_v1",
            name="Reprojection",
            version="1.0.0",
        ))

        ctx.set_quality(QualitySummary(
            overall_confidence=0.92,
            uncertainty_percent=8.0,
        ))

        # Finish tracking
        record = ctx.finish()

        # Verify everything
        assert record.product_id == "product_final"
        assert len(record.lineage) == 1
        assert record.quality_summary.overall_confidence == 0.92

        # Verify products
        assert manager.get(raw_product.product_id).status == ProductStatus.READY
        assert manager.get(norm_product.product_id).status == ProductStatus.READY

        # Verify retrieval
        retrieved = manager.retrieve_bytes(norm_product.product_id)
        assert retrieved == b"Normalized COG data"

        # Verify provenance export
        export_path = tmp_path / "provenance.json"
        tracker.export_to_json("product_final", export_path)
        assert export_path.exists()


class TestConcurrency:
    """Tests for concurrent access."""

    def test_concurrent_product_creation(self, tmp_path):
        """Test creating products from multiple threads."""
        storage_config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        config = ProductManagerConfig(
            storage_config=storage_config,
            db_path=str(tmp_path / "products.db"),
        )
        manager = ProductManager(config)

        products = []
        errors = []

        def create_product(idx):
            try:
                product = manager.register(
                    job_id=f"job_{idx}",
                    product_type=ProductType.RAW,
                )
                manager.store_bytes(product.product_id, f"data_{idx}".encode())
                products.append(product.product_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_product, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        manager.close()

        assert len(errors) == 0
        assert len(products) == 10


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestStorageEdgeCases:
    """Edge case tests for storage backend."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a local storage backend in a temp directory."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        return LocalStorageBackend(config)

    def test_upload_empty_bytes(self, storage):
        """Test uploading empty bytes."""
        result = storage.upload_bytes(b"", "empty.txt")
        assert result.size_bytes == 0
        assert storage.exists("empty.txt")
        data, _ = storage.download_bytes("empty.txt")
        assert data == b""

    def test_upload_bytes_special_characters_in_key(self, storage):
        """Test uploading with special characters in key."""
        data = b"test data"
        result = storage.upload_bytes(data, "path/with spaces/file.txt")
        assert storage.exists("path/with spaces/file.txt")

    def test_list_objects_empty_storage(self, storage):
        """Test listing objects in empty storage."""
        objects = list(storage.list_objects())
        assert len(objects) == 0

    def test_delete_many_empty_list(self, storage):
        """Test bulk delete with empty list."""
        deleted = storage.delete_many([])
        assert deleted == []

    def test_copy_to_same_key_raises_error(self, storage):
        """Test copying object to same key raises SameFileError."""
        import shutil
        storage.upload_bytes(b"original", "same.txt")
        with pytest.raises(shutil.SameFileError):
            storage.copy("same.txt", "same.txt")

    def test_upload_stream_empty(self, storage):
        """Test uploading empty stream."""
        import io
        stream = io.BytesIO(b"")
        result = storage.upload_stream(stream, "empty_stream.txt")
        assert result.size_bytes == 0

    def test_md5_checksum_algorithm(self, tmp_path):
        """Test MD5 checksum algorithm."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
            checksum_algorithm=ChecksumAlgorithm.MD5,
        )
        storage = LocalStorageBackend(config)
        result = storage.upload_bytes(b"test", "file.txt")
        assert result.checksum_algorithm == ChecksumAlgorithm.MD5
        assert len(result.checksum) == 32  # MD5 hex length


class TestProductManagerEdgeCases:
    """Edge case tests for product manager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a product manager with temp storage."""
        storage_config = StorageConfig(
            storage_type=StorageType.LOCAL,
            root_path=str(tmp_path / "storage"),
        )
        config = ProductManagerConfig(
            storage_config=storage_config,
            db_path=str(tmp_path / "products.db"),
            default_ttl_hours=1,
        )
        mgr = ProductManager(config)
        yield mgr
        mgr.close()

    def test_register_with_empty_job_id(self, manager):
        """Test registering with empty job ID."""
        product = manager.register(
            job_id="",
            product_type=ProductType.RAW,
        )
        assert product.job_id == ""

    def test_store_empty_bytes(self, manager):
        """Test storing empty bytes."""
        product = manager.register(
            job_id="job_1",
            product_type=ProductType.RAW,
        )
        updated = manager.store_bytes(product.product_id, b"")
        assert updated.status == ProductStatus.READY
        assert updated.size_bytes == 0

    def test_cleanup_job_empty(self, manager):
        """Test cleanup on job with no products."""
        deleted = manager.cleanup_job("nonexistent_job")
        assert deleted == []

    def test_list_by_job_nonexistent(self, manager):
        """Test listing products for non-existent job."""
        products = manager.list_by_job("nonexistent_job")
        assert products == []

    def test_search_by_tags_empty_set(self, manager):
        """Test searching with empty tag set."""
        products = manager.search_by_tags(set())
        # With empty tags and match_all=True, all products match
        assert isinstance(products, list)

    def test_get_dependencies_nonexistent(self, manager):
        """Test getting dependencies of non-existent product."""
        deps = manager.get_dependencies("nonexistent")
        assert deps == []

    def test_get_dependents_nonexistent(self, manager):
        """Test getting dependents of non-existent product."""
        deps = manager.get_dependents("nonexistent")
        assert deps == []

    def test_get_uri(self, manager):
        """Test getting storage URI for product."""
        product = manager.register(
            job_id="job_1",
            product_type=ProductType.RAW,
        )
        uri = manager.get_uri(product.product_id)
        assert "file://" in uri

    def test_mark_expired_nonexistent(self, manager):
        """Test marking non-existent product as expired."""
        with pytest.raises(ValueError, match="Product not found"):
            manager.mark_expired("nonexistent")

    def test_extend_expiration_nonexistent(self, manager):
        """Test extending expiration of non-existent product."""
        with pytest.raises(ValueError, match="Product not found"):
            manager.extend_expiration("nonexistent", hours=24)


class TestLineageTrackerEdgeCases:
    """Edge case tests for lineage tracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a lineage tracker with temp database."""
        db_path = str(tmp_path / "lineage.db")
        trk = LineageTracker(db_path)
        yield trk
        trk.close()

    def test_get_nonexistent_record(self, tracker):
        """Test getting non-existent provenance record."""
        record = tracker.get_record("nonexistent")
        assert record is None

    def test_get_context_nonexistent(self, tracker):
        """Test getting non-existent tracking context."""
        ctx = tracker.get_context("nonexistent")
        assert ctx is None

    def test_export_nonexistent_record(self, tracker, tmp_path):
        """Test exporting non-existent record."""
        with pytest.raises(ValueError, match="No record found"):
            tracker.export_to_json("nonexistent", tmp_path / "out.json")

    def test_list_records_empty(self, tracker):
        """Test listing records when none exist."""
        records = tracker.list_records()
        assert records == []

    def test_get_events_nonexistent(self, tracker):
        """Test getting events for non-existent product."""
        events = tracker.get_events("nonexistent")
        assert events == []

    def test_step_with_empty_name(self, tracker):
        """Test step with empty name."""
        ctx = tracker.start_tracking("prod_1", "evt_1")
        with ctx.step("") as step:
            pass
        assert ctx._steps[0].step == ""
        ctx.finish()

    def test_add_step_directly(self, tracker):
        """Test adding step directly without context manager."""
        ctx = tracker.start_tracking("prod_1", "evt_1")
        step = ProcessingStep(
            step="direct_step",
            timestamp=datetime.now(timezone.utc),
            inputs=["in1"],
            outputs=["out1"],
        )
        ctx.add_step(step)
        assert len(ctx._steps) == 1
        ctx.finish()

    def test_quality_summary_defaults(self, tracker):
        """Test quality summary with default values."""
        ctx = tracker.start_tracking("prod_1", "evt_1")
        ctx.set_quality(QualitySummary())
        record = ctx.finish()
        assert record.quality_summary.overall_confidence == 1.0
        assert record.quality_summary.uncertainty_percent == 0.0

    def test_input_dataset_without_optional_fields(self, tracker):
        """Test input dataset with only required fields."""
        ctx = tracker.start_tracking("prod_1", "evt_1")
        ctx.add_input(InputDataset(
            dataset_id="ds_1",
            provider="test",
            data_type="test",
        ))
        record = ctx.finish()
        assert len(record.input_datasets) == 1
        assert record.input_datasets[0].acquisition_time is None

    def test_reproducibility_hash_consistency(self, tracker):
        """Test that reproducibility hash is deterministic."""
        ctx1 = tracker.start_tracking("prod_1", "evt_1")
        ctx1.add_input(InputDataset(
            dataset_id="ds_1",
            provider="test",
            data_type="test",
        ))
        record1 = ctx1.finish()

        ctx2 = tracker.start_tracking("prod_2", "evt_2")
        ctx2.add_input(InputDataset(
            dataset_id="ds_1",
            provider="test",
            data_type="test",
        ))
        record2 = ctx2.finish()

        # Same inputs should produce same selection hash
        assert record1.reproducibility["selection_hash"] == record2.reproducibility["selection_hash"]


class TestDataclassEdgeCases:
    """Edge case tests for dataclasses."""

    def test_storage_object_to_dict_with_none(self):
        """Test StorageObject to_dict with None values."""
        obj = StorageObject(
            key="test.txt",
            size_bytes=0,
            last_modified=datetime.now(timezone.utc),
            checksum=None,
            content_type=None,
        )
        d = obj.to_dict()
        assert d["checksum"] is None
        assert d["content_type"] is None

    def test_upload_result_to_dict(self):
        """Test UploadResult to_dict."""
        result = UploadResult(
            key="test.txt",
            size_bytes=100,
            checksum="abc123",
            checksum_algorithm=ChecksumAlgorithm.SHA256,
            uri="file:///test.txt",
        )
        d = result.to_dict()
        assert d["checksum_algorithm"] == "sha256"

    def test_product_info_with_all_fields(self):
        """Test ProductInfo with all fields populated."""
        product = ProductInfo(
            product_id="prod_1",
            job_id="job_1",
            product_type=ProductType.FINAL,
            status=ProductStatus.READY,
            storage_key="products/prod_1.tif",
            size_bytes=1000,
            checksum="abc123",
            content_type="image/tiff",
            format="cog",
            dependencies=["dep_1"],
            dependents=["child_1"],
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            metadata={"key": "value"},
            tags={"flood", "urgent"},
        )
        d = product.to_dict()
        assert d["product_type"] == "final"
        assert "flood" in d["tags"]

    def test_processing_step_with_error(self):
        """Test ProcessingStep with error state."""
        step = ProcessingStep(
            step="failed_step",
            timestamp=datetime.now(timezone.utc),
            inputs=["in1"],
            outputs=[],
            status="failed",
            error_message="Test error message",
        )
        d = step.to_dict()
        assert d["status"] == "failed"
        assert d["error_message"] == "Test error message"
