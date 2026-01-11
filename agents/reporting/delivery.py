"""
Delivery Management for Reporting Agent.

Handles distribution of generated products to various destinations:
- Local filesystem delivery
- S3/cloud storage upload
- Webhook notifications
- Email notifications (optional)
- API response preparation
- Delivery confirmation tracking

Uses existing infrastructure from:
- core/data/ingestion/persistence/storage.py for storage backends
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DeliveryStatus(Enum):
    """Status of a delivery operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class DeliveryPriority(Enum):
    """Priority level for delivery."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DeliveryTask:
    """
    A delivery task to be processed.

    Attributes:
        task_id: Unique task identifier
        product_path: Path to product file
        product_id: Product identifier
        destination: Delivery destination URI
        method: Delivery method
        priority: Delivery priority
        retry_count: Number of retries attempted
        max_retries: Maximum retries allowed
        status: Current status
        created_at: When task was created
        completed_at: When task completed
        error_message: Error if failed
        metadata: Additional metadata
    """

    task_id: str
    product_path: Path
    product_id: str
    destination: str
    method: str
    priority: DeliveryPriority = DeliveryPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    status: DeliveryStatus = DeliveryStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "product_path": str(self.product_path),
            "product_id": self.product_id,
            "destination": self.destination,
            "method": self.method,
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class DeliveryReceipt:
    """
    Receipt confirming delivery completion.

    Attributes:
        task_id: Task that was delivered
        product_id: Product that was delivered
        destination: Where it was delivered
        method: Delivery method used
        status: Final status
        checksum: Checksum of delivered file
        size_bytes: Size of delivered file
        timestamp: When delivery completed
        response: Response from delivery target
        metadata: Additional metadata
    """

    task_id: str
    product_id: str
    destination: str
    method: str
    status: DeliveryStatus
    checksum: Optional[str] = None
    size_bytes: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "product_id": self.product_id,
            "destination": self.destination,
            "method": self.method,
            "status": self.status.value,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp.isoformat(),
            "response": self.response,
            "metadata": self.metadata,
        }


@dataclass
class WebhookConfig:
    """
    Configuration for webhook delivery.

    Attributes:
        url: Webhook URL
        headers: HTTP headers
        timeout: Request timeout in seconds
        retry_count: Number of retries
        retry_delay: Delay between retries
        verify_ssl: Verify SSL certificates
        include_product_data: Include product data in payload
    """

    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    include_product_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "headers": {k: "***" for k in self.headers},  # Mask headers
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "verify_ssl": self.verify_ssl,
            "include_product_data": self.include_product_data,
        }


@dataclass
class S3Config:
    """
    Configuration for S3 delivery.

    Attributes:
        bucket: S3 bucket name
        prefix: Key prefix
        region: AWS region
        endpoint_url: Custom endpoint URL (for S3-compatible)
        access_key_id: AWS access key (optional, use env vars)
        secret_access_key: AWS secret key (optional, use env vars)
        storage_class: S3 storage class
        acl: Access control list
        server_side_encryption: SSE algorithm
        metadata: Additional metadata to add
    """

    bucket: str
    prefix: str = ""
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    storage_class: str = "STANDARD"
    acl: str = "private"
    server_side_encryption: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without credentials)."""
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "storage_class": self.storage_class,
            "acl": self.acl,
            "server_side_encryption": self.server_side_encryption,
        }


class DeliveryManager:
    """
    Manager for product delivery operations.

    Handles delivery to various destinations with retry logic,
    confirmation tracking, and notification support.

    Example:
        manager = DeliveryManager()

        # Upload to S3
        result = manager.upload_to_s3(
            local_path=Path("product.tif"),
            bucket="my-bucket",
            key="products/product.tif",
        )

        # Send webhook notification
        result = manager.send_webhook(
            url="https://example.com/webhook",
            payload={"product_id": "prod_001", "status": "ready"},
        )

        # Deliver with retry
        receipt = manager.deliver(
            product_path=Path("product.tif"),
            destination="s3://my-bucket/products/",
            method="s3",
            retry_count=3,
        )
    """

    def __init__(
        self,
        default_retry_count: int = 3,
        default_timeout: int = 300,
    ):
        """
        Initialize DeliveryManager.

        Args:
            default_retry_count: Default number of retries
            default_timeout: Default timeout in seconds
        """
        self.default_retry_count = default_retry_count
        self.default_timeout = default_timeout
        self._receipts: Dict[str, DeliveryReceipt] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    def deliver(
        self,
        product_path: Path,
        destination: str,
        method: str,
        product_id: Optional[str] = None,
        retry_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliveryReceipt:
        """
        Deliver a product to a destination.

        Args:
            product_path: Path to product file
            destination: Destination URI
            method: Delivery method (local, s3, webhook, etc.)
            product_id: Optional product identifier
            retry_count: Number of retries
            metadata: Additional metadata

        Returns:
            DeliveryReceipt with delivery details
        """
        task_id = self._generate_task_id(product_path, destination)
        product_id = product_id or product_path.stem
        retry_count = retry_count or self.default_retry_count

        task = DeliveryTask(
            task_id=task_id,
            product_path=product_path,
            product_id=product_id,
            destination=destination,
            method=method,
            max_retries=retry_count,
            metadata=metadata or {},
        )

        # Execute delivery with retries
        for attempt in range(retry_count + 1):
            task.retry_count = attempt
            task.status = DeliveryStatus.IN_PROGRESS

            try:
                receipt = self._execute_delivery(task)
                if receipt.status == DeliveryStatus.SUCCESS:
                    self._receipts[task_id] = receipt
                    self._emit_event("delivery_success", receipt)
                    return receipt
            except Exception as e:
                logger.warning(f"Delivery attempt {attempt + 1} failed: {e}")
                task.error_message = str(e)

                if attempt < retry_count:
                    task.status = DeliveryStatus.RETRYING
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    task.status = DeliveryStatus.FAILED
                    task.completed_at = datetime.now(timezone.utc)

        # All retries failed
        receipt = DeliveryReceipt(
            task_id=task_id,
            product_id=product_id,
            destination=destination,
            method=method,
            status=DeliveryStatus.FAILED,
            metadata={"error": task.error_message, "attempts": retry_count + 1},
        )
        self._receipts[task_id] = receipt
        self._emit_event("delivery_failed", receipt)
        return receipt

    def _generate_task_id(self, product_path: Path, destination: str) -> str:
        """Generate unique task ID."""
        data = f"{product_path}:{destination}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _execute_delivery(self, task: DeliveryTask) -> DeliveryReceipt:
        """Execute a delivery task."""
        method = task.method.lower()

        if method == "local":
            return self._deliver_local(task)
        elif method == "s3":
            return self._deliver_s3(task)
        elif method == "gcs":
            return self._deliver_gcs(task)
        elif method == "webhook":
            return self._deliver_webhook(task)
        elif method == "api":
            return self._deliver_api(task)
        else:
            raise ValueError(f"Unknown delivery method: {method}")

    def _deliver_local(self, task: DeliveryTask) -> DeliveryReceipt:
        """Deliver to local filesystem."""
        dest_path = Path(task.destination)

        # Handle directory destination
        if dest_path.is_dir() or not dest_path.suffix:
            dest_path = dest_path / task.product_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(task.product_path, dest_path)

        # Calculate checksum
        checksum = self._compute_checksum(dest_path)

        return DeliveryReceipt(
            task_id=task.task_id,
            product_id=task.product_id,
            destination=str(dest_path),
            method="local",
            status=DeliveryStatus.SUCCESS,
            checksum=checksum,
            size_bytes=dest_path.stat().st_size,
        )

    def _deliver_s3(self, task: DeliveryTask) -> DeliveryReceipt:
        """Deliver to S3."""
        # Parse destination
        if task.destination.startswith("s3://"):
            parts = task.destination[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 destination: {task.destination}")

        # Add filename if key is a prefix
        if not key or key.endswith("/"):
            key = f"{key}{task.product_path.name}"

        result = self.upload_to_s3(
            local_path=task.product_path,
            bucket=bucket,
            key=key,
        )

        return DeliveryReceipt(
            task_id=task.task_id,
            product_id=task.product_id,
            destination=f"s3://{bucket}/{key}",
            method="s3",
            status=DeliveryStatus.SUCCESS,
            checksum=result.get("checksum"),
            size_bytes=result.get("size_bytes", 0),
            response=result,
        )

    def _deliver_gcs(self, task: DeliveryTask) -> DeliveryReceipt:
        """Deliver to Google Cloud Storage."""
        # Parse destination
        if task.destination.startswith("gs://"):
            parts = task.destination[5:].split("/", 1)
            bucket = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid GCS destination: {task.destination}")

        if not blob_name or blob_name.endswith("/"):
            blob_name = f"{blob_name}{task.product_path.name}"

        result = self.upload_to_gcs(
            local_path=task.product_path,
            bucket=bucket,
            blob_name=blob_name,
        )

        return DeliveryReceipt(
            task_id=task.task_id,
            product_id=task.product_id,
            destination=f"gs://{bucket}/{blob_name}",
            method="gcs",
            status=DeliveryStatus.SUCCESS,
            checksum=result.get("checksum"),
            size_bytes=result.get("size_bytes", 0),
            response=result,
        )

    def _deliver_webhook(self, task: DeliveryTask) -> DeliveryReceipt:
        """Deliver webhook notification."""
        payload = {
            "product_id": task.product_id,
            "product_path": str(task.product_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": task.metadata,
        }

        result = self.send_webhook(
            url=task.destination,
            payload=payload,
        )

        return DeliveryReceipt(
            task_id=task.task_id,
            product_id=task.product_id,
            destination=task.destination,
            method="webhook",
            status=DeliveryStatus.SUCCESS if result.get("success") else DeliveryStatus.FAILED,
            response=result,
        )

    def _deliver_api(self, task: DeliveryTask) -> DeliveryReceipt:
        """Prepare API response."""
        # For API delivery, we just prepare the response data
        response_data = {
            "product_id": task.product_id,
            "path": str(task.product_path),
            "size_bytes": task.product_path.stat().st_size,
            "checksum": self._compute_checksum(task.product_path),
            "metadata": task.metadata,
        }

        return DeliveryReceipt(
            task_id=task.task_id,
            product_id=task.product_id,
            destination="api_response",
            method="api",
            status=DeliveryStatus.SUCCESS,
            size_bytes=task.product_path.stat().st_size,
            response=response_data,
        )

    def upload_to_s3(
        self,
        local_path: Path,
        bucket: str,
        key: str,
        config: Optional[S3Config] = None,
    ) -> Dict[str, Any]:
        """
        Upload file to S3.

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: S3 object key
            config: S3 configuration

        Returns:
            Upload result dictionary
        """
        config = config or S3Config(bucket=bucket)

        try:
            import boto3
            from botocore.exceptions import ClientError

            # Create client
            client_kwargs = {}
            if config.region:
                client_kwargs["region_name"] = config.region
            if config.endpoint_url:
                client_kwargs["endpoint_url"] = config.endpoint_url
            if config.access_key_id and config.secret_access_key:
                client_kwargs["aws_access_key_id"] = config.access_key_id
                client_kwargs["aws_secret_access_key"] = config.secret_access_key

            s3_client = boto3.client("s3", **client_kwargs)

            # Prepare upload kwargs
            extra_args = {}
            if config.storage_class:
                extra_args["StorageClass"] = config.storage_class
            if config.acl:
                extra_args["ACL"] = config.acl
            if config.server_side_encryption:
                extra_args["ServerSideEncryption"] = config.server_side_encryption
            if config.metadata:
                extra_args["Metadata"] = config.metadata

            # Determine content type
            content_type = self._guess_content_type(local_path)
            if content_type:
                extra_args["ContentType"] = content_type

            # Upload
            s3_client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )

            # Get ETag
            response = s3_client.head_object(Bucket=bucket, Key=key)
            etag = response.get("ETag", "").strip('"')

            logger.info(f"Uploaded to s3://{bucket}/{key}")

            return {
                "success": True,
                "bucket": bucket,
                "key": key,
                "etag": etag,
                "size_bytes": local_path.stat().st_size,
                "checksum": self._compute_checksum(local_path),
                "uri": f"s3://{bucket}/{key}",
            }

        except ImportError:
            raise ImportError("boto3 is required for S3 uploads")
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise

    def upload_to_gcs(
        self,
        local_path: Path,
        bucket: str,
        blob_name: str,
    ) -> Dict[str, Any]:
        """
        Upload file to Google Cloud Storage.

        Args:
            local_path: Local file path
            bucket: GCS bucket name
            blob_name: GCS blob name

        Returns:
            Upload result dictionary
        """
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket_obj = client.bucket(bucket)
            blob = bucket_obj.blob(blob_name)

            # Upload
            blob.upload_from_filename(str(local_path))

            logger.info(f"Uploaded to gs://{bucket}/{blob_name}")

            return {
                "success": True,
                "bucket": bucket,
                "blob_name": blob_name,
                "size_bytes": local_path.stat().st_size,
                "checksum": self._compute_checksum(local_path),
                "uri": f"gs://{bucket}/{blob_name}",
            }

        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS uploads")
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            raise

    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Send webhook notification.

        Args:
            url: Webhook URL
            payload: JSON payload
            headers: HTTP headers
            timeout: Request timeout

        Returns:
            Response result dictionary
        """
        try:
            import urllib.request
            import urllib.error

            headers = headers or {}
            headers.setdefault("Content-Type", "application/json")

            data = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                status_code = response.status

                logger.info(f"Webhook sent to {url}: {status_code}")

                return {
                    "success": 200 <= status_code < 300,
                    "status_code": status_code,
                    "response": response_body,
                }

        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "status_code": e.code,
                "error": str(e),
            }
        except urllib.error.URLError as e:
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[Path]] = None,
        smtp_host: str = "localhost",
        smtp_port: int = 25,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_addr: str = "noreply@multiverse-dive.local",
    ) -> Dict[str, Any]:
        """
        Send email notification.

        Args:
            to: Recipient email addresses
            subject: Email subject
            body: Email body
            attachments: Optional file attachments
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_addr: Sender address

        Returns:
            Send result dictionary
        """
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        try:
            msg = MIMEMultipart()
            msg["From"] = from_addr
            msg["To"] = ", ".join(to)
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    with open(attachment_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={attachment_path.name}",
                        )
                        msg.attach(part)

            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.sendmail(from_addr, to, msg.as_string())

            logger.info(f"Email sent to {to}")

            return {
                "success": True,
                "recipients": to,
                "subject": subject,
            }

        except Exception as e:
            logger.error(f"Email failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_receipt(self, task_id: str) -> Optional[DeliveryReceipt]:
        """
        Get delivery receipt by task ID.

        Args:
            task_id: Task identifier

        Returns:
            DeliveryReceipt if found
        """
        return self._receipts.get(task_id)

    def list_receipts(
        self,
        status: Optional[DeliveryStatus] = None,
        method: Optional[str] = None,
        limit: int = 100,
    ) -> List[DeliveryReceipt]:
        """
        List delivery receipts.

        Args:
            status: Filter by status
            method: Filter by method
            limit: Maximum receipts to return

        Returns:
            List of DeliveryReceipts
        """
        receipts = list(self._receipts.values())

        if status:
            receipts = [r for r in receipts if r.status == status]
        if method:
            receipts = [r for r in receipts if r.method == method]

        # Sort by timestamp descending
        receipts.sort(key=lambda r: r.timestamp, reverse=True)

        return receipts[:limit]

    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for delivery events.

        Args:
            event: Event name (delivery_success, delivery_failed)
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit_event(self, event: str, data: Any):
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _guess_content_type(self, path: Path) -> Optional[str]:
        """Guess content type from file extension."""
        ext = path.suffix.lower()
        content_types = {
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".geojson": "application/geo+json",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".pdf": "application/pdf",
            ".html": "text/html",
            ".xml": "application/xml",
            ".zip": "application/zip",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
        }
        return content_types.get(ext)

    def verify_delivery(
        self,
        receipt: DeliveryReceipt,
    ) -> Tuple[bool, str]:
        """
        Verify a delivery was successful.

        Args:
            receipt: Delivery receipt to verify

        Returns:
            Tuple of (success, message)
        """
        if receipt.status != DeliveryStatus.SUCCESS:
            return False, f"Delivery status is {receipt.status.value}"

        if receipt.method == "local":
            return self._verify_local(receipt)
        elif receipt.method == "s3":
            return self._verify_s3(receipt)
        elif receipt.method == "gcs":
            return self._verify_gcs(receipt)
        else:
            return True, "Verification not supported for this method"

    def _verify_local(self, receipt: DeliveryReceipt) -> Tuple[bool, str]:
        """Verify local delivery."""
        path = Path(receipt.destination)

        if not path.exists():
            return False, f"File does not exist: {path}"

        if receipt.checksum:
            actual_checksum = self._compute_checksum(path)
            if actual_checksum != receipt.checksum:
                return False, f"Checksum mismatch: expected {receipt.checksum}, got {actual_checksum}"

        if receipt.size_bytes > 0:
            actual_size = path.stat().st_size
            if actual_size != receipt.size_bytes:
                return False, f"Size mismatch: expected {receipt.size_bytes}, got {actual_size}"

        return True, "Verification successful"

    def _verify_s3(self, receipt: DeliveryReceipt) -> Tuple[bool, str]:
        """Verify S3 delivery."""
        try:
            import boto3

            # Parse destination
            if receipt.destination.startswith("s3://"):
                parts = receipt.destination[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""
            else:
                return False, f"Invalid S3 destination: {receipt.destination}"

            s3_client = boto3.client("s3")
            response = s3_client.head_object(Bucket=bucket, Key=key)

            # Verify size
            if receipt.size_bytes > 0:
                actual_size = response.get("ContentLength", 0)
                if actual_size != receipt.size_bytes:
                    return False, f"Size mismatch: expected {receipt.size_bytes}, got {actual_size}"

            return True, "S3 verification successful"

        except ImportError:
            return True, "boto3 not available for verification"
        except Exception as e:
            return False, f"S3 verification failed: {e}"

    def _verify_gcs(self, receipt: DeliveryReceipt) -> Tuple[bool, str]:
        """Verify GCS delivery."""
        try:
            from google.cloud import storage

            # Parse destination
            if receipt.destination.startswith("gs://"):
                parts = receipt.destination[5:].split("/", 1)
                bucket = parts[0]
                blob_name = parts[1] if len(parts) > 1 else ""
            else:
                return False, f"Invalid GCS destination: {receipt.destination}"

            client = storage.Client()
            bucket_obj = client.bucket(bucket)
            blob = bucket_obj.blob(blob_name)

            if not blob.exists():
                return False, f"Blob does not exist: gs://{bucket}/{blob_name}"

            return True, "GCS verification successful"

        except ImportError:
            return True, "google-cloud-storage not available for verification"
        except Exception as e:
            return False, f"GCS verification failed: {e}"


class DeliveryTracker:
    """
    Tracks delivery progress and history.

    Provides methods for querying delivery status and generating
    delivery reports.
    """

    def __init__(self):
        """Initialize DeliveryTracker."""
        self._deliveries: Dict[str, List[DeliveryReceipt]] = {}

    def track(self, product_id: str, receipt: DeliveryReceipt):
        """
        Track a delivery.

        Args:
            product_id: Product identifier
            receipt: Delivery receipt
        """
        if product_id not in self._deliveries:
            self._deliveries[product_id] = []
        self._deliveries[product_id].append(receipt)

    def get_history(self, product_id: str) -> List[DeliveryReceipt]:
        """
        Get delivery history for a product.

        Args:
            product_id: Product identifier

        Returns:
            List of delivery receipts
        """
        return self._deliveries.get(product_id, [])

    def get_summary(self) -> Dict[str, Any]:
        """
        Get delivery summary.

        Returns:
            Summary dictionary
        """
        total = 0
        success = 0
        failed = 0
        by_method: Dict[str, int] = {}

        for receipts in self._deliveries.values():
            for receipt in receipts:
                total += 1
                if receipt.status == DeliveryStatus.SUCCESS:
                    success += 1
                elif receipt.status == DeliveryStatus.FAILED:
                    failed += 1

                method = receipt.method
                by_method[method] = by_method.get(method, 0) + 1

        return {
            "total_deliveries": total,
            "successful": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0.0,
            "by_method": by_method,
            "products_delivered": len(self._deliveries),
        }
