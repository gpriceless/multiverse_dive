"""
Reporting Agent for Final Product Generation and Delivery.

Handles generation of all final output products and their delivery to
various destinations. Coordinates with the Orchestrator agent to report
completion status.

Products Generated:
- GeoTIFF (COG) - Flood/fire/storm extent rasters
- GeoJSON - Vector outputs for web mapping
- PDF/HTML reports - QA reports with methodology and quality summary
- Provenance records - Full lineage following provenance.schema.json
- Thumbnails - PNG preview images

Delivery Options:
- Local filesystem
- S3/cloud storage
- Webhook notifications
- API response preparation
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Base Agent Classes
# ============================================================================


class AgentStatus(Enum):
    """Status of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentMessage:
    """
    Message for inter-agent communication.

    Attributes:
        sender: Agent sending the message
        recipient: Target agent
        message_type: Type of message
        payload: Message data
        correlation_id: ID to correlate request/response
        timestamp: When message was created
    """

    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.payload = payload
        self.correlation_id = correlation_id or self._generate_id()
        self.timestamp = datetime.now(timezone.utc)

    def _generate_id(self) -> str:
        """Generate unique correlation ID."""
        data = f"{self.sender}:{self.recipient}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentState:
    """
    Persistent state for an agent.

    Attributes:
        agent_id: Unique agent identifier
        status: Current status
        current_task: Current task being processed
        progress: Progress percentage (0-100)
        last_activity: Timestamp of last activity
        metadata: Additional state metadata
        error_message: Error message if failed
    """

    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    progress: float = 0.0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "current_task": self.current_task,
            "progress": self.progress,
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common functionality for lifecycle management, message passing,
    and state persistence.
    """

    def __init__(
        self,
        agent_id: str,
        message_handler: Optional[Callable[[AgentMessage], None]] = None,
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique agent identifier
            message_handler: Callback for outgoing messages
        """
        self.agent_id = agent_id
        self._message_handler = message_handler
        self._state = AgentState(agent_id=agent_id)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    def status(self) -> AgentStatus:
        """Get current status."""
        return self._state.status

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    def _update_status(self, status: AgentStatus):
        """Update agent status."""
        self._state.status = status
        self._state.last_activity = datetime.now(timezone.utc)
        logger.debug(f"Agent {self.agent_id} status: {status.value}")

    def _update_progress(self, progress: float, task: Optional[str] = None):
        """Update progress."""
        self._state.progress = max(0.0, min(100.0, progress))
        if task:
            self._state.current_task = task
        self._state.last_activity = datetime.now(timezone.utc)

    def send_message(
        self,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        """
        Send a message to another agent.

        Args:
            recipient: Target agent ID
            message_type: Type of message
            payload: Message data
            correlation_id: Optional correlation ID
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )

        if self._message_handler:
            self._message_handler(message)
        else:
            logger.warning(f"No message handler configured for {self.agent_id}")

    async def receive_message(self, message: AgentMessage):
        """
        Receive a message from another agent.

        Args:
            message: Incoming message
        """
        await self._message_queue.put(message)

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit_event(self, event: str, data: Any = None):
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    @abstractmethod
    async def start(self):
        """Start the agent."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the agent."""
        pass

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task.

        Args:
            task: Task to process

        Returns:
            Processing result
        """
        pass


# ============================================================================
# Product Configuration
# ============================================================================


class OutputFormat(Enum):
    """Supported output formats."""

    GEOTIFF = "geotiff"
    COG = "cog"
    GEOJSON = "geojson"
    PDF = "pdf"
    HTML = "html"
    PNG = "png"
    JSON = "json"
    ZARR = "zarr"


class DeliveryMethod(Enum):
    """Supported delivery methods."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class ProductConfig:
    """
    Configuration for product generation.

    Attributes:
        formats: Output formats to generate
        include_uncertainty: Include uncertainty layer
        include_thumbnail: Generate thumbnail
        include_qa_report: Generate QA report
        include_provenance: Generate provenance record
        thumbnail_size: Thumbnail dimensions
        report_format: Format for QA report
        compression: Compression for rasters
        crs: Output CRS
        resolution_m: Output resolution in meters
    """

    formats: List[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.COG, OutputFormat.GEOJSON]
    )
    include_uncertainty: bool = True
    include_thumbnail: bool = True
    include_qa_report: bool = True
    include_provenance: bool = True
    thumbnail_size: Tuple[int, int] = (512, 512)
    report_format: OutputFormat = OutputFormat.HTML
    compression: str = "deflate"
    crs: Optional[str] = None
    resolution_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formats": [f.value for f in self.formats],
            "include_uncertainty": self.include_uncertainty,
            "include_thumbnail": self.include_thumbnail,
            "include_qa_report": self.include_qa_report,
            "include_provenance": self.include_provenance,
            "thumbnail_size": self.thumbnail_size,
            "report_format": self.report_format.value,
            "compression": self.compression,
            "crs": self.crs,
            "resolution_m": self.resolution_m,
        }


@dataclass
class DeliveryConfig:
    """
    Configuration for product delivery.

    Attributes:
        methods: Delivery methods to use
        local_path: Local output directory
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix
        webhook_url: Webhook notification URL
        webhook_headers: Headers for webhook
        notify_on_complete: Send notification on completion
        notify_on_error: Send notification on error
        retry_count: Number of delivery retries
    """

    methods: List[DeliveryMethod] = field(
        default_factory=lambda: [DeliveryMethod.LOCAL]
    )
    local_path: Optional[Path] = None
    s3_bucket: Optional[str] = None
    s3_prefix: str = "products/"
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    notify_on_complete: bool = True
    notify_on_error: bool = True
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "methods": [m.value for m in self.methods],
            "local_path": str(self.local_path) if self.local_path else None,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "webhook_url": self.webhook_url,
            "notify_on_complete": self.notify_on_complete,
            "notify_on_error": self.notify_on_error,
            "retry_count": self.retry_count,
        }


# ============================================================================
# Product Data Classes
# ============================================================================


@dataclass
class Product:
    """
    A generated product.

    Attributes:
        product_id: Unique product identifier
        event_id: Associated event ID
        format: Product format
        path: Path to product file
        size_bytes: File size
        checksum: File checksum
        checksum_algorithm: Checksum algorithm
        created_at: Creation timestamp
        metadata: Product metadata
    """

    product_id: str
    event_id: str
    format: OutputFormat
    path: Path
    size_bytes: int = 0
    checksum: Optional[str] = None
    checksum_algorithm: str = "sha256"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "event_id": self.event_id,
            "format": self.format.value,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DeliveryResult:
    """
    Result of a delivery operation.

    Attributes:
        product_id: Product that was delivered
        method: Delivery method used
        destination: Delivery destination URI
        success: Whether delivery succeeded
        timestamp: When delivery occurred
        error_message: Error if delivery failed
        metadata: Additional delivery metadata
    """

    product_id: str
    method: DeliveryMethod
    destination: str
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "method": self.method.value,
            "destination": self.destination,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class CompletionReport:
    """
    Report of execution completion.

    Attributes:
        execution_id: Execution identifier
        event_id: Event that was processed
        status: Completion status
        products: Generated products
        deliveries: Delivery results
        processing_time_seconds: Total processing time
        quality_summary: Quality assessment summary
        timestamp: When processing completed
        error_message: Error if processing failed
    """

    execution_id: str
    event_id: str
    status: str  # success, partial_success, failed
    products: List[Product] = field(default_factory=list)
    deliveries: List[DeliveryResult] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    quality_summary: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "event_id": self.event_id,
            "status": self.status,
            "products": [p.to_dict() for p in self.products],
            "deliveries": [d.to_dict() for d in self.deliveries],
            "processing_time_seconds": self.processing_time_seconds,
            "quality_summary": self.quality_summary,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
        }


# ============================================================================
# Reporting Agent
# ============================================================================


class ReportingAgent(BaseAgent):
    """
    Agent for final product generation and delivery.

    Generates all output products from validated analysis results and
    delivers them to configured destinations. Reports completion status
    to the Orchestrator agent.

    Example:
        from agents.reporting import ReportingAgent, ProductConfig, DeliveryConfig

        agent = ReportingAgent("reporting_agent_001")

        # Configure products
        product_config = ProductConfig(
            formats=[OutputFormat.COG, OutputFormat.GEOJSON],
            include_uncertainty=True,
            include_qa_report=True,
        )

        # Configure delivery
        delivery_config = DeliveryConfig(
            methods=[DeliveryMethod.LOCAL, DeliveryMethod.S3],
            local_path=Path("/output"),
            s3_bucket="my-bucket",
        )

        # Generate and deliver products
        products = await agent.generate_products(
            validated_results=results,
            config=product_config,
        )

        await agent.deliver_products(
            products=products,
            config=delivery_config,
        )
    """

    def __init__(
        self,
        agent_id: str = "reporting_agent",
        message_handler: Optional[Callable[[AgentMessage], None]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize ReportingAgent.

        Args:
            agent_id: Unique agent identifier
            message_handler: Callback for outgoing messages
            output_dir: Default output directory
        """
        super().__init__(agent_id, message_handler)
        self.output_dir = output_dir or Path("/tmp/multiverse_dive/products")
        self._product_generator: Optional["ProductGenerator"] = None
        self._format_converter: Optional["FormatConverter"] = None
        self._delivery_manager: Optional["DeliveryManager"] = None

    def _ensure_components(self):
        """Ensure component instances are initialized."""
        # Lazy import to avoid circular dependencies
        from agents.reporting.products import ProductGenerator
        from agents.reporting.formats import FormatConverter
        from agents.reporting.delivery import DeliveryManager

        if self._product_generator is None:
            self._product_generator = ProductGenerator(self.output_dir)
        if self._format_converter is None:
            self._format_converter = FormatConverter()
        if self._delivery_manager is None:
            self._delivery_manager = DeliveryManager()

    async def start(self):
        """Start the reporting agent."""
        self._running = True
        self._update_status(AgentStatus.IDLE)
        self._ensure_components()
        logger.info(f"ReportingAgent {self.agent_id} started")

    async def stop(self):
        """Stop the reporting agent."""
        self._running = False
        self._update_status(AgentStatus.STOPPED)
        logger.info(f"ReportingAgent {self.agent_id} stopped")

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a reporting task.

        Args:
            task: Task containing:
                - validated_results: Analysis results
                - product_config: Product configuration
                - delivery_config: Delivery configuration
                - execution_id: Execution identifier
                - event_id: Event identifier

        Returns:
            CompletionReport as dictionary
        """
        self._update_status(AgentStatus.RUNNING)
        start_time = datetime.now(timezone.utc)

        try:
            execution_id = task.get("execution_id", self._generate_execution_id())
            event_id = task.get("event_id", "unknown")

            # Parse configurations
            product_config = self._parse_product_config(
                task.get("product_config", {})
            )
            delivery_config = self._parse_delivery_config(
                task.get("delivery_config", {})
            )

            self._update_progress(10, "Generating products")

            # Generate products
            products = await self.generate_products(
                validated_results=task.get("validated_results", {}),
                config=product_config,
                event_id=event_id,
            )

            self._update_progress(60, "Delivering products")

            # Deliver products
            deliveries = await self.deliver_products(
                products=products,
                config=delivery_config,
            )

            self._update_progress(90, "Notifying completion")

            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            # Build completion report
            success_count = sum(1 for d in deliveries if d.success)
            status = "success" if success_count == len(deliveries) else (
                "partial_success" if success_count > 0 else "failed"
            )

            report = CompletionReport(
                execution_id=execution_id,
                event_id=event_id,
                status=status,
                products=products,
                deliveries=deliveries,
                processing_time_seconds=processing_time,
                quality_summary=task.get("validated_results", {}).get("qa_report"),
            )

            # Notify orchestrator
            await self.notify_completion(execution_id, products)

            self._update_progress(100, "Complete")
            self._update_status(AgentStatus.COMPLETED)

            return report.to_dict()

        except Exception as e:
            logger.error(f"ReportingAgent error: {e}")
            self._state.error_message = str(e)
            self._update_status(AgentStatus.FAILED)

            return CompletionReport(
                execution_id=task.get("execution_id", "unknown"),
                event_id=task.get("event_id", "unknown"),
                status="failed",
                error_message=str(e),
            ).to_dict()

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        suffix = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]
        return f"exec_{timestamp}_{suffix}"

    def _parse_product_config(self, config: Dict[str, Any]) -> ProductConfig:
        """Parse product configuration from dictionary."""
        if isinstance(config, ProductConfig):
            return config

        formats = [
            OutputFormat(f) if isinstance(f, str) else f
            for f in config.get("formats", ["cog", "geojson"])
        ]

        report_format = config.get("report_format", "html")
        if isinstance(report_format, str):
            report_format = OutputFormat(report_format)

        return ProductConfig(
            formats=formats,
            include_uncertainty=config.get("include_uncertainty", True),
            include_thumbnail=config.get("include_thumbnail", True),
            include_qa_report=config.get("include_qa_report", True),
            include_provenance=config.get("include_provenance", True),
            thumbnail_size=tuple(config.get("thumbnail_size", [512, 512])),
            report_format=report_format,
            compression=config.get("compression", "deflate"),
            crs=config.get("crs"),
            resolution_m=config.get("resolution_m"),
        )

    def _parse_delivery_config(self, config: Dict[str, Any]) -> DeliveryConfig:
        """Parse delivery configuration from dictionary."""
        if isinstance(config, DeliveryConfig):
            return config

        methods = [
            DeliveryMethod(m) if isinstance(m, str) else m
            for m in config.get("methods", ["local"])
        ]

        local_path = config.get("local_path")
        if local_path and not isinstance(local_path, Path):
            local_path = Path(local_path)

        return DeliveryConfig(
            methods=methods,
            local_path=local_path or self.output_dir,
            s3_bucket=config.get("s3_bucket"),
            s3_prefix=config.get("s3_prefix", "products/"),
            webhook_url=config.get("webhook_url"),
            webhook_headers=config.get("webhook_headers", {}),
            notify_on_complete=config.get("notify_on_complete", True),
            notify_on_error=config.get("notify_on_error", True),
            retry_count=config.get("retry_count", 3),
        )

    async def generate_products(
        self,
        validated_results: Dict[str, Any],
        config: ProductConfig,
        event_id: Optional[str] = None,
    ) -> List[Product]:
        """
        Generate all output products from validated results.

        Args:
            validated_results: Analysis results with:
                - extent_data: Primary extent raster (numpy array)
                - extent_metadata: Metadata for extent
                - uncertainty_data: Uncertainty layer (optional)
                - features: Vector features (optional)
                - qa_report: QA report (optional)
                - provenance: Provenance record (optional)
            config: Product generation configuration
            event_id: Event identifier

        Returns:
            List of generated products
        """
        self._ensure_components()
        products: List[Product] = []
        event_id = event_id or validated_results.get("event_id", "unknown")

        logger.info(f"Generating products for event {event_id}")

        # Generate extent products
        if "extent_data" in validated_results:
            extent_products = await self._generate_extent_products(
                data=validated_results["extent_data"],
                metadata=validated_results.get("extent_metadata", {}),
                config=config,
                event_id=event_id,
            )
            products.extend(extent_products)

        # Generate uncertainty layer
        if config.include_uncertainty and "uncertainty_data" in validated_results:
            uncertainty_product = await self._generate_uncertainty_product(
                data=validated_results["uncertainty_data"],
                metadata=validated_results.get("uncertainty_metadata", {}),
                config=config,
                event_id=event_id,
            )
            if uncertainty_product:
                products.append(uncertainty_product)

        # Generate QA report
        if config.include_qa_report and "qa_report" in validated_results:
            report_product = await self._generate_qa_report_product(
                qa_report=validated_results["qa_report"],
                config=config,
                event_id=event_id,
            )
            if report_product:
                products.append(report_product)

        # Generate provenance record
        if config.include_provenance:
            provenance_product = await self._generate_provenance_product(
                provenance=validated_results.get("provenance", {}),
                products=products,
                event_id=event_id,
            )
            if provenance_product:
                products.append(provenance_product)

        # Generate thumbnail
        if config.include_thumbnail and "extent_data" in validated_results:
            thumbnail_product = await self._generate_thumbnail(
                data=validated_results["extent_data"],
                config=config,
                event_id=event_id,
            )
            if thumbnail_product:
                products.append(thumbnail_product)

        logger.info(f"Generated {len(products)} products for event {event_id}")
        return products

    async def _generate_extent_products(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        config: ProductConfig,
        event_id: str,
    ) -> List[Product]:
        """Generate extent products in configured formats."""
        products = []

        for fmt in config.formats:
            if fmt in (OutputFormat.GEOTIFF, OutputFormat.COG):
                product = await self.generate_geotiff(
                    data=data,
                    metadata=metadata,
                    event_id=event_id,
                    as_cog=fmt == OutputFormat.COG,
                    compression=config.compression,
                )
                if product:
                    products.append(product)

            elif fmt == OutputFormat.GEOJSON:
                # Convert raster to vector features
                features = self._raster_to_features(data, metadata)
                product = await self.generate_geojson(
                    features=features,
                    metadata=metadata,
                    event_id=event_id,
                )
                if product:
                    products.append(product)

        return products

    async def _generate_uncertainty_product(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        config: ProductConfig,
        event_id: str,
    ) -> Optional[Product]:
        """Generate uncertainty layer product."""
        return await self.generate_geotiff(
            data=data,
            metadata={**metadata, "layer_type": "uncertainty"},
            event_id=event_id,
            suffix="_uncertainty",
            as_cog=OutputFormat.COG in config.formats,
            compression=config.compression,
        )

    async def _generate_qa_report_product(
        self,
        qa_report: Dict[str, Any],
        config: ProductConfig,
        event_id: str,
    ) -> Optional[Product]:
        """Generate QA report product."""
        return await self.generate_report(
            results=qa_report,
            qa_report=qa_report,
            event_id=event_id,
            format=config.report_format,
        )

    async def _generate_provenance_product(
        self,
        provenance: Dict[str, Any],
        products: List[Product],
        event_id: str,
    ) -> Optional[Product]:
        """Generate provenance record product."""
        self._ensure_components()

        # Build complete provenance
        provenance_data = {
            "product_id": f"prod_{event_id}_provenance",
            "event_id": event_id,
            "generated_products": [p.to_dict() for p in products],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **provenance,
        }

        # Write provenance JSON
        output_path = self.output_dir / event_id / f"{event_id}_provenance.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        checksum = self._compute_checksum(output_path)

        return Product(
            product_id=f"prod_{event_id}_provenance",
            event_id=event_id,
            format=OutputFormat.JSON,
            path=output_path,
            size_bytes=output_path.stat().st_size,
            checksum=checksum,
            metadata={"type": "provenance"},
        )

    async def _generate_thumbnail(
        self,
        data: np.ndarray,
        config: ProductConfig,
        event_id: str,
    ) -> Optional[Product]:
        """Generate thumbnail preview."""
        self._ensure_components()

        try:
            thumbnail_data = self._product_generator.generate_thumbnail(
                data=data,
                size=config.thumbnail_size,
            )

            output_path = self.output_dir / event_id / f"{event_id}_thumbnail.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._product_generator.save_thumbnail(thumbnail_data, output_path)

            checksum = self._compute_checksum(output_path)

            return Product(
                product_id=f"prod_{event_id}_thumbnail",
                event_id=event_id,
                format=OutputFormat.PNG,
                path=output_path,
                size_bytes=output_path.stat().st_size,
                checksum=checksum,
                metadata={"type": "thumbnail", "size": config.thumbnail_size},
            )
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None

    def _raster_to_features(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Convert raster data to GeoJSON features."""
        self._ensure_components()
        return self._product_generator.raster_to_features(data, metadata)

    async def generate_geotiff(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        event_id: Optional[str] = None,
        suffix: str = "_extent",
        as_cog: bool = True,
        compression: str = "deflate",
    ) -> Optional[Product]:
        """
        Generate a GeoTIFF product.

        Args:
            data: Raster data array
            metadata: Metadata with transform, crs, etc.
            event_id: Event identifier
            suffix: Filename suffix
            as_cog: Generate as Cloud-Optimized GeoTIFF
            compression: Compression method

        Returns:
            Generated Product or None on error
        """
        self._ensure_components()

        event_id = event_id or "unknown"
        filename = f"{event_id}{suffix}.tif"
        output_path = self.output_dir / event_id / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = await asyncio.to_thread(
                self._product_generator.generate_geotiff,
                data=data,
                output_path=output_path,
                metadata=metadata,
                as_cog=as_cog,
                compression=compression,
            )

            checksum = self._compute_checksum(output_path)

            return Product(
                product_id=f"prod_{event_id}{suffix}",
                event_id=event_id,
                format=OutputFormat.COG if as_cog else OutputFormat.GEOTIFF,
                path=output_path,
                size_bytes=output_path.stat().st_size,
                checksum=checksum,
                metadata={
                    "type": "extent",
                    "compression": compression,
                    "cog": as_cog,
                    **result,
                },
            )
        except Exception as e:
            logger.error(f"Failed to generate GeoTIFF: {e}")
            return None

    async def generate_geojson(
        self,
        features: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> Optional[Product]:
        """
        Generate a GeoJSON product.

        Args:
            features: List of GeoJSON features
            metadata: Additional metadata
            event_id: Event identifier

        Returns:
            Generated Product or None on error
        """
        self._ensure_components()

        event_id = event_id or "unknown"
        filename = f"{event_id}_extent.geojson"
        output_path = self.output_dir / event_id / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = await asyncio.to_thread(
                self._product_generator.generate_geojson,
                features=features,
                output_path=output_path,
                metadata=metadata or {},
            )

            checksum = self._compute_checksum(output_path)

            return Product(
                product_id=f"prod_{event_id}_geojson",
                event_id=event_id,
                format=OutputFormat.GEOJSON,
                path=output_path,
                size_bytes=output_path.stat().st_size,
                checksum=checksum,
                metadata={
                    "type": "extent",
                    "feature_count": len(features),
                    **result,
                },
            )
        except Exception as e:
            logger.error(f"Failed to generate GeoJSON: {e}")
            return None

    async def generate_report(
        self,
        results: Dict[str, Any],
        qa_report: Dict[str, Any],
        event_id: Optional[str] = None,
        format: OutputFormat = OutputFormat.HTML,
    ) -> Optional[Product]:
        """
        Generate a QA report product.

        Args:
            results: Analysis results
            qa_report: QA report data
            event_id: Event identifier
            format: Output format (HTML or PDF)

        Returns:
            Generated Product or None on error
        """
        self._ensure_components()

        event_id = event_id or "unknown"
        extension = "html" if format == OutputFormat.HTML else "pdf"
        filename = f"{event_id}_qa_report.{extension}"
        output_path = self.output_dir / event_id / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(
                self._product_generator.generate_qa_report,
                qa_report=qa_report,
                results=results,
                output_path=output_path,
                format=format,
            )

            checksum = self._compute_checksum(output_path)

            return Product(
                product_id=f"prod_{event_id}_qa_report",
                event_id=event_id,
                format=format,
                path=output_path,
                size_bytes=output_path.stat().st_size,
                checksum=checksum,
                metadata={"type": "qa_report"},
            )
        except Exception as e:
            logger.error(f"Failed to generate QA report: {e}")
            return None

    async def deliver_products(
        self,
        products: List[Product],
        config: DeliveryConfig,
    ) -> List[DeliveryResult]:
        """
        Deliver products to configured destinations.

        Args:
            products: Products to deliver
            config: Delivery configuration

        Returns:
            List of delivery results
        """
        self._ensure_components()

        results: List[DeliveryResult] = []

        for product in products:
            for method in config.methods:
                try:
                    result = await self._deliver_single(
                        product=product,
                        method=method,
                        config=config,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Delivery failed for {product.product_id}: {e}")
                    results.append(
                        DeliveryResult(
                            product_id=product.product_id,
                            method=method,
                            destination="error",
                            success=False,
                            error_message=str(e),
                        )
                    )

        return results

    async def _deliver_single(
        self,
        product: Product,
        method: DeliveryMethod,
        config: DeliveryConfig,
    ) -> DeliveryResult:
        """Deliver a single product."""
        if method == DeliveryMethod.LOCAL:
            return await self._deliver_local(product, config)
        elif method == DeliveryMethod.S3:
            return await self._deliver_s3(product, config)
        elif method == DeliveryMethod.WEBHOOK:
            return await self._deliver_webhook(product, config)
        elif method == DeliveryMethod.API:
            return await self._deliver_api(product, config)
        else:
            raise ValueError(f"Unsupported delivery method: {method}")

    async def _deliver_local(
        self,
        product: Product,
        config: DeliveryConfig,
    ) -> DeliveryResult:
        """Deliver to local filesystem."""
        destination = config.local_path or self.output_dir
        dest_path = destination / product.path.name

        # Product is already at output location
        if product.path == dest_path:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.LOCAL,
                destination=str(dest_path),
                success=True,
            )

        # Copy to destination
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(product.path, dest_path)

            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.LOCAL,
                destination=str(dest_path),
                success=True,
            )
        except Exception as e:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.LOCAL,
                destination=str(dest_path),
                success=False,
                error_message=str(e),
            )

    async def _deliver_s3(
        self,
        product: Product,
        config: DeliveryConfig,
    ) -> DeliveryResult:
        """Deliver to S3."""
        if not config.s3_bucket:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.S3,
                destination="",
                success=False,
                error_message="S3 bucket not configured",
            )

        try:
            key = f"{config.s3_prefix}{product.event_id}/{product.path.name}"

            result = await asyncio.to_thread(
                self._delivery_manager.upload_to_s3,
                local_path=product.path,
                bucket=config.s3_bucket,
                key=key,
            )

            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.S3,
                destination=f"s3://{config.s3_bucket}/{key}",
                success=True,
                metadata=result,
            )
        except Exception as e:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.S3,
                destination=f"s3://{config.s3_bucket}/{config.s3_prefix}",
                success=False,
                error_message=str(e),
            )

    async def _deliver_webhook(
        self,
        product: Product,
        config: DeliveryConfig,
    ) -> DeliveryResult:
        """Deliver webhook notification."""
        if not config.webhook_url:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.WEBHOOK,
                destination="",
                success=False,
                error_message="Webhook URL not configured",
            )

        try:
            result = await asyncio.to_thread(
                self._delivery_manager.send_webhook,
                url=config.webhook_url,
                payload=product.to_dict(),
                headers=config.webhook_headers,
            )

            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.WEBHOOK,
                destination=config.webhook_url,
                success=True,
                metadata=result,
            )
        except Exception as e:
            return DeliveryResult(
                product_id=product.product_id,
                method=DeliveryMethod.WEBHOOK,
                destination=config.webhook_url,
                success=False,
                error_message=str(e),
            )

    async def _deliver_api(
        self,
        product: Product,
        config: DeliveryConfig,
    ) -> DeliveryResult:
        """Prepare API response."""
        # For API delivery, we just prepare the response data
        return DeliveryResult(
            product_id=product.product_id,
            method=DeliveryMethod.API,
            destination="api_response",
            success=True,
            metadata={"product": product.to_dict()},
        )

    async def notify_completion(
        self,
        execution_id: str,
        products: List[Product],
    ):
        """
        Notify orchestrator of completion.

        Args:
            execution_id: Execution identifier
            products: Generated products
        """
        self.send_message(
            recipient="orchestrator",
            message_type="execution_complete",
            payload={
                "execution_id": execution_id,
                "status": "success",
                "products": [p.to_dict() for p in products],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        self._emit_event("completion", {
            "execution_id": execution_id,
            "product_count": len(products),
        })

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
