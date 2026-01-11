"""
Reporting Agent for Final Product Generation and Delivery.

This module provides the ReportingAgent and supporting classes for generating
and delivering final output products from the multiverse_dive analysis pipeline.

Components:
- ReportingAgent: Main agent for coordinating product generation and delivery
- ProductGenerator: Generates GeoTIFF/COG, GeoJSON, reports, thumbnails
- FormatConverter: Handles CRS transformation, resampling, compression
- DeliveryManager: Manages delivery to local, S3, GCS, webhooks

Output Products:
- Flood/Fire/Storm extent - GeoTIFF (COG) + GeoJSON
- Uncertainty layer - GeoTIFF showing per-pixel confidence
- QA Report - PDF or HTML with methodology, data sources, quality summary
- Provenance record - JSON following provenance.schema.json
- Thumbnail - PNG preview image

Example:
    from agents.reporting import (
        ReportingAgent,
        ProductConfig,
        DeliveryConfig,
        OutputFormat,
        DeliveryMethod,
    )

    # Create agent
    agent = ReportingAgent("reporting_agent_001")
    await agent.start()

    # Configure products
    config = ProductConfig(
        formats=[OutputFormat.COG, OutputFormat.GEOJSON],
        include_uncertainty=True,
        include_qa_report=True,
        include_provenance=True,
        include_thumbnail=True,
    )

    # Generate products
    products = await agent.generate_products(
        validated_results=analysis_results,
        config=config,
        event_id="evt_flood_miami_2024",
    )

    # Deliver products
    delivery_config = DeliveryConfig(
        methods=[DeliveryMethod.LOCAL, DeliveryMethod.S3],
        local_path=Path("/output"),
        s3_bucket="products-bucket",
    )

    deliveries = await agent.deliver_products(
        products=products,
        config=delivery_config,
    )

    # Notify completion
    await agent.notify_completion(execution_id, products)
"""

# Main agent
from agents.reporting.main import (
    ReportingAgent,
    BaseAgent,
    AgentStatus,
    AgentMessage,
    AgentState,
    ProductConfig,
    DeliveryConfig,
    Product,
    DeliveryResult,
    CompletionReport,
    OutputFormat,
    DeliveryMethod,
)

# Product generation
from agents.reporting.products import (
    ProductGenerator,
    GeoTIFFResult,
    GeoJSONResult,
    COGConfig,
    ThumbnailColormap,
)

# Format conversion
from agents.reporting.formats import (
    FormatConverter,
    FormatConfig,
    ConversionResult,
    ValidationResult,
    ResamplingMethod,
    CompressionMethod,
    OutputCRS,
)

# Delivery management
from agents.reporting.delivery import (
    DeliveryManager,
    DeliveryTask,
    DeliveryReceipt,
    DeliveryTracker,
    DeliveryStatus,
    DeliveryPriority,
    WebhookConfig,
    S3Config,
)

__all__ = [
    # Main agent
    "ReportingAgent",
    "BaseAgent",
    "AgentStatus",
    "AgentMessage",
    "AgentState",
    "ProductConfig",
    "DeliveryConfig",
    "Product",
    "DeliveryResult",
    "CompletionReport",
    "OutputFormat",
    "DeliveryMethod",
    # Product generation
    "ProductGenerator",
    "GeoTIFFResult",
    "GeoJSONResult",
    "COGConfig",
    "ThumbnailColormap",
    # Format conversion
    "FormatConverter",
    "FormatConfig",
    "ConversionResult",
    "ValidationResult",
    "ResamplingMethod",
    "CompressionMethod",
    "OutputCRS",
    # Delivery management
    "DeliveryManager",
    "DeliveryTask",
    "DeliveryReceipt",
    "DeliveryTracker",
    "DeliveryStatus",
    "DeliveryPriority",
    "WebhookConfig",
    "S3Config",
]
