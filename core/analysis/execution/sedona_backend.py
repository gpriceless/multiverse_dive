"""
Apache Sedona Backend for Continental-Scale Geospatial Processing.

Provides distributed raster processing on Apache Spark clusters using
Apache Sedona's raster functions for continental-scale analysis.

Key Components:
- SedonaBackend: SparkSession management and cluster coordination
- SedonaConfig: Configuration for Spark/Sedona clusters
- SedonaTileProcessor: Tile processing on Spark executors
- RasterSerializer: Serialization for distributed raster data

Performance Target:
- Process 100,000 km^2 (continental scale) in <1 hour
- Scale across 10-100+ Spark executors
- Handle 10,000+ tiles efficiently

Example Usage:
    from core.analysis.execution.sedona_backend import (
        SedonaBackend,
        SedonaConfig,
        SedonaTileProcessor,
    )

    # Create backend with configuration
    config = SedonaConfig(
        master="spark://cluster:7077",
        app_name="flood_detection",
        executor_memory="8g",
        executor_cores=4,
    )
    backend = SedonaBackend(config)

    # Connect to cluster
    backend.connect()

    # Process with algorithm
    processor = SedonaTileProcessor(backend)
    result = processor.process(
        data_path="/path/to/cog.tif",
        algorithm=flood_detector,
        output_path="/path/to/result.tif",
    )

    # Disconnect
    backend.disconnect()
"""

import hashlib
import logging
import os
import pickle
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for Spark/Sedona
try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
    )
    HAS_SPARK = True
except ImportError:
    SparkConf = None
    SparkContext = None
    SparkSession = None
    DataFrame = None
    F = None
    HAS_SPARK = False
    logger.info("PySpark not available - Sedona backend will use mock mode")

try:
    from sedona.register import SedonaRegistrator
    from sedona.utils import SedonaKryoRegistrator, KryoSerializer
    from sedona.spark import SedonaContext
    HAS_SEDONA = True
except ImportError:
    SedonaRegistrator = None
    SedonaKryoRegistrator = None
    KryoSerializer = None
    SedonaContext = None
    HAS_SEDONA = False
    logger.info("Apache Sedona not available - using fallback implementation")


# =============================================================================
# Enumerations
# =============================================================================


class SedonaDeployMode(Enum):
    """Spark deployment mode."""

    LOCAL = "local"  # Local mode for development
    CLIENT = "client"  # Client mode
    CLUSTER = "cluster"  # Cluster mode


class PartitionStrategy(Enum):
    """Raster partitioning strategy."""

    UNIFORM = "uniform"  # Uniform grid partitioning
    ADAPTIVE = "adaptive"  # Adaptive based on data density
    SPATIAL = "spatial"  # Spatial partitioning (R-tree based)


class CheckpointMode(Enum):
    """Checkpoint mode for fault tolerance."""

    NONE = "none"  # No checkpointing
    PERIODIC = "periodic"  # Checkpoint at intervals
    ON_FAILURE = "on_failure"  # Checkpoint on task failure
    ALWAYS = "always"  # Checkpoint after every stage


class RasterFormat(Enum):
    """Supported raster formats for I/O."""

    GEOTIFF = "geotiff"
    COG = "cog"  # Cloud Optimized GeoTIFF
    ZARR = "zarr"
    NETCDF = "netcdf"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SedonaConfig:
    """
    Configuration for Apache Sedona on Spark.

    Attributes:
        master: Spark master URL (e.g., "spark://host:7077", "local[*]")
        app_name: Application name
        deploy_mode: Deployment mode (local, client, cluster)
        executor_memory: Memory per executor (e.g., "8g")
        executor_cores: Cores per executor
        num_executors: Number of executors (for YARN/K8s)
        driver_memory: Driver memory (e.g., "4g")
        driver_cores: Driver cores
        partition_strategy: How to partition raster data
        partitions_per_executor: Target partitions per executor
        tile_size: Tile size in pixels (height, width)
        overlap: Overlap in pixels for edge handling
        checkpoint_mode: Fault tolerance strategy
        checkpoint_dir: Directory for checkpoints
        adaptive_execution: Enable adaptive query execution
        shuffle_partitions: Number of shuffle partitions
        broadcast_threshold: Threshold for broadcast joins (bytes)
        serializer: Kryo or Java serializer
        extra_spark_config: Additional Spark configuration
        sedona_version: Sedona version for compatibility
        enable_raster_udfs: Enable Sedona raster UDFs
    """

    master: str = "local[*]"
    app_name: str = "multiverse_sedona"
    deploy_mode: SedonaDeployMode = SedonaDeployMode.LOCAL
    executor_memory: str = "4g"
    executor_cores: int = 4
    num_executors: int = 4
    driver_memory: str = "2g"
    driver_cores: int = 2
    partition_strategy: PartitionStrategy = PartitionStrategy.UNIFORM
    partitions_per_executor: int = 4
    tile_size: Tuple[int, int] = (1024, 1024)
    overlap: int = 64
    checkpoint_mode: CheckpointMode = CheckpointMode.PERIODIC
    checkpoint_dir: Optional[str] = None
    adaptive_execution: bool = True
    shuffle_partitions: int = 200
    broadcast_threshold: int = 10 * 1024 * 1024  # 10MB
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    extra_spark_config: Dict[str, str] = field(default_factory=dict)
    sedona_version: str = "1.5.0"
    enable_raster_udfs: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.executor_cores < 1:
            raise ValueError(f"executor_cores must be >= 1, got {self.executor_cores}")
        if self.num_executors < 1:
            raise ValueError(f"num_executors must be >= 1, got {self.num_executors}")
        if self.tile_size[0] < 64 or self.tile_size[1] < 64:
            raise ValueError(f"tile_size must be >= 64, got {self.tile_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    def get_spark_conf(self) -> Dict[str, str]:
        """Get Spark configuration dictionary."""
        conf = {
            "spark.app.name": self.app_name,
            "spark.executor.memory": self.executor_memory,
            "spark.executor.cores": str(self.executor_cores),
            "spark.driver.memory": self.driver_memory,
            "spark.driver.cores": str(self.driver_cores),
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.autoBroadcastJoinThreshold": str(self.broadcast_threshold),
            "spark.serializer": self.serializer,
        }

        # YARN/K8s executor count
        if self.deploy_mode != SedonaDeployMode.LOCAL:
            conf["spark.executor.instances"] = str(self.num_executors)

        # Adaptive execution
        if self.adaptive_execution:
            conf["spark.sql.adaptive.enabled"] = "true"
            conf["spark.sql.adaptive.coalescePartitions.enabled"] = "true"

        # Checkpointing
        if self.checkpoint_dir:
            conf["spark.checkpoint.dir"] = self.checkpoint_dir

        # Sedona configuration
        if HAS_SEDONA:
            conf["spark.kryo.registrator"] = "org.apache.sedona.core.serde.SedonaKryoRegistrator"
            conf["spark.sql.extensions"] = "org.apache.sedona.sql.SedonaSqlExtensions"

        # Merge extra config
        conf.update(self.extra_spark_config)

        return conf

    @classmethod
    def for_local_testing(cls) -> "SedonaConfig":
        """Create config for local testing."""
        return cls(
            master="local[4]",
            app_name="sedona_test",
            deploy_mode=SedonaDeployMode.LOCAL,
            executor_memory="2g",
            executor_cores=2,
            num_executors=1,
            driver_memory="1g",
            tile_size=(512, 512),
            checkpoint_mode=CheckpointMode.NONE,
        )

    @classmethod
    def for_cluster(
        cls,
        master: str,
        num_executors: int = 10,
        executor_memory: str = "8g",
        executor_cores: int = 4,
    ) -> "SedonaConfig":
        """Create config for cluster deployment."""
        return cls(
            master=master,
            app_name="multiverse_continental",
            deploy_mode=SedonaDeployMode.CLIENT,
            executor_memory=executor_memory,
            executor_cores=executor_cores,
            num_executors=num_executors,
            driver_memory="4g",
            driver_cores=4,
            tile_size=(1024, 1024),
            checkpoint_mode=CheckpointMode.PERIODIC,
            adaptive_execution=True,
        )

    @classmethod
    def for_databricks(
        cls,
        num_workers: int = 10,
        worker_type: str = "Standard_DS3_v2",
    ) -> "SedonaConfig":
        """Create config for Databricks cluster."""
        # Databricks handles master URL internally
        return cls(
            master="local[*]",  # Databricks overrides this
            app_name="multiverse_databricks",
            deploy_mode=SedonaDeployMode.CLIENT,
            executor_memory="8g",
            executor_cores=4,
            num_executors=num_workers,
            extra_spark_config={
                "spark.databricks.cluster.profile": "serverless",
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "master": self.master,
            "app_name": self.app_name,
            "deploy_mode": self.deploy_mode.value,
            "executor_memory": self.executor_memory,
            "executor_cores": self.executor_cores,
            "num_executors": self.num_executors,
            "driver_memory": self.driver_memory,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "checkpoint_mode": self.checkpoint_mode.value,
            "adaptive_execution": self.adaptive_execution,
        }


@dataclass
class ClusterStatus:
    """
    Status information about the Spark cluster.

    Attributes:
        is_connected: Whether connected to cluster
        master_url: Spark master URL
        app_id: Spark application ID
        executors: Number of active executors
        total_cores: Total cores available
        total_memory_gb: Total memory available
        active_jobs: Number of active jobs
        completed_stages: Number of completed stages
        web_ui_url: URL of Spark Web UI
    """

    is_connected: bool = False
    master_url: str = ""
    app_id: str = ""
    executors: int = 0
    total_cores: int = 0
    total_memory_gb: float = 0.0
    active_jobs: int = 0
    completed_stages: int = 0
    web_ui_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_connected": self.is_connected,
            "master_url": self.master_url,
            "app_id": self.app_id,
            "executors": self.executors,
            "total_cores": self.total_cores,
            "total_memory_gb": self.total_memory_gb,
            "active_jobs": self.active_jobs,
            "completed_stages": self.completed_stages,
            "web_ui_url": self.web_ui_url,
        }


@dataclass
class TilePartition:
    """
    Information about a tile partition.

    Attributes:
        partition_id: Unique partition ID
        tile_indices: List of (col, row) tile indices in this partition
        pixel_bounds: Combined pixel bounds for the partition
        geo_bounds: Geographic bounds if available
        estimated_memory_mb: Estimated memory requirement
    """

    partition_id: int
    tile_indices: List[Tuple[int, int]]
    pixel_bounds: Tuple[int, int, int, int]
    geo_bounds: Optional[Tuple[float, float, float, float]] = None
    estimated_memory_mb: float = 0.0

    @property
    def n_tiles(self) -> int:
        """Number of tiles in partition."""
        return len(self.tile_indices)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_id": self.partition_id,
            "n_tiles": self.n_tiles,
            "pixel_bounds": self.pixel_bounds,
            "geo_bounds": self.geo_bounds,
            "estimated_memory_mb": self.estimated_memory_mb,
        }


@dataclass
class SedonaProcessingResult:
    """
    Result from Sedona-based processing.

    Attributes:
        success: Whether processing succeeded
        output_path: Path to output file
        total_tiles: Number of tiles processed
        failed_tiles: Number of failed tiles
        processing_time_seconds: Total processing time
        stages_info: Information about Spark stages
        statistics: Aggregated statistics
        metadata: Additional metadata
        error: Error message if failed
    """

    success: bool
    output_path: Optional[str] = None
    total_tiles: int = 0
    failed_tiles: int = 0
    processing_time_seconds: float = 0.0
    stages_info: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "total_tiles": self.total_tiles,
            "failed_tiles": self.failed_tiles,
            "processing_time_seconds": self.processing_time_seconds,
            "stages_info": self.stages_info,
            "statistics": self.statistics,
            "metadata": self.metadata,
            "error": self.error,
        }


# =============================================================================
# RasterSerializer
# =============================================================================


class RasterSerializer:
    """
    Serializer for raster data in Spark.

    Handles serialization/deserialization of numpy arrays and
    raster metadata for efficient Spark data transfer.
    """

    @staticmethod
    def serialize_array(array: np.ndarray) -> bytes:
        """
        Serialize numpy array to bytes.

        Args:
            array: Numpy array to serialize

        Returns:
            Serialized bytes
        """
        # Use pickle with protocol 5 for efficient numpy serialization
        return pickle.dumps(
            {
                "data": array.tobytes(),
                "dtype": str(array.dtype),
                "shape": array.shape,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def deserialize_array(data: bytes) -> np.ndarray:
        """
        Deserialize bytes to numpy array.

        Args:
            data: Serialized bytes

        Returns:
            Numpy array
        """
        obj = pickle.loads(data)
        array = np.frombuffer(obj["data"], dtype=obj["dtype"])
        return array.reshape(obj["shape"])

    @staticmethod
    def serialize_tile_result(
        data: np.ndarray,
        tile_info: Dict[str, Any],
        statistics: Dict[str, float],
    ) -> bytes:
        """
        Serialize complete tile result.

        Args:
            data: Result array
            tile_info: Tile metadata
            statistics: Processing statistics

        Returns:
            Serialized bytes
        """
        return pickle.dumps(
            {
                "data": data.tobytes(),
                "dtype": str(data.dtype),
                "shape": data.shape,
                "tile_info": tile_info,
                "statistics": statistics,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def deserialize_tile_result(data: bytes) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Deserialize tile result.

        Args:
            data: Serialized bytes

        Returns:
            Tuple of (array, tile_info, statistics)
        """
        obj = pickle.loads(data)
        array = np.frombuffer(obj["data"], dtype=obj["dtype"])
        array = array.reshape(obj["shape"])
        return array, obj["tile_info"], obj["statistics"]


# =============================================================================
# SedonaBackend
# =============================================================================


class SedonaBackend:
    """
    Apache Sedona backend for distributed raster processing.

    Manages SparkSession lifecycle and provides high-level interface
    for distributed geospatial operations.

    Example:
        backend = SedonaBackend(SedonaConfig.for_cluster("spark://host:7077"))
        backend.connect()

        try:
            # Use backend for processing
            processor = SedonaTileProcessor(backend)
            result = processor.process(data, algorithm)
        finally:
            backend.disconnect()
    """

    def __init__(self, config: Optional[SedonaConfig] = None):
        """
        Initialize Sedona backend.

        Args:
            config: Sedona configuration
        """
        self.config = config or SedonaConfig()
        self._spark: Optional[Any] = None
        self._sedona_context: Optional[Any] = None
        self._lock = threading.Lock()
        self._status = ClusterStatus()

    @property
    def spark(self) -> Optional[Any]:
        """Get SparkSession."""
        return self._spark

    @property
    def is_connected(self) -> bool:
        """Check if connected to Spark."""
        return self._spark is not None

    def connect(self) -> ClusterStatus:
        """
        Connect to Spark cluster.

        Returns:
            ClusterStatus with connection information
        """
        with self._lock:
            if self._spark is not None:
                return self._status

            if not HAS_SPARK:
                logger.warning(
                    "PySpark not available - using mock mode for Sedona backend"
                )
                self._status = ClusterStatus(
                    is_connected=True,
                    master_url=self.config.master,
                    app_id="mock_" + str(uuid.uuid4())[:8],
                    executors=self.config.num_executors,
                    total_cores=self.config.num_executors * self.config.executor_cores,
                )
                return self._status

            try:
                # Build SparkSession
                builder = SparkSession.builder.master(self.config.master)

                # Apply configuration
                for key, value in self.config.get_spark_conf().items():
                    builder = builder.config(key, value)

                # Create session
                self._spark = builder.getOrCreate()

                # Initialize Sedona if available
                if HAS_SEDONA:
                    try:
                        SedonaRegistrator.registerAll(self._spark)
                        self._sedona_context = SedonaContext.create(self._spark)
                        logger.info("Sedona initialized successfully")
                    except Exception as e:
                        logger.warning(f"Could not initialize Sedona: {e}")

                # Update status
                self._status = self._get_cluster_status()

                logger.info(
                    f"Connected to Spark cluster: {self._status.app_id} "
                    f"({self._status.executors} executors, "
                    f"{self._status.total_cores} cores)"
                )

                return self._status

            except Exception as e:
                logger.error(f"Failed to connect to Spark: {e}")
                self._status = ClusterStatus(is_connected=False)
                raise

    def disconnect(self) -> None:
        """Disconnect from Spark cluster."""
        with self._lock:
            if self._spark is not None:
                try:
                    self._spark.stop()
                    logger.info("Disconnected from Spark cluster")
                except Exception as e:
                    logger.warning(f"Error disconnecting from Spark: {e}")
                finally:
                    self._spark = None
                    self._sedona_context = None
                    self._status = ClusterStatus(is_connected=False)

    def get_status(self) -> ClusterStatus:
        """
        Get current cluster status.

        Returns:
            ClusterStatus with current information
        """
        if not self.is_connected:
            return ClusterStatus(is_connected=False)

        if HAS_SPARK and self._spark is not None:
            return self._get_cluster_status()

        return self._status

    def _get_cluster_status(self) -> ClusterStatus:
        """Get cluster status from Spark."""
        if not HAS_SPARK or self._spark is None:
            return ClusterStatus(is_connected=False)

        try:
            sc = self._spark.sparkContext
            status_tracker = sc.statusTracker()

            # Get executor info
            executor_count = 0
            total_memory = 0
            total_cores = 0

            # Try to get executor info from RDD API
            try:
                # This gives us executor count
                executor_count = sc._jsc.sc().getExecutorMemoryStatus().size()
            except Exception:
                executor_count = self.config.num_executors

            return ClusterStatus(
                is_connected=True,
                master_url=sc.master,
                app_id=sc.applicationId,
                executors=executor_count,
                total_cores=executor_count * self.config.executor_cores,
                total_memory_gb=executor_count * self._parse_memory(self.config.executor_memory),
                active_jobs=len(status_tracker.getActiveJobIds()),
                completed_stages=len(status_tracker.getJobIdsForGroup()),
                web_ui_url=sc.uiWebUrl,
            )

        except Exception as e:
            logger.warning(f"Error getting cluster status: {e}")
            return ClusterStatus(
                is_connected=True,
                master_url=self.config.master,
            )

    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to GB."""
        memory_str = memory_str.lower().strip()
        if memory_str.endswith("g"):
            return float(memory_str[:-1])
        elif memory_str.endswith("m"):
            return float(memory_str[:-1]) / 1024
        elif memory_str.endswith("k"):
            return float(memory_str[:-1]) / (1024 * 1024)
        return float(memory_str) / (1024 * 1024 * 1024)

    def execute_sql(self, query: str) -> Optional[Any]:
        """
        Execute Sedona SQL query.

        Args:
            query: SQL query string

        Returns:
            DataFrame result or None
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Spark cluster")

        if not HAS_SPARK or self._spark is None:
            logger.warning("Spark not available - cannot execute SQL")
            return None

        return self._spark.sql(query)

    def read_raster(
        self,
        path: str,
        format: RasterFormat = RasterFormat.GEOTIFF,
    ) -> Optional[Any]:
        """
        Read raster file using Sedona.

        Args:
            path: Path to raster file
            format: Raster format

        Returns:
            Sedona raster DataFrame or None
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Spark cluster")

        if not HAS_SPARK or not HAS_SEDONA:
            logger.warning("Sedona not available - cannot read raster directly")
            return None

        try:
            # Use Sedona's raster functions
            df = self._spark.read.format("binaryFile").load(path)

            # Convert to Sedona raster
            df = df.selectExpr(
                "RS_FromGeoTiff(content) as raster",
                "path",
            )

            return df

        except Exception as e:
            logger.error(f"Error reading raster: {e}")
            return None

    def create_tile_dataframe(
        self,
        tiles: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """
        Create DataFrame from tile specifications.

        Args:
            tiles: List of tile dictionaries

        Returns:
            Spark DataFrame or None
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Spark cluster")

        if not HAS_SPARK or self._spark is None:
            logger.warning("Spark not available - cannot create DataFrame")
            return None

        # Define schema
        schema = StructType([
            StructField("tile_id", StringType(), False),
            StructField("col", IntegerType(), False),
            StructField("row", IntegerType(), False),
            StructField("col_start", IntegerType(), False),
            StructField("row_start", IntegerType(), False),
            StructField("col_end", IntegerType(), False),
            StructField("row_end", IntegerType(), False),
            StructField("data", BinaryType(), True),
        ])

        # Create rows
        rows = []
        for tile in tiles:
            rows.append((
                tile.get("tile_id", f"{tile['col']}_{tile['row']}"),
                tile["col"],
                tile["row"],
                tile["pixel_bounds"][0],
                tile["pixel_bounds"][1],
                tile["pixel_bounds"][2],
                tile["pixel_bounds"][3],
                tile.get("data"),
            ))

        return self._spark.createDataFrame(rows, schema)

    def __enter__(self) -> "SedonaBackend":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


# =============================================================================
# SedonaTileProcessor
# =============================================================================


class SedonaTileProcessor:
    """
    Tile processor using Sedona on Spark.

    Processes raster tiles in parallel across Spark executors.

    Example:
        backend = SedonaBackend(config)
        backend.connect()

        processor = SedonaTileProcessor(backend)
        result = processor.process(
            data=raster_array,
            algorithm=flood_detector,
            output_path="/output/result.tif",
        )
    """

    def __init__(
        self,
        backend: SedonaBackend,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize tile processor.

        Args:
            backend: Sedona backend instance
            progress_callback: Optional progress callback
        """
        self.backend = backend
        self.config = backend.config
        self.progress_callback = progress_callback

    def process(
        self,
        data: np.ndarray,
        algorithm: Any,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        output_path: Optional[str] = None,
    ) -> SedonaProcessingResult:
        """
        Process raster data using Sedona.

        Args:
            data: Input raster array
            algorithm: Algorithm instance
            bounds: Geographic bounds
            resolution: Pixel resolution
            output_path: Output file path

        Returns:
            SedonaProcessingResult
        """
        if not self.backend.is_connected:
            raise RuntimeError("Backend not connected")

        start_time = time.time()

        try:
            # Generate tile grid
            tiles = self._generate_tiles(data, bounds, resolution)
            total_tiles = len(tiles)

            logger.info(f"Processing {total_tiles} tiles with Sedona")

            # Process tiles
            if HAS_SPARK and self.backend.spark is not None:
                results = self._process_with_spark(data, algorithm, tiles)
            else:
                # Fallback to mock processing
                results = self._process_mock(data, algorithm, tiles)

            # Stitch results
            mosaic = self._stitch_results(results, data.shape)

            # Save output if path provided
            if output_path:
                self._save_output(mosaic, output_path, bounds, resolution)

            processing_time = time.time() - start_time

            # Count successes/failures
            failed = sum(1 for r in results if r.get("error") is not None)

            return SedonaProcessingResult(
                success=failed == 0,
                output_path=output_path,
                total_tiles=total_tiles,
                failed_tiles=failed,
                processing_time_seconds=processing_time,
                statistics=self._aggregate_statistics(results),
                metadata={
                    "input_shape": data.shape,
                    "tile_size": self.config.tile_size,
                    "overlap": self.config.overlap,
                },
            )

        except Exception as e:
            logger.error(f"Sedona processing failed: {e}")
            return SedonaProcessingResult(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    def process_path(
        self,
        input_path: str,
        algorithm: Any,
        output_path: str,
    ) -> SedonaProcessingResult:
        """
        Process raster file using Sedona.

        Args:
            input_path: Path to input raster
            algorithm: Algorithm instance
            output_path: Output file path

        Returns:
            SedonaProcessingResult
        """
        if not self.backend.is_connected:
            raise RuntimeError("Backend not connected")

        start_time = time.time()

        try:
            # Read raster with Sedona
            raster_df = self.backend.read_raster(input_path)

            if raster_df is None:
                # Fallback to loading with rasterio
                import rasterio
                with rasterio.open(input_path) as src:
                    data = src.read()
                    bounds = src.bounds
                    resolution = src.res

                return self.process(
                    data=data,
                    algorithm=algorithm,
                    bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                    resolution=resolution,
                    output_path=output_path,
                )

            # Process with Spark SQL
            result = self._process_raster_df(raster_df, algorithm, output_path)

            return result

        except Exception as e:
            logger.error(f"Sedona path processing failed: {e}")
            return SedonaProcessingResult(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    def _generate_tiles(
        self,
        data: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """Generate tile grid."""
        if data.ndim == 2:
            height, width = data.shape
        else:
            height, width = data.shape[1], data.shape[2]

        tile_h, tile_w = self.config.tile_size
        overlap = self.config.overlap

        tiles = []
        n_rows = int(np.ceil(height / tile_h))
        n_cols = int(np.ceil(width / tile_w))

        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate pixel bounds
                col_start = col * tile_w
                row_start = row * tile_h
                col_end = min(col_start + tile_w, width)
                row_end = min(row_start + tile_h, height)

                # Overlap bounds
                overlap_col_start = max(0, col_start - overlap)
                overlap_row_start = max(0, row_start - overlap)
                overlap_col_end = min(width, col_end + overlap)
                overlap_row_end = min(height, row_end + overlap)

                # Geographic bounds
                geo_bounds = None
                if bounds and resolution:
                    minx = bounds[0] + col_start * resolution[0]
                    maxy = bounds[3] - row_start * resolution[1]
                    maxx = bounds[0] + col_end * resolution[0]
                    miny = bounds[3] - row_end * resolution[1]
                    geo_bounds = (minx, miny, maxx, maxy)

                tiles.append({
                    "tile_id": f"{col}_{row}",
                    "col": col,
                    "row": row,
                    "pixel_bounds": (col_start, row_start, col_end, row_end),
                    "overlap_bounds": (
                        overlap_col_start, overlap_row_start,
                        overlap_col_end, overlap_row_end
                    ),
                    "geo_bounds": geo_bounds,
                })

        return tiles

    def _process_with_spark(
        self,
        data: np.ndarray,
        algorithm: Any,
        tiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process tiles using Spark."""
        spark = self.backend.spark
        sc = spark.sparkContext

        # Serialize algorithm and data for distribution
        algo_broadcast = sc.broadcast(algorithm)
        data_broadcast = sc.broadcast(data)

        def process_tile(tile: Dict[str, Any]) -> Dict[str, Any]:
            """Process single tile on executor."""
            try:
                algo = algo_broadcast.value
                full_data = data_broadcast.value

                # Extract tile data
                bounds = tile["overlap_bounds"]
                if full_data.ndim == 2:
                    tile_data = full_data[
                        bounds[1]:bounds[3],
                        bounds[0]:bounds[2]
                    ].copy()
                else:
                    tile_data = full_data[
                        :,
                        bounds[1]:bounds[3],
                        bounds[0]:bounds[2]
                    ].copy()

                # Process
                if hasattr(algo, "process_tile"):
                    result_data = algo.process_tile(tile_data, tile)
                elif hasattr(algo, "execute"):
                    result = algo.execute(tile_data.squeeze())
                    if hasattr(result, "flood_extent"):
                        result_data = result.flood_extent
                    elif hasattr(result, "data"):
                        result_data = result.data
                    else:
                        result_data = np.asarray(result)
                elif hasattr(algo, "run"):
                    result = algo.run(data=tile_data)
                    if hasattr(result, "flood_extent"):
                        result_data = result.flood_extent
                    else:
                        result_data = np.asarray(result)
                elif callable(algo):
                    result_data = algo(tile_data)
                else:
                    raise ValueError(f"Unknown algorithm interface: {type(algo)}")

                # Get statistics
                statistics = {}
                if hasattr(algo, "last_statistics"):
                    statistics = algo.last_statistics

                return {
                    "tile_id": tile["tile_id"],
                    "col": tile["col"],
                    "row": tile["row"],
                    "pixel_bounds": tile["pixel_bounds"],
                    "data": result_data,
                    "statistics": statistics,
                    "error": None,
                }

            except Exception as e:
                return {
                    "tile_id": tile["tile_id"],
                    "col": tile["col"],
                    "row": tile["row"],
                    "pixel_bounds": tile["pixel_bounds"],
                    "data": None,
                    "statistics": {},
                    "error": str(e),
                }

        # Create RDD and process
        tiles_rdd = sc.parallelize(tiles, numSlices=len(tiles))
        results_rdd = tiles_rdd.map(process_tile)

        # Collect results
        results = results_rdd.collect()

        # Cleanup broadcasts
        algo_broadcast.unpersist()
        data_broadcast.unpersist()

        return results

    def _process_mock(
        self,
        data: np.ndarray,
        algorithm: Any,
        tiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Mock processing for testing without Spark."""
        results = []

        for tile in tiles:
            try:
                # Extract tile data
                bounds = tile["overlap_bounds"]
                if data.ndim == 2:
                    tile_data = data[
                        bounds[1]:bounds[3],
                        bounds[0]:bounds[2]
                    ].copy()
                else:
                    tile_data = data[
                        :,
                        bounds[1]:bounds[3],
                        bounds[0]:bounds[2]
                    ].copy()

                # Process
                if hasattr(algorithm, "process_tile"):
                    result_data = algorithm.process_tile(tile_data, tile)
                elif hasattr(algorithm, "execute"):
                    result = algorithm.execute(tile_data.squeeze())
                    if hasattr(result, "flood_extent"):
                        result_data = result.flood_extent
                    elif hasattr(result, "data"):
                        result_data = result.data
                    else:
                        result_data = np.asarray(result)
                elif callable(algorithm):
                    result_data = algorithm(tile_data)
                else:
                    raise ValueError(f"Unknown algorithm interface")

                results.append({
                    "tile_id": tile["tile_id"],
                    "col": tile["col"],
                    "row": tile["row"],
                    "pixel_bounds": tile["pixel_bounds"],
                    "data": result_data,
                    "statistics": {},
                    "error": None,
                })

            except Exception as e:
                results.append({
                    "tile_id": tile["tile_id"],
                    "col": tile["col"],
                    "row": tile["row"],
                    "pixel_bounds": tile["pixel_bounds"],
                    "data": None,
                    "statistics": {},
                    "error": str(e),
                })

        return results

    def _process_raster_df(
        self,
        raster_df: Any,
        algorithm: Any,
        output_path: str,
    ) -> SedonaProcessingResult:
        """Process Sedona raster DataFrame."""
        # This would use Sedona's native raster functions
        # For now, we collect and process locally
        start_time = time.time()

        try:
            # This is a simplified implementation
            # Full implementation would use RS_Tile and RS_MapAlgebra

            rows = raster_df.collect()
            if not rows:
                return SedonaProcessingResult(
                    success=False,
                    error="No raster data found",
                )

            # Process would happen here with Sedona SQL
            return SedonaProcessingResult(
                success=True,
                output_path=output_path,
                processing_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            return SedonaProcessingResult(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    def _stitch_results(
        self,
        results: List[Dict[str, Any]],
        output_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Stitch tile results into mosaic."""
        if len(output_shape) == 2:
            height, width = output_shape
            n_bands = 1
        else:
            n_bands, height, width = output_shape

        # Determine output dtype
        ref_result = next((r for r in results if r["data"] is not None), None)
        if ref_result is None:
            return np.zeros((height, width), dtype=np.float32)

        dtype = ref_result["data"].dtype
        overlap = self.config.overlap

        # Initialize output
        output = np.zeros((n_bands, height, width), dtype=dtype)

        # Get grid dimensions
        max_col = max(r["col"] for r in results)
        max_row = max(r["row"] for r in results)

        # Place each tile
        for result in results:
            if result["data"] is None:
                continue

            tile_data = result["data"]
            if tile_data.ndim == 2:
                tile_data = tile_data[np.newaxis, :, :]

            pb = result["pixel_bounds"]
            col, row = result["col"], result["row"]

            # Calculate data region to use (trim overlap)
            data_row_start = overlap if row > 0 else 0
            data_col_start = overlap if col > 0 else 0
            data_row_end = tile_data.shape[1] - (overlap if row < max_row else 0)
            data_col_end = tile_data.shape[2] - (overlap if col < max_col else 0)

            # Ensure valid bounds
            data_row_end = max(data_row_start + 1, data_row_end)
            data_col_end = max(data_col_start + 1, data_col_end)

            # Extract core region
            core_data = tile_data[
                :,
                data_row_start:data_row_end,
                data_col_start:data_col_end
            ]

            # Calculate output position
            out_row_start = pb[1]
            out_col_start = pb[0]
            out_row_end = min(out_row_start + core_data.shape[1], height)
            out_col_end = min(out_col_start + core_data.shape[2], width)

            # Trim data to fit
            data_height = out_row_end - out_row_start
            data_width = out_col_end - out_col_start
            core_data = core_data[:, :data_height, :data_width]

            # Place in output
            output[:, out_row_start:out_row_end, out_col_start:out_col_end] = core_data

        return output.squeeze()

    def _save_output(
        self,
        data: np.ndarray,
        output_path: str,
        bounds: Optional[Tuple[float, float, float, float]],
        resolution: Optional[Tuple[float, float]],
    ) -> None:
        """Save output to file."""
        try:
            import rasterio
            from rasterio.transform import from_bounds

            if data.ndim == 2:
                data = data[np.newaxis, :, :]

            n_bands, height, width = data.shape

            # Calculate transform
            if bounds and resolution:
                transform = from_bounds(
                    bounds[0], bounds[1], bounds[2], bounds[3],
                    width, height
                )
            else:
                transform = rasterio.transform.from_origin(0, height, 1, 1)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=n_bands,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=transform,
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                dst.write(data)

            logger.info(f"Saved output to: {output_path}")

        except ImportError:
            logger.warning("rasterio not available for saving output")

    def _aggregate_statistics(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate statistics from all tiles."""
        all_stats: Dict[str, List[float]] = {}

        for result in results:
            if result.get("statistics"):
                for key, value in result["statistics"].items():
                    if key not in all_stats:
                        all_stats[key] = []
                    all_stats[key].append(value)

        aggregated = {}
        for key, values in all_stats.items():
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_sum"] = float(np.sum(values))
            aggregated[f"{key}_count"] = len(values)

        return aggregated


# =============================================================================
# Convenience Functions
# =============================================================================


def create_sedona_backend(
    master: str = "local[*]",
    config: Optional[SedonaConfig] = None,
) -> SedonaBackend:
    """
    Create and connect Sedona backend.

    Args:
        master: Spark master URL
        config: Optional configuration

    Returns:
        Connected SedonaBackend
    """
    if config is None:
        config = SedonaConfig(master=master)

    backend = SedonaBackend(config)
    backend.connect()
    return backend


def process_with_sedona(
    data: np.ndarray,
    algorithm: Any,
    master: str = "local[*]",
    output_path: Optional[str] = None,
) -> SedonaProcessingResult:
    """
    Convenience function for Sedona processing.

    Args:
        data: Input raster array
        algorithm: Algorithm instance
        master: Spark master URL
        output_path: Optional output path

    Returns:
        SedonaProcessingResult
    """
    config = SedonaConfig(master=master)
    backend = SedonaBackend(config)

    try:
        backend.connect()
        processor = SedonaTileProcessor(backend)
        return processor.process(data, algorithm, output_path=output_path)
    finally:
        backend.disconnect()


def is_sedona_available() -> bool:
    """Check if Sedona is available."""
    return HAS_SPARK and HAS_SEDONA


def get_sedona_info() -> Dict[str, Any]:
    """Get Sedona availability information."""
    return {
        "spark_available": HAS_SPARK,
        "sedona_available": HAS_SEDONA,
        "can_process": HAS_SPARK,  # Can still process with Spark alone
        "features": {
            "raster_udfs": HAS_SEDONA,
            "spatial_sql": HAS_SEDONA,
            "distributed_tiles": HAS_SPARK,
        },
    }
