"""
Sedona Adapters for Algorithm Execution on Spark.

Provides adapters to wrap existing algorithms for execution on
Apache Sedona/Spark clusters, handling serialization, UDF creation,
and result collection.

Key Components:
- SedonaAlgorithmAdapter: Wrap algorithms for Sedona execution
- SparkUDFFactory: Create Spark UDFs from algorithms
- ResultCollector: Collect and aggregate distributed results
- RasterBroadcaster: Efficient raster data broadcasting

Example Usage:
    from core.analysis.execution.sedona_adapters import (
        SedonaAlgorithmAdapter,
        wrap_algorithm_for_sedona,
    )

    # Wrap existing algorithm
    flood_algo = ThresholdSARAlgorithm()
    sedona_algo = SedonaAlgorithmAdapter(flood_algo)

    # Use with Sedona processor
    processor = SedonaTileProcessor(backend)
    result = processor.process(data, sedona_algo)

    # Or use factory function
    adapted = wrap_algorithm_for_sedona(flood_algo)
"""

import hashlib
import logging
import pickle
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional Spark imports
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        DoubleType,
        FloatType,
        IntegerType,
        StructField,
        StructType,
    )
    from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
    HAS_SPARK = True
except ImportError:
    SparkSession = None
    DataFrame = None
    F = None
    pandas_udf = None
    PandasUDFType = None
    HAS_SPARK = False

try:
    from sedona.sql.st_functions import ST_GeomFromText
    HAS_SEDONA = True
except ImportError:
    ST_GeomFromText = None
    HAS_SEDONA = False


# =============================================================================
# Protocols
# =============================================================================


class SedonaAlgorithmProtocol(Protocol):
    """Protocol for Sedona-compatible algorithms."""

    def process_partition(
        self,
        partition_data: List[Tuple[np.ndarray, Dict]],
    ) -> List[Tuple[np.ndarray, Dict]]:
        """Process a partition of tiles."""
        ...


class SerializableAlgorithmProtocol(Protocol):
    """Protocol for serializable algorithms."""

    def serialize(self) -> bytes:
        """Serialize algorithm state."""
        ...

    @classmethod
    def deserialize(cls, data: bytes) -> "SerializableAlgorithmProtocol":
        """Deserialize algorithm from bytes."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TileData:
    """
    Data container for a tile in Sedona processing.

    Attributes:
        tile_id: Unique tile identifier
        col: Column index in grid
        row: Row index in grid
        data: Raster data array
        pixel_bounds: Pixel bounds (col_start, row_start, col_end, row_end)
        geo_bounds: Geographic bounds
        metadata: Additional metadata
    """

    tile_id: str
    col: int
    row: int
    data: np.ndarray
    pixel_bounds: Tuple[int, int, int, int]
    geo_bounds: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "tile_id": self.tile_id,
            "col": self.col,
            "row": self.row,
            "pixel_bounds": self.pixel_bounds,
            "geo_bounds": self.geo_bounds,
            "metadata": self.metadata,
        }


@dataclass
class TileResult:
    """
    Result from processing a tile.

    Attributes:
        tile_id: Tile identifier
        data: Result array
        confidence: Optional confidence array
        statistics: Processing statistics
        success: Whether processing succeeded
        error: Error message if failed
    """

    tile_id: str
    data: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "tile_id": self.tile_id,
            "success": self.success,
            "statistics": self.statistics,
            "error": self.error,
        }


@dataclass
class AdapterConfig:
    """
    Configuration for Sedona adapters.

    Attributes:
        serialize_algorithm: Whether to serialize algorithm for broadcast
        cache_results: Whether to cache intermediate results
        collect_statistics: Whether to collect per-tile statistics
        retry_failed: Whether to retry failed tiles
        max_retries: Maximum retry attempts
        log_progress: Whether to log progress
    """

    serialize_algorithm: bool = True
    cache_results: bool = False
    collect_statistics: bool = True
    retry_failed: bool = True
    max_retries: int = 2
    log_progress: bool = True


# =============================================================================
# AlgorithmSerializer
# =============================================================================


class AlgorithmSerializer:
    """
    Serializer for algorithms to enable Spark broadcast.

    Handles serialization of algorithm instances including their
    state and parameters for distribution to Spark executors.
    """

    @staticmethod
    def can_serialize(algorithm: Any) -> bool:
        """
        Check if algorithm can be serialized.

        Args:
            algorithm: Algorithm instance

        Returns:
            True if serializable
        """
        try:
            # Try pickle serialization
            pickle.dumps(algorithm)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            return False

    @staticmethod
    def serialize(algorithm: Any) -> bytes:
        """
        Serialize algorithm to bytes.

        Args:
            algorithm: Algorithm instance

        Returns:
            Serialized bytes
        """
        return pickle.dumps(algorithm, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize algorithm from bytes.

        Args:
            data: Serialized bytes

        Returns:
            Algorithm instance
        """
        return pickle.loads(data)

    @staticmethod
    def get_signature(algorithm: Any) -> str:
        """
        Get unique signature for algorithm.

        Args:
            algorithm: Algorithm instance

        Returns:
            Hash signature
        """
        # Create signature from class name and parameters
        sig_data = {
            "class": type(algorithm).__name__,
            "module": type(algorithm).__module__,
        }

        # Add parameters if available
        if hasattr(algorithm, "__dict__"):
            for key, value in algorithm.__dict__.items():
                if not key.startswith("_"):
                    try:
                        # Only include serializable values
                        pickle.dumps(value)
                        sig_data[key] = str(value)
                    except Exception:
                        pass

        sig_str = str(sorted(sig_data.items()))
        return hashlib.md5(sig_str.encode()).hexdigest()[:16]


# =============================================================================
# SedonaAlgorithmAdapter
# =============================================================================


class SedonaAlgorithmAdapter:
    """
    Adapter to make algorithms compatible with Sedona/Spark execution.

    Wraps an existing algorithm and provides the interface expected
    by SedonaTileProcessor for distributed execution.

    Example:
        # Wrap flood detection algorithm
        flood_algo = ThresholdSARAlgorithm()
        adapted = SedonaAlgorithmAdapter(flood_algo)

        # Now works with Sedona processor
        processor = SedonaTileProcessor(backend)
        result = processor.process(data, adapted)
    """

    def __init__(
        self,
        algorithm: Any,
        config: Optional[AdapterConfig] = None,
        result_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        preprocess: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None,
        postprocess: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None,
    ):
        """
        Initialize adapter.

        Args:
            algorithm: Algorithm instance to wrap
            config: Adapter configuration
            result_extractor: Function to extract array from algorithm result
            preprocess: Optional preprocessing function
            postprocess: Optional postprocessing function
        """
        self.algorithm = algorithm
        self.config = config or AdapterConfig()
        self.result_extractor = result_extractor or self._default_extractor
        self.preprocess = preprocess
        self.postprocess = postprocess

        # Detect algorithm interface
        self._detect_interface()

        # Track statistics
        self.last_statistics: Dict[str, float] = {}

        # Verify serializability
        if self.config.serialize_algorithm:
            if not AlgorithmSerializer.can_serialize(algorithm):
                logger.warning(
                    f"Algorithm {type(algorithm).__name__} may not serialize properly"
                )

    def _detect_interface(self) -> None:
        """Detect the algorithm's interface."""
        self._has_execute = hasattr(self.algorithm, "execute")
        self._has_run = hasattr(self.algorithm, "run")
        self._has_process = hasattr(self.algorithm, "process")
        self._has_process_tile = hasattr(self.algorithm, "process_tile")
        self._is_callable = callable(self.algorithm)

        if not any([
            self._has_execute,
            self._has_run,
            self._has_process,
            self._has_process_tile,
            self._is_callable,
        ]):
            raise ValueError(
                f"Algorithm {type(self.algorithm)} has no recognized interface. "
                "Expected one of: execute(), run(), process(), process_tile(), or __call__()"
            )

        logger.debug(
            f"Detected interfaces for {type(self.algorithm).__name__}: "
            f"execute={self._has_execute}, run={self._has_run}, "
            f"process={self._has_process}, process_tile={self._has_process_tile}"
        )

    def process_tile(
        self,
        data: np.ndarray,
        tile_info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Process a single tile.

        This is the main interface used by SedonaTileProcessor.

        Args:
            data: Tile data array
            tile_info: Tile metadata dictionary

        Returns:
            Processed tile data
        """
        # Preprocess if configured
        if self.preprocess is not None:
            data = self.preprocess(data, tile_info)

        # Execute algorithm
        result = self._execute_algorithm(data, tile_info)

        # Extract result array
        result_data = self.result_extractor(result)

        # Postprocess if configured
        if self.postprocess is not None:
            result_data = self.postprocess(result_data, tile_info)

        # Update statistics
        self._update_statistics(result)

        return result_data

    def process_partition(
        self,
        partition_data: List[Tuple[np.ndarray, Dict[str, Any]]],
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process a partition of tiles.

        Optimized for batch processing on Spark executors.

        Args:
            partition_data: List of (data, tile_info) tuples

        Returns:
            List of (result_data, result_info) tuples
        """
        results = []

        for data, tile_info in partition_data:
            try:
                result_data = self.process_tile(data, tile_info)
                result_info = {
                    "tile_id": tile_info.get("tile_id"),
                    "col": tile_info.get("col"),
                    "row": tile_info.get("row"),
                    "success": True,
                    "statistics": self.last_statistics.copy(),
                }
                results.append((result_data, result_info))

            except Exception as e:
                logger.error(f"Error processing tile {tile_info.get('tile_id')}: {e}")
                # Return empty result on error
                empty_data = np.zeros((1, 1), dtype=np.uint8)
                result_info = {
                    "tile_id": tile_info.get("tile_id"),
                    "col": tile_info.get("col"),
                    "row": tile_info.get("row"),
                    "success": False,
                    "error": str(e),
                }
                results.append((empty_data, result_info))

        return results

    def _execute_algorithm(
        self,
        data: np.ndarray,
        tile_info: Dict[str, Any],
    ) -> Any:
        """Execute the wrapped algorithm."""
        # Squeeze to 2D if needed
        if data.ndim == 3 and data.shape[0] == 1:
            input_data = data.squeeze(axis=0)
        else:
            input_data = data

        # Native tiled support
        if self._has_process_tile:
            return self.algorithm.process_tile(data, tile_info)

        # Execute method
        if self._has_execute:
            return self.algorithm.execute(input_data)

        # Run method
        if self._has_run:
            return self.algorithm.run(data=data)

        # Process method
        if self._has_process:
            return self.algorithm.process(input_data)

        # Callable
        if self._is_callable:
            return self.algorithm(input_data)

        raise RuntimeError("No valid algorithm interface found")

    def _default_extractor(self, result: Any) -> np.ndarray:
        """Default result extractor."""
        if isinstance(result, np.ndarray):
            return result

        # Common result attributes
        if hasattr(result, "flood_extent"):
            return result.flood_extent
        if hasattr(result, "burn_mask"):
            return result.burn_mask
        if hasattr(result, "damage_mask"):
            return result.damage_mask
        if hasattr(result, "data"):
            return result.data
        if hasattr(result, "mask"):
            return result.mask

        try:
            return np.asarray(result)
        except Exception:
            raise ValueError(
                f"Cannot extract array from result type {type(result)}. "
                "Provide a custom result_extractor function."
            )

    def _update_statistics(self, result: Any) -> None:
        """Update statistics from result."""
        self.last_statistics = {}

        if hasattr(result, "statistics") and result.statistics:
            self.last_statistics = result.statistics.copy()
        elif hasattr(self.algorithm, "last_statistics"):
            self.last_statistics = self.algorithm.last_statistics.copy()

    @property
    def signature(self) -> str:
        """Get algorithm signature."""
        return AlgorithmSerializer.get_signature(self.algorithm)

    @property
    def METADATA(self) -> Dict[str, Any]:
        """Forward METADATA from wrapped algorithm."""
        if hasattr(self.algorithm, "METADATA"):
            return self.algorithm.METADATA
        return {}

    def __repr__(self) -> str:
        return f"SedonaAlgorithmAdapter({type(self.algorithm).__name__})"


# =============================================================================
# Specialized Adapters
# =============================================================================


class FloodSedonaAdapter(SedonaAlgorithmAdapter):
    """
    Specialized adapter for flood detection algorithms.

    Handles common patterns in flood algorithms:
    - SAR backscatter thresholding
    - NDWI optical detection
    - HAND model processing
    """

    def __init__(self, algorithm: Any, config: Optional[AdapterConfig] = None):
        """Initialize flood adapter."""
        super().__init__(
            algorithm=algorithm,
            config=config,
            result_extractor=self._extract_flood_result,
        )

    def _extract_flood_result(self, result: Any) -> np.ndarray:
        """Extract flood mask from result."""
        if isinstance(result, np.ndarray):
            return result

        if hasattr(result, "flood_extent"):
            return result.flood_extent
        if hasattr(result, "water_mask"):
            return result.water_mask
        if hasattr(result, "inundation_mask"):
            return result.inundation_mask

        return self._default_extractor(result)


class WildfireSedonaAdapter(SedonaAlgorithmAdapter):
    """
    Specialized adapter for wildfire detection algorithms.

    Handles common patterns in wildfire algorithms:
    - dNBR burn severity
    - Thermal anomaly detection
    - Active fire detection
    """

    def __init__(self, algorithm: Any, config: Optional[AdapterConfig] = None):
        """Initialize wildfire adapter."""
        super().__init__(
            algorithm=algorithm,
            config=config,
            result_extractor=self._extract_wildfire_result,
        )

    def _extract_wildfire_result(self, result: Any) -> np.ndarray:
        """Extract burn/fire mask from result."""
        if isinstance(result, np.ndarray):
            return result

        if hasattr(result, "burn_mask"):
            return result.burn_mask
        if hasattr(result, "severity"):
            return result.severity
        if hasattr(result, "active_fire_mask"):
            return result.active_fire_mask
        if hasattr(result, "burned_area"):
            return result.burned_area

        return self._default_extractor(result)


class StormSedonaAdapter(SedonaAlgorithmAdapter):
    """
    Specialized adapter for storm damage algorithms.

    Handles common patterns in storm algorithms:
    - Wind damage assessment
    - Structural damage detection
    """

    def __init__(self, algorithm: Any, config: Optional[AdapterConfig] = None):
        """Initialize storm adapter."""
        super().__init__(
            algorithm=algorithm,
            config=config,
            result_extractor=self._extract_storm_result,
        )

    def _extract_storm_result(self, result: Any) -> np.ndarray:
        """Extract damage mask from result."""
        if isinstance(result, np.ndarray):
            return result

        if hasattr(result, "damage_mask"):
            return result.damage_mask
        if hasattr(result, "damage_severity"):
            return result.damage_severity
        if hasattr(result, "wind_damage"):
            return result.wind_damage

        return self._default_extractor(result)


# =============================================================================
# SparkUDFFactory
# =============================================================================


class SparkUDFFactory:
    """
    Factory for creating Spark UDFs from algorithms.

    Creates User-Defined Functions that can be applied to
    Spark DataFrames for distributed processing.
    """

    @staticmethod
    def create_tile_udf(
        algorithm: Any,
        return_type: str = "binary",
    ) -> Optional[Callable]:
        """
        Create a Spark UDF for tile processing.

        Args:
            algorithm: Algorithm instance
            return_type: Return type ("binary", "array")

        Returns:
            Spark UDF function or None if Spark not available
        """
        if not HAS_SPARK:
            logger.warning("Spark not available - cannot create UDF")
            return None

        # Serialize algorithm for closure
        algo_bytes = AlgorithmSerializer.serialize(algorithm)

        def process_tile_udf(tile_data: bytes, tile_info_json: str) -> bytes:
            """UDF for processing a single tile."""
            import json
            import numpy as np
            import pickle

            # Deserialize algorithm
            algo = pickle.loads(algo_bytes)

            # Deserialize tile data
            tile_array = pickle.loads(tile_data)
            tile_info = json.loads(tile_info_json)

            # Process
            if hasattr(algo, "process_tile"):
                result = algo.process_tile(tile_array, tile_info)
            elif hasattr(algo, "execute"):
                result = algo.execute(tile_array.squeeze())
                if hasattr(result, "flood_extent"):
                    result = result.flood_extent
                elif hasattr(result, "data"):
                    result = result.data
            elif callable(algo):
                result = algo(tile_array)
            else:
                result = tile_array

            # Serialize result
            return pickle.dumps(np.asarray(result), protocol=pickle.HIGHEST_PROTOCOL)

        # Register as UDF
        from pyspark.sql.functions import udf
        return udf(process_tile_udf, BinaryType())

    @staticmethod
    def create_pandas_udf(
        algorithm: Any,
        output_schema: Optional[StructType] = None,
    ) -> Optional[Callable]:
        """
        Create a Pandas UDF for batch processing.

        Args:
            algorithm: Algorithm instance
            output_schema: Output schema for the UDF

        Returns:
            Pandas UDF or None if not available
        """
        if not HAS_SPARK or pandas_udf is None:
            logger.warning("Pandas UDFs not available")
            return None

        # Serialize algorithm
        algo_bytes = AlgorithmSerializer.serialize(algorithm)

        # Default output schema
        if output_schema is None:
            output_schema = StructType([
                StructField("tile_id", StringType()),
                StructField("result", BinaryType()),
                StructField("success", BooleanType()),
            ])

        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def process_tiles_pandas(pdf):
            """Pandas UDF for batch tile processing."""
            import pickle
            import numpy as np
            import pandas as pd

            algo = pickle.loads(algo_bytes)
            results = []

            for idx, row in pdf.iterrows():
                try:
                    tile_data = pickle.loads(row["data"])
                    tile_info = {
                        "tile_id": row["tile_id"],
                        "col": row["col"],
                        "row": row["row"],
                    }

                    if hasattr(algo, "process_tile"):
                        result = algo.process_tile(tile_data, tile_info)
                    elif hasattr(algo, "execute"):
                        result = algo.execute(tile_data.squeeze())
                    else:
                        result = tile_data

                    result_bytes = pickle.dumps(
                        np.asarray(result),
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
                    results.append({
                        "tile_id": row["tile_id"],
                        "result": result_bytes,
                        "success": True,
                    })

                except Exception as e:
                    results.append({
                        "tile_id": row["tile_id"],
                        "result": None,
                        "success": False,
                    })

            return pd.DataFrame(results)

        return process_tiles_pandas


# =============================================================================
# ResultCollector
# =============================================================================


class ResultCollector:
    """
    Collector for aggregating distributed processing results.

    Handles collection, validation, and aggregation of results
    from Spark executors.
    """

    def __init__(self):
        """Initialize collector."""
        self._results: Dict[str, TileResult] = {}
        self._lock = threading.Lock()
        self._statistics: Dict[str, List[float]] = {}

    def add_result(self, result: TileResult) -> None:
        """
        Add a tile result.

        Args:
            result: Tile processing result
        """
        with self._lock:
            self._results[result.tile_id] = result

            # Collect statistics
            for key, value in result.statistics.items():
                if key not in self._statistics:
                    self._statistics[key] = []
                self._statistics[key].append(value)

    def add_results(self, results: List[TileResult]) -> None:
        """
        Add multiple results.

        Args:
            results: List of tile results
        """
        for result in results:
            self.add_result(result)

    def get_result(self, tile_id: str) -> Optional[TileResult]:
        """
        Get result for a tile.

        Args:
            tile_id: Tile identifier

        Returns:
            TileResult or None
        """
        return self._results.get(tile_id)

    def get_all_results(self) -> Dict[str, TileResult]:
        """Get all results."""
        return self._results.copy()

    def get_successful_results(self) -> List[TileResult]:
        """Get all successful results."""
        return [r for r in self._results.values() if r.success]

    def get_failed_results(self) -> List[TileResult]:
        """Get all failed results."""
        return [r for r in self._results.values() if not r.success]

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if not self._results:
            return 0.0
        successful = sum(1 for r in self._results.values() if r.success)
        return successful / len(self._results)

    def aggregate_statistics(self) -> Dict[str, float]:
        """
        Aggregate statistics from all results.

        Returns:
            Dictionary of aggregated statistics
        """
        aggregated = {}

        for key, values in self._statistics.items():
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_sum"] = float(np.sum(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
                aggregated[f"{key}_count"] = len(values)

        return aggregated

    def clear(self) -> None:
        """Clear all results."""
        with self._lock:
            self._results.clear()
            self._statistics.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def wrap_algorithm_for_sedona(
    algorithm: Any,
    algorithm_type: Optional[str] = None,
    config: Optional[AdapterConfig] = None,
) -> SedonaAlgorithmAdapter:
    """
    Wrap an algorithm for Sedona execution.

    Args:
        algorithm: Algorithm instance
        algorithm_type: Type hint ("flood", "wildfire", "storm") or None for auto
        config: Adapter configuration

    Returns:
        SedonaAlgorithmAdapter instance
    """
    # Auto-detect type from algorithm metadata
    if algorithm_type is None and hasattr(algorithm, "METADATA"):
        meta = algorithm.METADATA
        algo_id = meta.get("id", "")
        if algo_id.startswith("flood"):
            algorithm_type = "flood"
        elif algo_id.startswith("wildfire") or algo_id.startswith("fire"):
            algorithm_type = "wildfire"
        elif algo_id.startswith("storm") or algo_id.startswith("wind"):
            algorithm_type = "storm"

    # Use specialized adapter if type known
    if algorithm_type == "flood":
        return FloodSedonaAdapter(algorithm, config)
    elif algorithm_type == "wildfire":
        return WildfireSedonaAdapter(algorithm, config)
    elif algorithm_type == "storm":
        return StormSedonaAdapter(algorithm, config)

    # Default adapter
    return SedonaAlgorithmAdapter(algorithm, config)


def adapt_algorithms_for_sedona(
    algorithms: List[Any],
    config: Optional[AdapterConfig] = None,
) -> List[SedonaAlgorithmAdapter]:
    """
    Wrap multiple algorithms for Sedona execution.

    Args:
        algorithms: List of algorithm instances
        config: Shared adapter configuration

    Returns:
        List of adapted algorithms
    """
    return [wrap_algorithm_for_sedona(algo, config=config) for algo in algorithms]


def check_sedona_compatibility(algorithm: Any) -> Dict[str, Any]:
    """
    Check if an algorithm is compatible with Sedona execution.

    Args:
        algorithm: Algorithm to check

    Returns:
        Dictionary with compatibility information
    """
    return {
        "has_execute": hasattr(algorithm, "execute"),
        "has_run": hasattr(algorithm, "run"),
        "has_process": hasattr(algorithm, "process"),
        "has_process_tile": hasattr(algorithm, "process_tile"),
        "is_callable": callable(algorithm),
        "is_serializable": AlgorithmSerializer.can_serialize(algorithm),
        "has_metadata": hasattr(algorithm, "METADATA"),
        "can_be_wrapped": any([
            hasattr(algorithm, "execute"),
            hasattr(algorithm, "run"),
            hasattr(algorithm, "process"),
            hasattr(algorithm, "process_tile"),
            callable(algorithm),
        ]),
        "signature": AlgorithmSerializer.get_signature(algorithm),
    }


def validate_sedona_adapter(adapter: SedonaAlgorithmAdapter) -> Tuple[bool, List[str]]:
    """
    Validate that an adapter is properly configured.

    Args:
        adapter: Adapter to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check wrapped algorithm
    if adapter.algorithm is None:
        issues.append("No algorithm wrapped")
        return False, issues

    # Check serializability
    if adapter.config.serialize_algorithm:
        if not AlgorithmSerializer.can_serialize(adapter.algorithm):
            issues.append("Algorithm cannot be serialized for Spark broadcast")

    # Test process_tile with dummy data
    try:
        test_data = np.random.rand(1, 64, 64).astype(np.float32)
        test_info = {"tile_id": "test", "col": 0, "row": 0}
        result = adapter.process_tile(test_data, test_info)

        if not isinstance(result, np.ndarray):
            issues.append(f"process_tile returned {type(result)}, expected np.ndarray")

    except Exception as e:
        issues.append(f"process_tile failed: {str(e)}")

    return len(issues) == 0, issues


# =============================================================================
# Type imports for external use
# =============================================================================

# Make StringType and BooleanType available if Spark is available
if HAS_SPARK:
    from pyspark.sql.types import StringType, BooleanType
else:
    StringType = None
    BooleanType = None
