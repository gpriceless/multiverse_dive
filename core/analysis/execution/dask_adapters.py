"""
Dask Adapters for Existing Algorithms.

Provides adapters and mixins to make existing algorithms compatible with
the Dask-based parallel tile processing system without modifying the
original algorithm code.

Key Components:
- DaskAlgorithmAdapter: Wrap any algorithm for Dask execution
- TiledAlgorithmMixin: Mixin class for adding tiled support
- AlgorithmWrapper: Functional wrapper for simple functions

Example Usage:
    from core.analysis.execution.dask_adapters import (
        DaskAlgorithmAdapter,
        wrap_algorithm_for_dask,
    )

    # Wrap existing algorithm
    sar_algorithm = ThresholdSARAlgorithm()
    dask_sar = DaskAlgorithmAdapter(sar_algorithm)

    # Use with Dask processor
    processor = DaskTileProcessor()
    result = processor.process(data, dask_sar)

    # Or use convenience function
    adapted = wrap_algorithm_for_dask(sar_algorithm)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class AlgorithmProtocol(Protocol):
    """Protocol for standard algorithms."""

    def run(self, **kwargs) -> Any:
        """Run the algorithm."""
        ...


class ExecutableProtocol(Protocol):
    """Protocol for algorithms with execute method."""

    def execute(self, *args, **kwargs) -> Any:
        """Execute the algorithm."""
        ...


class TiledProtocol(Protocol):
    """Protocol for tile-aware algorithms."""

    def process_tile(
        self,
        data: np.ndarray,
        tile_info: Any,
    ) -> np.ndarray:
        """Process a single tile."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TileContext:
    """
    Context information passed to tile processing.

    Attributes:
        col: Tile column index
        row: Tile row index
        bounds: Geographic bounds (if available)
        pixel_bounds: Pixel bounds in source image
        is_edge: Whether tile is on image edge
        overlap: Overlap size in pixels
        metadata: Additional metadata
    """

    col: int
    row: int
    bounds: Optional[Tuple[float, float, float, float]] = None
    pixel_bounds: Optional[Tuple[int, int, int, int]] = None
    is_edge: bool = False
    overlap: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "col": self.col,
            "row": self.row,
            "bounds": self.bounds,
            "pixel_bounds": self.pixel_bounds,
            "is_edge": self.is_edge,
            "overlap": self.overlap,
            "metadata": self.metadata,
        }


@dataclass
class TileResult:
    """
    Result from processing a tile.

    Attributes:
        data: Result data array
        confidence: Optional confidence array
        statistics: Tile statistics
        metadata: Additional metadata
    """

    data: np.ndarray
    confidence: Optional[np.ndarray] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DaskAlgorithmAdapter
# =============================================================================


class DaskAlgorithmAdapter:
    """
    Adapter to make any algorithm compatible with Dask tile processing.

    Wraps an existing algorithm and provides the `process_tile` interface
    expected by DaskTileProcessor.

    Example:
        # Wrap SAR threshold algorithm
        sar_algo = ThresholdSARAlgorithm()
        adapted = DaskAlgorithmAdapter(sar_algo)

        # Now works with Dask processor
        processor = DaskTileProcessor()
        result = processor.process(data, adapted)
    """

    def __init__(
        self,
        algorithm: Any,
        result_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        preprocess_tile: Optional[Callable[[np.ndarray, TileContext], np.ndarray]] = None,
        postprocess_tile: Optional[Callable[[np.ndarray, TileContext], np.ndarray]] = None,
        pass_context: bool = False,
    ):
        """
        Initialize adapter.

        Args:
            algorithm: Algorithm instance to wrap
            result_extractor: Function to extract array from algorithm result
            preprocess_tile: Optional preprocessing function
            postprocess_tile: Optional postprocessing function
            pass_context: Whether to pass tile context to algorithm
        """
        self.algorithm = algorithm
        self.result_extractor = result_extractor or self._default_extractor
        self.preprocess_tile = preprocess_tile
        self.postprocess_tile = postprocess_tile
        self.pass_context = pass_context

        # Track statistics from last execution
        self.last_statistics: Dict[str, float] = {}

        # Detect algorithm interface
        self._detect_interface()

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
        tile_info: Any,
    ) -> np.ndarray:
        """
        Process a single tile.

        This is the main interface expected by DaskTileProcessor.

        Args:
            data: Tile data array
            tile_info: Tile information (TileContext or similar)

        Returns:
            Processed tile data
        """
        # Convert tile_info to TileContext if needed
        if isinstance(tile_info, TileContext):
            context = tile_info
        else:
            context = self._convert_tile_info(tile_info)

        # Preprocess if configured
        if self.preprocess_tile is not None:
            data = self.preprocess_tile(data, context)

        # Execute algorithm
        result = self._execute_algorithm(data, context)

        # Extract result array
        result_data = self.result_extractor(result)

        # Postprocess if configured
        if self.postprocess_tile is not None:
            result_data = self.postprocess_tile(result_data, context)

        # Extract statistics
        self._extract_statistics(result)

        return result_data

    def _execute_algorithm(
        self,
        data: np.ndarray,
        context: TileContext,
    ) -> Any:
        """Execute the wrapped algorithm."""
        # Squeeze to 2D if needed (most algorithms expect 2D input)
        if data.ndim == 3 and data.shape[0] == 1:
            input_data = data.squeeze(axis=0)
        else:
            input_data = data

        # Native tiled support
        if self._has_process_tile:
            return self.algorithm.process_tile(data, context)

        # Execute method (common in this codebase)
        if self._has_execute:
            return self.algorithm.execute(input_data)

        # Run method with keyword argument
        if self._has_run:
            if self.pass_context:
                return self.algorithm.run(data=data, context=context.to_dict())
            return self.algorithm.run(data=data)

        # Process method
        if self._has_process:
            return self.algorithm.process(input_data)

        # Callable
        if self._is_callable:
            return self.algorithm(input_data)

        raise RuntimeError("No valid algorithm interface found")

    def _convert_tile_info(self, tile_info: Any) -> TileContext:
        """Convert tile_info to TileContext."""
        if hasattr(tile_info, "col") and hasattr(tile_info, "row"):
            return TileContext(
                col=tile_info.col,
                row=tile_info.row,
                bounds=getattr(tile_info, "geo_bounds", None),
                pixel_bounds=getattr(tile_info, "pixel_bounds", None),
                overlap=getattr(tile_info, "overlap", 0),
            )
        elif isinstance(tile_info, dict):
            return TileContext(**tile_info)
        else:
            return TileContext(col=0, row=0)

    def _default_extractor(self, result: Any) -> np.ndarray:
        """Default result extractor."""
        # Direct array
        if isinstance(result, np.ndarray):
            return result

        # Result object with common attributes
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

        # Try to convert to array
        try:
            return np.asarray(result)
        except Exception:
            raise ValueError(
                f"Cannot extract array from result type {type(result)}. "
                "Provide a custom result_extractor function."
            )

    def _extract_statistics(self, result: Any) -> None:
        """Extract statistics from result."""
        self.last_statistics = {}

        if hasattr(result, "statistics"):
            self.last_statistics = result.statistics.copy() if result.statistics else {}
        elif hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            if "statistics" in result_dict:
                self.last_statistics = result_dict["statistics"]

    @property
    def METADATA(self) -> Dict[str, Any]:
        """Forward METADATA from wrapped algorithm."""
        if hasattr(self.algorithm, "METADATA"):
            return self.algorithm.METADATA
        return {}

    def __repr__(self) -> str:
        return f"DaskAlgorithmAdapter({type(self.algorithm).__name__})"


# =============================================================================
# TiledAlgorithmMixin
# =============================================================================


class TiledAlgorithmMixin:
    """
    Mixin class to add tiled processing support to algorithms.

    Inherit from this mixin to add `process_tile` method that works
    with the Dask tile processor.

    Example:
        class MyAlgorithm(TiledAlgorithmMixin, BaseAlgorithm):
            def _process_single(self, data):
                # Core processing logic
                return result

        # Now has process_tile method
        algo = MyAlgorithm()
        result = algo.process_tile(tile_data, tile_context)
    """

    def process_tile(
        self,
        data: np.ndarray,
        tile_info: Any,
    ) -> np.ndarray:
        """
        Process a single tile.

        Override this method for custom tile handling, or implement
        `_process_single` for the default behavior.

        Args:
            data: Tile data array
            tile_info: Tile context information

        Returns:
            Processed tile data
        """
        # Preprocess
        data = self._preprocess_tile(data, tile_info)

        # Process
        if hasattr(self, "_process_single"):
            result = self._process_single(data)
        elif hasattr(self, "execute"):
            result = self.execute(data.squeeze())
        elif hasattr(self, "run"):
            result = self.run(data=data)
        else:
            raise NotImplementedError(
                "Implement _process_single, execute, or run method"
            )

        # Extract array from result
        result_data = self._extract_result_array(result)

        # Postprocess
        result_data = self._postprocess_tile(result_data, tile_info)

        return result_data

    def _preprocess_tile(
        self,
        data: np.ndarray,
        tile_info: Any,
    ) -> np.ndarray:
        """
        Preprocess tile before main processing.

        Override for custom preprocessing.
        """
        return data

    def _postprocess_tile(
        self,
        result: np.ndarray,
        tile_info: Any,
    ) -> np.ndarray:
        """
        Postprocess tile result.

        Override for custom postprocessing.
        """
        return result

    def _extract_result_array(self, result: Any) -> np.ndarray:
        """Extract array from result object."""
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "flood_extent"):
            return result.flood_extent
        if hasattr(result, "data"):
            return result.data
        return np.asarray(result)

    @property
    def supports_tiled(self) -> bool:
        """Indicate that this algorithm supports tiled processing."""
        return True


# =============================================================================
# AlgorithmWrapper
# =============================================================================


class AlgorithmWrapper:
    """
    Wrap a simple function as an algorithm compatible with Dask processing.

    Example:
        # Wrap a simple function
        def detect_water(data):
            return data < -15

        algo = AlgorithmWrapper(detect_water)
        result = processor.process(data, algo)

        # With preprocessing
        algo = AlgorithmWrapper(
            detect_water,
            preprocess=lambda x: x.squeeze(),
            postprocess=lambda x: x.astype(np.uint8),
        )
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        name: str = "wrapped_function",
    ):
        """
        Initialize wrapper.

        Args:
            func: Processing function
            preprocess: Optional preprocessing function
            postprocess: Optional postprocessing function
            name: Name for the wrapper
        """
        self.func = func
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.name = name
        self.last_statistics: Dict[str, float] = {}

    def process_tile(
        self,
        data: np.ndarray,
        tile_info: Any,
    ) -> np.ndarray:
        """Process a single tile."""
        if self.preprocess is not None:
            data = self.preprocess(data)

        result = self.func(data)

        if self.postprocess is not None:
            result = self.postprocess(result)

        return result

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Execute on data (non-tiled interface)."""
        if self.preprocess is not None:
            data = self.preprocess(data)

        result = self.func(data)

        if self.postprocess is not None:
            result = self.postprocess(result)

        return result

    @property
    def supports_tiled(self) -> bool:
        """Support tiled processing."""
        return True

    def __repr__(self) -> str:
        return f"AlgorithmWrapper({self.name})"


# =============================================================================
# Algorithm-Specific Adapters
# =============================================================================


class FloodAlgorithmAdapter(DaskAlgorithmAdapter):
    """
    Specialized adapter for flood detection algorithms.

    Handles common patterns in flood algorithms:
    - SAR backscatter thresholding
    - NDWI optical detection
    - HAND model processing
    """

    def __init__(self, algorithm: Any):
        """
        Initialize flood adapter.

        Args:
            algorithm: Flood algorithm instance
        """
        super().__init__(
            algorithm=algorithm,
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


class WildfireAlgorithmAdapter(DaskAlgorithmAdapter):
    """
    Specialized adapter for wildfire detection algorithms.

    Handles common patterns in wildfire algorithms:
    - dNBR burn severity
    - Thermal anomaly detection
    - Active fire detection
    """

    def __init__(self, algorithm: Any):
        """
        Initialize wildfire adapter.

        Args:
            algorithm: Wildfire algorithm instance
        """
        super().__init__(
            algorithm=algorithm,
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


class StormAlgorithmAdapter(DaskAlgorithmAdapter):
    """
    Specialized adapter for storm damage algorithms.

    Handles common patterns in storm algorithms:
    - Wind damage assessment
    - Structural damage detection
    """

    def __init__(self, algorithm: Any):
        """
        Initialize storm adapter.

        Args:
            algorithm: Storm algorithm instance
        """
        super().__init__(
            algorithm=algorithm,
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
# Factory Functions
# =============================================================================


def wrap_algorithm_for_dask(
    algorithm: Any,
    algorithm_type: Optional[str] = None,
) -> DaskAlgorithmAdapter:
    """
    Wrap an algorithm for Dask processing.

    Args:
        algorithm: Algorithm instance
        algorithm_type: Type hint ("flood", "wildfire", "storm") or None for auto

    Returns:
        DaskAlgorithmAdapter instance
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
        return FloodAlgorithmAdapter(algorithm)
    elif algorithm_type == "wildfire":
        return WildfireAlgorithmAdapter(algorithm)
    elif algorithm_type == "storm":
        return StormAlgorithmAdapter(algorithm)

    # Default adapter
    return DaskAlgorithmAdapter(algorithm)


def create_tiled_algorithm(
    func: Callable[[np.ndarray], np.ndarray],
    name: str = "custom_algorithm",
    preprocess: Optional[Callable] = None,
    postprocess: Optional[Callable] = None,
) -> AlgorithmWrapper:
    """
    Create a tiled algorithm from a function.

    Args:
        func: Processing function
        name: Algorithm name
        preprocess: Optional preprocessing
        postprocess: Optional postprocessing

    Returns:
        AlgorithmWrapper instance
    """
    return AlgorithmWrapper(
        func=func,
        preprocess=preprocess,
        postprocess=postprocess,
        name=name,
    )


def adapt_all_algorithms(
    algorithms: List[Any],
) -> List[DaskAlgorithmAdapter]:
    """
    Wrap multiple algorithms for Dask processing.

    Args:
        algorithms: List of algorithm instances

    Returns:
        List of adapted algorithms
    """
    return [wrap_algorithm_for_dask(algo) for algo in algorithms]


# =============================================================================
# Validation Functions
# =============================================================================


def check_algorithm_compatibility(algorithm: Any) -> Dict[str, bool]:
    """
    Check if an algorithm is compatible with Dask processing.

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
        "supports_tiled_flag": getattr(algorithm, "supports_tiled", False),
        "has_metadata": hasattr(algorithm, "METADATA"),
        "can_be_wrapped": any([
            hasattr(algorithm, "execute"),
            hasattr(algorithm, "run"),
            hasattr(algorithm, "process"),
            hasattr(algorithm, "process_tile"),
            callable(algorithm),
        ]),
    }


def validate_adapter(adapter: DaskAlgorithmAdapter) -> Tuple[bool, List[str]]:
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

    # Test process_tile with dummy data
    try:
        test_data = np.random.rand(1, 64, 64).astype(np.float32)
        test_context = TileContext(col=0, row=0)
        result = adapter.process_tile(test_data, test_context)

        if not isinstance(result, np.ndarray):
            issues.append(f"process_tile returned {type(result)}, expected np.ndarray")

    except Exception as e:
        issues.append(f"process_tile failed: {str(e)}")

    return len(issues) == 0, issues
