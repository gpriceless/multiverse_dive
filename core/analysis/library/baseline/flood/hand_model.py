"""
HAND (Height Above Nearest Drainage) Flood Susceptibility Model

Uses topographic analysis to identify areas susceptible to flooding based on
height above the nearest drainage channel.

Algorithm ID: flood.baseline.hand_model
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class HANDModelConfig:
    """Configuration for HAND flood susceptibility model."""

    hand_threshold_m: float = 10.0  # Maximum HAND value for flood susceptibility (meters)
    channel_threshold_area_km2: float = 1.0  # Minimum drainage area for channel definition
    use_slope_factor: bool = True  # Include slope in susceptibility calculation
    slope_weight: float = 0.3  # Weight for slope factor (0-1)
    min_area_ha: float = 0.5  # Minimum flood polygon area (hectares)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hand_threshold_m <= 0:
            raise ValueError(f"hand_threshold_m must be positive, got {self.hand_threshold_m}")
        if self.channel_threshold_area_km2 <= 0:
            raise ValueError(f"channel_threshold_area_km2 must be positive, got {self.channel_threshold_area_km2}")
        if not 0.0 <= self.slope_weight <= 1.0:
            raise ValueError(f"slope_weight must be in [0, 1], got {self.slope_weight}")


@dataclass
class HANDModelResult:
    """Results from HAND flood susceptibility analysis."""

    susceptibility_mask: np.ndarray  # Binary mask of flood susceptibility
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    hand_raster: np.ndarray  # HAND values (meters)
    drainage_network: np.ndarray  # Identified drainage channels
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, float]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "susceptibility_mask": self.susceptibility_mask,
            "confidence_raster": self.confidence_raster,
            "hand_raster": self.hand_raster,
            "drainage_network": self.drainage_network,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class HANDModelAlgorithm:
    """
    HAND (Height Above Nearest Drainage) Flood Susceptibility Model.

    HAND represents the vertical distance between each location and the nearest
    drainage channel. Lower HAND values indicate higher flood susceptibility.

    This is a simplified implementation suitable for rapid assessment.
    For production use, consider more sophisticated implementations using:
    - TauDEM for flow routing and drainage delineation
    - GRASS GIS r.watershed
    - WhiteboxTools hydrological analysis

    Requirements:
        - Digital Elevation Model (DEM)
        - Optional: Flow accumulation raster
        - Optional: Slope raster

    Outputs:
        - susceptibility_mask: Binary mask of flood-susceptible areas
        - confidence_raster: Per-pixel susceptibility confidence (0-1)
        - hand_raster: HAND values for analysis (meters)
        - drainage_network: Identified drainage channels
    """

    METADATA = {
        "id": "flood.baseline.hand_model",
        "name": "HAND Flood Susceptibility Model",
        "category": "baseline",
        "event_types": ["flood.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "experimental": True,
        "requirements": {
            "data": {
                "dem": {"type": "elevation", "unit": "meters"}
            },
            "optional": {
                "flow_accumulation": {"benefit": "improved_channel_detection"},
                "slope": {"benefit": "improved_susceptibility"}
            },
            "compute": {"memory_gb": 8, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.65, 0.80],
            "validated_regions": [],
            "citations": ["doi:10.1016/j.envsoft.2012.11.009"],
            "notes": "Simplified implementation for rapid assessment"
        }
    }

    def __init__(self, config: Optional[HANDModelConfig] = None):
        """
        Initialize HAND model algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or HANDModelConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: HAND threshold={self.config.hand_threshold_m}m, "
                   f"channel area={self.config.channel_threshold_area_km2}km²")

    def execute(
        self,
        dem: np.ndarray,
        flow_accumulation: Optional[np.ndarray] = None,
        slope: Optional[np.ndarray] = None,
        pixel_size_m: float = 30.0,
        nodata_value: Optional[float] = None
    ) -> HANDModelResult:
        """
        Execute HAND flood susceptibility analysis.

        Args:
            dem: Digital Elevation Model (meters), shape (H, W)
            flow_accumulation: Optional flow accumulation (number of cells), shape (H, W)
            slope: Optional slope (degrees), shape (H, W)
            pixel_size_m: Pixel size in meters
            nodata_value: NoData value to mask out

        Returns:
            HANDModelResult containing susceptibility and HAND values
        """
        logger.info("Starting HAND flood susceptibility analysis")

        # Validate inputs
        if dem.ndim != 2:
            raise ValueError(f"Expected 2D DEM array, got shape {dem.shape}")

        # Create valid data mask
        valid_mask = np.ones_like(dem, dtype=bool)
        if nodata_value is not None:
            valid_mask &= (dem != nodata_value)
        valid_mask &= np.isfinite(dem)

        # Identify drainage network
        if flow_accumulation is not None:
            logger.info("Using provided flow accumulation")
            drainage_network = self._identify_channels_from_flow_accum(
                flow_accumulation, pixel_size_m, valid_mask
            )
        else:
            logger.info("Estimating drainage network from DEM")
            drainage_network = self._estimate_drainage_network(
                dem, pixel_size_m, valid_mask
            )

        # Calculate HAND values
        hand_raster = self._calculate_hand(dem, drainage_network, valid_mask)

        # Calculate susceptibility
        susceptibility_mask, confidence = self._calculate_susceptibility(
            hand_raster, slope, valid_mask
        )

        # Calculate statistics
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0  # m² to hectares
        susceptible_pixels = np.sum(susceptibility_mask)
        susceptible_area_ha = susceptible_pixels * pixel_area_ha
        channel_pixels = np.sum(drainage_network)

        statistics = {
            "total_pixels": int(dem.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "susceptible_pixels": int(susceptible_pixels),
            "susceptible_area_ha": float(susceptible_area_ha),
            "susceptible_percent": float(100.0 * susceptible_pixels / np.sum(valid_mask)) if np.sum(valid_mask) > 0 else 0.0,
            "channel_pixels": int(channel_pixels),
            "mean_hand": float(np.mean(hand_raster[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "mean_hand_susceptible": float(np.mean(hand_raster[susceptibility_mask])) if susceptible_pixels > 0 else 0.0,
            "mean_confidence": float(np.mean(confidence[susceptibility_mask])) if susceptible_pixels > 0 else 0.0
        }

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "hand_threshold_m": self.config.hand_threshold_m,
                "channel_threshold_area_km2": self.config.channel_threshold_area_km2,
                "use_slope_factor": self.config.use_slope_factor,
                "slope_weight": self.config.slope_weight,
                "min_area_ha": self.config.min_area_ha,
                "pixel_size_m": pixel_size_m
            },
            "execution": {
                "flow_accumulation_provided": flow_accumulation is not None,
                "slope_provided": slope is not None
            }
        }

        logger.info(f"Analysis complete: {susceptible_area_ha:.2f} ha susceptible area "
                   f"({statistics['susceptible_percent']:.1f}% of valid area)")
        logger.info(f"Mean HAND: {statistics['mean_hand']:.2f}m, "
                   f"Susceptible HAND: {statistics['mean_hand_susceptible']:.2f}m")

        return HANDModelResult(
            susceptibility_mask=susceptibility_mask,
            confidence_raster=confidence,
            hand_raster=hand_raster,
            drainage_network=drainage_network,
            metadata=metadata,
            statistics=statistics
        )

    def _identify_channels_from_flow_accum(
        self,
        flow_accumulation: np.ndarray,
        pixel_size_m: float,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Identify drainage channels from flow accumulation.

        Args:
            flow_accumulation: Flow accumulation (number of cells)
            pixel_size_m: Pixel size in meters
            valid_mask: Valid data mask

        Returns:
            Binary mask of drainage channels
        """
        # Convert area threshold to number of cells
        pixel_area_km2 = (pixel_size_m ** 2) / 1e6
        cells_threshold = self.config.channel_threshold_area_km2 / pixel_area_km2

        # Identify channels
        channels = (flow_accumulation >= cells_threshold) & valid_mask

        n_channels = np.sum(channels)
        logger.info(f"Identified {n_channels} channel pixels "
                   f"({100.0 * n_channels / np.sum(valid_mask):.2f}% of area)")

        return channels

    def _estimate_drainage_network(
        self,
        dem: np.ndarray,
        pixel_size_m: float,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate drainage network from DEM using simple topographic analysis.

        This is a simplified approach. For production use, proper flow routing
        algorithms should be used.

        Args:
            dem: Digital Elevation Model
            pixel_size_m: Pixel size in meters
            valid_mask: Valid data mask

        Returns:
            Binary mask of estimated drainage channels
        """
        logger.warning("Using simplified drainage network estimation. "
                      "For better results, provide flow accumulation data.")

        # Fill local depressions (simple approach)
        dem_filled = self._fill_depressions_simple(dem, valid_mask)

        # Calculate flow accumulation using simple D8 flow routing
        flow_accum = self._calculate_flow_accumulation_d8(dem_filled, valid_mask)

        # Identify channels
        channels = self._identify_channels_from_flow_accum(
            flow_accum, pixel_size_m, valid_mask
        )

        return channels

    def _fill_depressions_simple(
        self,
        dem: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """Simple depression filling using morphological reconstruction."""
        # Create a seed image (DEM + small constant at borders)
        seed = dem.copy()
        seed[1:-1, 1:-1] = np.inf

        # Morphological reconstruction using iterative grey_dilation
        # For depression filling, use iterative dilation approach
        filled = np.minimum(dem, ndimage.grey_dilation(seed, size=(3, 3)))
        filled[~valid_mask] = dem[~valid_mask]

        return filled

    def _calculate_flow_accumulation_d8(
        self,
        dem: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Simplified D8 flow accumulation.

        WARNING: This is a simplified placeholder. For production use, integrate pysheds or richdem.

        This is a very basic implementation for demonstration.
        Production systems should use proper hydrological algorithms.
        """
        logger.warning("D8 flow accumulation is experimental and not validated. "
                      "For production use, integrate pysheds or richdem.")
        # Initialize flow accumulation
        flow_accum = np.ones_like(dem, dtype=np.float32)

        # Simple approach: count downslope neighbors
        # This is NOT a proper flow accumulation algorithm
        for i in range(1, dem.shape[0] - 1):
            for j in range(1, dem.shape[1] - 1):
                if not valid_mask[i, j]:
                    continue

                # Check 8 neighbors
                neighbors = [
                    (i-1, j-1), (i-1, j), (i-1, j+1),
                    (i, j-1),             (i, j+1),
                    (i+1, j-1), (i+1, j), (i+1, j+1)
                ]

                for ni, nj in neighbors:
                    if valid_mask[ni, nj] and dem[ni, nj] > dem[i, j]:
                        flow_accum[i, j] += 1

        return flow_accum

    def _calculate_hand(
        self,
        dem: np.ndarray,
        drainage_network: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Height Above Nearest Drainage (HAND).

        Args:
            dem: Digital Elevation Model
            drainage_network: Binary mask of drainage channels
            valid_mask: Valid data mask

        Returns:
            HAND raster (meters)
        """
        logger.info("Calculating HAND values")

        # Initialize HAND
        hand = np.full_like(dem, np.inf, dtype=np.float32)

        # Drainage channels have HAND = 0
        hand[drainage_network] = 0.0

        # Calculate distance transform to find nearest drainage
        # This gives pixel distance, not accounting for elevation
        distance_pixels = ndimage.distance_transform_edt(~drainage_network)

        # For each pixel, calculate elevation difference to nearest drainage
        # Simplified approach: use elevation difference as HAND
        # (A proper implementation would trace flow paths)

        # Find nearest drainage pixel for each location
        _, indices = ndimage.distance_transform_edt(
            ~drainage_network,
            return_indices=True
        )

        # Calculate HAND as elevation difference to nearest drainage
        # Vectorized computation for efficiency
        nearest_i = indices[0]
        nearest_j = indices[1]

        # Get elevation at nearest drainage points
        nearest_elevations = dem[nearest_i, nearest_j]

        # Calculate HAND for all non-channel pixels
        hand = dem - nearest_elevations

        # Set channel pixels to 0
        hand[drainage_network] = 0.0

        # Ensure non-negative HAND
        hand = np.maximum(hand, 0.0)
        hand[~valid_mask] = np.nan

        logger.info(f"HAND range: {np.nanmin(hand):.2f} - {np.nanmax(hand):.2f} m")

        return hand

    def _calculate_susceptibility(
        self,
        hand: np.ndarray,
        slope: Optional[np.ndarray],
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate flood susceptibility from HAND and optional slope.

        Args:
            hand: HAND values (meters)
            slope: Optional slope (degrees)
            valid_mask: Valid data mask

        Returns:
            Tuple of (susceptibility_mask, confidence)
        """
        # Base susceptibility from HAND
        susceptibility_mask = (hand <= self.config.hand_threshold_m) & valid_mask

        # Calculate confidence
        confidence = np.zeros_like(hand, dtype=np.float32)

        # HAND-based confidence: lower HAND = higher confidence
        hand_conf = np.clip(
            1.0 - (hand / self.config.hand_threshold_m),
            0.0, 1.0
        )

        if slope is not None and self.config.use_slope_factor:
            logger.info("Including slope factor in susceptibility")
            # Slope-based confidence: lower slope = higher confidence
            # Assume slopes > 20 degrees are unlikely to flood
            slope_conf = np.clip(1.0 - (slope / 20.0), 0.0, 1.0)

            # Weighted combination
            confidence = (
                (1.0 - self.config.slope_weight) * hand_conf +
                self.config.slope_weight * slope_conf
            )
        else:
            confidence = hand_conf

        confidence[~valid_mask] = 0.0

        return susceptibility_mask, confidence

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return HANDModelAlgorithm.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'HANDModelAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = HANDModelConfig(**params)
        return HANDModelAlgorithm(config)
