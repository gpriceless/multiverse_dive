"""
NDWI Optical Flood Detection Algorithm

Uses Normalized Difference Water Index (NDWI) from optical imagery
to detect standing water and flood extent.

Algorithm ID: flood.baseline.ndwi_optical
Version: 1.1.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NDWIOpticalConfig:
    """Configuration for NDWI optical flood detection."""

    ndwi_threshold: float = 0.0  # NDWI threshold for water classification
    min_area_ha: float = 0.5  # Minimum flood polygon area (hectares)
    cloud_mask_enabled: bool = True  # Use cloud masking if available
    shadow_mask_enabled: bool = True  # Use shadow masking if available
    use_change_detection: bool = False  # Enable pre/post change detection
    change_threshold: float = 0.2  # NDWI change threshold

    def __post_init__(self):
        """Validate configuration parameters."""
        if not -1.0 <= self.ndwi_threshold <= 1.0:
            raise ValueError(f"ndwi_threshold must be in [-1.0, 1.0], got {self.ndwi_threshold}")
        if self.min_area_ha < 0:
            raise ValueError(f"min_area_ha must be non-negative, got {self.min_area_ha}")


@dataclass
class NDWIOpticalResult:
    """Results from NDWI optical flood detection."""

    flood_extent: np.ndarray  # Binary mask of flood extent
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    ndwi_raster: np.ndarray  # NDWI values
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, float]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "flood_extent": self.flood_extent,
            "confidence_raster": self.confidence_raster,
            "ndwi_raster": self.ndwi_raster,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class NDWIOpticalAlgorithm:
    """
    NDWI Optical Flood Detection.

    Calculates Normalized Difference Water Index (NDWI) from optical imagery:
    NDWI = (Green - NIR) / (Green + NIR)

    Water surfaces have high NDWI values (typically > 0.0) due to:
    - Strong absorption in NIR
    - Relatively high reflectance in green

    Requirements:
        - Optical imagery with Green and NIR bands
        - Optional: Cloud/shadow masks
        - Optional: Pre-event imagery for change detection

    Outputs:
        - flood_extent: Binary vector polygon of flood extent
        - confidence_raster: Per-pixel confidence (0-1)
        - ndwi_raster: NDWI values for analysis
    """

    METADATA = {
        "id": "flood.baseline.ndwi_optical",
        "name": "NDWI Optical Flood Detection",
        "category": "baseline",
        "event_types": ["flood.*"],
        "version": "1.1.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "optical": {
                    "bands": ["green", "nir"],
                    "temporal": "post_event",
                    "max_cloud_cover": 0.3
                }
            },
            "optional": {
                "optical_pre": {"temporal": "pre_event", "benefit": "enables_change_detection"},
                "cloud_mask": {"benefit": "improved_accuracy"},
                "shadow_mask": {"benefit": "reduced_false_positives"}
            },
            "compute": {"memory_gb": 4, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.70, 0.85],
            "validated_regions": ["global"],
            "citations": ["doi:10.1016/j.isprsjprs.2006.01.003"]
        }
    }

    def __init__(self, config: Optional[NDWIOpticalConfig] = None):
        """
        Initialize NDWI optical algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or NDWIOpticalConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: threshold={self.config.ndwi_threshold}, "
                   f"min_area={self.config.min_area_ha}ha")

    def execute(
        self,
        green_band: np.ndarray,
        nir_band: np.ndarray,
        green_pre: Optional[np.ndarray] = None,
        nir_pre: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
        shadow_mask: Optional[np.ndarray] = None,
        pixel_size_m: float = 10.0,
        nodata_value: Optional[float] = None
    ) -> NDWIOpticalResult:
        """
        Execute NDWI optical flood detection.

        Args:
            green_band: Post-event green band reflectance (0-1 or 0-10000), shape (H, W)
            nir_band: Post-event NIR band reflectance (0-1 or 0-10000), shape (H, W)
            green_pre: Optional pre-event green band
            nir_pre: Optional pre-event NIR band
            cloud_mask: Optional cloud mask (True = cloud)
            shadow_mask: Optional shadow mask (True = shadow)
            pixel_size_m: Pixel size in meters (for area calculation)
            nodata_value: NoData value to mask out

        Returns:
            NDWIOpticalResult containing flood extent and confidence
        """
        logger.info("Starting NDWI optical flood detection")

        # Validate inputs
        if green_band.shape != nir_band.shape:
            raise ValueError(f"Green/NIR shape mismatch: {green_band.shape} vs {nir_band.shape}")

        # Normalize reflectance to 0-1 if needed (handle DN values 0-10000)
        green_band = self._normalize_reflectance(green_band)
        nir_band = self._normalize_reflectance(nir_band)

        if green_pre is not None and nir_pre is not None:
            green_pre = self._normalize_reflectance(green_pre)
            nir_pre = self._normalize_reflectance(nir_pre)

        # Create valid data mask
        valid_mask = self._create_valid_mask(
            green_band, nir_band, cloud_mask, shadow_mask, nodata_value
        )

        # Calculate NDWI
        ndwi_post = self._calculate_ndwi(green_band, nir_band, valid_mask)

        # Apply threshold detection
        if green_pre is not None and nir_pre is not None and self.config.use_change_detection:
            logger.info("Using change detection mode")
            ndwi_pre = self._calculate_ndwi(green_pre, nir_pre, valid_mask)
            flood_extent, confidence = self._detect_with_change(
                ndwi_post, ndwi_pre, valid_mask
            )
        else:
            logger.info("Using simple threshold mode")
            flood_extent, confidence = self._detect_simple(ndwi_post, valid_mask)

        # Calculate statistics
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0  # m² to hectares
        flood_pixels = np.sum(flood_extent)
        flood_area_ha = flood_pixels * pixel_area_ha

        statistics = {
            "total_pixels": int(flood_extent.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "flood_pixels": int(flood_pixels),
            "flood_area_ha": float(flood_area_ha),
            "flood_percent": float(100.0 * flood_pixels / np.sum(valid_mask)) if np.sum(valid_mask) > 0 else 0.0,
            "mean_ndwi": float(np.mean(ndwi_post[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "mean_ndwi_flood": float(np.mean(ndwi_post[flood_extent])) if flood_pixels > 0 else 0.0,
            "mean_confidence": float(np.mean(confidence[flood_extent])) if flood_pixels > 0 else 0.0
        }

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "ndwi_threshold": self.config.ndwi_threshold,
                "min_area_ha": self.config.min_area_ha,
                "cloud_mask_enabled": self.config.cloud_mask_enabled,
                "shadow_mask_enabled": self.config.shadow_mask_enabled,
                "use_change_detection": self.config.use_change_detection,
                "pixel_size_m": pixel_size_m
            },
            "execution": {
                "mode": "change_detection" if (green_pre is not None and self.config.use_change_detection) else "simple_threshold",
                "cloud_masking_applied": cloud_mask is not None and self.config.cloud_mask_enabled,
                "shadow_masking_applied": shadow_mask is not None and self.config.shadow_mask_enabled
            }
        }

        logger.info(f"Detection complete: {flood_area_ha:.2f} ha flood extent "
                   f"({statistics['flood_percent']:.1f}% of valid area)")
        logger.info(f"Mean NDWI: {statistics['mean_ndwi']:.3f}, "
                   f"Flood NDWI: {statistics['mean_ndwi_flood']:.3f}")

        return NDWIOpticalResult(
            flood_extent=flood_extent,
            confidence_raster=confidence,
            ndwi_raster=ndwi_post,
            metadata=metadata,
            statistics=statistics
        )

    def _normalize_reflectance(self, band: np.ndarray) -> np.ndarray:
        """Normalize reflectance values to 0-1 range."""
        if np.nanmax(band) > 1.0:
            # Assume DN values (0-10000 scale)
            return band / 10000.0
        return band

    def _create_valid_mask(
        self,
        green_band: np.ndarray,
        nir_band: np.ndarray,
        cloud_mask: Optional[np.ndarray],
        shadow_mask: Optional[np.ndarray],
        nodata_value: Optional[float]
    ) -> np.ndarray:
        """Create mask of valid pixels."""
        valid_mask = np.ones_like(green_band, dtype=bool)

        # Mask NoData
        if nodata_value is not None:
            valid_mask &= (green_band != nodata_value)
            valid_mask &= (nir_band != nodata_value)

        # Mask NaN/Inf
        valid_mask &= np.isfinite(green_band)
        valid_mask &= np.isfinite(nir_band)

        # Mask clouds
        if cloud_mask is not None and self.config.cloud_mask_enabled:
            valid_mask &= ~cloud_mask
            logger.info(f"Cloud masking removed {np.sum(cloud_mask)} pixels")

        # Mask shadows
        if shadow_mask is not None and self.config.shadow_mask_enabled:
            valid_mask &= ~shadow_mask
            logger.info(f"Shadow masking removed {np.sum(shadow_mask)} pixels")

        return valid_mask

    def _calculate_ndwi(
        self,
        green: np.ndarray,
        nir: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate NDWI: (Green - NIR) / (Green + NIR)

        Args:
            green: Green band reflectance (0-1)
            nir: NIR band reflectance (0-1)
            valid_mask: Valid data mask

        Returns:
            NDWI array (-1 to 1)
        """
        ndwi = np.full_like(green, np.nan, dtype=np.float32)

        # Calculate NDWI only for valid pixels
        denominator = green + nir
        valid_calc = valid_mask & (denominator > 1e-6)  # Avoid division by zero

        ndwi[valid_calc] = (green[valid_calc] - nir[valid_calc]) / denominator[valid_calc]

        # Clip to valid range
        ndwi = np.clip(ndwi, -1.0, 1.0)

        return ndwi

    def _detect_simple(
        self,
        ndwi: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple threshold-based detection.

        Args:
            ndwi: NDWI values
            valid_mask: Valid data mask

        Returns:
            Tuple of (flood_extent, confidence)
        """
        # Detect water where NDWI > threshold
        flood_extent = (ndwi > self.config.ndwi_threshold) & valid_mask

        # Calculate confidence based on distance from threshold
        # Higher NDWI = higher confidence
        confidence = np.zeros_like(ndwi, dtype=np.float32)

        # Normalize confidence: threshold -> 0.5, 1.0 -> 1.0
        confidence[valid_mask] = np.clip(
            0.5 + 0.5 * (ndwi[valid_mask] - self.config.ndwi_threshold) / (1.0 - self.config.ndwi_threshold),
            0.0,
            1.0
        )

        return flood_extent, confidence

    def _detect_with_change(
        self,
        ndwi_post: np.ndarray,
        ndwi_pre: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Change detection mode (pre/post comparison).

        Args:
            ndwi_post: Post-event NDWI
            ndwi_pre: Pre-event NDWI
            valid_mask: Valid data mask

        Returns:
            Tuple of (flood_extent, confidence)
        """
        if ndwi_pre.shape != ndwi_post.shape:
            raise ValueError(f"Pre/post NDWI shape mismatch: {ndwi_pre.shape} vs {ndwi_post.shape}")

        # Calculate NDWI change (increase indicates flooding)
        change = ndwi_post - ndwi_pre

        # Flood detection: post > threshold AND significant increase
        threshold_met = ndwi_post > self.config.ndwi_threshold
        change_met = change > self.config.change_threshold
        flood_extent = threshold_met & change_met & valid_mask

        # Confidence based on both absolute value and change magnitude
        confidence = np.zeros_like(ndwi_post, dtype=np.float32)

        # Combine threshold confidence and change confidence
        threshold_conf = np.clip(
            0.5 + 0.5 * (ndwi_post - self.config.ndwi_threshold) / (1.0 - self.config.ndwi_threshold),
            0.0, 1.0
        )
        change_conf = np.clip(
            change / (self.config.change_threshold * 2.0),
            0.0, 1.0
        )

        # Average both confidence measures
        confidence[valid_mask] = (threshold_conf[valid_mask] + change_conf[valid_mask]) / 2.0

        return flood_extent, confidence

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return NDWIOpticalAlgorithm.METADATA

    def process_tile(
        self,
        tile_data: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Process a single tile for memory-efficient execution.

        This method enables tiled/windowed processing of large optical datasets.
        The tile should contain stacked green and NIR bands.

        Args:
            tile_data: Stacked optical tile, shape (2, H, W) where:
                - tile_data[0] = green band
                - tile_data[1] = NIR band
                OR shape (H, W) with pre-computed NDWI values
            context: Optional context dictionary containing:
                - nodata_value: Value to treat as nodata
                - pixel_size_m: Pixel size in meters
                - return_confidence: If True, return confidence instead of binary mask
                - return_ndwi: If True, return NDWI values instead of binary mask
                - is_ndwi: If True, input is pre-computed NDWI (not green/NIR bands)

        Returns:
            Flood extent mask for the tile (binary), or confidence/NDWI raster
            based on context flags
        """
        context = context or {}
        nodata_value = context.get("nodata_value")
        return_confidence = context.get("return_confidence", False)
        return_ndwi = context.get("return_ndwi", False)
        is_ndwi = context.get("is_ndwi", False)

        # Handle pre-computed NDWI input
        if is_ndwi or tile_data.ndim == 2:
            if tile_data.ndim == 3 and tile_data.shape[0] == 2:
                # It's green/NIR bands - compute NDWI
                green_band = self._normalize_reflectance(tile_data[0])
                nir_band = self._normalize_reflectance(tile_data[1])
                valid_mask = np.isfinite(green_band) & np.isfinite(nir_band)
                if nodata_value is not None:
                    valid_mask &= (green_band != nodata_value) & (nir_band != nodata_value)
                ndwi = self._calculate_ndwi(green_band, nir_band, valid_mask)
            else:
                # Pre-computed NDWI
                ndwi = tile_data
                valid_mask = np.isfinite(ndwi)
                if nodata_value is not None:
                    valid_mask &= (ndwi != nodata_value)
        else:
            # Extract green and NIR bands
            if tile_data.shape[0] != 2:
                raise ValueError(
                    f"Expected tile_data shape (2, H, W) for green/NIR bands, "
                    f"got {tile_data.shape}"
                )
            green_band = self._normalize_reflectance(tile_data[0])
            nir_band = self._normalize_reflectance(tile_data[1])

            # Create valid mask
            valid_mask = np.isfinite(green_band) & np.isfinite(nir_band)
            if nodata_value is not None:
                valid_mask &= (green_band != nodata_value) & (nir_band != nodata_value)

            # Calculate NDWI
            ndwi = self._calculate_ndwi(green_band, nir_band, valid_mask)

        if return_ndwi:
            return ndwi

        # Detect flood extent
        flood_extent, confidence = self._detect_simple(ndwi, valid_mask)

        if return_confidence:
            return confidence
        return flood_extent.astype(np.uint8)

    def compute_ndwi_windowed(
        self,
        green_band: np.ndarray,
        nir_band: np.ndarray,
        nodata_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute NDWI for a window/tile of optical data.

        This is a convenience method for windowed NDWI computation,
        useful when reading rasters in chunks.

        Args:
            green_band: Green band reflectance tile
            nir_band: NIR band reflectance tile
            nodata_value: NoData value to mask

        Returns:
            NDWI array for the tile (-1 to 1)
        """
        green_band = self._normalize_reflectance(green_band)
        nir_band = self._normalize_reflectance(nir_band)

        valid_mask = np.isfinite(green_band) & np.isfinite(nir_band)
        if nodata_value is not None:
            valid_mask &= (green_band != nodata_value) & (nir_band != nodata_value)

        return self._calculate_ndwi(green_band, nir_band, valid_mask)

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'NDWIOpticalAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = NDWIOpticalConfig(**params)
        return NDWIOpticalAlgorithm(config)
