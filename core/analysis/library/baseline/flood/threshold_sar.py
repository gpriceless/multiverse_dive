"""
SAR Backscatter Threshold Flood Detection Algorithm

Simple thresholding of SAR backscatter coefficient to detect standing water.
Reliable, interpretable, and well-validated baseline approach.

Algorithm ID: flood.baseline.threshold_sar
Version: 1.2.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSARConfig:
    """Configuration for SAR threshold flood detection."""

    threshold_db: float = -15.0  # Backscatter threshold for water classification (dB)
    min_area_ha: float = 0.5  # Minimum flood polygon area (hectares)
    polarization: str = "VV"  # Preferred polarization (VV or VH)
    use_change_detection: bool = False  # Enable pre/post change detection
    change_threshold_db: float = 3.0  # Change detection threshold (dB)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not -20.0 <= self.threshold_db <= -10.0:
            raise ValueError(f"threshold_db must be in [-20.0, -10.0], got {self.threshold_db}")
        if self.min_area_ha < 0:
            raise ValueError(f"min_area_ha must be non-negative, got {self.min_area_ha}")
        if self.polarization not in ["VV", "VH"]:
            raise ValueError(f"polarization must be VV or VH, got {self.polarization}")


@dataclass
class ThresholdSARResult:
    """Results from SAR threshold flood detection."""

    flood_extent: np.ndarray  # Binary mask of flood extent
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, float]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "flood_extent": self.flood_extent,
            "confidence_raster": self.confidence_raster,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class ThresholdSARAlgorithm:
    """
    SAR Backscatter Threshold Flood Detection.

    This algorithm detects flood extent by thresholding SAR backscatter values.
    Water surfaces appear dark in SAR imagery due to specular reflection,
    resulting in low backscatter coefficients (typically < -15 dB).

    Requirements:
        - SAR imagery (VV or VH polarization)
        - Optional: Pre-event SAR for change detection

    Outputs:
        - flood_extent: Binary vector polygon of flood extent
        - confidence_raster: Per-pixel confidence (0-1)
    """

    METADATA = {
        "id": "flood.baseline.threshold_sar",
        "name": "SAR Backscatter Threshold",
        "category": "baseline",
        "event_types": ["flood.*"],
        "version": "1.2.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "sar": {"polarization": ["VV", "VH"], "temporal": "post_event"}
            },
            "optional": {
                "sar_pre": {"temporal": "pre_event", "benefit": "enables_change_detection"}
            },
            "compute": {"memory_gb": 4, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.75, 0.90],
            "validated_regions": ["north_america", "europe", "southeast_asia"],
            "citations": ["doi:10.1016/j.rse.2019.111489"]
        }
    }

    def __init__(self, config: Optional[ThresholdSARConfig] = None):
        """
        Initialize SAR threshold algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or ThresholdSARConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: threshold={self.config.threshold_db}dB, "
                   f"min_area={self.config.min_area_ha}ha")

    def execute(
        self,
        sar_post: np.ndarray,
        sar_pre: Optional[np.ndarray] = None,
        pixel_size_m: float = 10.0,
        nodata_value: Optional[float] = None
    ) -> ThresholdSARResult:
        """
        Execute SAR threshold flood detection.

        Args:
            sar_post: Post-event SAR backscatter (dB), shape (H, W)
            sar_pre: Optional pre-event SAR backscatter (dB), shape (H, W)
            pixel_size_m: Pixel size in meters (for area calculation)
            nodata_value: NoData value to mask out

        Returns:
            ThresholdSARResult containing flood extent and confidence
        """
        logger.info("Starting SAR threshold flood detection")

        # Validate inputs
        if sar_post.ndim != 2:
            raise ValueError(f"Expected 2D SAR array, got shape {sar_post.shape}")

        # Create valid data mask
        valid_mask = np.ones_like(sar_post, dtype=bool)
        if nodata_value is not None:
            valid_mask &= (sar_post != nodata_value)
        valid_mask &= np.isfinite(sar_post)

        # Apply threshold detection
        if sar_pre is not None and self.config.use_change_detection:
            logger.info("Using change detection mode")
            flood_extent, confidence = self._detect_with_change(
                sar_post, sar_pre, valid_mask
            )
        else:
            logger.info("Using simple threshold mode")
            flood_extent, confidence = self._detect_simple(sar_post, valid_mask)

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
            "mean_confidence": float(np.mean(confidence[flood_extent])) if flood_pixels > 0 else 0.0
        }

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "threshold_db": self.config.threshold_db,
                "min_area_ha": self.config.min_area_ha,
                "polarization": self.config.polarization,
                "use_change_detection": self.config.use_change_detection,
                "pixel_size_m": pixel_size_m
            },
            "execution": {
                "mode": "change_detection" if (sar_pre is not None and self.config.use_change_detection) else "simple_threshold"
            }
        }

        logger.info(f"Detection complete: {flood_area_ha:.2f} ha flood extent "
                   f"({statistics['flood_percent']:.1f}% of valid area)")

        return ThresholdSARResult(
            flood_extent=flood_extent,
            confidence_raster=confidence,
            metadata=metadata,
            statistics=statistics
        )

    def _detect_simple(
        self,
        sar: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple threshold-based detection.

        Args:
            sar: SAR backscatter (dB)
            valid_mask: Valid data mask

        Returns:
            Tuple of (flood_extent, confidence)
        """
        # Detect water where backscatter < threshold
        flood_extent = (sar < self.config.threshold_db) & valid_mask

        # Calculate confidence based on distance from threshold
        # Stronger signal (more negative) = higher confidence
        confidence = np.zeros_like(sar, dtype=np.float32)

        # Normalize confidence: 0 dB -> 0.0, threshold -> 0.5, very dark -> 1.0
        confidence[valid_mask] = np.clip(
            (self.config.threshold_db - sar[valid_mask]) / abs(self.config.threshold_db),
            0.0,
            1.0
        )

        return flood_extent, confidence

    def _detect_with_change(
        self,
        sar_post: np.ndarray,
        sar_pre: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Change detection mode (pre/post comparison).

        Args:
            sar_post: Post-event SAR backscatter (dB)
            sar_pre: Pre-event SAR backscatter (dB)
            valid_mask: Valid data mask

        Returns:
            Tuple of (flood_extent, confidence)
        """
        if sar_pre.shape != sar_post.shape:
            raise ValueError(f"Pre/post SAR shape mismatch: {sar_pre.shape} vs {sar_post.shape}")

        # Calculate backscatter change (decrease indicates flooding)
        change_db = sar_pre - sar_post

        # Flood detection: post < threshold AND significant decrease
        threshold_met = sar_post < self.config.threshold_db
        change_met = change_db > self.config.change_threshold_db
        flood_extent = threshold_met & change_met & valid_mask

        # Confidence based on both absolute value and change magnitude
        confidence = np.zeros_like(sar_post, dtype=np.float32)

        # Combine threshold confidence and change confidence
        threshold_conf = np.clip(
            (self.config.threshold_db - sar_post) / abs(self.config.threshold_db),
            0.0, 1.0
        )
        change_conf = np.clip(
            change_db / (self.config.change_threshold_db * 3.0),
            0.0, 1.0
        )

        # Average both confidence measures
        confidence[valid_mask] = (threshold_conf[valid_mask] + change_conf[valid_mask]) / 2.0

        return flood_extent, confidence

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return ThresholdSARAlgorithm.METADATA

    def process_tile(
        self,
        tile_data: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Process a single tile for memory-efficient execution.

        This method enables tiled processing of large datasets that don't fit
        in memory. The tile should contain SAR backscatter values in dB.

        Args:
            tile_data: SAR backscatter tile (dB), shape (H, W)
            context: Optional context dictionary containing:
                - nodata_value: Value to treat as nodata
                - pixel_size_m: Pixel size in meters
                - return_confidence: If True, return confidence instead of binary mask

        Returns:
            Flood extent mask for the tile (binary), or confidence raster if
            return_confidence=True in context
        """
        context = context or {}
        nodata_value = context.get("nodata_value")
        return_confidence = context.get("return_confidence", False)

        # Create valid data mask
        valid_mask = np.ones_like(tile_data, dtype=bool)
        if nodata_value is not None:
            valid_mask &= (tile_data != nodata_value)
        valid_mask &= np.isfinite(tile_data)

        # Detect flood extent
        flood_extent, confidence = self._detect_simple(tile_data, valid_mask)

        if return_confidence:
            return confidence
        return flood_extent.astype(np.uint8)

    def detect(
        self,
        sar_data: np.ndarray,
        nodata_value: Optional[float] = None,
        pixel_size_m: float = 10.0
    ) -> ThresholdSARResult:
        """
        Simplified detection method for tile processing compatibility.

        This is an alias for execute() with sensible defaults, designed
        for use in tiled processing pipelines.

        Args:
            sar_data: SAR backscatter array (dB)
            nodata_value: NoData value to mask
            pixel_size_m: Pixel size in meters

        Returns:
            ThresholdSARResult with flood extent and confidence
        """
        return self.execute(sar_data, pixel_size_m=pixel_size_m, nodata_value=nodata_value)

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'ThresholdSARAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = ThresholdSARConfig(**params)
        return ThresholdSARAlgorithm(config)
