"""
Differenced Normalized Burn Ratio (dNBR) Algorithm

Gold standard for burn severity mapping using optical imagery.
Compares NBR before and after fire event to quantify burn severity.

Algorithm ID: wildfire.baseline.nbr_differenced
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# USGS/Key & Benson severity thresholds (standard values)
DEFAULT_SEVERITY_THRESHOLDS = {
    "high": 0.66,
    "moderate_high": 0.44,
    "moderate_low": 0.27,
    "low": 0.10,
    "unburned": -0.10,
}


@dataclass
class DifferencedNBRConfig:
    """Configuration for Differenced NBR burn severity mapping."""

    high_severity_threshold: float = 0.66
    moderate_high_threshold: float = 0.44
    moderate_low_threshold: float = 0.27
    low_severity_threshold: float = 0.10
    unburned_threshold: float = -0.10
    min_burn_area_ha: float = 1.0
    cloud_mask_enabled: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (self.high_severity_threshold > self.moderate_high_threshold
                > self.moderate_low_threshold > self.low_severity_threshold):
            raise ValueError("Severity thresholds must be in descending order: "
                           f"high ({self.high_severity_threshold}) > moderate_high ({self.moderate_high_threshold}) "
                           f"> moderate_low ({self.moderate_low_threshold}) > low ({self.low_severity_threshold})")
        if self.min_burn_area_ha < 0:
            raise ValueError(f"min_burn_area_ha must be non-negative, got {self.min_burn_area_ha}")


@dataclass
class DifferencedNBRResult:
    """Results from Differenced NBR analysis."""

    burn_extent: np.ndarray  # Binary mask of burned area
    burn_severity: np.ndarray  # Classified severity (0-5)
    dnbr_map: np.ndarray  # Continuous dNBR values
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, Any]  # Summary statistics by severity class

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "burn_extent": self.burn_extent,
            "burn_severity": self.burn_severity,
            "dnbr_map": self.dnbr_map,
            "confidence_raster": self.confidence_raster,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class DifferencedNBRAlgorithm:
    """
    Differenced Normalized Burn Ratio (dNBR) for burn severity mapping.

    The NBR is calculated as: NBR = (NIR - SWIR) / (NIR + SWIR)
    The dNBR is: dNBR = NBR_pre - NBR_post

    Positive dNBR values indicate burn severity (higher = more severe).
    Standard USGS severity thresholds are applied for classification.

    Requirements:
        - Pre-fire optical imagery with NIR and SWIR bands
        - Post-fire optical imagery with NIR and SWIR bands
        - Cloud-free conditions preferred

    Outputs:
        - burn_extent: Binary vector polygon of burned area
        - burn_severity: Classified severity map (0-5)
        - dnbr_map: Continuous dNBR values
    """

    # Severity class names
    SEVERITY_CLASSES = {
        0: "unburned",
        1: "enhanced_regrowth",
        2: "low",
        3: "moderate_low",
        4: "moderate_high",
        5: "high"
    }

    METADATA = {
        "id": "wildfire.baseline.nbr_differenced",
        "name": "Differenced Normalized Burn Ratio",
        "category": "baseline",
        "event_types": ["wildfire.*", "fire.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "optical_pre": {
                    "bands": ["nir", "swir"],
                    "temporal": "pre_event"
                },
                "optical_post": {
                    "bands": ["nir", "swir"],
                    "temporal": "post_event"
                }
            },
            "optional": {
                "cloud_mask": {"benefit": "improved_accuracy"}
            },
            "compute": {"memory_gb": 2, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.85, 0.96],
            "validated_regions": ["global"],
            "citations": [
                "Key & Benson (2006) - Landscape Assessment",
                "Miller & Thode (2007) - NBR Quantification"
            ]
        }
    }

    def __init__(self, config: Optional[DifferencedNBRConfig] = None):
        """
        Initialize dNBR algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or DifferencedNBRConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: high_threshold={self.config.high_severity_threshold}, "
                   f"min_burn_area={self.config.min_burn_area_ha}ha")

    def execute(
        self,
        nir_pre: np.ndarray,
        swir_pre: np.ndarray,
        nir_post: np.ndarray,
        swir_post: np.ndarray,
        pixel_size_m: float = 30.0,
        cloud_mask: Optional[np.ndarray] = None,
        nodata_value: Optional[float] = None
    ) -> DifferencedNBRResult:
        """
        Execute dNBR burn severity analysis.

        Args:
            nir_pre: Pre-fire NIR band, shape (H, W)
            swir_pre: Pre-fire SWIR band, shape (H, W)
            nir_post: Post-fire NIR band, shape (H, W)
            swir_post: Post-fire SWIR band, shape (H, W)
            pixel_size_m: Pixel size in meters
            cloud_mask: Optional cloud mask (True = cloudy/invalid)
            nodata_value: NoData value to mask out

        Returns:
            DifferencedNBRResult containing burn extent, severity, and dNBR map
        """
        logger.info("Starting Differenced NBR burn severity analysis")

        # Validate inputs
        shapes = [nir_pre.shape, swir_pre.shape, nir_post.shape, swir_post.shape]
        if len(set(shapes)) != 1:
            raise ValueError(f"All input bands must have same shape. Got: {shapes}")

        if nir_pre.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got shape {nir_pre.shape}")

        # Create valid data mask
        valid_mask = np.ones_like(nir_pre, dtype=bool)

        for arr in [nir_pre, swir_pre, nir_post, swir_post]:
            valid_mask &= np.isfinite(arr)
            if nodata_value is not None:
                valid_mask &= (arr != nodata_value)

        if cloud_mask is not None and self.config.cloud_mask_enabled:
            valid_mask &= ~cloud_mask

        # Calculate NBR for pre and post
        nbr_pre = self._calculate_nbr(nir_pre, swir_pre, valid_mask)
        nbr_post = self._calculate_nbr(nir_post, swir_post, valid_mask)

        # Calculate dNBR
        dnbr = np.zeros_like(nir_pre, dtype=np.float32)
        dnbr[valid_mask] = nbr_pre[valid_mask] - nbr_post[valid_mask]

        # Classify severity
        burn_severity = self._classify_severity(dnbr, valid_mask)

        # Create burn extent (anything above unburned threshold)
        burn_extent = (dnbr > self.config.low_severity_threshold) & valid_mask

        # Calculate confidence
        confidence = self._calculate_confidence(dnbr, valid_mask)

        # Calculate statistics
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0
        statistics = self._calculate_statistics(
            burn_extent, burn_severity, dnbr, valid_mask, pixel_area_ha
        )

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "high_severity_threshold": self.config.high_severity_threshold,
                "moderate_high_threshold": self.config.moderate_high_threshold,
                "moderate_low_threshold": self.config.moderate_low_threshold,
                "low_severity_threshold": self.config.low_severity_threshold,
                "min_burn_area_ha": self.config.min_burn_area_ha,
                "pixel_size_m": pixel_size_m
            }
        }

        logger.info(f"Analysis complete: {statistics['total_burned_area_ha']:.2f} ha burned, "
                   f"{statistics['high_severity_percent']:.1f}% high severity")

        return DifferencedNBRResult(
            burn_extent=burn_extent,
            burn_severity=burn_severity,
            dnbr_map=dnbr,
            confidence_raster=confidence,
            metadata=metadata,
            statistics=statistics
        )

    def _calculate_nbr(
        self,
        nir: np.ndarray,
        swir: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Normalized Burn Ratio.

        NBR = (NIR - SWIR) / (NIR + SWIR)

        Args:
            nir: Near-infrared band
            swir: Shortwave infrared band
            valid_mask: Valid data mask

        Returns:
            NBR values (-1 to 1)
        """
        nbr = np.zeros_like(nir, dtype=np.float32)

        denominator = nir + swir
        valid_denom = (denominator != 0) & valid_mask

        nbr[valid_denom] = (nir[valid_denom] - swir[valid_denom]) / denominator[valid_denom]

        # Clip to valid range
        nbr = np.clip(nbr, -1.0, 1.0)

        return nbr

    def _classify_severity(
        self,
        dnbr: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Classify burn severity using USGS thresholds.

        Classes:
            0: Unburned or regrowth
            1: Enhanced regrowth (negative dNBR)
            2: Low severity
            3: Moderate-low severity
            4: Moderate-high severity
            5: High severity

        Args:
            dnbr: Differenced NBR values
            valid_mask: Valid data mask

        Returns:
            Severity classification (0-5)
        """
        severity = np.zeros_like(dnbr, dtype=np.uint8)

        # Enhanced regrowth (negative dNBR - more vegetation post-fire)
        severity[(dnbr < self.config.unburned_threshold) & valid_mask] = 1

        # Unburned
        severity[(dnbr >= self.config.unburned_threshold) &
                (dnbr < self.config.low_severity_threshold) & valid_mask] = 0

        # Low severity
        severity[(dnbr >= self.config.low_severity_threshold) &
                (dnbr < self.config.moderate_low_threshold) & valid_mask] = 2

        # Moderate-low severity
        severity[(dnbr >= self.config.moderate_low_threshold) &
                (dnbr < self.config.moderate_high_threshold) & valid_mask] = 3

        # Moderate-high severity
        severity[(dnbr >= self.config.moderate_high_threshold) &
                (dnbr < self.config.high_severity_threshold) & valid_mask] = 4

        # High severity
        severity[(dnbr >= self.config.high_severity_threshold) & valid_mask] = 5

        return severity

    def _calculate_confidence(
        self,
        dnbr: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate per-pixel confidence score.

        Confidence is higher when dNBR values are further from threshold boundaries.

        Args:
            dnbr: Differenced NBR values
            valid_mask: Valid data mask

        Returns:
            Confidence scores (0-1)
        """
        confidence = np.zeros_like(dnbr, dtype=np.float32)

        # For burned pixels, confidence based on how far above threshold
        burned_mask = (dnbr > self.config.low_severity_threshold) & valid_mask

        # Normalize: threshold -> 0.5, high severity threshold -> 1.0
        if np.any(burned_mask):
            range_val = self.config.high_severity_threshold - self.config.low_severity_threshold
            if range_val > 0:
                confidence[burned_mask] = np.clip(
                    0.5 + 0.5 * (dnbr[burned_mask] - self.config.low_severity_threshold) / range_val,
                    0.5, 1.0
                )

        # For unburned pixels, confidence based on how far below threshold
        unburned_mask = (dnbr <= self.config.low_severity_threshold) & valid_mask
        if np.any(unburned_mask):
            confidence[unburned_mask] = np.clip(
                0.5 + 0.5 * (self.config.low_severity_threshold - dnbr[unburned_mask]) / 0.3,
                0.0, 1.0
            )

        return confidence

    def _calculate_statistics(
        self,
        burn_extent: np.ndarray,
        burn_severity: np.ndarray,
        dnbr: np.ndarray,
        valid_mask: np.ndarray,
        pixel_area_ha: float
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""

        total_valid = np.sum(valid_mask)
        total_burned = np.sum(burn_extent)

        # Calculate area by severity class
        severity_areas = {}
        severity_counts = {}
        for class_id, class_name in self.SEVERITY_CLASSES.items():
            count = np.sum(burn_severity == class_id)
            severity_counts[class_name] = int(count)
            severity_areas[f"{class_name}_area_ha"] = float(count * pixel_area_ha)

        # Calculate percentages for burned area
        high_count = severity_counts.get("high", 0)
        moderate_high_count = severity_counts.get("moderate_high", 0)
        moderate_low_count = severity_counts.get("moderate_low", 0)

        return {
            "total_pixels": int(valid_mask.size),
            "valid_pixels": int(total_valid),
            "burned_pixels": int(total_burned),
            "total_burned_area_ha": float(total_burned * pixel_area_ha),
            "burned_percent": float(100.0 * total_burned / total_valid) if total_valid > 0 else 0.0,
            "high_severity_percent": float(100.0 * high_count / total_burned) if total_burned > 0 else 0.0,
            "moderate_severity_percent": float(100.0 * (moderate_high_count + moderate_low_count) / total_burned) if total_burned > 0 else 0.0,
            "mean_dnbr": float(np.mean(dnbr[valid_mask])) if total_valid > 0 else 0.0,
            "mean_dnbr_burned": float(np.mean(dnbr[burn_extent])) if total_burned > 0 else 0.0,
            "severity_counts": severity_counts,
            **severity_areas
        }

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return DifferencedNBRAlgorithm.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'DifferencedNBRAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = DifferencedNBRConfig(**params)
        return DifferencedNBRAlgorithm(config)
