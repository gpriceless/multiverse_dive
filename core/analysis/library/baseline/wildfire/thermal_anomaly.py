"""
Thermal Anomaly Detection Algorithm for Active Fire Detection

Active fire detection using thermal infrared bands.
Identifies high-temperature anomalies indicative of active burning.
Works day and night, through smoke.

Algorithm ID: wildfire.baseline.thermal_anomaly
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Fire detection constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
PLANCK_C1 = 1.191042e8  # W·µm⁴/m²/sr
PLANCK_C2 = 1.4387774e4  # µm·K


@dataclass
class ThermalAnomalyConfig:
    """Configuration for thermal anomaly fire detection."""

    temperature_threshold_k: float = 320.0  # Minimum brightness temperature for fire (Kelvin)
    background_window_size: int = 21  # Window size for background calculation
    min_temperature_delta_k: float = 10.0  # Min difference from background
    contextual_algorithm: bool = True  # Use contextual detection (MODIS/VIIRS style)
    confidence_high_threshold_k: float = 360.0  # High confidence threshold
    confidence_low_threshold_k: float = 330.0  # Low confidence threshold
    min_fire_size_pixels: int = 1  # Minimum fire cluster size
    frp_calculation: bool = True  # Calculate Fire Radiative Power

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature_threshold_k < 273:
            raise ValueError(f"temperature_threshold_k must be >= 273K, got {self.temperature_threshold_k}")
        if self.background_window_size < 3 or self.background_window_size % 2 == 0:
            raise ValueError(f"background_window_size must be odd and >= 3, got {self.background_window_size}")
        if self.min_temperature_delta_k < 0:
            raise ValueError(f"min_temperature_delta_k must be non-negative, got {self.min_temperature_delta_k}")


@dataclass
class ThermalAnomalyResult:
    """Results from thermal anomaly fire detection."""

    active_fires: np.ndarray  # Binary mask of detected fires
    fire_radiative_power: np.ndarray  # FRP values (MW) or zeros if not calculated
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    brightness_temp: np.ndarray  # Brightness temperature map (K)
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, Any]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "active_fires": self.active_fires,
            "fire_radiative_power": self.fire_radiative_power,
            "confidence_raster": self.confidence_raster,
            "brightness_temp": self.brightness_temp,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class ThermalAnomalyAlgorithm:
    """
    Thermal Anomaly Detection for Active Fires.

    This algorithm detects active fires by identifying thermal anomalies
    in infrared imagery. It implements a contextual algorithm inspired by
    the MODIS Collection 6 and VIIRS active fire products.

    The algorithm:
    1. Calculates brightness temperature from thermal band(s)
    2. Identifies potential fire pixels exceeding threshold
    3. Applies contextual tests comparing to background
    4. Calculates Fire Radiative Power (FRP) for detected fires
    5. Assigns confidence based on temperature and context

    Requirements:
        - Thermal infrared band (MIR ~4µm or TIR ~11µm)
        - Optional: Second thermal band for improved detection

    Outputs:
        - active_fires: Binary mask of fire detections
        - fire_radiative_power: Per-pixel FRP in MW
        - confidence_raster: Detection confidence (0-1)
    """

    METADATA = {
        "id": "wildfire.baseline.thermal_anomaly",
        "name": "Thermal Anomaly Detection",
        "category": "baseline",
        "event_types": ["wildfire.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "thermal": {
                    "bands": ["thermal_mir", "thermal_tir"],
                    "description": "Mid-infrared (~4µm) preferred, thermal infrared (~11µm) also accepted"
                }
            },
            "optional": {
                "solar_zenith": {"benefit": "day_night_distinction"},
                "land_mask": {"benefit": "exclude_water"}
            },
            "compute": {"memory_gb": 0.5, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.78, 0.92],
            "validated_regions": ["global"],
            "citations": [
                "Giglio et al. (2016) - MODIS Collection 6",
                "Schroeder et al. (2014) - VIIRS active fire"
            ]
        }
    }

    def __init__(self, config: Optional[ThermalAnomalyConfig] = None):
        """
        Initialize thermal anomaly algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or ThermalAnomalyConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: threshold={self.config.temperature_threshold_k}K, "
                   f"contextual={self.config.contextual_algorithm}")

    def execute(
        self,
        thermal_band: np.ndarray,
        thermal_band_wavelength_um: float = 4.0,
        pixel_size_m: float = 375.0,
        water_mask: Optional[np.ndarray] = None,
        solar_zenith: Optional[np.ndarray] = None,
        nodata_value: Optional[float] = None
    ) -> ThermalAnomalyResult:
        """
        Execute thermal anomaly fire detection.

        Args:
            thermal_band: Thermal radiance or brightness temperature, shape (H, W)
                         If values < 200, assumed to be radiance (W/m²/sr/µm)
                         If values > 200, assumed to be brightness temperature (K)
            thermal_band_wavelength_um: Central wavelength in micrometers
            pixel_size_m: Pixel size in meters (for FRP calculation)
            water_mask: Optional water mask (True = water, exclude)
            solar_zenith: Optional solar zenith angle (degrees)
            nodata_value: NoData value to mask out

        Returns:
            ThermalAnomalyResult containing fire detections and FRP
        """
        logger.info("Starting thermal anomaly fire detection")

        # Validate inputs
        if thermal_band.ndim != 2:
            raise ValueError(f"Expected 2D thermal array, got shape {thermal_band.shape}")

        # Create valid data mask
        valid_mask = np.ones_like(thermal_band, dtype=bool)
        if nodata_value is not None:
            valid_mask &= (thermal_band != nodata_value)
        valid_mask &= np.isfinite(thermal_band)

        if water_mask is not None:
            valid_mask &= ~water_mask

        # Convert to brightness temperature if needed
        if np.nanmax(thermal_band[valid_mask]) < 200:
            logger.info("Converting radiance to brightness temperature")
            brightness_temp = self._radiance_to_brightness_temp(
                thermal_band, thermal_band_wavelength_um, valid_mask
            )
        else:
            logger.info("Input is already brightness temperature")
            brightness_temp = thermal_band.astype(np.float32)

        # Detect fires
        if self.config.contextual_algorithm:
            logger.info("Using contextual detection algorithm")
            active_fires, confidence = self._detect_contextual(brightness_temp, valid_mask)
        else:
            logger.info("Using simple threshold detection")
            active_fires, confidence = self._detect_simple(brightness_temp, valid_mask)

        # Calculate Fire Radiative Power
        if self.config.frp_calculation:
            frp = self._calculate_frp(
                brightness_temp, active_fires, pixel_size_m, thermal_band_wavelength_um
            )
        else:
            frp = np.zeros_like(brightness_temp, dtype=np.float32)

        # Calculate statistics
        statistics = self._calculate_statistics(
            active_fires, frp, brightness_temp, valid_mask, pixel_size_m
        )

        # Determine if day or night
        is_daytime = None
        if solar_zenith is not None:
            is_daytime = np.mean(solar_zenith[valid_mask]) < 85.0

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "temperature_threshold_k": self.config.temperature_threshold_k,
                "background_window_size": self.config.background_window_size,
                "min_temperature_delta_k": self.config.min_temperature_delta_k,
                "contextual_algorithm": self.config.contextual_algorithm,
                "frp_calculation": self.config.frp_calculation,
                "pixel_size_m": pixel_size_m,
                "wavelength_um": thermal_band_wavelength_um
            },
            "execution": {
                "is_daytime": is_daytime
            }
        }

        logger.info(f"Detection complete: {statistics['fire_count']} fire pixels, "
                   f"total FRP: {statistics['total_frp_mw']:.2f} MW")

        return ThermalAnomalyResult(
            active_fires=active_fires,
            fire_radiative_power=frp,
            confidence_raster=confidence,
            brightness_temp=brightness_temp,
            metadata=metadata,
            statistics=statistics
        )

    def _radiance_to_brightness_temp(
        self,
        radiance: np.ndarray,
        wavelength_um: float,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Convert spectral radiance to brightness temperature.

        Uses inverse Planck function.

        Args:
            radiance: Spectral radiance (W/m²/sr/µm)
            wavelength_um: Central wavelength (µm)
            valid_mask: Valid data mask

        Returns:
            Brightness temperature (K)
        """
        brightness_temp = np.zeros_like(radiance, dtype=np.float32)

        # Planck function inversion: T = c2 / (λ * ln(c1 / (λ^5 * L) + 1))
        valid_radiance = (radiance > 0) & valid_mask

        if np.any(valid_radiance):
            c1_term = PLANCK_C1 / (wavelength_um ** 5 * radiance[valid_radiance])
            brightness_temp[valid_radiance] = PLANCK_C2 / (
                wavelength_um * np.log(c1_term + 1)
            )

        return brightness_temp

    def _detect_simple(
        self,
        brightness_temp: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple threshold-based fire detection.

        Args:
            brightness_temp: Brightness temperature (K)
            valid_mask: Valid data mask

        Returns:
            Tuple of (fire_mask, confidence)
        """
        # Threshold detection
        active_fires = (brightness_temp > self.config.temperature_threshold_k) & valid_mask

        # Confidence based on temperature
        confidence = np.zeros_like(brightness_temp, dtype=np.float32)

        fire_pixels = active_fires & valid_mask
        if np.any(fire_pixels):
            # Normalize between low and high confidence thresholds
            temp_range = self.config.confidence_high_threshold_k - self.config.confidence_low_threshold_k
            if temp_range > 0:
                confidence[fire_pixels] = np.clip(
                    (brightness_temp[fire_pixels] - self.config.confidence_low_threshold_k) / temp_range,
                    0.0, 1.0
                )

        return active_fires, confidence

    def _detect_contextual(
        self,
        brightness_temp: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Contextual fire detection algorithm.

        Implements a simplified version of MODIS/VIIRS contextual algorithm:
        1. Absolute threshold test
        2. Background temperature comparison
        3. Contextual tests against neighboring pixels

        Args:
            brightness_temp: Brightness temperature (K)
            valid_mask: Valid data mask

        Returns:
            Tuple of (fire_mask, confidence)
        """
        h, w = brightness_temp.shape
        half_win = self.config.background_window_size // 2

        # Initial threshold test
        potential_fires = (brightness_temp > self.config.temperature_threshold_k) & valid_mask

        # Calculate background statistics for each pixel
        background_mean = np.zeros_like(brightness_temp, dtype=np.float32)
        background_std = np.zeros_like(brightness_temp, dtype=np.float32)

        # Pad array for window operations
        padded = np.pad(
            brightness_temp,
            pad_width=half_win,
            mode='reflect'
        )
        padded_valid = np.pad(
            valid_mask.astype(np.float32),
            pad_width=half_win,
            mode='constant',
            constant_values=0
        )

        # Calculate background statistics using sliding window
        for i in range(h):
            for j in range(w):
                if not valid_mask[i, j]:
                    continue

                # Extract window
                window = padded[i:i + 2*half_win + 1, j:j + 2*half_win + 1]
                window_valid = padded_valid[i:i + 2*half_win + 1, j:j + 2*half_win + 1]

                # Exclude center pixel and any potential fire pixels from background
                center_i, center_j = half_win, half_win
                window_valid[center_i, center_j] = 0

                # Exclude hot pixels from background
                hot_threshold = self.config.temperature_threshold_k - 10
                background_pixels = (window < hot_threshold) & (window_valid > 0.5)

                if np.sum(background_pixels) >= 3:
                    background_mean[i, j] = np.mean(window[background_pixels])
                    background_std[i, j] = np.std(window[background_pixels])

        # Apply contextual tests
        # Test 1: Must exceed absolute threshold (already done)
        # Test 2: Must exceed background by delta
        delta_test = (brightness_temp - background_mean) > self.config.min_temperature_delta_k

        # Test 3: Must be significantly above background (3-sigma)
        sigma_test = (brightness_temp - background_mean) > (3 * background_std)

        # Combine tests
        active_fires = potential_fires & delta_test & sigma_test & valid_mask

        # Calculate confidence
        confidence = np.zeros_like(brightness_temp, dtype=np.float32)

        fire_pixels = active_fires
        if np.any(fire_pixels):
            # Combine temperature-based and context-based confidence
            temp_conf = np.clip(
                (brightness_temp[fire_pixels] - self.config.confidence_low_threshold_k) /
                (self.config.confidence_high_threshold_k - self.config.confidence_low_threshold_k),
                0.0, 1.0
            )

            delta_conf = np.clip(
                (brightness_temp[fire_pixels] - background_mean[fire_pixels]) /
                (self.config.min_temperature_delta_k * 5),
                0.0, 1.0
            )

            confidence[fire_pixels] = (temp_conf + delta_conf) / 2

        return active_fires, confidence

    def _calculate_frp(
        self,
        brightness_temp: np.ndarray,
        active_fires: np.ndarray,
        pixel_size_m: float,
        wavelength_um: float
    ) -> np.ndarray:
        """
        Calculate Fire Radiative Power (FRP).

        Uses the MIR method based on Wooster et al. (2003).

        Args:
            brightness_temp: Brightness temperature (K)
            active_fires: Fire mask
            pixel_size_m: Pixel size in meters
            wavelength_um: Wavelength in micrometers

        Returns:
            FRP in megawatts per pixel
        """
        frp = np.zeros_like(brightness_temp, dtype=np.float32)

        if not np.any(active_fires):
            return frp

        # Simplified FRP calculation (Wooster et al. 2003)
        # FRP ≈ A_pixel * σ * (T_fire^4 - T_bg^4) * ε
        # where A_pixel is pixel area, σ is Stefan-Boltzmann constant

        pixel_area_m2 = pixel_size_m ** 2
        emissivity = 0.98  # Typical fire emissivity

        # Estimate background temperature (median of non-fire pixels)
        non_fire = ~active_fires & np.isfinite(brightness_temp)
        if np.any(non_fire):
            t_background = np.median(brightness_temp[non_fire])
        else:
            t_background = 300.0  # Default background

        # Calculate FRP for fire pixels
        fire_temps = brightness_temp[active_fires]
        frp_values = (
            pixel_area_m2 * STEFAN_BOLTZMANN * emissivity *
            (fire_temps ** 4 - t_background ** 4)
        )

        # Convert to MW
        frp_values = frp_values / 1e6

        # Ensure non-negative
        frp_values = np.maximum(frp_values, 0)

        frp[active_fires] = frp_values

        return frp

    def _calculate_statistics(
        self,
        active_fires: np.ndarray,
        frp: np.ndarray,
        brightness_temp: np.ndarray,
        valid_mask: np.ndarray,
        pixel_size_m: float
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""

        fire_count = int(np.sum(active_fires))
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0

        stats = {
            "total_pixels": int(valid_mask.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "fire_count": fire_count,
            "fire_area_ha": float(fire_count * pixel_area_ha),
            "total_frp_mw": float(np.sum(frp)),
            "mean_frp_mw": float(np.mean(frp[active_fires])) if fire_count > 0 else 0.0,
            "max_frp_mw": float(np.max(frp)) if fire_count > 0 else 0.0,
            "max_temperature_k": float(np.max(brightness_temp[active_fires])) if fire_count > 0 else 0.0,
            "mean_fire_temperature_k": float(np.mean(brightness_temp[active_fires])) if fire_count > 0 else 0.0,
            "mean_background_k": float(np.mean(brightness_temp[valid_mask & ~active_fires])) if np.any(valid_mask & ~active_fires) else 0.0
        }

        return stats

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return ThermalAnomalyAlgorithm.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'ThermalAnomalyAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = ThermalAnomalyConfig(**params)
        return ThermalAnomalyAlgorithm(config)
