"""
Terrain and Atmospheric Corrections for Multi-Sensor Fusion.

Provides tools for correcting geospatial data for physical effects including:
- Terrain correction (slope, aspect, shadowing)
- Atmospheric correction (scattering, absorption, haze)
- Radiometric normalization across sensors
- Cross-sensor calibration

Key Concepts:
- Terrain correction removes topographic distortions from SAR/optical data
- Atmospheric correction removes atmospheric effects from optical imagery
- Radiometric normalization ensures comparable values across sensors/dates
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TerrainCorrectionMethod(Enum):
    """Methods for terrain correction."""
    COSINE = "cosine"                    # Simple cosine correction
    MINNAERT = "minnaert"                # Minnaert correction with k factor
    C_CORRECTION = "c_correction"        # C correction (Teillet et al.)
    SCS = "scs"                          # Sun-Canopy-Sensor correction
    SCS_C = "scs_c"                      # SCS + C correction
    GAMMA = "gamma"                      # Gamma terrain correction for SAR
    FLAT_EARTH = "flat_earth"            # Flat Earth removal for SAR


class AtmosphericCorrectionMethod(Enum):
    """Methods for atmospheric correction."""
    DOS = "dos"                          # Dark Object Subtraction
    DOS_IMPROVED = "dos_improved"        # Improved DOS (Chavez)
    COST = "cost"                        # COST model
    TOAR = "toar"                        # Top of Atmosphere Reflectance
    FLAASH = "flaash"                    # Fast Line-of-sight Atm. Analysis
    SEN2COR = "sen2cor"                  # Sentinel-2 L2A processor style
    LUT_BASED = "lut_based"              # Lookup table based


class NormalizationMethod(Enum):
    """Methods for radiometric normalization."""
    HISTOGRAM_MATCHING = "histogram_matching"      # Match histograms
    PIF = "pif"                                   # Pseudo-Invariant Features
    RELATIVE = "relative"                         # Relative normalization
    ABSOLUTE = "absolute"                         # Absolute calibration
    IR_MAD = "ir_mad"                            # Iteratively Reweighted MAD


@dataclass
class TerrainCorrectionConfig:
    """
    Configuration for terrain correction.

    Attributes:
        method: Correction method to use
        dem: DEM array for terrain calculations
        dem_resolution: DEM resolution in meters
        sun_azimuth_deg: Sun azimuth angle in degrees (0=North, clockwise)
        sun_elevation_deg: Sun elevation angle in degrees above horizon
        sensor_azimuth_deg: Sensor azimuth for SAR corrections
        sensor_elevation_deg: Sensor elevation/incidence angle
        minnaert_k: Minnaert k factor (0-1, higher for rougher surfaces)
        slope_threshold_deg: Maximum slope for reliable correction
        shadow_detection: Enable shadow detection
    """
    method: TerrainCorrectionMethod = TerrainCorrectionMethod.COSINE
    dem: Optional[np.ndarray] = None
    dem_resolution: float = 30.0
    sun_azimuth_deg: float = 180.0
    sun_elevation_deg: float = 45.0
    sensor_azimuth_deg: float = 0.0
    sensor_elevation_deg: float = 90.0
    minnaert_k: float = 0.5
    slope_threshold_deg: float = 70.0
    shadow_detection: bool = True


@dataclass
class AtmosphericCorrectionConfig:
    """
    Configuration for atmospheric correction.

    Attributes:
        method: Correction method to use
        sensor_type: Type of sensor (sentinel2, landsat8, etc.)
        band_wavelengths: Center wavelengths per band (nm)
        solar_zenith_deg: Solar zenith angle in degrees
        solar_azimuth_deg: Solar azimuth angle in degrees
        view_zenith_deg: Sensor view zenith angle
        view_azimuth_deg: Sensor view azimuth angle
        ozone_content: Ozone column in DU
        water_vapor: Precipitable water vapor in cm
        aod550: Aerosol Optical Depth at 550nm
        altitude_m: Ground altitude in meters
        dark_object_threshold: Percentile for DOS dark object selection
    """
    method: AtmosphericCorrectionMethod = AtmosphericCorrectionMethod.DOS
    sensor_type: str = "generic"
    band_wavelengths: Optional[List[float]] = None
    solar_zenith_deg: float = 30.0
    solar_azimuth_deg: float = 180.0
    view_zenith_deg: float = 0.0
    view_azimuth_deg: float = 0.0
    ozone_content: float = 300.0
    water_vapor: float = 2.0
    aod550: float = 0.1
    altitude_m: float = 0.0
    dark_object_threshold: float = 1.0


@dataclass
class NormalizationConfig:
    """
    Configuration for radiometric normalization.

    Attributes:
        method: Normalization method to use
        reference_dataset_id: ID of reference dataset for relative norm
        use_pif: Use Pseudo-Invariant Features for normalization
        pif_threshold: Threshold for PIF selection
        histogram_match_percentiles: Percentiles for histogram matching
        clip_outliers: Clip extreme values
        outlier_sigma: Standard deviations for outlier clipping
    """
    method: NormalizationMethod = NormalizationMethod.HISTOGRAM_MATCHING
    reference_dataset_id: Optional[str] = None
    use_pif: bool = True
    pif_threshold: float = 0.95
    histogram_match_percentiles: Tuple[float, float] = (2.0, 98.0)
    clip_outliers: bool = True
    outlier_sigma: float = 3.0


@dataclass
class CorrectionResult:
    """
    Result from a correction operation.

    Attributes:
        corrected_data: Corrected data array
        correction_factor: Per-pixel correction factor applied
        quality_mask: Quality mask for corrected data
        correction_type: Type of correction applied
        parameters: Parameters used for correction
        diagnostics: Diagnostic information
    """
    corrected_data: np.ndarray
    correction_factor: Optional[np.ndarray] = None
    quality_mask: Optional[np.ndarray] = None
    correction_type: str = "unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "correction_type": self.correction_type,
            "data_shape": list(self.corrected_data.shape),
            "data_dtype": str(self.corrected_data.dtype),
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
        }


class TerrainCorrector:
    """
    Corrects imagery for terrain effects.

    Provides methods for:
    - Calculating slope and aspect from DEM
    - Computing illumination models
    - Applying terrain correction to optical/SAR data
    """

    def __init__(self, config: Optional[TerrainCorrectionConfig] = None):
        """
        Initialize terrain corrector.

        Args:
            config: Terrain correction configuration
        """
        self.config = config or TerrainCorrectionConfig()
        self._slope = None
        self._aspect = None
        self._illumination = None

    def correct(
        self,
        data: np.ndarray,
        dem: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """
        Apply terrain correction to data.

        Args:
            data: Input data array (2D for single band, 3D for multi-band)
            dem: DEM array (uses config DEM if not provided)

        Returns:
            CorrectionResult with terrain-corrected data
        """
        dem_data = dem if dem is not None else self.config.dem
        if dem_data is None:
            logger.warning("No DEM provided, skipping terrain correction")
            return CorrectionResult(
                corrected_data=data,
                correction_type="none",
                diagnostics={"reason": "no_dem"}
            )

        # Calculate terrain derivatives if not cached
        if self._slope is None or self._aspect is None:
            self._slope, self._aspect = self._calculate_slope_aspect(dem_data)

        # Calculate illumination
        self._illumination = self._calculate_illumination()

        # Apply correction based on method
        method = self.config.method

        if method == TerrainCorrectionMethod.COSINE:
            corrected, factor = self._apply_cosine_correction(data)
        elif method == TerrainCorrectionMethod.MINNAERT:
            corrected, factor = self._apply_minnaert_correction(data)
        elif method == TerrainCorrectionMethod.C_CORRECTION:
            corrected, factor = self._apply_c_correction(data)
        elif method == TerrainCorrectionMethod.SCS:
            corrected, factor = self._apply_scs_correction(data)
        elif method == TerrainCorrectionMethod.GAMMA:
            corrected, factor = self._apply_gamma_correction(data, dem_data)
        else:
            logger.warning(f"Method {method} not implemented, using cosine")
            corrected, factor = self._apply_cosine_correction(data)

        # Create quality mask
        quality_mask = self._create_quality_mask(factor)

        return CorrectionResult(
            corrected_data=corrected,
            correction_factor=factor,
            quality_mask=quality_mask,
            correction_type=f"terrain_{method.value}",
            parameters={
                "sun_azimuth_deg": self.config.sun_azimuth_deg,
                "sun_elevation_deg": self.config.sun_elevation_deg,
                "minnaert_k": self.config.minnaert_k if method == TerrainCorrectionMethod.MINNAERT else None,
            },
            diagnostics={
                "slope_mean_deg": float(np.nanmean(np.degrees(self._slope))),
                "slope_max_deg": float(np.nanmax(np.degrees(self._slope))),
                "illumination_mean": float(np.nanmean(self._illumination)),
                "correction_factor_mean": float(np.nanmean(factor)),
            }
        )

    def _calculate_slope_aspect(
        self,
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect from DEM.

        Returns slope in radians, aspect in radians from north (clockwise).
        """
        resolution = self.config.dem_resolution

        # Calculate gradients
        dy, dx = np.gradient(dem, resolution)

        # Slope in radians
        slope = np.arctan(np.sqrt(dx**2 + dy**2))

        # Aspect in radians (0=north, clockwise)
        aspect = np.arctan2(-dx, dy)  # Adjusted for north=0
        aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

        return slope, aspect

    def _calculate_illumination(self) -> np.ndarray:
        """
        Calculate illumination coefficient (cos(i)).

        Uses the slope-aspect DEM model.
        """
        # Convert angles to radians
        sun_zen = np.radians(90 - self.config.sun_elevation_deg)
        sun_az = np.radians(self.config.sun_azimuth_deg)

        # Illumination coefficient: cos(i) = cos(slope)*cos(sun_zen) + sin(slope)*sin(sun_zen)*cos(sun_az - aspect)
        cos_i = (
            np.cos(self._slope) * np.cos(sun_zen) +
            np.sin(self._slope) * np.sin(sun_zen) * np.cos(sun_az - self._aspect)
        )

        # Clamp to valid range
        cos_i = np.clip(cos_i, 0.001, 1.0)

        return cos_i

    def _apply_cosine_correction(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply simple cosine correction."""
        sun_zen = np.radians(90 - self.config.sun_elevation_deg)
        cos_sun_zen = np.cos(sun_zen)

        # Correction factor
        factor = cos_sun_zen / self._illumination

        # Apply correction
        if len(data.shape) == 3:
            # Multi-band
            corrected = data * factor[..., np.newaxis]
        else:
            corrected = data * factor

        return corrected.astype(data.dtype), factor

    def _apply_minnaert_correction(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Minnaert correction with k factor."""
        k = self.config.minnaert_k
        sun_zen = np.radians(90 - self.config.sun_elevation_deg)
        cos_sun_zen = np.cos(sun_zen)

        # Minnaert correction: L_corrected = L * (cos(sun_zen) / cos(i))^k
        factor = (cos_sun_zen / self._illumination) ** k

        if len(data.shape) == 3:
            corrected = data * factor[..., np.newaxis]
        else:
            corrected = data * factor

        return corrected.astype(data.dtype), factor

    def _apply_c_correction(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply C correction (Teillet et al.)."""
        sun_zen = np.radians(90 - self.config.sun_elevation_deg)
        cos_sun_zen = np.cos(sun_zen)

        # Estimate C factor from regression of L vs cos(i)
        # For simplicity, use a fixed C value here
        # In practice, C should be estimated from the data
        c = 0.5

        # C correction: L_corrected = L * (cos(sun_zen) + C) / (cos(i) + C)
        factor = (cos_sun_zen + c) / (self._illumination + c)

        if len(data.shape) == 3:
            corrected = data * factor[..., np.newaxis]
        else:
            corrected = data * factor

        return corrected.astype(data.dtype), factor

    def _apply_scs_correction(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Sun-Canopy-Sensor correction."""
        sun_zen = np.radians(90 - self.config.sun_elevation_deg)
        cos_sun_zen = np.cos(sun_zen)

        # SCS correction: L_corrected = L * (cos(sun_zen) * cos(slope)) / cos(i)
        factor = (cos_sun_zen * np.cos(self._slope)) / self._illumination

        if len(data.shape) == 3:
            corrected = data * factor[..., np.newaxis]
        else:
            corrected = data * factor

        return corrected.astype(data.dtype), factor

    def _apply_gamma_correction(
        self,
        data: np.ndarray,
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gamma terrain correction for SAR data.

        Converts from beta naught to gamma naught using local incidence angle.
        """
        # Calculate local incidence angle
        sensor_zen = np.radians(90 - self.config.sensor_elevation_deg)
        sensor_az = np.radians(self.config.sensor_azimuth_deg)

        # Local incidence angle
        cos_local = (
            np.cos(self._slope) * np.cos(sensor_zen) +
            np.sin(self._slope) * np.sin(sensor_zen) * np.cos(sensor_az - self._aspect)
        )
        cos_local = np.clip(cos_local, 0.001, 1.0)

        # Gamma correction: gamma0 = beta0 * sin(local_incidence) / sin(sensor_incidence)
        factor = cos_local / np.cos(sensor_zen)

        if len(data.shape) == 3:
            corrected = data * factor[..., np.newaxis]
        else:
            corrected = data * factor

        return corrected.astype(data.dtype), factor

    def _create_quality_mask(self, factor: np.ndarray) -> np.ndarray:
        """Create quality mask based on correction factor."""
        quality = np.ones_like(factor, dtype=np.float32)

        # Reduce quality for extreme correction factors
        quality = np.where(factor < 0.5, quality * 0.5, quality)
        quality = np.where(factor > 2.0, quality * 0.5, quality)

        # Reduce quality for steep slopes
        if self._slope is not None:
            slope_deg = np.degrees(self._slope)
            quality = np.where(
                slope_deg > self.config.slope_threshold_deg,
                quality * 0.3,
                quality
            )

        # Zero quality for shadowed areas
        if self.config.shadow_detection and self._illumination is not None:
            quality = np.where(self._illumination < 0.1, 0.0, quality)

        return quality


class AtmosphericCorrector:
    """
    Corrects optical imagery for atmospheric effects.

    Provides methods for:
    - Dark Object Subtraction (DOS)
    - Top of Atmosphere reflectance conversion
    - Basic surface reflectance estimation
    """

    def __init__(self, config: Optional[AtmosphericCorrectionConfig] = None):
        """
        Initialize atmospheric corrector.

        Args:
            config: Atmospheric correction configuration
        """
        self.config = config or AtmosphericCorrectionConfig()

    def correct(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
    ) -> CorrectionResult:
        """
        Apply atmospheric correction to optical data.

        Args:
            data: Input data array (2D for single band, 3D for multi-band)
            wavelengths: Band center wavelengths in nm

        Returns:
            CorrectionResult with atmospherically corrected data
        """
        wavelengths = wavelengths or self.config.band_wavelengths

        method = self.config.method

        if method == AtmosphericCorrectionMethod.DOS:
            corrected, params = self._apply_dos(data)
        elif method == AtmosphericCorrectionMethod.DOS_IMPROVED:
            corrected, params = self._apply_dos_improved(data, wavelengths)
        elif method == AtmosphericCorrectionMethod.TOAR:
            corrected, params = self._apply_toar(data)
        elif method == AtmosphericCorrectionMethod.COST:
            corrected, params = self._apply_cost(data, wavelengths)
        else:
            logger.warning(f"Method {method} not implemented, using DOS")
            corrected, params = self._apply_dos(data)

        # Create quality mask based on valid data
        quality_mask = np.ones(corrected.shape[:2] if len(corrected.shape) == 3 else corrected.shape, dtype=np.float32)
        quality_mask[np.isnan(corrected if len(corrected.shape) == 2 else corrected[..., 0])] = 0

        return CorrectionResult(
            corrected_data=corrected,
            quality_mask=quality_mask,
            correction_type=f"atmospheric_{method.value}",
            parameters=params,
            diagnostics={
                "method": method.value,
                "sensor_type": self.config.sensor_type,
            }
        )

    def _apply_dos(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Dark Object Subtraction."""
        if len(data.shape) == 2:
            # Single band
            dark_value = np.nanpercentile(data, self.config.dark_object_threshold)
            corrected = data - dark_value
            corrected = np.clip(corrected, 0, None)
            params = {"dark_object_value": float(dark_value)}
        else:
            # Multi-band
            corrected = np.zeros_like(data, dtype=np.float32)
            dark_values = []
            for i in range(data.shape[-1]):
                dark_value = np.nanpercentile(data[..., i], self.config.dark_object_threshold)
                corrected[..., i] = np.clip(data[..., i] - dark_value, 0, None)
                dark_values.append(float(dark_value))
            params = {"dark_object_values": dark_values}

        return corrected, params

    def _apply_dos_improved(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply improved DOS with wavelength-dependent haze correction."""
        # Apply basic DOS first
        corrected, params = self._apply_dos(data)

        if wavelengths is not None and len(data.shape) == 3:
            # Apply wavelength-dependent correction
            # Haze has stronger effect on shorter wavelengths
            reference_wavelength = 550.0  # Green reference
            for i, wl in enumerate(wavelengths):
                # Rayleigh-like correction factor
                rayleigh_factor = (reference_wavelength / wl) ** 4
                # Apply small adjustment based on wavelength
                adjustment = 0.01 * rayleigh_factor
                corrected[..., i] = corrected[..., i] * (1 + adjustment)

            params["wavelength_adjusted"] = True
            params["wavelengths"] = wavelengths

        return corrected, params

    def _apply_toar(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert to Top of Atmosphere Reflectance."""
        # Calculate solar zenith cosine
        cos_sza = np.cos(np.radians(self.config.solar_zenith_deg))

        # Simple TOA reflectance: rho = (pi * L * d^2) / (ESUN * cos(sza))
        # For simplicity, assume data is already in radiance units
        # and use a simplified conversion

        # Assume data is DN, convert to reflectance-like values
        corrected = data.astype(np.float32) / 10000.0  # Typical scaling

        # Apply solar angle correction
        corrected = corrected / cos_sza

        # Clamp to valid reflectance range
        corrected = np.clip(corrected, 0, 1)

        params = {
            "solar_zenith_deg": self.config.solar_zenith_deg,
            "cos_sza": float(cos_sza),
        }

        return corrected, params

    def _apply_cost(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply COST model correction."""
        # COST = Cosine of the Sun Angle (Chavez, 1996)
        cos_sza = np.cos(np.radians(self.config.solar_zenith_deg))

        # Apply DOS
        corrected, dos_params = self._apply_dos(data)

        # Apply cosine correction
        corrected = corrected / cos_sza

        # Apply simple transmittance model
        transmittance = self._estimate_transmittance(wavelengths)

        if len(corrected.shape) == 3 and transmittance is not None:
            for i, t in enumerate(transmittance):
                corrected[..., i] = corrected[..., i] / t

        params = {
            **dos_params,
            "solar_zenith_deg": self.config.solar_zenith_deg,
            "transmittance": transmittance,
        }

        return np.clip(corrected, 0, 1), params

    def _estimate_transmittance(
        self,
        wavelengths: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        """Estimate atmospheric transmittance."""
        if wavelengths is None:
            return None

        # Simple exponential model based on AOD
        tau = self.config.aod550

        transmittances = []
        for wl in wavelengths:
            # Angstrom exponent approximation
            angstrom = 1.4
            tau_wl = tau * (wl / 550.0) ** (-angstrom)
            # Beer-Lambert law for 2-way path
            air_mass = 1 / np.cos(np.radians(self.config.solar_zenith_deg))
            air_mass += 1 / np.cos(np.radians(self.config.view_zenith_deg))
            t = np.exp(-tau_wl * air_mass)
            transmittances.append(float(t))

        return transmittances


class RadiometricNormalizer:
    """
    Normalizes radiometric values across sensors and dates.

    Provides methods for:
    - Histogram matching
    - Pseudo-Invariant Feature based normalization
    - Cross-sensor calibration
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize radiometric normalizer.

        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self._reference_data = None
        self._reference_stats = None

    def set_reference(
        self,
        reference_data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Set reference data for normalization.

        Args:
            reference_data: Reference data array
            mask: Optional mask for valid pixels
        """
        self._reference_data = reference_data
        self._reference_stats = self._compute_stats(reference_data, mask)

    def normalize(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """
        Normalize data to reference.

        Args:
            data: Input data to normalize
            mask: Optional mask for valid pixels

        Returns:
            CorrectionResult with normalized data
        """
        if self._reference_data is None:
            logger.warning("No reference set, returning unchanged data")
            return CorrectionResult(
                corrected_data=data,
                correction_type="none",
                diagnostics={"reason": "no_reference"}
            )

        method = self.config.method

        if method == NormalizationMethod.HISTOGRAM_MATCHING:
            normalized, params = self._histogram_matching(data, mask)
        elif method == NormalizationMethod.PIF:
            normalized, params = self._pif_normalization(data, mask)
        elif method == NormalizationMethod.RELATIVE:
            normalized, params = self._relative_normalization(data, mask)
        else:
            logger.warning(f"Method {method} not implemented, using histogram matching")
            normalized, params = self._histogram_matching(data, mask)

        # Compute quality mask
        quality_mask = np.ones(normalized.shape[:2] if len(normalized.shape) == 3 else normalized.shape, dtype=np.float32)
        if mask is not None:
            quality_mask *= mask.astype(np.float32)

        return CorrectionResult(
            corrected_data=normalized,
            quality_mask=quality_mask,
            correction_type=f"normalization_{method.value}",
            parameters=params,
            diagnostics={
                "method": method.value,
                "reference_stats": self._reference_stats,
            }
        )

    def _compute_stats(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute statistics for data."""
        if mask is not None:
            valid_data = data[mask > 0]
        else:
            valid_data = data[~np.isnan(data)]

        return {
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "p2": float(np.percentile(valid_data, 2)),
            "p98": float(np.percentile(valid_data, 98)),
        }

    def _histogram_matching(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Match histogram to reference."""
        p_low, p_high = self.config.histogram_match_percentiles

        def match_band(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
            """Match a single band's histogram."""
            # Get valid values
            src_valid = source[~np.isnan(source)]
            ref_valid = reference[~np.isnan(reference)]

            if len(src_valid) == 0 or len(ref_valid) == 0:
                return source

            # Compute percentiles
            src_p = np.percentile(src_valid, [p_low, p_high])
            ref_p = np.percentile(ref_valid, [p_low, p_high])

            # Avoid division by zero
            if src_p[1] - src_p[0] == 0:
                return source

            # Linear transformation to match
            scale = (ref_p[1] - ref_p[0]) / (src_p[1] - src_p[0])
            offset = ref_p[0] - src_p[0] * scale

            matched = source * scale + offset

            if self.config.clip_outliers:
                matched = np.clip(matched, ref_p[0], ref_p[1])

            return matched

        if len(data.shape) == 2:
            normalized = match_band(data, self._reference_data)
        else:
            normalized = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[-1]):
                ref_band = self._reference_data[..., i] if len(self._reference_data.shape) == 3 else self._reference_data
                normalized[..., i] = match_band(data[..., i], ref_band)

        params = {
            "percentiles": [p_low, p_high],
            "clip_outliers": self.config.clip_outliers,
        }

        return normalized, params

    def _pif_normalization(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize using Pseudo-Invariant Features."""
        # Find PIF pixels (stable features between dates)
        pif_mask = self._identify_pif(data, self._reference_data)

        if mask is not None:
            pif_mask = pif_mask & (mask > 0)

        pif_count = np.sum(pif_mask)
        if pif_count < 100:
            logger.warning(f"Too few PIF pixels ({pif_count}), falling back to histogram matching")
            return self._histogram_matching(data, mask)

        # Compute regression on PIF pixels
        if len(data.shape) == 2:
            src_pif = data[pif_mask]
            ref_pif = self._reference_data[pif_mask]
            slope, intercept = self._linear_regression(src_pif, ref_pif)
            normalized = data * slope + intercept
        else:
            normalized = np.zeros_like(data, dtype=np.float32)
            slopes = []
            intercepts = []
            for i in range(data.shape[-1]):
                ref_band = self._reference_data[..., i] if len(self._reference_data.shape) == 3 else self._reference_data
                src_pif = data[..., i][pif_mask]
                ref_pif = ref_band[pif_mask]
                slope, intercept = self._linear_regression(src_pif, ref_pif)
                normalized[..., i] = data[..., i] * slope + intercept
                slopes.append(float(slope))
                intercepts.append(float(intercept))

        params = {
            "pif_count": int(pif_count),
            "slopes": slopes if len(data.shape) == 3 else float(slope),
            "intercepts": intercepts if len(data.shape) == 3 else float(intercept),
        }

        return normalized, params

    def _relative_normalization(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply simple relative normalization (mean/std matching)."""
        src_stats = self._compute_stats(data, mask)

        # Normalize: (x - src_mean) / src_std * ref_std + ref_mean
        if src_stats["std"] == 0:
            return data, {"warning": "zero_std"}

        normalized = (data - src_stats["mean"]) / src_stats["std"]
        normalized = normalized * self._reference_stats["std"] + self._reference_stats["mean"]

        params = {
            "source_stats": src_stats,
            "reference_stats": self._reference_stats,
        }

        return normalized.astype(data.dtype), params

    def _identify_pif(
        self,
        data: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Identify Pseudo-Invariant Features."""
        # Simple approach: pixels with low relative change
        threshold = self.config.pif_threshold

        if len(data.shape) == 2:
            # Compute normalized difference
            diff = np.abs(data - reference) / (np.abs(data) + np.abs(reference) + 1e-10)
            pif_mask = diff < (1 - threshold)
        else:
            # Multi-band: require low difference in all bands
            masks = []
            for i in range(data.shape[-1]):
                ref_band = reference[..., i] if len(reference.shape) == 3 else reference
                diff = np.abs(data[..., i] - ref_band) / (np.abs(data[..., i]) + np.abs(ref_band) + 1e-10)
                masks.append(diff < (1 - threshold))
            pif_mask = np.all(np.stack(masks), axis=0)

        # Remove NaN pixels
        pif_mask = pif_mask & ~np.isnan(data if len(data.shape) == 2 else data[..., 0])
        pif_mask = pif_mask & ~np.isnan(reference if len(reference.shape) == 2 else reference[..., 0])

        return pif_mask

    def _linear_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute linear regression (y = slope * x + intercept)."""
        # Remove NaN
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            return 1.0, 0.0

        # Least squares
        x_mean = np.mean(x_valid)
        y_mean = np.mean(y_valid)

        denominator = np.sum((x_valid - x_mean) ** 2)
        if denominator == 0:
            return 1.0, 0.0

        slope = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / denominator
        intercept = y_mean - slope * x_mean

        return float(slope), float(intercept)


class CorrectionPipeline:
    """
    Orchestrates multiple correction steps in sequence.

    Applies terrain, atmospheric, and radiometric corrections
    in the appropriate order for multi-sensor fusion.
    """

    def __init__(
        self,
        terrain_config: Optional[TerrainCorrectionConfig] = None,
        atmospheric_config: Optional[AtmosphericCorrectionConfig] = None,
        normalization_config: Optional[NormalizationConfig] = None,
    ):
        """
        Initialize correction pipeline.

        Args:
            terrain_config: Terrain correction config
            atmospheric_config: Atmospheric correction config
            normalization_config: Radiometric normalization config
        """
        self.terrain_corrector = TerrainCorrector(terrain_config) if terrain_config else None
        self.atmospheric_corrector = AtmosphericCorrector(atmospheric_config) if atmospheric_config else None
        self.normalizer = RadiometricNormalizer(normalization_config) if normalization_config else None

    def apply(
        self,
        data: np.ndarray,
        sensor_type: str = "optical",
        dem: Optional[np.ndarray] = None,
        wavelengths: Optional[List[float]] = None,
        reference_data: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """
        Apply full correction pipeline.

        Args:
            data: Input data array
            sensor_type: Type of sensor (optical, sar, thermal)
            dem: DEM for terrain correction
            wavelengths: Band wavelengths for atmospheric correction
            reference_data: Reference for normalization

        Returns:
            CorrectionResult with all corrections applied
        """
        current_data = data.copy()
        all_diagnostics = {"steps_applied": []}

        # Step 1: Atmospheric correction (optical only)
        if self.atmospheric_corrector and sensor_type == "optical":
            result = self.atmospheric_corrector.correct(current_data, wavelengths)
            current_data = result.corrected_data
            all_diagnostics["atmospheric"] = result.diagnostics
            all_diagnostics["steps_applied"].append("atmospheric")

        # Step 2: Terrain correction
        if self.terrain_corrector and dem is not None:
            self.terrain_corrector.config.dem = dem
            result = self.terrain_corrector.correct(current_data, dem)
            current_data = result.corrected_data
            all_diagnostics["terrain"] = result.diagnostics
            all_diagnostics["steps_applied"].append("terrain")

        # Step 3: Radiometric normalization
        if self.normalizer and reference_data is not None:
            self.normalizer.set_reference(reference_data)
            result = self.normalizer.normalize(current_data)
            current_data = result.corrected_data
            all_diagnostics["normalization"] = result.diagnostics
            all_diagnostics["steps_applied"].append("normalization")

        return CorrectionResult(
            corrected_data=current_data,
            correction_type="pipeline",
            diagnostics=all_diagnostics,
        )


# Convenience functions

def apply_terrain_correction(
    data: np.ndarray,
    dem: np.ndarray,
    sun_elevation_deg: float = 45.0,
    sun_azimuth_deg: float = 180.0,
    method: TerrainCorrectionMethod = TerrainCorrectionMethod.COSINE,
) -> CorrectionResult:
    """
    Apply terrain correction to data.

    Args:
        data: Input data array
        dem: DEM array
        sun_elevation_deg: Sun elevation angle
        sun_azimuth_deg: Sun azimuth angle
        method: Correction method

    Returns:
        CorrectionResult
    """
    config = TerrainCorrectionConfig(
        method=method,
        dem=dem,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
    )
    corrector = TerrainCorrector(config)
    return corrector.correct(data)


def apply_atmospheric_correction(
    data: np.ndarray,
    method: AtmosphericCorrectionMethod = AtmosphericCorrectionMethod.DOS,
    solar_zenith_deg: float = 30.0,
) -> CorrectionResult:
    """
    Apply atmospheric correction to optical data.

    Args:
        data: Input data array
        method: Correction method
        solar_zenith_deg: Solar zenith angle

    Returns:
        CorrectionResult
    """
    config = AtmosphericCorrectionConfig(
        method=method,
        solar_zenith_deg=solar_zenith_deg,
    )
    corrector = AtmosphericCorrector(config)
    return corrector.correct(data)


def normalize_to_reference(
    data: np.ndarray,
    reference: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.HISTOGRAM_MATCHING,
) -> CorrectionResult:
    """
    Normalize data to match reference.

    Args:
        data: Input data to normalize
        reference: Reference data to match
        method: Normalization method

    Returns:
        CorrectionResult
    """
    config = NormalizationConfig(method=method)
    normalizer = RadiometricNormalizer(config)
    normalizer.set_reference(reference)
    return normalizer.normalize(data)
