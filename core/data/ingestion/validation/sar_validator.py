"""
SAR Validator for Synthetic Aperture Radar Imagery.

Validates SAR images with speckle-aware thresholds and
polarization-specific checks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data.ingestion.validation.config import ValidationConfig
from core.data.ingestion.validation.exceptions import SARValidationError
from core.data.ingestion.validation.image_validator import (
    BandStatistics,
    BandValidationResult,
)

logger = logging.getLogger(__name__)

# Optional rasterio import
try:
    import rasterio
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    Window = None
    HAS_RASTERIO = False


@dataclass
class SARValidationResult:
    """
    Result of SAR-specific validation.

    Attributes:
        is_valid: Whether validation passed
        band_results: Per-polarization validation results
        polarizations_found: List of polarizations found
        acquisition_mode: SAR acquisition mode (if detected)
        orbit_direction: Ascending or Descending
        processing_level: GRD or SLC
        warnings: Warning messages
        errors: Error messages
    """

    is_valid: bool = True
    band_results: Dict[str, BandValidationResult] = field(default_factory=dict)
    polarizations_found: List[str] = field(default_factory=list)
    acquisition_mode: Optional[str] = None
    orbit_direction: Optional[str] = None
    processing_level: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "band_results": {k: v.to_dict() for k, v in self.band_results.items()},
            "polarizations_found": self.polarizations_found,
            "acquisition_mode": self.acquisition_mode,
            "orbit_direction": self.orbit_direction,
            "processing_level": self.processing_level,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class SARValidator:
    """
    Validator for SAR imagery.

    SAR images have different characteristics than optical:
    - Speckle noise creates inherent variability
    - Values are typically in dB (backscatter)
    - Different polarizations (VV, VH, HH, HV)
    - Different valid value ranges

    Example:
        validator = SARValidator(config)
        result = validator.validate(
            dataset=rasterio_dataset,
            data_source_spec={"sensor": "Sentinel-1"},
        )
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize SAR validator.

        Args:
            config: Validation configuration
        """
        self.config = config

        # Standard polarization band patterns
        self.polarization_patterns = {
            "VV": ["VV", "vv", "Sigma0_VV", "sigma0_vv"],
            "VH": ["VH", "vh", "Sigma0_VH", "sigma0_vh"],
            "HH": ["HH", "hh", "Sigma0_HH", "sigma0_hh"],
            "HV": ["HV", "hv", "Sigma0_HV", "sigma0_hv"],
        }

    def validate(
        self,
        dataset: Any,
        data_source_spec: Optional[Dict[str, Any]] = None,
    ) -> SARValidationResult:
        """
        Validate SAR imagery.

        Args:
            dataset: Open rasterio dataset
            data_source_spec: Data source specification

        Returns:
            SARValidationResult
        """
        result = SARValidationResult()

        # Step 1: Extract SAR metadata
        self._extract_sar_metadata(dataset, data_source_spec, result)

        # Step 2: Identify polarization bands
        polarization_bands = self._identify_polarization_bands(dataset)
        result.polarizations_found = list(polarization_bands.keys())

        # Step 3: Check required polarizations
        required_pols = self.config.required_sar_polarizations
        found_required = [p for p in required_pols if p in polarization_bands]

        if not found_required:
            result.is_valid = False
            result.errors.append(
                f"No required polarization found. Required: {required_pols}, Found: {result.polarizations_found}"
            )
            return result

        # Step 4: Validate each polarization band
        for pol_name, band_index in polarization_bands.items():
            is_required = pol_name in required_pols
            band_result = self._validate_polarization_band(
                dataset, pol_name, band_index, is_required
            )
            result.band_results[pol_name] = band_result

            if not band_result.is_valid and is_required:
                result.is_valid = False

        return result

    def _extract_sar_metadata(
        self,
        dataset: Any,
        data_source_spec: Optional[Dict[str, Any]],
        result: SARValidationResult,
    ) -> None:
        """Extract SAR-specific metadata from dataset."""
        # Try to get from dataset tags/metadata
        tags = dataset.tags() if hasattr(dataset, "tags") else {}

        # Acquisition mode
        if "ACQUISITION_MODE" in tags:
            result.acquisition_mode = tags["ACQUISITION_MODE"]
        elif data_source_spec and "acquisition_mode" in data_source_spec:
            result.acquisition_mode = data_source_spec["acquisition_mode"]
        else:
            # Try to detect from common patterns
            for key in tags:
                if "IW" in str(tags.get(key, "")):
                    result.acquisition_mode = "IW"
                    break
                elif "EW" in str(tags.get(key, "")):
                    result.acquisition_mode = "EW"
                    break

        # Orbit direction
        if "ORBIT_DIRECTION" in tags:
            result.orbit_direction = tags["ORBIT_DIRECTION"]
        elif data_source_spec and "orbit_direction" in data_source_spec:
            result.orbit_direction = data_source_spec["orbit_direction"]

        # Processing level
        if "PROCESSING_LEVEL" in tags:
            result.processing_level = tags["PROCESSING_LEVEL"]
        elif data_source_spec and "processing_level" in data_source_spec:
            result.processing_level = data_source_spec["processing_level"]

    def _identify_polarization_bands(
        self,
        dataset: Any,
    ) -> Dict[str, int]:
        """
        Identify which bands correspond to which polarizations.

        Returns:
            Mapping of polarization name to 1-indexed band number
        """
        found_pols: Dict[str, int] = {}

        # Check band descriptions
        for band_idx in range(1, dataset.count + 1):
            desc = dataset.descriptions[band_idx - 1] if dataset.descriptions else None

            # Also check tags
            try:
                band_tags = dataset.tags(band_idx) or {}
            except Exception:
                band_tags = {}

            # Combine description and tags for matching
            search_text = str(desc or "").upper()
            for key, val in band_tags.items():
                search_text += f" {str(key).upper()} {str(val).upper()}"

            # Match against polarization patterns
            for pol_name, patterns in self.polarization_patterns.items():
                if pol_name in found_pols:
                    continue
                for pattern in patterns:
                    if pattern.upper() in search_text:
                        found_pols[pol_name] = band_idx
                        break

        # If no polarizations found by name, use positional assumptions
        if not found_pols and dataset.count >= 1:
            # Assume first band is VV for single-band SAR
            found_pols["VV"] = 1
            if dataset.count >= 2:
                found_pols["VH"] = 2

        return found_pols

    def _validate_polarization_band(
        self,
        dataset: Any,
        pol_name: str,
        band_index: int,
        is_required: bool,
    ) -> BandValidationResult:
        """
        Validate a single polarization band.

        SAR-specific validation with speckle-aware thresholds.

        Args:
            dataset: Open rasterio dataset
            pol_name: Polarization name (VV, VH, etc.)
            band_index: 1-indexed band number
            is_required: Whether this polarization is required

        Returns:
            BandValidationResult
        """
        result = BandValidationResult(
            band_name=pol_name,
            band_index=band_index,
            is_valid=True,
            is_required=is_required,
        )

        try:
            # Calculate statistics
            stats = self._calculate_sar_statistics(dataset, band_index)
            result.statistics = stats

            # Check if band is blank (using SAR threshold)
            if stats.std_dev < self.config.sar.std_dev_min_db:
                result.is_valid = False
                result.errors.append(
                    f"SAR band '{pol_name}' appears blank "
                    f"(std_dev={stats.std_dev:.4f} dB < {self.config.sar.std_dev_min_db} dB)"
                )

            # Check non-zero ratio
            if stats.non_zero_ratio < self.config.optical.non_zero_ratio_min:
                result.is_valid = False
                result.errors.append(
                    f"SAR band '{pol_name}' has too few valid pixels "
                    f"(ratio={stats.non_zero_ratio:.4f})"
                )

            # Check backscatter range
            self._check_backscatter_range(pol_name, stats, result)

            # Check for extreme values
            extreme_ratio = self._calculate_extreme_value_ratio(dataset, band_index)
            if extreme_ratio > self.config.sar.extreme_value_ratio_max:
                result.warnings.append(
                    f"SAR band '{pol_name}' has high ratio of extreme values "
                    f"({extreme_ratio:.2%})"
                )

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error validating SAR band '{pol_name}': {str(e)}")
            logger.error(f"Error validating SAR band {pol_name}: {e}")

        return result

    def _calculate_sar_statistics(
        self,
        dataset: Any,
        band_index: int,
    ) -> BandStatistics:
        """
        Calculate statistics for SAR band.

        Handles dB conversion and NoData masking.

        Args:
            dataset: Open rasterio dataset
            band_index: 1-indexed band number

        Returns:
            BandStatistics
        """
        total_pixels = dataset.width * dataset.height
        nodata = dataset.nodata

        # Determine if sampling is needed
        should_sample = total_pixels > self.config.performance.sample_threshold_pixels
        sample_ratio = self.config.performance.sample_ratio if should_sample else 1.0

        if should_sample:
            data = self._read_sampled_data(dataset, band_index, sample_ratio)
        else:
            data = dataset.read(band_index)

        data = data.flatten().astype(np.float64)

        # Create mask for valid data
        if nodata is not None:
            if np.isnan(nodata):
                valid_mask = ~np.isnan(data)
            else:
                valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data) & (data != 0)

        valid_data = data[valid_mask]
        valid_count = len(valid_data)

        if valid_count == 0:
            return BandStatistics(
                mean=np.nan,
                std_dev=0.0,
                min_val=np.nan,
                max_val=np.nan,
                non_zero_ratio=0.0,
                nodata_ratio=1.0,
                valid_pixel_count=0,
                total_pixel_count=total_pixels,
            )

        # Check if data is likely in linear or dB scale
        # Linear backscatter values are typically < 1
        # dB values are typically -30 to +10
        is_db_scale = np.median(valid_data) < 50  # Heuristic

        if not is_db_scale:
            # Convert from linear to dB (assuming sigma0)
            # dB = 10 * log10(linear)
            valid_data = np.where(valid_data > 0, 10 * np.log10(valid_data), -50)

        # Calculate statistics in dB
        mean_val = float(np.mean(valid_data))
        std_val = float(np.std(valid_data))
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))

        # Calculate ratios
        actual_pixels = len(data)
        non_zero = np.count_nonzero(valid_data)
        non_zero_ratio = non_zero / valid_count if valid_count > 0 else 0.0

        nodata_count = actual_pixels - valid_count
        nodata_ratio = nodata_count / actual_pixels if actual_pixels > 0 else 0.0

        return BandStatistics(
            mean=mean_val,
            std_dev=std_val,
            min_val=min_val,
            max_val=max_val,
            non_zero_ratio=non_zero_ratio,
            nodata_ratio=nodata_ratio,
            valid_pixel_count=valid_count,
            total_pixel_count=total_pixels,
        )

    def _read_sampled_data(
        self,
        dataset: Any,
        band_index: int,
        sample_ratio: float,
    ) -> np.ndarray:
        """Read sampled data from dataset."""
        height, width = dataset.height, dataset.width
        window_size = 512
        n_windows_x = max(1, width // window_size)
        n_windows_y = max(1, height // window_size)
        total_windows = n_windows_x * n_windows_y

        n_samples = max(1, int(total_windows * sample_ratio))

        rng = np.random.default_rng(42)
        window_indices = rng.choice(total_windows, size=n_samples, replace=False)

        all_values = []
        for idx in window_indices:
            wy = idx // n_windows_x
            wx = idx % n_windows_x

            col_off = wx * window_size
            row_off = wy * window_size
            w = min(window_size, width - col_off)
            h = min(window_size, height - row_off)

            window = Window(col_off, row_off, w, h)
            data = dataset.read(band_index, window=window)
            all_values.append(data.flatten())

        return np.concatenate(all_values)

    def _check_backscatter_range(
        self,
        pol_name: str,
        stats: BandStatistics,
        result: BandValidationResult,
    ) -> None:
        """Check if backscatter values are within expected range."""
        min_db, max_db = self.config.sar.backscatter_range_db

        if not np.isnan(stats.min_val) and stats.min_val < min_db:
            result.warnings.append(
                f"SAR band '{pol_name}' has values below typical backscatter range "
                f"(min={stats.min_val:.1f} dB)"
            )

        if not np.isnan(stats.max_val) and stats.max_val > max_db:
            result.warnings.append(
                f"SAR band '{pol_name}' has values above typical backscatter range "
                f"(max={stats.max_val:.1f} dB)"
            )

    def _calculate_extreme_value_ratio(
        self,
        dataset: Any,
        band_index: int,
    ) -> float:
        """
        Calculate ratio of pixels with extreme backscatter values.

        Extreme values are outside the typical land/water range.

        Args:
            dataset: Open rasterio dataset
            band_index: 1-indexed band number

        Returns:
            Ratio of extreme value pixels (0.0 to 1.0)
        """
        # Sample for efficiency
        total_pixels = dataset.width * dataset.height
        if total_pixels > self.config.performance.sample_threshold_pixels:
            data = self._read_sampled_data(
                dataset, band_index, self.config.performance.sample_ratio
            )
        else:
            data = dataset.read(band_index).flatten()

        # Convert to dB if needed
        data = data.astype(np.float64)
        nodata = dataset.nodata
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data) & (data != 0)

        valid_data = data[valid_mask]
        if len(valid_data) == 0:
            return 1.0

        # Check if linear scale and convert
        if np.median(valid_data) > 50:
            valid_data = np.where(valid_data > 0, 10 * np.log10(valid_data), -50)

        # Count extreme values
        min_thresh, max_thresh = self.config.sar.extreme_value_threshold_db
        extreme_count = np.sum((valid_data < min_thresh) | (valid_data > max_thresh))

        return extreme_count / len(valid_data)

    def validate_sar_specific(
        self,
        dataset: Any,
        polarization: str = "VV",
    ) -> BandValidationResult:
        """
        Validate SAR-specific characteristics.

        Convenience method for single-polarization validation.

        Args:
            dataset: Open rasterio dataset
            polarization: Polarization to validate

        Returns:
            BandValidationResult
        """
        pol_bands = self._identify_polarization_bands(dataset)

        if polarization not in pol_bands:
            return BandValidationResult(
                band_name=polarization,
                band_index=0,
                is_valid=False,
                is_required=True,
                errors=[f"Polarization {polarization} not found in dataset"],
            )

        return self._validate_polarization_band(
            dataset,
            polarization,
            pol_bands[polarization],
            is_required=True,
        )
