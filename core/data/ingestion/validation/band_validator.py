"""
Band Validator for Optical Imagery.

Validates individual bands for presence, content (not blank),
and statistical validity.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data.ingestion.validation.config import ValidationConfig
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
class BandInfo:
    """
    Information about a band in the raster.

    Attributes:
        index: 1-based band index
        name: Band name (if available from metadata)
        dtype: Data type
        nodata: NoData value
    """

    index: int
    name: str = ""
    dtype: str = ""
    nodata: Optional[float] = None


class BandValidator:
    """
    Validates optical imagery bands.

    Performs the following checks:
    - Band presence (required vs optional)
    - Band content (not blank, valid statistics)
    - Value range validation
    - NoData handling

    Example:
        validator = BandValidator(config)
        results = validator.validate_bands(
            dataset=rasterio_dataset,
            expected_bands={"nir": ["B08", "B8A"]},
            required_bands=["blue", "green", "red", "nir"],
        )
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize band validator.

        Args:
            config: Validation configuration
        """
        self.config = config

    def validate_bands(
        self,
        dataset: Any,
        expected_bands: Dict[str, List[str]],
        required_bands: Optional[List[str]] = None,
    ) -> Dict[str, BandValidationResult]:
        """
        Validate all bands in the dataset.

        Args:
            dataset: Open rasterio dataset
            expected_bands: Mapping of generic band names to possible sensor band names
            required_bands: List of required band names (generic)

        Returns:
            Dictionary mapping band names to validation results
        """
        if required_bands is None:
            required_bands = self.config.required_optical_bands

        results: Dict[str, BandValidationResult] = {}

        # Get band descriptions/names from dataset
        band_descriptions = self._get_band_descriptions(dataset)

        # Match expected bands to dataset bands
        band_mapping = self._match_bands(expected_bands, band_descriptions, dataset.count)

        # Determine which bands to validate
        bands_to_validate = []
        for generic_name, band_index in band_mapping.items():
            is_required = generic_name in required_bands
            bands_to_validate.append((generic_name, band_index, is_required))

        # Also check for missing required bands
        for required_name in required_bands:
            if required_name not in band_mapping:
                results[required_name] = BandValidationResult(
                    band_name=required_name,
                    band_index=0,
                    is_valid=False,
                    is_required=True,
                    errors=[f"Required band '{required_name}' not found in dataset"],
                )

        # Validate bands (optionally in parallel)
        if self.config.performance.parallel_bands and len(bands_to_validate) > 1:
            results.update(
                self._validate_bands_parallel(dataset, bands_to_validate)
            )
        else:
            for generic_name, band_index, is_required in bands_to_validate:
                result = self._validate_single_band(
                    dataset, generic_name, band_index, is_required
                )
                results[generic_name] = result

        return results

    def _get_band_descriptions(self, dataset: Any) -> Dict[int, str]:
        """Get band descriptions from dataset metadata."""
        descriptions = {}
        for i in range(1, dataset.count + 1):
            desc = dataset.descriptions[i - 1] if dataset.descriptions else None
            if desc:
                descriptions[i] = desc
            else:
                # Try to get from tags
                tags = dataset.tags(i)
                if tags and "DESCRIPTION" in tags:
                    descriptions[i] = tags["DESCRIPTION"]
        return descriptions

    def _match_bands(
        self,
        expected_bands: Dict[str, List[str]],
        band_descriptions: Dict[int, str],
        band_count: int,
    ) -> Dict[str, int]:
        """
        Match expected band names to dataset band indices.

        Args:
            expected_bands: Mapping of generic names to possible band names
            band_descriptions: Dataset band descriptions
            band_count: Total number of bands

        Returns:
            Mapping of generic band names to 1-indexed band numbers
        """
        mapping = {}

        # First try to match by band description
        for generic_name, possible_names in expected_bands.items():
            for band_idx, desc in band_descriptions.items():
                if desc:
                    desc_upper = desc.upper()
                    for possible in possible_names:
                        if possible.upper() in desc_upper or desc_upper == possible.upper():
                            mapping[generic_name] = band_idx
                            break
                if generic_name in mapping:
                    break

        # For unmatched bands, use positional fallback for common band orders
        if band_count >= 4:
            positional_defaults = {
                "blue": 1,
                "green": 2,
                "red": 3,
                "nir": 4,
            }
            if band_count >= 6:
                positional_defaults["swir1"] = 5
                positional_defaults["swir2"] = 6

            for generic_name, default_idx in positional_defaults.items():
                if generic_name not in mapping and default_idx <= band_count:
                    # Only use default if it's in expected bands
                    if generic_name in expected_bands:
                        mapping[generic_name] = default_idx

        return mapping

    def _validate_single_band(
        self,
        dataset: Any,
        band_name: str,
        band_index: int,
        is_required: bool,
    ) -> BandValidationResult:
        """
        Validate a single band.

        Args:
            dataset: Open rasterio dataset
            band_name: Generic band name
            band_index: 1-indexed band number
            is_required: Whether this band is required

        Returns:
            BandValidationResult
        """
        result = BandValidationResult(
            band_name=band_name,
            band_index=band_index,
            is_valid=True,
            is_required=is_required,
        )

        try:
            # Calculate statistics
            stats = self._calculate_band_statistics(dataset, band_index)
            result.statistics = stats

            # Check if band is blank
            if stats.std_dev < self.config.optical.std_dev_min:
                result.is_valid = False
                result.errors.append(
                    f"Band '{band_name}' appears blank (std_dev={stats.std_dev:.4f} < {self.config.optical.std_dev_min})"
                )

            # Check non-zero ratio
            if stats.non_zero_ratio < self.config.optical.non_zero_ratio_min:
                result.is_valid = False
                result.errors.append(
                    f"Band '{band_name}' has too few non-zero pixels (ratio={stats.non_zero_ratio:.4f} < {self.config.optical.non_zero_ratio_min})"
                )

            # Check NoData ratio (warning only)
            if stats.nodata_ratio > self.config.optical.nodata_ratio_max:
                result.warnings.append(
                    f"Band '{band_name}' has high NoData ratio ({stats.nodata_ratio:.2%})"
                )

            # Check value range
            self._check_value_range(band_name, stats, dataset.dtypes[band_index - 1], result)

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error validating band '{band_name}': {str(e)}")
            logger.error(f"Error validating band {band_name}: {e}")

        return result

    def _calculate_band_statistics(
        self,
        dataset: Any,
        band_index: int,
    ) -> BandStatistics:
        """
        Calculate statistics for a band.

        Uses windowed/sampled reading for large images to stay within
        memory constraints.

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
            # Use windowed sampling
            return self._calculate_sampled_statistics(
                dataset, band_index, nodata, sample_ratio
            )
        else:
            # Read full band
            data = dataset.read(band_index)
            return self._calculate_array_statistics(data, nodata, total_pixels)

    def _calculate_sampled_statistics(
        self,
        dataset: Any,
        band_index: int,
        nodata: Optional[float],
        sample_ratio: float,
    ) -> BandStatistics:
        """
        Calculate statistics using random window sampling.

        Args:
            dataset: Open rasterio dataset
            band_index: Band index
            nodata: NoData value
            sample_ratio: Fraction of image to sample

        Returns:
            BandStatistics
        """
        height, width = dataset.height, dataset.width
        total_pixels = height * width

        # Use a grid of windows
        window_size = 512
        n_windows_x = max(1, width // window_size)
        n_windows_y = max(1, height // window_size)
        total_windows = n_windows_x * n_windows_y

        # Sample a subset of windows
        n_samples = max(1, int(total_windows * sample_ratio))

        # Generate random window indices
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        window_indices = rng.choice(total_windows, size=n_samples, replace=False)

        # Collect samples
        all_values = []
        sampled_pixels = 0

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
            sampled_pixels += data.size

        # Combine and calculate statistics
        combined = np.concatenate(all_values)
        return self._calculate_array_statistics(combined, nodata, total_pixels, sampled_pixels)

    def _calculate_array_statistics(
        self,
        data: np.ndarray,
        nodata: Optional[float],
        total_pixels: int,
        sampled_pixels: Optional[int] = None,
    ) -> BandStatistics:
        """
        Calculate statistics from array data.

        Args:
            data: Band data array
            nodata: NoData value
            total_pixels: Total pixels in image
            sampled_pixels: Number of sampled pixels (if sampling)

        Returns:
            BandStatistics
        """
        data = data.flatten()
        actual_pixels = sampled_pixels or len(data)

        # Create mask for valid data
        if nodata is not None:
            if np.isnan(nodata):
                valid_mask = ~np.isnan(data)
            else:
                valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data)

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

        # Calculate statistics
        mean_val = float(np.mean(valid_data))
        std_val = float(np.std(valid_data))
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))

        # Calculate ratios
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

    def _check_value_range(
        self,
        band_name: str,
        stats: BandStatistics,
        dtype: str,
        result: BandValidationResult,
    ) -> None:
        """Check if values are within expected range."""
        dtype_str = str(dtype).lower()

        if "uint16" in dtype_str or "int16" in dtype_str:
            expected_range = self.config.optical.value_range_uint16
        elif "float" in dtype_str:
            expected_range = self.config.optical.value_range_float32
        else:
            return  # Unknown dtype, skip range check

        # Check if min/max are within range
        if not np.isnan(stats.min_val) and not np.isnan(stats.max_val):
            if stats.min_val < expected_range[0]:
                result.warnings.append(
                    f"Band '{band_name}' has values below expected range (min={stats.min_val})"
                )
            if stats.max_val > expected_range[1]:
                result.warnings.append(
                    f"Band '{band_name}' has values above expected range (max={stats.max_val})"
                )

    def _validate_bands_parallel(
        self,
        dataset: Any,
        bands_to_validate: List[Tuple[str, int, bool]],
    ) -> Dict[str, BandValidationResult]:
        """Validate multiple bands in parallel."""
        results = {}

        # Note: rasterio datasets are not thread-safe for reading
        # So we need to be careful here. For true parallelism,
        # we would need to re-open the file in each thread.
        # For now, we use sequential processing with the parallel
        # flag mainly as an optimization indicator.

        for generic_name, band_index, is_required in bands_to_validate:
            result = self._validate_single_band(
                dataset, generic_name, band_index, is_required
            )
            results[generic_name] = result

        return results

    def validate_band_content(
        self,
        dataset: Any,
        band_index: int,
    ) -> BandValidationResult:
        """
        Validate content of a specific band.

        Convenience method for validating a single band by index.

        Args:
            dataset: Open rasterio dataset
            band_index: 1-indexed band number

        Returns:
            BandValidationResult
        """
        return self._validate_single_band(
            dataset,
            band_name=f"band_{band_index}",
            band_index=band_index,
            is_required=True,
        )
