"""
Spatial and Temporal Alignment for Multi-Sensor Data Fusion.

Provides tools for aligning geospatial data from multiple sensors to a
common reference frame, including:
- Spatial alignment (co-registration, reprojection to common grid)
- Temporal alignment (interpolation, binning, gap-filling)
- Multi-sensor alignment with quality-aware weighting

Key Concepts:
- Reference grid: The target spatial reference for all aligned data
- Temporal bins: Time intervals for aggregating multi-temporal observations
- Alignment confidence: Quality metric reflecting alignment accuracy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SpatialAlignmentMethod(Enum):
    """Methods for spatial alignment/co-registration."""
    REPROJECT = "reproject"              # Simple reprojection to target grid
    COREGISTER = "coregister"            # Feature-based co-registration
    RESAMPLE_NEAREST = "resample_nearest"  # Nearest neighbor resampling
    RESAMPLE_BILINEAR = "resample_bilinear"  # Bilinear interpolation
    RESAMPLE_CUBIC = "resample_cubic"    # Cubic interpolation
    RESAMPLE_LANCZOS = "resample_lanczos"  # Lanczos windowed sinc


class TemporalAlignmentMethod(Enum):
    """Methods for temporal alignment."""
    NEAREST = "nearest"          # Use nearest observation
    LINEAR = "linear"            # Linear interpolation between observations
    CUBIC = "cubic"              # Cubic spline interpolation
    STEP = "step"                # Forward-fill (step function)
    MEAN = "mean"                # Average within time window
    MEDIAN = "median"            # Median within time window
    WEIGHTED = "weighted"        # Quality-weighted average


class AlignmentQuality(Enum):
    """Quality levels for aligned data."""
    EXCELLENT = "excellent"      # Sub-pixel accuracy, no interpolation
    GOOD = "good"                # Sub-pixel accuracy, minor interpolation
    FAIR = "fair"                # Pixel-level accuracy, significant interpolation
    POOR = "poor"                # Multi-pixel uncertainty, heavy interpolation
    DEGRADED = "degraded"        # Large gaps, extrapolation used


@dataclass
class ReferenceGrid:
    """
    Defines the target spatial reference grid for alignment.

    Attributes:
        crs: Coordinate reference system (EPSG code or WKT)
        bounds: Bounding box (minx, miny, maxx, maxy) in CRS units
        resolution_x: Pixel size in x direction (CRS units)
        resolution_y: Pixel size in y direction (CRS units)
        width: Grid width in pixels
        height: Grid height in pixels
        nodata: Nodata value for output
    """
    crs: str
    bounds: Tuple[float, float, float, float]
    resolution_x: float
    resolution_y: float
    width: Optional[int] = None
    height: Optional[int] = None
    nodata: float = np.nan

    def __post_init__(self):
        """Calculate dimensions if not provided."""
        if self.width is None:
            self.width = int(
                (self.bounds[2] - self.bounds[0]) / abs(self.resolution_x)
            )
        if self.height is None:
            self.height = int(
                (self.bounds[3] - self.bounds[1]) / abs(self.resolution_y)
            )

    @property
    def transform(self) -> Tuple[float, float, float, float, float, float]:
        """Return affine transform (GDAL-style)."""
        return (
            self.bounds[0],              # x origin
            self.resolution_x,           # x pixel size
            0.0,                         # row rotation
            self.bounds[3],              # y origin
            0.0,                         # column rotation
            -abs(self.resolution_y)      # y pixel size (negative for north-up)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crs": self.crs,
            "bounds": self.bounds,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata,
        }


@dataclass
class TemporalBin:
    """
    Defines a time interval for temporal aggregation.

    Attributes:
        start: Bin start time (inclusive)
        end: Bin end time (exclusive)
        center: Representative timestamp for the bin
        label: Human-readable label
    """
    start: datetime
    end: datetime
    center: Optional[datetime] = None
    label: Optional[str] = None

    def __post_init__(self):
        """Calculate center if not provided."""
        if self.center is None:
            duration = self.end - self.start
            self.center = self.start + duration / 2

    @property
    def duration_hours(self) -> float:
        """Return bin duration in hours."""
        return (self.end - self.start).total_seconds() / 3600

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this bin."""
        return self.start <= timestamp < self.end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "center": self.center.isoformat() if self.center else None,
            "label": self.label,
            "duration_hours": self.duration_hours,
        }


@dataclass
class SpatialAlignmentConfig:
    """
    Configuration for spatial alignment operations.

    Attributes:
        method: Alignment method to use
        reference_grid: Target spatial reference grid
        max_offset_pixels: Maximum allowed offset for co-registration
        subpixel_precision: Enable sub-pixel precision
        nodata_handling: How to handle nodata values ("mask", "fill", "ignore")
        quality_threshold: Minimum quality for including aligned data
    """
    method: SpatialAlignmentMethod = SpatialAlignmentMethod.RESAMPLE_BILINEAR
    reference_grid: Optional[ReferenceGrid] = None
    max_offset_pixels: float = 5.0
    subpixel_precision: bool = True
    nodata_handling: str = "mask"
    quality_threshold: float = 0.5


@dataclass
class TemporalAlignmentConfig:
    """
    Configuration for temporal alignment operations.

    Attributes:
        method: Temporal alignment method
        target_timestamps: Target timestamps for alignment (or None for bins)
        bin_duration_hours: Duration of temporal bins
        max_gap_hours: Maximum gap for interpolation
        extrapolate: Allow extrapolation beyond observations
        quality_weights: Use quality metrics for weighting
    """
    method: TemporalAlignmentMethod = TemporalAlignmentMethod.LINEAR
    target_timestamps: Optional[List[datetime]] = None
    bin_duration_hours: float = 24.0
    max_gap_hours: float = 72.0
    extrapolate: bool = False
    quality_weights: bool = True


@dataclass
class AlignedLayer:
    """
    A spatially and/or temporally aligned data layer.

    Attributes:
        data: Aligned raster data array
        source_id: Identifier of source dataset
        sensor_type: Type of sensor (sar, optical, dem, etc.)
        timestamp: Observation timestamp (or bin center)
        quality_mask: Per-pixel quality mask (0-1)
        alignment_quality: Overall alignment quality
        metadata: Additional layer metadata
    """
    data: np.ndarray
    source_id: str
    sensor_type: str
    timestamp: datetime
    quality_mask: Optional[np.ndarray] = None
    alignment_quality: AlignmentQuality = AlignmentQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize quality mask if not provided."""
        if self.quality_mask is None:
            # Default to full quality where data is valid
            valid_mask = ~np.isnan(self.data) if self.data.dtype.kind == 'f' else np.ones_like(self.data, dtype=bool)
            self.quality_mask = valid_mask.astype(np.float32)

    @property
    def valid_fraction(self) -> float:
        """Return fraction of valid (non-nodata) pixels."""
        if self.quality_mask is not None:
            return float(np.mean(self.quality_mask > 0))
        return float(np.mean(~np.isnan(self.data)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data array)."""
        return {
            "source_id": self.source_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp.isoformat(),
            "alignment_quality": self.alignment_quality.value,
            "valid_fraction": self.valid_fraction,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "metadata": self.metadata,
        }


@dataclass
class AlignmentResult:
    """
    Result from an alignment operation.

    Attributes:
        layers: List of aligned data layers
        reference_grid: The reference grid used
        temporal_bins: Temporal bins used (if applicable)
        overall_quality: Combined quality assessment
        coverage_map: Per-pixel count of contributing sources
        diagnostics: Detailed alignment diagnostics
    """
    layers: List[AlignedLayer]
    reference_grid: ReferenceGrid
    temporal_bins: Optional[List[TemporalBin]] = None
    overall_quality: AlignmentQuality = AlignmentQuality.GOOD
    coverage_map: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "num_layers": len(self.layers),
            "layers": [layer.to_dict() for layer in self.layers],
            "reference_grid": self.reference_grid.to_dict(),
            "temporal_bins": [b.to_dict() for b in self.temporal_bins] if self.temporal_bins else None,
            "overall_quality": self.overall_quality.value,
            "diagnostics": self.diagnostics,
        }


class SpatialAligner:
    """
    Aligns multi-sensor data to a common spatial reference grid.

    Provides methods for:
    - Reprojecting rasters to common CRS and grid
    - Co-registration for sub-pixel alignment
    - Quality-aware resampling
    """

    def __init__(self, config: Optional[SpatialAlignmentConfig] = None):
        """
        Initialize spatial aligner.

        Args:
            config: Spatial alignment configuration
        """
        self.config = config or SpatialAlignmentConfig()

    def align(
        self,
        data: np.ndarray,
        source_crs: str,
        source_bounds: Tuple[float, float, float, float],
        source_resolution: Tuple[float, float],
        source_id: str = "unknown",
        sensor_type: str = "unknown",
        timestamp: Optional[datetime] = None,
        quality_data: Optional[np.ndarray] = None,
    ) -> AlignedLayer:
        """
        Align a single data layer to the reference grid.

        Args:
            data: Input raster data (2D or 3D array)
            source_crs: Source CRS (EPSG code or WKT)
            source_bounds: Source bounds (minx, miny, maxx, maxy)
            source_resolution: Source resolution (x, y)
            source_id: Identifier for the source dataset
            sensor_type: Type of sensor
            timestamp: Observation timestamp
            quality_data: Optional quality/weight data

        Returns:
            AlignedLayer with resampled data
        """
        if self.config.reference_grid is None:
            raise ValueError("Reference grid not configured")

        reference = self.config.reference_grid

        # Check if CRS transformation is needed
        needs_transform = not self._crs_matches(source_crs, reference.crs)

        # Calculate alignment parameters
        alignment_quality = AlignmentQuality.EXCELLENT

        if needs_transform:
            # Perform reprojection
            aligned_data, aligned_quality = self._reproject_data(
                data,
                source_crs,
                source_bounds,
                source_resolution,
                reference,
                quality_data
            )
            alignment_quality = AlignmentQuality.GOOD
        else:
            # Just resample to target grid
            aligned_data, aligned_quality = self._resample_data(
                data,
                source_bounds,
                source_resolution,
                reference,
                quality_data
            )

        # Handle nodata
        aligned_data, aligned_quality = self._handle_nodata(
            aligned_data,
            aligned_quality,
            reference.nodata
        )

        # Check offset if co-registration requested
        if self.config.method == SpatialAlignmentMethod.COREGISTER:
            offset_pixels = self._estimate_offset(aligned_data, reference)
            if offset_pixels > self.config.max_offset_pixels:
                alignment_quality = AlignmentQuality.POOR
                logger.warning(
                    f"Large offset detected: {offset_pixels:.2f} pixels for {source_id}"
                )

        return AlignedLayer(
            data=aligned_data,
            source_id=source_id,
            sensor_type=sensor_type,
            timestamp=timestamp or datetime.now(timezone.utc),
            quality_mask=aligned_quality,
            alignment_quality=alignment_quality,
            metadata={
                "source_crs": source_crs,
                "source_bounds": source_bounds,
                "source_resolution": source_resolution,
                "method": self.config.method.value,
            }
        )

    def align_multiple(
        self,
        datasets: List[Dict[str, Any]],
    ) -> List[AlignedLayer]:
        """
        Align multiple datasets to the reference grid.

        Args:
            datasets: List of dictionaries containing:
                - data: Input array
                - source_crs: CRS
                - source_bounds: Bounds
                - source_resolution: Resolution
                - source_id: ID
                - sensor_type: Sensor type
                - timestamp: Timestamp
                - quality_data: Optional quality array

        Returns:
            List of AlignedLayer objects
        """
        layers = []
        for dataset in datasets:
            layer = self.align(
                data=dataset["data"],
                source_crs=dataset["source_crs"],
                source_bounds=dataset["source_bounds"],
                source_resolution=dataset["source_resolution"],
                source_id=dataset.get("source_id", "unknown"),
                sensor_type=dataset.get("sensor_type", "unknown"),
                timestamp=dataset.get("timestamp"),
                quality_data=dataset.get("quality_data"),
            )
            layers.append(layer)
        return layers

    def create_reference_grid(
        self,
        bounds: Tuple[float, float, float, float],
        crs: str = "EPSG:4326",
        resolution: Optional[float] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> ReferenceGrid:
        """
        Create a reference grid for alignment.

        Args:
            bounds: Bounding box (minx, miny, maxx, maxy)
            crs: Target CRS
            resolution: Target resolution (same for x and y)
            target_shape: Target shape (height, width) - overrides resolution

        Returns:
            ReferenceGrid instance
        """
        if target_shape is not None:
            height, width = target_shape
            resolution_x = (bounds[2] - bounds[0]) / width
            resolution_y = (bounds[3] - bounds[1]) / height
        elif resolution is not None:
            resolution_x = resolution
            resolution_y = resolution
            width = int((bounds[2] - bounds[0]) / resolution)
            height = int((bounds[3] - bounds[1]) / resolution)
        else:
            raise ValueError("Either resolution or target_shape must be provided")

        return ReferenceGrid(
            crs=crs,
            bounds=bounds,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            width=width,
            height=height,
        )

    def _crs_matches(self, crs1: str, crs2: str) -> bool:
        """Check if two CRS definitions match."""
        # Normalize EPSG codes
        def normalize(crs: str) -> str:
            crs = crs.upper().strip()
            if crs.startswith("EPSG:"):
                return crs
            if crs.isdigit():
                return f"EPSG:{crs}"
            return crs

        return normalize(crs1) == normalize(crs2)

    def _reproject_data(
        self,
        data: np.ndarray,
        source_crs: str,
        source_bounds: Tuple[float, float, float, float],
        source_resolution: Tuple[float, float],
        reference: ReferenceGrid,
        quality_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reproject data to reference grid."""
        # Simplified reprojection using array resampling
        # In production, this would use rasterio.warp or similar

        target_shape = (reference.height, reference.width)

        # For now, use simple interpolation
        aligned_data = self._resample_array(data, target_shape)

        if quality_data is not None:
            aligned_quality = self._resample_array(quality_data, target_shape)
        else:
            aligned_quality = np.ones(target_shape, dtype=np.float32)

        # Mark areas outside source bounds as nodata
        # This is a simplified version - real implementation would
        # compute the actual footprint transformation

        return aligned_data, aligned_quality

    def _resample_data(
        self,
        data: np.ndarray,
        source_bounds: Tuple[float, float, float, float],
        source_resolution: Tuple[float, float],
        reference: ReferenceGrid,
        quality_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample data to reference grid (same CRS)."""
        target_shape = (reference.height, reference.width)

        # Compute overlapping region
        overlap_bounds = self._compute_overlap(source_bounds, reference.bounds)
        if overlap_bounds is None:
            # No overlap - return empty
            empty_data = np.full(target_shape, reference.nodata)
            empty_quality = np.zeros(target_shape, dtype=np.float32)
            return empty_data, empty_quality

        # Resample to target
        aligned_data = self._resample_array(data, target_shape)

        if quality_data is not None:
            aligned_quality = self._resample_array(quality_data, target_shape)
        else:
            aligned_quality = np.ones(target_shape, dtype=np.float32)

        return aligned_data, aligned_quality

    def _resample_array(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resample array to target shape using configured method.

        Uses numpy/scipy for basic interpolation.
        """
        from scipy import ndimage

        if data.shape == target_shape:
            return data.copy()

        # Calculate zoom factors
        zoom_factors = (
            target_shape[0] / data.shape[0],
            target_shape[1] / data.shape[1],
        )

        # Handle multi-band data
        if len(data.shape) == 3:
            zoom_factors = (*zoom_factors, 1)

        # Choose interpolation order based on method
        method = self.config.method
        if method == SpatialAlignmentMethod.RESAMPLE_NEAREST:
            order = 0
        elif method == SpatialAlignmentMethod.RESAMPLE_BILINEAR:
            order = 1
        elif method == SpatialAlignmentMethod.RESAMPLE_CUBIC:
            order = 3
        else:
            order = 1  # Default to bilinear

        # Handle nodata before resampling
        nodata_mask = np.isnan(data) if data.dtype.kind == 'f' else None

        # Perform resampling
        resampled = ndimage.zoom(data.astype(np.float64), zoom_factors, order=order)

        # Restore nodata mask if present
        if nodata_mask is not None and nodata_mask.any():
            # Resample mask too
            mask_resampled = ndimage.zoom(
                nodata_mask.astype(np.float64),
                zoom_factors,
                order=0
            )
            resampled[mask_resampled > 0.5] = np.nan

        return resampled.astype(data.dtype)

    def _compute_overlap(
        self,
        bounds1: Tuple[float, float, float, float],
        bounds2: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute overlap between two bounding boxes."""
        minx = max(bounds1[0], bounds2[0])
        miny = max(bounds1[1], bounds2[1])
        maxx = min(bounds1[2], bounds2[2])
        maxy = min(bounds1[3], bounds2[3])

        if minx >= maxx or miny >= maxy:
            return None

        return (minx, miny, maxx, maxy)

    def _handle_nodata(
        self,
        data: np.ndarray,
        quality: np.ndarray,
        nodata_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle nodata values according to configuration."""
        # Find nodata locations
        if np.isnan(nodata_value):
            nodata_mask = np.isnan(data)
        else:
            nodata_mask = data == nodata_value

        handling = self.config.nodata_handling

        if handling == "mask":
            # Set quality to 0 for nodata pixels
            quality = quality.copy()
            quality[nodata_mask] = 0.0

        elif handling == "fill":
            # Fill nodata with nearest valid value
            if nodata_mask.any():
                from scipy import ndimage
                indices = ndimage.distance_transform_edt(
                    nodata_mask,
                    return_distances=False,
                    return_indices=True
                )
                data = data.copy()
                data[nodata_mask] = data[tuple(indices[:, nodata_mask])]
                quality = quality.copy()
                quality[nodata_mask] = 0.5  # Reduced quality for filled pixels

        # "ignore" handling doesn't modify anything

        return data, quality

    def _estimate_offset(
        self,
        data: np.ndarray,
        reference: ReferenceGrid,
    ) -> float:
        """
        Estimate spatial offset for co-registration.

        Returns estimated offset in pixels.
        """
        # Simplified - real implementation would use phase correlation
        # or feature matching
        return 0.0


class TemporalAligner:
    """
    Aligns multi-temporal observations to common time references.

    Provides methods for:
    - Interpolating to specific timestamps
    - Binning observations into time intervals
    - Gap-filling with quality tracking
    """

    def __init__(self, config: Optional[TemporalAlignmentConfig] = None):
        """
        Initialize temporal aligner.

        Args:
            config: Temporal alignment configuration
        """
        self.config = config or TemporalAlignmentConfig()

    def align_to_timestamps(
        self,
        layers: List[AlignedLayer],
        target_timestamps: List[datetime],
    ) -> List[AlignedLayer]:
        """
        Align layers to specific target timestamps.

        Args:
            layers: Input aligned layers with timestamps
            target_timestamps: Target timestamps for output

        Returns:
            List of aligned layers at target timestamps
        """
        if not layers:
            return []

        # Sort layers by timestamp
        sorted_layers = sorted(layers, key=lambda x: x.timestamp)

        result = []
        for target_ts in target_timestamps:
            aligned_layer = self._interpolate_to_timestamp(
                sorted_layers,
                target_ts
            )
            if aligned_layer is not None:
                result.append(aligned_layer)

        return result

    def align_to_bins(
        self,
        layers: List[AlignedLayer],
        start_time: datetime,
        end_time: datetime,
        bin_duration_hours: Optional[float] = None,
    ) -> Tuple[List[AlignedLayer], List[TemporalBin]]:
        """
        Bin observations into time intervals.

        Args:
            layers: Input aligned layers
            start_time: Start of temporal range
            end_time: End of temporal range
            bin_duration_hours: Duration of each bin (default from config)

        Returns:
            Tuple of (aligned layers, temporal bins)
        """
        duration = bin_duration_hours or self.config.bin_duration_hours

        # Create temporal bins
        bins = self._create_bins(start_time, end_time, duration)

        # Assign layers to bins and aggregate
        result_layers = []
        for temporal_bin in bins:
            bin_layers = [
                layer for layer in layers
                if temporal_bin.contains(layer.timestamp)
            ]

            if bin_layers:
                aggregated = self._aggregate_layers(bin_layers, temporal_bin)
                result_layers.append(aggregated)

        return result_layers, bins

    def create_dense_timeseries(
        self,
        layers: List[AlignedLayer],
        time_step_hours: float,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AlignedLayer]:
        """
        Create dense time series through interpolation.

        Args:
            layers: Sparse input layers
            time_step_hours: Time step for dense output
            start_time: Start time (default: first observation)
            end_time: End time (default: last observation)

        Returns:
            Dense time series of aligned layers
        """
        if not layers:
            return []

        sorted_layers = sorted(layers, key=lambda x: x.timestamp)

        if start_time is None:
            start_time = sorted_layers[0].timestamp
        if end_time is None:
            end_time = sorted_layers[-1].timestamp

        # Generate target timestamps
        timestamps = []
        current = start_time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(hours=time_step_hours)

        return self.align_to_timestamps(sorted_layers, timestamps)

    def _create_bins(
        self,
        start_time: datetime,
        end_time: datetime,
        duration_hours: float,
    ) -> List[TemporalBin]:
        """Create temporal bins covering the time range."""
        bins = []
        current = start_time
        duration = timedelta(hours=duration_hours)
        bin_num = 0

        while current < end_time:
            bin_end = min(current + duration, end_time)
            bins.append(TemporalBin(
                start=current,
                end=bin_end,
                label=f"bin_{bin_num}"
            ))
            current = bin_end
            bin_num += 1

        return bins

    def _interpolate_to_timestamp(
        self,
        sorted_layers: List[AlignedLayer],
        target_ts: datetime,
    ) -> Optional[AlignedLayer]:
        """Interpolate to a single target timestamp."""
        if not sorted_layers:
            return None

        # Find bracketing observations
        before = None
        after = None

        for layer in sorted_layers:
            if layer.timestamp <= target_ts:
                before = layer
            else:
                if after is None:
                    after = layer
                break

        # Handle edge cases
        if before is None and after is None:
            return None

        if before is None:
            # Target is before all observations
            if not self.config.extrapolate:
                return None
            return self._extrapolate(after, target_ts, "before")

        if after is None:
            # Target is after all observations
            if not self.config.extrapolate:
                return None
            return self._extrapolate(before, target_ts, "after")

        # Check if gap is too large
        gap_hours = (after.timestamp - before.timestamp).total_seconds() / 3600
        if gap_hours > self.config.max_gap_hours:
            logger.warning(
                f"Gap of {gap_hours:.1f} hours exceeds max {self.config.max_gap_hours}"
            )
            # Use nearest instead of interpolating across large gap
            return self._use_nearest(before, after, target_ts)

        # Perform interpolation based on method
        method = self.config.method

        if method == TemporalAlignmentMethod.NEAREST:
            return self._use_nearest(before, after, target_ts)
        elif method == TemporalAlignmentMethod.LINEAR:
            return self._interpolate_linear(before, after, target_ts)
        elif method == TemporalAlignmentMethod.STEP:
            return self._interpolate_step(before, target_ts)
        else:
            # Default to linear
            return self._interpolate_linear(before, after, target_ts)

    def _use_nearest(
        self,
        before: AlignedLayer,
        after: AlignedLayer,
        target_ts: datetime,
    ) -> AlignedLayer:
        """Use nearest observation."""
        dt_before = abs((target_ts - before.timestamp).total_seconds())
        dt_after = abs((after.timestamp - target_ts).total_seconds())

        nearest = before if dt_before <= dt_after else after

        return AlignedLayer(
            data=nearest.data.copy(),
            source_id=f"nearest_{nearest.source_id}",
            sensor_type=nearest.sensor_type,
            timestamp=target_ts,
            quality_mask=nearest.quality_mask.copy() if nearest.quality_mask is not None else None,
            alignment_quality=nearest.alignment_quality,
            metadata={
                **nearest.metadata,
                "temporal_method": "nearest",
                "source_timestamp": nearest.timestamp.isoformat(),
            }
        )

    def _interpolate_linear(
        self,
        before: AlignedLayer,
        after: AlignedLayer,
        target_ts: datetime,
    ) -> AlignedLayer:
        """Linearly interpolate between two observations."""
        # Calculate interpolation weight
        total_dt = (after.timestamp - before.timestamp).total_seconds()
        target_dt = (target_ts - before.timestamp).total_seconds()

        if total_dt == 0:
            weight_after = 0.5
        else:
            weight_after = target_dt / total_dt
        weight_before = 1.0 - weight_after

        # Interpolate data
        interpolated = (
            weight_before * before.data.astype(np.float64) +
            weight_after * after.data.astype(np.float64)
        )

        # Combine quality masks
        if before.quality_mask is not None and after.quality_mask is not None:
            combined_quality = np.minimum(before.quality_mask, after.quality_mask)
        else:
            combined_quality = None

        # Interpolated data has reduced quality
        quality_reduction = 1.0 - 0.3 * min(total_dt / 3600, 24) / 24  # Max 30% reduction for 24h gap

        return AlignedLayer(
            data=interpolated.astype(before.data.dtype),
            source_id=f"interp_{before.source_id}_{after.source_id}",
            sensor_type=before.sensor_type,
            timestamp=target_ts,
            quality_mask=combined_quality * quality_reduction if combined_quality is not None else None,
            alignment_quality=AlignmentQuality.FAIR,
            metadata={
                "temporal_method": "linear",
                "source_before": before.timestamp.isoformat(),
                "source_after": after.timestamp.isoformat(),
                "weight_before": weight_before,
                "weight_after": weight_after,
            }
        )

    def _interpolate_step(
        self,
        before: AlignedLayer,
        target_ts: datetime,
    ) -> AlignedLayer:
        """Use step function (forward fill)."""
        return AlignedLayer(
            data=before.data.copy(),
            source_id=f"step_{before.source_id}",
            sensor_type=before.sensor_type,
            timestamp=target_ts,
            quality_mask=before.quality_mask.copy() if before.quality_mask is not None else None,
            alignment_quality=before.alignment_quality,
            metadata={
                **before.metadata,
                "temporal_method": "step",
                "source_timestamp": before.timestamp.isoformat(),
            }
        )

    def _extrapolate(
        self,
        layer: AlignedLayer,
        target_ts: datetime,
        direction: str,
    ) -> AlignedLayer:
        """Extrapolate beyond observations (with quality penalty)."""
        gap_hours = abs((target_ts - layer.timestamp).total_seconds()) / 3600

        return AlignedLayer(
            data=layer.data.copy(),
            source_id=f"extrap_{layer.source_id}",
            sensor_type=layer.sensor_type,
            timestamp=target_ts,
            quality_mask=layer.quality_mask * 0.5 if layer.quality_mask is not None else None,  # 50% quality for extrapolation
            alignment_quality=AlignmentQuality.POOR,
            metadata={
                **layer.metadata,
                "temporal_method": f"extrapolate_{direction}",
                "source_timestamp": layer.timestamp.isoformat(),
                "extrapolation_hours": gap_hours,
            }
        )

    def _aggregate_layers(
        self,
        layers: List[AlignedLayer],
        temporal_bin: TemporalBin,
    ) -> AlignedLayer:
        """Aggregate multiple layers within a temporal bin."""
        if len(layers) == 1:
            # Single layer - just update timestamp
            layer = layers[0]
            return AlignedLayer(
                data=layer.data.copy(),
                source_id=f"bin_{layer.source_id}",
                sensor_type=layer.sensor_type,
                timestamp=temporal_bin.center,
                quality_mask=layer.quality_mask.copy() if layer.quality_mask is not None else None,
                alignment_quality=layer.alignment_quality,
                metadata={
                    **layer.metadata,
                    "temporal_method": "single_in_bin",
                    "bin_label": temporal_bin.label,
                }
            )

        # Multiple layers - aggregate
        method = self.config.method

        if method == TemporalAlignmentMethod.MEAN:
            aggregated = self._aggregate_mean(layers)
        elif method == TemporalAlignmentMethod.MEDIAN:
            aggregated = self._aggregate_median(layers)
        elif method == TemporalAlignmentMethod.WEIGHTED:
            aggregated = self._aggregate_weighted(layers)
        else:
            # Default to mean
            aggregated = self._aggregate_mean(layers)

        # Combine quality masks
        quality_masks = [l.quality_mask for l in layers if l.quality_mask is not None]
        if quality_masks:
            combined_quality = np.mean(np.stack(quality_masks), axis=0)
        else:
            combined_quality = None

        return AlignedLayer(
            data=aggregated,
            source_id=f"bin_{temporal_bin.label}",
            sensor_type=layers[0].sensor_type,
            timestamp=temporal_bin.center,
            quality_mask=combined_quality,
            alignment_quality=AlignmentQuality.GOOD,
            metadata={
                "temporal_method": method.value if hasattr(method, 'value') else str(method),
                "bin_label": temporal_bin.label,
                "num_sources": len(layers),
                "source_ids": [l.source_id for l in layers],
            }
        )

    def _aggregate_mean(self, layers: List[AlignedLayer]) -> np.ndarray:
        """Compute mean of layers."""
        stacked = np.stack([l.data.astype(np.float64) for l in layers])
        return np.nanmean(stacked, axis=0)

    def _aggregate_median(self, layers: List[AlignedLayer]) -> np.ndarray:
        """Compute median of layers."""
        stacked = np.stack([l.data.astype(np.float64) for l in layers])
        return np.nanmedian(stacked, axis=0)

    def _aggregate_weighted(self, layers: List[AlignedLayer]) -> np.ndarray:
        """Compute quality-weighted mean of layers."""
        data_stack = np.stack([l.data.astype(np.float64) for l in layers])

        # Get weights from quality masks
        if all(l.quality_mask is not None for l in layers):
            weight_stack = np.stack([l.quality_mask for l in layers])
        else:
            # Equal weights if quality not available
            weight_stack = np.ones_like(data_stack)

        # Avoid division by zero
        weight_sum = np.sum(weight_stack, axis=0)
        weight_sum[weight_sum == 0] = 1.0

        weighted_sum = np.sum(data_stack * weight_stack, axis=0)
        return weighted_sum / weight_sum


class MultiSensorAligner:
    """
    Coordinates alignment of multiple sensors to a common reference.

    Combines spatial and temporal alignment for multi-sensor fusion.
    """

    def __init__(
        self,
        spatial_config: Optional[SpatialAlignmentConfig] = None,
        temporal_config: Optional[TemporalAlignmentConfig] = None,
    ):
        """
        Initialize multi-sensor aligner.

        Args:
            spatial_config: Spatial alignment configuration
            temporal_config: Temporal alignment configuration
        """
        self.spatial_aligner = SpatialAligner(spatial_config)
        self.temporal_aligner = TemporalAligner(temporal_config)

    def align(
        self,
        datasets: List[Dict[str, Any]],
        reference_grid: ReferenceGrid,
        target_timestamps: Optional[List[datetime]] = None,
        temporal_bins: Optional[Tuple[datetime, datetime, float]] = None,
    ) -> AlignmentResult:
        """
        Align multiple datasets to common spatial and temporal references.

        Args:
            datasets: List of dataset dictionaries with:
                - data: Array data
                - source_crs: CRS
                - source_bounds: Bounds
                - source_resolution: Resolution
                - source_id: ID
                - sensor_type: Sensor type
                - timestamp: Observation timestamp
                - quality_data: Optional quality array
            reference_grid: Target spatial reference
            target_timestamps: Target timestamps for alignment (optional)
            temporal_bins: Tuple of (start, end, duration_hours) for binning

        Returns:
            AlignmentResult with aligned layers
        """
        # Configure spatial aligner
        self.spatial_aligner.config.reference_grid = reference_grid

        # Phase 1: Spatial alignment
        spatially_aligned = self.spatial_aligner.align_multiple(datasets)

        # Phase 2: Temporal alignment
        if target_timestamps is not None:
            # Align to specific timestamps
            final_layers = self.temporal_aligner.align_to_timestamps(
                spatially_aligned,
                target_timestamps
            )
            bins = None

        elif temporal_bins is not None:
            # Align to temporal bins
            start, end, duration = temporal_bins
            final_layers, bins = self.temporal_aligner.align_to_bins(
                spatially_aligned,
                start,
                end,
                duration
            )

        else:
            # No temporal alignment - keep original timestamps
            final_layers = spatially_aligned
            bins = None

        # Calculate coverage map
        coverage_map = self._calculate_coverage(final_layers)

        # Determine overall quality
        overall_quality = self._assess_overall_quality(final_layers)

        # Build diagnostics
        diagnostics = {
            "num_input_datasets": len(datasets),
            "num_output_layers": len(final_layers),
            "spatial_method": self.spatial_aligner.config.method.value,
            "temporal_method": self.temporal_aligner.config.method.value,
            "sensor_types": list(set(l.sensor_type for l in final_layers)),
        }

        return AlignmentResult(
            layers=final_layers,
            reference_grid=reference_grid,
            temporal_bins=bins,
            overall_quality=overall_quality,
            coverage_map=coverage_map,
            diagnostics=diagnostics,
        )

    def _calculate_coverage(self, layers: List[AlignedLayer]) -> np.ndarray:
        """Calculate per-pixel count of contributing sources."""
        if not layers:
            return np.array([])

        shape = layers[0].data.shape
        coverage = np.zeros(shape, dtype=np.int32)

        for layer in layers:
            if layer.quality_mask is not None:
                coverage += (layer.quality_mask > 0).astype(np.int32)
            else:
                valid = ~np.isnan(layer.data) if layer.data.dtype.kind == 'f' else np.ones_like(layer.data, dtype=bool)
                coverage += valid.astype(np.int32)

        return coverage

    def _assess_overall_quality(
        self,
        layers: List[AlignedLayer],
    ) -> AlignmentQuality:
        """Assess overall alignment quality from layers."""
        if not layers:
            return AlignmentQuality.DEGRADED

        quality_scores = {
            AlignmentQuality.EXCELLENT: 4,
            AlignmentQuality.GOOD: 3,
            AlignmentQuality.FAIR: 2,
            AlignmentQuality.POOR: 1,
            AlignmentQuality.DEGRADED: 0,
        }

        scores = [quality_scores[l.alignment_quality] for l in layers]
        avg_score = sum(scores) / len(scores)

        if avg_score >= 3.5:
            return AlignmentQuality.EXCELLENT
        elif avg_score >= 2.5:
            return AlignmentQuality.GOOD
        elif avg_score >= 1.5:
            return AlignmentQuality.FAIR
        elif avg_score >= 0.5:
            return AlignmentQuality.POOR
        else:
            return AlignmentQuality.DEGRADED


# Convenience functions

def create_reference_grid(
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    resolution: Optional[float] = None,
    target_shape: Optional[Tuple[int, int]] = None,
) -> ReferenceGrid:
    """
    Create a reference grid for alignment.

    Args:
        bounds: Bounding box (minx, miny, maxx, maxy)
        crs: Target CRS (default WGS84)
        resolution: Target resolution
        target_shape: Target shape (height, width)

    Returns:
        ReferenceGrid instance
    """
    aligner = SpatialAligner()
    return aligner.create_reference_grid(bounds, crs, resolution, target_shape)


def align_datasets(
    datasets: List[Dict[str, Any]],
    reference_grid: ReferenceGrid,
    target_timestamps: Optional[List[datetime]] = None,
) -> AlignmentResult:
    """
    Convenience function to align multiple datasets.

    Args:
        datasets: List of dataset dictionaries
        reference_grid: Target reference grid
        target_timestamps: Optional target timestamps

    Returns:
        AlignmentResult
    """
    aligner = MultiSensorAligner()
    return aligner.align(datasets, reference_grid, target_timestamps)
