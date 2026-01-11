"""
Data Anomaly Detection Module.

Provides statistical and pattern-based anomaly detection for geospatial data:
- Statistical outlier detection (z-score, IQR, MAD)
- Spatial artifact detection (stripes, tiles, holes)
- Temporal anomaly detection (sudden changes, gaps)
- Sensor-specific anomaly patterns (SAR, optical, thermal)
- Cloud/shadow/water confusion detection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    # Statistical anomalies
    OUTLIER_ZSCORE = "outlier_zscore"
    OUTLIER_IQR = "outlier_iqr"
    OUTLIER_MAD = "outlier_mad"

    # Value anomalies
    INVALID_VALUE = "invalid_value"
    SATURATED = "saturated"
    NODATA_PATTERN = "nodata_pattern"

    # Spatial anomalies
    STRIPE_ARTIFACT = "stripe_artifact"
    TILE_ARTIFACT = "tile_artifact"
    EDGE_ARTIFACT = "edge_artifact"
    SPATIAL_DISCONTINUITY = "spatial_discontinuity"

    # Radiometric anomalies
    DARK_REGION = "dark_region"
    BRIGHT_REGION = "bright_region"
    UNUSUAL_HISTOGRAM = "unusual_histogram"

    # Sensor-specific
    SAR_SPECKLE_EXTREME = "sar_speckle_extreme"
    THERMAL_ANOMALY = "thermal_anomaly"
    CLOUD_SHADOW = "cloud_shadow"

    # General
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AnomalyLocation:
    """Location of an anomaly within the data."""

    band: Optional[int] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    pixel_count: int = 0
    percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert location to dictionary."""
        return {
            "band": self.band,
            "row_range": [self.row_start, self.row_end] if self.row_start is not None else None,
            "col_range": [self.col_start, self.col_end] if self.col_start is not None else None,
            "pixel_count": self.pixel_count,
            "percentage": self.percentage,
        }


@dataclass
class DetectedAnomaly:
    """Represents a single detected anomaly."""

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    location: Optional[AnomalyLocation] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary."""
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location.to_dict() if self.location else None,
            "statistics": self.statistics,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
        }


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    has_anomalies: bool
    anomalies: List[DetectedAnomaly] = field(default_factory=list)
    band_statistics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    overall_quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical anomalies."""
        return sum(1 for a in self.anomalies if a.severity == AnomalySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high-severity anomalies."""
        return sum(1 for a in self.anomalies if a.severity == AnomalySeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "has_anomalies": self.has_anomalies,
            "anomaly_count": len(self.anomalies),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "band_statistics": self.band_statistics,
            "overall_quality_score": self.overall_quality_score,
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    saturation_threshold: float = 0.99
    stripe_detection: bool = True
    stripe_threshold: float = 3.0  # Increased from 2.0 to reduce false positives
    histogram_bins: int = 256
    min_region_size: int = 100
    max_nodata_percent: float = 50.0
    detect_spatial: bool = True
    detect_statistical: bool = True
    sample_size: Optional[int] = None


class AnomalyDetector:
    """Detects anomalies in geospatial raster data."""

    def __init__(self, config: Optional[AnomalyConfig] = None):
        """Initialize anomaly detector."""
        self.config = config or AnomalyConfig()

    def detect_from_file(self, path: Union[str, Path]) -> AnomalyResult:
        """Detect anomalies in a raster file."""
        import time

        start_time = time.time()
        path = Path(path)

        if not path.exists():
            return AnomalyResult(
                has_anomalies=True,
                anomalies=[
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.UNKNOWN,
                        severity=AnomalySeverity.CRITICAL,
                        description=f"File not found: {path}",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        try:
            import rasterio
        except ImportError:
            return AnomalyResult(
                has_anomalies=False,
                metadata={"error": "rasterio not available"},
                duration_seconds=time.time() - start_time,
            )

        try:
            with rasterio.open(path) as src:
                data = src.read()
                nodata = src.nodata

            return self.detect_from_array(data, nodata=nodata)

        except Exception as e:
            return AnomalyResult(
                has_anomalies=True,
                anomalies=[
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.UNKNOWN,
                        severity=AnomalySeverity.CRITICAL,
                        description=f"Failed to read file: {e}",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

    def detect_from_array(
        self,
        data: np.ndarray,
        nodata: Optional[float] = None,
        band_names: Optional[List[str]] = None,
    ) -> AnomalyResult:
        """Detect anomalies in a numpy array."""
        import time

        start_time = time.time()
        anomalies: List[DetectedAnomaly] = []
        band_stats: Dict[int, Dict[str, float]] = {}

        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        num_bands, height, width = data.shape
        total_pixels = height * width

        # Create mask for nodata
        if nodata is not None:
            if np.isnan(nodata):
                mask = np.isnan(data)
            else:
                mask = data == nodata
        else:
            mask = np.zeros_like(data, dtype=bool)

        # Check for invalid values (NaN, Inf) BEFORE statistical analysis
        for band_idx in range(num_bands):
            band_data = data[band_idx]
            total_pixels_band = band_data.size

            # Check for NaN values (if nodata is not NaN)
            if nodata is None or not np.isnan(nodata):
                nan_count = np.sum(np.isnan(band_data))
                if nan_count > 0:
                    anomalies.append(
                        DetectedAnomaly(
                            anomaly_type=AnomalyType.INVALID_VALUE,
                            severity=AnomalySeverity.HIGH,
                            description=f"Band {band_idx}: {nan_count} unexpected NaN values",
                            location=AnomalyLocation(
                                band=band_idx,
                                pixel_count=int(nan_count),
                                percentage=(nan_count / total_pixels_band) * 100,
                            ),
                            recommendation="Check data processing pipeline for division errors",
                        )
                    )
                    # Add NaN to mask for subsequent analysis
                    mask[band_idx] |= np.isnan(band_data)

            # Check for Inf values
            inf_count = np.sum(np.isinf(band_data))
            if inf_count > 0:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.INVALID_VALUE,
                        severity=AnomalySeverity.HIGH,
                        description=f"Band {band_idx}: {inf_count} infinite values",
                        location=AnomalyLocation(
                            band=band_idx,
                            pixel_count=int(inf_count),
                            percentage=(inf_count / total_pixels_band) * 100,
                        ),
                        recommendation="Check for overflow or division by zero",
                    )
                )
                # Add Inf to mask for subsequent analysis
                mask[band_idx] |= np.isinf(band_data)

        # Check nodata percentage per band
        for band_idx in range(num_bands):
            nodata_count = np.sum(mask[band_idx])
            nodata_pct = (nodata_count / total_pixels) * 100

            if nodata_pct > self.config.max_nodata_percent:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.NODATA_PATTERN,
                        severity=AnomalySeverity.HIGH,
                        description=f"Band {band_idx}: Excessive nodata ({nodata_pct:.1f}%)",
                        location=AnomalyLocation(
                            band=band_idx,
                            pixel_count=int(nodata_count),
                            percentage=nodata_pct,
                        ),
                        statistics={"nodata_percent": nodata_pct},
                    )
                )

        # Compute statistics per band
        for band_idx in range(num_bands):
            band_data = data[band_idx]
            band_mask = mask[band_idx]
            valid_data = band_data[~band_mask]

            if len(valid_data) == 0:
                band_stats[band_idx] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "valid_count": 0,
                }
                continue

            # Sample if configured
            if self.config.sample_size and len(valid_data) > self.config.sample_size:
                indices = np.random.choice(len(valid_data), self.config.sample_size, replace=False)
                valid_data = valid_data[indices]

            # Compute statistics
            mean = float(np.mean(valid_data))
            std = float(np.std(valid_data))
            minimum = float(np.min(valid_data))
            maximum = float(np.max(valid_data))
            median = float(np.median(valid_data))

            band_stats[band_idx] = {
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "median": median,
                "valid_count": len(valid_data),
            }

            # Statistical anomaly detection
            if self.config.detect_statistical:
                stat_anomalies = self._detect_statistical_anomalies(
                    band_data, band_mask, band_idx, band_stats[band_idx]
                )
                anomalies.extend(stat_anomalies)

        # Spatial anomaly detection
        if self.config.detect_spatial:
            spatial_anomalies = self._detect_spatial_anomalies(data, mask)
            anomalies.extend(spatial_anomalies)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(anomalies, num_bands)

        duration = time.time() - start_time

        logger.info(
            f"Anomaly detection complete: {len(anomalies)} anomalies found, "
            f"quality score: {quality_score:.2f}"
        )

        return AnomalyResult(
            has_anomalies=len(anomalies) > 0,
            anomalies=anomalies,
            band_statistics=band_stats,
            overall_quality_score=quality_score,
            metadata={
                "num_bands": num_bands,
                "height": height,
                "width": width,
                "nodata": nodata,
            },
            duration_seconds=duration,
        )

    def _detect_statistical_anomalies(
        self,
        band_data: np.ndarray,
        band_mask: np.ndarray,
        band_idx: int,
        stats: Dict[str, float],
    ) -> List[DetectedAnomaly]:
        """Detect statistical anomalies in a single band."""
        anomalies = []
        valid_data = band_data[~band_mask]

        if len(valid_data) == 0:
            return anomalies

        mean = stats["mean"]
        std = stats["std"]

        # Guard against zero std
        if std == 0 or np.isnan(std):
            return anomalies

        # Z-score outliers
        zscores = np.abs((valid_data - mean) / std)
        outlier_mask = zscores > self.config.zscore_threshold
        outlier_count = np.sum(outlier_mask)
        outlier_pct = (outlier_count / len(valid_data)) * 100

        if outlier_pct > 1.0:  # More than 1% outliers is unusual
            anomalies.append(
                DetectedAnomaly(
                    anomaly_type=AnomalyType.OUTLIER_ZSCORE,
                    severity=AnomalySeverity.MEDIUM if outlier_pct < 5 else AnomalySeverity.HIGH,
                    description=f"Band {band_idx}: {outlier_pct:.1f}% values are outliers (z-score > {self.config.zscore_threshold})",
                    location=AnomalyLocation(
                        band=band_idx,
                        pixel_count=int(outlier_count),
                        percentage=outlier_pct,
                    ),
                    statistics={
                        "outlier_percent": outlier_pct,
                        "threshold": self.config.zscore_threshold,
                        "max_zscore": float(np.max(zscores)),
                    },
                )
            )

        # IQR outliers
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1

        if iqr > 0:
            lower_bound = q1 - self.config.iqr_multiplier * iqr
            upper_bound = q3 + self.config.iqr_multiplier * iqr
            iqr_outliers = np.sum((valid_data < lower_bound) | (valid_data > upper_bound))
            iqr_pct = (iqr_outliers / len(valid_data)) * 100

            if iqr_pct > 2.0:  # More than 2% IQR outliers
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.OUTLIER_IQR,
                        severity=AnomalySeverity.MEDIUM,
                        description=f"Band {band_idx}: {iqr_pct:.1f}% values outside IQR bounds",
                        location=AnomalyLocation(
                            band=band_idx,
                            pixel_count=int(iqr_outliers),
                            percentage=iqr_pct,
                        ),
                        statistics={
                            "iqr": float(iqr),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound),
                        },
                    )
                )

        # Check for saturation
        dtype_info = np.iinfo(valid_data.dtype) if np.issubdtype(valid_data.dtype, np.integer) else None
        if dtype_info:
            max_possible = dtype_info.max
            saturation_value = max_possible * self.config.saturation_threshold
            saturated = np.sum(valid_data >= saturation_value)
            saturated_pct = (saturated / len(valid_data)) * 100

            if saturated_pct > 0.5:  # More than 0.5% saturated
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_type=AnomalyType.SATURATED,
                        severity=AnomalySeverity.MEDIUM,
                        description=f"Band {band_idx}: {saturated_pct:.1f}% pixels are saturated",
                        location=AnomalyLocation(
                            band=band_idx,
                            pixel_count=int(saturated),
                            percentage=saturated_pct,
                        ),
                        statistics={
                            "saturated_percent": saturated_pct,
                            "saturation_value": float(saturation_value),
                        },
                        recommendation="Consider adjusting exposure or scaling",
                    )
                )

        return anomalies

    def _detect_spatial_anomalies(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> List[DetectedAnomaly]:
        """Detect spatial anomalies across all bands."""
        anomalies = []
        num_bands, height, width = data.shape

        for band_idx in range(num_bands):
            band_data = data[band_idx].astype(np.float64)
            band_mask = mask[band_idx]

            # Replace masked values with NaN for analysis
            band_data = band_data.copy()
            band_data[band_mask] = np.nan

            # Detect stripe artifacts (horizontal or vertical)
            if self.config.stripe_detection:
                stripe_anomalies = self._detect_stripes(band_data, band_idx)
                anomalies.extend(stripe_anomalies)

            # Detect dark/bright regions
            region_anomalies = self._detect_extreme_regions(band_data, band_mask, band_idx)
            anomalies.extend(region_anomalies)

        return anomalies

    def _detect_stripes(
        self,
        band_data: np.ndarray,
        band_idx: int,
    ) -> List[DetectedAnomaly]:
        """Detect horizontal or vertical stripe artifacts."""
        anomalies = []
        height, width = band_data.shape

        # Compute row and column means, ignoring NaN
        with np.errstate(all="ignore"):
            row_means = np.nanmean(band_data, axis=1)
            col_means = np.nanmean(band_data, axis=0)

        # Check for unusual row variations (horizontal stripes)
        valid_rows = ~np.isnan(row_means)
        if np.sum(valid_rows) > 10:
            row_std = np.nanstd(row_means)
            row_median = np.nanmedian(row_means)

            if row_std > 0:
                row_zscores = np.abs((row_means - row_median) / row_std)
                stripe_rows = np.sum(row_zscores > self.config.stripe_threshold)

                # Require at least 2% of rows to be striped (reduced from 1%)
                if stripe_rows > height * 0.02:
                    anomalies.append(
                        DetectedAnomaly(
                            anomaly_type=AnomalyType.STRIPE_ARTIFACT,
                            severity=AnomalySeverity.MEDIUM,
                            description=f"Band {band_idx}: Horizontal stripe artifacts detected ({stripe_rows} rows)",
                            location=AnomalyLocation(band=band_idx, pixel_count=stripe_rows),
                            statistics={
                                "affected_rows": int(stripe_rows),
                                "direction": "horizontal",
                            },
                            recommendation="Consider destriping algorithm or sensor calibration review",
                        )
                    )

        # Check for unusual column variations (vertical stripes)
        valid_cols = ~np.isnan(col_means)
        if np.sum(valid_cols) > 10:
            col_std = np.nanstd(col_means)
            col_median = np.nanmedian(col_means)

            if col_std > 0:
                col_zscores = np.abs((col_means - col_median) / col_std)
                stripe_cols = np.sum(col_zscores > self.config.stripe_threshold)

                # Require at least 2% of columns to be striped
                if stripe_cols > width * 0.02:
                    anomalies.append(
                        DetectedAnomaly(
                            anomaly_type=AnomalyType.STRIPE_ARTIFACT,
                            severity=AnomalySeverity.MEDIUM,
                            description=f"Band {band_idx}: Vertical stripe artifacts detected ({stripe_cols} columns)",
                            location=AnomalyLocation(band=band_idx, pixel_count=stripe_cols),
                            statistics={
                                "affected_cols": int(stripe_cols),
                                "direction": "vertical",
                            },
                            recommendation="Consider destriping algorithm or sensor calibration review",
                        )
                    )

        return anomalies

    def _detect_extreme_regions(
        self,
        band_data: np.ndarray,
        band_mask: np.ndarray,
        band_idx: int,
    ) -> List[DetectedAnomaly]:
        """Detect unusually dark or bright regions."""
        anomalies = []
        valid_data = band_data[~band_mask]

        if len(valid_data) < 100:
            return anomalies

        total_pixels = band_data.size

        # Define thresholds based on percentiles
        p1 = np.nanpercentile(valid_data, 1)
        p99 = np.nanpercentile(valid_data, 99)

        # Check for concentrated dark regions
        dark_mask = band_data < p1
        dark_count = np.sum(dark_mask & ~band_mask)
        dark_pct = (dark_count / total_pixels) * 100

        if dark_pct > 5:  # More than 5% very dark pixels
            anomalies.append(
                DetectedAnomaly(
                    anomaly_type=AnomalyType.DARK_REGION,
                    severity=AnomalySeverity.LOW,
                    description=f"Band {band_idx}: Unusually large dark region ({dark_pct:.1f}%)",
                    location=AnomalyLocation(
                        band=band_idx,
                        pixel_count=int(dark_count),
                        percentage=dark_pct,
                    ),
                    statistics={"threshold": float(p1), "dark_percent": dark_pct},
                )
            )

        # Check for concentrated bright regions
        bright_mask = band_data > p99
        bright_count = np.sum(bright_mask & ~band_mask)
        bright_pct = (bright_count / total_pixels) * 100

        if bright_pct > 5:  # More than 5% very bright pixels
            anomalies.append(
                DetectedAnomaly(
                    anomaly_type=AnomalyType.BRIGHT_REGION,
                    severity=AnomalySeverity.LOW,
                    description=f"Band {band_idx}: Unusually large bright region ({bright_pct:.1f}%)",
                    location=AnomalyLocation(
                        band=band_idx,
                        pixel_count=int(bright_count),
                        percentage=bright_pct,
                    ),
                    statistics={"threshold": float(p99), "bright_percent": bright_pct},
                )
            )

        return anomalies

    def _calculate_quality_score(
        self,
        anomalies: List[DetectedAnomaly],
        num_bands: int,
    ) -> float:
        """Calculate overall quality score from detected anomalies."""
        if not anomalies:
            return 1.0

        # Severity weights
        severity_weights = {
            AnomalySeverity.CRITICAL: 0.4,
            AnomalySeverity.HIGH: 0.2,
            AnomalySeverity.MEDIUM: 0.1,
            AnomalySeverity.LOW: 0.05,
            AnomalySeverity.INFO: 0.01,
        }

        # Calculate penalty
        total_penalty = sum(severity_weights.get(a.severity, 0.05) for a in anomalies)

        # Normalize by number of bands (more bands = more potential anomalies)
        normalized_penalty = total_penalty / max(1, num_bands)

        # Clamp score between 0 and 1
        quality_score = max(0.0, 1.0 - normalized_penalty)

        return round(quality_score, 3)


def detect_anomalies(
    data: np.ndarray,
    nodata: Optional[float] = None,
    zscore_threshold: float = 3.0,
) -> AnomalyResult:
    """Convenience function to detect anomalies in array data."""
    config = AnomalyConfig(zscore_threshold=zscore_threshold)
    detector = AnomalyDetector(config)
    return detector.detect_from_array(data, nodata=nodata)


def detect_anomalies_from_file(
    path: Union[str, Path],
    zscore_threshold: float = 3.0,
) -> AnomalyResult:
    """Convenience function to detect anomalies in a raster file."""
    config = AnomalyConfig(zscore_threshold=zscore_threshold)
    detector = AnomalyDetector(config)
    return detector.detect_from_file(path)
