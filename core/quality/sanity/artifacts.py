"""
Artifact Detection Sanity Checks.

Detects processing and sensor artifacts in analysis outputs:
- Stripe artifacts (scan-line errors, striping)
- Tile boundary artifacts (seams, edge effects)
- Geometric artifacts (warping errors, misregistration)
- Radiometric artifacts (hot pixels, dead pixels, saturation)
- Compression artifacts (blocking, ringing)
- Cloud/shadow confusion patterns
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts that can be detected."""
    # Geometric artifacts
    STRIPE = "stripe"                       # Horizontal/vertical stripes
    DIAGONAL_STRIPE = "diagonal_stripe"     # Diagonal striping
    TILE_SEAM = "tile_seam"                # Visible tile boundaries
    CHECKERBOARD = "checkerboard"          # Alternating pattern
    WARPING = "warping"                    # Geometric distortion

    # Radiometric artifacts
    HOT_PIXEL = "hot_pixel"                # Anomalously bright pixels
    DEAD_PIXEL = "dead_pixel"              # Non-responding pixels
    SATURATION = "saturation"              # Saturated values
    BANDING = "banding"                    # Quantization banding

    # Sensor-specific
    SAR_SCALLOPING = "sar_scalloping"      # SAR antenna pattern
    SAR_SIDELOBE = "sar_sidelobe"          # SAR sidelobe artifacts
    THERMAL_STRIPING = "thermal_striping"  # Thermal detector striping

    # Processing artifacts
    COMPRESSION = "compression"            # JPEG/compression blocking
    RESAMPLING = "resampling"              # Resampling artifacts
    EDGE_EFFECT = "edge_effect"            # Processing edge effects
    NODATA_PATTERN = "nodata_pattern"      # Unexpected nodata distribution

    # Classification artifacts
    CLOUD_SHADOW_CONFUSION = "cloud_shadow"  # Cloud/shadow misclassification
    WATER_SHADOW_CONFUSION = "water_shadow"  # Water/shadow confusion


class ArtifactSeverity(Enum):
    """Severity levels for detected artifacts."""
    CRITICAL = "critical"   # Severely impacts usability
    HIGH = "high"           # Significant quality degradation
    MEDIUM = "medium"       # Noticeable but may be acceptable
    LOW = "low"             # Minor artifact
    INFO = "info"           # Informational only


@dataclass
class ArtifactLocation:
    """Location of an artifact in the image."""
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    pixel_count: int = 0
    percentage: float = 0.0
    band: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row_range": [self.row_start, self.row_end] if self.row_start is not None else None,
            "col_range": [self.col_start, self.col_end] if self.col_start is not None else None,
            "pixel_count": self.pixel_count,
            "percentage": self.percentage,
            "band": self.band,
        }


@dataclass
class DetectedArtifact:
    """
    A single detected artifact.

    Attributes:
        artifact_type: Type of artifact detected
        severity: Severity level
        description: Human-readable description
        location: Where the artifact is located
        confidence: Detection confidence (0-1)
        statistics: Additional statistics about the artifact
        recommendation: Suggested action
    """
    artifact_type: ArtifactType
    severity: ArtifactSeverity
    description: str
    location: Optional[ArtifactLocation] = None
    confidence: float = 1.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.artifact_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location.to_dict() if self.location else None,
            "confidence": self.confidence,
            "statistics": self.statistics,
            "recommendation": self.recommendation,
        }


@dataclass
class ArtifactDetectionConfig:
    """
    Configuration for artifact detection.

    Attributes:
        detect_stripes: Enable stripe artifact detection
        detect_tile_seams: Enable tile seam detection
        detect_hot_pixels: Enable hot pixel detection
        detect_saturation: Enable saturation detection
        detect_compression: Enable compression artifact detection

        stripe_threshold: Z-score threshold for stripe detection
        hot_pixel_zscore: Z-score threshold for hot pixels
        saturation_margin: Margin below max for saturation (0-1)
        seam_gradient_threshold: Gradient threshold for seam detection

        tile_size: Expected tile size for seam detection
        data_max_value: Maximum valid data value (for saturation)
    """
    detect_stripes: bool = True
    detect_tile_seams: bool = True
    detect_hot_pixels: bool = True
    detect_saturation: bool = True
    detect_compression: bool = False  # Optional, computationally expensive

    stripe_threshold: float = 3.0
    hot_pixel_zscore: float = 5.0
    saturation_margin: float = 0.99
    seam_gradient_threshold: float = 3.0

    tile_size: Optional[int] = None  # Auto-detect if None
    data_max_value: Optional[float] = None  # Auto-detect if None

    # Minimum affected area to report
    min_artifact_pixels: int = 10
    min_artifact_percentage: float = 0.01  # 0.01%


@dataclass
class ArtifactDetectionResult:
    """
    Result of artifact detection.

    Attributes:
        has_artifacts: Whether any artifacts were detected
        artifacts: List of detected artifacts
        metrics: Computed detection metrics
        duration_seconds: Time taken for detection
    """
    has_artifacts: bool
    artifacts: List[DetectedArtifact] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical artifacts."""
        return sum(1 for a in self.artifacts if a.severity == ArtifactSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high-severity artifacts."""
        return sum(1 for a in self.artifacts if a.severity == ArtifactSeverity.HIGH)

    @property
    def artifact_types(self) -> List[str]:
        """List of artifact types detected."""
        return list(set(a.artifact_type.value for a in self.artifacts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_artifacts": self.has_artifacts,
            "artifact_count": len(self.artifacts),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "artifact_types": self.artifact_types,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class ArtifactDetector:
    """
    Detects various artifacts in analysis outputs.

    Identifies processing errors, sensor artifacts, and other issues
    that may affect data quality.

    Example:
        detector = ArtifactDetector()
        result = detector.detect(flood_extent_array)
        for artifact in result.artifacts:
            print(f"{artifact.severity}: {artifact.artifact_type.value}")
    """

    def __init__(self, config: Optional[ArtifactDetectionConfig] = None):
        """
        Initialize the artifact detector.

        Args:
            config: Configuration options
        """
        self.config = config or ArtifactDetectionConfig()

    def detect(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ArtifactDetectionResult:
        """
        Detect all configured artifact types.

        Args:
            data: 2D or 3D array (H, W) or (B, H, W)
            mask: Optional boolean mask of valid pixels

        Returns:
            ArtifactDetectionResult with all detected artifacts
        """
        import time
        start_time = time.time()

        artifacts = []
        metrics = {}

        # Handle 3D data (multi-band)
        if data.ndim == 3:
            bands, height, width = data.shape
            metrics["shape"] = {"bands": bands, "height": height, "width": width}

            # Detect on each band
            for b in range(bands):
                band_data = data[b]
                band_mask = mask[b] if mask is not None and mask.ndim == 3 else mask
                band_artifacts = self._detect_single_band(band_data, band_mask, band=b)
                artifacts.extend(band_artifacts)
        else:
            height, width = data.shape
            metrics["shape"] = {"height": height, "width": width}
            artifacts = self._detect_single_band(data, mask)

        # Deduplicate similar artifacts
        artifacts = self._deduplicate_artifacts(artifacts)

        # Sort by severity
        severity_order = {
            ArtifactSeverity.CRITICAL: 0,
            ArtifactSeverity.HIGH: 1,
            ArtifactSeverity.MEDIUM: 2,
            ArtifactSeverity.LOW: 3,
            ArtifactSeverity.INFO: 4,
        }
        artifacts.sort(key=lambda a: severity_order.get(a.severity, 5))

        has_artifacts = len(artifacts) > 0
        duration = time.time() - start_time

        logger.info(f"Artifact detection completed in {duration:.2f}s, found {len(artifacts)} artifacts")

        return ArtifactDetectionResult(
            has_artifacts=has_artifacts,
            artifacts=artifacts,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _detect_single_band(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        band: Optional[int] = None,
    ) -> List[DetectedArtifact]:
        """Detect artifacts in a single band."""
        artifacts = []

        # Apply mask
        if mask is not None:
            data = np.where(mask, data, np.nan)

        # Detect stripes
        if self.config.detect_stripes:
            stripe_artifacts = self._detect_stripes(data, band)
            artifacts.extend(stripe_artifacts)

        # Detect tile seams
        if self.config.detect_tile_seams:
            seam_artifacts = self._detect_tile_seams(data, band)
            artifacts.extend(seam_artifacts)

        # Detect hot pixels
        if self.config.detect_hot_pixels:
            hot_artifacts = self._detect_hot_pixels(data, band)
            artifacts.extend(hot_artifacts)

        # Detect saturation
        if self.config.detect_saturation:
            sat_artifacts = self._detect_saturation(data, band)
            artifacts.extend(sat_artifacts)

        # Detect compression artifacts
        if self.config.detect_compression:
            comp_artifacts = self._detect_compression(data, band)
            artifacts.extend(comp_artifacts)

        return artifacts

    def _detect_stripes(
        self, data: np.ndarray, band: Optional[int] = None
    ) -> List[DetectedArtifact]:
        """Detect horizontal and vertical stripe artifacts."""
        artifacts = []

        # Replace NaN with 0 for analysis
        valid_mask = np.isfinite(data)
        if not valid_mask.any():
            return artifacts

        clean_data = np.nan_to_num(data, nan=0.0)
        height, width = data.shape

        # Detect horizontal stripes (row anomalies)
        row_means = np.nanmean(data, axis=1)
        valid_row_means = row_means[np.isfinite(row_means)]

        if len(valid_row_means) > 10:
            row_mean_global = np.mean(valid_row_means)
            row_std_global = np.std(valid_row_means)

            if row_std_global > 1e-10:
                row_zscores = (row_means - row_mean_global) / row_std_global
                anomalous_rows = np.where(np.abs(row_zscores) > self.config.stripe_threshold)[0]

                if len(anomalous_rows) > 0:
                    # Group consecutive rows
                    stripe_groups = self._group_consecutive(anomalous_rows)

                    for group in stripe_groups:
                        if len(group) >= 2:  # At least 2 consecutive rows
                            affected_pixels = len(group) * width
                            pct = 100.0 * affected_pixels / data.size

                            if affected_pixels >= self.config.min_artifact_pixels:
                                artifacts.append(DetectedArtifact(
                                    artifact_type=ArtifactType.STRIPE,
                                    severity=self._stripe_severity(len(group), height),
                                    description=f"Horizontal stripe at rows {group[0]}-{group[-1]}",
                                    location=ArtifactLocation(
                                        row_start=int(group[0]),
                                        row_end=int(group[-1]),
                                        col_start=0,
                                        col_end=width - 1,
                                        pixel_count=affected_pixels,
                                        percentage=pct,
                                        band=band,
                                    ),
                                    confidence=min(1.0, np.mean(np.abs(row_zscores[group])) / self.config.stripe_threshold),
                                    statistics={"max_zscore": float(np.max(np.abs(row_zscores[group])))},
                                    recommendation="Apply destriping filter or check sensor calibration",
                                ))

        # Detect vertical stripes (column anomalies)
        col_means = np.nanmean(data, axis=0)
        valid_col_means = col_means[np.isfinite(col_means)]

        if len(valid_col_means) > 10:
            col_mean_global = np.mean(valid_col_means)
            col_std_global = np.std(valid_col_means)

            if col_std_global > 1e-10:
                col_zscores = (col_means - col_mean_global) / col_std_global
                anomalous_cols = np.where(np.abs(col_zscores) > self.config.stripe_threshold)[0]

                if len(anomalous_cols) > 0:
                    stripe_groups = self._group_consecutive(anomalous_cols)

                    for group in stripe_groups:
                        if len(group) >= 2:
                            affected_pixels = height * len(group)
                            pct = 100.0 * affected_pixels / data.size

                            if affected_pixels >= self.config.min_artifact_pixels:
                                artifacts.append(DetectedArtifact(
                                    artifact_type=ArtifactType.STRIPE,
                                    severity=self._stripe_severity(len(group), width),
                                    description=f"Vertical stripe at columns {group[0]}-{group[-1]}",
                                    location=ArtifactLocation(
                                        row_start=0,
                                        row_end=height - 1,
                                        col_start=int(group[0]),
                                        col_end=int(group[-1]),
                                        pixel_count=affected_pixels,
                                        percentage=pct,
                                        band=band,
                                    ),
                                    confidence=min(1.0, np.mean(np.abs(col_zscores[group])) / self.config.stripe_threshold),
                                    statistics={"max_zscore": float(np.max(np.abs(col_zscores[group])))},
                                    recommendation="Apply destriping filter or check sensor calibration",
                                ))

        return artifacts

    def _detect_tile_seams(
        self, data: np.ndarray, band: Optional[int] = None
    ) -> List[DetectedArtifact]:
        """Detect tile boundary artifacts (seams)."""
        artifacts = []

        height, width = data.shape
        tile_size = self.config.tile_size or self._estimate_tile_size(data.shape)

        if tile_size is None or tile_size >= min(height, width):
            return artifacts

        # Calculate gradient magnitude
        gy, gx = np.gradient(np.nan_to_num(data, nan=0.0))
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Get background gradient statistics
        valid_gradient = gradient_mag[np.isfinite(data)]
        if len(valid_gradient) < 100:
            return artifacts

        mean_grad = np.mean(valid_gradient)
        std_grad = np.std(valid_gradient)

        if std_grad < 1e-10:
            return artifacts

        # Check horizontal seams
        h_seam_rows = list(range(tile_size, height, tile_size))
        for row in h_seam_rows:
            if row >= height:
                continue
            seam_gradient = np.nanmean(gradient_mag[row, :])
            z_score = (seam_gradient - mean_grad) / std_grad

            if z_score > self.config.seam_gradient_threshold:
                artifacts.append(DetectedArtifact(
                    artifact_type=ArtifactType.TILE_SEAM,
                    severity=ArtifactSeverity.MEDIUM,
                    description=f"Horizontal tile seam at row {row}",
                    location=ArtifactLocation(
                        row_start=row,
                        row_end=row,
                        col_start=0,
                        col_end=width - 1,
                        pixel_count=width,
                        percentage=100.0 * width / data.size,
                        band=band,
                    ),
                    confidence=min(1.0, z_score / (2 * self.config.seam_gradient_threshold)),
                    statistics={"gradient_zscore": float(z_score)},
                    recommendation="Check mosaic blending or tile processing",
                ))

        # Check vertical seams
        v_seam_cols = list(range(tile_size, width, tile_size))
        for col in v_seam_cols:
            if col >= width:
                continue
            seam_gradient = np.nanmean(gradient_mag[:, col])
            z_score = (seam_gradient - mean_grad) / std_grad

            if z_score > self.config.seam_gradient_threshold:
                artifacts.append(DetectedArtifact(
                    artifact_type=ArtifactType.TILE_SEAM,
                    severity=ArtifactSeverity.MEDIUM,
                    description=f"Vertical tile seam at column {col}",
                    location=ArtifactLocation(
                        row_start=0,
                        row_end=height - 1,
                        col_start=col,
                        col_end=col,
                        pixel_count=height,
                        percentage=100.0 * height / data.size,
                        band=band,
                    ),
                    confidence=min(1.0, z_score / (2 * self.config.seam_gradient_threshold)),
                    statistics={"gradient_zscore": float(z_score)},
                    recommendation="Check mosaic blending or tile processing",
                ))

        return artifacts

    def _detect_hot_pixels(
        self, data: np.ndarray, band: Optional[int] = None
    ) -> List[DetectedArtifact]:
        """Detect hot (anomalously bright) pixels."""
        artifacts = []

        valid_data = data[np.isfinite(data)]
        if len(valid_data) < 100:
            return artifacts

        mean = np.mean(valid_data)
        std = np.std(valid_data)

        if std < 1e-10:
            return artifacts

        threshold = mean + self.config.hot_pixel_zscore * std
        hot_mask = (data > threshold) & np.isfinite(data)
        hot_count = hot_mask.sum()

        if hot_count >= self.config.min_artifact_pixels:
            finite_count = np.isfinite(data).sum()
            pct = 100.0 * hot_count / finite_count if finite_count > 0 else 0.0

            # Determine severity based on count
            if hot_count > 1000:
                severity = ArtifactSeverity.HIGH
            elif hot_count > 100:
                severity = ArtifactSeverity.MEDIUM
            else:
                severity = ArtifactSeverity.LOW

            # Get sample locations
            hot_rows, hot_cols = np.where(hot_mask)
            sample_locs = list(zip(hot_rows[:5].tolist(), hot_cols[:5].tolist()))

            artifacts.append(DetectedArtifact(
                artifact_type=ArtifactType.HOT_PIXEL,
                severity=severity,
                description=f"Detected {hot_count} hot pixels (>{self.config.hot_pixel_zscore} sigma)",
                location=ArtifactLocation(
                    pixel_count=int(hot_count),
                    percentage=pct,
                    band=band,
                ),
                confidence=0.9,
                statistics={
                    "threshold": float(threshold),
                    "max_value": float(np.max(data[hot_mask])),
                    "sample_locations": sample_locs,
                },
                recommendation="Apply hot pixel filter or median filter",
            ))

        # Similarly detect dead/cold pixels
        cold_threshold = mean - self.config.hot_pixel_zscore * std
        cold_mask = (data < cold_threshold) & np.isfinite(data)
        cold_count = cold_mask.sum()

        if cold_count >= self.config.min_artifact_pixels:
            finite_count = np.isfinite(data).sum()
            pct = 100.0 * cold_count / finite_count if finite_count > 0 else 0.0

            if cold_count > 1000:
                severity = ArtifactSeverity.HIGH
            elif cold_count > 100:
                severity = ArtifactSeverity.MEDIUM
            else:
                severity = ArtifactSeverity.LOW

            artifacts.append(DetectedArtifact(
                artifact_type=ArtifactType.DEAD_PIXEL,
                severity=severity,
                description=f"Detected {cold_count} dead/cold pixels (<{-self.config.hot_pixel_zscore} sigma)",
                location=ArtifactLocation(
                    pixel_count=int(cold_count),
                    percentage=pct,
                    band=band,
                ),
                confidence=0.9,
                statistics={
                    "threshold": float(cold_threshold),
                    "min_value": float(np.min(data[cold_mask])),
                },
                recommendation="Apply dead pixel interpolation",
            ))

        return artifacts

    def _detect_saturation(
        self, data: np.ndarray, band: Optional[int] = None
    ) -> List[DetectedArtifact]:
        """Detect saturated pixels."""
        artifacts = []

        valid_data = data[np.isfinite(data)]
        if len(valid_data) < 100:
            return artifacts

        # Determine max value
        data_max = self.config.data_max_value
        if data_max is None:
            data_max = np.max(valid_data)

        # Saturation threshold
        sat_threshold = data_max * self.config.saturation_margin

        # Count saturated pixels
        sat_mask = (data >= sat_threshold) & np.isfinite(data)
        sat_count = sat_mask.sum()

        if sat_count >= self.config.min_artifact_pixels:
            pct = 100.0 * sat_count / np.isfinite(data).sum()

            if pct > 10:
                severity = ArtifactSeverity.HIGH
            elif pct > 1:
                severity = ArtifactSeverity.MEDIUM
            else:
                severity = ArtifactSeverity.LOW

            artifacts.append(DetectedArtifact(
                artifact_type=ArtifactType.SATURATION,
                severity=severity,
                description=f"Detected {sat_count} saturated pixels ({pct:.2f}% of valid pixels)",
                location=ArtifactLocation(
                    pixel_count=int(sat_count),
                    percentage=pct,
                    band=band,
                ),
                confidence=0.95,
                statistics={
                    "threshold": float(sat_threshold),
                    "data_max": float(data_max),
                },
                recommendation="Flag saturated regions as uncertain",
            ))

        return artifacts

    def _detect_compression(
        self, data: np.ndarray, band: Optional[int] = None
    ) -> List[DetectedArtifact]:
        """Detect compression artifacts (blocking)."""
        artifacts = []

        # Look for 8x8 or 16x16 block patterns (JPEG-like)
        for block_size in [8, 16]:
            block_score = self._calculate_block_score(data, block_size)
            if block_score > 0.5:  # Threshold for significant blocking
                artifacts.append(DetectedArtifact(
                    artifact_type=ArtifactType.COMPRESSION,
                    severity=ArtifactSeverity.MEDIUM if block_score > 0.7 else ArtifactSeverity.LOW,
                    description=f"Compression blocking artifacts detected ({block_size}x{block_size} pattern)",
                    confidence=min(1.0, block_score),
                    statistics={"block_score": block_score, "block_size": block_size},
                    recommendation="Use lossless compression or higher quality",
                ))
                break  # Only report one

        return artifacts

    def _calculate_block_score(self, data: np.ndarray, block_size: int) -> float:
        """Calculate blocking artifact score using gradient analysis."""
        height, width = data.shape

        if height < block_size * 2 or width < block_size * 2:
            return 0.0

        # Calculate gradient
        gy, gx = np.gradient(np.nan_to_num(data, nan=0.0))
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Calculate gradient at block boundaries vs elsewhere
        block_boundary_grads = []
        non_boundary_grads = []

        for row in range(height):
            for col in range(width):
                g = gradient_mag[row, col]
                if np.isfinite(g):
                    if row % block_size == 0 or col % block_size == 0:
                        block_boundary_grads.append(g)
                    else:
                        non_boundary_grads.append(g)

        if not block_boundary_grads or not non_boundary_grads:
            return 0.0

        boundary_mean = np.mean(block_boundary_grads)
        non_boundary_mean = np.mean(non_boundary_grads)

        if non_boundary_mean < 1e-10:
            return 0.0

        # Score based on ratio
        ratio = boundary_mean / non_boundary_mean
        return min(1.0, max(0.0, (ratio - 1.0)))

    def _group_consecutive(self, indices: np.ndarray) -> List[List[int]]:
        """Group consecutive indices into runs."""
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]

        groups.append(current_group)
        return groups

    def _stripe_severity(self, stripe_width: int, dimension_size: int) -> ArtifactSeverity:
        """Determine stripe severity based on width."""
        ratio = stripe_width / dimension_size
        if ratio > 0.1:
            return ArtifactSeverity.HIGH
        elif ratio > 0.02:
            return ArtifactSeverity.MEDIUM
        else:
            return ArtifactSeverity.LOW

    def _estimate_tile_size(self, shape: Tuple[int, ...]) -> Optional[int]:
        """Estimate tile size from common values."""
        height, width = shape[:2]

        # Common tile sizes
        common_sizes = [256, 512, 1024, 2048]

        for size in common_sizes:
            if height % size == 0 or width % size == 0:
                return size

        return None

    def _deduplicate_artifacts(
        self, artifacts: List[DetectedArtifact]
    ) -> List[DetectedArtifact]:
        """Deduplicate similar artifacts."""
        if len(artifacts) <= 1:
            return artifacts

        # Simple dedup by type and location
        seen = set()
        unique = []

        for artifact in artifacts:
            key = (
                artifact.artifact_type,
                artifact.location.row_start if artifact.location else None,
                artifact.location.col_start if artifact.location else None,
                artifact.location.band if artifact.location else None,
            )
            if key not in seen:
                seen.add(key)
                unique.append(artifact)

        return unique


def detect_artifacts(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[ArtifactDetectionConfig] = None,
) -> ArtifactDetectionResult:
    """
    Convenience function to detect artifacts.

    Args:
        data: 2D or 3D array
        mask: Optional boolean mask
        config: Optional configuration

    Returns:
        ArtifactDetectionResult with all detected artifacts
    """
    detector = ArtifactDetector(config)
    return detector.detect(data, mask)
