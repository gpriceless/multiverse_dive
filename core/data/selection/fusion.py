"""
Fusion strategy for multi-sensor data blending.

Provides intelligent strategies for combining data from multiple sensors,
including:
- Complementary vs redundant sensor combinations
- Multi-sensor blending rules
- Temporal densification for improved time series
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import logging
import math

from core.data.discovery.base import DiscoveryResult

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Types of multi-sensor fusion strategies."""
    COMPLEMENTARY = "complementary"  # Sensors provide different information
    REDUNDANT = "redundant"          # Sensors provide overlapping information
    TEMPORAL = "temporal"            # Sensors provide different time coverage
    HIERARCHICAL = "hierarchical"    # Primary sensor with fallback
    ENSEMBLE = "ensemble"            # Multiple sensors for consensus


class SensorRole(Enum):
    """Role of a sensor in a fusion configuration."""
    PRIMARY = "primary"              # Main source of information
    SECONDARY = "secondary"          # Supporting/validating source
    FALLBACK = "fallback"            # Used when primary unavailable
    GAP_FILL = "gap_fill"            # Fills temporal/spatial gaps
    VALIDATION = "validation"        # Cross-validation source


class BlendingMethod(Enum):
    """Methods for blending multi-sensor data."""
    WEIGHTED_AVERAGE = "weighted_average"    # Weight by quality/confidence
    QUALITY_MOSAIC = "quality_mosaic"        # Select best quality per pixel
    TEMPORAL_COMPOSITE = "temporal_composite" # Composite over time
    CONSENSUS = "consensus"                   # Majority voting
    PRIORITY_STACK = "priority_stack"         # Use by priority order
    KALMAN_FILTER = "kalman_filter"           # Optimal state estimation


@dataclass
class SensorContribution:
    """
    Defines a sensor's contribution to a fusion configuration.

    Attributes:
        sensor_type: Type of sensor (optical, sar, thermal, etc.)
        role: Role in the fusion (primary, secondary, fallback, etc.)
        weight: Weight for blending (0.0 to 1.0)
        provides: List of observables this sensor provides
        requirements: Conditions required for this sensor
        fallback_for: Sensor types this can serve as fallback for
    """
    sensor_type: str
    role: SensorRole
    weight: float = 1.0
    provides: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    fallback_for: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sensor_type": self.sensor_type,
            "role": self.role.value,
            "weight": self.weight,
            "provides": self.provides,
            "requirements": self.requirements,
            "fallback_for": self.fallback_for
        }


@dataclass
class FusionConfiguration:
    """
    Configuration for a multi-sensor fusion strategy.

    Attributes:
        name: Configuration name
        strategy: Type of fusion strategy
        sensors: List of sensor contributions
        blending_method: How to blend the data
        confidence_threshold: Minimum confidence for inclusion
        temporal_tolerance_hours: Max time gap for temporal compositing
        priority_order: Ordered list of sensor types by priority
    """
    name: str
    strategy: FusionStrategy
    sensors: List[SensorContribution]
    blending_method: BlendingMethod = BlendingMethod.WEIGHTED_AVERAGE
    confidence_threshold: float = 0.5
    temporal_tolerance_hours: float = 24.0
    priority_order: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "sensors": [s.to_dict() for s in self.sensors],
            "blending_method": self.blending_method.value,
            "confidence_threshold": self.confidence_threshold,
            "temporal_tolerance_hours": self.temporal_tolerance_hours,
            "priority_order": self.priority_order
        }


@dataclass
class FusionPlan:
    """
    A concrete plan for fusing specific datasets.

    Attributes:
        configuration: The fusion configuration used
        datasets: Mapping of sensor_type -> selected datasets
        blending_weights: Computed weights for each dataset
        temporal_coverage: Overall temporal coverage
        spatial_coverage: Overall spatial coverage estimate
        confidence: Overall confidence in fusion
        rationale: Explanation of fusion decisions
    """
    configuration: FusionConfiguration
    datasets: Dict[str, List[DiscoveryResult]]
    blending_weights: Dict[str, float]
    temporal_coverage: Dict[str, Any]
    spatial_coverage: float
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "configuration": self.configuration.to_dict(),
            "datasets": {
                k: [d.to_dict() for d in v]
                for k, v in self.datasets.items()
            },
            "blending_weights": self.blending_weights,
            "temporal_coverage": self.temporal_coverage,
            "spatial_coverage": self.spatial_coverage,
            "confidence": self.confidence,
            "rationale": self.rationale
        }


@dataclass
class TemporalGap:
    """Represents a gap in temporal coverage."""
    start: datetime
    end: datetime
    duration_hours: float
    sensor_types_available: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_hours": self.duration_hours,
            "sensor_types_available": self.sensor_types_available
        }


class FusionStrategyEngine:
    """
    Engine for determining and applying multi-sensor fusion strategies.

    Provides methods for:
    - Determining optimal fusion strategy based on data availability
    - Computing blending weights
    - Identifying temporal gaps and densification opportunities
    - Creating fusion plans from discovered datasets
    """

    def __init__(self):
        """Initialize fusion strategy engine."""
        self.configurations = self._initialize_default_configurations()

    def _initialize_default_configurations(self) -> Dict[str, FusionConfiguration]:
        """Initialize default fusion configurations for common scenarios."""
        configs = {}

        # Flood detection: SAR + optical complementary
        configs["flood_complementary"] = FusionConfiguration(
            name="flood_complementary",
            strategy=FusionStrategy.COMPLEMENTARY,
            sensors=[
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.PRIMARY,
                    weight=0.6,
                    provides=["water_extent", "flood_boundary"],
                    requirements={"max_resolution_m": 30}  # 30m or finer is acceptable
                ),
                SensorContribution(
                    sensor_type="optical",
                    role=SensorRole.SECONDARY,
                    weight=0.4,
                    provides=["water_quality", "land_use_context"],
                    requirements={"max_cloud_cover": 50},
                    fallback_for=["sar"]
                ),
                SensorContribution(
                    sensor_type="dem",
                    role=SensorRole.SECONDARY,
                    weight=0.3,
                    provides=["terrain_height", "flood_depth"],
                    requirements={}
                )
            ],
            blending_method=BlendingMethod.WEIGHTED_AVERAGE,
            confidence_threshold=0.6,
            priority_order=["sar", "optical", "dem"]
        )

        # Flood detection: SAR only (all-weather)
        configs["flood_sar_only"] = FusionConfiguration(
            name="flood_sar_only",
            strategy=FusionStrategy.HIERARCHICAL,
            sensors=[
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.PRIMARY,
                    weight=1.0,
                    provides=["water_extent", "flood_boundary"],
                    requirements={"max_resolution_m": 30}  # 30m or finer is acceptable
                ),
                SensorContribution(
                    sensor_type="dem",
                    role=SensorRole.SECONDARY,
                    weight=0.5,
                    provides=["terrain_height", "flood_depth"],
                    requirements={}
                )
            ],
            blending_method=BlendingMethod.PRIORITY_STACK,
            confidence_threshold=0.5,
            priority_order=["sar", "dem"]
        )

        # Wildfire: Thermal + optical complementary
        configs["wildfire_complementary"] = FusionConfiguration(
            name="wildfire_complementary",
            strategy=FusionStrategy.COMPLEMENTARY,
            sensors=[
                SensorContribution(
                    sensor_type="thermal",
                    role=SensorRole.PRIMARY,
                    weight=0.7,
                    provides=["active_fire", "fire_temperature"],
                    requirements={}
                ),
                SensorContribution(
                    sensor_type="optical",
                    role=SensorRole.SECONDARY,
                    weight=0.5,
                    provides=["burn_severity", "vegetation_damage"],
                    requirements={"max_cloud_cover": 70}
                ),
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.FALLBACK,
                    weight=0.3,
                    provides=["burn_boundary"],
                    requirements={},
                    fallback_for=["optical"]
                )
            ],
            blending_method=BlendingMethod.QUALITY_MOSAIC,
            confidence_threshold=0.5,
            priority_order=["thermal", "optical", "sar"]
        )

        # Storm damage: SAR + optical redundant
        configs["storm_redundant"] = FusionConfiguration(
            name="storm_redundant",
            strategy=FusionStrategy.REDUNDANT,
            sensors=[
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.PRIMARY,
                    weight=0.5,
                    provides=["structural_damage", "vegetation_damage"],
                    requirements={}
                ),
                SensorContribution(
                    sensor_type="optical",
                    role=SensorRole.PRIMARY,
                    weight=0.5,
                    provides=["structural_damage", "vegetation_damage"],
                    requirements={"max_cloud_cover": 50}
                )
            ],
            blending_method=BlendingMethod.CONSENSUS,
            confidence_threshold=0.6,
            priority_order=["sar", "optical"]
        )

        # Multi-temporal: Dense time series
        configs["temporal_densification"] = FusionConfiguration(
            name="temporal_densification",
            strategy=FusionStrategy.TEMPORAL,
            sensors=[
                SensorContribution(
                    sensor_type="optical",
                    role=SensorRole.GAP_FILL,
                    weight=1.0,
                    provides=["change_detection"],
                    requirements={"max_cloud_cover": 30}
                ),
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.GAP_FILL,
                    weight=0.8,
                    provides=["change_detection"],
                    requirements={}
                )
            ],
            blending_method=BlendingMethod.TEMPORAL_COMPOSITE,
            confidence_threshold=0.4,
            temporal_tolerance_hours=48.0,
            priority_order=["optical", "sar"]
        )

        # Ensemble: Multiple sensors for consensus
        configs["ensemble_consensus"] = FusionConfiguration(
            name="ensemble_consensus",
            strategy=FusionStrategy.ENSEMBLE,
            sensors=[
                SensorContribution(
                    sensor_type="optical",
                    role=SensorRole.VALIDATION,
                    weight=0.4,
                    provides=["observation"],
                    requirements={"max_cloud_cover": 30}
                ),
                SensorContribution(
                    sensor_type="sar",
                    role=SensorRole.VALIDATION,
                    weight=0.4,
                    provides=["observation"],
                    requirements={}
                ),
                SensorContribution(
                    sensor_type="thermal",
                    role=SensorRole.VALIDATION,
                    weight=0.2,
                    provides=["observation"],
                    requirements={}
                )
            ],
            blending_method=BlendingMethod.CONSENSUS,
            confidence_threshold=0.7,
            priority_order=["optical", "sar", "thermal"]
        )

        return configs

    def get_configuration(self, name: str) -> Optional[FusionConfiguration]:
        """
        Get a fusion configuration by name.

        Args:
            name: Configuration name

        Returns:
            FusionConfiguration if found, None otherwise
        """
        return self.configurations.get(name)

    def determine_strategy(
        self,
        event_class: str,
        available_sensors: Set[str],
        atmospheric_conditions: Optional[Dict[str, Any]] = None
    ) -> FusionConfiguration:
        """
        Determine optimal fusion strategy based on context.

        Args:
            event_class: Event classification (e.g., "flood.coastal")
            available_sensors: Set of available sensor types
            atmospheric_conditions: Current atmospheric conditions

        Returns:
            Recommended FusionConfiguration
        """
        conditions = atmospheric_conditions or {}
        cloud_cover = conditions.get("cloud_cover_percent", 0)

        # Flood events
        if event_class.startswith("flood"):
            # High cloud cover: SAR only
            if cloud_cover > 80:
                logger.info("High cloud cover - using SAR-only flood strategy")
                return self.configurations["flood_sar_only"]
            # SAR + optical available: complementary
            if "sar" in available_sensors and "optical" in available_sensors:
                return self.configurations["flood_complementary"]
            # SAR only
            if "sar" in available_sensors:
                return self.configurations["flood_sar_only"]

        # Wildfire events
        elif event_class.startswith("wildfire"):
            return self.configurations["wildfire_complementary"]

        # Storm events
        elif event_class.startswith("storm"):
            return self.configurations["storm_redundant"]

        # Default: temporal densification
        logger.info(f"No specific strategy for {event_class}, using temporal densification")
        return self.configurations["temporal_densification"]

    def create_fusion_plan(
        self,
        configuration: FusionConfiguration,
        candidates: List[DiscoveryResult],
        temporal_extent: Dict[str, str]
    ) -> FusionPlan:
        """
        Create a concrete fusion plan from discovered datasets.

        Args:
            configuration: Fusion configuration to use
            candidates: List of discovered dataset candidates
            temporal_extent: Temporal extent {start, end, reference_time}

        Returns:
            FusionPlan with selected datasets and blending weights
        """
        # Group candidates by sensor type
        by_sensor: Dict[str, List[DiscoveryResult]] = {}
        for candidate in candidates:
            sensor_type = candidate.data_type
            if sensor_type not in by_sensor:
                by_sensor[sensor_type] = []
            by_sensor[sensor_type].append(candidate)

        # Select datasets for each sensor in configuration
        selected_datasets: Dict[str, List[DiscoveryResult]] = {}
        blending_weights: Dict[str, float] = {}

        for sensor_config in configuration.sensors:
            sensor_type = sensor_config.sensor_type
            candidates_for_sensor = by_sensor.get(sensor_type, [])

            if not candidates_for_sensor:
                logger.warning(f"No candidates available for {sensor_type}")
                continue

            # Filter by requirements
            filtered = self._filter_by_requirements(
                candidates_for_sensor,
                sensor_config.requirements
            )

            if not filtered:
                logger.warning(f"No candidates pass requirements for {sensor_type}")
                continue

            # Select best candidates
            selected = self._select_best_candidates(
                filtered,
                configuration.strategy,
                temporal_extent
            )

            selected_datasets[sensor_type] = selected
            blending_weights[sensor_type] = sensor_config.weight

        # Normalize weights
        blending_weights = self._normalize_weights(blending_weights)

        # Calculate coverage
        temporal_coverage = self._calculate_temporal_coverage(
            selected_datasets,
            temporal_extent
        )

        spatial_coverage = self._calculate_spatial_coverage(selected_datasets)

        # Calculate confidence
        confidence = self._calculate_fusion_confidence(
            configuration,
            selected_datasets,
            blending_weights
        )

        # Generate rationale
        rationale = self._generate_rationale(
            configuration,
            selected_datasets,
            confidence
        )

        return FusionPlan(
            configuration=configuration,
            datasets=selected_datasets,
            blending_weights=blending_weights,
            temporal_coverage=temporal_coverage,
            spatial_coverage=spatial_coverage,
            confidence=confidence,
            rationale=rationale
        )

    def identify_temporal_gaps(
        self,
        datasets: Dict[str, List[DiscoveryResult]],
        temporal_extent: Dict[str, str],
        max_gap_hours: float = 24.0
    ) -> List[TemporalGap]:
        """
        Identify gaps in temporal coverage.

        Args:
            datasets: Mapping of sensor_type -> datasets
            temporal_extent: Temporal extent {start, end}
            max_gap_hours: Maximum acceptable gap in hours

        Returns:
            List of TemporalGap objects
        """
        # Parse temporal extent
        start = datetime.fromisoformat(
            temporal_extent["start"].replace('Z', '+00:00')
        )
        end = datetime.fromisoformat(
            temporal_extent["end"].replace('Z', '+00:00')
        )

        # Collect all acquisition times
        acquisitions: List[Tuple[datetime, str]] = []
        for sensor_type, sensor_datasets in datasets.items():
            for dataset in sensor_datasets:
                acquisitions.append((dataset.acquisition_time, sensor_type))

        # Sort by time
        acquisitions.sort(key=lambda x: x[0])

        # Find gaps
        gaps: List[TemporalGap] = []

        # Check gap at start
        if acquisitions and acquisitions[0][0] > start:
            gap_duration = (acquisitions[0][0] - start).total_seconds() / 3600
            if gap_duration > max_gap_hours:
                gaps.append(TemporalGap(
                    start=start,
                    end=acquisitions[0][0],
                    duration_hours=gap_duration,
                    sensor_types_available=[]
                ))

        # Check gaps between acquisitions
        for i in range(len(acquisitions) - 1):
            current_time, current_sensor = acquisitions[i]
            next_time, next_sensor = acquisitions[i + 1]

            gap_duration = (next_time - current_time).total_seconds() / 3600
            if gap_duration > max_gap_hours:
                gaps.append(TemporalGap(
                    start=current_time,
                    end=next_time,
                    duration_hours=gap_duration,
                    sensor_types_available=[current_sensor, next_sensor]
                ))

        # Check gap at end
        if acquisitions and acquisitions[-1][0] < end:
            gap_duration = (end - acquisitions[-1][0]).total_seconds() / 3600
            if gap_duration > max_gap_hours:
                gaps.append(TemporalGap(
                    start=acquisitions[-1][0],
                    end=end,
                    duration_hours=gap_duration,
                    sensor_types_available=[]
                ))

        return gaps

    def densify_temporal_coverage(
        self,
        primary_datasets: List[DiscoveryResult],
        gap_fill_datasets: List[DiscoveryResult],
        max_gap_hours: float = 24.0
    ) -> List[DiscoveryResult]:
        """
        Densify temporal coverage by filling gaps with additional sensors.

        Args:
            primary_datasets: Primary sensor datasets
            gap_fill_datasets: Datasets available for gap filling
            max_gap_hours: Maximum acceptable gap in hours

        Returns:
            Combined list of datasets providing dense temporal coverage
        """
        result: List[DiscoveryResult] = list(primary_datasets)

        # Sort primary by time
        primary_sorted = sorted(primary_datasets, key=lambda x: x.acquisition_time)

        # Find gaps and fill with secondary data
        for i in range(len(primary_sorted) - 1):
            current = primary_sorted[i]
            next_dataset = primary_sorted[i + 1]

            gap_hours = (
                next_dataset.acquisition_time - current.acquisition_time
            ).total_seconds() / 3600

            if gap_hours > max_gap_hours:
                # Find gap-fill candidates within this window
                gap_start = current.acquisition_time
                gap_end = next_dataset.acquisition_time

                for filler in gap_fill_datasets:
                    if gap_start < filler.acquisition_time < gap_end:
                        if filler not in result:
                            result.append(filler)
                            logger.debug(
                                f"Added {filler.data_type} dataset for gap-fill "
                                f"at {filler.acquisition_time}"
                            )

        # Sort result by time
        result.sort(key=lambda x: x.acquisition_time)

        return result

    def compute_blending_weights(
        self,
        datasets: Dict[str, List[DiscoveryResult]],
        method: BlendingMethod,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute blending weights for each dataset.

        Args:
            datasets: Mapping of sensor_type -> datasets
            method: Blending method to use
            base_weights: Base weights by sensor type

        Returns:
            Nested dict: sensor_type -> dataset_id -> weight
        """
        weights: Dict[str, Dict[str, float]] = {}
        base = base_weights or {}

        for sensor_type, sensor_datasets in datasets.items():
            weights[sensor_type] = {}
            sensor_base = base.get(sensor_type, 1.0)

            if method == BlendingMethod.WEIGHTED_AVERAGE:
                # Weight by quality and coverage
                for dataset in sensor_datasets:
                    quality_factor = self._quality_factor(dataset)
                    coverage_factor = dataset.spatial_coverage_percent / 100.0
                    weights[sensor_type][dataset.dataset_id] = (
                        sensor_base * quality_factor * coverage_factor
                    )

            elif method == BlendingMethod.QUALITY_MOSAIC:
                # All datasets get equal base weight, quality used for per-pixel selection
                for dataset in sensor_datasets:
                    weights[sensor_type][dataset.dataset_id] = sensor_base

            elif method == BlendingMethod.PRIORITY_STACK:
                # Decreasing weights by priority order
                for i, dataset in enumerate(sensor_datasets):
                    priority_factor = 1.0 / (i + 1)
                    weights[sensor_type][dataset.dataset_id] = (
                        sensor_base * priority_factor
                    )

            elif method == BlendingMethod.CONSENSUS:
                # Equal weights for voting
                for dataset in sensor_datasets:
                    weights[sensor_type][dataset.dataset_id] = sensor_base

            elif method == BlendingMethod.TEMPORAL_COMPOSITE:
                # Weight by temporal proximity to reference
                for dataset in sensor_datasets:
                    weights[sensor_type][dataset.dataset_id] = sensor_base

            else:
                # Default: equal weights
                for dataset in sensor_datasets:
                    weights[sensor_type][dataset.dataset_id] = sensor_base

        # Normalize within each sensor type
        for sensor_type in weights:
            total = sum(weights[sensor_type].values())
            if total > 0:
                for dataset_id in weights[sensor_type]:
                    weights[sensor_type][dataset_id] /= total

        return weights

    def is_complementary(
        self,
        sensor_a: str,
        sensor_b: str
    ) -> bool:
        """
        Check if two sensor types provide complementary information.

        Args:
            sensor_a: First sensor type
            sensor_b: Second sensor type

        Returns:
            True if sensors are complementary
        """
        complementary_pairs = {
            ("sar", "optical"),
            ("optical", "sar"),
            ("thermal", "optical"),
            ("optical", "thermal"),
            ("sar", "dem"),
            ("dem", "sar"),
            ("optical", "dem"),
            ("dem", "optical")
        }
        return (sensor_a, sensor_b) in complementary_pairs

    def is_redundant(
        self,
        sensor_a: str,
        sensor_b: str
    ) -> bool:
        """
        Check if two sensor types provide redundant (overlapping) information.

        Args:
            sensor_a: First sensor type
            sensor_b: Second sensor type

        Returns:
            True if sensors are redundant
        """
        redundant_pairs = {
            ("sar", "sar"),
            ("optical", "optical"),
            ("thermal", "thermal"),
            ("dem", "dem")
        }
        # Same type is always redundant
        if sensor_a == sensor_b:
            return True
        return (sensor_a, sensor_b) in redundant_pairs

    def _filter_by_requirements(
        self,
        candidates: List[DiscoveryResult],
        requirements: Dict[str, Any]
    ) -> List[DiscoveryResult]:
        """Filter candidates by sensor requirements."""
        filtered = []

        for candidate in candidates:
            passes = True

            # Resolution requirement
            if "min_resolution_m" in requirements:
                if candidate.resolution_m < requirements["min_resolution_m"]:
                    passes = False

            if "max_resolution_m" in requirements:
                if candidate.resolution_m > requirements["max_resolution_m"]:
                    passes = False

            # Cloud cover requirement
            if "max_cloud_cover" in requirements:
                if candidate.cloud_cover_percent is not None:
                    if candidate.cloud_cover_percent > requirements["max_cloud_cover"]:
                        passes = False

            # Spatial coverage requirement
            if "min_spatial_coverage" in requirements:
                if candidate.spatial_coverage_percent < requirements["min_spatial_coverage"]:
                    passes = False

            if passes:
                filtered.append(candidate)

        return filtered

    def _select_best_candidates(
        self,
        candidates: List[DiscoveryResult],
        strategy: FusionStrategy,
        temporal_extent: Dict[str, str]
    ) -> List[DiscoveryResult]:
        """Select best candidates based on fusion strategy."""

        if strategy == FusionStrategy.TEMPORAL:
            # Select all to maximize temporal coverage
            return sorted(candidates, key=lambda x: x.acquisition_time)

        elif strategy == FusionStrategy.REDUNDANT:
            # Select multiple for consensus
            sorted_by_quality = sorted(
                candidates,
                key=lambda x: self._quality_factor(x),
                reverse=True
            )
            return sorted_by_quality[:3]  # Top 3 for redundancy

        elif strategy == FusionStrategy.ENSEMBLE:
            # Select all for ensemble
            return sorted(
                candidates,
                key=lambda x: self._quality_factor(x),
                reverse=True
            )[:5]

        else:  # COMPLEMENTARY, HIERARCHICAL
            # Select best single candidate
            best = max(candidates, key=lambda x: self._quality_factor(x))
            return [best]

    def _quality_factor(self, candidate: DiscoveryResult) -> float:
        """Compute quality factor for a candidate (0.0 to 1.0)."""
        factors = []

        # Cloud cover (inverted - lower is better)
        if candidate.cloud_cover_percent is not None:
            cloud_factor = 1.0 - (candidate.cloud_cover_percent / 100.0)
            factors.append(cloud_factor)

        # Spatial coverage
        coverage_factor = candidate.spatial_coverage_percent / 100.0
        factors.append(coverage_factor)

        # Resolution (normalized - lower is better)
        resolution_factor = math.exp(-candidate.resolution_m / 100.0)
        factors.append(resolution_factor)

        # Quality flag
        quality_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3,
            'bad': 0.1
        }
        if candidate.quality_flag:
            factors.append(quality_scores.get(candidate.quality_flag, 0.5))

        # Average factors
        if factors:
            return sum(factors) / len(factors)
        return 0.5

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def _calculate_temporal_coverage(
        self,
        datasets: Dict[str, List[DiscoveryResult]],
        temporal_extent: Dict[str, str]
    ) -> Dict[str, Any]:
        """Calculate temporal coverage statistics."""
        start = datetime.fromisoformat(
            temporal_extent["start"].replace('Z', '+00:00')
        )
        end = datetime.fromisoformat(
            temporal_extent["end"].replace('Z', '+00:00')
        )
        total_hours = (end - start).total_seconds() / 3600

        # Collect all acquisition times
        all_acquisitions = []
        by_sensor: Dict[str, List[datetime]] = {}

        for sensor_type, sensor_datasets in datasets.items():
            by_sensor[sensor_type] = []
            for dataset in sensor_datasets:
                all_acquisitions.append(dataset.acquisition_time)
                by_sensor[sensor_type].append(dataset.acquisition_time)

        # Calculate metrics
        result: Dict[str, Any] = {
            "total_hours": total_hours,
            "total_acquisitions": len(all_acquisitions),
            "by_sensor": {},
        }

        for sensor_type, times in by_sensor.items():
            result["by_sensor"][sensor_type] = len(times)

        # Average revisit time
        if len(all_acquisitions) > 1:
            sorted_times = sorted(all_acquisitions)
            gaps = [
                (sorted_times[i + 1] - sorted_times[i]).total_seconds() / 3600
                for i in range(len(sorted_times) - 1)
            ]
            result["avg_revisit_hours"] = sum(gaps) / len(gaps)
            result["max_gap_hours"] = max(gaps)
        else:
            result["avg_revisit_hours"] = total_hours
            result["max_gap_hours"] = total_hours

        return result

    def _calculate_spatial_coverage(
        self,
        datasets: Dict[str, List[DiscoveryResult]]
    ) -> float:
        """Calculate combined spatial coverage estimate."""
        if not datasets:
            return 0.0

        # Find maximum coverage from any single dataset
        max_coverage = 0.0
        for sensor_datasets in datasets.values():
            for dataset in sensor_datasets:
                max_coverage = max(max_coverage, dataset.spatial_coverage_percent)

        return max_coverage

    def _calculate_fusion_confidence(
        self,
        configuration: FusionConfiguration,
        datasets: Dict[str, List[DiscoveryResult]],
        weights: Dict[str, float]
    ) -> float:
        """Calculate overall confidence in fusion plan."""
        if not datasets:
            return 0.0

        # Check sensor coverage
        required_sensors = {s.sensor_type for s in configuration.sensors}
        available_sensors = set(datasets.keys())

        # Guard against empty configuration (no required sensors)
        if not required_sensors:
            coverage_ratio = 0.0
        else:
            coverage_ratio = len(available_sensors & required_sensors) / len(required_sensors)

        # Quality-weighted confidence
        quality_scores = []
        for sensor_type, sensor_datasets in datasets.items():
            weight = weights.get(sensor_type, 1.0)
            for dataset in sensor_datasets:
                quality_scores.append(self._quality_factor(dataset) * weight)

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
        else:
            avg_quality = 0.5

        # Combine factors
        confidence = coverage_ratio * 0.4 + avg_quality * 0.6

        return min(max(confidence, 0.0), 1.0)

    def _generate_rationale(
        self,
        configuration: FusionConfiguration,
        datasets: Dict[str, List[DiscoveryResult]],
        confidence: float
    ) -> str:
        """Generate human-readable rationale for fusion plan."""
        parts = [
            f"Using {configuration.strategy.value} fusion strategy",
            f"with {configuration.blending_method.value} blending"
        ]

        # Sensor summary
        sensor_counts = []
        for sensor_type, sensor_datasets in datasets.items():
            sensor_counts.append(f"{len(sensor_datasets)} {sensor_type}")

        if sensor_counts:
            parts.append(f"datasets: {', '.join(sensor_counts)}")

        parts.append(f"confidence: {confidence:.2f}")

        return "; ".join(parts)


# Convenience functions

def determine_fusion_strategy(
    event_class: str,
    available_sensors: Set[str],
    atmospheric_conditions: Optional[Dict[str, Any]] = None
) -> FusionConfiguration:
    """
    Convenience function to determine optimal fusion strategy.

    Args:
        event_class: Event classification
        available_sensors: Set of available sensor types
        atmospheric_conditions: Current atmospheric conditions

    Returns:
        Recommended FusionConfiguration
    """
    engine = FusionStrategyEngine()
    return engine.determine_strategy(
        event_class,
        available_sensors,
        atmospheric_conditions
    )


def create_fusion_plan(
    event_class: str,
    candidates: List[DiscoveryResult],
    temporal_extent: Dict[str, str],
    atmospheric_conditions: Optional[Dict[str, Any]] = None
) -> FusionPlan:
    """
    Convenience function to create a fusion plan.

    Args:
        event_class: Event classification
        candidates: List of discovered dataset candidates
        temporal_extent: Temporal extent {start, end, reference_time}
        atmospheric_conditions: Current atmospheric conditions

    Returns:
        FusionPlan with selected datasets and blending weights
    """
    engine = FusionStrategyEngine()

    # Determine available sensors
    available_sensors = {c.data_type for c in candidates}

    # Get configuration
    config = engine.determine_strategy(
        event_class,
        available_sensors,
        atmospheric_conditions
    )

    # Create plan
    return engine.create_fusion_plan(config, candidates, temporal_extent)


def identify_temporal_gaps(
    datasets: List[DiscoveryResult],
    temporal_extent: Dict[str, str],
    max_gap_hours: float = 24.0
) -> List[TemporalGap]:
    """
    Convenience function to identify temporal gaps.

    Args:
        datasets: List of datasets
        temporal_extent: Temporal extent {start, end}
        max_gap_hours: Maximum acceptable gap in hours

    Returns:
        List of TemporalGap objects
    """
    engine = FusionStrategyEngine()

    # Group by sensor type
    by_sensor: Dict[str, List[DiscoveryResult]] = {}
    for dataset in datasets:
        if dataset.data_type not in by_sensor:
            by_sensor[dataset.data_type] = []
        by_sensor[dataset.data_type].append(dataset)

    return engine.identify_temporal_gaps(
        by_sensor,
        temporal_extent,
        max_gap_hours
    )
