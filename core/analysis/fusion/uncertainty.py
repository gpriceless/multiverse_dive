"""
Uncertainty Propagation for Multi-Sensor Data Fusion.

Provides tools for tracking and propagating uncertainty through fusion pipelines,
including:
- Per-pixel uncertainty estimation
- Error propagation through operations
- Confidence bound calculation
- Uncertainty visualization and reporting

Key Concepts:
- Uncertainty quantifies our confidence in derived values
- Propagation rules depend on the operations performed
- Multiple uncertainty sources are combined (sensor, algorithm, fusion)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty representation."""
    STANDARD_DEVIATION = "std"          # 1-sigma standard deviation
    VARIANCE = "variance"               # Variance (std^2)
    CONFIDENCE_INTERVAL = "ci"          # Confidence interval bounds
    RELATIVE = "relative"               # Relative uncertainty (%)
    QUANTILES = "quantiles"             # Distribution quantiles
    ENSEMBLE = "ensemble"               # Ensemble of realizations


class UncertaintySource(Enum):
    """Sources of uncertainty in the fusion pipeline."""
    SENSOR = "sensor"                   # Sensor measurement uncertainty
    ATMOSPHERIC = "atmospheric"         # Atmospheric correction uncertainty
    GEOMETRIC = "geometric"             # Registration/alignment uncertainty
    ALGORITHM = "algorithm"             # Algorithm/model uncertainty
    FUSION = "fusion"                   # Multi-source fusion uncertainty
    INTERPOLATION = "interpolation"     # Temporal/spatial interpolation
    EXTRAPOLATION = "extrapolation"     # Extrapolation beyond data
    UNKNOWN = "unknown"                 # Uncharacterized uncertainty


class PropagationMethod(Enum):
    """Methods for propagating uncertainty through operations."""
    LINEAR = "linear"                   # Linear error propagation
    MONTE_CARLO = "monte_carlo"         # Monte Carlo sampling
    ANALYTICAL = "analytical"           # Analytical (closed-form)
    ENSEMBLE = "ensemble"               # Ensemble propagation
    TAYLOR = "taylor"                   # Taylor series expansion


@dataclass
class UncertaintyComponent:
    """
    A single component of uncertainty.

    Attributes:
        source: Source of this uncertainty
        value: Uncertainty value (interpretation depends on type)
        uncertainty_type: How the value is represented
        correlation_length: Spatial correlation length (meters)
        is_systematic: True if systematic (correlated), False if random
        metadata: Additional component metadata
    """
    source: UncertaintySource
    value: Union[float, np.ndarray]
    uncertainty_type: UncertaintyType = UncertaintyType.STANDARD_DEVIATION
    correlation_length: Optional[float] = None
    is_systematic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_variance(self) -> Union[float, np.ndarray]:
        """Convert to variance."""
        if self.uncertainty_type == UncertaintyType.VARIANCE:
            return self.value
        elif self.uncertainty_type == UncertaintyType.STANDARD_DEVIATION:
            return self.value ** 2
        elif self.uncertainty_type == UncertaintyType.RELATIVE:
            # Relative uncertainty - need reference value
            raise ValueError("Cannot convert relative uncertainty to variance without reference value")
        else:
            return self.value ** 2  # Assume std-like

    def to_std(self) -> Union[float, np.ndarray]:
        """Convert to standard deviation."""
        if self.uncertainty_type == UncertaintyType.STANDARD_DEVIATION:
            return self.value
        elif self.uncertainty_type == UncertaintyType.VARIANCE:
            return np.sqrt(self.value)
        else:
            return self.value  # Assume std-like


@dataclass
class UncertaintyBudget:
    """
    Complete uncertainty budget for a data product.

    Attributes:
        components: List of uncertainty components
        total_uncertainty: Combined total uncertainty
        dominant_source: Dominant source of uncertainty
        correlation_matrix: Inter-component correlations
        metadata: Budget metadata
    """
    components: List[UncertaintyComponent]
    total_uncertainty: Optional[Union[float, np.ndarray]] = None
    dominant_source: Optional[UncertaintySource] = None
    correlation_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total uncertainty if not provided."""
        if self.total_uncertainty is None and self.components:
            self.total_uncertainty = self.calculate_total()
            self.dominant_source = self.find_dominant_source()

    def calculate_total(self) -> Union[float, np.ndarray]:
        """Calculate total combined uncertainty."""
        if not self.components:
            return 0.0

        # Sum variances (assuming uncorrelated by default)
        total_variance = sum(c.to_variance() for c in self.components)

        return np.sqrt(total_variance)

    def find_dominant_source(self) -> Optional[UncertaintySource]:
        """Find the dominant source of uncertainty."""
        if not self.components:
            return None

        variances = [(c.source, np.mean(c.to_variance())) for c in self.components]
        dominant = max(variances, key=lambda x: x[1])
        return dominant[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_components": len(self.components),
            "total_uncertainty": float(np.mean(self.total_uncertainty)) if self.total_uncertainty is not None else None,
            "dominant_source": self.dominant_source.value if self.dominant_source else None,
            "components": [
                {
                    "source": c.source.value,
                    "mean_uncertainty": float(np.mean(c.value)),
                    "type": c.uncertainty_type.value,
                    "is_systematic": c.is_systematic,
                }
                for c in self.components
            ],
            "metadata": self.metadata,
        }


@dataclass
class UncertaintyMap:
    """
    Spatial uncertainty map for a data product.

    Attributes:
        uncertainty: Per-pixel uncertainty array
        uncertainty_type: How uncertainty is represented
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        lower_bound: Lower confidence bound (optional)
        upper_bound: Upper confidence bound (optional)
        budget: Full uncertainty budget
        metadata: Map metadata
    """
    uncertainty: np.ndarray
    uncertainty_type: UncertaintyType = UncertaintyType.STANDARD_DEVIATION
    confidence_level: float = 0.68  # 1-sigma
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    budget: Optional[UncertaintyBudget] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence interval bounds.

        Args:
            data: Central values
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Z-score for confidence level (assuming normal distribution)
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)

        std = self.to_std()
        lower = data - z * std
        upper = data + z * std

        return lower, upper

    def to_std(self) -> np.ndarray:
        """Convert to standard deviation."""
        if self.uncertainty_type == UncertaintyType.STANDARD_DEVIATION:
            return self.uncertainty
        elif self.uncertainty_type == UncertaintyType.VARIANCE:
            return np.sqrt(self.uncertainty)
        else:
            return self.uncertainty

    def to_relative(self, data: np.ndarray) -> np.ndarray:
        """Convert to relative uncertainty (%)."""
        std = self.to_std()
        with np.errstate(divide='ignore', invalid='ignore'):
            relative = np.where(np.abs(data) > 1e-10, 100 * std / np.abs(data), 0)
        return relative

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary summary."""
        return {
            "shape": list(self.uncertainty.shape),
            "uncertainty_type": self.uncertainty_type.value,
            "confidence_level": self.confidence_level,
            "mean_uncertainty": float(np.nanmean(self.uncertainty)),
            "max_uncertainty": float(np.nanmax(self.uncertainty)),
            "min_uncertainty": float(np.nanmin(self.uncertainty)),
            "budget": self.budget.to_dict() if self.budget else None,
            "metadata": self.metadata,
        }


@dataclass
class PropagationConfig:
    """
    Configuration for uncertainty propagation.

    Attributes:
        method: Propagation method to use
        num_samples: Number of samples for Monte Carlo
        correlation_aware: Account for spatial correlations
        track_components: Track individual uncertainty components
        min_uncertainty: Minimum uncertainty floor
        max_relative_uncertainty: Maximum allowed relative uncertainty
    """
    method: PropagationMethod = PropagationMethod.LINEAR
    num_samples: int = 1000
    correlation_aware: bool = False
    track_components: bool = True
    min_uncertainty: float = 1e-10
    max_relative_uncertainty: float = 100.0


class UncertaintyPropagator:
    """
    Propagates uncertainty through operations.

    Provides methods for:
    - Linear error propagation
    - Monte Carlo uncertainty estimation
    - Operation-specific propagation rules
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize uncertainty propagator.

        Args:
            config: Propagation configuration
        """
        self.config = config or PropagationConfig()

    def propagate_addition(
        self,
        data_a: np.ndarray,
        uncertainty_a: np.ndarray,
        data_b: np.ndarray,
        uncertainty_b: np.ndarray,
        correlation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through addition: c = a + b.

        Args:
            data_a: First operand data
            uncertainty_a: First operand uncertainty (std)
            data_b: Second operand data
            uncertainty_b: Second operand uncertainty (std)
            correlation: Correlation coefficient between a and b

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        result = data_a + data_b

        # Variance propagation: var(c) = var(a) + var(b) + 2*cov(a,b)
        var_a = uncertainty_a ** 2
        var_b = uncertainty_b ** 2
        cov_ab = correlation * uncertainty_a * uncertainty_b

        var_c = var_a + var_b + 2 * cov_ab
        uncertainty_c = np.sqrt(np.maximum(var_c, self.config.min_uncertainty**2))

        return result, uncertainty_c

    def propagate_subtraction(
        self,
        data_a: np.ndarray,
        uncertainty_a: np.ndarray,
        data_b: np.ndarray,
        uncertainty_b: np.ndarray,
        correlation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through subtraction: c = a - b.

        Args:
            data_a: First operand data
            uncertainty_a: First operand uncertainty (std)
            data_b: Second operand data
            uncertainty_b: Second operand uncertainty (std)
            correlation: Correlation coefficient between a and b

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        result = data_a - data_b

        # Variance propagation: var(c) = var(a) + var(b) - 2*cov(a,b)
        var_a = uncertainty_a ** 2
        var_b = uncertainty_b ** 2
        cov_ab = correlation * uncertainty_a * uncertainty_b

        var_c = var_a + var_b - 2 * cov_ab
        uncertainty_c = np.sqrt(np.maximum(var_c, self.config.min_uncertainty**2))

        return result, uncertainty_c

    def propagate_multiplication(
        self,
        data_a: np.ndarray,
        uncertainty_a: np.ndarray,
        data_b: np.ndarray,
        uncertainty_b: np.ndarray,
        correlation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through multiplication: c = a * b.

        Args:
            data_a: First operand data
            uncertainty_a: First operand uncertainty (std)
            data_b: Second operand data
            uncertainty_b: Second operand uncertainty (std)
            correlation: Correlation coefficient between a and b

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        result = data_a * data_b

        # Relative uncertainty propagation for multiplication
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_a = np.where(np.abs(data_a) > 1e-10, uncertainty_a / np.abs(data_a), 0)
            rel_b = np.where(np.abs(data_b) > 1e-10, uncertainty_b / np.abs(data_b), 0)

        # Relative variance: (sigma_c/c)^2 = (sigma_a/a)^2 + (sigma_b/b)^2 + 2*rho*(sigma_a/a)*(sigma_b/b)
        rel_var_c = rel_a**2 + rel_b**2 + 2 * correlation * rel_a * rel_b

        uncertainty_c = np.abs(result) * np.sqrt(np.maximum(rel_var_c, 0))
        uncertainty_c = np.maximum(uncertainty_c, self.config.min_uncertainty)

        return result, uncertainty_c

    def propagate_division(
        self,
        data_a: np.ndarray,
        uncertainty_a: np.ndarray,
        data_b: np.ndarray,
        uncertainty_b: np.ndarray,
        correlation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through division: c = a / b.

        Args:
            data_a: Numerator data
            uncertainty_a: Numerator uncertainty (std)
            data_b: Denominator data
            uncertainty_b: Denominator uncertainty (std)
            correlation: Correlation coefficient between a and b

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(data_b) > 1e-10, data_a / data_b, np.nan)

        # Relative uncertainty propagation for division
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_a = np.where(np.abs(data_a) > 1e-10, uncertainty_a / np.abs(data_a), 0)
            rel_b = np.where(np.abs(data_b) > 1e-10, uncertainty_b / np.abs(data_b), 0)

        # Relative variance: (sigma_c/c)^2 = (sigma_a/a)^2 + (sigma_b/b)^2 - 2*rho*(sigma_a/a)*(sigma_b/b)
        rel_var_c = rel_a**2 + rel_b**2 - 2 * correlation * rel_a * rel_b

        uncertainty_c = np.abs(result) * np.sqrt(np.maximum(rel_var_c, 0))
        uncertainty_c = np.maximum(uncertainty_c, self.config.min_uncertainty)
        uncertainty_c = np.where(np.isnan(result), np.nan, uncertainty_c)

        return result, uncertainty_c

    def propagate_power(
        self,
        data: np.ndarray,
        uncertainty: np.ndarray,
        exponent: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through power operation: c = a^n.

        Args:
            data: Base data
            uncertainty: Base uncertainty (std)
            exponent: Power exponent

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        result = np.power(data, exponent)

        # sigma_c = |n| * |a|^(n-1) * sigma_a = |n| * |c/a| * sigma_a
        with np.errstate(divide='ignore', invalid='ignore'):
            derivative = np.abs(exponent) * np.abs(result / data)

        uncertainty_c = derivative * uncertainty
        uncertainty_c = np.maximum(uncertainty_c, self.config.min_uncertainty)
        uncertainty_c = np.where(np.isnan(result), np.nan, uncertainty_c)

        return result, uncertainty_c

    def propagate_function(
        self,
        data: np.ndarray,
        uncertainty: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
        derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through an arbitrary function.

        Args:
            data: Input data
            uncertainty: Input uncertainty (std)
            func: Function to apply
            derivative: Derivative of function (computed numerically if not provided)

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        result = func(data)

        if derivative is not None:
            deriv = derivative(data)
        else:
            # Numerical derivative
            h = 1e-7 * (np.abs(data) + 1)
            deriv = (func(data + h) - func(data - h)) / (2 * h)

        uncertainty_c = np.abs(deriv) * uncertainty
        uncertainty_c = np.maximum(uncertainty_c, self.config.min_uncertainty)

        return result, uncertainty_c

    def propagate_weighted_average(
        self,
        data_list: List[np.ndarray],
        uncertainty_list: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through weighted average.

        Args:
            data_list: List of data arrays to average
            uncertainty_list: List of uncertainty arrays
            weights: Optional weights (default: equal weighting)

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        n = len(data_list)
        if weights is None:
            weights = [1.0 / n] * n

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted average
        result = sum(w * d for w, d in zip(weights, data_list))

        # Uncertainty: sqrt(sum(w^2 * sigma^2))
        variance = sum(w**2 * u**2 for w, u in zip(weights, uncertainty_list))
        uncertainty_c = np.sqrt(np.maximum(variance, self.config.min_uncertainty**2))

        return result, uncertainty_c

    def propagate_monte_carlo(
        self,
        data: np.ndarray,
        uncertainty: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty using Monte Carlo sampling.

        Args:
            data: Input data
            uncertainty: Input uncertainty (std)
            func: Function to apply

        Returns:
            Tuple of (result_data, result_uncertainty)
        """
        n_samples = self.config.num_samples

        # Generate samples
        samples = np.random.normal(
            loc=data[np.newaxis, ...],
            scale=uncertainty[np.newaxis, ...],
            size=(n_samples,) + data.shape
        )

        # Apply function to each sample
        results = np.array([func(s) for s in samples])

        # Compute statistics
        result_mean = np.mean(results, axis=0)
        result_std = np.std(results, axis=0)

        return result_mean, result_std


class UncertaintyCombiner:
    """
    Combines uncertainty from multiple sources.

    Handles correlated and uncorrelated uncertainty components.
    """

    def __init__(self):
        """Initialize uncertainty combiner."""
        pass

    def combine_uncorrelated(
        self,
        components: List[UncertaintyComponent],
    ) -> UncertaintyComponent:
        """
        Combine uncorrelated uncertainty components.

        Root sum of squares combination.

        Args:
            components: List of uncertainty components

        Returns:
            Combined uncertainty component
        """
        if not components:
            raise ValueError("No components to combine")

        # Sum variances
        total_variance = sum(c.to_variance() for c in components)
        total_std = np.sqrt(total_variance)

        return UncertaintyComponent(
            source=UncertaintySource.UNKNOWN,
            value=total_std,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            is_systematic=False,
            metadata={"combined_sources": [c.source.value for c in components]},
        )

    def combine_correlated(
        self,
        components: List[UncertaintyComponent],
        correlation_matrix: np.ndarray,
    ) -> UncertaintyComponent:
        """
        Combine correlated uncertainty components.

        Args:
            components: List of uncertainty components
            correlation_matrix: Correlation matrix between components

        Returns:
            Combined uncertainty component
        """
        if not components:
            raise ValueError("No components to combine")

        n = len(components)
        if correlation_matrix.shape != (n, n):
            raise ValueError(f"Correlation matrix shape {correlation_matrix.shape} doesn't match {n} components")

        # Build covariance matrix
        stds = np.array([np.mean(c.to_std()) if isinstance(c.to_std(), np.ndarray) else c.to_std()
                        for c in components])

        covariance = np.outer(stds, stds) * correlation_matrix

        # Total variance = sum of all covariance matrix elements
        total_variance = np.sum(covariance)
        total_std = np.sqrt(max(total_variance, 0))

        return UncertaintyComponent(
            source=UncertaintySource.UNKNOWN,
            value=total_std,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            is_systematic=any(c.is_systematic for c in components),
            metadata={
                "combined_sources": [c.source.value for c in components],
                "correlation_matrix": correlation_matrix.tolist(),
            },
        )

    def combine_systematic_random(
        self,
        systematic: UncertaintyComponent,
        random: UncertaintyComponent,
    ) -> UncertaintyBudget:
        """
        Combine systematic and random uncertainty.

        Args:
            systematic: Systematic uncertainty component
            random: Random uncertainty component

        Returns:
            UncertaintyBudget with both components
        """
        # Systematic and random combine as RSS
        total_variance = systematic.to_variance() + random.to_variance()
        total_std = np.sqrt(total_variance)

        return UncertaintyBudget(
            components=[systematic, random],
            total_uncertainty=total_std,
            dominant_source=systematic.source if systematic.to_variance() > random.to_variance() else random.source,
            metadata={"combination_method": "systematic_random_rss"},
        )


class FusionUncertaintyEstimator:
    """
    Estimates uncertainty specific to multi-sensor fusion.

    Handles fusion-specific uncertainty sources:
    - Sensor disagreement
    - Temporal interpolation
    - Spatial alignment
    - Algorithm differences
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize fusion uncertainty estimator.

        Args:
            config: Propagation configuration
        """
        self.config = config or PropagationConfig()
        self.propagator = UncertaintyPropagator(config)
        self.combiner = UncertaintyCombiner()

    def estimate_from_disagreement(
        self,
        data_list: List[np.ndarray],
        quality_list: Optional[List[np.ndarray]] = None,
    ) -> UncertaintyMap:
        """
        Estimate uncertainty from sensor disagreement.

        Args:
            data_list: List of data arrays from different sources
            quality_list: Optional quality weights for each source

        Returns:
            UncertaintyMap based on source disagreement
        """
        if len(data_list) < 2:
            # Single source - return minimum uncertainty
            shape = data_list[0].shape if data_list else (0, 0)
            return UncertaintyMap(
                uncertainty=np.full(shape, self.config.min_uncertainty),
                uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
                metadata={"source": "single_source_minimum"},
            )

        # Stack data
        data_stack = np.stack([d.astype(np.float64) for d in data_list])

        if quality_list:
            # Quality-weighted statistics
            quality_stack = np.stack(quality_list)
            valid_mask = quality_stack > 0
            masked_data = np.where(valid_mask, data_stack, np.nan)
        else:
            masked_data = data_stack

        # Calculate standard deviation as disagreement
        disagreement_std = np.nanstd(masked_data, axis=0)

        # Floor at minimum uncertainty
        disagreement_std = np.maximum(disagreement_std, self.config.min_uncertainty)

        return UncertaintyMap(
            uncertainty=disagreement_std,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            budget=UncertaintyBudget(
                components=[UncertaintyComponent(
                    source=UncertaintySource.FUSION,
                    value=disagreement_std,
                    metadata={"num_sources": len(data_list)},
                )],
            ),
            metadata={"source": "sensor_disagreement", "num_sources": len(data_list)},
        )

    def estimate_interpolation_uncertainty(
        self,
        data_before: np.ndarray,
        data_after: np.ndarray,
        time_fraction: float,
        method: str = "linear",
    ) -> UncertaintyMap:
        """
        Estimate uncertainty from temporal interpolation.

        Args:
            data_before: Data at earlier time
            data_after: Data at later time
            time_fraction: Fraction of interval (0=before, 1=after)
            method: Interpolation method used

        Returns:
            UncertaintyMap for interpolated data
        """
        # Uncertainty grows with interpolation distance from observations
        # Maximum at midpoint (time_fraction = 0.5)
        distance_factor = 2 * abs(time_fraction - 0.5)  # 0 at edges, 1 at center

        # Base uncertainty from difference between observations
        temporal_change = np.abs(data_after - data_before)

        # Interpolation uncertainty proportional to change and distance
        interpolation_std = distance_factor * temporal_change * 0.5

        # Add minimum floor
        interpolation_std = np.maximum(interpolation_std, self.config.min_uncertainty)

        return UncertaintyMap(
            uncertainty=interpolation_std,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            budget=UncertaintyBudget(
                components=[UncertaintyComponent(
                    source=UncertaintySource.INTERPOLATION,
                    value=interpolation_std,
                    metadata={
                        "method": method,
                        "time_fraction": time_fraction,
                    },
                )],
            ),
            metadata={"source": "temporal_interpolation", "method": method},
        )

    def estimate_alignment_uncertainty(
        self,
        offset_pixels: float,
        gradient_magnitude: np.ndarray,
        pixel_size: float = 10.0,
    ) -> UncertaintyMap:
        """
        Estimate uncertainty from spatial alignment.

        Args:
            offset_pixels: Estimated alignment offset in pixels
            gradient_magnitude: Spatial gradient magnitude of data
            pixel_size: Pixel size in meters

        Returns:
            UncertaintyMap for aligned data
        """
        # Uncertainty proportional to offset and local gradient
        # Higher gradients = more sensitivity to alignment errors
        alignment_std = offset_pixels * gradient_magnitude

        # Minimum floor
        alignment_std = np.maximum(alignment_std, self.config.min_uncertainty)

        return UncertaintyMap(
            uncertainty=alignment_std,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            budget=UncertaintyBudget(
                components=[UncertaintyComponent(
                    source=UncertaintySource.GEOMETRIC,
                    value=alignment_std,
                    correlation_length=pixel_size * 3,  # Correlation ~3 pixels
                    is_systematic=True,
                    metadata={"offset_pixels": offset_pixels},
                )],
            ),
            metadata={"source": "spatial_alignment", "offset_pixels": offset_pixels},
        )

    def combine_fusion_uncertainties(
        self,
        sensor_uncertainty: UncertaintyMap,
        disagreement_uncertainty: Optional[UncertaintyMap] = None,
        interpolation_uncertainty: Optional[UncertaintyMap] = None,
        alignment_uncertainty: Optional[UncertaintyMap] = None,
    ) -> UncertaintyMap:
        """
        Combine all fusion-related uncertainties.

        Args:
            sensor_uncertainty: Base sensor measurement uncertainty
            disagreement_uncertainty: Uncertainty from sensor disagreement
            interpolation_uncertainty: Uncertainty from interpolation
            alignment_uncertainty: Uncertainty from alignment

        Returns:
            Combined UncertaintyMap
        """
        components = []

        # Collect all components
        if sensor_uncertainty.budget:
            components.extend(sensor_uncertainty.budget.components)
        else:
            components.append(UncertaintyComponent(
                source=UncertaintySource.SENSOR,
                value=sensor_uncertainty.uncertainty,
            ))

        if disagreement_uncertainty and disagreement_uncertainty.budget:
            components.extend(disagreement_uncertainty.budget.components)
        elif disagreement_uncertainty:
            components.append(UncertaintyComponent(
                source=UncertaintySource.FUSION,
                value=disagreement_uncertainty.uncertainty,
            ))

        if interpolation_uncertainty and interpolation_uncertainty.budget:
            components.extend(interpolation_uncertainty.budget.components)
        elif interpolation_uncertainty:
            components.append(UncertaintyComponent(
                source=UncertaintySource.INTERPOLATION,
                value=interpolation_uncertainty.uncertainty,
            ))

        if alignment_uncertainty and alignment_uncertainty.budget:
            components.extend(alignment_uncertainty.budget.components)
        elif alignment_uncertainty:
            components.append(UncertaintyComponent(
                source=UncertaintySource.GEOMETRIC,
                value=alignment_uncertainty.uncertainty,
            ))

        # Combine components
        combined = self.combiner.combine_uncorrelated(components)

        budget = UncertaintyBudget(
            components=components,
            total_uncertainty=combined.value,
            dominant_source=combined.source,
        )

        return UncertaintyMap(
            uncertainty=combined.value,
            uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
            budget=budget,
            metadata={"source": "combined_fusion"},
        )


# Convenience functions

def estimate_uncertainty_from_samples(
    data_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> UncertaintyMap:
    """
    Estimate uncertainty from multiple samples.

    Args:
        data_list: List of data arrays (samples/realizations)
        weights: Optional sample weights

    Returns:
        UncertaintyMap based on sample spread
    """
    estimator = FusionUncertaintyEstimator()
    return estimator.estimate_from_disagreement(data_list)


def propagate_through_operation(
    data: np.ndarray,
    uncertainty: np.ndarray,
    operation: str,
    operand: Optional[Union[float, np.ndarray]] = None,
    operand_uncertainty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainty through a simple operation.

    Args:
        data: Input data
        uncertainty: Input uncertainty (std)
        operation: Operation name (add, subtract, multiply, divide, power)
        operand: Second operand (for binary operations)
        operand_uncertainty: Uncertainty of second operand

    Returns:
        Tuple of (result_data, result_uncertainty)
    """
    propagator = UncertaintyPropagator()

    if operation in ("add", "+"):
        if operand is None:
            raise ValueError("Operand required for addition")
        operand = np.asarray(operand)
        operand_uncertainty = operand_uncertainty if operand_uncertainty is not None else np.zeros_like(operand)
        return propagator.propagate_addition(data, uncertainty, operand, operand_uncertainty)

    elif operation in ("subtract", "-"):
        if operand is None:
            raise ValueError("Operand required for subtraction")
        operand = np.asarray(operand)
        operand_uncertainty = operand_uncertainty if operand_uncertainty is not None else np.zeros_like(operand)
        return propagator.propagate_subtraction(data, uncertainty, operand, operand_uncertainty)

    elif operation in ("multiply", "*"):
        if operand is None:
            raise ValueError("Operand required for multiplication")
        operand = np.asarray(operand)
        operand_uncertainty = operand_uncertainty if operand_uncertainty is not None else np.zeros_like(operand)
        return propagator.propagate_multiplication(data, uncertainty, operand, operand_uncertainty)

    elif operation in ("divide", "/"):
        if operand is None:
            raise ValueError("Operand required for division")
        operand = np.asarray(operand)
        operand_uncertainty = operand_uncertainty if operand_uncertainty is not None else np.zeros_like(operand)
        return propagator.propagate_division(data, uncertainty, operand, operand_uncertainty)

    elif operation in ("power", "**"):
        if operand is None:
            raise ValueError("Exponent required for power")
        return propagator.propagate_power(data, uncertainty, float(operand))

    else:
        raise ValueError(f"Unknown operation: {operation}")


def combine_uncertainties(
    uncertainties: List[np.ndarray],
    method: str = "rss",
) -> np.ndarray:
    """
    Combine multiple uncertainty arrays.

    Args:
        uncertainties: List of uncertainty arrays (std)
        method: Combination method (rss, linear, max)

    Returns:
        Combined uncertainty array
    """
    if not uncertainties:
        raise ValueError("No uncertainties to combine")

    if method == "rss":
        # Root sum of squares (uncorrelated)
        total_variance = sum(u**2 for u in uncertainties)
        return np.sqrt(total_variance)

    elif method == "linear":
        # Linear sum (fully correlated)
        return sum(uncertainties)

    elif method == "max":
        # Maximum (conservative)
        return np.maximum.reduce(uncertainties)

    else:
        raise ValueError(f"Unknown method: {method}")
