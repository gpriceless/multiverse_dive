"""
Fusion Core Module - Multi-Sensor Data Fusion for Geospatial Analysis.

This module provides comprehensive tools for fusing data from multiple sensors,
including spatial/temporal alignment, corrections, conflict resolution, and
uncertainty propagation.

Submodules:
- alignment: Spatial and temporal alignment of multi-sensor data
- corrections: Terrain, atmospheric, and radiometric corrections
- conflict: Conflict detection and resolution between data sources
- uncertainty: Uncertainty propagation through fusion operations

Example Usage:
    from core.analysis.fusion import (
        # Alignment
        MultiSensorAligner,
        ReferenceGrid,
        create_reference_grid,
        align_datasets,

        # Corrections
        CorrectionPipeline,
        apply_terrain_correction,
        apply_atmospheric_correction,

        # Conflict Resolution
        ConflictResolver,
        resolve_conflicts,
        build_consensus,

        # Uncertainty
        UncertaintyPropagator,
        FusionUncertaintyEstimator,
        combine_uncertainties,
    )
"""

# Alignment module exports
from core.analysis.fusion.alignment import (
    # Enums
    SpatialAlignmentMethod,
    TemporalAlignmentMethod,
    AlignmentQuality,
    # Config dataclasses
    ReferenceGrid,
    TemporalBin,
    SpatialAlignmentConfig,
    TemporalAlignmentConfig,
    # Result dataclasses
    AlignedLayer,
    AlignmentResult,
    # Core classes
    SpatialAligner,
    TemporalAligner,
    MultiSensorAligner,
    # Convenience functions
    create_reference_grid,
    align_datasets,
)

# Corrections module exports
from core.analysis.fusion.corrections import (
    # Enums
    TerrainCorrectionMethod,
    AtmosphericCorrectionMethod,
    NormalizationMethod,
    # Config dataclasses
    TerrainCorrectionConfig,
    AtmosphericCorrectionConfig,
    NormalizationConfig,
    # Result dataclasses
    CorrectionResult,
    # Core classes
    TerrainCorrector,
    AtmosphericCorrector,
    RadiometricNormalizer,
    CorrectionPipeline,
    # Convenience functions
    apply_terrain_correction,
    apply_atmospheric_correction,
    normalize_to_reference,
)

# Conflict resolution module exports
from core.analysis.fusion.conflict import (
    # Enums
    ConflictResolutionStrategy,
    ConflictSeverity,
    # Config dataclasses
    ConflictThresholds,
    ConflictConfig,
    SourceLayer,
    # Result dataclasses
    ConflictMap,
    ConflictResolutionResult,
    # Core classes
    ConflictDetector,
    ConflictResolver,
    ConsensusBuilder,
    # Convenience functions
    detect_conflicts,
    resolve_conflicts,
    build_consensus,
)

# Uncertainty module exports
from core.analysis.fusion.uncertainty import (
    # Enums
    UncertaintyType,
    UncertaintySource,
    PropagationMethod,
    # Config dataclasses
    UncertaintyComponent,
    UncertaintyBudget,
    UncertaintyMap,
    PropagationConfig,
    # Core classes
    UncertaintyPropagator,
    UncertaintyCombiner,
    FusionUncertaintyEstimator,
    # Convenience functions
    estimate_uncertainty_from_samples,
    propagate_through_operation,
    combine_uncertainties,
)

__all__ = [
    # Alignment
    "SpatialAlignmentMethod",
    "TemporalAlignmentMethod",
    "AlignmentQuality",
    "ReferenceGrid",
    "TemporalBin",
    "SpatialAlignmentConfig",
    "TemporalAlignmentConfig",
    "AlignedLayer",
    "AlignmentResult",
    "SpatialAligner",
    "TemporalAligner",
    "MultiSensorAligner",
    "create_reference_grid",
    "align_datasets",
    # Corrections
    "TerrainCorrectionMethod",
    "AtmosphericCorrectionMethod",
    "NormalizationMethod",
    "TerrainCorrectionConfig",
    "AtmosphericCorrectionConfig",
    "NormalizationConfig",
    "CorrectionResult",
    "TerrainCorrector",
    "AtmosphericCorrector",
    "RadiometricNormalizer",
    "CorrectionPipeline",
    "apply_terrain_correction",
    "apply_atmospheric_correction",
    "normalize_to_reference",
    # Conflict Resolution
    "ConflictResolutionStrategy",
    "ConflictSeverity",
    "ConflictThresholds",
    "ConflictConfig",
    "SourceLayer",
    "ConflictMap",
    "ConflictResolutionResult",
    "ConflictDetector",
    "ConflictResolver",
    "ConsensusBuilder",
    "detect_conflicts",
    "resolve_conflicts",
    "build_consensus",
    # Uncertainty
    "UncertaintyType",
    "UncertaintySource",
    "PropagationMethod",
    "UncertaintyComponent",
    "UncertaintyBudget",
    "UncertaintyMap",
    "PropagationConfig",
    "UncertaintyPropagator",
    "UncertaintyCombiner",
    "FusionUncertaintyEstimator",
    "estimate_uncertainty_from_samples",
    "propagate_through_operation",
    "combine_uncertainties",
]
