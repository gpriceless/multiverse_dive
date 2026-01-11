"""
Quality Control Module for Geospatial Event Intelligence.

Provides comprehensive quality control capabilities for analysis outputs:
- Sanity checks for spatial, value, and temporal coherence
- Cross-validation between models, sensors, and historical data
- Uncertainty quantification and spatial mapping
- Action management (gating, flagging, routing)
- QA reporting and diagnostics

Group I of the development roadmap - Quality Control Citadel.

Submodules:
- sanity: Coherence and plausibility checks
- validation: Cross-validation framework
- uncertainty: Uncertainty quantification and propagation
- actions: Gating, flagging, and expert review routing
- reporting: QA report generation

Example:
    from core.quality import uncertainty
    from core.quality.uncertainty import (
        UncertaintyQuantifier,
        SpatialUncertaintyMapper,
        propagate_quality_uncertainty,
    )
"""

# Import submodules for easy access
from core.quality import uncertainty
from core.quality import sanity
from core.quality import reporting

__all__ = [
    "uncertainty",
    "sanity",
    "reporting",
]
