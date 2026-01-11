"""
Execution Infrastructure for Multiverse Dive.

Provides execution profiles, state persistence, and tiled processing
for resource-constrained environments.

Key Components (Group L):
- ExecutionProfile: Resource constraint definitions (Tracks 7-8)
- ProfileManager: Profile selection and validation
- WorkflowState: Workflow state tracking
- StateManager: State persistence and recovery
- TileScheme: Define tile sizes and overlap configurations (Track 1)
- TileGrid: Generate tile grids covering an AOI
- TileManager: Track tile processing state with resume support
- OverlapHandler: Handle edge effects through overlap blending
"""

from core.execution.profiles import (
    ExecutionProfile,
    ProcessingMode,
    ProfileManager,
    ProfileName,
    SystemResources,
    auto_select_profile,
    detect_resources,
    get_profile,
)
from core.execution.state import (
    Checkpoint,
    StateManager,
    StepStatus,
    WorkflowStage,
    WorkflowState,
    WorkflowStep,
    create_workflow_state,
    get_state_manager,
)
from core.execution.tiling import (
    # Enums
    TileSizePreset,
    BlendMode,
    TileStatus,
    CoordinateSystem,
    # Data classes
    TileIndex,
    TileBounds,
    PixelBounds,
    TileInfo,
    # Core classes
    TileScheme,
    TileGrid,
    TileManager,
    OverlapHandler,
    # Utility functions
    create_tile_grid,
    estimate_memory_per_tile,
    suggest_tile_size,
    compute_grid_hash,
)

__all__ = [
    # Profiles (Track 7)
    "ExecutionProfile",
    "ProcessingMode",
    "ProfileManager",
    "ProfileName",
    "SystemResources",
    "auto_select_profile",
    "detect_resources",
    "get_profile",
    # State (Track 8)
    "Checkpoint",
    "StateManager",
    "StepStatus",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowStep",
    "create_workflow_state",
    "get_state_manager",
    # Tiling (Group L, Track 1)
    "TileSizePreset",
    "BlendMode",
    "TileStatus",
    "CoordinateSystem",
    "TileIndex",
    "TileBounds",
    "PixelBounds",
    "TileInfo",
    "TileScheme",
    "TileGrid",
    "TileManager",
    "OverlapHandler",
    "create_tile_grid",
    "estimate_memory_per_tile",
    "suggest_tile_size",
    "compute_grid_hash",
]
