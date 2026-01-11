"""
Pipeline Agent Module for Event Intelligence Platform.

Provides pipeline orchestration capabilities including:
- Pipeline assembly from specifications or dynamic construction
- Algorithm selection based on event type and data availability
- Step-by-step execution with progress tracking
- Parallel and distributed execution support
- Checkpointing and recovery
- Real-time progress monitoring and metrics

Main Components:
- PipelineAgent: Main agent for pipeline orchestration
- PipelineBuilder: Builds pipelines from event types and data
- ExecutionEngine: Manages pipeline execution
- ProgressTracker: Tracks execution progress and metrics

Example Usage:
    from agents.pipeline import (
        PipelineAgent,
        PipelineBuilder,
        ExecutionEngine,
        ProgressTracker,
    )

    # Create and start agent
    agent = PipelineAgent()
    await agent.start()

    # Assemble pipeline for flood event
    pipeline = await agent.assemble_pipeline(
        event_type="flood.coastal",
        available_data={"sar": DataType.RASTER, "dem": DataType.RASTER}
    )

    # Execute pipeline
    result = await agent.execute_pipeline(
        pipeline_spec=spec,
        inputs=data
    )

    await agent.stop()

Integration Points:
- Uses core/analysis/assembly/ for DAG building and validation
- Uses core/analysis/execution/ for distributed execution
- Uses core/analysis/library/ for algorithm registry
- Coordinates with Quality agent for inline validation
"""

# Main agent
from agents.pipeline.main import (
    # Agent classes
    BaseAgent,
    PipelineAgent,
    # Status enums
    AgentStatus,
    StepStatus,
    # Data classes
    PipelineStep,
    StepResult,
    PipelineResult,
)

# Assembly module
from agents.pipeline.assembly import (
    # Builder class
    PipelineBuilder,
    # Configuration
    BuildConfig,
    BuildStrategy,
    SelectionCriteria,
    # Results
    BuildResult,
    AlgorithmScore,
)

# Execution module
from agents.pipeline.execution import (
    # Engine class
    ExecutionEngine,
    # Configuration
    ExecutionEngineConfig,
    ResourceLimits,
    ResourceLimitPolicy,
    # Results and state
    StepExecutionResult,
    ExecutionState,
)

# Monitoring module
from agents.pipeline.monitoring import (
    # Tracker class
    ProgressTracker,
    # Metrics
    PipelineMetrics,
    StepMetrics,
    GroupMetrics,
    ProgressSnapshot,
    # Events
    ExecutionEvent,
    EventType,
    MetricType,
)

__all__ = [
    # Main agent
    "BaseAgent",
    "PipelineAgent",
    "AgentStatus",
    "StepStatus",
    "PipelineStep",
    "StepResult",
    "PipelineResult",
    # Assembly
    "PipelineBuilder",
    "BuildConfig",
    "BuildStrategy",
    "SelectionCriteria",
    "BuildResult",
    "AlgorithmScore",
    # Execution
    "ExecutionEngine",
    "ExecutionEngineConfig",
    "ResourceLimits",
    "ResourceLimitPolicy",
    "StepExecutionResult",
    "ExecutionState",
    # Monitoring
    "ProgressTracker",
    "PipelineMetrics",
    "StepMetrics",
    "GroupMetrics",
    "ProgressSnapshot",
    "ExecutionEvent",
    "EventType",
    "MetricType",
]
