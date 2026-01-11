"""
Pipeline Agent for Event Intelligence Platform.

Orchestrates analysis pipeline assembly and execution, coordinating with
Quality agent for inline validation, handling checkpointing and resume,
and reporting progress to the Orchestrator.

Based on the agent architecture from OPENSPEC.md.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    NodeType,
    NodeStatus,
    DataType,
)
from core.analysis.assembly.assembler import (
    PipelineAssembler,
    DynamicAssembler,
    PipelineSpec,
    AssemblyResult,
    AssemblyError,
)
from core.analysis.assembly.validator import (
    PipelineValidator,
    ValidationResult,
    ValidationSeverity,
)
from core.analysis.assembly.optimizer import (
    PipelineOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    ExecutionPlan,
)
from core.analysis.execution.runner import (
    ExecutionConfig,
    ExecutionMode,
    ExecutionResult as CoreExecutionResult,
    TaskStatus,
    TaskResult,
    ExecutionProgress,
)
from core.analysis.execution.checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetadata,
    CheckpointType,
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    get_global_registry,
    load_default_algorithms,
)

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of the pipeline agent."""
    IDLE = "idle"
    ASSEMBLING = "assembling"
    VALIDATING = "validating"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """
    Represents a single step in the pipeline.

    Attributes:
        step_id: Unique step identifier
        name: Human-readable step name
        processor: Algorithm ID for processing
        inputs: Input references (node IDs or data names)
        parameters: Step-specific parameters
        status: Current step status
        result: Step execution result
        start_time: When step started
        end_time: When step completed
        error_message: Error description if failed
    """
    step_id: str
    name: str
    processor: str
    inputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "processor": self.processor,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class StepResult:
    """
    Result from executing a pipeline step.

    Attributes:
        step_id: Step identifier
        success: Whether step succeeded
        output_data: Output data from step
        metrics: Execution metrics
        quality_scores: Quality assessment scores
        warnings: Any warnings generated
        error: Error message if failed
    """
    step_id: str
    success: bool
    output_data: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "has_output": self.output_data is not None,
            "metrics": self.metrics,
            "quality_scores": self.quality_scores,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """
    Complete result of pipeline execution.

    Attributes:
        pipeline_id: Pipeline identifier
        execution_id: Unique execution run ID
        success: Whether execution succeeded
        step_results: Results from each step
        final_outputs: Final pipeline outputs
        total_duration_seconds: Total execution time
        checkpoints_created: Number of checkpoints created
        metadata: Additional execution metadata
    """
    pipeline_id: str
    execution_id: str
    success: bool
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    final_outputs: Dict[str, Any] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    checkpoints_created: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "execution_id": self.execution_id,
            "success": self.success,
            "step_results": {
                k: v.to_dict() for k, v in self.step_results.items()
            },
            "final_outputs": list(self.final_outputs.keys()),
            "total_duration_seconds": self.total_duration_seconds,
            "checkpoints_created": self.checkpoints_created,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides common functionality for:
    - Agent lifecycle management
    - Progress reporting
    - Error handling
    - Communication with orchestrator
    """

    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique agent identifier (generated if not provided)
        """
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.status = AgentStatus.IDLE
        self._progress_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._created_at = datetime.now(timezone.utc)
        logger.info(f"Initialized agent {self.__class__.__name__} ({self.agent_id})")

    def register_progress_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _report_progress(self, progress_data: Dict[str, Any]) -> None:
        """Report progress to registered callbacks."""
        progress_data["agent_id"] = self.agent_id
        progress_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        for callback in self._progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    @abstractmethod
    async def start(self) -> None:
        """Start the agent."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent."""
        pass


class PipelineAgent(BaseAgent):
    """
    Pipeline Agent for orchestrating analysis pipeline assembly and execution.

    Responsibilities:
    - Assemble pipelines from specifications or dynamically
    - Validate pipelines before execution
    - Execute pipelines with progress tracking
    - Handle checkpointing and recovery
    - Coordinate with Quality agent for inline validation
    - Report progress to Orchestrator

    Usage:
        agent = PipelineAgent()
        await agent.start()

        # Assemble and execute
        pipeline = await agent.assemble_pipeline("flood", available_data)
        result = await agent.execute_pipeline(pipeline_spec, inputs)

        await agent.stop()
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        registry: Optional[AlgorithmRegistry] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """
        Initialize the Pipeline Agent.

        Args:
            agent_id: Agent identifier
            registry: Algorithm registry (uses global if None)
            checkpoint_config: Configuration for checkpointing
            execution_config: Configuration for execution
        """
        super().__init__(agent_id)

        # Initialize registry
        self.registry = registry
        if self.registry is None:
            self.registry = get_global_registry()
            if not self.registry.algorithms:
                load_default_algorithms()

        # Initialize components
        self.assembler = PipelineAssembler(registry=self.registry)
        self.dynamic_assembler = DynamicAssembler(registry=self.registry)
        self.validator = PipelineValidator(registry=self.registry)
        self.optimizer = PipelineOptimizer(registry=self.registry)

        # Checkpoint management
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_config)

        # Execution configuration
        self.execution_config = execution_config or ExecutionConfig()

        # State tracking
        self._current_execution_id: Optional[str] = None
        self._current_pipeline: Optional[PipelineGraph] = None
        self._step_states: Dict[str, PipelineStep] = {}
        self._cancelled = False

        # Quality agent reference (set by orchestrator)
        self._quality_agent: Optional[Any] = None

        logger.info(
            f"PipelineAgent initialized with {len(self.registry.algorithms)} algorithms"
        )

    async def start(self) -> None:
        """Start the pipeline agent."""
        self.status = AgentStatus.IDLE
        self._cancelled = False
        logger.info(f"PipelineAgent {self.agent_id} started")

    async def stop(self) -> None:
        """Stop the pipeline agent."""
        self._cancelled = True
        self.status = AgentStatus.IDLE
        logger.info(f"PipelineAgent {self.agent_id} stopped")

    def set_quality_agent(self, quality_agent: Any) -> None:
        """Set reference to quality agent for inline validation."""
        self._quality_agent = quality_agent

    async def execute_pipeline(
        self,
        pipeline_spec: Dict[str, Any],
        inputs: Dict[str, Any],
        resume_from_checkpoint: bool = False,
    ) -> PipelineResult:
        """
        Execute a complete analysis pipeline.

        Args:
            pipeline_spec: Pipeline specification dictionary
            inputs: Input data for the pipeline
            resume_from_checkpoint: Whether to resume from last checkpoint

        Returns:
            PipelineResult with execution details
        """
        execution_id = str(uuid.uuid4())[:16]
        self._current_execution_id = execution_id
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting pipeline execution: {execution_id}")
        self.status = AgentStatus.ASSEMBLING

        self._report_progress({
            "phase": "starting",
            "execution_id": execution_id,
            "message": "Beginning pipeline execution",
        })

        try:
            # Phase 1: Assemble pipeline
            self.status = AgentStatus.ASSEMBLING
            spec = PipelineSpec.from_dict(pipeline_spec)
            assembly_result = self.assembler.assemble(spec)

            if not assembly_result.success:
                raise AssemblyError("Pipeline assembly failed")

            graph = assembly_result.graph
            self._current_pipeline = graph

            # Phase 2: Validate pipeline
            self.status = AgentStatus.VALIDATING
            validation_result = self.validator.validate(graph)

            if not validation_result.is_valid:
                errors = [
                    issue.message
                    for issue in validation_result.errors
                ]
                raise AssemblyError(f"Pipeline validation failed: {errors}")

            # Phase 3: Optimize execution plan
            optimization_result = self.optimizer.optimize(graph)
            execution_plan = optimization_result.execution_plan

            # Phase 4: Execute pipeline
            self.status = AgentStatus.EXECUTING
            step_results: Dict[str, StepResult] = {}
            final_outputs: Dict[str, Any] = {}
            checkpoints_created = 0

            # Handle checkpoint resume
            if resume_from_checkpoint:
                checkpoint_data = self.checkpoint_manager.load_latest(execution_id)
                if checkpoint_data:
                    step_results = self._restore_step_results(
                        checkpoint_data.get("step_results", {})
                    )
                    logger.info(
                        f"Resumed from checkpoint with {len(step_results)} completed steps"
                    )

            # Execute steps in order from execution plan
            completed_steps = set(step_results.keys())

            for group in execution_plan.groups:
                if self._cancelled:
                    break

                # Execute steps in group (potentially parallel)
                for step_id in group.node_ids:
                    if step_id in completed_steps:
                        continue

                    node = graph.get_node(step_id)
                    if node is None or node.node_type != NodeType.PROCESSOR:
                        continue

                    # Create step representation
                    step = PipelineStep(
                        step_id=step_id,
                        name=node.name,
                        processor=node.processor or "",
                        inputs=[],
                        parameters=node.parameters,
                    )
                    self._step_states[step_id] = step

                    # Prepare step inputs from previous results
                    step_inputs = self._prepare_step_inputs(
                        node, inputs, step_results, graph
                    )

                    # Execute step
                    result = await self.run_step(step, step_inputs)
                    step_results[step_id] = result

                    if not result.success:
                        await self.handle_step_failure(step_id, Exception(result.error))
                        if self.execution_config.fail_fast:
                            break

                    # Checkpoint after group
                    if group.checkpoint_before:
                        await self.checkpoint(
                            step_id,
                            {"step_results": step_results, "inputs": inputs}
                        )
                        checkpoints_created += 1

            # Collect final outputs
            for output_node in graph.get_output_nodes():
                incoming_edges = graph.get_incoming_edges(output_node.id)
                for edge in incoming_edges:
                    source_result = step_results.get(edge.source_node)
                    if source_result and source_result.output_data:
                        final_outputs[output_node.name] = source_result.output_data

            # Determine success
            success = all(r.success for r in step_results.values())
            self.status = AgentStatus.COMPLETED if success else AgentStatus.FAILED

            end_time = datetime.now(timezone.utc)

            result = PipelineResult(
                pipeline_id=spec.id,
                execution_id=execution_id,
                success=success,
                step_results=step_results,
                final_outputs=final_outputs,
                total_duration_seconds=(end_time - start_time).total_seconds(),
                checkpoints_created=checkpoints_created,
                start_time=start_time,
                end_time=end_time,
                metadata={
                    "algorithms_used": list(assembly_result.algorithms_used.keys()),
                    "validation_warnings": len(validation_result.warnings),
                    "optimization_applied": optimization_result.execution_plan.optimizations_applied,
                },
            )

            self._report_progress({
                "phase": "completed",
                "execution_id": execution_id,
                "success": success,
                "duration_seconds": result.total_duration_seconds,
            })

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.status = AgentStatus.FAILED

            self._report_progress({
                "phase": "failed",
                "execution_id": execution_id,
                "error": str(e),
            })

            return PipelineResult(
                pipeline_id=pipeline_spec.get("id", "unknown"),
                execution_id=execution_id,
                success=False,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                metadata={"error": str(e)},
            )

        finally:
            self._current_execution_id = None
            self._current_pipeline = None

    async def assemble_pipeline(
        self,
        event_type: str,
        available_data: Dict[str, Any],
        max_algorithms: int = 1,
    ) -> PipelineGraph:
        """
        Dynamically assemble a pipeline for the given event type.

        Args:
            event_type: Type of event (e.g., "flood", "wildfire")
            available_data: Available data sources and types
            max_algorithms: Maximum algorithms to include

        Returns:
            Assembled PipelineGraph
        """
        logger.info(f"Assembling pipeline for event type: {event_type}")
        self.status = AgentStatus.ASSEMBLING

        # Convert available data to DataType mapping
        data_types = {}
        for name, info in available_data.items():
            if isinstance(info, dict):
                dtype = info.get("type", "raster")
            else:
                dtype = str(info)
            data_types[name] = DataType(dtype) if dtype in [
                dt.value for dt in DataType
            ] else DataType.RASTER

        # Use dynamic assembler
        result = self.dynamic_assembler.create_for_event(
            event_class=event_type,
            available_data=data_types,
            max_algorithms=max_algorithms,
        )

        self.status = AgentStatus.IDLE
        return result.graph

    async def run_step(
        self,
        step: PipelineStep,
        inputs: Dict[str, Any],
    ) -> StepResult:
        """
        Execute a single pipeline step.

        Args:
            step: Pipeline step to execute
            inputs: Input data for the step

        Returns:
            StepResult with execution details
        """
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now(timezone.utc)

        logger.info(f"Executing step: {step.step_id} ({step.processor})")

        self._report_progress({
            "phase": "step_started",
            "step_id": step.step_id,
            "processor": step.processor,
        })

        try:
            # Get algorithm from registry
            algorithm = self.registry.get(step.processor)

            if algorithm is None:
                raise ValueError(f"Algorithm {step.processor} not found in registry")

            # Execute algorithm (simulated for now - real implementation would
            # call the actual algorithm module)
            output_data = await self._execute_algorithm(
                algorithm, inputs, step.parameters
            )

            # Run quality checks if quality agent available
            quality_scores = {}
            if self._quality_agent is not None:
                quality_scores = await self._run_quality_checks(
                    step.step_id, output_data
                )

            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now(timezone.utc)

            self._report_progress({
                "phase": "step_completed",
                "step_id": step.step_id,
                "duration_seconds": step.duration_seconds,
            })

            return StepResult(
                step_id=step.step_id,
                success=True,
                output_data=output_data,
                metrics={
                    "duration_seconds": step.duration_seconds or 0.0,
                },
                quality_scores=quality_scores,
            )

        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")

            step.status = StepStatus.FAILED
            step.end_time = datetime.now(timezone.utc)
            step.error_message = str(e)

            self._report_progress({
                "phase": "step_failed",
                "step_id": step.step_id,
                "error": str(e),
            })

            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(e),
            )

    async def checkpoint(
        self,
        step_id: str,
        state: Dict[str, Any],
    ) -> CheckpointMetadata:
        """
        Create a checkpoint at the current execution state.

        Args:
            step_id: Current step ID for checkpoint naming
            state: State data to checkpoint

        Returns:
            CheckpointMetadata for the created checkpoint
        """
        if self._current_execution_id is None:
            raise RuntimeError("No active execution to checkpoint")

        logger.info(f"Creating checkpoint at step {step_id}")

        # Add execution context to state
        checkpoint_state = {
            **state,
            "step_id": step_id,
            "execution_id": self._current_execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        metadata = self.checkpoint_manager.save(
            execution_id=self._current_execution_id,
            data=checkpoint_state,
            checkpoint_type=CheckpointType.AUTO,
        )

        self._report_progress({
            "phase": "checkpoint_created",
            "checkpoint_id": metadata.checkpoint_id,
            "step_id": step_id,
        })

        return metadata

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        """
        Resume execution from a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint to resume from

        Returns:
            Restored state dictionary
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_id}")

        checkpoint_data = self.checkpoint_manager.load(checkpoint_id)

        if checkpoint_data is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        self._report_progress({
            "phase": "checkpoint_restored",
            "checkpoint_id": checkpoint_id,
        })

        return checkpoint_data

    async def handle_step_failure(
        self,
        step: str,
        error: Exception,
    ) -> None:
        """
        Handle a step execution failure.

        Args:
            step: Failed step ID
            error: Exception that caused failure
        """
        logger.error(f"Handling failure for step {step}: {error}")

        # Update step state
        if step in self._step_states:
            self._step_states[step].status = StepStatus.FAILED
            self._step_states[step].error_message = str(error)

        # Create recovery checkpoint
        if self._current_execution_id:
            await self.checkpoint(
                step_id=f"{step}_failure",
                state={
                    "failed_step": step,
                    "error": str(error),
                    "step_states": {
                        k: v.to_dict() for k, v in self._step_states.items()
                    },
                },
            )

        self._report_progress({
            "phase": "step_failure_handled",
            "step_id": step,
            "error": str(error),
        })

    def cancel_execution(self) -> None:
        """Cancel the current pipeline execution."""
        self._cancelled = True
        self.status = AgentStatus.CANCELLED
        logger.info("Pipeline execution cancelled")

        self._report_progress({
            "phase": "cancelled",
            "execution_id": self._current_execution_id,
        })

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "execution_id": self._current_execution_id,
            "pipeline_id": self._current_pipeline.id if self._current_pipeline else None,
            "step_count": len(self._step_states),
            "completed_steps": sum(
                1 for s in self._step_states.values()
                if s.status == StepStatus.COMPLETED
            ),
            "failed_steps": sum(
                1 for s in self._step_states.values()
                if s.status == StepStatus.FAILED
            ),
        }

    async def _execute_algorithm(
        self,
        algorithm: Any,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute an algorithm with given inputs and parameters.

        This is a stub implementation - real implementation would:
        1. Load the algorithm module
        2. Instantiate the algorithm class
        3. Execute with inputs and parameters
        4. Return results
        """
        # Simulate algorithm execution
        await asyncio.sleep(0.01)  # Simulate processing time

        return {
            "algorithm_id": algorithm.id,
            "parameters_used": parameters,
            "output_type": "raster",
            "processed": True,
        }

    async def _run_quality_checks(
        self,
        step_id: str,
        output_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run quality checks on step output using quality agent.

        Args:
            step_id: Step identifier
            output_data: Output data to check

        Returns:
            Dictionary of quality scores
        """
        # Stub implementation - would call quality agent
        return {
            "completeness": 0.95,
            "consistency": 0.92,
            "accuracy": 0.88,
        }

    def _prepare_step_inputs(
        self,
        node: PipelineNode,
        pipeline_inputs: Dict[str, Any],
        step_results: Dict[str, StepResult],
        graph: PipelineGraph,
    ) -> Dict[str, Any]:
        """Prepare inputs for a step from pipeline inputs and previous results."""
        inputs = {}

        # Get inputs from incoming edges
        for edge in graph.get_incoming_edges(node.id):
            source_node = graph.get_node(edge.source_node)
            if source_node is None:
                continue

            if source_node.node_type == NodeType.INPUT:
                # Get from pipeline inputs
                input_name = source_node.name
                if input_name in pipeline_inputs:
                    inputs[edge.target_port] = pipeline_inputs[input_name]
            else:
                # Get from previous step results
                source_result = step_results.get(edge.source_node)
                if source_result and source_result.output_data:
                    inputs[edge.target_port] = source_result.output_data

        return inputs

    def _restore_step_results(
        self,
        results_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, StepResult]:
        """Restore step results from checkpoint data."""
        results = {}
        for step_id, data in results_dict.items():
            results[step_id] = StepResult(
                step_id=step_id,
                success=data.get("success", False),
                output_data=data.get("output_data"),
                metrics=data.get("metrics", {}),
                quality_scores=data.get("quality_scores", {}),
                warnings=data.get("warnings", []),
                error=data.get("error"),
            )
        return results
