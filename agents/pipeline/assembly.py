"""
Pipeline Assembly Module for Pipeline Agent.

Provides pipeline building logic, algorithm selection based on event type
and data availability, execution DAG construction, and validation before
execution.

Integrates with core/analysis/assembly/ for underlying functionality.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    Port,
    QCGate,
    NodeType,
    DataType,
    EdgeType,
    NodeStatus,
    CycleDetectedError,
)
from core.analysis.assembly.assembler import (
    PipelineAssembler,
    DynamicAssembler,
    PipelineSpec,
    InputSpec,
    OutputSpec,
    StepSpec,
    AssemblyResult,
    AssemblyError,
    TemporalRole,
)
from core.analysis.assembly.validator import (
    PipelineValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
    ResourceEstimate,
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType as AlgorithmDataType,
    get_global_registry,
    load_default_algorithms,
)

logger = logging.getLogger(__name__)


class BuildStrategy(Enum):
    """Strategy for building pipelines."""
    SINGLE_ALGORITHM = "single"        # Use best single algorithm
    ENSEMBLE = "ensemble"              # Use multiple algorithms
    CASCADED = "cascaded"              # Chain algorithms in sequence
    ADAPTIVE = "adaptive"              # Select based on data quality


class SelectionCriteria(Enum):
    """Criteria for algorithm selection."""
    ACCURACY = "accuracy"              # Prioritize accuracy
    SPEED = "speed"                    # Prioritize execution speed
    RESOURCE_EFFICIENCY = "resource"   # Prioritize low resource usage
    COVERAGE = "coverage"              # Prioritize data coverage
    DETERMINISTIC = "deterministic"    # Only deterministic algorithms


@dataclass
class BuildConfig:
    """
    Configuration for pipeline building.

    Attributes:
        strategy: Pipeline building strategy
        selection_criteria: Primary algorithm selection criteria
        max_algorithms: Maximum algorithms to include
        require_deterministic: Only use deterministic algorithms
        max_memory_mb: Maximum memory budget
        gpu_available: Whether GPU is available
        enable_qc_gates: Insert QC gates between steps
        qc_gate_checks: QC checks to run at gates
        optimize_execution: Apply execution optimization
        optimization_strategy: Strategy for optimization
    """
    strategy: BuildStrategy = BuildStrategy.SINGLE_ALGORITHM
    selection_criteria: SelectionCriteria = SelectionCriteria.ACCURACY
    max_algorithms: int = 3
    require_deterministic: bool = False
    max_memory_mb: Optional[int] = None
    gpu_available: bool = False
    enable_qc_gates: bool = True
    qc_gate_checks: List[str] = field(default_factory=lambda: ["validity", "range"])
    optimize_execution: bool = True
    optimization_strategy: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy": self.strategy.value,
            "selection_criteria": self.selection_criteria.value,
            "max_algorithms": self.max_algorithms,
            "require_deterministic": self.require_deterministic,
            "max_memory_mb": self.max_memory_mb,
            "gpu_available": self.gpu_available,
            "enable_qc_gates": self.enable_qc_gates,
            "qc_gate_checks": self.qc_gate_checks,
            "optimize_execution": self.optimize_execution,
            "optimization_strategy": self.optimization_strategy,
        }


@dataclass
class AlgorithmScore:
    """
    Score for algorithm selection.

    Attributes:
        algorithm: Algorithm metadata
        total_score: Combined score
        accuracy_score: Score based on validation metrics
        speed_score: Score based on resource requirements
        coverage_score: Score based on data coverage
        match_score: Score based on event type match
    """
    algorithm: AlgorithmMetadata
    total_score: float = 0.0
    accuracy_score: float = 0.0
    speed_score: float = 0.0
    coverage_score: float = 0.0
    match_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm_id": self.algorithm.id,
            "total_score": self.total_score,
            "accuracy_score": self.accuracy_score,
            "speed_score": self.speed_score,
            "coverage_score": self.coverage_score,
            "match_score": self.match_score,
        }


@dataclass
class BuildResult:
    """
    Result of pipeline building.

    Attributes:
        pipeline: Built pipeline graph
        spec: Generated pipeline specification
        selected_algorithms: Algorithms selected with scores
        build_time: Time taken to build
        validation: Validation result
        warnings: Build warnings
    """
    pipeline: PipelineGraph
    spec: PipelineSpec
    selected_algorithms: List[AlgorithmScore]
    build_time: float = 0.0
    validation: Optional[ValidationResult] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether build was successful."""
        if self.validation is None:
            return self.pipeline is not None
        return self.validation.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "pipeline_id": self.pipeline.id if self.pipeline else None,
            "selected_algorithms": [a.to_dict() for a in self.selected_algorithms],
            "build_time": self.build_time,
            "validation_valid": self.validation.is_valid if self.validation else None,
            "warnings": self.warnings,
        }


class PipelineBuilder:
    """
    Builds analysis pipelines based on event type and available data.

    Features:
    - Automatic algorithm selection based on multiple criteria
    - Support for different building strategies (single, ensemble, cascaded)
    - QC gate insertion between steps
    - Execution DAG construction and validation
    - Integration with algorithm registry

    Usage:
        builder = PipelineBuilder(registry)
        result = builder.build(
            event_type="flood.coastal",
            available_data={"sar": DataType.RASTER, "dem": DataType.RASTER},
            config=BuildConfig(strategy=BuildStrategy.ENSEMBLE)
        )
        if result.success:
            pipeline = result.pipeline
    """

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        validator: Optional[PipelineValidator] = None,
    ):
        """
        Initialize the pipeline builder.

        Args:
            registry: Algorithm registry (uses global if None)
            validator: Pipeline validator (creates new if None)
        """
        self.registry = registry
        if self.registry is None:
            self.registry = get_global_registry()
            if not self.registry.algorithms:
                load_default_algorithms()

        self.validator = validator or PipelineValidator(registry=self.registry)
        self._assembler = PipelineAssembler(registry=self.registry, strict_mode=False)

        logger.info(f"PipelineBuilder initialized with {len(self.registry.algorithms)} algorithms")

    def build(
        self,
        event_type: str,
        available_data: Dict[str, DataType],
        config: Optional[BuildConfig] = None,
        pipeline_id: Optional[str] = None,
    ) -> BuildResult:
        """
        Build a pipeline for the given event type and data.

        Args:
            event_type: Event type (e.g., "flood.coastal")
            available_data: Available data sources and types
            config: Build configuration
            pipeline_id: Optional pipeline ID (auto-generated if None)

        Returns:
            BuildResult with pipeline and metadata
        """
        start_time = datetime.now(timezone.utc)
        config = config or BuildConfig()
        warnings: List[str] = []

        logger.info(f"Building pipeline for {event_type} with strategy {config.strategy.value}")

        # Step 1: Find candidate algorithms
        candidates = self._find_candidates(event_type, available_data, config)

        if not candidates:
            warnings.append(
                f"No algorithms found for event type '{event_type}' "
                f"with available data: {list(available_data.keys())}"
            )
            return BuildResult(
                pipeline=None,
                spec=None,
                selected_algorithms=[],
                warnings=warnings,
            )

        # Step 2: Score and select algorithms
        scored_algorithms = self._score_algorithms(candidates, available_data, config)
        selected = self._select_algorithms(scored_algorithms, config)

        if not selected:
            warnings.append("No algorithms selected after scoring")
            return BuildResult(
                pipeline=None,
                spec=None,
                selected_algorithms=[],
                warnings=warnings,
            )

        # Step 3: Build pipeline specification
        spec = self._build_spec(
            event_type=event_type,
            available_data=available_data,
            selected_algorithms=selected,
            config=config,
            pipeline_id=pipeline_id,
        )

        # Step 4: Assemble pipeline graph
        assembly_result = self._assembler.assemble(spec)

        if not assembly_result.success:
            warnings.extend(assembly_result.warnings)
            warnings.append("Pipeline assembly failed")
            return BuildResult(
                pipeline=None,
                spec=spec,
                selected_algorithms=selected,
                warnings=warnings,
            )

        warnings.extend(assembly_result.warnings)
        pipeline = assembly_result.graph

        # Step 5: Validate pipeline
        validation = self.validator.validate(pipeline)

        build_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = BuildResult(
            pipeline=pipeline,
            spec=spec,
            selected_algorithms=selected,
            build_time=build_time,
            validation=validation,
            warnings=warnings,
        )

        logger.info(
            f"Built pipeline {pipeline.id} with {len(selected)} algorithms "
            f"in {build_time:.3f}s (valid: {validation.is_valid})"
        )

        return result

    def build_from_spec(
        self,
        spec: Union[PipelineSpec, Dict[str, Any]],
        validate: bool = True,
    ) -> BuildResult:
        """
        Build a pipeline from an explicit specification.

        Args:
            spec: Pipeline specification
            validate: Whether to validate the built pipeline

        Returns:
            BuildResult with pipeline
        """
        start_time = datetime.now(timezone.utc)
        warnings: List[str] = []

        if isinstance(spec, dict):
            spec = PipelineSpec.from_dict(spec)

        assembly_result = self._assembler.assemble(spec)
        warnings.extend(assembly_result.warnings)

        if not assembly_result.success:
            return BuildResult(
                pipeline=None,
                spec=spec,
                selected_algorithms=[],
                warnings=warnings,
            )

        validation = None
        if validate:
            validation = self.validator.validate(assembly_result.graph)

        build_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return BuildResult(
            pipeline=assembly_result.graph,
            spec=spec,
            selected_algorithms=[],
            build_time=build_time,
            validation=validation,
            warnings=warnings,
        )

    def _find_candidates(
        self,
        event_type: str,
        available_data: Dict[str, DataType],
        config: BuildConfig,
    ) -> List[AlgorithmMetadata]:
        """Find candidate algorithms based on event type and data."""
        # Convert to algorithm data types
        available_types = []
        for dtype in available_data.values():
            try:
                algo_type = AlgorithmDataType(dtype.value)
                available_types.append(algo_type)
            except (ValueError, KeyError):
                # Try to map common types
                if dtype == DataType.RASTER:
                    available_types.append(AlgorithmDataType.OPTICAL)

        # Search with all filters
        candidates = self.registry.search_by_requirements(
            event_type=event_type,
            available_data_types=available_types,
            max_memory_mb=config.max_memory_mb,
            gpu_available=config.gpu_available,
            require_deterministic=config.require_deterministic,
        )

        logger.debug(f"Found {len(candidates)} candidate algorithms for {event_type}")
        return candidates

    def _score_algorithms(
        self,
        candidates: List[AlgorithmMetadata],
        available_data: Dict[str, DataType],
        config: BuildConfig,
    ) -> List[AlgorithmScore]:
        """Score candidate algorithms based on selection criteria."""
        scored = []

        for algo in candidates:
            score = AlgorithmScore(algorithm=algo)

            # Accuracy score from validation metrics
            if algo.validation:
                if algo.validation.accuracy_median is not None:
                    score.accuracy_score = algo.validation.accuracy_median
                elif algo.validation.f1_score is not None:
                    score.accuracy_score = algo.validation.f1_score
                else:
                    score.accuracy_score = 0.5  # Default

                # Bonus for more validated regions
                if algo.validation.validated_regions:
                    score.accuracy_score *= (1 + 0.1 * min(len(algo.validation.validated_regions), 5))
            else:
                score.accuracy_score = 0.3  # Low score for unvalidated

            # Speed score from resource requirements
            if algo.resources:
                if algo.resources.max_runtime_minutes:
                    # Faster = higher score (normalize to 0-1)
                    runtime = algo.resources.max_runtime_minutes
                    score.speed_score = max(0.0, 1.0 - (runtime / 60.0))
                else:
                    score.speed_score = 0.5

                if algo.resources.memory_mb:
                    # Less memory = higher score
                    memory = algo.resources.memory_mb
                    score.speed_score *= max(0.1, 1.0 - (memory / 16000.0))
            else:
                score.speed_score = 0.5

            # Coverage score based on required vs available data
            required_count = len(algo.required_data_types)
            optional_count = len(algo.optional_data_types)
            available_count = len(available_data)

            if required_count > 0:
                # Higher score if we have more than required
                score.coverage_score = min(1.0, available_count / required_count)
                # Bonus if we can use optional inputs
                if optional_count > 0 and available_count > required_count:
                    score.coverage_score *= 1.1
            else:
                score.coverage_score = 0.5

            # Match score based on event type specificity
            for pattern in algo.event_types:
                if pattern == config.selection_criteria.value:
                    score.match_score = 1.0
                    break
                elif ".*" not in pattern:
                    score.match_score = 0.9
                else:
                    score.match_score = 0.7

            # Calculate total score based on criteria weights
            weights = self._get_criteria_weights(config.selection_criteria)
            score.total_score = (
                weights["accuracy"] * score.accuracy_score +
                weights["speed"] * score.speed_score +
                weights["coverage"] * score.coverage_score +
                weights["match"] * score.match_score
            )

            # Category bonus/penalty
            if algo.category == AlgorithmCategory.BASELINE:
                score.total_score *= 0.9  # Slight penalty for basic
            elif algo.category == AlgorithmCategory.ADVANCED:
                score.total_score *= 1.0  # Standard
            elif algo.category == AlgorithmCategory.EXPERIMENTAL:
                score.total_score *= 0.7  # Penalty for experimental

            scored.append(score)

        # Sort by total score (highest first)
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    def _get_criteria_weights(self, criteria: SelectionCriteria) -> Dict[str, float]:
        """Get scoring weights based on selection criteria."""
        if criteria == SelectionCriteria.ACCURACY:
            return {"accuracy": 0.5, "speed": 0.1, "coverage": 0.2, "match": 0.2}
        elif criteria == SelectionCriteria.SPEED:
            return {"accuracy": 0.2, "speed": 0.5, "coverage": 0.1, "match": 0.2}
        elif criteria == SelectionCriteria.RESOURCE_EFFICIENCY:
            return {"accuracy": 0.2, "speed": 0.4, "coverage": 0.2, "match": 0.2}
        elif criteria == SelectionCriteria.COVERAGE:
            return {"accuracy": 0.2, "speed": 0.1, "coverage": 0.5, "match": 0.2}
        else:  # DETERMINISTIC or default
            return {"accuracy": 0.3, "speed": 0.2, "coverage": 0.2, "match": 0.3}

    def _select_algorithms(
        self,
        scored: List[AlgorithmScore],
        config: BuildConfig,
    ) -> List[AlgorithmScore]:
        """Select algorithms based on strategy and configuration."""
        if not scored:
            return []

        if config.strategy == BuildStrategy.SINGLE_ALGORITHM:
            return [scored[0]]

        elif config.strategy == BuildStrategy.ENSEMBLE:
            # Select top N diverse algorithms
            selected = []
            seen_categories = set()

            for score in scored:
                if len(selected) >= config.max_algorithms:
                    break

                # Try to get diverse categories
                if score.algorithm.category not in seen_categories or len(selected) < 2:
                    selected.append(score)
                    seen_categories.add(score.algorithm.category)

            return selected

        elif config.strategy == BuildStrategy.CASCADED:
            # Select algorithms that can be chained
            selected = [scored[0]]

            for score in scored[1:]:
                if len(selected) >= config.max_algorithms:
                    break

                # Check if outputs of previous can feed inputs of this
                # (simplified: just take top algorithms)
                selected.append(score)

            return selected

        elif config.strategy == BuildStrategy.ADAPTIVE:
            # Return all for adaptive selection at runtime
            return scored[:config.max_algorithms]

        return [scored[0]]

    def _build_spec(
        self,
        event_type: str,
        available_data: Dict[str, DataType],
        selected_algorithms: List[AlgorithmScore],
        config: BuildConfig,
        pipeline_id: Optional[str] = None,
    ) -> PipelineSpec:
        """Build pipeline specification from selected algorithms."""
        # Generate pipeline ID
        if pipeline_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            pipeline_id = f"pipeline_{event_type.replace('.', '_')}_{timestamp}"

        # Create inputs
        inputs = []
        for name, dtype in available_data.items():
            inputs.append(InputSpec(
                name=name,
                data_type=dtype,
                required=True,
            ))

        # Create steps for each algorithm
        steps = []
        previous_step_id = None

        for i, scored in enumerate(selected_algorithms):
            algo = scored.algorithm
            step_id = f"step_{i}_{algo.id.replace('.', '_')}"

            # Determine inputs
            if i == 0:
                # First step uses pipeline inputs
                step_inputs = list(available_data.keys())
            else:
                # Subsequent steps use previous output
                step_inputs = [previous_step_id]

            # Create QC gate if enabled
            qc_gate = None
            if config.enable_qc_gates:
                qc_gate = QCGate(
                    enabled=True,
                    checks=config.qc_gate_checks,
                    on_fail="warn",
                )

            step = StepSpec(
                id=step_id,
                processor=algo.id,
                inputs=step_inputs,
                parameters=algo.default_parameters.copy(),
                outputs={"output": "raster"},
                qc_gate=qc_gate,
            )
            steps.append(step)
            previous_step_id = step_id

        # Create outputs
        outputs = []
        if steps:
            # Output from last step
            outputs.append(OutputSpec(
                name="result",
                data_type=DataType.RASTER,
                format="cog",
                source_step=steps[-1].id,
            ))

            # If ensemble, also output from each algorithm
            if config.strategy == BuildStrategy.ENSEMBLE and len(steps) > 1:
                for i, step in enumerate(steps):
                    outputs.append(OutputSpec(
                        name=f"result_{i}",
                        data_type=DataType.RASTER,
                        format="cog",
                        source_step=step.id,
                    ))

        return PipelineSpec(
            id=pipeline_id,
            name=f"Pipeline for {event_type}",
            version="1.0.0",
            description=f"Auto-generated pipeline for {event_type}",
            applicable_classes=[event_type],
            inputs=inputs,
            steps=steps,
            outputs=outputs,
            metadata={
                "build_strategy": config.strategy.value,
                "selection_criteria": config.selection_criteria.value,
                "algorithms": [s.algorithm.id for s in selected_algorithms],
            },
        )

    def validate_pipeline(
        self,
        pipeline: PipelineGraph,
        available_resources: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a pipeline before execution.

        Args:
            pipeline: Pipeline to validate
            available_resources: Available compute resources

        Returns:
            ValidationResult with issues and resource estimates
        """
        return self.validator.validate(pipeline, available_resources)

    def get_algorithm_options(
        self,
        event_type: str,
        available_data: Dict[str, DataType],
    ) -> List[AlgorithmScore]:
        """
        Get scored algorithm options for given event type and data.

        Useful for presenting options to users before building.

        Args:
            event_type: Event type
            available_data: Available data sources

        Returns:
            List of scored algorithms
        """
        config = BuildConfig()
        candidates = self._find_candidates(event_type, available_data, config)
        return self._score_algorithms(candidates, available_data, config)
