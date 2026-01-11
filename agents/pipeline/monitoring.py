"""
Progress Tracking and Monitoring Module for Pipeline Agent.

Provides real-time progress updates, step timing and performance metrics,
resource usage monitoring, and execution timeline generation.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    DURATION = "duration"              # Time-based metrics
    THROUGHPUT = "throughput"          # Processing rate
    MEMORY = "memory"                  # Memory usage
    CPU = "cpu"                        # CPU utilization
    IO = "io"                          # I/O operations
    QUALITY = "quality"                # Quality scores
    ERROR = "error"                    # Error counts


class EventType(Enum):
    """Types of execution events."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    GROUP_START = "group_start"
    GROUP_END = "group_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    WARNING = "warning"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class ExecutionEvent:
    """
    An event during pipeline execution.

    Attributes:
        event_type: Type of event
        timestamp: When event occurred
        step_id: Related step ID (if applicable)
        group_id: Related group ID (if applicable)
        message: Event description
        data: Additional event data
    """
    event_type: EventType
    timestamp: datetime
    step_id: Optional[str] = None
    group_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step_id": self.step_id,
            "group_id": self.group_id,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class StepMetrics:
    """
    Metrics for a single pipeline step.

    Attributes:
        step_id: Step identifier
        start_time: When step started
        end_time: When step ended
        duration_seconds: Total duration
        queue_time_seconds: Time waiting in queue
        execution_time_seconds: Actual execution time
        memory_peak_mb: Peak memory usage
        cpu_percent: Average CPU utilization
        input_size_bytes: Total input data size
        output_size_bytes: Total output data size
        retry_count: Number of retries
        quality_scores: Quality assessment scores
    """
    step_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    queue_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    retry_count: int = 0
    quality_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "queue_time_seconds": self.queue_time_seconds,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_percent": self.cpu_percent,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "retry_count": self.retry_count,
            "quality_scores": self.quality_scores,
        }


@dataclass
class GroupMetrics:
    """
    Metrics for an execution group.

    Attributes:
        group_id: Group identifier
        step_count: Number of steps in group
        parallel_steps: Steps that ran in parallel
        start_time: When group started
        end_time: When group ended
        duration_seconds: Total group duration
        parallelism_factor: Effective parallelism achieved
        memory_peak_mb: Peak memory during group
    """
    group_id: str
    step_count: int = 0
    parallel_steps: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    parallelism_factor: float = 1.0
    memory_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_id": self.group_id,
            "step_count": self.step_count,
            "parallel_steps": self.parallel_steps,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "parallelism_factor": self.parallelism_factor,
            "memory_peak_mb": self.memory_peak_mb,
        }


@dataclass
class PipelineMetrics:
    """
    Overall metrics for pipeline execution.

    Attributes:
        pipeline_id: Pipeline identifier
        execution_id: Execution run identifier
        start_time: Execution start
        end_time: Execution end
        total_duration_seconds: Total execution time
        step_metrics: Metrics for each step
        group_metrics: Metrics for each group
        total_steps: Total steps executed
        completed_steps: Successfully completed steps
        failed_steps: Steps that failed
        skipped_steps: Steps that were skipped
        total_retries: Total retry attempts
        checkpoints_created: Number of checkpoints
        memory_peak_mb: Peak memory usage
        average_parallelism: Average parallelism achieved
    """
    pipeline_id: str
    execution_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    step_metrics: Dict[str, StepMetrics] = field(default_factory=dict)
    group_metrics: Dict[str, GroupMetrics] = field(default_factory=dict)
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    total_retries: int = 0
    checkpoints_created: int = 0
    memory_peak_mb: float = 0.0
    average_parallelism: float = 1.0

    @property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps

    @property
    def is_complete(self) -> bool:
        """Whether execution is complete."""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "success_rate": self.success_rate,
            "total_retries": self.total_retries,
            "checkpoints_created": self.checkpoints_created,
            "memory_peak_mb": self.memory_peak_mb,
            "average_parallelism": self.average_parallelism,
            "step_metrics": {k: v.to_dict() for k, v in self.step_metrics.items()},
            "group_metrics": {k: v.to_dict() for k, v in self.group_metrics.items()},
        }


@dataclass
class ProgressSnapshot:
    """
    A snapshot of current execution progress.

    Attributes:
        timestamp: When snapshot was taken
        progress_percent: Overall progress percentage
        current_phase: Current execution phase
        active_steps: Currently running steps
        completed_steps: Completed step count
        failed_steps: Failed step count
        pending_steps: Remaining step count
        elapsed_seconds: Time elapsed
        estimated_remaining_seconds: Estimated time remaining
        throughput_steps_per_minute: Processing rate
    """
    timestamp: datetime
    progress_percent: float = 0.0
    current_phase: str = ""
    active_steps: List[str] = field(default_factory=list)
    completed_steps: int = 0
    failed_steps: int = 0
    pending_steps: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    throughput_steps_per_minute: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "progress_percent": self.progress_percent,
            "current_phase": self.current_phase,
            "active_steps": self.active_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "pending_steps": self.pending_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "throughput_steps_per_minute": self.throughput_steps_per_minute,
        }


class ProgressTracker:
    """
    Tracks execution progress and collects metrics.

    Features:
    - Real-time progress updates
    - Step and group timing
    - Resource usage monitoring
    - Throughput calculation
    - ETA estimation
    - Event timeline generation

    Usage:
        tracker = ProgressTracker(pipeline_id, execution_id)
        tracker.start_tracking()

        tracker.record_step_start("step_1")
        # ... step execution ...
        tracker.record_step_end("step_1", success=True)

        metrics = tracker.get_metrics()
        timeline = tracker.get_timeline()
    """

    def __init__(
        self,
        pipeline_id: str,
        execution_id: str,
        total_steps: int = 0,
        progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None,
        update_interval_seconds: float = 1.0,
    ):
        """
        Initialize the progress tracker.

        Args:
            pipeline_id: Pipeline identifier
            execution_id: Execution run identifier
            total_steps: Total number of steps (for progress calculation)
            progress_callback: Callback for progress updates
            update_interval_seconds: Interval between progress updates
        """
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.total_steps = total_steps
        self.progress_callback = progress_callback
        self.update_interval = update_interval_seconds

        # Metrics
        self._metrics = PipelineMetrics(
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            total_steps=total_steps,
        )

        # Event timeline
        self._events: List[ExecutionEvent] = []

        # Current state
        self._active_steps: Set[str] = set()
        self._step_queue_times: Dict[str, datetime] = {}
        self._current_group: Optional[str] = None

        # Threading
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(f"ProgressTracker initialized for execution {execution_id}")

    def start_tracking(self) -> None:
        """Start progress tracking and background updates."""
        self._metrics.start_time = datetime.now(timezone.utc)

        self._record_event(
            EventType.PIPELINE_START,
            message="Pipeline execution started",
        )

        # Start background update thread
        if self.progress_callback:
            self._stop_event.clear()
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
            )
            self._update_thread.start()

    def stop_tracking(self, success: bool = True) -> None:
        """Stop progress tracking."""
        self._metrics.end_time = datetime.now(timezone.utc)
        if self._metrics.start_time:
            self._metrics.total_duration_seconds = (
                self._metrics.end_time - self._metrics.start_time
            ).total_seconds()

        self._record_event(
            EventType.PIPELINE_END,
            message=f"Pipeline execution {'completed' if success else 'failed'}",
            data={"success": success},
        )

        # Calculate final metrics
        self._calculate_final_metrics()

        # Stop update thread
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)

    def record_step_queued(self, step_id: str) -> None:
        """Record that a step has been queued for execution."""
        with self._lock:
            self._step_queue_times[step_id] = datetime.now(timezone.utc)

    def record_step_start(
        self,
        step_id: str,
        group_id: Optional[str] = None,
    ) -> None:
        """Record step execution start."""
        now = datetime.now(timezone.utc)

        with self._lock:
            self._active_steps.add(step_id)

            # Initialize step metrics
            metrics = StepMetrics(step_id=step_id, start_time=now)

            # Calculate queue time if step was queued
            if step_id in self._step_queue_times:
                queue_time = (now - self._step_queue_times[step_id]).total_seconds()
                metrics.queue_time_seconds = queue_time
                del self._step_queue_times[step_id]

            self._metrics.step_metrics[step_id] = metrics

        self._record_event(
            EventType.STEP_START,
            step_id=step_id,
            group_id=group_id,
            message=f"Step {step_id} started",
        )

    def record_step_end(
        self,
        step_id: str,
        success: bool = True,
        output_size_bytes: int = 0,
        quality_scores: Optional[Dict[str, float]] = None,
        retry_count: int = 0,
    ) -> None:
        """Record step execution end."""
        now = datetime.now(timezone.utc)

        with self._lock:
            self._active_steps.discard(step_id)

            if step_id in self._metrics.step_metrics:
                metrics = self._metrics.step_metrics[step_id]
                metrics.end_time = now
                if metrics.start_time:
                    metrics.duration_seconds = (now - metrics.start_time).total_seconds()
                    metrics.execution_time_seconds = (
                        metrics.duration_seconds - metrics.queue_time_seconds
                    )
                metrics.output_size_bytes = output_size_bytes
                metrics.retry_count = retry_count
                if quality_scores:
                    metrics.quality_scores = quality_scores

            # Update pipeline metrics
            if success:
                self._metrics.completed_steps += 1
            else:
                self._metrics.failed_steps += 1

            self._metrics.total_retries += retry_count

        self._record_event(
            EventType.STEP_END,
            step_id=step_id,
            message=f"Step {step_id} {'completed' if success else 'failed'}",
            data={"success": success, "retry_count": retry_count},
        )

    def record_step_skipped(self, step_id: str, reason: str = "") -> None:
        """Record that a step was skipped."""
        with self._lock:
            self._metrics.skipped_steps += 1

        self._record_event(
            EventType.STEP_END,
            step_id=step_id,
            message=f"Step {step_id} skipped: {reason}",
            data={"skipped": True, "reason": reason},
        )

    def record_group_start(self, group_id: str, step_count: int) -> None:
        """Record execution group start."""
        now = datetime.now(timezone.utc)

        with self._lock:
            self._current_group = group_id
            self._metrics.group_metrics[group_id] = GroupMetrics(
                group_id=group_id,
                step_count=step_count,
                start_time=now,
            )

        self._record_event(
            EventType.GROUP_START,
            group_id=group_id,
            message=f"Group {group_id} started with {step_count} steps",
            data={"step_count": step_count},
        )

    def record_group_end(
        self,
        group_id: str,
        parallel_steps: int = 0,
    ) -> None:
        """Record execution group end."""
        now = datetime.now(timezone.utc)

        with self._lock:
            if group_id in self._metrics.group_metrics:
                metrics = self._metrics.group_metrics[group_id]
                metrics.end_time = now
                if metrics.start_time:
                    metrics.duration_seconds = (now - metrics.start_time).total_seconds()
                metrics.parallel_steps = parallel_steps
                if metrics.step_count > 0:
                    metrics.parallelism_factor = parallel_steps / metrics.step_count

            self._current_group = None

        self._record_event(
            EventType.GROUP_END,
            group_id=group_id,
            message=f"Group {group_id} completed",
            data={"parallel_steps": parallel_steps},
        )

    def record_checkpoint(self, checkpoint_id: str) -> None:
        """Record checkpoint creation."""
        with self._lock:
            self._metrics.checkpoints_created += 1

        self._record_event(
            EventType.CHECKPOINT,
            message=f"Checkpoint {checkpoint_id} created",
            data={"checkpoint_id": checkpoint_id},
        )

    def record_error(
        self,
        message: str,
        step_id: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error event."""
        self._record_event(
            EventType.ERROR,
            step_id=step_id,
            message=message,
            data=error_data or {},
        )

    def record_warning(
        self,
        message: str,
        step_id: Optional[str] = None,
    ) -> None:
        """Record a warning event."""
        self._record_event(
            EventType.WARNING,
            step_id=step_id,
            message=message,
        )

    def record_resource_usage(
        self,
        memory_mb: float,
        cpu_percent: float = 0.0,
        step_id: Optional[str] = None,
    ) -> None:
        """Record resource usage."""
        with self._lock:
            # Update peak memory
            if memory_mb > self._metrics.memory_peak_mb:
                self._metrics.memory_peak_mb = memory_mb

            # Update step metrics if applicable
            if step_id and step_id in self._metrics.step_metrics:
                metrics = self._metrics.step_metrics[step_id]
                if memory_mb > metrics.memory_peak_mb:
                    metrics.memory_peak_mb = memory_mb
                metrics.cpu_percent = cpu_percent

    def get_progress_snapshot(self) -> ProgressSnapshot:
        """Get current progress snapshot."""
        now = datetime.now(timezone.utc)

        with self._lock:
            completed = self._metrics.completed_steps
            failed = self._metrics.failed_steps
            total = self._metrics.total_steps

            # Calculate progress
            if total > 0:
                progress = (completed + failed) / total * 100
            else:
                progress = 0.0

            # Calculate elapsed time
            elapsed = 0.0
            if self._metrics.start_time:
                elapsed = (now - self._metrics.start_time).total_seconds()

            # Calculate throughput and ETA
            throughput = 0.0
            eta = None
            if elapsed > 0 and completed > 0:
                throughput = completed / elapsed * 60  # steps per minute
                remaining = total - completed - failed
                if throughput > 0:
                    eta = remaining / throughput * 60  # seconds

            return ProgressSnapshot(
                timestamp=now,
                progress_percent=progress,
                current_phase=self._current_group or "executing",
                active_steps=list(self._active_steps),
                completed_steps=completed,
                failed_steps=failed,
                pending_steps=total - completed - failed,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=eta,
                throughput_steps_per_minute=throughput,
            )

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        with self._lock:
            return self._metrics

    def get_timeline(self) -> List[ExecutionEvent]:
        """Get execution event timeline."""
        with self._lock:
            return list(self._events)

    def get_step_metrics(self, step_id: str) -> Optional[StepMetrics]:
        """Get metrics for a specific step."""
        with self._lock:
            return self._metrics.step_metrics.get(step_id)

    def _record_event(
        self,
        event_type: EventType,
        step_id: Optional[str] = None,
        group_id: Optional[str] = None,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an execution event."""
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            step_id=step_id,
            group_id=group_id,
            message=message,
            data=data or {},
        )

        with self._lock:
            self._events.append(event)

        logger.debug(f"Event: {event_type.value} - {message}")

    def _update_loop(self) -> None:
        """Background loop for periodic progress updates."""
        while not self._stop_event.wait(self.update_interval):
            try:
                snapshot = self.get_progress_snapshot()
                if self.progress_callback:
                    self.progress_callback(snapshot)
            except Exception as e:
                logger.warning(f"Progress update failed: {e}")

    def _calculate_final_metrics(self) -> None:
        """Calculate final aggregate metrics."""
        with self._lock:
            # Calculate average parallelism
            total_parallelism = 0.0
            group_count = 0

            for metrics in self._metrics.group_metrics.values():
                if metrics.parallelism_factor > 0:
                    total_parallelism += metrics.parallelism_factor
                    group_count += 1

            if group_count > 0:
                self._metrics.average_parallelism = total_parallelism / group_count

    def generate_summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        metrics = self.get_metrics()
        snapshot = self.get_progress_snapshot()

        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "status": "complete" if metrics.is_complete else "in_progress",
            "progress_percent": snapshot.progress_percent,
            "duration_seconds": metrics.total_duration_seconds,
            "steps": {
                "total": metrics.total_steps,
                "completed": metrics.completed_steps,
                "failed": metrics.failed_steps,
                "skipped": metrics.skipped_steps,
            },
            "performance": {
                "success_rate": metrics.success_rate,
                "average_parallelism": metrics.average_parallelism,
                "throughput_steps_per_minute": snapshot.throughput_steps_per_minute,
                "memory_peak_mb": metrics.memory_peak_mb,
            },
            "checkpoints": metrics.checkpoints_created,
            "retries": metrics.total_retries,
        }
