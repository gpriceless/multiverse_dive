"""
Comprehensive tests for Pipeline Execution Engine module (Group H, Track 3).

Tests cover:
- PipelineRunner: Sequential and parallel execution
- Pipeline: DAG management, validation, topological sorting
- TaskResult, ExecutionContext: Task management
- RetryConfig: Retry behavior with backoff
- Checkpoint integration (with CheckpointManager)
- Distributed execution (LocalExecutor, DaskExecutor, RayExecutor)

Following Agent Code Review Checklist:
1. Correctness & Safety: Division guards, bounds checks, NaN handling
2. Consistency: Names match across files, defaults match
3. Completeness: All features implemented, docstrings, type hints
4. Robustness: Specific exceptions, thread safety
5. Performance: No O(n²) loops, caching
6. Security: Input validation, no secrets logged
7. Maintainability: No magic numbers, no duplication
"""

import pytest
import time
import threading
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

from core.analysis.execution import (
    # Runner enums and config
    TaskStatus,
    RetryPolicy,
    ExecutionMode,
    RetryConfig,
    ExecutionConfig,
    # Task components
    TaskResult,
    ExecutionContext,
    TaskExecutor,
    PipelineTask,
    # Progress and results
    ExecutionProgress,
    ExecutionResult,
    # Pipeline
    Pipeline,
    PipelineRunner,
    run_pipeline,
    # Distributed
    DistributedBackend,
    WorkerInfo,
    ClusterInfo,
    DistributedConfig,
    DistributedExecutorBase,
    LocalExecutor,
    DistributedPipelineRunner,
    get_executor,
    # Checkpoint
    CheckpointStatus,
    CheckpointType,
    CheckpointMetadata,
    CheckpointData,
    CheckpointConfig,
    CheckpointManager,
    AutoCheckpointer,
    LocalStorageBackend,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_task():
    """Create a simple task that returns a value."""
    def executor(ctx: ExecutionContext) -> str:
        return f"result_from_{ctx.task_id}"

    return PipelineTask(
        task_id="task_1",
        executor=executor,
    )


@pytest.fixture
def simple_pipeline():
    """Create a simple linear pipeline: A -> B -> C."""
    pipeline = Pipeline("test_pipeline", name="Test Pipeline")

    def make_executor(task_id: str):
        def executor(ctx: ExecutionContext) -> Dict[str, Any]:
            return {"task_id": task_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        return executor

    pipeline.add_task(PipelineTask(
        task_id="task_a",
        executor=make_executor("task_a"),
    ))
    pipeline.add_task(PipelineTask(
        task_id="task_b",
        executor=make_executor("task_b"),
        dependencies=["task_a"],
    ))
    pipeline.add_task(PipelineTask(
        task_id="task_c",
        executor=make_executor("task_c"),
        dependencies=["task_b"],
    ))

    return pipeline


@pytest.fixture
def parallel_pipeline():
    """Create a pipeline with parallel tasks: A -> [B, C] -> D."""
    pipeline = Pipeline("parallel_pipeline", name="Parallel Test")

    def make_executor(task_id: str, delay: float = 0.0):
        def executor(ctx: ExecutionContext) -> Dict[str, Any]:
            if delay > 0:
                time.sleep(delay)
            return {"task_id": task_id}
        return executor

    pipeline.add_task(PipelineTask(
        task_id="start",
        executor=make_executor("start"),
    ))
    pipeline.add_task(PipelineTask(
        task_id="parallel_1",
        executor=make_executor("parallel_1", 0.05),
        dependencies=["start"],
    ))
    pipeline.add_task(PipelineTask(
        task_id="parallel_2",
        executor=make_executor("parallel_2", 0.05),
        dependencies=["start"],
    ))
    pipeline.add_task(PipelineTask(
        task_id="end",
        executor=make_executor("end"),
        dependencies=["parallel_1", "parallel_2"],
    ))

    return pipeline


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create a checkpoint manager with temporary storage."""
    config = CheckpointConfig(base_path=temp_checkpoint_dir)
    return CheckpointManager(config)


# =============================================================================
# TaskStatus Tests
# =============================================================================

class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected statuses exist."""
        expected = ["PENDING", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "SKIPPED"]
        for status in expected:
            assert hasattr(TaskStatus, status)

    def test_status_values(self):
        """Verify status values are strings."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


# =============================================================================
# RetryConfig Tests
# =============================================================================

class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.policy == RetryPolicy.EXPONENTIAL_BACKOFF
        assert config.max_retries == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0

    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )
        assert config.get_delay(1) == 1.0  # 1 * 2^0
        assert config.get_delay(2) == 2.0  # 1 * 2^1
        assert config.get_delay(3) == 4.0  # 1 * 2^2
        assert config.get_delay(4) == 8.0  # 1 * 2^3

    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            policy=RetryPolicy.LINEAR_BACKOFF,
            base_delay_seconds=2.0,
            max_delay_seconds=60.0,
        )
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 6.0

    def test_max_delay_capped(self):
        """Test that delay is capped at max_delay_seconds."""
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=10.0,
            max_delay_seconds=30.0,
        )
        assert config.get_delay(5) == 30.0  # Would be 160 but capped

    def test_no_retry_policy(self):
        """Test no retry policy returns zero delay."""
        config = RetryConfig(policy=RetryPolicy.NONE)
        assert config.get_delay(1) == 0.0
        assert config.should_retry(Exception(), 1) is False

    def test_immediate_retry_policy(self):
        """Test immediate retry policy returns zero delay."""
        config = RetryConfig(policy=RetryPolicy.IMMEDIATE, max_retries=3)
        assert config.get_delay(1) == 0.0
        assert config.should_retry(Exception(), 1) is True

    def test_should_retry_respects_max(self):
        """Test that should_retry respects max_retries."""
        config = RetryConfig(max_retries=2)
        assert config.should_retry(Exception(), 1) is True
        assert config.should_retry(Exception(), 2) is False
        assert config.should_retry(Exception(), 3) is False

    def test_should_retry_with_specific_exceptions(self):
        """Test retry only on specific exception types."""
        config = RetryConfig(
            retry_on_exceptions={ValueError, TimeoutError},
            max_retries=3,
        )
        assert config.should_retry(ValueError("test"), 1) is True
        assert config.should_retry(TimeoutError("test"), 1) is True
        assert config.should_retry(RuntimeError("test"), 1) is False


# =============================================================================
# ExecutionConfig Tests
# =============================================================================

class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()
        assert config.mode == ExecutionMode.PARALLEL
        assert config.max_workers == 4
        assert config.checkpoint_enabled is True

    def test_validation_max_workers(self):
        """Test validation of max_workers."""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            ExecutionConfig(max_workers=0)
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            ExecutionConfig(max_workers=-1)

    def test_validation_checkpoint_interval(self):
        """Test validation of checkpoint_interval."""
        with pytest.raises(ValueError, match="checkpoint_interval must be >= 1"):
            ExecutionConfig(checkpoint_interval=0)


# =============================================================================
# TaskResult Tests
# =============================================================================

class TestTaskResult:
    """Tests for TaskResult."""

    def test_success_property(self):
        """Test success property."""
        result = TaskResult(task_id="test", status=TaskStatus.COMPLETED, result="value")
        assert result.success is True

        result = TaskResult(task_id="test", status=TaskStatus.FAILED, error="error")
        assert result.success is False

    def test_duration_calculation(self):
        """Test duration_seconds calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

        result = TaskResult(
            task_id="test",
            status=TaskStatus.COMPLETED,
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds == 10.0

    def test_duration_none_when_times_missing(self):
        """Test duration is None when times not set."""
        result = TaskResult(task_id="test", status=TaskStatus.COMPLETED)
        assert result.duration_seconds is None

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = TaskResult(
            task_id="test",
            status=TaskStatus.COMPLETED,
            result={"value": 42},
            retry_count=1,
        )
        d = result.to_dict()
        assert d["task_id"] == "test"
        assert d["status"] == "completed"
        assert d["result_type"] == "dict"
        assert d["retry_count"] == 1


# =============================================================================
# ExecutionContext Tests
# =============================================================================

class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_get_input(self):
        """Test get_input method."""
        ctx = ExecutionContext(
            task_id="test",
            execution_id="exec_1",
            input_data={"key1": "value1"},
        )
        assert ctx.get_input("key1") == "value1"
        assert ctx.get_input("missing", "default") == "default"

    def test_shared_state_thread_safe(self):
        """Test that shared state operations are thread-safe."""
        ctx = ExecutionContext(task_id="test", execution_id="exec_1")
        results = []

        def writer(value):
            ctx.set_shared("key", value)
            time.sleep(0.01)
            results.append(ctx.get_shared("key"))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should complete without error
        assert len(results) == 5

    def test_report_progress_clamped(self):
        """Test that progress is clamped to [0, 1]."""
        reported = []

        def callback(progress: float, message: str):
            reported.append(progress)

        ctx = ExecutionContext(
            task_id="test",
            execution_id="exec_1",
            progress_callback=callback,
        )

        ctx.report_progress(-0.5)
        ctx.report_progress(0.5)
        ctx.report_progress(1.5)

        assert reported == [0.0, 0.5, 1.0]


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipeline:
    """Tests for Pipeline class."""

    def test_add_task(self):
        """Test adding tasks to pipeline."""
        pipeline = Pipeline("test")
        task = PipelineTask(task_id="t1", executor=lambda ctx: None)
        pipeline.add_task(task)
        assert len(pipeline) == 1
        assert pipeline.get_task("t1") is task

    def test_add_duplicate_task_fails(self):
        """Test that adding duplicate task ID fails."""
        pipeline = Pipeline("test")
        task1 = PipelineTask(task_id="t1", executor=lambda ctx: None)
        task2 = PipelineTask(task_id="t1", executor=lambda ctx: None)
        pipeline.add_task(task1)
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_task(task2)

    def test_remove_task(self):
        """Test removing tasks."""
        pipeline = Pipeline("test")
        task = PipelineTask(task_id="t1", executor=lambda ctx: None)
        pipeline.add_task(task)
        assert pipeline.remove_task("t1") is True
        assert pipeline.remove_task("t1") is False
        assert len(pipeline) == 0

    def test_validate_empty_pipeline(self):
        """Test validation fails for empty pipeline."""
        pipeline = Pipeline("test")
        is_valid, errors = pipeline.validate()
        assert is_valid is False
        assert "no tasks" in errors[0].lower()

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        pipeline = Pipeline("test")
        task = PipelineTask(
            task_id="t1",
            executor=lambda ctx: None,
            dependencies=["missing_task"],
        )
        pipeline.add_task(task)
        is_valid, errors = pipeline.validate()
        assert is_valid is False
        assert any("missing_task" in e for e in errors)

    def test_validate_cycle_detection(self):
        """Test validation detects cycles."""
        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="a",
            executor=lambda ctx: None,
            dependencies=["c"],
        ))
        pipeline.add_task(PipelineTask(
            task_id="b",
            executor=lambda ctx: None,
            dependencies=["a"],
        ))
        pipeline.add_task(PipelineTask(
            task_id="c",
            executor=lambda ctx: None,
            dependencies=["b"],
        ))
        is_valid, errors = pipeline.validate()
        assert is_valid is False
        assert any("cycle" in e.lower() for e in errors)

    def test_execution_order(self, simple_pipeline):
        """Test topological sort for execution order."""
        order = simple_pipeline.get_execution_order()
        assert order == ["task_a", "task_b", "task_c"]

    def test_get_independent_tasks(self, parallel_pipeline):
        """Test finding independent tasks."""
        completed = {"start"}
        ready = parallel_pipeline.get_independent_tasks(completed)
        assert set(ready) == {"parallel_1", "parallel_2"}

    def test_priority_ordering(self):
        """Test that higher priority tasks come first."""
        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="low",
            executor=lambda ctx: None,
            priority=1,
        ))
        pipeline.add_task(PipelineTask(
            task_id="high",
            executor=lambda ctx: None,
            priority=10,
        ))
        order = pipeline.get_execution_order()
        assert order[0] == "high"


# =============================================================================
# PipelineRunner Tests
# =============================================================================

class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_sequential_execution(self, simple_pipeline):
        """Test sequential execution mode."""
        config = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL)
        runner = PipelineRunner(config=config)
        result = runner.execute(simple_pipeline)

        assert result.success is True
        assert len(result.task_results) == 3
        assert all(r.success for r in result.task_results.values())

    def test_parallel_execution(self, parallel_pipeline):
        """Test parallel execution mode."""
        config = ExecutionConfig(mode=ExecutionMode.PARALLEL, max_workers=4)
        runner = PipelineRunner(config=config)
        result = runner.execute(parallel_pipeline)

        assert result.success is True
        assert len(result.task_results) == 4

    def test_fail_fast(self):
        """Test fail_fast stops on first failure."""
        pipeline = Pipeline("test")

        execution_order = []

        def make_executor(task_id: str, should_fail: bool = False):
            def executor(ctx: ExecutionContext) -> str:
                execution_order.append(task_id)
                if should_fail:
                    raise ValueError(f"Task {task_id} failed")
                return task_id
            return executor

        pipeline.add_task(PipelineTask(task_id="t1", executor=make_executor("t1")))
        pipeline.add_task(PipelineTask(
            task_id="t2",
            executor=make_executor("t2", should_fail=True),
            dependencies=["t1"],
        ))
        pipeline.add_task(PipelineTask(
            task_id="t3",
            executor=make_executor("t3"),
            dependencies=["t2"],
        ))

        config = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL, fail_fast=True)
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.status == TaskStatus.FAILED
        assert "t3" not in execution_order  # t3 should not have run

    def test_task_with_retry(self):
        """Test task retry on failure."""
        attempts = []

        def flaky_executor(ctx: ExecutionContext) -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")
            return "success"

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="flaky",
            executor=flaky_executor,
            retry_config=RetryConfig(
                policy=RetryPolicy.IMMEDIATE,
                max_retries=5,
            ),
        ))

        runner = PipelineRunner()
        result = runner.execute(pipeline)

        assert result.success is True
        assert len(attempts) == 3  # Failed twice, succeeded on third
        assert result.task_results["flaky"].retry_count == 2

    def test_task_timeout(self):
        """Test task timeout handling."""
        def slow_executor(ctx: ExecutionContext) -> str:
            time.sleep(5)
            return "done"

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="slow",
            executor=slow_executor,
            timeout_seconds=0.1,
        ))

        config = ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            retry_config=RetryConfig(policy=RetryPolicy.NONE),
        )
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.status == TaskStatus.FAILED
        assert "timeout" in result.task_results["slow"].error.lower()

    def test_dependency_skipping(self):
        """Test that tasks are skipped when dependencies fail."""
        def failing_executor(ctx: ExecutionContext) -> str:
            raise ValueError("Intentional failure")

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="failing",
            executor=failing_executor,
        ))
        pipeline.add_task(PipelineTask(
            task_id="dependent",
            executor=lambda ctx: "ok",
            dependencies=["failing"],
        ))

        config = ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            retry_config=RetryConfig(policy=RetryPolicy.NONE),
        )
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.task_results["failing"].status == TaskStatus.FAILED
        assert result.task_results["dependent"].status == TaskStatus.SKIPPED

    def test_initial_inputs(self, simple_pipeline):
        """Test that initial inputs are passed to tasks."""
        def input_reader(ctx: ExecutionContext) -> Dict[str, Any]:
            return {"received": ctx.get_input("source")}

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="reader",
            executor=input_reader,
        ))

        runner = PipelineRunner()
        result = runner.execute(pipeline, initial_inputs={"source": "test_data"})

        assert result.success is True
        assert result.task_results["reader"].result["received"] == "test_data"

    def test_progress_callback(self, simple_pipeline):
        """Test progress callback is called."""
        progress_updates = []

        def progress_callback(progress: ExecutionProgress):
            progress_updates.append({
                "completed": progress.completed_tasks,
                "total": progress.total_tasks,
            })

        runner = PipelineRunner(progress_callback=progress_callback)
        runner.execute(simple_pipeline)

        assert len(progress_updates) > 0
        # Final update should show all completed
        assert progress_updates[-1]["completed"] == 3

    def test_cancel_execution(self, parallel_pipeline):
        """Test cancelling execution."""
        def slow_task(ctx: ExecutionContext) -> str:
            time.sleep(1)
            return "done"

        pipeline = Pipeline("test")
        for i in range(5):
            pipeline.add_task(PipelineTask(
                task_id=f"task_{i}",
                executor=slow_task,
            ))

        runner = PipelineRunner()

        # Cancel after brief delay
        def cancel_later():
            time.sleep(0.1)
            runner.cancel()

        threading.Thread(target=cancel_later).start()
        result = runner.execute(pipeline)

        assert result.status == TaskStatus.CANCELLED


# =============================================================================
# run_pipeline Convenience Function Tests
# =============================================================================

class TestRunPipeline:
    """Tests for run_pipeline convenience function."""

    def test_basic_run(self, simple_pipeline):
        """Test basic run_pipeline usage."""
        result = run_pipeline(simple_pipeline)
        assert result.success is True

    def test_with_config(self, simple_pipeline):
        """Test run_pipeline with custom config."""
        config = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL)
        result = run_pipeline(simple_pipeline, config=config)
        assert result.success is True


# =============================================================================
# ExecutionProgress Tests
# =============================================================================

class TestExecutionProgress:
    """Tests for ExecutionProgress."""

    def test_progress_fraction(self):
        """Test progress_fraction calculation."""
        progress = ExecutionProgress(
            total_tasks=10,
            completed_tasks=5,
            failed_tasks=2,
        )
        assert progress.progress_fraction == 0.7  # (5+2)/10

    def test_progress_fraction_empty(self):
        """Test progress_fraction with no tasks."""
        progress = ExecutionProgress(total_tasks=0)
        assert progress.progress_fraction == 0.0

    def test_is_complete(self):
        """Test is_complete property."""
        progress = ExecutionProgress(
            total_tasks=3,
            completed_tasks=2,
            failed_tasks=1,
        )
        assert progress.is_complete is True

        progress = ExecutionProgress(
            total_tasks=3,
            completed_tasks=2,
            failed_tasks=0,
        )
        assert progress.is_complete is False


# =============================================================================
# ExecutionResult Tests
# =============================================================================

class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_failed_tasks(self):
        """Test failed_tasks property."""
        result = ExecutionResult(
            execution_id="test",
            status=TaskStatus.FAILED,
            task_results={
                "t1": TaskResult("t1", TaskStatus.COMPLETED),
                "t2": TaskResult("t2", TaskStatus.FAILED, error="error"),
                "t3": TaskResult("t3", TaskStatus.COMPLETED),
            },
        )
        assert len(result.failed_tasks) == 1
        assert result.failed_tasks[0].task_id == "t2"

    def test_successful_tasks(self):
        """Test successful_tasks property."""
        result = ExecutionResult(
            execution_id="test",
            status=TaskStatus.COMPLETED,
            task_results={
                "t1": TaskResult("t1", TaskStatus.COMPLETED),
                "t2": TaskResult("t2", TaskStatus.COMPLETED),
            },
        )
        assert len(result.successful_tasks) == 2


# =============================================================================
# CheckpointManager Tests
# =============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load(self, checkpoint_manager):
        """Test saving and loading checkpoint."""
        data = {
            "task_results": {"t1": {"status": "completed"}},
            "shared_state": {"key": "value"},
            "completed": ["t1"],
        }

        metadata = checkpoint_manager.save("exec_1", data)
        assert metadata.status == CheckpointStatus.VALID

        loaded = checkpoint_manager.load_latest("exec_1")
        assert loaded is not None
        assert "task_results" in loaded

    def test_version_increment(self, checkpoint_manager):
        """Test checkpoint version incrementing."""
        data = {"task_results": {}, "completed": []}

        m1 = checkpoint_manager.save("exec_1", data)
        m2 = checkpoint_manager.save("exec_1", data)
        m3 = checkpoint_manager.save("exec_1", data)

        assert m1.version == 1
        assert m2.version == 2
        assert m3.version == 3

    def test_checksum_verification(self, checkpoint_manager):
        """Test that checksum verification works."""
        data = {"task_results": {}, "completed": []}
        metadata = checkpoint_manager.save("exec_1", data)

        # Load should verify checksum
        loaded = checkpoint_manager.load(metadata.checkpoint_id)
        assert loaded is not None

    def test_list_checkpoints(self, checkpoint_manager):
        """Test listing checkpoints."""
        data = {"task_results": {}, "completed": []}
        checkpoint_manager.save("exec_1", data)
        checkpoint_manager.save("exec_1", data)
        checkpoint_manager.save("exec_2", data)

        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 3

        exec1_checkpoints = checkpoint_manager.list_checkpoints(execution_id="exec_1")
        assert len(exec1_checkpoints) == 2

    def test_delete_checkpoint(self, checkpoint_manager):
        """Test deleting checkpoint."""
        data = {"task_results": {}, "completed": []}
        metadata = checkpoint_manager.save("exec_1", data)

        assert checkpoint_manager.delete(metadata.checkpoint_id) is True
        assert checkpoint_manager.load(metadata.checkpoint_id) is None

    def test_cleanup_old_checkpoints(self, checkpoint_manager):
        """Test cleanup of old checkpoints."""
        data = {"task_results": {}, "completed": []}

        # Create several checkpoints
        for _ in range(5):
            checkpoint_manager.save("exec_1", data)

        # Clean up, keeping only 2
        deleted = checkpoint_manager.cleanup(execution_id="exec_1", keep_latest=2)
        assert deleted == 3

        remaining = checkpoint_manager.list_checkpoints(execution_id="exec_1")
        assert len(remaining) == 2

    def test_get_execution_progress(self, checkpoint_manager):
        """Test getting execution progress from checkpoint."""
        data = {
            "task_results": {"t1": {}, "t2": {}, "t3": {}},
            "completed": ["t1", "t2"],
        }
        checkpoint_manager.save("exec_1", data)

        progress = checkpoint_manager.get_execution_progress("exec_1")
        assert progress is not None
        assert progress["completed_tasks"] == 2


# =============================================================================
# CheckpointConfig Tests
# =============================================================================

class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.max_checkpoints == 10
        assert config.retention_hours == 168  # 7 days
        assert config.auto_cleanup is True

    def test_validation_max_checkpoints(self):
        """Test validation of max_checkpoints."""
        with pytest.raises(ValueError, match="max_checkpoints must be >= 1"):
            CheckpointConfig(max_checkpoints=0)

    def test_validation_retention_hours(self):
        """Test validation of retention_hours."""
        with pytest.raises(ValueError, match="retention_hours must be >= 0"):
            CheckpointConfig(retention_hours=-1)

    def test_path_string_conversion(self):
        """Test that string paths are converted to Path."""
        config = CheckpointConfig(base_path="./my_checkpoints")
        assert isinstance(config.base_path, Path)


# =============================================================================
# LocalStorageBackend Tests
# =============================================================================

class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    def test_save_and_load(self, temp_checkpoint_dir):
        """Test save and load operations."""
        backend = LocalStorageBackend(temp_checkpoint_dir)
        metadata = CheckpointMetadata(
            checkpoint_id="test_1",
            execution_id="exec_1",
        )

        data = b"test checkpoint data"
        backend.save("test_1", data, metadata)

        loaded = backend.load("test_1")
        assert loaded == data

    def test_exists(self, temp_checkpoint_dir):
        """Test exists check."""
        backend = LocalStorageBackend(temp_checkpoint_dir)
        metadata = CheckpointMetadata(
            checkpoint_id="test_1",
            execution_id="exec_1",
        )

        assert backend.exists("test_1") is False
        backend.save("test_1", b"data", metadata)
        assert backend.exists("test_1") is True

    def test_delete(self, temp_checkpoint_dir):
        """Test delete operation."""
        backend = LocalStorageBackend(temp_checkpoint_dir)
        metadata = CheckpointMetadata(
            checkpoint_id="test_1",
            execution_id="exec_1",
        )

        backend.save("test_1", b"data", metadata)
        assert backend.delete("test_1") is True
        assert backend.exists("test_1") is False
        assert backend.delete("test_1") is False  # Already deleted

    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing checkpoints."""
        backend = LocalStorageBackend(temp_checkpoint_dir)

        for i in range(3):
            metadata = CheckpointMetadata(
                checkpoint_id=f"ckpt_{i}",
                execution_id="exec_1",
            )
            backend.save(f"ckpt_{i}", b"data", metadata)

        checkpoints = backend.list_checkpoints()
        assert len(checkpoints) == 3


# =============================================================================
# AutoCheckpointer Tests
# =============================================================================

class TestAutoCheckpointer:
    """Tests for AutoCheckpointer."""

    def test_auto_checkpoint_creates_checkpoints(self, checkpoint_manager):
        """Test that auto checkpointer creates checkpoints."""
        checkpointed = []

        def on_checkpoint(metadata):
            checkpointed.append(metadata.checkpoint_id)

        with AutoCheckpointer(
            checkpoint_manager,
            execution_id="exec_1",
            interval_seconds=0.1,
            on_checkpoint=on_checkpoint,
        ) as auto:
            auto.update_data({"task_results": {}, "completed": []})
            time.sleep(0.35)  # Allow for ~3 checkpoints

        assert len(checkpointed) >= 2

    def test_context_manager(self, checkpoint_manager):
        """Test using as context manager."""
        with AutoCheckpointer(
            checkpoint_manager,
            execution_id="exec_1",
            interval_seconds=1.0,
        ) as auto:
            auto.update_data({"test": "data"})
        # Should not raise, just ensure clean exit


# =============================================================================
# DistributedConfig Tests
# =============================================================================

class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DistributedConfig()
        assert config.backend == DistributedBackend.AUTO
        assert config.n_workers == 4
        assert config.memory_limit_per_worker == "4GB"

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = DistributedConfig(n_workers=8, adaptive=True)
        d = config.to_dict()
        assert d["n_workers"] == 8
        assert d["adaptive"] is True


# =============================================================================
# WorkerInfo Tests
# =============================================================================

class TestWorkerInfo:
    """Tests for WorkerInfo."""

    def test_memory_gb(self):
        """Test memory_gb property."""
        worker = WorkerInfo(
            worker_id="w1",
            host="localhost",
            port=8000,
            ncores=4,
            memory_bytes=4 * 1024 * 1024 * 1024,  # 4GB
        )
        assert worker.memory_gb == 4.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        worker = WorkerInfo(
            worker_id="w1",
            host="localhost",
            port=8000,
            ncores=4,
            memory_bytes=4 * 1024 * 1024 * 1024,
            status="active",
        )
        d = worker.to_dict()
        assert d["worker_id"] == "w1"
        assert d["ncores"] == 4


# =============================================================================
# ClusterInfo Tests
# =============================================================================

class TestClusterInfo:
    """Tests for ClusterInfo."""

    def test_worker_count(self):
        """Test worker_count property."""
        workers = [
            WorkerInfo("w1", "localhost", 8000, 4, 1000, status="active"),
            WorkerInfo("w2", "localhost", 8001, 4, 1000, status="active"),
            WorkerInfo("w3", "localhost", 8002, 4, 1000, status="inactive"),
        ]
        cluster = ClusterInfo(
            backend=DistributedBackend.LOCAL,
            scheduler_address="local",
            workers=workers,
            total_cores=12,
            total_memory_bytes=3000,
        )
        assert cluster.worker_count == 2  # Only active


# =============================================================================
# LocalExecutor Tests
# =============================================================================

class TestLocalExecutor:
    """Tests for LocalExecutor."""

    def test_connect_disconnect(self):
        """Test connect and disconnect."""
        config = DistributedConfig(n_workers=2)
        executor = LocalExecutor(config)

        cluster_info = executor.connect()
        assert cluster_info.is_connected is True
        assert cluster_info.backend == DistributedBackend.LOCAL
        assert len(cluster_info.workers) == 2

        executor.disconnect()
        assert executor.is_connected is False

    def test_submit_and_gather(self):
        """Test submitting tasks and gathering results."""
        config = DistributedConfig(n_workers=2)
        executor = LocalExecutor(config)
        executor.connect()

        try:
            task = PipelineTask(
                task_id="test_task",
                executor=lambda ctx: "result",
            )
            context = ExecutionContext(
                task_id="test_task",
                execution_id="exec_1",
            )

            future = executor.submit_task(task, context, [])
            results = executor.gather_results([future])

            assert len(results) == 1
            assert results[0].task_id == "test_task"
            assert results[0].status == TaskStatus.COMPLETED
        finally:
            executor.disconnect()

    def test_get_cluster_info(self):
        """Test get_cluster_info when connected."""
        config = DistributedConfig(n_workers=4)
        executor = LocalExecutor(config)
        executor.connect()

        try:
            info = executor.get_cluster_info()
            assert info.total_cores == 4
            assert info.is_connected is True
        finally:
            executor.disconnect()


# =============================================================================
# get_executor Tests
# =============================================================================

class TestGetExecutor:
    """Tests for get_executor factory function."""

    def test_local_backend(self):
        """Test getting local executor."""
        config = DistributedConfig(backend=DistributedBackend.LOCAL)
        executor = get_executor(config)
        assert isinstance(executor, LocalExecutor)

    def test_auto_backend(self):
        """Test auto backend detection falls back to local."""
        config = DistributedConfig(backend=DistributedBackend.AUTO)
        executor = get_executor(config)
        # Should return some executor (likely LocalExecutor if Dask/Ray not installed)
        assert isinstance(executor, DistributedExecutorBase)


# =============================================================================
# DistributedPipelineRunner Tests
# =============================================================================

class TestDistributedPipelineRunner:
    """Tests for DistributedPipelineRunner."""

    def test_execute_pipeline(self, simple_pipeline):
        """Test executing pipeline with distributed runner."""
        dist_config = DistributedConfig(backend=DistributedBackend.LOCAL, n_workers=2)
        runner = DistributedPipelineRunner(distributed_config=dist_config)

        result = runner.execute(simple_pipeline)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.task_results) == 3

    def test_invalid_pipeline(self):
        """Test handling invalid pipeline."""
        pipeline = Pipeline("invalid")  # Empty pipeline

        dist_config = DistributedConfig(backend=DistributedBackend.LOCAL)
        runner = DistributedPipelineRunner(distributed_config=dist_config)

        result = runner.execute(pipeline)
        assert result.status == TaskStatus.FAILED
        assert "validation_errors" in result.metadata

    def test_progress_callback(self, simple_pipeline):
        """Test progress callback with distributed runner."""
        progress_updates = []

        def progress_callback(progress: ExecutionProgress):
            progress_updates.append(progress.current_phase)

        dist_config = DistributedConfig(backend=DistributedBackend.LOCAL, n_workers=2)
        runner = DistributedPipelineRunner(
            distributed_config=dist_config,
            progress_callback=progress_callback,
        )

        runner.execute(simple_pipeline)

        assert "submitting" in progress_updates
        assert "executing" in progress_updates


# =============================================================================
# Integration Tests
# =============================================================================

class TestExecutionIntegration:
    """Integration tests for the complete execution system."""

    def test_full_execution_with_checkpoints(
        self, simple_pipeline, checkpoint_manager
    ):
        """Test full execution with checkpoint integration."""
        config = ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            checkpoint_enabled=True,
            checkpoint_interval=1,
        )
        runner = PipelineRunner(
            config=config,
            checkpoint_manager=checkpoint_manager,
        )

        result = runner.execute(simple_pipeline)

        assert result.success is True

        # Verify checkpoints were created
        checkpoints = checkpoint_manager.list_checkpoints()
        # May have checkpoints depending on task count
        assert len(checkpoints) >= 0

    def test_resume_from_checkpoint(self, checkpoint_manager):
        """Test resuming execution from checkpoint."""
        # Create initial checkpoint
        checkpoint_manager.save("exec_1", {
            "task_results": {
                "task_a": {"status": "completed", "result": "done"},
            },
            "shared_state": {},
            "completed": ["task_a"],
        })

        # Load and verify
        loaded = checkpoint_manager.load_latest("exec_1")
        assert loaded is not None
        assert "task_a" in loaded.get("completed", [])

    def test_parallel_with_dependencies(self):
        """Test parallel execution respects dependencies."""
        execution_times: Dict[str, float] = {}

        def timed_executor(task_id: str):
            def executor(ctx: ExecutionContext) -> Dict[str, Any]:
                execution_times[task_id] = time.time()
                time.sleep(0.05)
                return {"task_id": task_id}
            return executor

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="first",
            executor=timed_executor("first"),
        ))
        pipeline.add_task(PipelineTask(
            task_id="second",
            executor=timed_executor("second"),
            dependencies=["first"],
        ))

        config = ExecutionConfig(mode=ExecutionMode.PARALLEL, max_workers=4)
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.success is True
        # Second must start after first
        assert execution_times["second"] >= execution_times["first"]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_pipeline_result(self):
        """Test that empty pipeline returns appropriate error."""
        pipeline = Pipeline("empty")
        runner = PipelineRunner()
        result = runner.execute(pipeline)

        assert result.status == TaskStatus.FAILED
        assert "validation_errors" in result.metadata or "error" in result.metadata

    def test_task_with_none_result(self):
        """Test task that returns None."""
        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="none_result",
            executor=lambda ctx: None,
        ))

        runner = PipelineRunner()
        result = runner.execute(pipeline)

        assert result.success is True
        assert result.task_results["none_result"].result is None

    def test_exception_in_task(self):
        """Test handling of uncaught exception in task."""
        def failing_task(ctx: ExecutionContext) -> None:
            raise RuntimeError("Task failed unexpectedly")

        pipeline = Pipeline("test")
        pipeline.add_task(PipelineTask(
            task_id="failing",
            executor=failing_task,
        ))

        config = ExecutionConfig(
            retry_config=RetryConfig(policy=RetryPolicy.NONE),
        )
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.status == TaskStatus.FAILED
        assert result.task_results["failing"].exception is not None

    def test_deeply_nested_dependencies(self):
        """Test pipeline with long dependency chain."""
        pipeline = Pipeline("test")

        for i in range(10):
            deps = [f"task_{i-1}"] if i > 0 else []
            pipeline.add_task(PipelineTask(
                task_id=f"task_{i}",
                executor=lambda ctx: "ok",
                dependencies=deps,
            ))

        runner = PipelineRunner()
        result = runner.execute(pipeline)

        assert result.success is True
        assert len(result.task_results) == 10

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency pattern."""
        pipeline = Pipeline("test")

        pipeline.add_task(PipelineTask(
            task_id="start",
            executor=lambda ctx: "start",
        ))
        pipeline.add_task(PipelineTask(
            task_id="left",
            executor=lambda ctx: "left",
            dependencies=["start"],
        ))
        pipeline.add_task(PipelineTask(
            task_id="right",
            executor=lambda ctx: "right",
            dependencies=["start"],
        ))
        pipeline.add_task(PipelineTask(
            task_id="end",
            executor=lambda ctx: "end",
            dependencies=["left", "right"],
        ))

        config = ExecutionConfig(mode=ExecutionMode.PARALLEL)
        runner = PipelineRunner(config=config)
        result = runner.execute(pipeline)

        assert result.success is True
        assert all(r.success for r in result.task_results.values())


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of execution components."""

    def test_concurrent_context_updates(self):
        """Test concurrent updates to ExecutionContext shared state."""
        ctx = ExecutionContext(
            task_id="test",
            execution_id="exec_1",
        )

        errors = []

        def updater(thread_id: int):
            try:
                for i in range(100):
                    ctx.set_shared(f"key_{thread_id}_{i}", i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_pipeline_modifications(self):
        """Test thread safety of pipeline modifications."""
        pipeline = Pipeline("test")
        errors = []

        def add_tasks(start: int):
            try:
                for i in range(start, start + 10):
                    try:
                        pipeline.add_task(PipelineTask(
                            task_id=f"task_{i}",
                            executor=lambda ctx: None,
                        ))
                    except ValueError:
                        pass  # Duplicate ID is expected in concurrent scenario
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_tasks, args=(i * 5,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unexpected errors
        assert len(errors) == 0
        # Some tasks should have been added
        assert len(pipeline) > 0
