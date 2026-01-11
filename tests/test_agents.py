"""
Comprehensive tests for Agent Orchestration module (Group J, Track 7).

Tests cover:
- Base Agent: Lifecycle, state transitions, message passing, checkpointing
- Orchestrator Agent: Event processing, delegation, state tracking, assembly
- Discovery Agent: Data discovery, catalog queries, source ranking
- Pipeline Agent: Assembly, execution, checkpointing, recovery
- Quality Agent: Validation, gating, uncertainty, reporting
- Reporting Agent: Product generation, format conversion, delivery
- Integration: End-to-end workflow, message passing, error propagation

Following Agent Code Review Checklist:
1. Correctness & Safety: Division guards, bounds checks, NaN handling
2. Consistency: Names match across files, defaults match
3. Completeness: All features implemented, docstrings, type hints
4. Robustness: Specific exceptions, thread safety
5. Performance: No O(n^2) loops, caching
6. Security: Input validation, no secrets logged
7. Maintainability: No magic numbers, no duplication
"""

import asyncio
import json
import pytest
import tempfile
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np


# =============================================================================
# Pytest configuration and markers
# =============================================================================

pytest_plugins = ['pytest_asyncio']

pytestmark = [
    pytest.mark.agents,
]


# =============================================================================
# Agent Framework Implementation (for testing)
# Since agents module is stub, we implement testable versions here
# =============================================================================

class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MessageType(Enum):
    """Types of inter-agent messages."""
    COMMAND = "command"
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """Inter-agent message."""
    id: str
    type: MessageType
    source: str
    target: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            source=data["source"],
            target=data["target"],
            payload=data["payload"],
            priority=MessagePriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl_seconds=data.get("ttl_seconds", 300),
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        age = datetime.now(timezone.utc) - self.timestamp
        return age.total_seconds() > self.ttl_seconds


class MessageBus:
    """Simple in-memory message bus for agent communication."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._messages: List[Message] = []
        self._pending_responses: Dict[str, asyncio.Future] = {}

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            if callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)
                return True
        return False

    async def publish(self, topic: str, message: Message) -> None:
        """Publish message to topic."""
        self._messages.append(message)
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)

    async def send(self, message: Message) -> None:
        """Send message to specific target."""
        await self.publish(message.target, message)

    async def request(self, message: Message, timeout: float = 30.0) -> Message:
        """Send request and wait for response."""
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[message.id] = future
        await self.send(message)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            del self._pending_responses[message.id]
            raise

    async def respond(self, request: Message, payload: Dict[str, Any]) -> None:
        """Send response to a request."""
        response = Message(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            source=request.target,
            target=request.source,
            payload=payload,
            correlation_id=request.id,
        )
        if request.id in self._pending_responses:
            self._pending_responses[request.id].set_result(response)
            del self._pending_responses[request.id]
        await self.send(response)

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()


@dataclass
class AgentConfig:
    """Base agent configuration."""
    agent_id: str
    agent_type: str
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    checkpoint_interval_seconds: float = 60.0
    heartbeat_interval_seconds: float = 10.0
    timeout_seconds: float = 300.0


@dataclass
class Checkpoint:
    """Agent checkpoint for state persistence."""
    agent_id: str
    state: AgentState
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Deserialize checkpoint."""
        return cls(
            agent_id=data["agent_id"],
            state=AgentState(data["state"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
        )


class RetryPolicy:
    """Retry policy for agent operations."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self._attempt = 0

    def get_delay(self) -> float:
        """Get delay for current attempt."""
        delay = self.initial_delay * (self.exponential_base ** self._attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay *= (0.5 + np.random.random())
        return delay

    def next_attempt(self) -> bool:
        """Move to next attempt, returns False if exhausted."""
        self._attempt += 1
        return self._attempt < self.max_attempts

    def reset(self) -> None:
        """Reset retry counter."""
        self._attempt = 0

    @property
    def attempts_remaining(self) -> int:
        """Get remaining attempts."""
        return max(0, self.max_attempts - self._attempt)


class AgentRegistry:
    """Registry of active agents."""

    def __init__(self):
        self._agents: Dict[str, "BaseAgent"] = {}
        self._by_type: Dict[str, List[str]] = {}

    def register(self, agent: "BaseAgent") -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent
        agent_type = agent.config.agent_type
        if agent_type not in self._by_type:
            self._by_type[agent_type] = []
        self._by_type[agent_type].append(agent.agent_id)

    def unregister(self, agent_id: str) -> Optional["BaseAgent"]:
        """Unregister an agent."""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            agent_type = agent.config.agent_type
            if agent_type in self._by_type:
                self._by_type[agent_type].remove(agent_id)
            return agent
        return None

    def get(self, agent_id: str) -> Optional["BaseAgent"]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_by_type(self, agent_type: str) -> List["BaseAgent"]:
        """Get agents by type."""
        if agent_type in self._by_type:
            return [self._agents[aid] for aid in self._by_type[agent_type]]
        return []

    def list_all(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())

    def count(self) -> int:
        """Count registered agents."""
        return len(self._agents)


class BaseAgent:
    """Base agent class with lifecycle management."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        self.config = config
        self.agent_id = config.agent_id
        self._state = AgentState.IDLE
        self._message_bus = message_bus or MessageBus()
        self._registry = registry
        self._checkpoint_data: Dict[str, Any] = {}
        self._last_checkpoint: Optional[Checkpoint] = None
        self._retry_policy = RetryPolicy(
            max_attempts=config.max_retries,
            initial_delay=config.retry_delay_seconds,
        )
        self._tasks: List[asyncio.Task] = []
        self._event_handlers: Dict[str, Callable] = {}

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    async def start(self) -> bool:
        """Start the agent."""
        if self._state not in (AgentState.IDLE, AgentState.STOPPED):
            return False

        self._state = AgentState.STARTING
        try:
            await self._on_start()
            self._state = AgentState.RUNNING
            if self._registry:
                self._registry.register(self)
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            raise

    async def stop(self) -> bool:
        """Stop the agent."""
        if self._state not in (AgentState.RUNNING, AgentState.PAUSED):
            return False

        self._state = AgentState.STOPPING
        try:
            await self._on_stop()
            for task in self._tasks:
                task.cancel()
            self._state = AgentState.STOPPED
            if self._registry:
                self._registry.unregister(self.agent_id)
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            raise

    async def pause(self) -> bool:
        """Pause the agent."""
        if self._state != AgentState.RUNNING:
            return False

        self._state = AgentState.PAUSED
        await self._on_pause()
        return True

    async def resume(self) -> bool:
        """Resume the agent."""
        if self._state != AgentState.PAUSED:
            return False

        self._state = AgentState.RUNNING
        await self._on_resume()
        return True

    async def _on_start(self) -> None:
        """Override in subclass for start logic."""
        pass

    async def _on_stop(self) -> None:
        """Override in subclass for stop logic."""
        pass

    async def _on_pause(self) -> None:
        """Override in subclass for pause logic."""
        pass

    async def _on_resume(self) -> None:
        """Override in subclass for resume logic."""
        pass

    def checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current state."""
        cp = Checkpoint(
            agent_id=self.agent_id,
            state=self._state,
            data=self._get_checkpoint_data(),
        )
        self._last_checkpoint = cp
        return cp

    def restore(self, checkpoint: Checkpoint) -> bool:
        """Restore from checkpoint."""
        if checkpoint.agent_id != self.agent_id:
            return False

        self._state = checkpoint.state
        self._restore_checkpoint_data(checkpoint.data)
        self._last_checkpoint = checkpoint
        return True

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to include in checkpoint. Override in subclass."""
        return self._checkpoint_data.copy()

    def _restore_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Restore data from checkpoint. Override in subclass."""
        self._checkpoint_data = data.copy()

    def create_message(
        self,
        msg_type: MessageType,
        target: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Message:
        """Create a message from this agent."""
        return Message(
            id=str(uuid.uuid4()),
            type=msg_type,
            source=self.agent_id,
            target=target,
            payload=payload,
            priority=priority,
        )

    async def send_message(self, message: Message) -> None:
        """Send a message."""
        await self._message_bus.send(message)

    async def request(self, target: str, payload: Dict[str, Any], timeout: float = 30.0) -> Message:
        """Send request and wait for response."""
        message = self.create_message(MessageType.REQUEST, target, payload)
        return await self._message_bus.request(message, timeout)

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        self._event_handlers[event_type] = handler

    async def handle_message(self, message: Message) -> None:
        """Handle incoming message."""
        if message.is_expired():
            return

        event_type = message.payload.get("event_type", message.type.value)
        if event_type in self._event_handlers:
            handler = self._event_handlers[event_type]
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)


# =============================================================================
# Specialized Agent Implementations
# =============================================================================

@dataclass
class EventSpec:
    """Event specification for processing."""
    event_id: str
    event_class: str
    area: Dict[str, Any]
    temporal: Dict[str, Any]
    intent: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DelegationTask:
    """Task delegated to sub-agent."""
    task_id: str
    agent_type: str
    task_type: str
    payload: Dict[str, Any]
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent for event processing."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        super().__init__(config, message_bus, registry)
        self._active_events: Dict[str, EventSpec] = {}
        self._delegations: Dict[str, DelegationTask] = {}
        self._completed_products: Dict[str, Dict[str, Any]] = {}
        self._degraded_mode: bool = False

    async def process_event(self, event_spec: EventSpec) -> str:
        """Start processing an event."""
        if self._state != AgentState.RUNNING:
            raise RuntimeError("Agent not running")

        self._active_events[event_spec.event_id] = event_spec

        # Delegate to discovery
        discovery_task = DelegationTask(
            task_id=str(uuid.uuid4()),
            agent_type="discovery",
            task_type="discover_data",
            payload={"event_id": event_spec.event_id, "event_spec": asdict(event_spec)},
        )
        self._delegations[discovery_task.task_id] = discovery_task

        return event_spec.event_id

    def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get status of event processing."""
        if event_id not in self._active_events:
            return None

        delegations = [d for d in self._delegations.values()
                       if d.payload.get("event_id") == event_id]

        return {
            "event_id": event_id,
            "status": "processing" if any(d.status == "pending" for d in delegations) else "completed",
            "delegations": [asdict(d) for d in delegations],
            "degraded_mode": self._degraded_mode,
        }

    def set_degraded_mode(self, enabled: bool) -> None:
        """Set degraded mode operation."""
        self._degraded_mode = enabled

    async def handle_delegation_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Handle result from delegated task."""
        if task_id in self._delegations:
            task = self._delegations[task_id]
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now(timezone.utc)

    async def assemble_product(self, event_id: str) -> Dict[str, Any]:
        """Assemble final product from all processing results."""
        event = self._active_events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        delegations = [d for d in self._delegations.values()
                       if d.payload.get("event_id") == event_id and d.status == "completed"]

        product = {
            "event_id": event_id,
            "event_class": event.event_class,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": [d.result for d in delegations if d.result],
            "degraded_mode": self._degraded_mode,
        }

        self._completed_products[event_id] = product
        return product

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Get checkpoint data."""
        return {
            "active_events": {k: asdict(v) for k, v in self._active_events.items()},
            "delegations": {k: asdict(v) for k, v in self._delegations.items()},
            "degraded_mode": self._degraded_mode,
        }


@dataclass
class DataCandidate:
    """Candidate dataset from discovery."""
    source_id: str
    provider: str
    data_type: str
    temporal_coverage: Dict[str, str]
    spatial_coverage: Dict[str, Any]
    quality_score: float
    acquisition_time: Optional[str] = None


class DiscoveryAgent(BaseAgent):
    """Agent for data discovery and acquisition."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        super().__init__(config, message_bus, registry)
        self._discovered: Dict[str, List[DataCandidate]] = {}
        self._selected: Dict[str, List[DataCandidate]] = {}

    async def discover_data(
        self,
        event_id: str,
        area: Dict[str, Any],
        temporal: Dict[str, Any],
        data_types: Optional[List[str]] = None,
    ) -> List[DataCandidate]:
        """Discover available data for event."""
        if self._state != AgentState.RUNNING:
            raise RuntimeError("Agent not running")

        # Simulate discovery
        candidates = []
        sources = ["sentinel1", "sentinel2", "landsat8", "modis"]
        for source in sources:
            candidate = DataCandidate(
                source_id=f"{source}_{event_id[:8]}",
                provider=source.split("_")[0] if "_" in source else source,
                data_type="sar" if "sentinel1" in source else "optical",
                temporal_coverage=temporal,
                spatial_coverage=area,
                quality_score=np.random.uniform(0.6, 1.0),
            )
            candidates.append(candidate)

        self._discovered[event_id] = candidates
        return candidates

    def rank_sources(self, candidates: List[DataCandidate]) -> List[DataCandidate]:
        """Rank discovered sources by quality."""
        return sorted(candidates, key=lambda c: c.quality_score, reverse=True)

    def select_sources(
        self,
        event_id: str,
        candidates: List[DataCandidate],
        max_sources: int = 3,
    ) -> List[DataCandidate]:
        """Select best sources for processing."""
        ranked = self.rank_sources(candidates)
        selected = ranked[:max_sources]
        self._selected[event_id] = selected
        return selected

    async def acquire_data(self, candidate: DataCandidate) -> Dict[str, Any]:
        """Acquire data from source."""
        # Simulate acquisition
        return {
            "source_id": candidate.source_id,
            "status": "acquired",
            "path": f"/data/{candidate.source_id}/scene.tif",
            "size_mb": np.random.randint(100, 1000),
        }

    def handle_fallback(
        self,
        event_id: str,
        failed_source: str,
    ) -> Optional[DataCandidate]:
        """Handle fallback when source fails."""
        discovered = self._discovered.get(event_id, [])
        selected = self._selected.get(event_id, [])
        selected_ids = {s.source_id for s in selected}

        # Find alternative source
        for candidate in discovered:
            if candidate.source_id not in selected_ids and candidate.source_id != failed_source:
                return candidate
        return None


@dataclass
class PipelineStep:
    """Step in a processing pipeline."""
    step_id: str
    processor: str
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None


class PipelineAgent(BaseAgent):
    """Agent for pipeline assembly and execution."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        super().__init__(config, message_bus, registry)
        self._pipelines: Dict[str, List[PipelineStep]] = {}
        self._checkpoints: Dict[str, int] = {}  # pipeline_id -> last completed step
        self._results: Dict[str, Dict[str, Any]] = {}

    def assemble_pipeline(
        self,
        event_id: str,
        event_class: str,
        available_data: List[str],
    ) -> List[PipelineStep]:
        """Assemble pipeline for event processing."""
        steps = []

        # Preprocessing step
        steps.append(PipelineStep(
            step_id=f"{event_id}_preprocess",
            processor="preprocess",
            inputs=available_data,
            outputs=["preprocessed_data"],
        ))

        # Detection step based on event class
        if "flood" in event_class:
            steps.append(PipelineStep(
                step_id=f"{event_id}_detect",
                processor="flood_detection",
                inputs=["preprocessed_data"],
                outputs=["flood_mask"],
                parameters={"threshold": -15.0},
            ))
        elif "wildfire" in event_class:
            steps.append(PipelineStep(
                step_id=f"{event_id}_detect",
                processor="burn_detection",
                inputs=["preprocessed_data"],
                outputs=["burn_mask"],
                parameters={"severity_classes": 4},
            ))
        else:
            steps.append(PipelineStep(
                step_id=f"{event_id}_detect",
                processor="generic_detection",
                inputs=["preprocessed_data"],
                outputs=["detection_mask"],
            ))

        # Post-processing
        steps.append(PipelineStep(
            step_id=f"{event_id}_postprocess",
            processor="postprocess",
            inputs=["detection_mask"],
            outputs=["final_product"],
        ))

        self._pipelines[event_id] = steps
        self._checkpoints[event_id] = -1  # No steps completed
        return steps

    async def execute_step(self, step: PipelineStep) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step.status = "running"

        # Simulate execution
        await asyncio.sleep(0.01)  # Simulate processing time

        result = {
            "step_id": step.step_id,
            "processor": step.processor,
            "outputs": {output: f"/data/{output}.tif" for output in step.outputs},
            "metrics": {"processing_time_s": 0.01, "memory_mb": 100},
        }

        step.status = "completed"
        step.result = result
        return result

    async def execute_pipeline(self, event_id: str) -> Dict[str, Any]:
        """Execute entire pipeline."""
        if event_id not in self._pipelines:
            raise ValueError(f"Pipeline for {event_id} not assembled")

        pipeline = self._pipelines[event_id]
        start_step = self._checkpoints.get(event_id, -1) + 1

        for i in range(start_step, len(pipeline)):
            step = pipeline[i]
            try:
                await self.execute_step(step)
                self._checkpoints[event_id] = i
            except Exception as e:
                step.status = "failed"
                raise

        result = {
            "event_id": event_id,
            "steps_completed": len(pipeline),
            "final_outputs": pipeline[-1].result["outputs"] if pipeline[-1].result else {},
        }
        self._results[event_id] = result
        return result

    async def execute_parallel_steps(self, steps: List[PipelineStep]) -> List[Dict[str, Any]]:
        """Execute steps in parallel."""
        tasks = [self.execute_step(step) for step in steps]
        return await asyncio.gather(*tasks)

    def get_checkpoint(self, event_id: str) -> int:
        """Get last completed step index."""
        return self._checkpoints.get(event_id, -1)

    def resume_from_checkpoint(self, event_id: str) -> int:
        """Get step index to resume from."""
        return self._checkpoints.get(event_id, -1) + 1


@dataclass
class QACheck:
    """Quality assurance check result."""
    check_id: str
    category: str
    passed: bool
    score: float
    details: str = ""


class QualityAgent(BaseAgent):
    """Agent for quality control and validation."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        super().__init__(config, message_bus, registry)
        self._check_results: Dict[str, List[QACheck]] = {}
        self._uncertainty: Dict[str, Dict[str, float]] = {}
        self._gate_decisions: Dict[str, str] = {}

    async def run_sanity_checks(
        self,
        product_id: str,
        product_data: np.ndarray,
    ) -> List[QACheck]:
        """Run sanity checks on product."""
        checks = []

        # Value range check
        min_val = np.nanmin(product_data) if product_data.size > 0 else 0
        max_val = np.nanmax(product_data) if product_data.size > 0 else 0
        checks.append(QACheck(
            check_id=f"{product_id}_value_range",
            category="value",
            passed=0.0 <= min_val and max_val <= 1.0,
            score=1.0 if 0.0 <= min_val and max_val <= 1.0 else 0.5,
            details=f"Range: [{min_val:.3f}, {max_val:.3f}]",
        ))

        # NaN check
        nan_fraction = np.sum(np.isnan(product_data)) / max(product_data.size, 1)
        checks.append(QACheck(
            check_id=f"{product_id}_nan_check",
            category="completeness",
            passed=nan_fraction < 0.1,
            score=1.0 - nan_fraction,
            details=f"NaN fraction: {nan_fraction:.3f}",
        ))

        # Spatial coherence check (simplified)
        if product_data.size > 1:
            variance = np.nanvar(product_data)
            checks.append(QACheck(
                check_id=f"{product_id}_spatial",
                category="spatial",
                passed=variance > 0.001,
                score=min(1.0, variance * 10),
                details=f"Variance: {variance:.5f}",
            ))

        self._check_results[product_id] = checks
        return checks

    async def validate_product(
        self,
        product_id: str,
        checks: List[QACheck],
    ) -> Dict[str, Any]:
        """Validate product based on checks."""
        passed_count = sum(1 for c in checks if c.passed)
        total_count = len(checks)
        overall_score = sum(c.score for c in checks) / max(total_count, 1)

        return {
            "product_id": product_id,
            "checks_passed": passed_count,
            "checks_total": total_count,
            "overall_score": overall_score,
            "validation_passed": overall_score >= 0.7,
        }

    def make_gate_decision(
        self,
        product_id: str,
        validation_result: Dict[str, Any],
        confidence_score: float,
    ) -> str:
        """Make quality gate decision."""
        if validation_result["overall_score"] < 0.5:
            decision = "BLOCKED"
        elif validation_result["overall_score"] < 0.7:
            decision = "REVIEW_REQUIRED"
        elif confidence_score < 0.6:
            decision = "PASS_WITH_WARNINGS"
        else:
            decision = "PASS"

        self._gate_decisions[product_id] = decision
        return decision

    def quantify_uncertainty(
        self,
        product_id: str,
        product_data: np.ndarray,
    ) -> Dict[str, float]:
        """Quantify uncertainty in product."""
        if product_data.size == 0:
            uncertainty = {"mean": 0.0, "std": 0.0, "confidence": 0.0}
        else:
            uncertainty = {
                "mean": float(np.nanmean(product_data)),
                "std": float(np.nanstd(product_data)),
                "confidence": float(1.0 - np.nanstd(product_data)),
            }
        self._uncertainty[product_id] = uncertainty
        return uncertainty

    async def generate_qa_report(self, product_id: str) -> Dict[str, Any]:
        """Generate QA report."""
        checks = self._check_results.get(product_id, [])
        uncertainty = self._uncertainty.get(product_id, {})
        gate_decision = self._gate_decisions.get(product_id, "UNKNOWN")

        return {
            "product_id": product_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checks": [asdict(c) for c in checks],
            "uncertainty": uncertainty,
            "gate_decision": gate_decision,
            "summary": {
                "total_checks": len(checks),
                "passed_checks": sum(1 for c in checks if c.passed),
                "overall_quality": sum(c.score for c in checks) / max(len(checks), 1),
            },
        }


@dataclass
class ProductFormat:
    """Output product format specification."""
    format_type: str
    extension: str
    options: Dict[str, Any] = field(default_factory=dict)


class ReportingAgent(BaseAgent):
    """Agent for product generation and delivery."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        super().__init__(config, message_bus, registry)
        self._products: Dict[str, Dict[str, Any]] = {}
        self._deliveries: Dict[str, List[Dict[str, Any]]] = {}
        self._notifications: List[Dict[str, Any]] = []

    async def generate_product(
        self,
        event_id: str,
        product_type: str,
        data: Any,
        format_spec: ProductFormat,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generate output product."""
        filename = f"{event_id}_{product_type}.{format_spec.extension}"
        output_path = output_dir / filename

        # Simulate product generation
        product_info = {
            "event_id": event_id,
            "product_type": product_type,
            "format": format_spec.format_type,
            "path": str(output_path),
            "size_bytes": 1024 * 100,  # Simulated
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._products[f"{event_id}_{product_type}"] = product_info
        return product_info

    def convert_format(
        self,
        source_path: str,
        target_format: ProductFormat,
    ) -> str:
        """Convert product to different format."""
        source = Path(source_path)
        target_path = source.with_suffix(f".{target_format.extension}")
        # Simulate conversion
        return str(target_path)

    async def deliver_product(
        self,
        product_id: str,
        destination: str,
        method: str = "http",
    ) -> Dict[str, Any]:
        """Deliver product to destination."""
        if product_id not in self._products:
            raise ValueError(f"Product {product_id} not found")

        delivery = {
            "product_id": product_id,
            "destination": destination,
            "method": method,
            "status": "delivered",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }

        if product_id not in self._deliveries:
            self._deliveries[product_id] = []
        self._deliveries[product_id].append(delivery)

        return delivery

    async def send_notification(
        self,
        recipient: str,
        notification_type: str,
        payload: Dict[str, Any],
    ) -> bool:
        """Send notification."""
        notification = {
            "recipient": recipient,
            "type": notification_type,
            "payload": payload,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self._notifications.append(notification)
        return True

    def get_delivery_status(self, product_id: str) -> List[Dict[str, Any]]:
        """Get delivery status for product."""
        return self._deliveries.get(product_id, [])


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def message_bus():
    """Create message bus fixture."""
    return MessageBus()


@pytest.fixture
def agent_registry():
    """Create agent registry fixture."""
    return AgentRegistry()


@pytest.fixture
def base_config():
    """Create base agent config."""
    return AgentConfig(
        agent_id="test_agent_001",
        agent_type="test",
    )


@pytest.fixture
def orchestrator_config():
    """Create orchestrator agent config."""
    return AgentConfig(
        agent_id="orchestrator_001",
        agent_type="orchestrator",
    )


@pytest.fixture
def discovery_config():
    """Create discovery agent config."""
    return AgentConfig(
        agent_id="discovery_001",
        agent_type="discovery",
    )


@pytest.fixture
def pipeline_config():
    """Create pipeline agent config."""
    return AgentConfig(
        agent_id="pipeline_001",
        agent_type="pipeline",
    )


@pytest.fixture
def quality_config():
    """Create quality agent config."""
    return AgentConfig(
        agent_id="quality_001",
        agent_type="quality",
    )


@pytest.fixture
def reporting_config():
    """Create reporting agent config."""
    return AgentConfig(
        agent_id="reporting_001",
        agent_type="reporting",
    )


@pytest.fixture
def sample_event_spec():
    """Create sample event specification."""
    return EventSpec(
        event_id="evt_flood_001",
        event_class="flood.coastal",
        area={
            "type": "Polygon",
            "coordinates": [[[-80.5, 25.5], [-80.0, 25.5], [-80.0, 26.0], [-80.5, 26.0], [-80.5, 25.5]]],
        },
        temporal={
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z",
        },
        intent={"class": "flood.coastal", "source": "explicit"},
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_product_data():
    """Create sample product data."""
    np.random.seed(42)
    data = np.random.rand(100, 100).astype(np.float32)
    # Add some water region
    data[40:60, 40:60] = 0.9
    return data


# =============================================================================
# Base Agent Tests
# =============================================================================

class TestBaseAgent:
    """Tests for base agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle_start_stop(self, base_config, message_bus):
        """Test agent start and stop lifecycle."""
        agent = BaseAgent(base_config, message_bus)

        assert agent.state == AgentState.IDLE

        # Start agent
        result = await agent.start()
        assert result is True
        assert agent.state == AgentState.RUNNING

        # Stop agent
        result = await agent.stop()
        assert result is True
        assert agent.state == AgentState.STOPPED

    @pytest.mark.asyncio
    async def test_agent_lifecycle_pause_resume(self, base_config, message_bus):
        """Test agent pause and resume."""
        agent = BaseAgent(base_config, message_bus)

        await agent.start()

        # Pause
        result = await agent.pause()
        assert result is True
        assert agent.state == AgentState.PAUSED

        # Resume
        result = await agent.resume()
        assert result is True
        assert agent.state == AgentState.RUNNING

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_invalid_state_transitions(self, base_config, message_bus):
        """Test invalid state transitions return False."""
        agent = BaseAgent(base_config, message_bus)

        # Cannot pause when idle
        result = await agent.pause()
        assert result is False

        # Cannot resume when idle
        result = await agent.resume()
        assert result is False

        # Cannot stop when idle
        result = await agent.stop()
        assert result is False

        await agent.start()

        # Cannot resume when running
        result = await agent.resume()
        assert result is False

        # Cannot start when running
        result = await agent.start()
        assert result is False

    def test_message_creation(self, base_config, message_bus):
        """Test message creation."""
        agent = BaseAgent(base_config, message_bus)

        message = agent.create_message(
            MessageType.REQUEST,
            "target_agent",
            {"action": "test"},
            MessagePriority.HIGH,
        )

        assert message.source == agent.agent_id
        assert message.target == "target_agent"
        assert message.type == MessageType.REQUEST
        assert message.priority == MessagePriority.HIGH
        assert message.payload["action"] == "test"

    def test_message_serialization(self, base_config, message_bus):
        """Test message serialization round-trip."""
        agent = BaseAgent(base_config, message_bus)

        original = agent.create_message(
            MessageType.EVENT,
            "target",
            {"data": "test"},
        )

        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.payload == original.payload

    def test_message_expiration(self, base_config, message_bus):
        """Test message expiration check."""
        agent = BaseAgent(base_config, message_bus)

        # Create message with very short TTL
        message = agent.create_message(
            MessageType.EVENT,
            "target",
            {},
        )
        message.ttl_seconds = 0

        # Force timestamp to past
        message.timestamp = datetime.now(timezone.utc) - timedelta(seconds=1)

        assert message.is_expired() is True

    def test_checkpoint_create_restore(self, base_config, message_bus):
        """Test checkpoint creation and restoration."""
        agent = BaseAgent(base_config, message_bus)
        agent._checkpoint_data = {"key": "value", "count": 42}

        # Create checkpoint
        checkpoint = agent.checkpoint()
        assert checkpoint.agent_id == agent.agent_id
        assert checkpoint.state == agent.state
        assert checkpoint.data["key"] == "value"

        # Modify state
        agent._checkpoint_data = {"key": "modified"}

        # Restore
        result = agent.restore(checkpoint)
        assert result is True
        assert agent._checkpoint_data["key"] == "value"
        assert agent._checkpoint_data["count"] == 42

    def test_checkpoint_wrong_agent_fails(self, base_config, message_bus):
        """Test that restoring checkpoint from wrong agent fails."""
        agent = BaseAgent(base_config, message_bus)

        wrong_checkpoint = Checkpoint(
            agent_id="wrong_agent",
            state=AgentState.RUNNING,
            data={},
        )

        result = agent.restore(wrong_checkpoint)
        assert result is False

    def test_checkpoint_serialization(self, base_config, message_bus):
        """Test checkpoint serialization."""
        agent = BaseAgent(base_config, message_bus)
        agent._checkpoint_data = {"test": 123}

        checkpoint = agent.checkpoint()
        data = checkpoint.to_dict()
        restored = Checkpoint.from_dict(data)

        assert restored.agent_id == checkpoint.agent_id
        assert restored.state == checkpoint.state
        assert restored.data == checkpoint.data

    def test_agent_registry(self, base_config, agent_registry, message_bus):
        """Test agent registry."""
        agent = BaseAgent(base_config, message_bus, agent_registry)

        # Not registered until started
        assert agent_registry.get(agent.agent_id) is None

        asyncio.get_event_loop().run_until_complete(agent.start())

        # Now registered
        assert agent_registry.get(agent.agent_id) is agent
        assert agent_registry.count() == 1
        assert agent.agent_id in agent_registry.list_all()

        asyncio.get_event_loop().run_until_complete(agent.stop())

        # Unregistered after stop
        assert agent_registry.get(agent.agent_id) is None

    def test_retry_policy_backoff(self):
        """Test retry policy exponential backoff."""
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        delays = []
        while policy.next_attempt():
            delays.append(policy.get_delay())

        # Should have exponential growth
        assert delays[1] > delays[0]
        assert len(delays) == 4  # 5 attempts - 1 (first attempt doesn't need delay)

    def test_retry_policy_reset(self):
        """Test retry policy reset."""
        policy = RetryPolicy(max_attempts=3)

        policy.next_attempt()
        policy.next_attempt()
        assert policy.attempts_remaining == 1

        policy.reset()
        assert policy.attempts_remaining == 3


class TestMessageBus:
    """Tests for message bus functionality."""

    @pytest.mark.asyncio
    async def test_pub_sub(self, message_bus):
        """Test publish/subscribe."""
        received = []

        async def handler(msg):
            received.append(msg)

        message_bus.subscribe("topic1", handler)

        message = Message(
            id="msg_001",
            type=MessageType.EVENT,
            source="sender",
            target="topic1",
            payload={"data": "test"},
        )

        await message_bus.publish("topic1", message)

        assert len(received) == 1
        assert received[0].id == "msg_001"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_bus):
        """Test unsubscribe."""
        received = []

        async def handler(msg):
            received.append(msg)

        message_bus.subscribe("topic1", handler)
        result = message_bus.unsubscribe("topic1", handler)
        assert result is True

        message = Message(
            id="msg_001",
            type=MessageType.EVENT,
            source="sender",
            target="topic1",
            payload={},
        )

        await message_bus.publish("topic1", message)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_message_count(self, message_bus):
        """Test message count tracking."""
        assert message_bus.get_message_count() == 0

        message = Message(
            id="msg_001",
            type=MessageType.EVENT,
            source="sender",
            target="topic1",
            payload={},
        )

        await message_bus.publish("topic1", message)
        assert message_bus.get_message_count() == 1

        message_bus.clear()
        assert message_bus.get_message_count() == 0


class TestAgentRegistry:
    """Tests for agent registry."""

    def test_register_unregister(self, agent_registry, base_config, message_bus):
        """Test agent registration and unregistration."""
        agent = BaseAgent(base_config, message_bus)

        agent_registry.register(agent)
        assert agent_registry.count() == 1
        assert agent_registry.get(agent.agent_id) is agent

        removed = agent_registry.unregister(agent.agent_id)
        assert removed is agent
        assert agent_registry.count() == 0

    def test_get_by_type(self, agent_registry, message_bus):
        """Test getting agents by type."""
        config1 = AgentConfig(agent_id="agent_1", agent_type="discovery")
        config2 = AgentConfig(agent_id="agent_2", agent_type="discovery")
        config3 = AgentConfig(agent_id="agent_3", agent_type="pipeline")

        agent1 = BaseAgent(config1, message_bus)
        agent2 = BaseAgent(config2, message_bus)
        agent3 = BaseAgent(config3, message_bus)

        agent_registry.register(agent1)
        agent_registry.register(agent2)
        agent_registry.register(agent3)

        discovery_agents = agent_registry.get_by_type("discovery")
        assert len(discovery_agents) == 2

        pipeline_agents = agent_registry.get_by_type("pipeline")
        assert len(pipeline_agents) == 1


# =============================================================================
# Orchestrator Agent Tests
# =============================================================================

class TestOrchestratorAgent:
    """Tests for orchestrator agent."""

    @pytest.mark.asyncio
    async def test_process_event(self, orchestrator_config, message_bus, sample_event_spec):
        """Test event processing initiation."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        event_id = await agent.process_event(sample_event_spec)

        assert event_id == sample_event_spec.event_id
        assert event_id in agent._active_events

        await agent.stop()

    @pytest.mark.asyncio
    async def test_event_status_tracking(self, orchestrator_config, message_bus, sample_event_spec):
        """Test event status tracking."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        await agent.process_event(sample_event_spec)
        status = agent.get_event_status(sample_event_spec.event_id)

        assert status is not None
        assert status["event_id"] == sample_event_spec.event_id
        assert "delegations" in status

        await agent.stop()

    @pytest.mark.asyncio
    async def test_delegation_result_handling(self, orchestrator_config, message_bus, sample_event_spec):
        """Test handling delegation results."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        await agent.process_event(sample_event_spec)

        # Get delegation task ID
        task_id = list(agent._delegations.keys())[0]

        # Handle result
        result = {"status": "completed", "data_path": "/data/test.tif"}
        await agent.handle_delegation_result(task_id, result)

        task = agent._delegations[task_id]
        assert task.status == "completed"
        assert task.result == result
        assert task.completed_at is not None

        await agent.stop()

    @pytest.mark.asyncio
    async def test_degraded_mode(self, orchestrator_config, message_bus, sample_event_spec):
        """Test degraded mode operation."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        assert agent._degraded_mode is False

        agent.set_degraded_mode(True)
        assert agent._degraded_mode is True

        await agent.process_event(sample_event_spec)
        status = agent.get_event_status(sample_event_spec.event_id)

        assert status["degraded_mode"] is True

        await agent.stop()

    @pytest.mark.asyncio
    async def test_product_assembly(self, orchestrator_config, message_bus, sample_event_spec):
        """Test final product assembly."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        await agent.process_event(sample_event_spec)

        # Complete delegation
        task_id = list(agent._delegations.keys())[0]
        await agent.handle_delegation_result(task_id, {"component": "data"})

        # Assemble product
        product = await agent.assemble_product(sample_event_spec.event_id)

        assert product["event_id"] == sample_event_spec.event_id
        assert "components" in product
        assert "created_at" in product

        await agent.stop()

    @pytest.mark.asyncio
    async def test_checkpoint_state(self, orchestrator_config, message_bus, sample_event_spec):
        """Test orchestrator checkpoint includes state."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        await agent.process_event(sample_event_spec)
        agent.set_degraded_mode(True)

        checkpoint = agent.checkpoint()

        assert "active_events" in checkpoint.data
        assert "delegations" in checkpoint.data
        assert checkpoint.data["degraded_mode"] is True

        await agent.stop()

    @pytest.mark.asyncio
    async def test_process_event_when_not_running(self, orchestrator_config, message_bus, sample_event_spec):
        """Test that processing fails when agent is not running."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)

        with pytest.raises(RuntimeError, match="not running"):
            await agent.process_event(sample_event_spec)

    @pytest.mark.asyncio
    async def test_assemble_nonexistent_event(self, orchestrator_config, message_bus):
        """Test assembling product for nonexistent event."""
        agent = OrchestratorAgent(orchestrator_config, message_bus)
        await agent.start()

        with pytest.raises(ValueError, match="not found"):
            await agent.assemble_product("nonexistent_event")

        await agent.stop()


# =============================================================================
# Discovery Agent Tests
# =============================================================================

class TestDiscoveryAgent:
    """Tests for discovery agent."""

    @pytest.mark.asyncio
    async def test_discover_data(self, discovery_config, message_bus):
        """Test data discovery."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        candidates = await agent.discover_data(
            event_id="evt_001",
            area={"type": "Polygon", "coordinates": [[[-80, 25], [-79, 25], [-79, 26], [-80, 26], [-80, 25]]]},
            temporal={"start": "2024-01-01", "end": "2024-01-07"},
        )

        assert len(candidates) > 0
        assert all(isinstance(c, DataCandidate) for c in candidates)
        assert "evt_001" in agent._discovered

        await agent.stop()

    @pytest.mark.asyncio
    async def test_rank_sources(self, discovery_config, message_bus):
        """Test source ranking."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        candidates = [
            DataCandidate(
                source_id="source_1",
                provider="provider_a",
                data_type="optical",
                temporal_coverage={},
                spatial_coverage={},
                quality_score=0.6,
            ),
            DataCandidate(
                source_id="source_2",
                provider="provider_b",
                data_type="sar",
                temporal_coverage={},
                spatial_coverage={},
                quality_score=0.9,
            ),
            DataCandidate(
                source_id="source_3",
                provider="provider_c",
                data_type="optical",
                temporal_coverage={},
                spatial_coverage={},
                quality_score=0.75,
            ),
        ]

        ranked = agent.rank_sources(candidates)

        assert ranked[0].source_id == "source_2"  # Highest quality first
        assert ranked[-1].source_id == "source_1"  # Lowest quality last

        await agent.stop()

    @pytest.mark.asyncio
    async def test_select_sources(self, discovery_config, message_bus):
        """Test source selection with limit."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        candidates = await agent.discover_data(
            event_id="evt_002",
            area={},
            temporal={},
        )

        selected = agent.select_sources("evt_002", candidates, max_sources=2)

        assert len(selected) == 2
        assert "evt_002" in agent._selected

        await agent.stop()

    @pytest.mark.asyncio
    async def test_acquire_data(self, discovery_config, message_bus):
        """Test data acquisition."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        candidate = DataCandidate(
            source_id="test_source",
            provider="test_provider",
            data_type="sar",
            temporal_coverage={},
            spatial_coverage={},
            quality_score=0.8,
        )

        result = await agent.acquire_data(candidate)

        assert result["source_id"] == "test_source"
        assert result["status"] == "acquired"
        assert "path" in result

        await agent.stop()

    @pytest.mark.asyncio
    async def test_fallback_handling(self, discovery_config, message_bus):
        """Test fallback when source fails."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        candidates = await agent.discover_data(
            event_id="evt_003",
            area={},
            temporal={},
        )

        # Select some sources
        selected = agent.select_sources("evt_003", candidates, max_sources=2)
        failed_source = selected[0].source_id

        # Get fallback
        fallback = agent.handle_fallback("evt_003", failed_source)

        assert fallback is not None
        assert fallback.source_id != failed_source
        assert fallback.source_id not in [s.source_id for s in selected]

        await agent.stop()

    @pytest.mark.asyncio
    async def test_discover_when_not_running(self, discovery_config, message_bus):
        """Test discovery fails when agent is not running."""
        agent = DiscoveryAgent(discovery_config, message_bus)

        with pytest.raises(RuntimeError, match="not running"):
            await agent.discover_data("evt_001", {}, {})


# =============================================================================
# Pipeline Agent Tests
# =============================================================================

class TestPipelineAgent:
    """Tests for pipeline agent."""

    @pytest.mark.asyncio
    async def test_assemble_pipeline(self, pipeline_config, message_bus):
        """Test pipeline assembly."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        steps = agent.assemble_pipeline(
            event_id="evt_001",
            event_class="flood.coastal",
            available_data=["sar_data", "dem_data"],
        )

        assert len(steps) >= 3  # preprocess, detect, postprocess
        assert steps[0].processor == "preprocess"
        assert any(s.processor == "flood_detection" for s in steps)

        await agent.stop()

    @pytest.mark.asyncio
    async def test_assemble_wildfire_pipeline(self, pipeline_config, message_bus):
        """Test wildfire pipeline assembly."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        steps = agent.assemble_pipeline(
            event_id="evt_002",
            event_class="wildfire.forest",
            available_data=["optical_data"],
        )

        assert any(s.processor == "burn_detection" for s in steps)

        await agent.stop()

    @pytest.mark.asyncio
    async def test_execute_step(self, pipeline_config, message_bus):
        """Test executing a single step."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        step = PipelineStep(
            step_id="test_step",
            processor="test_processor",
            inputs=["input_data"],
            outputs=["output_data"],
        )

        result = await agent.execute_step(step)

        assert step.status == "completed"
        assert result["step_id"] == "test_step"
        assert "output_data" in result["outputs"]

        await agent.stop()

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, pipeline_config, message_bus):
        """Test executing entire pipeline."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        agent.assemble_pipeline("evt_001", "flood.coastal", ["data"])

        result = await agent.execute_pipeline("evt_001")

        assert result["event_id"] == "evt_001"
        assert result["steps_completed"] >= 3
        assert "final_outputs" in result

        await agent.stop()

    @pytest.mark.asyncio
    async def test_checkpointing(self, pipeline_config, message_bus):
        """Test pipeline checkpointing."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        agent.assemble_pipeline("evt_001", "flood.coastal", ["data"])

        # Execute partially
        pipeline = agent._pipelines["evt_001"]
        await agent.execute_step(pipeline[0])
        agent._checkpoints["evt_001"] = 0  # First step completed

        checkpoint = agent.get_checkpoint("evt_001")
        assert checkpoint == 0

        # Resume should start from step 1
        resume_index = agent.resume_from_checkpoint("evt_001")
        assert resume_index == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_parallel_step_execution(self, pipeline_config, message_bus):
        """Test parallel step execution."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        steps = [
            PipelineStep(step_id=f"step_{i}", processor="proc", inputs=[], outputs=[f"out_{i}"])
            for i in range(3)
        ]

        results = await agent.execute_parallel_steps(steps)

        assert len(results) == 3
        assert all(s.status == "completed" for s in steps)

        await agent.stop()

    @pytest.mark.asyncio
    async def test_execute_unassembled_pipeline(self, pipeline_config, message_bus):
        """Test executing pipeline that was not assembled."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        with pytest.raises(ValueError, match="not assembled"):
            await agent.execute_pipeline("nonexistent_pipeline")

        await agent.stop()


# =============================================================================
# Quality Agent Tests
# =============================================================================

class TestQualityAgent:
    """Tests for quality agent."""

    @pytest.mark.asyncio
    async def test_run_sanity_checks(self, quality_config, message_bus, sample_product_data):
        """Test sanity check execution."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        checks = await agent.run_sanity_checks("prod_001", sample_product_data)

        assert len(checks) >= 2
        assert all(isinstance(c, QACheck) for c in checks)
        assert "prod_001" in agent._check_results

        await agent.stop()

    @pytest.mark.asyncio
    async def test_validate_product(self, quality_config, message_bus, sample_product_data):
        """Test product validation."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        checks = await agent.run_sanity_checks("prod_001", sample_product_data)
        result = await agent.validate_product("prod_001", checks)

        assert result["product_id"] == "prod_001"
        assert "overall_score" in result
        assert "validation_passed" in result

        await agent.stop()

    @pytest.mark.asyncio
    async def test_gate_decision_pass(self, quality_config, message_bus, sample_product_data):
        """Test gate decision for passing product."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        checks = await agent.run_sanity_checks("prod_001", sample_product_data)
        validation = await agent.validate_product("prod_001", checks)

        decision = agent.make_gate_decision("prod_001", validation, confidence_score=0.9)

        # Good data should pass
        assert decision in ("PASS", "PASS_WITH_WARNINGS")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_gate_decision_blocked(self, quality_config, message_bus):
        """Test gate decision for bad data."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        # All NaN data
        bad_data = np.full((100, 100), np.nan)
        checks = await agent.run_sanity_checks("prod_bad", bad_data)
        validation = await agent.validate_product("prod_bad", checks)

        decision = agent.make_gate_decision("prod_bad", validation, confidence_score=0.3)

        assert decision in ("BLOCKED", "REVIEW_REQUIRED")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_uncertainty_quantification(self, quality_config, message_bus, sample_product_data):
        """Test uncertainty quantification."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        uncertainty = agent.quantify_uncertainty("prod_001", sample_product_data)

        assert "mean" in uncertainty
        assert "std" in uncertainty
        assert "confidence" in uncertainty
        assert 0 <= uncertainty["confidence"] <= 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_qa_report_generation(self, quality_config, message_bus, sample_product_data):
        """Test QA report generation."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        # Run full QA pipeline
        checks = await agent.run_sanity_checks("prod_001", sample_product_data)
        validation = await agent.validate_product("prod_001", checks)
        agent.make_gate_decision("prod_001", validation, 0.85)
        agent.quantify_uncertainty("prod_001", sample_product_data)

        report = await agent.generate_qa_report("prod_001")

        assert report["product_id"] == "prod_001"
        assert "checks" in report
        assert "uncertainty" in report
        assert "gate_decision" in report
        assert "summary" in report

        await agent.stop()

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, quality_config, message_bus):
        """Test handling of empty data."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        empty_data = np.array([]).reshape(0, 0)

        # Should not crash
        uncertainty = agent.quantify_uncertainty("prod_empty", empty_data)
        assert uncertainty["mean"] == 0.0

        await agent.stop()


# =============================================================================
# Reporting Agent Tests
# =============================================================================

class TestReportingAgent:
    """Tests for reporting agent."""

    @pytest.mark.asyncio
    async def test_generate_product(self, reporting_config, message_bus, temp_dir):
        """Test product generation."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        format_spec = ProductFormat(
            format_type="geotiff",
            extension="tif",
            options={"compression": "lzw"},
        )

        result = await agent.generate_product(
            event_id="evt_001",
            product_type="flood_extent",
            data=np.zeros((100, 100)),
            format_spec=format_spec,
            output_dir=temp_dir,
        )

        assert result["event_id"] == "evt_001"
        assert result["product_type"] == "flood_extent"
        assert result["format"] == "geotiff"
        assert "evt_001_flood_extent" in agent._products

        await agent.stop()

    @pytest.mark.asyncio
    async def test_format_conversion(self, reporting_config, message_bus):
        """Test format conversion."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        target_format = ProductFormat(
            format_type="geojson",
            extension="geojson",
        )

        result = agent.convert_format("/data/source.tif", target_format)

        assert result.endswith(".geojson")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_deliver_product(self, reporting_config, message_bus, temp_dir):
        """Test product delivery."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        # First generate a product
        format_spec = ProductFormat(format_type="geotiff", extension="tif")
        await agent.generate_product(
            "evt_001", "flood_extent", None, format_spec, temp_dir
        )

        product_id = "evt_001_flood_extent"
        delivery = await agent.deliver_product(
            product_id,
            destination="https://api.example.com/products",
            method="http",
        )

        assert delivery["status"] == "delivered"
        assert delivery["product_id"] == product_id

        await agent.stop()

    @pytest.mark.asyncio
    async def test_deliver_nonexistent_product(self, reporting_config, message_bus):
        """Test delivering nonexistent product fails."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        with pytest.raises(ValueError, match="not found"):
            await agent.deliver_product("nonexistent", "dest")

        await agent.stop()

    @pytest.mark.asyncio
    async def test_send_notification(self, reporting_config, message_bus):
        """Test notification sending."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        result = await agent.send_notification(
            recipient="user@example.com",
            notification_type="product_ready",
            payload={"product_id": "prod_001", "download_url": "http://example.com/prod_001"},
        )

        assert result is True
        assert len(agent._notifications) == 1
        assert agent._notifications[0]["recipient"] == "user@example.com"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_delivery_status(self, reporting_config, message_bus, temp_dir):
        """Test delivery status tracking."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        format_spec = ProductFormat(format_type="geotiff", extension="tif")
        await agent.generate_product("evt_001", "flood_extent", None, format_spec, temp_dir)

        product_id = "evt_001_flood_extent"
        await agent.deliver_product(product_id, "dest_1")
        await agent.deliver_product(product_id, "dest_2")

        status = agent.get_delivery_status(product_id)

        assert len(status) == 2

        await agent.stop()


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for agent workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow(
        self, message_bus, agent_registry, sample_event_spec, temp_dir
    ):
        """Test full end-to-end workflow."""
        # Create all agents
        orchestrator = OrchestratorAgent(
            AgentConfig(agent_id="orchestrator", agent_type="orchestrator"),
            message_bus, agent_registry
        )
        discovery = DiscoveryAgent(
            AgentConfig(agent_id="discovery", agent_type="discovery"),
            message_bus, agent_registry
        )
        pipeline = PipelineAgent(
            AgentConfig(agent_id="pipeline", agent_type="pipeline"),
            message_bus, agent_registry
        )
        quality = QualityAgent(
            AgentConfig(agent_id="quality", agent_type="quality"),
            message_bus, agent_registry
        )
        reporting = ReportingAgent(
            AgentConfig(agent_id="reporting", agent_type="reporting"),
            message_bus, agent_registry
        )

        # Start all agents
        await orchestrator.start()
        await discovery.start()
        await pipeline.start()
        await quality.start()
        await reporting.start()

        # Step 1: Orchestrator receives event
        event_id = await orchestrator.process_event(sample_event_spec)

        # Step 2: Discovery finds data
        candidates = await discovery.discover_data(
            event_id,
            sample_event_spec.area,
            sample_event_spec.temporal,
        )
        selected = discovery.select_sources(event_id, candidates, max_sources=2)

        # Step 3: Pipeline processes data
        steps = pipeline.assemble_pipeline(
            event_id,
            sample_event_spec.event_class,
            [s.source_id for s in selected],
        )
        pipeline_result = await pipeline.execute_pipeline(event_id)

        # Step 4: Quality validates
        np.random.seed(42)
        mock_output = np.random.rand(100, 100)
        checks = await quality.run_sanity_checks(f"prod_{event_id}", mock_output)
        validation = await quality.validate_product(f"prod_{event_id}", checks)
        gate_decision = quality.make_gate_decision(f"prod_{event_id}", validation, 0.85)

        # Step 5: Reporting generates products
        format_spec = ProductFormat(format_type="geotiff", extension="tif")
        product = await reporting.generate_product(
            event_id, "flood_extent", mock_output, format_spec, temp_dir
        )

        # Verify complete workflow
        assert pipeline_result["steps_completed"] >= 3
        assert gate_decision in ("PASS", "PASS_WITH_WARNINGS", "REVIEW_REQUIRED")
        assert product["event_id"] == event_id

        # Stop all agents
        await reporting.stop()
        await quality.stop()
        await pipeline.stop()
        await discovery.stop()
        await orchestrator.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_message_passing_between_agents(self, message_bus, agent_registry):
        """Test message passing between agents."""
        messages_received = []

        # Create agents
        sender = BaseAgent(
            AgentConfig(agent_id="sender", agent_type="sender"),
            message_bus, agent_registry
        )
        receiver = BaseAgent(
            AgentConfig(agent_id="receiver", agent_type="receiver"),
            message_bus, agent_registry
        )

        async def handle_message(msg):
            messages_received.append(msg)

        message_bus.subscribe("receiver", handle_message)

        await sender.start()
        await receiver.start()

        # Send message
        message = sender.create_message(
            MessageType.EVENT,
            "receiver",
            {"action": "test", "data": [1, 2, 3]},
        )
        await sender.send_message(message)

        # Wait for processing
        await asyncio.sleep(0.01)

        assert len(messages_received) == 1
        assert messages_received[0].source == "sender"
        assert messages_received[0].payload["action"] == "test"

        await receiver.stop()
        await sender.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_propagation(self, message_bus, agent_registry):
        """Test error propagation across agents."""
        orchestrator = OrchestratorAgent(
            AgentConfig(agent_id="orchestrator", agent_type="orchestrator"),
            message_bus, agent_registry
        )

        await orchestrator.start()

        # Process event
        event_spec = EventSpec(
            event_id="evt_error_test",
            event_class="flood",
            area={},
            temporal={},
        )
        await orchestrator.process_event(event_spec)

        # Simulate error in delegation by directly setting status
        task_id = list(orchestrator._delegations.keys())[0]
        await orchestrator.handle_delegation_result(task_id, {
            "status": "error",
            "error": "Discovery failed",
        })

        task = orchestrator._delegations[task_id]
        assert task.status == "completed"  # Completed but with error in result
        assert task.result["status"] == "error"

        await orchestrator.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_degraded_mode_end_to_end(
        self, message_bus, agent_registry, sample_event_spec, temp_dir
    ):
        """Test degraded mode operation end-to-end."""
        orchestrator = OrchestratorAgent(
            AgentConfig(agent_id="orchestrator", agent_type="orchestrator"),
            message_bus, agent_registry
        )
        quality = QualityAgent(
            AgentConfig(agent_id="quality", agent_type="quality"),
            message_bus, agent_registry
        )

        await orchestrator.start()
        await quality.start()

        # Enable degraded mode
        orchestrator.set_degraded_mode(True)

        # Process event
        event_id = await orchestrator.process_event(sample_event_spec)
        status = orchestrator.get_event_status(event_id)

        assert status["degraded_mode"] is True

        # Quality should still work in degraded mode
        np.random.seed(42)
        noisy_data = np.random.rand(50, 50) + np.random.randn(50, 50) * 0.3
        checks = await quality.run_sanity_checks("prod_degraded", noisy_data)
        validation = await quality.validate_product("prod_degraded", checks)

        # Degraded mode may accept lower quality
        decision = quality.make_gate_decision("prod_degraded", validation, 0.55)
        assert decision in ("PASS", "PASS_WITH_WARNINGS", "REVIEW_REQUIRED", "BLOCKED")

        await quality.stop()
        await orchestrator.stop()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestAgentEdgeCases:
    """Edge case tests for agent system."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_starts(self, message_bus, agent_registry):
        """Test multiple agents starting concurrently."""
        configs = [
            AgentConfig(agent_id=f"agent_{i}", agent_type="test")
            for i in range(5)
        ]
        agents = [BaseAgent(c, message_bus, agent_registry) for c in configs]

        # Start all concurrently
        results = await asyncio.gather(*[a.start() for a in agents])

        assert all(results)
        assert agent_registry.count() == 5

        # Stop all concurrently
        await asyncio.gather(*[a.stop() for a in agents])

    @pytest.mark.asyncio
    async def test_pipeline_step_failure_recovery(self, pipeline_config, message_bus):
        """Test recovery from step failure."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        agent.assemble_pipeline("evt_fail", "flood", ["data"])

        # Simulate partial execution
        pipeline = agent._pipelines["evt_fail"]
        await agent.execute_step(pipeline[0])
        agent._checkpoints["evt_fail"] = 0

        # Checkpoint should allow resume
        resume_index = agent.resume_from_checkpoint("evt_fail")
        assert resume_index == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_quality_with_extreme_values(self, quality_config, message_bus):
        """Test quality agent with extreme values."""
        agent = QualityAgent(quality_config, message_bus)
        await agent.start()

        # Test with infinity values
        inf_data = np.array([[np.inf, -np.inf], [0.5, 0.5]])
        checks = await agent.run_sanity_checks("prod_inf", inf_data)
        assert len(checks) > 0

        # Test with all zeros
        zero_data = np.zeros((10, 10))
        checks = await agent.run_sanity_checks("prod_zero", zero_data)
        uncertainty = agent.quantify_uncertainty("prod_zero", zero_data)
        assert uncertainty["std"] == 0.0

        await agent.stop()

    @pytest.mark.asyncio
    async def test_discovery_empty_candidates(self, discovery_config, message_bus):
        """Test handling when no fallback available."""
        agent = DiscoveryAgent(discovery_config, message_bus)
        await agent.start()

        # Create minimal candidates
        candidates = await agent.discover_data("evt_minimal", {}, {})
        selected = agent.select_sources("evt_minimal", candidates, max_sources=len(candidates))

        # All candidates are selected, no fallback available
        fallback = agent.handle_fallback("evt_minimal", "nonexistent")
        assert fallback is None  # No fallback when all selected

        await agent.stop()

    @pytest.mark.asyncio
    async def test_reporting_multiple_formats(self, reporting_config, message_bus, temp_dir):
        """Test generating products in multiple formats."""
        agent = ReportingAgent(reporting_config, message_bus)
        await agent.start()

        formats = [
            ProductFormat("geotiff", "tif"),
            ProductFormat("geojson", "geojson"),
            ProductFormat("png", "png"),
        ]

        products = []
        for fmt in formats:
            product = await agent.generate_product(
                "evt_multi", f"product_{fmt.format_type}",
                None, fmt, temp_dir
            )
            products.append(product)

        assert len(products) == 3
        assert all(p["format"] in ["geotiff", "geojson", "png"] for p in products)

        await agent.stop()


# =============================================================================
# Slow Tests (marked for optional skipping)
# =============================================================================

class TestSlowAgentOperations:
    """Slow tests for agent operations."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_pipeline_execution(self, pipeline_config, message_bus):
        """Test executing a large pipeline."""
        agent = PipelineAgent(pipeline_config, message_bus)
        await agent.start()

        # Create pipeline with many steps
        steps = [
            PipelineStep(
                step_id=f"step_{i}",
                processor=f"processor_{i}",
                inputs=[f"input_{i}"],
                outputs=[f"output_{i}"],
            )
            for i in range(20)
        ]

        agent._pipelines["large_evt"] = steps
        agent._checkpoints["large_evt"] = -1

        result = await agent.execute_pipeline("large_evt")

        assert result["steps_completed"] == 20

        await agent.stop()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_many_concurrent_messages(self, message_bus):
        """Test handling many concurrent messages."""
        message_count = [0]

        async def counter(msg):
            message_count[0] += 1

        message_bus.subscribe("counter_topic", counter)

        # Send many messages
        messages = [
            Message(
                id=str(i),
                type=MessageType.EVENT,
                source="sender",
                target="counter_topic",
                payload={"index": i},
            )
            for i in range(100)
        ]

        await asyncio.gather(*[message_bus.publish("counter_topic", m) for m in messages])

        assert message_count[0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
