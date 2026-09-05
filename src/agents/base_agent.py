"""
Base agent interface for state management decision-making.

All strategies (baselines and proposed) must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.service.state import ServiceState
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel, LinkMetrics


class Action(Enum):
    """Available actions for the state management agent."""
    LOCAL_EXECUTION = "local_execution"
    CHECKPOINT = "checkpoint"
    REPLICATE = "replicate"
    SYNCHRONIZE = "synchronize"
    MIGRATE = "migrate"
    DEGRADE = "degrade"
    RECOVER = "recover"


@dataclass
class Observation:
    """Environment observation for the agent.

    Consolidates all observable state into a single structure.
    """
    # Network
    bandwidth: float = 0.0
    latency: float = 0.0
    packet_loss: float = 0.0
    connectivity_state: str = "disconnected"
    predicted_contact_duration: float = 0.0
    num_neighbors: int = 0

    # UAV
    uav_x: float = 0.0
    uav_y: float = 0.0
    uav_velocity: float = 0.0
    uav_direction: float = 0.0

    # Edge resources
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0

    # Service
    state_size: float = 0.0
    state_version: int = 0
    state_update_rate: float = 0.0
    priority_weight: float = 1.0
    has_checkpoint: bool = False
    checkpoint_staleness: int = 0
    num_replicas: int = 0
    replica_staleness: int = 0
    time_since_sync: int = 0
    is_running: bool = True
    is_migrating: bool = False

    # Best neighbor info
    best_neighbor_id: Optional[str] = None
    best_neighbor_bandwidth: float = 0.0
    best_neighbor_contact_duration: float = 0.0

    def to_vector(self) -> List[float]:
        """Convert to numeric vector for RL/ML agents."""
        return [
            self.bandwidth, self.latency, self.packet_loss,
            1.0 if self.connectivity_state == "connected" else
            0.5 if self.connectivity_state == "degraded" else 0.0,
            min(self.predicted_contact_duration, 1000.0) / 1000.0,
            self.num_neighbors / 20.0,
            self.uav_velocity / 30.0,
            self.cpu_utilization, self.memory_utilization,
            self.storage_utilization,
            min(self.state_size, 1000.0) / 1000.0,
            self.state_update_rate / 10.0,
            self.priority_weight / 2.0,
            1.0 if self.has_checkpoint else 0.0,
            min(self.checkpoint_staleness, 100) / 100.0,
            self.num_replicas / 5.0,
            min(self.replica_staleness, 100) / 100.0,
            min(self.time_since_sync, 100) / 100.0,
            1.0 if self.is_running else 0.0,
            self.best_neighbor_bandwidth / 100.0,
            min(self.best_neighbor_contact_duration, 1000.0) / 1000.0,
        ]


@dataclass
class Decision:
    """Agent decision with metadata."""
    action: Action
    target_node: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""


def build_observation(service: ServiceState, source_uav: UAVNode,
                      all_uavs: List[UAVNode], network: NetworkModel,
                      current_step: int) -> Observation:
    """Build an Observation from current environment state.

    Args:
        service: The stateful service.
        source_uav: UAV hosting the service.
        all_uavs: All UAV nodes.
        network: Network model.
        current_step: Current simulation step.

    Returns:
        Populated Observation.
    """
    neighbors = source_uav.get_neighbors(all_uavs)
    best = network.get_best_neighbor(source_uav, all_uavs)

    obs = Observation(
        num_neighbors=len(neighbors),
        uav_x=source_uav.x,
        uav_y=source_uav.y,
        uav_velocity=source_uav.velocity,
        uav_direction=source_uav.direction,
        cpu_utilization=source_uav.edge.cpu_utilization,
        memory_utilization=source_uav.edge.memory_utilization,
        storage_utilization=source_uav.edge.storage_utilization,
        state_size=service.state_size,
        state_version=service.state_version,
        state_update_rate=service.state_update_rate,
        priority_weight=service.priority_weight,
        has_checkpoint=service.has_checkpoint,
        checkpoint_staleness=service.checkpoint_staleness,
        num_replicas=service.num_replicas,
        replica_staleness=service.replica_staleness,
        time_since_sync=current_step - service.last_sync_time,
        is_running=service.is_running,
        is_migrating=service.is_migrating,
    )

    if best:
        best_node, best_link = best
        obs.best_neighbor_id = best_node.node_id
        obs.best_neighbor_bandwidth = best_link.bandwidth
        obs.best_neighbor_contact_duration = best_link.predicted_contact_duration
        obs.bandwidth = best_link.bandwidth
        obs.latency = best_link.latency
        obs.packet_loss = best_link.packet_loss
        obs.connectivity_state = best_link.state.value
        obs.predicted_contact_duration = best_link.predicted_contact_duration

    return obs


class BaseAgent(ABC):
    """Abstract base class for state management agents."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.decision_count: int = 0
        self.action_history: List[Action] = []

    @abstractmethod
    def decide(self, observation: Observation, step: int) -> Decision:
        """Select an action based on the current observation.

        Args:
            observation: Current environment state.
            step: Current simulation step.

        Returns:
            Decision with selected action.
        """
        ...

    def record_outcome(self, observation: Observation, action: Action,
                       reward: float, next_observation: Observation) -> None:
        """Record the outcome of an action (for learning agents).

        Default implementation does nothing. Override in learning agents.
        """
        pass

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.decision_count = 0
        self.action_history = []
