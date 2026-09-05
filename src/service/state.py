"""
Stateful service model.

Represents a service running on an edge node with versioned state,
configurable size and update rate, and tracking of checkpoint/replica status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import time as _time


@dataclass
class ServiceState:
    """A stateful service running on a UAV edge node.

    Attributes:
        service_id: Unique identifier for this service.
        current_node: Node ID where the service is currently running.
        state_size: Current state size in MB.
        state_version: Monotonically increasing version counter.
        state_update_rate: Rate at which state grows (MB per second/step).
        checkpoint_version: Version of the latest checkpoint (-1 if none).
        checkpoint_node: Node where latest checkpoint is stored.
        replicated_nodes: List of node IDs holding replicas.
        replica_versions: Dict mapping node_id -> version of their replica.
        last_sync_time: Simulation step of last successful synchronization.
        priority: Service priority level ('low', 'medium', 'high').
        is_running: Whether the service is currently active.
        is_migrating: Whether a migration is in progress.
        interruption_start: Step when current interruption began (None if running).
        total_interruption_time: Accumulated interruption time (steps).
        total_state_loss: Accumulated state loss (in version deltas).
        migration_count: Number of completed migrations.
        sla_max_interruption: Maximum allowed interruption (steps) from SLA.
        sla_max_recovery_time: Maximum allowed recovery time (steps) from SLA.
        sla_violations: Count of SLA violations.
    """

    service_id: str
    current_node: str
    state_size: float                     # MB
    state_version: int = 0
    state_update_rate: float = 1.0        # MB/step
    checkpoint_version: int = -1
    checkpoint_node: Optional[str] = None
    replicated_nodes: List[str] = field(default_factory=list)
    replica_versions: dict = field(default_factory=dict)
    last_sync_time: int = 0
    priority: str = "medium"
    is_running: bool = True
    is_migrating: bool = False
    interruption_start: Optional[int] = None
    total_interruption_time: float = 0.0
    total_state_loss: int = 0
    migration_count: int = 0
    sla_max_interruption: float = 10.0
    sla_max_recovery_time: float = 30.0
    sla_violations: int = 0

    # Initial state size (for reference)
    _initial_state_size: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._initial_state_size = self.state_size

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #

    def update_state(self, step: int) -> None:
        """Advance the service state by one step.

        Increments version and grows state size according to update rate.
        Only updates if the service is running and not migrating.
        """
        if not self.is_running or self.is_migrating:
            return
        self.state_version += 1
        self.state_size += self.state_update_rate

    # ------------------------------------------------------------------ #
    # Interruption tracking
    # ------------------------------------------------------------------ #

    def begin_interruption(self, step: int) -> None:
        """Mark the service as interrupted."""
        if self.interruption_start is None:
            self.interruption_start = step
            self.is_running = False

    def end_interruption(self, step: int) -> float:
        """End the current interruption and return its duration.

        Returns:
            Duration of the interruption in steps, or 0 if not interrupted.
        """
        if self.interruption_start is None:
            return 0.0
        duration = step - self.interruption_start
        self.total_interruption_time += duration
        if duration > self.sla_max_interruption:
            self.sla_violations += 1
        self.interruption_start = None
        self.is_running = True
        return duration

    # ------------------------------------------------------------------ #
    # Priority helpers
    # ------------------------------------------------------------------ #

    @property
    def priority_weight(self) -> float:
        """Numeric weight for priority (higher = more important)."""
        weights = {"low": 0.5, "medium": 1.0, "high": 2.0}
        return weights.get(self.priority, 1.0)

    # ------------------------------------------------------------------ #
    # Checkpoint status
    # ------------------------------------------------------------------ #

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_version >= 0

    @property
    def checkpoint_staleness(self) -> int:
        """How many versions behind the checkpoint is."""
        if not self.has_checkpoint:
            return self.state_version
        return self.state_version - self.checkpoint_version

    # ------------------------------------------------------------------ #
    # Replica status
    # ------------------------------------------------------------------ #

    @property
    def num_replicas(self) -> int:
        return len(self.replicated_nodes)

    def best_replica_version(self) -> int:
        """Return the most recent replica version, or -1 if none."""
        if not self.replica_versions:
            return -1
        return max(self.replica_versions.values())

    @property
    def replica_staleness(self) -> int:
        """How many versions behind the best replica is."""
        best = self.best_replica_version()
        if best < 0:
            return self.state_version
        return self.state_version - best

    # ------------------------------------------------------------------ #
    # Observation vector (for agents)
    # ------------------------------------------------------------------ #

    def get_observation(self, current_step: int) -> dict:
        """Return a dict of observable features for the decision layer."""
        return {
            "state_size": self.state_size,
            "state_version": self.state_version,
            "state_update_rate": self.state_update_rate,
            "priority": self.priority,
            "priority_weight": self.priority_weight,
            "has_checkpoint": self.has_checkpoint,
            "checkpoint_staleness": self.checkpoint_staleness,
            "num_replicas": self.num_replicas,
            "replica_staleness": self.replica_staleness,
            "time_since_last_sync": current_step - self.last_sync_time,
            "is_running": self.is_running,
            "is_migrating": self.is_migrating,
        }

    def __repr__(self) -> str:
        status = "running" if self.is_running else "interrupted"
        if self.is_migrating:
            status = "migrating"
        return (
            f"ServiceState(id={self.service_id}, node={self.current_node}, "
            f"size={self.state_size:.1f}MB, v{self.state_version}, "
            f"status={status}, ckpt=v{self.checkpoint_version}, "
            f"replicas={self.num_replicas})"
        )


def create_service(config: dict, node_id: str,
                   service_id: str = "service_0") -> ServiceState:
    """Create a ServiceState from configuration.

    Args:
        config: Full configuration dict.
        node_id: ID of the node hosting this service.
        service_id: Unique service identifier.

    Returns:
        Initialized ServiceState.
    """
    svc_cfg = config["service"]
    return ServiceState(
        service_id=service_id,
        current_node=node_id,
        state_size=svc_cfg["state_size"],
        state_update_rate=svc_cfg["state_update_rate"],
        priority=svc_cfg["priority"],
        sla_max_interruption=svc_cfg["sla_max_interruption"],
        sla_max_recovery_time=svc_cfg["sla_max_recovery_time"],
    )
