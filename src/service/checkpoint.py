"""
Checkpoint management for stateful services.

Creates, stores, and manages checkpoints of service state.
Tracks checkpoint costs (CPU, storage, time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from src.service.state import ServiceState
from src.environment.edge_node import EdgeNode


@dataclass
class Checkpoint:
    """A stored checkpoint of service state.

    Attributes:
        checkpoint_id: Unique identifier.
        service_id: Service this checkpoint belongs to.
        node_id: Node where the checkpoint is stored.
        version: State version at time of checkpoint.
        size: Size of checkpoint in MB.
        creation_step: Simulation step when created.
    """
    checkpoint_id: str
    service_id: str
    node_id: str
    version: int
    size: float           # MB
    creation_step: int


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation."""
    success: bool
    checkpoint: Optional[Checkpoint] = None
    cpu_cost: float = 0.0       # GHz consumed
    storage_cost: float = 0.0   # MB consumed
    time_cost: float = 0.0      # steps (or fractional)
    reason: str = ""


class CheckpointManager:
    """Manages checkpoints for services on edge nodes.

    Parameters are loaded from the configuration dict.
    """

    def __init__(self, config: dict):
        ckpt_cfg = config["checkpoint"]
        self.enabled = ckpt_cfg.get("enabled", True)
        self.interval = ckpt_cfg.get("interval", 50)
        self.cpu_overhead_fraction = ckpt_cfg.get("cpu_overhead_fraction", 0.1)
        self.storage_multiplier = ckpt_cfg.get("storage_multiplier", 1.1)
        self.max_checkpoints = ckpt_cfg.get("max_checkpoints", 3)

        # Storage: service_id -> list of Checkpoints (oldest first)
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------ #
    # Checkpoint creation
    # ------------------------------------------------------------------ #

    def create_checkpoint(self, service: ServiceState,
                          edge: EdgeNode, step: int) -> CheckpointResult:
        """Create a checkpoint of the service's current state.

        Args:
            service: The service to checkpoint.
            edge: The edge node hosting the service.
            step: Current simulation step.

        Returns:
            CheckpointResult with success/failure and cost metrics.
        """
        if not self.enabled:
            return CheckpointResult(success=False, reason="Checkpointing disabled")

        if not service.is_running:
            return CheckpointResult(success=False, reason="Service not running")

        if service.is_migrating:
            return CheckpointResult(success=False, reason="Service is migrating")

        # Calculate costs
        checkpoint_size = service.state_size * self.storage_multiplier
        cpu_needed = edge.cpu_capacity * self.cpu_overhead_fraction

        # Check resources
        if not edge.allocate_storage(checkpoint_size):
            return CheckpointResult(
                success=False,
                reason="Insufficient storage",
                storage_cost=checkpoint_size,
            )

        # CPU overhead is temporary (released after checkpoint completes)
        cpu_available = edge.available_cpu() >= cpu_needed

        if not cpu_available:
            edge.release_storage(checkpoint_size)
            return CheckpointResult(
                success=False,
                reason="Insufficient CPU",
                cpu_cost=cpu_needed,
            )

        # Create checkpoint
        ckpt_id = f"ckpt_{self._next_id}"
        self._next_id += 1

        checkpoint = Checkpoint(
            checkpoint_id=ckpt_id,
            service_id=service.service_id,
            node_id=service.current_node,
            version=service.state_version,
            size=checkpoint_size,
            creation_step=step,
        )

        # Store checkpoint, evict oldest if at max
        if service.service_id not in self._checkpoints:
            self._checkpoints[service.service_id] = []

        ckpts = self._checkpoints[service.service_id]
        if len(ckpts) >= self.max_checkpoints:
            evicted = ckpts.pop(0)
            # Release storage for evicted checkpoint
            edge.release_storage(evicted.size)

        ckpts.append(checkpoint)

        # Update service state
        service.checkpoint_version = service.state_version
        service.checkpoint_node = service.current_node

        # Time cost: proportional to state size (simplified)
        time_cost = checkpoint_size / 100.0  # rough: 100 MB/step

        return CheckpointResult(
            success=True,
            checkpoint=checkpoint,
            cpu_cost=cpu_needed,
            storage_cost=checkpoint_size,
            time_cost=time_cost,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint queries
    # ------------------------------------------------------------------ #

    def get_latest_checkpoint(self, service_id: str) -> Optional[Checkpoint]:
        """Return the most recent checkpoint for a service."""
        ckpts = self._checkpoints.get(service_id, [])
        return ckpts[-1] if ckpts else None

    def get_checkpoints(self, service_id: str) -> List[Checkpoint]:
        """Return all checkpoints for a service."""
        return list(self._checkpoints.get(service_id, []))

    def has_checkpoint(self, service_id: str) -> bool:
        """Check if any checkpoint exists for a service."""
        return bool(self._checkpoints.get(service_id))

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def remove_checkpoints(self, service_id: str, edge: EdgeNode) -> float:
        """Remove all checkpoints for a service, freeing storage.

        Returns total storage freed in MB.
        """
        ckpts = self._checkpoints.pop(service_id, [])
        total_freed = 0.0
        for ckpt in ckpts:
            if ckpt.node_id == edge.node_id:
                edge.release_storage(ckpt.size)
                total_freed += ckpt.size
        return total_freed
