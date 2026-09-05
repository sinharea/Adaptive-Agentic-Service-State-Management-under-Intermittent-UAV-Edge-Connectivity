"""
Edge Node resource model.

Represents a UAV-mounted or ground-based edge computing node with
CPU, memory, and storage resources. Tracks utilization over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeNode:
    """An edge computing node with trackable resource utilization.

    Attributes:
        node_id: Unique identifier for this node.
        cpu_capacity: Total CPU capacity in GHz.
        memory_capacity: Total memory in MB.
        storage_capacity: Total storage in MB.
        cpu_used: Currently used CPU in GHz.
        memory_used: Currently used memory in MB.
        storage_used: Currently used storage in MB.
    """

    node_id: str
    cpu_capacity: float        # GHz
    memory_capacity: float     # MB
    storage_capacity: float    # MB
    cpu_used: float = 0.0
    memory_used: float = 0.0
    storage_used: float = 0.0

    # ------------------------------------------------------------------ #
    # Utilization queries
    # ------------------------------------------------------------------ #

    @property
    def cpu_utilization(self) -> float:
        """Return CPU utilization as a fraction in [0, 1]."""
        if self.cpu_capacity <= 0:
            return 1.0
        return min(self.cpu_used / self.cpu_capacity, 1.0)

    @property
    def memory_utilization(self) -> float:
        """Return memory utilization as a fraction in [0, 1]."""
        if self.memory_capacity <= 0:
            return 1.0
        return min(self.memory_used / self.memory_capacity, 1.0)

    @property
    def storage_utilization(self) -> float:
        """Return storage utilization as a fraction in [0, 1]."""
        if self.storage_capacity <= 0:
            return 1.0
        return min(self.storage_used / self.storage_capacity, 1.0)

    # ------------------------------------------------------------------ #
    # Resource allocation
    # ------------------------------------------------------------------ #

    def available_cpu(self) -> float:
        """Return available CPU in GHz."""
        return max(self.cpu_capacity - self.cpu_used, 0.0)

    def available_memory(self) -> float:
        """Return available memory in MB."""
        return max(self.memory_capacity - self.memory_used, 0.0)

    def available_storage(self) -> float:
        """Return available storage in MB."""
        return max(self.storage_capacity - self.storage_used, 0.0)

    def allocate_cpu(self, amount: float) -> bool:
        """Allocate CPU. Returns True if successful, False if insufficient."""
        if amount < 0:
            raise ValueError(f"Cannot allocate negative CPU: {amount}")
        if self.cpu_used + amount > self.cpu_capacity:
            return False
        self.cpu_used += amount
        return True

    def release_cpu(self, amount: float) -> None:
        """Release previously allocated CPU."""
        if amount < 0:
            raise ValueError(f"Cannot release negative CPU: {amount}")
        self.cpu_used = max(self.cpu_used - amount, 0.0)

    def allocate_memory(self, amount: float) -> bool:
        """Allocate memory. Returns True if successful."""
        if amount < 0:
            raise ValueError(f"Cannot allocate negative memory: {amount}")
        if self.memory_used + amount > self.memory_capacity:
            return False
        self.memory_used += amount
        return True

    def release_memory(self, amount: float) -> None:
        """Release previously allocated memory."""
        if amount < 0:
            raise ValueError(f"Cannot release negative memory: {amount}")
        self.memory_used = max(self.memory_used - amount, 0.0)

    def allocate_storage(self, amount: float) -> bool:
        """Allocate storage. Returns True if successful."""
        if amount < 0:
            raise ValueError(f"Cannot allocate negative storage: {amount}")
        if self.storage_used + amount > self.storage_capacity:
            return False
        self.storage_used += amount
        return True

    def release_storage(self, amount: float) -> None:
        """Release previously allocated storage."""
        if amount < 0:
            raise ValueError(f"Cannot release negative storage: {amount}")
        self.storage_used = max(self.storage_used - amount, 0.0)

    def can_host_service(self, cpu_needed: float, memory_needed: float,
                         storage_needed: float) -> bool:
        """Check if the node has enough resources to host a service."""
        return (self.available_cpu() >= cpu_needed
                and self.available_memory() >= memory_needed
                and self.available_storage() >= storage_needed)

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def set_initial_utilization(self, cpu_frac: float, memory_frac: float,
                                storage_frac: float) -> None:
        """Set initial utilization from fractions (0-1)."""
        self.cpu_used = self.cpu_capacity * max(0.0, min(cpu_frac, 1.0))
        self.memory_used = self.memory_capacity * max(0.0, min(memory_frac, 1.0))
        self.storage_used = self.storage_capacity * max(0.0, min(storage_frac, 1.0))

    def __repr__(self) -> str:
        return (
            f"EdgeNode(id={self.node_id}, "
            f"cpu={self.cpu_used:.1f}/{self.cpu_capacity:.1f}GHz, "
            f"mem={self.memory_used:.0f}/{self.memory_capacity:.0f}MB, "
            f"stor={self.storage_used:.0f}/{self.storage_capacity:.0f}MB)"
        )
