"""
UAV node model.

Combines spatial position, velocity, mobility, and edge computing resources
into a single UAV entity used by the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from src.environment.edge_node import EdgeNode
from src.environment.mobility import MobilityModel


@dataclass
class UAVNode:
    """A UAV-mounted edge computing node.

    Attributes:
        node_id: Unique identifier.
        x: Current x-position (meters).
        y: Current y-position (meters).
        velocity: Current speed (m/s).
        direction: Current heading (radians, 0 = east, π/2 = north).
        communication_radius: Maximum communication range (meters).
        edge: The edge computing resources on this UAV.
        mobility_model: The mobility model driving this UAV's movement.
    """

    node_id: str
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    direction: float = 0.0
    communication_radius: float = 200.0
    edge: EdgeNode = field(default_factory=lambda: EdgeNode(
        node_id="default", cpu_capacity=4.0,
        memory_capacity=8192.0, storage_capacity=64000.0
    ))
    mobility_model: Optional[MobilityModel] = None

    # ------------------------------------------------------------------ #
    # Spatial operations
    # ------------------------------------------------------------------ #

    def distance_to(self, other: "UAVNode") -> float:
        """Euclidean distance to another UAV node."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def is_in_range(self, other: "UAVNode") -> bool:
        """Check if another UAV is within this UAV's communication radius."""
        return self.distance_to(other) <= self.communication_radius

    def get_neighbors(self, all_nodes: List["UAVNode"]) -> List["UAVNode"]:
        """Return all UAV nodes within communication range (excluding self)."""
        return [
            node for node in all_nodes
            if node.node_id != self.node_id and self.is_in_range(node)
        ]

    # ------------------------------------------------------------------ #
    # Mobility
    # ------------------------------------------------------------------ #

    def initialize_position(self, rng: np.random.Generator,
                            area_width: float, area_height: float) -> None:
        """Set initial position using the mobility model."""
        if self.mobility_model is None:
            raise RuntimeError(f"UAV {self.node_id} has no mobility model assigned.")
        self.x, self.y = self.mobility_model.initialize(rng, area_width, area_height)

    def update_position(self, dt: float, rng: np.random.Generator,
                        area_width: float, area_height: float) -> None:
        """Advance position by one time step using the mobility model."""
        if self.mobility_model is None:
            raise RuntimeError(f"UAV {self.node_id} has no mobility model assigned.")
        self.x, self.y, self.velocity, self.direction = (
            self.mobility_model.update(
                self.x, self.y, self.velocity, self.direction,
                dt, rng, area_width, area_height
            )
        )

    # ------------------------------------------------------------------ #
    # Contact duration prediction
    # ------------------------------------------------------------------ #

    def predicted_contact_duration(self, other: "UAVNode") -> float:
        """Estimate how long two UAVs will remain in communication range.

        Uses relative velocity and current distance to estimate the time
        until the nodes move out of range, assuming constant velocity and
        direction (a common simplification in mobile networking research).

        Returns:
            Estimated contact duration in seconds.
            Returns float('inf') if nodes are stationary relative to each other.
            Returns 0.0 if nodes are already out of range.
        """
        dist = self.distance_to(other)
        if dist > self.communication_radius:
            return 0.0

        # Relative velocity components
        vx_rel = (self.velocity * math.cos(self.direction)
                  - other.velocity * math.cos(other.direction))
        vy_rel = (self.velocity * math.sin(self.direction)
                  - other.velocity * math.sin(other.direction))
        v_rel = math.sqrt(vx_rel * vx_rel + vy_rel * vy_rel)

        if v_rel < 1e-6:
            # Essentially stationary relative to each other
            return float("inf")

        # Approximate: time for distance to grow from current to communication_radius
        remaining_range = self.communication_radius - dist
        return remaining_range / v_rel

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"UAVNode(id={self.node_id}, pos=({self.x:.1f}, {self.y:.1f}), "
            f"v={self.velocity:.1f}m/s, range={self.communication_radius:.0f}m, "
            f"edge={self.edge})"
        )


def create_uav_nodes(config: dict, mobility_model: MobilityModel,
                     rng: np.random.Generator) -> List[UAVNode]:
    """Factory function to create UAV nodes from configuration.

    Args:
        config: Full configuration dict.
        mobility_model: Base mobility model (will be cloned per UAV).
        rng: Random generator for initialization.

    Returns:
        List of initialized UAVNode instances.
    """
    uav_cfg = config["uav"]
    sim_cfg = config["simulation"]
    count = uav_cfg["count"]
    init_util = uav_cfg.get("initial_utilization", {})

    nodes = []
    for i in range(count):
        node_id = f"uav_{i}"

        # Create edge node with resources
        edge = EdgeNode(
            node_id=node_id,
            cpu_capacity=uav_cfg["cpu_capacity"],
            memory_capacity=uav_cfg["memory_capacity"],
            storage_capacity=uav_cfg["storage_capacity"],
        )
        edge.set_initial_utilization(
            cpu_frac=init_util.get("cpu", 0.0),
            memory_frac=init_util.get("memory", 0.0),
            storage_frac=init_util.get("storage", 0.0),
        )

        # Clone mobility model for independent per-UAV state
        mob = mobility_model.clone()

        uav = UAVNode(
            node_id=node_id,
            communication_radius=uav_cfg["communication_radius"],
            edge=edge,
            mobility_model=mob,
        )

        # Initialize position
        uav.initialize_position(
            rng, sim_cfg["area_width"], sim_cfg["area_height"]
        )

        nodes.append(uav)

    return nodes
