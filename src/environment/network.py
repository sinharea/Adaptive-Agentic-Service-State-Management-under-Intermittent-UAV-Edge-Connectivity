"""
Intermittent network model for UAV-edge connectivity.

Models distance-based connectivity with bandwidth decay, latency,
packet loss, and three discrete connectivity states
(connected / degraded / disconnected).

Also supports controlled link failure injection for experimental evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.uav import UAVNode


class ConnectivityState(Enum):
    """Discrete connectivity state between two nodes."""
    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


@dataclass
class LinkMetrics:
    """Metrics for a single communication link between two UAV nodes.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        distance: Euclidean distance (meters).
        bandwidth: Available bandwidth (Mbps).
        latency: One-way latency (ms).
        packet_loss: Packet loss rate [0, 1].
        state: Discrete connectivity state.
        predicted_contact_duration: Estimated time link will remain usable (seconds).
    """
    source_id: str
    target_id: str
    distance: float
    bandwidth: float
    latency: float
    packet_loss: float
    state: ConnectivityState
    predicted_contact_duration: float


class NetworkModel:
    """Distance-based intermittent network model.

    Bandwidth decays with distance. Latency and packet loss increase with
    distance. Connectivity state is determined by bandwidth thresholds.
    Random link failures can be injected.

    Parameters are loaded from the configuration dict.
    """

    def __init__(self, config: dict):
        net_cfg = config["network"]
        self.max_bandwidth = net_cfg["max_bandwidth"]          # Mbps
        self.min_bandwidth = net_cfg["min_bandwidth"]           # Mbps
        self.latency_base = net_cfg["latency_base"]             # ms
        self.latency_per_km = net_cfg["latency_per_km"]         # ms/km
        self.packet_loss_base = net_cfg["packet_loss_base"]     # fraction
        self.packet_loss_distance_factor = net_cfg["packet_loss_distance_factor"]

        thresholds = net_cfg["connectivity_thresholds"]
        self.connected_threshold = thresholds["connected"]      # bandwidth ratio
        self.degraded_threshold = thresholds["degraded"]        # bandwidth ratio

        # Link failure injection (configurable per experiment)
        self._failure_probability: float = 0.0
        self._failed_links: set = set()  # Set of (source_id, target_id) tuples

    # ------------------------------------------------------------------ #
    # Core link metrics computation
    # ------------------------------------------------------------------ #

    def compute_bandwidth(self, distance: float,
                          communication_radius: float) -> float:
        """Compute bandwidth based on distance.

        Uses inverse distance decay within communication range.
        Returns 0 if beyond communication radius.

        Args:
            distance: Distance between nodes (meters).
            communication_radius: Maximum range (meters).

        Returns:
            Bandwidth in Mbps.
        """
        if distance >= communication_radius or communication_radius <= 0:
            return 0.0

        # Linear decay: full bandwidth at distance=0, zero at range boundary
        ratio = 1.0 - (distance / communication_radius)
        bandwidth = self.max_bandwidth * ratio

        return max(bandwidth, 0.0)

    def compute_latency(self, distance: float) -> float:
        """Compute one-way latency based on distance.

        Args:
            distance: Distance between nodes (meters).

        Returns:
            Latency in milliseconds.
        """
        return self.latency_base + self.latency_per_km * (distance / 1000.0)

    def compute_packet_loss(self, distance: float,
                            communication_radius: float) -> float:
        """Compute packet loss rate based on distance.

        Args:
            distance: Distance between nodes (meters).
            communication_radius: Maximum range (meters).

        Returns:
            Packet loss rate as a fraction [0, 1].
        """
        if distance >= communication_radius:
            return 1.0

        dist_factor = self.packet_loss_distance_factor * (distance / 100.0)
        loss = self.packet_loss_base + dist_factor

        return min(loss, 1.0)

    def compute_connectivity_state(self, bandwidth: float) -> ConnectivityState:
        """Determine connectivity state from current bandwidth.

        Args:
            bandwidth: Current bandwidth (Mbps).

        Returns:
            ConnectivityState enum value.
        """
        if bandwidth <= 0:
            return ConnectivityState.DISCONNECTED

        ratio = bandwidth / self.max_bandwidth

        if ratio >= self.connected_threshold:
            return ConnectivityState.CONNECTED
        elif ratio >= self.degraded_threshold:
            return ConnectivityState.DEGRADED
        else:
            return ConnectivityState.DISCONNECTED

    # ------------------------------------------------------------------ #
    # Link evaluation
    # ------------------------------------------------------------------ #

    def evaluate_link(self, source: UAVNode, target: UAVNode) -> LinkMetrics:
        """Evaluate the communication link between two UAV nodes.

        Args:
            source: Source UAV node.
            target: Target UAV node.

        Returns:
            LinkMetrics with all computed values.
        """
        distance = source.distance_to(target)
        comm_radius = min(source.communication_radius,
                         target.communication_radius)

        bandwidth = self.compute_bandwidth(distance, comm_radius)
        latency = self.compute_latency(distance)
        packet_loss = self.compute_packet_loss(distance, comm_radius)

        # Check for injected failure
        link_key = (source.node_id, target.node_id)
        reverse_key = (target.node_id, source.node_id)
        if link_key in self._failed_links or reverse_key in self._failed_links:
            bandwidth = 0.0
            packet_loss = 1.0

        state = self.compute_connectivity_state(bandwidth)
        contact_duration = source.predicted_contact_duration(target)

        return LinkMetrics(
            source_id=source.node_id,
            target_id=target.node_id,
            distance=distance,
            bandwidth=bandwidth,
            latency=latency,
            packet_loss=packet_loss,
            state=state,
            predicted_contact_duration=contact_duration,
        )

    def evaluate_all_links(self, nodes: List[UAVNode]) -> Dict[Tuple[str, str], LinkMetrics]:
        """Evaluate all pairwise links between nodes.

        Only evaluates pairs where distance < max communication radius.

        Args:
            nodes: List of UAV nodes.

        Returns:
            Dict mapping (source_id, target_id) -> LinkMetrics.
        """
        links = {}
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i >= j:
                    continue
                metrics = self.evaluate_link(source, target)
                links[(source.node_id, target.node_id)] = metrics
                # Also store reverse direction (same metrics, swapped IDs)
                links[(target.node_id, source.node_id)] = LinkMetrics(
                    source_id=target.node_id,
                    target_id=source.node_id,
                    distance=metrics.distance,
                    bandwidth=metrics.bandwidth,
                    latency=metrics.latency,
                    packet_loss=metrics.packet_loss,
                    state=metrics.state,
                    predicted_contact_duration=metrics.predicted_contact_duration,
                )
        return links

    # ------------------------------------------------------------------ #
    # Failure injection
    # ------------------------------------------------------------------ #

    def set_failure_probability(self, probability: float) -> None:
        """Set the probability of random link failure per step."""
        self._failure_probability = max(0.0, min(probability, 1.0))

    def inject_failures(self, nodes: List[UAVNode],
                        rng: np.random.Generator) -> None:
        """Randomly fail links based on failure probability.

        Called once per simulation step to simulate intermittent connectivity.

        Args:
            nodes: All UAV nodes.
            rng: Random generator.
        """
        self._failed_links.clear()
        if self._failure_probability <= 0:
            return

        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i >= j:
                    continue
                if rng.random() < self._failure_probability:
                    self._failed_links.add((source.node_id, target.node_id))

    def clear_failures(self) -> None:
        """Remove all injected failures."""
        self._failed_links.clear()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_best_neighbor(self, source: UAVNode,
                          nodes: List[UAVNode]) -> Optional[Tuple[UAVNode, LinkMetrics]]:
        """Find the neighbor with the best (highest bandwidth) link.

        Args:
            source: The source UAV node.
            nodes: All UAV nodes.

        Returns:
            Tuple of (best_neighbor, link_metrics) or None if no neighbors connected.
        """
        best = None
        best_bw = 0.0

        for target in nodes:
            if target.node_id == source.node_id:
                continue
            metrics = self.evaluate_link(source, target)
            if metrics.state != ConnectivityState.DISCONNECTED and metrics.bandwidth > best_bw:
                best = (target, metrics)
                best_bw = metrics.bandwidth

        return best

    def can_transfer(self, source: UAVNode, target: UAVNode,
                     data_size_mb: float, bandwidth_fraction: float,
                     min_bandwidth: float = 0.0) -> Tuple[bool, float]:
        """Check if a data transfer is feasible between two nodes.

        Args:
            source: Source node.
            target: Target node.
            data_size_mb: Size of data to transfer (MB).
            bandwidth_fraction: Fraction of bandwidth to use for transfer.
            min_bandwidth: Minimum bandwidth required (Mbps).

        Returns:
            (feasible, estimated_transfer_time_seconds)
        """
        link = self.evaluate_link(source, target)

        if link.state == ConnectivityState.DISCONNECTED:
            return False, float("inf")

        effective_bandwidth = link.bandwidth * bandwidth_fraction  # Mbps
        if effective_bandwidth < min_bandwidth or effective_bandwidth <= 0:
            return False, float("inf")

        # Convert: data_size_mb (MB) / bandwidth (Mbps) = seconds
        # 1 MB = 8 Mbit
        transfer_time = (data_size_mb * 8.0) / effective_bandwidth  # seconds

        # Check if predicted contact duration is sufficient
        if link.predicted_contact_duration < transfer_time:
            return False, transfer_time

        return True, transfer_time

    def __repr__(self) -> str:
        return (
            f"NetworkModel(max_bw={self.max_bandwidth}Mbps, "
            f"failure_prob={self._failure_probability:.2f}, "
            f"failed_links={len(self._failed_links)})"
        )
