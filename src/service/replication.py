"""
State replication across edge nodes.

Manages creation and tracking of replicas on neighboring nodes.
Tracks bandwidth and storage costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from src.service.state import ServiceState
from src.environment.edge_node import EdgeNode
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel, ConnectivityState


@dataclass
class ReplicationResult:
    """Result of a replication operation."""
    success: bool
    target_node: str = ""
    version_replicated: int = -1
    bandwidth_cost: float = 0.0   # MB transferred
    storage_cost: float = 0.0     # MB on target
    transfer_time: float = 0.0    # steps
    reason: str = ""


class ReplicationManager:
    """Manages state replication to neighboring nodes."""

    def __init__(self, config: dict):
        rep_cfg = config["replication"]
        self.enabled = rep_cfg.get("enabled", True)
        self.max_replicas = rep_cfg.get("max_replicas", 2)
        self.sync_interval = rep_cfg.get("sync_interval", 20)
        self.bandwidth_fraction = rep_cfg.get("bandwidth_fraction", 0.2)

    def replicate(self, service: ServiceState,
                  source_uav: UAVNode, target_uav: UAVNode,
                  network: NetworkModel) -> ReplicationResult:
        """Replicate service state to a target node.

        Args:
            service: The service to replicate.
            source_uav: UAV hosting the service.
            target_uav: UAV to replicate to.
            network: Network model for link evaluation.

        Returns:
            ReplicationResult with metrics.
        """
        if not self.enabled:
            return ReplicationResult(success=False, reason="Replication disabled")

        if not service.is_running:
            return ReplicationResult(success=False, reason="Service not running")

        if len(service.replicated_nodes) >= self.max_replicas:
            return ReplicationResult(success=False, reason="Max replicas reached")

        if target_uav.node_id in service.replicated_nodes:
            return ReplicationResult(success=False,
                                     reason="Already replicated to this node")

        # Check network feasibility
        feasible, transfer_time = network.can_transfer(
            source_uav, target_uav, service.state_size,
            self.bandwidth_fraction
        )
        if not feasible:
            return ReplicationResult(
                success=False,
                reason="Network transfer not feasible",
                transfer_time=transfer_time,
            )

        # Check target storage
        if not target_uav.edge.allocate_storage(service.state_size):
            return ReplicationResult(
                success=False,
                target_node=target_uav.node_id,
                reason="Insufficient storage on target",
            )

        # Success
        service.replicated_nodes.append(target_uav.node_id)
        service.replica_versions[target_uav.node_id] = service.state_version

        return ReplicationResult(
            success=True,
            target_node=target_uav.node_id,
            version_replicated=service.state_version,
            bandwidth_cost=service.state_size,
            storage_cost=service.state_size,
            transfer_time=transfer_time,
        )

    def remove_replica(self, service: ServiceState,
                       node_id: str, edge: EdgeNode) -> float:
        """Remove a replica from a node. Returns storage freed."""
        if node_id not in service.replicated_nodes:
            return 0.0
        service.replicated_nodes.remove(node_id)
        version = service.replica_versions.pop(node_id, None)
        edge.release_storage(service.state_size)
        return service.state_size
