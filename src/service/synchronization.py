"""
State synchronization for replicated services.

Updates existing replicas with the latest state from the primary.
Tracks synchronization lag and bandwidth costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

from src.service.state import ServiceState
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel, ConnectivityState


@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    success: bool
    target_node: str = ""
    old_version: int = -1
    new_version: int = -1
    data_transferred: float = 0.0  # MB
    transfer_time: float = 0.0     # steps
    reason: str = ""


class SynchronizationManager:
    """Manages synchronization of replicas with primary state."""

    def __init__(self, config: dict):
        rep_cfg = config["replication"]
        self.sync_interval = rep_cfg.get("sync_interval", 20)
        self.bandwidth_fraction = rep_cfg.get("bandwidth_fraction", 0.2)

    def synchronize(self, service: ServiceState,
                    source_uav: UAVNode, target_uav: UAVNode,
                    network: NetworkModel,
                    step: int) -> SyncResult:
        """Synchronize a replica with the primary state.

        Transfers only the delta (version difference × update rate) if possible.

        Args:
            service: The service whose state to sync.
            source_uav: UAV hosting the primary service.
            target_uav: UAV hosting the replica.
            network: Network model for link evaluation.
            step: Current simulation step.

        Returns:
            SyncResult with metrics.
        """
        if target_uav.node_id not in service.replicated_nodes:
            return SyncResult(success=False,
                              reason="No replica on target node")

        if not service.is_running:
            return SyncResult(success=False, reason="Service not running")

        old_version = service.replica_versions.get(target_uav.node_id, 0)
        if old_version >= service.state_version:
            return SyncResult(
                success=True,
                target_node=target_uav.node_id,
                old_version=old_version,
                new_version=old_version,
                data_transferred=0.0,
                reason="Already up to date",
            )

        # Calculate delta size
        version_diff = service.state_version - old_version
        delta_size = version_diff * service.state_update_rate  # MB

        # Check network feasibility
        feasible, transfer_time = network.can_transfer(
            source_uav, target_uav, delta_size,
            self.bandwidth_fraction
        )
        if not feasible:
            return SyncResult(
                success=False,
                target_node=target_uav.node_id,
                reason="Network transfer not feasible",
                transfer_time=transfer_time,
            )

        # Update replica version
        service.replica_versions[target_uav.node_id] = service.state_version
        service.last_sync_time = step

        return SyncResult(
            success=True,
            target_node=target_uav.node_id,
            old_version=old_version,
            new_version=service.state_version,
            data_transferred=delta_size,
            transfer_time=transfer_time,
        )

    def synchronize_all(self, service: ServiceState,
                        source_uav: UAVNode,
                        uav_map: Dict[str, UAVNode],
                        network: NetworkModel,
                        step: int) -> List[SyncResult]:
        """Synchronize all replicas.

        Args:
            service: The service to synchronize.
            source_uav: UAV hosting the primary.
            uav_map: Dict mapping node_id -> UAVNode.
            network: Network model.
            step: Current simulation step.

        Returns:
            List of SyncResult for each replica.
        """
        results = []
        for node_id in list(service.replicated_nodes):
            target = uav_map.get(node_id)
            if target is None:
                continue
            result = self.synchronize(service, source_uav, target, network, step)
            results.append(result)
        return results

    def get_sync_lag(self, service: ServiceState, current_step: int) -> int:
        """Return the number of steps since last synchronization."""
        return current_step - service.last_sync_time
