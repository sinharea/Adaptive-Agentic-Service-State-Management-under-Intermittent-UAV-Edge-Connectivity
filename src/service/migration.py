"""
Service migration mechanism.

Handles full transfer of service state from one edge node to another.
Tracks migration time, interruption, bandwidth, success/failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.service.state import ServiceState
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel, ConnectivityState


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    source_node: str = ""
    destination_node: str = ""
    state_version: int = -1
    data_transferred: float = 0.0   # MB
    migration_time: float = 0.0     # steps/seconds
    interruption_time: float = 0.0  # steps
    reason: str = ""


class MigrationManager:
    """Manages service migration between edge nodes.

    Migration is stop-and-copy: the service is interrupted during transfer.
    """

    def __init__(self, config: dict):
        mig_cfg = config["migration"]
        self.bandwidth_fraction = mig_cfg.get("bandwidth_fraction", 0.5)
        self.cpu_overhead_fraction = mig_cfg.get("cpu_overhead_fraction", 0.15)
        self.min_bandwidth_required = mig_cfg.get("min_bandwidth_required", 10.0)
        self.min_contact_duration = mig_cfg.get("min_contact_duration", 30.0)

    def can_migrate(self, service: ServiceState,
                    source_uav: UAVNode, target_uav: UAVNode,
                    network: NetworkModel) -> tuple[bool, str, float]:
        """Check if migration is feasible.

        Returns:
            (feasible, reason, estimated_time)
        """
        if service.is_migrating:
            return False, "Migration already in progress", 0.0

        if not service.is_running:
            return False, "Service not running", 0.0

        # Check network
        link = network.evaluate_link(source_uav, target_uav)
        if link.state == ConnectivityState.DISCONNECTED:
            return False, "Nodes are disconnected", 0.0

        effective_bw = link.bandwidth * self.bandwidth_fraction
        if effective_bw < self.min_bandwidth_required:
            return (False,
                    f"Bandwidth {effective_bw:.1f}Mbps below minimum {self.min_bandwidth_required}Mbps",
                    0.0)

        # Estimate transfer time
        transfer_time = (service.state_size * 8.0) / effective_bw  # seconds

        # Check contact duration
        if link.predicted_contact_duration < transfer_time:
            return (False,
                    f"Contact duration {link.predicted_contact_duration:.1f}s insufficient for {transfer_time:.1f}s transfer",
                    transfer_time)

        # Check target resources
        if not target_uav.edge.can_host_service(
            service.state_size * 0.01,  # approximate CPU need
            service.state_size,         # memory for state
            service.state_size          # storage for state
        ):
            return False, "Insufficient resources on target", transfer_time

        return True, "Migration feasible", transfer_time

    def migrate(self, service: ServiceState,
                source_uav: UAVNode, target_uav: UAVNode,
                network: NetworkModel, step: int) -> MigrationResult:
        """Execute service migration from source to target.

        This is a simplified atomic migration: checks feasibility,
        then transfers state. In a real system, this would span
        multiple simulation steps.

        Args:
            service: The service to migrate.
            source_uav: Current host UAV.
            target_uav: Destination UAV.
            network: Network model.
            step: Current simulation step.

        Returns:
            MigrationResult with metrics.
        """
        feasible, reason, est_time = self.can_migrate(
            service, source_uav, target_uav, network
        )

        if not feasible:
            return MigrationResult(
                success=False,
                source_node=source_uav.node_id,
                destination_node=target_uav.node_id,
                reason=reason,
                migration_time=est_time,
            )

        # Begin migration
        service.is_migrating = True
        service.begin_interruption(step)

        # Calculate actual transfer metrics
        link = network.evaluate_link(source_uav, target_uav)
        effective_bw = link.bandwidth * self.bandwidth_fraction
        transfer_time = (service.state_size * 8.0) / effective_bw

        # Allocate resources on target
        target_uav.edge.allocate_memory(service.state_size)
        target_uav.edge.allocate_storage(service.state_size)

        # Release resources on source
        source_uav.edge.release_memory(service.state_size)
        source_uav.edge.release_storage(service.state_size)

        # Complete migration
        old_node = service.current_node
        service.current_node = target_uav.node_id
        service.is_migrating = False
        interruption_time = service.end_interruption(step + int(transfer_time))
        service.migration_count += 1

        return MigrationResult(
            success=True,
            source_node=old_node,
            destination_node=target_uav.node_id,
            state_version=service.state_version,
            data_transferred=service.state_size,
            migration_time=transfer_time,
            interruption_time=interruption_time,
        )
