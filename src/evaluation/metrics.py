"""
Evaluation metrics collection and computation.

Tracks all metrics defined in the research design:
service, migration, state management, network, and resource metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StepMetrics:
    """Metrics collected for a single simulation step."""
    step: int = 0
    action: str = ""
    action_success: bool = True

    # Service
    service_running: bool = True
    interruption: bool = False
    recovery_time: float = 0.0
    state_loss: int = 0
    sla_violation: bool = False

    # Migration
    migration_attempted: bool = False
    migration_success: bool = False
    migration_time: float = 0.0
    data_transferred: float = 0.0

    # State management
    checkpoint_created: bool = False
    checkpoint_cost_cpu: float = 0.0
    checkpoint_cost_storage: float = 0.0
    replication_performed: bool = False
    replication_cost_bw: float = 0.0
    sync_performed: bool = False
    sync_data: float = 0.0

    # Network
    bandwidth: float = 0.0
    latency: float = 0.0
    packet_loss: float = 0.0
    connectivity_state: str = "disconnected"

    # Resources
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0

    # Service state
    state_size: float = 0.0
    state_version: int = 0
    num_replicas: int = 0
    checkpoint_staleness: int = 0


@dataclass
class SimulationMetrics:
    """Aggregated metrics for an entire simulation run."""
    strategy: str = ""
    seed: int = 0
    steps: List[StepMetrics] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics from step-level metrics."""
        if not self.steps:
            return {}

        n = len(self.steps)
        total_steps = n

        # Service metrics
        running_steps = sum(1 for s in self.steps if s.service_running)
        service_availability = running_steps / total_steps if total_steps > 0 else 0.0
        total_interruption = sum(1 for s in self.steps if s.interruption)
        total_recovery_time = sum(s.recovery_time for s in self.steps)
        total_state_loss = sum(s.state_loss for s in self.steps)
        sla_violations = sum(1 for s in self.steps if s.sla_violation)

        # Migration metrics
        migrations_attempted = sum(1 for s in self.steps if s.migration_attempted)
        migrations_successful = sum(1 for s in self.steps
                                    if s.migration_attempted and s.migration_success)
        migration_success_rate = (migrations_successful / migrations_attempted
                                  if migrations_attempted > 0 else 0.0)
        total_migration_time = sum(s.migration_time for s in self.steps)

        # State management
        checkpoints_created = sum(1 for s in self.steps if s.checkpoint_created)
        total_checkpoint_cpu = sum(s.checkpoint_cost_cpu for s in self.steps)
        total_checkpoint_storage = sum(s.checkpoint_cost_storage for s in self.steps)
        replications = sum(1 for s in self.steps if s.replication_performed)
        total_replication_bw = sum(s.replication_cost_bw for s in self.steps)
        syncs = sum(1 for s in self.steps if s.sync_performed)
        total_sync_data = sum(s.sync_data for s in self.steps)

        # Network
        avg_bandwidth = sum(s.bandwidth for s in self.steps) / n
        avg_latency = sum(s.latency for s in self.steps) / n

        # Resources
        avg_cpu = sum(s.cpu_utilization for s in self.steps) / n
        avg_memory = sum(s.memory_utilization for s in self.steps) / n
        avg_storage = sum(s.storage_utilization for s in self.steps) / n

        # Data transfer volume
        total_data_transferred = sum(s.data_transferred + s.replication_cost_bw +
                                     s.sync_data for s in self.steps)

        # Energy proxy (CPU-time based)
        energy_proxy = sum(s.cpu_utilization for s in self.steps)

        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "total_steps": total_steps,
            "service_availability": service_availability,
            "total_interruption_time": total_interruption,
            "total_recovery_time": total_recovery_time,
            "total_state_loss": total_state_loss,
            "sla_violations": sla_violations,
            "migrations_attempted": migrations_attempted,
            "migrations_successful": migrations_successful,
            "migration_success_rate": migration_success_rate,
            "total_migration_time": total_migration_time,
            "migration_frequency": migrations_attempted,
            "checkpoints_created": checkpoints_created,
            "total_checkpoint_cpu": total_checkpoint_cpu,
            "total_checkpoint_storage": total_checkpoint_storage,
            "replications": replications,
            "total_replication_bw": total_replication_bw,
            "syncs": syncs,
            "total_sync_data": total_sync_data,
            "avg_bandwidth": avg_bandwidth,
            "avg_latency": avg_latency,
            "avg_cpu_utilization": avg_cpu,
            "avg_memory_utilization": avg_memory,
            "avg_storage_utilization": avg_storage,
            "total_data_transferred": total_data_transferred,
            "energy_proxy": energy_proxy,
        }


def compute_objective_cost(summary: Dict[str, float],
                           weights: Dict[str, float]) -> float:
    """Compute the weighted objective cost from summary metrics.

    Total Cost = w1 × interruption + w2 × state_transfer
               + w3 × state_loss + w4 × migration_cost
               + w5 × SLA_violation + w6 × resource_cost
    """
    cost = (
        weights.get("w_interruption", 1.0) * summary.get("total_interruption_time", 0)
        + weights.get("w_state_transfer", 0.3) * summary.get("total_data_transferred", 0) / 1000.0
        + weights.get("w_state_loss", 2.0) * summary.get("total_state_loss", 0)
        + weights.get("w_migration_cost", 0.5) * summary.get("total_migration_time", 0) / 100.0
        + weights.get("w_sla_violation", 1.5) * summary.get("sla_violations", 0)
        + weights.get("w_resource_cost", 0.2) * summary.get("energy_proxy", 0) / 100.0
    )
    return cost
