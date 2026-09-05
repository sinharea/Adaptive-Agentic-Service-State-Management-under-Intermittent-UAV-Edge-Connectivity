"""
Recovery mechanism for stateful services.

Restores service state from checkpoints or replicas after disruption.
Tracks recovery time and state loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.service.state import ServiceState
from src.service.checkpoint import CheckpointManager, Checkpoint


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    source: str = ""          # "checkpoint" or "replica" or ""
    recovered_version: int = -1
    state_loss: int = 0       # versions lost
    recovery_time: float = 0.0  # steps
    reason: str = ""


class RecoveryManager:
    """Manages service recovery from checkpoints and replicas.

    Attempts recovery in order of preference:
    1. Best replica (least stale)
    2. Latest checkpoint
    """

    def __init__(self, config: dict):
        svc_cfg = config.get("service", {})
        self.sla_max_recovery_time = svc_cfg.get("sla_max_recovery_time", 30.0)

    def recover_from_checkpoint(self, service: ServiceState,
                                 checkpoint: Checkpoint,
                                 step: int) -> RecoveryResult:
        """Recover service state from a checkpoint.

        Args:
            service: The service to recover.
            checkpoint: The checkpoint to recover from.
            step: Current simulation step.

        Returns:
            RecoveryResult with recovery metrics.
        """
        version_before = service.state_version
        state_loss = version_before - checkpoint.version

        # Recovery time proportional to checkpoint size
        recovery_time = checkpoint.size / 100.0  # simplified: 100 MB/step

        # Restore state
        service.state_version = checkpoint.version
        service.state_size = checkpoint.size / 1.1  # reverse multiplier approx
        service.total_state_loss += max(state_loss, 0)

        # End interruption
        service.end_interruption(step)

        # Check SLA
        if recovery_time > self.sla_max_recovery_time:
            service.sla_violations += 1

        return RecoveryResult(
            success=True,
            source="checkpoint",
            recovered_version=checkpoint.version,
            state_loss=max(state_loss, 0),
            recovery_time=recovery_time,
        )

    def recover_from_replica(self, service: ServiceState,
                              replica_node_id: str,
                              replica_version: int,
                              state_size: float,
                              step: int) -> RecoveryResult:
        """Recover service state from a replica.

        Args:
            service: The service to recover.
            replica_node_id: Node holding the replica.
            replica_version: Version of the replica state.
            state_size: Size of the replica state in MB.
            step: Current simulation step.

        Returns:
            RecoveryResult with recovery metrics.
        """
        version_before = service.state_version
        state_loss = version_before - replica_version

        # Recovery time proportional to state size
        recovery_time = state_size / 100.0

        # Restore state
        service.state_version = replica_version
        service.state_size = state_size
        service.current_node = replica_node_id
        service.total_state_loss += max(state_loss, 0)

        # End interruption
        service.end_interruption(step)

        if recovery_time > self.sla_max_recovery_time:
            service.sla_violations += 1

        return RecoveryResult(
            success=True,
            source="replica",
            recovered_version=replica_version,
            state_loss=max(state_loss, 0),
            recovery_time=recovery_time,
        )

    def attempt_recovery(self, service: ServiceState,
                          checkpoint_mgr: CheckpointManager,
                          step: int) -> RecoveryResult:
        """Attempt recovery using the best available source.

        Prefers replicas (typically more recent) over checkpoints.

        Args:
            service: The service to recover.
            checkpoint_mgr: CheckpointManager to query checkpoints.
            step: Current simulation step.

        Returns:
            RecoveryResult.
        """
        # Try replica first (best version)
        if service.replica_versions:
            best_node = max(service.replica_versions,
                           key=lambda n: service.replica_versions[n])
            best_version = service.replica_versions[best_node]

            # Check if replica is better than checkpoint
            ckpt = checkpoint_mgr.get_latest_checkpoint(service.service_id)
            if ckpt is None or best_version >= ckpt.version:
                return self.recover_from_replica(
                    service, best_node, best_version,
                    service.state_size, step
                )

        # Try checkpoint
        ckpt = checkpoint_mgr.get_latest_checkpoint(service.service_id)
        if ckpt is not None:
            return self.recover_from_checkpoint(service, ckpt, step)

        # No recovery source
        return RecoveryResult(
            success=False,
            reason="No checkpoint or replica available",
        )
