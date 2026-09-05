"""
Mobility-aware migration agent (Baseline 3).

Uses predicted UAV contact duration to decide when to migrate proactively.
"""

from src.agents.base_agent import BaseAgent, Action, Decision, Observation


class MobilityAgent(BaseAgent):
    """Mobility-aware proactive migration agent.

    Migrates when predicted contact duration falls below a threshold,
    provided sufficient bandwidth exists for state transfer. Also
    considers velocity trends for proactive action.
    """

    def __init__(self, config: dict):
        super().__init__("mobility_aware", config)
        mob_cfg = config.get("mobility_agent", {})
        self.contact_threshold = mob_cfg.get("contact_duration_threshold", 60.0)
        self.use_velocity_prediction = mob_cfg.get("use_velocity_prediction", True)

        mig_cfg = config.get("migration", {})
        self.min_bandwidth = mig_cfg.get("min_bandwidth_required", 10.0)

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_interval = ckpt_cfg.get("interval", 50)

        rep_cfg = config.get("replication", {})
        self.sync_interval = rep_cfg.get("sync_interval", 20)

    def decide(self, observation: Observation, step: int) -> Decision:
        self.decision_count += 1

        if not observation.is_running:
            return Decision(action=Action.RECOVER, reasoning="Service interrupted")

        # Check if we should migrate based on contact duration
        if (observation.predicted_contact_duration < self.contact_threshold
                and observation.predicted_contact_duration > 0
                and observation.best_neighbor_id
                and observation.best_neighbor_bandwidth >= self.min_bandwidth):

            # Estimate transfer time
            effective_bw = observation.best_neighbor_bandwidth * 0.5
            if effective_bw > 0:
                transfer_time = (observation.state_size * 8.0) / effective_bw
                # Only migrate if we have time
                if observation.best_neighbor_contact_duration > transfer_time:
                    self.action_history.append(Action.MIGRATE)
                    return Decision(
                        action=Action.MIGRATE,
                        target_node=observation.best_neighbor_id,
                        reasoning=(
                            f"Contact duration {observation.predicted_contact_duration:.1f}s "
                            f"below threshold {self.contact_threshold:.1f}s, "
                            f"migration feasible ({transfer_time:.1f}s needed)"
                        ),
                    )

        # If connectivity is degrading but migration not feasible, checkpoint
        if (observation.predicted_contact_duration < self.contact_threshold * 2
                and observation.predicted_contact_duration > 0):
            if not observation.has_checkpoint or observation.checkpoint_staleness > 5:
                self.action_history.append(Action.CHECKPOINT)
                return Decision(
                    action=Action.CHECKPOINT,
                    reasoning="Proactive checkpoint due to declining contact duration",
                )

        # Sync replicas if interval reached
        if (observation.num_replicas > 0
                and observation.time_since_sync >= self.sync_interval):
            self.action_history.append(Action.SYNCHRONIZE)
            return Decision(
                action=Action.SYNCHRONIZE,
                reasoning=f"Sync interval reached ({observation.time_since_sync} steps)",
            )

        # Replicate if no replicas and neighbor available
        if (observation.num_replicas == 0
                and observation.best_neighbor_id
                and observation.best_neighbor_bandwidth > self.min_bandwidth):
            self.action_history.append(Action.REPLICATE)
            return Decision(
                action=Action.REPLICATE,
                target_node=observation.best_neighbor_id,
                reasoning="Create initial replica for resilience",
            )

        # Periodic checkpoint
        if step > 0 and step % self.checkpoint_interval == 0:
            self.action_history.append(Action.CHECKPOINT)
            return Decision(
                action=Action.CHECKPOINT,
                reasoning=f"Periodic checkpoint at step {step}",
            )

        self.action_history.append(Action.LOCAL_EXECUTION)
        return Decision(
            action=Action.LOCAL_EXECUTION,
            reasoning="Conditions stable, continue locally",
        )
