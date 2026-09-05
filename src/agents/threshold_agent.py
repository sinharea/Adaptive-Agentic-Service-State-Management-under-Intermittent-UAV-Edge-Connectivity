"""
Threshold-based migration agent (Baseline 2).

Migrates when any metric exceeds a configurable threshold.
"""

from src.agents.base_agent import BaseAgent, Action, Decision, Observation


class ThresholdAgent(BaseAgent):
    """Threshold-based reactive migration agent.

    Migrates when:
    - bandwidth < bandwidth_threshold, OR
    - latency > latency_threshold, OR
    - CPU utilization > cpu_threshold

    Also performs periodic checkpointing.
    """

    def __init__(self, config: dict):
        super().__init__("threshold_based", config)
        th_cfg = config.get("threshold_agent", {})
        self.bandwidth_threshold = th_cfg.get("bandwidth_threshold", 20.0)
        self.latency_threshold = th_cfg.get("latency_threshold", 100.0)
        self.cpu_threshold = th_cfg.get("cpu_threshold", 0.85)
        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_interval = ckpt_cfg.get("interval", 50)

    def decide(self, observation: Observation, step: int) -> Decision:
        self.decision_count += 1

        if not observation.is_running:
            return Decision(action=Action.RECOVER, reasoning="Service interrupted")

        # Check thresholds
        reasons = []
        should_migrate = False

        if (observation.bandwidth > 0 and
                observation.bandwidth < self.bandwidth_threshold):
            reasons.append(
                f"bandwidth {observation.bandwidth:.1f} < {self.bandwidth_threshold}"
            )
            should_migrate = True

        if observation.latency > self.latency_threshold:
            reasons.append(
                f"latency {observation.latency:.1f} > {self.latency_threshold}"
            )
            should_migrate = True

        if observation.cpu_utilization > self.cpu_threshold:
            reasons.append(
                f"CPU {observation.cpu_utilization:.2f} > {self.cpu_threshold}"
            )
            should_migrate = True

        if should_migrate and observation.best_neighbor_id:
            action = Action.MIGRATE
            reasoning = f"Threshold exceeded: {'; '.join(reasons)}"
            self.action_history.append(action)
            return Decision(
                action=action,
                target_node=observation.best_neighbor_id,
                reasoning=reasoning,
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
            reasoning="All metrics within thresholds",
        )
