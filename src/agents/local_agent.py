"""
Local execution agent (Baseline 1).

Never migrates. Always continues local execution.
Serves as a lower bound for migration strategies.
"""

from src.agents.base_agent import BaseAgent, Action, Decision, Observation


class LocalAgent(BaseAgent):
    """Agent that never migrates — always runs locally.

    This baseline demonstrates the cost of inaction when connectivity
    degrades. It may perform checkpointing if configured.
    """

    def __init__(self, config: dict):
        super().__init__("local_execution", config)
        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_interval = ckpt_cfg.get("interval", 50)

    def decide(self, observation: Observation, step: int) -> Decision:
        self.decision_count += 1

        # Periodic checkpoint if not running
        if not observation.is_running:
            return Decision(action=Action.RECOVER, reasoning="Service interrupted")

        # Periodic checkpoint
        if step > 0 and step % self.checkpoint_interval == 0:
            action = Action.CHECKPOINT
            reasoning = f"Periodic checkpoint at step {step}"
        else:
            action = Action.LOCAL_EXECUTION
            reasoning = "Continue local execution (never migrate)"

        self.action_history.append(action)
        return Decision(action=action, reasoning=reasoning)
