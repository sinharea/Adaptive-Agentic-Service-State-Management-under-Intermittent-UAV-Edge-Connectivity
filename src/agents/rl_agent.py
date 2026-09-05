"""
RL-based agent (Baseline 4).

Lightweight Q-learning agent for state management decisions.
Uses tabular Q-learning with discretized state for transparency.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Optional, List, Tuple

from src.agents.base_agent import BaseAgent, Action, Decision, Observation


class RLAgent(BaseAgent):
    """Reinforcement learning agent using tabular Q-learning.

    Uses a discretized state representation and epsilon-greedy
    exploration. This is intentionally simple to maintain transparency
    and reproducibility.

    Note: The agent requires training episodes before it can make
    good decisions. If training is insufficient, the agent may
    perform poorly — this is reported as a valid finding.
    """

    ACTIONS = list(Action)
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, config: dict):
        super().__init__("rl_based", config)
        rl_cfg = config.get("rl_agent", {})
        self.learning_rate = rl_cfg.get("learning_rate", 0.001)
        self.gamma = rl_cfg.get("gamma", 0.99)
        self.epsilon = rl_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = rl_cfg.get("epsilon_end", 0.05)
        self.epsilon_decay = rl_cfg.get("epsilon_decay", 0.995)
        self.training = True

        # Q-table: state_key -> array of Q-values for each action
        self.q_table: dict = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))
        self.rng = np.random.default_rng(42)

        # For experience tracking
        self._last_state_key: Optional[str] = None
        self._last_action_idx: Optional[int] = None

    def _discretize_state(self, obs: Observation) -> str:
        """Convert continuous observation to discrete state key.

        Discretizes key features into bins for tabular Q-learning.
        """
        bw_bin = int(min(obs.bandwidth / 25.0, 3))  # 0-3
        lat_bin = int(min(obs.latency / 50.0, 3))   # 0-3
        cpu_bin = int(min(obs.cpu_utilization * 4, 3))  # 0-3
        contact_bin = int(min(obs.predicted_contact_duration / 60.0, 3))  # 0-3
        size_bin = int(min(obs.state_size / 250.0, 3))  # 0-3
        ckpt_bin = 1 if obs.has_checkpoint else 0
        replica_bin = min(obs.num_replicas, 2)
        running_bin = 1 if obs.is_running else 0
        neighbor_bin = 1 if obs.best_neighbor_id else 0

        return f"{bw_bin}{lat_bin}{cpu_bin}{contact_bin}{size_bin}{ckpt_bin}{replica_bin}{running_bin}{neighbor_bin}"

    def decide(self, observation: Observation, step: int) -> Decision:
        self.decision_count += 1
        state_key = self._discretize_state(observation)

        if not observation.is_running:
            return Decision(action=Action.RECOVER, reasoning="Service interrupted")

        # Epsilon-greedy action selection
        if self.training and self.rng.random() < self.epsilon:
            action_idx = self.rng.integers(0, self.NUM_ACTIONS)
        else:
            q_values = self.q_table[state_key]
            action_idx = int(np.argmax(q_values))

        action = self.ACTIONS[action_idx]

        # Filter infeasible actions
        if action == Action.MIGRATE and not observation.best_neighbor_id:
            action = Action.LOCAL_EXECUTION
            action_idx = self.ACTIONS.index(action)
        if action == Action.REPLICATE and not observation.best_neighbor_id:
            action = Action.LOCAL_EXECUTION
            action_idx = self.ACTIONS.index(action)
        if action == Action.SYNCHRONIZE and observation.num_replicas == 0:
            action = Action.LOCAL_EXECUTION
            action_idx = self.ACTIONS.index(action)
        if action == Action.RECOVER and observation.is_running:
            action = Action.LOCAL_EXECUTION
            action_idx = self.ACTIONS.index(action)

        self._last_state_key = state_key
        self._last_action_idx = action_idx
        self.action_history.append(action)

        target = observation.best_neighbor_id if action in (
            Action.MIGRATE, Action.REPLICATE
        ) else None

        return Decision(
            action=action,
            target_node=target,
            confidence=1.0 - self.epsilon if self.training else 1.0,
            reasoning=f"Q-learning (ε={self.epsilon:.3f}, state={state_key})",
        )

    def record_outcome(self, observation: Observation, action: Action,
                       reward: float, next_observation: Observation) -> None:
        """Update Q-table with observed transition."""
        if self._last_state_key is None or self._last_action_idx is None:
            return

        next_key = self._discretize_state(next_observation)
        next_q = np.max(self.q_table[next_key])

        current_q = self.q_table[self._last_state_key][self._last_action_idx]
        td_target = reward + self.gamma * next_q
        self.q_table[self._last_state_key][self._last_action_idx] += (
            self.learning_rate * (td_target - current_q)
        )

        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_end,
                             self.epsilon * self.epsilon_decay)

    def set_training(self, training: bool) -> None:
        """Toggle training mode."""
        self.training = training
        if not training:
            self.epsilon = 0.0  # Pure exploitation

    def reset(self) -> None:
        super().reset()
        self._last_state_key = None
        self._last_action_idx = None
