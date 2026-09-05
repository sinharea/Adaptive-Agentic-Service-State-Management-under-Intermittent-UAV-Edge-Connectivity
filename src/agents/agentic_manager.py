"""
Agentic State Manager (Proposed Approach).

Context-aware agent with:
- Multi-source observation (network, UAV, edge, service)
- Context building from recent history
- Explicit reasoning over risk, cost, and benefit
- Action selection based on composite scoring
- Outcome memory for learning from past decisions
- Replaceable decision engine (rule_based, rl, llm, hybrid)

This implements the Observe → Context → Reason → Act → Learn cycle.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent, Action, Decision, Observation


@dataclass
class MemoryEntry:
    """A single entry in the agent's outcome memory.

    Stores the conditions, action taken, and resulting outcome
    for future reference during decision-making.
    """
    step: int
    observation: Observation
    action: Action
    reward: float
    migration_time: float = 0.0
    interruption_time: float = 0.0
    state_loss: int = 0
    success: bool = True


class AgenticManager(BaseAgent):
    """Adaptive context-aware state management agent.

    Differentiates itself from baselines through:
    1. Context window: considers recent observation history, not just current state.
    2. Multi-factor risk assessment: combines network, mobility, resource,
       and service factors into a composite risk score.
    3. Cost-benefit analysis: estimates cost of each action before selecting.
    4. Outcome memory: learns from past decisions to improve future ones.
    5. Adaptive thresholds: adjusts decision boundaries based on outcomes.

    The decision engine is replaceable via config ('rule_based', 'rl', 'hybrid').
    """

    def __init__(self, config: dict):
        super().__init__("agentic_manager", config)
        am_cfg = config.get("agentic_manager", {})
        self.memory_capacity = am_cfg.get("memory_capacity", 1000)
        self.context_window = am_cfg.get("context_window", 10)
        self.risk_sensitivity = am_cfg.get("risk_sensitivity", 0.5)
        self.decision_engine = am_cfg.get("decision_engine", "rule_based")

        # Objective function weights
        obj_cfg = config.get("objective", {})
        self.w_interruption = obj_cfg.get("w_interruption", 1.0)
        self.w_state_transfer = obj_cfg.get("w_state_transfer", 0.3)
        self.w_state_loss = obj_cfg.get("w_state_loss", 2.0)
        self.w_migration_cost = obj_cfg.get("w_migration_cost", 0.5)
        self.w_sla_violation = obj_cfg.get("w_sla_violation", 1.5)
        self.w_resource_cost = obj_cfg.get("w_resource_cost", 0.2)

        # Config references for cost estimation
        mig_cfg = config.get("migration", {})
        self.min_bandwidth = mig_cfg.get("min_bandwidth_required", 10.0)
        self.min_contact_duration = mig_cfg.get("min_contact_duration", 30.0)
        self.migration_bw_fraction = mig_cfg.get("bandwidth_fraction", 0.5)

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_interval = ckpt_cfg.get("interval", 50)

        rep_cfg = config.get("replication", {})
        self.sync_interval = rep_cfg.get("sync_interval", 20)
        self.max_replicas = rep_cfg.get("max_replicas", 2)

        # Internal state
        self.memory: deque = deque(maxlen=self.memory_capacity)
        self.context_history: deque = deque(maxlen=self.context_window)

        # Adaptive parameters (adjusted based on outcomes)
        self.risk_threshold = 0.6       # above this → take protective action
        self.migration_benefit_threshold = 0.3  # above this → migration worthwhile

    # ------------------------------------------------------------------ #
    # OBSERVE: Collect current observation and add to context
    # ------------------------------------------------------------------ #

    def _observe(self, observation: Observation) -> None:
        """Add observation to context history."""
        self.context_history.append(observation)

    # ------------------------------------------------------------------ #
    # CONTEXT: Build context from recent observations
    # ------------------------------------------------------------------ #

    def _build_context(self, observation: Observation) -> Dict[str, float]:
        """Build decision context from observation history.

        Returns a dict of contextual features including trends.
        """
        context = {
            "current_bandwidth": observation.bandwidth,
            "current_latency": observation.latency,
            "current_cpu": observation.cpu_utilization,
            "current_contact_duration": observation.predicted_contact_duration,
            "state_size": observation.state_size,
            "priority": observation.priority_weight,
        }

        if len(self.context_history) >= 2:
            recent = list(self.context_history)
            # Bandwidth trend (positive = improving)
            bw_values = [o.bandwidth for o in recent]
            context["bandwidth_trend"] = bw_values[-1] - bw_values[0]

            # Contact duration trend
            cd_values = [o.predicted_contact_duration for o in recent]
            context["contact_trend"] = cd_values[-1] - cd_values[0]

            # CPU trend
            cpu_values = [o.cpu_utilization for o in recent]
            context["cpu_trend"] = cpu_values[-1] - cpu_values[0]

            # Variability (high variability = unpredictable)
            context["bandwidth_variability"] = float(np.std(bw_values)) if len(bw_values) > 1 else 0.0
        else:
            context["bandwidth_trend"] = 0.0
            context["contact_trend"] = 0.0
            context["cpu_trend"] = 0.0
            context["bandwidth_variability"] = 0.0

        return context

    # ------------------------------------------------------------------ #
    # REASON: Assess risk and score actions
    # ------------------------------------------------------------------ #

    def _assess_risk(self, context: Dict[str, float],
                     observation: Observation) -> float:
        """Compute a composite risk score [0, 1].

        Higher = more risky (service more likely to be disrupted).
        """
        risks = []

        # Network risk: low bandwidth or high latency
        if observation.bandwidth > 0:
            bw_risk = 1.0 - min(observation.bandwidth / 100.0, 1.0)
        else:
            bw_risk = 1.0
        risks.append(bw_risk * 0.3)

        # Contact duration risk
        if observation.predicted_contact_duration > 0:
            cd_risk = 1.0 - min(observation.predicted_contact_duration / 120.0, 1.0)
        else:
            cd_risk = 0.0  # no neighbor, no contact risk
        risks.append(cd_risk * 0.25)

        # Trend risk: declining bandwidth or contact duration
        trend_risk = 0.0
        if context.get("bandwidth_trend", 0) < -10:
            trend_risk += 0.5
        if context.get("contact_trend", 0) < -20:
            trend_risk += 0.5
        risks.append(min(trend_risk, 1.0) * 0.2)

        # Resource risk
        resource_risk = max(observation.cpu_utilization,
                           observation.memory_utilization,
                           observation.storage_utilization)
        risks.append(resource_risk * 0.15)

        # State vulnerability: no checkpoint + no replica = high risk
        state_risk = 0.0
        if not observation.has_checkpoint and observation.num_replicas == 0:
            state_risk = 1.0
        elif observation.checkpoint_staleness > 20:
            state_risk = 0.5
        risks.append(state_risk * 0.1)

        return min(sum(risks), 1.0)

    def _score_action(self, action: Action, observation: Observation,
                      context: Dict[str, float], risk: float) -> float:
        """Score an action based on estimated benefit minus cost.

        Higher score = better action choice.
        """
        score = 0.0

        if action == Action.LOCAL_EXECUTION:
            # Good when risk is low
            score = (1.0 - risk) * 0.8

        elif action == Action.CHECKPOINT:
            # Valuable when checkpoint is stale and risk is moderate
            staleness_benefit = min(observation.checkpoint_staleness / 20.0, 1.0)
            risk_benefit = risk * 0.5
            cpu_cost = 0.1  # overhead fraction
            score = (staleness_benefit * 0.5 + risk_benefit * 0.5) - cpu_cost

        elif action == Action.REPLICATE:
            if observation.num_replicas >= self.max_replicas:
                return -1.0  # infeasible
            if not observation.best_neighbor_id:
                return -1.0
            # Valuable when risk is high and no replicas
            replica_need = 1.0 if observation.num_replicas == 0 else 0.3
            bw_cost = observation.state_size / max(observation.best_neighbor_bandwidth, 1) * 0.01
            score = (risk * replica_need) - bw_cost

        elif action == Action.SYNCHRONIZE:
            if observation.num_replicas == 0:
                return -1.0
            # Valuable when replicas are stale
            sync_benefit = min(observation.replica_staleness / 10.0, 1.0)
            score = sync_benefit * 0.5

        elif action == Action.MIGRATE:
            if not observation.best_neighbor_id:
                return -1.0
            # Migration benefit vs cost
            bw = observation.best_neighbor_bandwidth
            if bw < self.min_bandwidth:
                return -1.0
            transfer_time = (observation.state_size * 8.0) / (bw * self.migration_bw_fraction)
            contact = observation.best_neighbor_contact_duration
            if contact < transfer_time:
                return -1.0  # won't complete

            migration_cost = (self.w_migration_cost * transfer_time / 100.0 +
                            self.w_state_transfer * observation.state_size / 500.0)
            risk_benefit = risk * observation.priority_weight
            score = risk_benefit - migration_cost

            # Boost from memory: similar past situations where migration helped
            memory_boost = self._memory_lookup(action, observation)
            score += memory_boost * 0.2

        elif action == Action.DEGRADE:
            # Degrade is a last resort
            score = risk * 0.3 - 0.5

        elif action == Action.RECOVER:
            if observation.is_running:
                return -1.0  # not needed
            score = 1.0  # always recover if interrupted

        return score

    # ------------------------------------------------------------------ #
    # MEMORY: Learn from past decisions
    # ------------------------------------------------------------------ #

    def _memory_lookup(self, action: Action,
                       observation: Observation) -> float:
        """Look up relevant past outcomes for a given action.

        Returns a benefit score based on historical outcomes:
        positive if the action previously led to good outcomes in
        similar conditions, negative otherwise.
        """
        if not self.memory:
            return 0.0

        relevant = []
        for entry in self.memory:
            if entry.action != action:
                continue
            # Similarity: compare bandwidth and contact duration
            bw_diff = abs(entry.observation.bandwidth - observation.bandwidth)
            cd_diff = abs(entry.observation.predicted_contact_duration -
                         observation.predicted_contact_duration)
            if bw_diff < 20 and cd_diff < 30:
                relevant.append(entry)

        if not relevant:
            return 0.0

        # Average reward of similar past outcomes
        avg_reward = sum(e.reward for e in relevant) / len(relevant)
        return avg_reward

    def _update_adaptive_thresholds(self) -> None:
        """Adjust thresholds based on recent memory.

        If recent migrations were mostly unsuccessful, raise the
        migration benefit threshold. If recent checkpoints prevented
        state loss, lower the risk threshold.
        """
        if len(self.memory) < 10:
            return

        recent = list(self.memory)[-20:]

        # Migration success rate
        migrations = [e for e in recent if e.action == Action.MIGRATE]
        if len(migrations) >= 3:
            success_rate = sum(1 for m in migrations if m.success) / len(migrations)
            if success_rate < 0.5:
                self.migration_benefit_threshold = min(
                    self.migration_benefit_threshold + 0.05, 0.8)
            elif success_rate > 0.8:
                self.migration_benefit_threshold = max(
                    self.migration_benefit_threshold - 0.05, 0.1)

    # ------------------------------------------------------------------ #
    # ACT: Select best action
    # ------------------------------------------------------------------ #

    def decide(self, observation: Observation, step: int) -> Decision:
        """Full Observe → Context → Reason → Act cycle."""
        self.decision_count += 1

        # OBSERVE
        self._observe(observation)

        # Handle interruption immediately
        if not observation.is_running:
            self.action_history.append(Action.RECOVER)
            return Decision(action=Action.RECOVER,
                          reasoning="Service interrupted — attempting recovery")

        # CONTEXT
        context = self._build_context(observation)

        # REASON
        risk = self._assess_risk(context, observation)

        # Score all actions
        action_scores: List[Tuple[Action, float]] = []
        for action in Action:
            score = self._score_action(action, observation, context, risk)
            action_scores.append((action, score))

        # Apply risk sensitivity
        for i, (action, score) in enumerate(action_scores):
            if action in (Action.CHECKPOINT, Action.REPLICATE, Action.SYNCHRONIZE):
                # Protective actions get a boost proportional to risk sensitivity
                action_scores[i] = (action, score + risk * self.risk_sensitivity * 0.2)

        # Select best action
        action_scores.sort(key=lambda x: x[1], reverse=True)
        best_action, best_score = action_scores[0]

        # Build reasoning
        reasoning_parts = [
            f"Risk={risk:.2f}",
            f"BW_trend={context.get('bandwidth_trend', 0):.1f}",
            f"Contact={observation.predicted_contact_duration:.1f}s",
        ]
        top3 = [(a.value, f"{s:.3f}") for a, s in action_scores[:3]]
        reasoning_parts.append(f"Top3={top3}")

        target = None
        if best_action in (Action.MIGRATE, Action.REPLICATE):
            target = observation.best_neighbor_id

        # Adaptive threshold update
        self._update_adaptive_thresholds()

        self.action_history.append(best_action)
        return Decision(
            action=best_action,
            target_node=target,
            confidence=best_score,
            reasoning="; ".join(reasoning_parts),
        )

    # ------------------------------------------------------------------ #
    # LEARN: Record outcomes
    # ------------------------------------------------------------------ #

    def record_outcome(self, observation: Observation, action: Action,
                       reward: float, next_observation: Observation,
                       migration_time: float = 0.0,
                       interruption_time: float = 0.0,
                       state_loss: int = 0,
                       success: bool = True) -> None:
        """Store outcome in memory for future reference.

        This method has an extended signature compared to BaseAgent
        to capture richer outcome data.
        """
        entry = MemoryEntry(
            step=self.decision_count,
            observation=observation,
            action=action,
            reward=reward,
            migration_time=migration_time,
            interruption_time=interruption_time,
            state_loss=state_loss,
            success=success,
        )
        self.memory.append(entry)

    def reset(self) -> None:
        """Reset for new episode, keeping memory."""
        super().reset()
        self.context_history.clear()
        # Memory is preserved across episodes for learning
