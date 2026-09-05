"""
Tests for the agentic state manager.
"""

import pytest
from src.agents.base_agent import Action, Observation
from src.agents.agentic_manager import AgenticManager, MemoryEntry


def _default_config():
    return {
        "checkpoint": {"enabled": True, "interval": 50, "cpu_overhead_fraction": 0.1,
                       "storage_multiplier": 1.1, "max_checkpoints": 3},
        "replication": {"enabled": True, "max_replicas": 2, "sync_interval": 20,
                       "bandwidth_fraction": 0.2},
        "migration": {"bandwidth_fraction": 0.5, "cpu_overhead_fraction": 0.15,
                      "min_bandwidth_required": 10.0, "min_contact_duration": 30.0},
        "objective": {"w_interruption": 1.0, "w_state_transfer": 0.3,
                     "w_state_loss": 2.0, "w_migration_cost": 0.5,
                     "w_sla_violation": 1.5, "w_resource_cost": 0.2},
        "agentic_manager": {"memory_capacity": 100, "context_window": 5,
                           "risk_sensitivity": 0.5, "decision_engine": "rule_based"},
    }


def _make_obs(**kwargs):
    defaults = dict(
        bandwidth=80.0, latency=10.0, packet_loss=0.01,
        connectivity_state="connected", predicted_contact_duration=120.0,
        num_neighbors=3, uav_velocity=10.0,
        cpu_utilization=0.3, memory_utilization=0.4, storage_utilization=0.2,
        state_size=100.0, state_version=10, state_update_rate=1.0,
        priority_weight=1.0, has_checkpoint=True, checkpoint_staleness=5,
        num_replicas=1, replica_staleness=3, time_since_sync=10,
        is_running=True, best_neighbor_id="uav_1",
        best_neighbor_bandwidth=60.0, best_neighbor_contact_duration=90.0,
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class TestAgenticManager:

    def test_returns_valid_action(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs()
        d = agent.decide(obs, step=1)
        assert isinstance(d.action, Action)
        assert d.reasoning  # should have reasoning

    def test_recovers_when_interrupted(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs(is_running=False)
        d = agent.decide(obs, step=1)
        assert d.action == Action.RECOVER

    def test_context_building(self):
        agent = AgenticManager(_default_config())
        # Feed multiple observations to build context
        for i in range(5):
            obs = _make_obs(bandwidth=80.0 - i * 10)
            agent.decide(obs, step=i)
        # Context should have trend
        context = agent._build_context(_make_obs(bandwidth=30.0))
        assert context["bandwidth_trend"] < 0  # declining

    def test_risk_assessment_low_risk(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs(bandwidth=90.0, predicted_contact_duration=200.0,
                        cpu_utilization=0.1, has_checkpoint=True, num_replicas=1)
        context = agent._build_context(obs)
        risk = agent._assess_risk(context, obs)
        assert risk < 0.4  # low risk

    def test_risk_assessment_high_risk(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs(bandwidth=5.0, predicted_contact_duration=10.0,
                        cpu_utilization=0.95, has_checkpoint=False, num_replicas=0)
        context = agent._build_context(obs)
        risk = agent._assess_risk(context, obs)
        assert risk > 0.5  # high risk

    def test_protective_action_under_risk(self):
        agent = AgenticManager(_default_config())
        # High risk scenario with no checkpoint/replicas
        obs = _make_obs(
            bandwidth=15.0, predicted_contact_duration=20.0,
            has_checkpoint=False, num_replicas=0,
            checkpoint_staleness=100,
        )
        d = agent.decide(obs, step=1)
        # Should take protective action (checkpoint, replicate, or migrate)
        assert d.action in (Action.CHECKPOINT, Action.REPLICATE,
                           Action.MIGRATE, Action.SYNCHRONIZE)

    def test_memory_recording(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs()
        agent.record_outcome(obs, Action.MIGRATE, reward=1.0,
                            next_observation=obs, migration_time=5.0,
                            success=True)
        assert len(agent.memory) == 1
        assert agent.memory[0].action == Action.MIGRATE

    def test_memory_lookup(self):
        agent = AgenticManager(_default_config())
        obs1 = _make_obs(bandwidth=50.0, predicted_contact_duration=60.0)
        agent.record_outcome(obs1, Action.MIGRATE, reward=0.8,
                            next_observation=obs1, success=True)
        agent.record_outcome(obs1, Action.MIGRATE, reward=0.9,
                            next_observation=obs1, success=True)
        # Lookup similar observation
        obs2 = _make_obs(bandwidth=55.0, predicted_contact_duration=65.0)
        boost = agent._memory_lookup(Action.MIGRATE, obs2)
        assert boost > 0  # positive past outcomes

    def test_memory_capacity(self):
        cfg = _default_config()
        cfg["agentic_manager"]["memory_capacity"] = 5
        agent = AgenticManager(cfg)
        obs = _make_obs()
        for i in range(10):
            agent.record_outcome(obs, Action.LOCAL_EXECUTION, reward=0.5,
                                next_observation=obs)
        assert len(agent.memory) == 5  # capped

    def test_reset_preserves_memory(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs()
        agent.record_outcome(obs, Action.CHECKPOINT, reward=0.5,
                            next_observation=obs)
        agent.decide(obs, step=1)
        agent.reset()
        assert len(agent.memory) == 1  # memory preserved
        assert len(agent.context_history) == 0  # context cleared
        assert agent.decision_count == 0

    def test_stable_conditions_local_execution(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs(
            bandwidth=90.0, predicted_contact_duration=300.0,
            cpu_utilization=0.1, has_checkpoint=True,
            checkpoint_staleness=2, num_replicas=1, replica_staleness=1,
        )
        d = agent.decide(obs, step=1)
        assert d.action == Action.LOCAL_EXECUTION

    def test_infeasible_migration_filtered(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs(best_neighbor_id=None)  # no target
        d = agent.decide(obs, step=1)
        assert d.action != Action.MIGRATE

    def test_reasoning_contains_risk(self):
        agent = AgenticManager(_default_config())
        obs = _make_obs()
        d = agent.decide(obs, step=1)
        assert "Risk=" in d.reasoning
