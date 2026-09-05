"""
Tests for baseline agent strategies.
"""

import pytest
import numpy as np

from src.agents.base_agent import Action, Observation, Decision, build_observation
from src.agents.local_agent import LocalAgent
from src.agents.threshold_agent import ThresholdAgent
from src.agents.mobility_agent import MobilityAgent
from src.agents.rl_agent import RLAgent
from src.service.state import ServiceState
from src.environment.edge_node import EdgeNode
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel


def _default_config():
    return {
        "checkpoint": {"enabled": True, "interval": 50, "cpu_overhead_fraction": 0.1,
                       "storage_multiplier": 1.1, "max_checkpoints": 3},
        "replication": {"enabled": True, "max_replicas": 2, "sync_interval": 20,
                       "bandwidth_fraction": 0.2},
        "migration": {"bandwidth_fraction": 0.5, "cpu_overhead_fraction": 0.15,
                      "min_bandwidth_required": 10.0, "min_contact_duration": 30.0},
        "threshold_agent": {"bandwidth_threshold": 20.0, "latency_threshold": 100.0,
                            "cpu_threshold": 0.85},
        "mobility_agent": {"contact_duration_threshold": 60.0,
                           "use_velocity_prediction": True},
        "rl_agent": {"learning_rate": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
                     "epsilon_end": 0.05, "epsilon_decay": 0.99,
                     "batch_size": 32, "memory_size": 1000,
                     "training_episodes": 100},
        "network": {
            "max_bandwidth": 100.0, "min_bandwidth": 1.0, "latency_base": 5.0,
            "latency_per_km": 10.0, "packet_loss_base": 0.001,
            "packet_loss_distance_factor": 0.01,
            "connectivity_thresholds": {"connected": 0.7, "degraded": 0.3},
        },
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


# ====================================================================== #
# Local Agent Tests
# ====================================================================== #


class TestLocalAgent:

    def test_always_local(self):
        agent = LocalAgent(_default_config())
        obs = _make_obs()
        for step in range(10):
            d = agent.decide(obs, step)
            if step == 0 or step % 50 != 0:
                assert d.action in (Action.LOCAL_EXECUTION, Action.CHECKPOINT)

    def test_never_migrates(self):
        agent = LocalAgent(_default_config())
        obs = _make_obs(bandwidth=1.0, latency=500.0)  # bad conditions
        for step in range(100):
            d = agent.decide(obs, step)
            assert d.action != Action.MIGRATE

    def test_recovers_when_interrupted(self):
        agent = LocalAgent(_default_config())
        obs = _make_obs(is_running=False)
        d = agent.decide(obs, step=5)
        assert d.action == Action.RECOVER

    def test_periodic_checkpoint(self):
        agent = LocalAgent(_default_config())
        obs = _make_obs()
        d = agent.decide(obs, step=50)
        assert d.action == Action.CHECKPOINT


# ====================================================================== #
# Threshold Agent Tests
# ====================================================================== #


class TestThresholdAgent:

    def test_no_migration_within_thresholds(self):
        agent = ThresholdAgent(_default_config())
        obs = _make_obs(bandwidth=80.0, latency=10.0, cpu_utilization=0.3)
        d = agent.decide(obs, step=1)
        assert d.action != Action.MIGRATE

    def test_migrates_on_low_bandwidth(self):
        agent = ThresholdAgent(_default_config())
        obs = _make_obs(bandwidth=10.0)  # below 20.0 threshold
        d = agent.decide(obs, step=1)
        assert d.action == Action.MIGRATE
        assert "bandwidth" in d.reasoning.lower()

    def test_migrates_on_high_latency(self):
        agent = ThresholdAgent(_default_config())
        obs = _make_obs(latency=150.0)  # above 100.0 threshold
        d = agent.decide(obs, step=1)
        assert d.action == Action.MIGRATE

    def test_migrates_on_high_cpu(self):
        agent = ThresholdAgent(_default_config())
        obs = _make_obs(cpu_utilization=0.9)  # above 0.85
        d = agent.decide(obs, step=1)
        assert d.action == Action.MIGRATE

    def test_no_migrate_without_neighbor(self):
        agent = ThresholdAgent(_default_config())
        obs = _make_obs(bandwidth=10.0, best_neighbor_id=None)
        d = agent.decide(obs, step=1)
        # Can't migrate without a neighbor
        assert d.action != Action.MIGRATE


# ====================================================================== #
# Mobility Agent Tests
# ====================================================================== #


class TestMobilityAgent:

    def test_stable_conditions(self):
        agent = MobilityAgent(_default_config())
        obs = _make_obs(predicted_contact_duration=200.0)
        d = agent.decide(obs, step=1)
        # Should not migrate with long contact duration
        assert d.action != Action.MIGRATE

    def test_migrate_on_short_contact(self):
        agent = MobilityAgent(_default_config())
        obs = _make_obs(
            predicted_contact_duration=30.0,  # below 60s threshold
            best_neighbor_bandwidth=80.0,
            best_neighbor_contact_duration=100.0,
            state_size=10.0,  # small state for fast transfer
        )
        d = agent.decide(obs, step=1)
        assert d.action == Action.MIGRATE

    def test_checkpoint_when_contact_declining(self):
        agent = MobilityAgent(_default_config())
        obs = _make_obs(
            predicted_contact_duration=100.0,  # below 2x threshold (120)
            has_checkpoint=False,
            best_neighbor_id=None,  # no migration target
        )
        d = agent.decide(obs, step=1)
        assert d.action == Action.CHECKPOINT

    def test_create_replica_when_none(self):
        agent = MobilityAgent(_default_config())
        obs = _make_obs(
            predicted_contact_duration=200.0,
            num_replicas=0,
            best_neighbor_bandwidth=50.0,
        )
        d = agent.decide(obs, step=1)
        assert d.action == Action.REPLICATE


# ====================================================================== #
# RL Agent Tests
# ====================================================================== #


class TestRLAgent:

    def test_decide_returns_valid_action(self):
        agent = RLAgent(_default_config())
        obs = _make_obs()
        d = agent.decide(obs, step=1)
        assert isinstance(d.action, Action)

    def test_exploration_in_training(self):
        agent = RLAgent(_default_config())
        agent.epsilon = 1.0  # 100% exploration
        obs = _make_obs()
        actions = set()
        for step in range(100):
            d = agent.decide(obs, step)
            actions.add(d.action)
        # With full exploration, should see multiple actions
        assert len(actions) > 1

    def test_q_table_update(self):
        agent = RLAgent(_default_config())
        obs1 = _make_obs(bandwidth=80.0)
        obs2 = _make_obs(bandwidth=60.0)
        d = agent.decide(obs1, step=1)
        agent.record_outcome(obs1, d.action, reward=1.0, next_observation=obs2)
        # Q-table should now have an entry
        state_key = agent._discretize_state(obs1)
        assert np.any(agent.q_table[state_key] != 0)

    def test_epsilon_decay(self):
        agent = RLAgent(_default_config())
        initial_eps = agent.epsilon
        obs = _make_obs()
        for step in range(10):
            d = agent.decide(obs, step)
            agent.record_outcome(obs, d.action, reward=0.0, next_observation=obs)
        assert agent.epsilon < initial_eps

    def test_set_training_false(self):
        agent = RLAgent(_default_config())
        agent.set_training(False)
        assert agent.epsilon == 0.0
        assert agent.training is False

    def test_infeasible_action_filtering(self):
        agent = RLAgent(_default_config())
        # Force action to MIGRATE but no neighbor
        obs = _make_obs(best_neighbor_id=None)
        for _ in range(50):
            d = agent.decide(obs, step=1)
            if d.action in (Action.MIGRATE, Action.REPLICATE):
                # Should have been filtered
                assert False, f"Infeasible action {d.action} not filtered"


# ====================================================================== #
# Observation Tests
# ====================================================================== #


class TestObservation:

    def test_to_vector(self):
        obs = _make_obs()
        vec = obs.to_vector()
        assert isinstance(vec, list)
        assert len(vec) == 21
        assert all(isinstance(v, float) for v in vec)

    def test_build_observation(self):
        edge = EdgeNode("uav_0", 4.0, 8192.0, 64000.0)
        uav0 = UAVNode("uav_0", x=0, y=0, velocity=10.0,
                       communication_radius=200.0, edge=edge)
        edge1 = EdgeNode("uav_1", 4.0, 8192.0, 64000.0)
        uav1 = UAVNode("uav_1", x=50, y=0, velocity=5.0,
                       communication_radius=200.0, edge=edge1)

        net = NetworkModel(_default_config())
        svc = ServiceState("svc_0", "uav_0", state_size=100.0)

        obs = build_observation(svc, uav0, [uav0, uav1], net, current_step=10)
        assert obs.num_neighbors == 1
        assert obs.best_neighbor_id == "uav_1"
        assert obs.bandwidth > 0
