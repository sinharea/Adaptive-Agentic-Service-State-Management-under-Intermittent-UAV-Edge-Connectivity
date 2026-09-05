"""
Tests for the stateful service model.
"""

import pytest
from src.service.state import ServiceState, create_service


class TestServiceState:
    """Tests for ServiceState."""

    def _make_service(self, **kwargs) -> ServiceState:
        defaults = dict(
            service_id="svc_0", current_node="uav_0",
            state_size=100.0, state_update_rate=1.0, priority="medium",
        )
        defaults.update(kwargs)
        return ServiceState(**defaults)

    def test_initial_state(self):
        svc = self._make_service()
        assert svc.state_version == 0
        assert svc.state_size == 100.0
        assert svc.is_running is True
        assert svc.is_migrating is False
        assert svc.has_checkpoint is False
        assert svc.num_replicas == 0

    def test_update_state(self):
        svc = self._make_service()
        svc.update_state(step=1)
        assert svc.state_version == 1
        assert abs(svc.state_size - 101.0) < 1e-9

    def test_update_state_not_running(self):
        svc = self._make_service()
        svc.is_running = False
        svc.update_state(step=1)
        assert svc.state_version == 0  # unchanged
        assert svc.state_size == 100.0

    def test_update_state_migrating(self):
        svc = self._make_service()
        svc.is_migrating = True
        svc.update_state(step=1)
        assert svc.state_version == 0  # unchanged

    def test_multiple_updates(self):
        svc = self._make_service(state_update_rate=2.0)
        for i in range(10):
            svc.update_state(step=i + 1)
        assert svc.state_version == 10
        assert abs(svc.state_size - 120.0) < 1e-9  # 100 + 10*2

    def test_interruption_tracking(self):
        svc = self._make_service()
        svc.begin_interruption(step=5)
        assert svc.is_running is False
        assert svc.interruption_start == 5

        duration = svc.end_interruption(step=8)
        assert duration == 3.0
        assert svc.is_running is True
        assert svc.total_interruption_time == 3.0

    def test_sla_violation(self):
        svc = self._make_service(sla_max_interruption=5.0)
        svc.begin_interruption(step=0)
        svc.end_interruption(step=10)
        assert svc.sla_violations == 1

    def test_no_sla_violation(self):
        svc = self._make_service(sla_max_interruption=10.0)
        svc.begin_interruption(step=0)
        svc.end_interruption(step=5)
        assert svc.sla_violations == 0

    def test_priority_weight(self):
        assert self._make_service(priority="low").priority_weight == 0.5
        assert self._make_service(priority="medium").priority_weight == 1.0
        assert self._make_service(priority="high").priority_weight == 2.0

    def test_checkpoint_staleness(self):
        svc = self._make_service()
        for i in range(5):
            svc.update_state(step=i + 1)
        assert svc.checkpoint_staleness == 5  # no checkpoint
        svc.checkpoint_version = 3
        assert svc.checkpoint_staleness == 2  # version 5 - 3

    def test_replica_staleness(self):
        svc = self._make_service()
        for i in range(10):
            svc.update_state(step=i + 1)
        assert svc.replica_staleness == 10  # no replicas
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 7}
        assert svc.replica_staleness == 3  # 10 - 7

    def test_best_replica_version(self):
        svc = self._make_service()
        assert svc.best_replica_version() == -1
        svc.replica_versions = {"uav_1": 5, "uav_2": 8}
        assert svc.best_replica_version() == 8

    def test_get_observation(self):
        svc = self._make_service()
        svc.update_state(step=1)
        obs = svc.get_observation(current_step=10)
        assert "state_size" in obs
        assert "state_version" in obs
        assert "has_checkpoint" in obs
        assert "num_replicas" in obs
        assert obs["time_since_last_sync"] == 10

    def test_end_interruption_when_not_interrupted(self):
        svc = self._make_service()
        duration = svc.end_interruption(step=5)
        assert duration == 0.0

    def test_begin_interruption_idempotent(self):
        svc = self._make_service()
        svc.begin_interruption(step=5)
        svc.begin_interruption(step=8)  # should not reset
        assert svc.interruption_start == 5


class TestCreateService:
    """Tests for factory function."""

    def test_create_from_config(self):
        config = {
            "service": {
                "state_size": 250.0,
                "state_update_rate": 2.0,
                "priority": "high",
                "sla_max_interruption": 15.0,
                "sla_max_recovery_time": 45.0,
            }
        }
        svc = create_service(config, node_id="uav_0")
        assert svc.state_size == 250.0
        assert svc.state_update_rate == 2.0
        assert svc.priority == "high"
        assert svc.current_node == "uav_0"
        assert svc.sla_max_interruption == 15.0
