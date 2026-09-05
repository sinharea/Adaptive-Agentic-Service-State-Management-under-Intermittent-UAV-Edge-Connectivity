"""
Tests for checkpoint and recovery mechanisms.
"""

import pytest
from src.service.state import ServiceState
from src.service.checkpoint import CheckpointManager, CheckpointResult
from src.service.recovery import RecoveryManager
from src.environment.edge_node import EdgeNode


def _make_edge(storage=64000.0, cpu=4.0):
    return EdgeNode("uav_0", cpu_capacity=cpu,
                    memory_capacity=8192.0, storage_capacity=storage)


def _make_service(**kwargs):
    defaults = dict(
        service_id="svc_0", current_node="uav_0",
        state_size=100.0, state_update_rate=1.0, priority="medium",
    )
    defaults.update(kwargs)
    return ServiceState(**defaults)


def _default_config():
    return {
        "checkpoint": {
            "enabled": True,
            "interval": 50,
            "cpu_overhead_fraction": 0.1,
            "storage_multiplier": 1.1,
            "max_checkpoints": 3,
        },
        "service": {
            "sla_max_recovery_time": 30.0,
        },
    }


class TestCheckpointManager:
    """Tests for checkpoint creation and management."""

    def test_create_checkpoint_success(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()
        for i in range(5):
            svc.update_state(i + 1)

        result = mgr.create_checkpoint(svc, edge, step=5)
        assert result.success is True
        assert result.checkpoint is not None
        assert result.checkpoint.version == 5
        assert result.storage_cost > 0
        assert svc.checkpoint_version == 5

    def test_checkpoint_disabled(self):
        cfg = _default_config()
        cfg["checkpoint"]["enabled"] = False
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()
        result = mgr.create_checkpoint(svc, edge, step=1)
        assert result.success is False
        assert "disabled" in result.reason.lower()

    def test_checkpoint_service_not_running(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        svc.is_running = False
        edge = _make_edge()
        result = mgr.create_checkpoint(svc, edge, step=1)
        assert result.success is False

    def test_checkpoint_insufficient_storage(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service(state_size=100.0)
        edge = _make_edge(storage=50.0)  # not enough
        result = mgr.create_checkpoint(svc, edge, step=1)
        assert result.success is False
        assert "storage" in result.reason.lower()

    def test_max_checkpoints_eviction(self):
        cfg = _default_config()
        cfg["checkpoint"]["max_checkpoints"] = 2
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()

        # Create 3 checkpoints
        for i in range(3):
            svc.update_state(i + 1)
            mgr.create_checkpoint(svc, edge, step=i + 1)

        # Should have only 2 checkpoints
        ckpts = mgr.get_checkpoints(svc.service_id)
        assert len(ckpts) == 2
        # Oldest evicted: versions should be 2 and 3
        assert ckpts[0].version == 2
        assert ckpts[1].version == 3

    def test_get_latest_checkpoint(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()

        assert mgr.get_latest_checkpoint(svc.service_id) is None

        svc.update_state(1)
        mgr.create_checkpoint(svc, edge, step=1)
        latest = mgr.get_latest_checkpoint(svc.service_id)
        assert latest is not None
        assert latest.version == 1

    def test_has_checkpoint(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()
        assert mgr.has_checkpoint(svc.service_id) is False
        svc.update_state(1)
        mgr.create_checkpoint(svc, edge, step=1)
        assert mgr.has_checkpoint(svc.service_id) is True

    def test_remove_checkpoints(self):
        cfg = _default_config()
        mgr = CheckpointManager(cfg)
        svc = _make_service()
        edge = _make_edge()
        svc.update_state(1)
        mgr.create_checkpoint(svc, edge, step=1)
        freed = mgr.remove_checkpoints(svc.service_id, edge)
        assert freed > 0
        assert mgr.has_checkpoint(svc.service_id) is False


class TestRecoveryManager:
    """Tests for recovery from checkpoint and replica."""

    def test_recover_from_checkpoint(self):
        cfg = _default_config()
        ckpt_mgr = CheckpointManager(cfg)
        rec_mgr = RecoveryManager(cfg)
        svc = _make_service()
        edge = _make_edge()

        # Advance to version 5, checkpoint at version 3
        for i in range(3):
            svc.update_state(i + 1)
        ckpt_mgr.create_checkpoint(svc, edge, step=3)
        for i in range(2):
            svc.update_state(i + 4)

        # Simulate disruption
        svc.begin_interruption(step=5)

        # Recover
        ckpt = ckpt_mgr.get_latest_checkpoint(svc.service_id)
        result = rec_mgr.recover_from_checkpoint(svc, ckpt, step=6)
        assert result.success is True
        assert result.source == "checkpoint"
        assert result.state_loss == 2  # version 5 - version 3
        assert svc.state_version == 3
        assert svc.is_running is True

    def test_recover_from_replica(self):
        cfg = _default_config()
        rec_mgr = RecoveryManager(cfg)
        svc = _make_service()
        for i in range(10):
            svc.update_state(i + 1)
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 8}

        svc.begin_interruption(step=10)
        result = rec_mgr.recover_from_replica(svc, "uav_1", 8, 108.0, step=11)
        assert result.success is True
        assert result.source == "replica"
        assert result.state_loss == 2  # version 10 - 8
        assert svc.state_version == 8

    def test_attempt_recovery_prefers_replica(self):
        cfg = _default_config()
        ckpt_mgr = CheckpointManager(cfg)
        rec_mgr = RecoveryManager(cfg)
        svc = _make_service()
        edge = _make_edge()

        # Create checkpoint at version 3
        for i in range(3):
            svc.update_state(i + 1)
        ckpt_mgr.create_checkpoint(svc, edge, step=3)

        # Advance and add replica at version 7
        for i in range(4):
            svc.update_state(i + 4)
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 7}

        svc.begin_interruption(step=7)
        result = rec_mgr.attempt_recovery(svc, ckpt_mgr, step=8)
        assert result.success is True
        assert result.source == "replica"  # replica is more recent

    def test_attempt_recovery_falls_back_to_checkpoint(self):
        cfg = _default_config()
        ckpt_mgr = CheckpointManager(cfg)
        rec_mgr = RecoveryManager(cfg)
        svc = _make_service()
        edge = _make_edge()

        for i in range(5):
            svc.update_state(i + 1)
        ckpt_mgr.create_checkpoint(svc, edge, step=5)

        svc.begin_interruption(step=5)
        result = rec_mgr.attempt_recovery(svc, ckpt_mgr, step=6)
        assert result.success is True
        assert result.source == "checkpoint"

    def test_attempt_recovery_no_source(self):
        cfg = _default_config()
        ckpt_mgr = CheckpointManager(cfg)
        rec_mgr = RecoveryManager(cfg)
        svc = _make_service()

        result = rec_mgr.attempt_recovery(svc, ckpt_mgr, step=1)
        assert result.success is False
        assert "no checkpoint" in result.reason.lower()
