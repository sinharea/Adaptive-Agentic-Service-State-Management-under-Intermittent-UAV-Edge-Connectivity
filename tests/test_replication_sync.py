"""
Tests for replication and synchronization.
"""

import pytest
import numpy as np

from src.service.state import ServiceState
from src.service.replication import ReplicationManager
from src.service.synchronization import SynchronizationManager
from src.environment.edge_node import EdgeNode
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel


def _default_config():
    return {
        "replication": {
            "enabled": True,
            "max_replicas": 2,
            "sync_interval": 20,
            "bandwidth_fraction": 0.2,
        },
        "network": {
            "max_bandwidth": 100.0,
            "min_bandwidth": 1.0,
            "latency_base": 5.0,
            "latency_per_km": 10.0,
            "packet_loss_base": 0.001,
            "packet_loss_distance_factor": 0.01,
            "connectivity_thresholds": {"connected": 0.7, "degraded": 0.3},
        },
    }


def _make_uav(node_id, x, y, comm_radius=200.0, storage=64000.0):
    edge = EdgeNode(node_id, cpu_capacity=4.0,
                    memory_capacity=8192.0, storage_capacity=storage)
    return UAVNode(node_id=node_id, x=x, y=y,
                   communication_radius=comm_radius, edge=edge)


def _make_service(**kwargs):
    defaults = dict(
        service_id="svc_0", current_node="uav_0",
        state_size=100.0, state_update_rate=1.0, priority="medium",
    )
    defaults.update(kwargs)
    return ServiceState(**defaults)


class TestReplicationManager:
    """Tests for ReplicationManager."""

    def test_replicate_success(self):
        cfg = _default_config()
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        for i in range(5):
            svc.update_state(i + 1)
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)

        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is True
        assert "uav_1" in svc.replicated_nodes
        assert svc.replica_versions["uav_1"] == 5

    def test_replicate_disabled(self):
        cfg = _default_config()
        cfg["replication"]["enabled"] = False
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)
        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is False

    def test_replicate_max_reached(self):
        cfg = _default_config()
        cfg["replication"]["max_replicas"] = 1
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 0}
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_2", 50, 0)
        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is False
        assert "max" in result.reason.lower()

    def test_replicate_already_exists(self):
        cfg = _default_config()
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)
        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is False

    def test_replicate_out_of_range(self):
        cfg = _default_config()
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 500, 0)
        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is False

    def test_replicate_insufficient_storage(self):
        cfg = _default_config()
        mgr = ReplicationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service(state_size=100.0)
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0, storage=50.0)
        result = mgr.replicate(svc, src, tgt, net)
        assert result.success is False
        assert "storage" in result.reason.lower()

    def test_remove_replica(self):
        cfg = _default_config()
        mgr = ReplicationManager(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 3}
        edge = EdgeNode("uav_1", cpu_capacity=4.0,
                        memory_capacity=8192.0, storage_capacity=64000.0)
        freed = mgr.remove_replica(svc, "uav_1", edge)
        assert freed > 0
        assert "uav_1" not in svc.replicated_nodes


class TestSynchronizationManager:
    """Tests for SynchronizationManager."""

    def test_sync_success(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 3}
        for i in range(10):
            svc.update_state(i + 1)

        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)
        result = sync_mgr.synchronize(svc, src, tgt, net, step=10)
        assert result.success is True
        assert result.new_version == 10
        assert result.old_version == 3
        assert result.data_transferred > 0

    def test_sync_already_up_to_date(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 5}
        for i in range(5):
            svc.update_state(i + 1)

        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)
        result = sync_mgr.synchronize(svc, src, tgt, net, step=5)
        assert result.success is True
        assert result.data_transferred == 0.0

    def test_sync_no_replica(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 50, 0)
        result = sync_mgr.synchronize(svc, src, tgt, net, step=1)
        assert result.success is False

    def test_sync_out_of_range(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1"]
        svc.replica_versions = {"uav_1": 0}
        for i in range(5):
            svc.update_state(i + 1)

        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 500, 0)
        result = sync_mgr.synchronize(svc, src, tgt, net, step=5)
        assert result.success is False

    def test_sync_all(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.replicated_nodes = ["uav_1", "uav_2"]
        svc.replica_versions = {"uav_1": 0, "uav_2": 3}
        for i in range(5):
            svc.update_state(i + 1)

        src = _make_uav("uav_0", 0, 0)
        uav_map = {
            "uav_0": src,
            "uav_1": _make_uav("uav_1", 50, 0),
            "uav_2": _make_uav("uav_2", 80, 0),
        }
        results = sync_mgr.synchronize_all(svc, src, uav_map, net, step=5)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_sync_lag(self):
        cfg = _default_config()
        sync_mgr = SynchronizationManager(cfg)
        svc = _make_service()
        svc.last_sync_time = 10
        assert sync_mgr.get_sync_lag(svc, current_step=25) == 15
