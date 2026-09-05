"""
Tests for migration mechanism.
"""

import pytest
from src.service.state import ServiceState
from src.service.migration import MigrationManager
from src.environment.edge_node import EdgeNode
from src.environment.uav import UAVNode
from src.environment.network import NetworkModel


def _default_config():
    return {
        "migration": {
            "bandwidth_fraction": 0.5,
            "cpu_overhead_fraction": 0.15,
            "min_bandwidth_required": 10.0,
            "min_contact_duration": 30.0,
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


def _make_uav(node_id, x, y, comm_radius=200.0, storage=64000.0,
              memory=8192.0):
    edge = EdgeNode(node_id, cpu_capacity=4.0,
                    memory_capacity=memory, storage_capacity=storage)
    return UAVNode(node_id=node_id, x=x, y=y,
                   communication_radius=comm_radius, edge=edge)


def _make_service(**kwargs):
    defaults = dict(
        service_id="svc_0", current_node="uav_0",
        state_size=100.0, state_update_rate=1.0, priority="medium",
    )
    defaults.update(kwargs)
    return ServiceState(**defaults)


class TestMigrationManager:
    """Tests for MigrationManager."""

    def test_can_migrate_feasible(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 30, 0)  # close
        feasible, reason, est_time = mgr.can_migrate(svc, src, tgt, net)
        assert feasible is True
        assert est_time > 0

    def test_can_migrate_disconnected(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 500, 0)  # far
        feasible, reason, _ = mgr.can_migrate(svc, src, tgt, net)
        assert feasible is False
        assert "disconnected" in reason.lower()

    def test_can_migrate_low_bandwidth(self):
        cfg = _default_config()
        cfg["migration"]["min_bandwidth_required"] = 50.0
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        # At 120m: bw = 100*(1-120/200) = 40 Mbps, effective = 20 < 50
        # ratio=0.4 is above degraded threshold (0.3) so it's degraded, not disconnected
        tgt = _make_uav("uav_1", 120, 0)
        feasible, reason, _ = mgr.can_migrate(svc, src, tgt, net)
        assert feasible is False
        assert "bandwidth" in reason.lower()

    def test_can_migrate_already_migrating(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        svc.is_migrating = True
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 30, 0)
        feasible, reason, _ = mgr.can_migrate(svc, src, tgt, net)
        assert feasible is False

    def test_migrate_success(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 30, 0)

        result = mgr.migrate(svc, src, tgt, net, step=10)
        assert result.success is True
        assert result.source_node == "uav_0"
        assert result.destination_node == "uav_1"
        assert result.data_transferred == 100.0
        assert result.migration_time > 0
        assert svc.current_node == "uav_1"
        assert svc.migration_count == 1
        assert svc.is_running is True

    def test_migrate_failure(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 500, 0)

        result = mgr.migrate(svc, src, tgt, net, step=10)
        assert result.success is False
        assert svc.current_node == "uav_0"  # unchanged
        assert svc.migration_count == 0

    def test_migrate_tracks_interruption(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service()
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 30, 0)

        result = mgr.migrate(svc, src, tgt, net, step=10)
        assert result.interruption_time >= 0
        assert svc.total_interruption_time >= 0

    def test_migrate_transfers_resources(self):
        cfg = _default_config()
        mgr = MigrationManager(cfg)
        net = NetworkModel(cfg)
        svc = _make_service(state_size=100.0)
        src = _make_uav("uav_0", 0, 0)
        tgt = _make_uav("uav_1", 30, 0)

        src_mem_before = src.edge.memory_used
        tgt_mem_before = tgt.edge.memory_used
        mgr.migrate(svc, src, tgt, net, step=10)
        # Target should have more memory used now
        assert tgt.edge.memory_used > tgt_mem_before
