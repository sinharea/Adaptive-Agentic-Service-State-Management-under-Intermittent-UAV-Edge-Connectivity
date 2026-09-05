"""
Tests for the intermittent network model.
"""

import math
import pytest
import numpy as np

from src.environment.edge_node import EdgeNode
from src.environment.network import NetworkModel, ConnectivityState, LinkMetrics
from src.environment.uav import UAVNode


# Default config for tests
def _default_config():
    return {
        "network": {
            "max_bandwidth": 100.0,
            "min_bandwidth": 1.0,
            "latency_base": 5.0,
            "latency_per_km": 10.0,
            "packet_loss_base": 0.001,
            "packet_loss_distance_factor": 0.01,
            "connectivity_thresholds": {
                "connected": 0.7,
                "degraded": 0.3,
            },
        }
    }


def _make_uav(node_id, x, y, velocity=0.0, direction=0.0, comm_radius=200.0):
    edge = EdgeNode(node_id, cpu_capacity=4.0,
                    memory_capacity=8192.0, storage_capacity=64000.0)
    return UAVNode(node_id=node_id, x=x, y=y, velocity=velocity,
                   direction=direction, communication_radius=comm_radius,
                   edge=edge)


class TestBandwidth:
    """Tests for bandwidth computation."""

    def test_bandwidth_at_zero_distance(self):
        net = NetworkModel(_default_config())
        bw = net.compute_bandwidth(0.0, 200.0)
        assert abs(bw - 100.0) < 1e-9

    def test_bandwidth_at_max_range(self):
        net = NetworkModel(_default_config())
        bw = net.compute_bandwidth(200.0, 200.0)
        assert bw == 0.0

    def test_bandwidth_at_half_range(self):
        net = NetworkModel(_default_config())
        bw = net.compute_bandwidth(100.0, 200.0)
        assert abs(bw - 50.0) < 1e-9

    def test_bandwidth_beyond_range(self):
        net = NetworkModel(_default_config())
        bw = net.compute_bandwidth(300.0, 200.0)
        assert bw == 0.0

    def test_bandwidth_decreases_with_distance(self):
        net = NetworkModel(_default_config())
        bw_near = net.compute_bandwidth(50.0, 200.0)
        bw_far = net.compute_bandwidth(150.0, 200.0)
        assert bw_near > bw_far


class TestLatency:
    """Tests for latency computation."""

    def test_latency_at_zero_distance(self):
        net = NetworkModel(_default_config())
        lat = net.compute_latency(0.0)
        assert abs(lat - 5.0) < 1e-9

    def test_latency_at_one_km(self):
        net = NetworkModel(_default_config())
        lat = net.compute_latency(1000.0)
        assert abs(lat - 15.0) < 1e-9  # 5 + 10*1

    def test_latency_increases_with_distance(self):
        net = NetworkModel(_default_config())
        lat_near = net.compute_latency(100.0)
        lat_far = net.compute_latency(500.0)
        assert lat_far > lat_near


class TestPacketLoss:
    """Tests for packet loss computation."""

    def test_packet_loss_at_zero_distance(self):
        net = NetworkModel(_default_config())
        loss = net.compute_packet_loss(0.0, 200.0)
        assert abs(loss - 0.001) < 1e-9

    def test_packet_loss_beyond_range(self):
        net = NetworkModel(_default_config())
        loss = net.compute_packet_loss(300.0, 200.0)
        assert loss == 1.0

    def test_packet_loss_increases_with_distance(self):
        net = NetworkModel(_default_config())
        loss_near = net.compute_packet_loss(50.0, 200.0)
        loss_far = net.compute_packet_loss(150.0, 200.0)
        assert loss_far > loss_near

    def test_packet_loss_capped_at_one(self):
        net = NetworkModel(_default_config())
        # With very high distance factor
        loss = net.compute_packet_loss(199.0, 200.0)
        assert 0.0 <= loss <= 1.0


class TestConnectivityState:
    """Tests for connectivity state determination."""

    def test_connected_state(self):
        net = NetworkModel(_default_config())
        # bandwidth ratio >= 0.7 → connected
        state = net.compute_connectivity_state(80.0)  # 80/100 = 0.8
        assert state == ConnectivityState.CONNECTED

    def test_degraded_state(self):
        net = NetworkModel(_default_config())
        # 0.3 <= ratio < 0.7 → degraded
        state = net.compute_connectivity_state(50.0)  # 50/100 = 0.5
        assert state == ConnectivityState.DEGRADED

    def test_disconnected_low_bandwidth(self):
        net = NetworkModel(_default_config())
        # ratio < 0.3 → disconnected
        state = net.compute_connectivity_state(20.0)  # 20/100 = 0.2
        assert state == ConnectivityState.DISCONNECTED

    def test_disconnected_zero_bandwidth(self):
        net = NetworkModel(_default_config())
        state = net.compute_connectivity_state(0.0)
        assert state == ConnectivityState.DISCONNECTED


class TestLinkEvaluation:
    """Tests for end-to-end link evaluation."""

    def test_evaluate_link_close_nodes(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)
        link = net.evaluate_link(uav1, uav2)
        assert link.state == ConnectivityState.CONNECTED
        assert link.bandwidth > 0
        assert link.distance == 50.0

    def test_evaluate_link_far_nodes(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 500.0, 0.0)
        link = net.evaluate_link(uav1, uav2)
        assert link.state == ConnectivityState.DISCONNECTED
        assert link.bandwidth == 0.0

    def test_evaluate_all_links(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)
        uav3 = _make_uav("u3", 500.0, 0.0)
        links = net.evaluate_all_links([uav1, uav2, uav3])
        # 3 nodes → 6 directed links (3 pairs × 2 directions)
        assert len(links) == 6
        assert ("u1", "u2") in links
        assert ("u2", "u1") in links


class TestFailureInjection:
    """Tests for link failure injection."""

    def test_no_failures_by_default(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)
        link = net.evaluate_link(uav1, uav2)
        assert link.state == ConnectivityState.CONNECTED

    def test_inject_deterministic_failure(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)

        # Manually inject failure
        net._failed_links.add(("u1", "u2"))
        link = net.evaluate_link(uav1, uav2)
        assert link.state == ConnectivityState.DISCONNECTED
        assert link.bandwidth == 0.0
        assert link.packet_loss == 1.0

    def test_inject_random_failures(self):
        net = NetworkModel(_default_config())
        net.set_failure_probability(1.0)  # 100% failure
        nodes = [_make_uav(f"u{i}", i * 50.0, 0.0) for i in range(3)]
        rng = np.random.default_rng(42)
        net.inject_failures(nodes, rng)
        # All links should be failed
        assert len(net._failed_links) > 0

    def test_clear_failures(self):
        net = NetworkModel(_default_config())
        net._failed_links.add(("u1", "u2"))
        net.clear_failures()
        assert len(net._failed_links) == 0

    def test_zero_failure_probability(self):
        net = NetworkModel(_default_config())
        net.set_failure_probability(0.0)
        nodes = [_make_uav(f"u{i}", i * 50.0, 0.0) for i in range(5)]
        rng = np.random.default_rng(42)
        net.inject_failures(nodes, rng)
        assert len(net._failed_links) == 0


class TestTransferFeasibility:
    """Tests for data transfer feasibility checks."""

    def test_transfer_feasible(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0, comm_radius=200.0)
        uav2 = _make_uav("u2", 50.0, 0.0, comm_radius=200.0)
        feasible, time = net.can_transfer(uav1, uav2, data_size_mb=10.0,
                                          bandwidth_fraction=0.5)
        assert feasible is True
        assert time > 0.0
        assert time < float("inf")

    def test_transfer_disconnected(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 500.0, 0.0)
        feasible, time = net.can_transfer(uav1, uav2, data_size_mb=10.0,
                                          bandwidth_fraction=0.5)
        assert feasible is False
        assert time == float("inf")

    def test_transfer_time_increases_with_size(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)
        _, time_small = net.can_transfer(uav1, uav2, data_size_mb=10.0,
                                          bandwidth_fraction=0.5)
        _, time_large = net.can_transfer(uav1, uav2, data_size_mb=100.0,
                                          bandwidth_fraction=0.5)
        assert time_large > time_small

    def test_transfer_with_min_bandwidth(self):
        net = NetworkModel(_default_config())
        # Node at 180m → low bandwidth
        uav1 = _make_uav("u1", 0.0, 0.0, comm_radius=200.0)
        uav2 = _make_uav("u2", 180.0, 0.0, comm_radius=200.0)
        feasible, _ = net.can_transfer(uav1, uav2, data_size_mb=10.0,
                                       bandwidth_fraction=0.5,
                                       min_bandwidth=20.0)
        # bandwidth at 180m = 100 * (1 - 180/200) = 10 Mbps
        # effective = 10 * 0.5 = 5 Mbps < 20 Mbps minimum
        assert feasible is False


class TestBestNeighbor:
    """Tests for best neighbor selection."""

    def test_best_neighbor_closest(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 50.0, 0.0)
        uav3 = _make_uav("u3", 150.0, 0.0)
        result = net.get_best_neighbor(uav1, [uav1, uav2, uav3])
        assert result is not None
        best_node, best_link = result
        assert best_node.node_id == "u2"

    def test_no_neighbor(self):
        net = NetworkModel(_default_config())
        uav1 = _make_uav("u1", 0.0, 0.0)
        uav2 = _make_uav("u2", 500.0, 0.0)
        result = net.get_best_neighbor(uav1, [uav1, uav2])
        assert result is None
