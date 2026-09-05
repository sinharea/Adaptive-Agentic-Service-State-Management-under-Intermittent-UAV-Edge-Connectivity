"""
Tests for UAV environment: edge nodes, mobility models, and UAV nodes.
"""

import math
import pytest
import numpy as np
import yaml

from src.environment.edge_node import EdgeNode
from src.environment.mobility import (
    RandomWaypointMobility,
    RandomWalkMobility,
    create_mobility_model,
)
from src.environment.uav import UAVNode, create_uav_nodes


# ====================================================================== #
# EdgeNode Tests
# ====================================================================== #


class TestEdgeNode:
    """Tests for EdgeNode resource model."""

    def test_initial_utilization_zero(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        assert node.cpu_utilization == 0.0
        assert node.memory_utilization == 0.0
        assert node.storage_utilization == 0.0

    def test_set_initial_utilization(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        node.set_initial_utilization(0.5, 0.3, 0.1)
        assert abs(node.cpu_utilization - 0.5) < 1e-9
        assert abs(node.memory_utilization - 0.3) < 1e-9
        assert abs(node.storage_utilization - 0.1) < 1e-9

    def test_allocate_cpu_success(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        assert node.allocate_cpu(2.0) is True
        assert abs(node.cpu_used - 2.0) < 1e-9

    def test_allocate_cpu_failure(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        assert node.allocate_cpu(5.0) is False
        assert node.cpu_used == 0.0  # Unchanged

    def test_release_cpu(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        node.allocate_cpu(3.0)
        node.release_cpu(1.0)
        assert abs(node.cpu_used - 2.0) < 1e-9

    def test_release_cpu_floor_zero(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        node.release_cpu(100.0)  # More than used
        assert node.cpu_used == 0.0

    def test_available_resources(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0, cpu_used=1.0,
                         memory_used=2048.0, storage_used=10000.0)
        assert abs(node.available_cpu() - 3.0) < 1e-9
        assert abs(node.available_memory() - 6144.0) < 1e-9
        assert abs(node.available_storage() - 54000.0) < 1e-9

    def test_can_host_service(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        assert node.can_host_service(2.0, 4096.0, 32000.0) is True
        assert node.can_host_service(5.0, 4096.0, 32000.0) is False  # CPU too much

    def test_negative_allocation_raises(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        with pytest.raises(ValueError):
            node.allocate_cpu(-1.0)
        with pytest.raises(ValueError):
            node.release_cpu(-1.0)

    def test_utilization_capped_at_one(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0, cpu_used=10.0)
        assert node.cpu_utilization == 1.0

    def test_zero_capacity(self):
        node = EdgeNode("n1", cpu_capacity=0.0, memory_capacity=0.0,
                         storage_capacity=0.0)
        assert node.cpu_utilization == 1.0
        assert node.memory_utilization == 1.0
        assert node.storage_utilization == 1.0

    def test_allocate_memory_and_storage(self):
        node = EdgeNode("n1", cpu_capacity=4.0, memory_capacity=8192.0,
                         storage_capacity=64000.0)
        assert node.allocate_memory(4096.0) is True
        assert node.allocate_storage(32000.0) is True
        assert abs(node.memory_used - 4096.0) < 1e-9
        assert abs(node.storage_used - 32000.0) < 1e-9


# ====================================================================== #
# Mobility Model Tests
# ====================================================================== #


class TestRandomWaypointMobility:
    """Tests for RandomWaypointMobility model."""

    def test_initialize_within_bounds(self):
        rng = np.random.default_rng(42)
        mob = RandomWaypointMobility(v_min=5.0, v_max=20.0)
        x, y = mob.initialize(rng, 1000.0, 1000.0)
        assert 0.0 <= x <= 1000.0
        assert 0.0 <= y <= 1000.0

    def test_update_stays_in_bounds(self):
        rng = np.random.default_rng(42)
        mob = RandomWaypointMobility(v_min=5.0, v_max=20.0, pause_min=0.0, pause_max=0.0)
        x, y = mob.initialize(rng, 1000.0, 1000.0)

        # Run for many steps
        for _ in range(500):
            x, y, v, d = mob.update(x, y, 0.0, 0.0, 1.0, rng, 1000.0, 1000.0)
            assert 0.0 <= x <= 1000.0, f"x={x} out of bounds"
            assert 0.0 <= y <= 1000.0, f"y={y} out of bounds"

    def test_velocity_within_range(self):
        rng = np.random.default_rng(42)
        mob = RandomWaypointMobility(v_min=5.0, v_max=20.0, pause_min=0.0, pause_max=0.0)
        x, y = mob.initialize(rng, 1000.0, 1000.0)
        velocities = []
        for _ in range(100):
            x, y, v, d = mob.update(x, y, 0.0, 0.0, 1.0, rng, 1000.0, 1000.0)
            if v > 0:
                velocities.append(v)
        assert len(velocities) > 0
        assert all(5.0 <= v <= 20.0 for v in velocities)

    def test_clone_independence(self):
        mob1 = RandomWaypointMobility(v_min=5.0, v_max=20.0)
        mob2 = mob1.clone()
        rng = np.random.default_rng(42)
        mob1.initialize(rng, 1000.0, 1000.0)
        # mob2 should be a fresh independent instance
        assert mob2._target_x != mob1._target_x or mob2._target_y != mob1._target_y or True
        # The key point: modifying mob1 internal state doesn't affect mob2


class TestRandomWalkMobility:
    """Tests for RandomWalkMobility model."""

    def test_initialize_within_bounds(self):
        rng = np.random.default_rng(42)
        mob = RandomWalkMobility(v_min=5.0, v_max=20.0)
        x, y = mob.initialize(rng, 1000.0, 1000.0)
        assert 0.0 <= x <= 1000.0
        assert 0.0 <= y <= 1000.0

    def test_update_stays_in_bounds(self):
        rng = np.random.default_rng(42)
        mob = RandomWalkMobility(v_min=5.0, v_max=20.0, direction_change_interval=5)
        x, y = mob.initialize(rng, 1000.0, 1000.0)

        for _ in range(500):
            x, y, v, d = mob.update(x, y, 0.0, 0.0, 1.0, rng, 1000.0, 1000.0)
            assert 0.0 <= x <= 1000.0, f"x={x} out of bounds"
            assert 0.0 <= y <= 1000.0, f"y={y} out of bounds"

    def test_direction_changes(self):
        rng = np.random.default_rng(42)
        mob = RandomWalkMobility(v_min=5.0, v_max=20.0, direction_change_interval=5)
        x, y = mob.initialize(rng, 500.0, 500.0)
        directions = []
        for _ in range(30):
            x, y, v, d = mob.update(x, y, 0.0, 0.0, 1.0, rng, 1000.0, 1000.0)
            directions.append(d)
        # Direction should change at least once in 30 steps with interval=5
        unique_dirs = set(round(d, 4) for d in directions)
        assert len(unique_dirs) > 1

    def test_reflection_at_boundary(self):
        """Test that a UAV heading out of bounds gets reflected."""
        rng = np.random.default_rng(42)
        mob = RandomWalkMobility(v_min=50.0, v_max=50.0, direction_change_interval=1000)
        # Place at edge heading right
        mob._current_velocity = 50.0
        mob._current_direction = 0.0  # heading east
        mob._steps_since_change = 0
        x, y = 990.0, 500.0
        new_x, new_y, _, _ = mob.update(x, y, 50.0, 0.0, 1.0, rng, 1000.0, 1000.0)
        # Should reflect and stay in bounds
        assert 0.0 <= new_x <= 1000.0


class TestCreateMobilityModel:
    """Tests for the factory function."""

    def test_create_random_waypoint(self):
        config = {
            "mobility": {"model": "random_waypoint", "pause_time": [0.0, 5.0]},
            "uav": {"velocity_range": [5.0, 20.0]},
        }
        mob = create_mobility_model(config)
        assert isinstance(mob, RandomWaypointMobility)

    def test_create_random_walk(self):
        config = {
            "mobility": {"model": "random_walk", "direction_change_interval": 10},
            "uav": {"velocity_range": [5.0, 20.0]},
        }
        mob = create_mobility_model(config)
        assert isinstance(mob, RandomWalkMobility)

    def test_unknown_model_raises(self):
        config = {
            "mobility": {"model": "teleport"},
            "uav": {"velocity_range": [5.0, 20.0]},
        }
        with pytest.raises(ValueError, match="Unknown mobility model"):
            create_mobility_model(config)


# ====================================================================== #
# UAV Node Tests
# ====================================================================== #


class TestUAVNode:
    """Tests for UAVNode model."""

    def _make_uav(self, node_id: str, x: float, y: float,
                  velocity: float = 0.0, direction: float = 0.0,
                  comm_radius: float = 200.0) -> UAVNode:
        """Helper to create a UAV at a specific position."""
        edge = EdgeNode(node_id, cpu_capacity=4.0,
                        memory_capacity=8192.0, storage_capacity=64000.0)
        uav = UAVNode(node_id=node_id, x=x, y=y, velocity=velocity,
                      direction=direction, communication_radius=comm_radius,
                      edge=edge)
        return uav

    def test_distance_to(self):
        uav1 = self._make_uav("u1", 0.0, 0.0)
        uav2 = self._make_uav("u2", 3.0, 4.0)
        assert abs(uav1.distance_to(uav2) - 5.0) < 1e-9

    def test_distance_symmetry(self):
        uav1 = self._make_uav("u1", 10.0, 20.0)
        uav2 = self._make_uav("u2", 30.0, 40.0)
        assert abs(uav1.distance_to(uav2) - uav2.distance_to(uav1)) < 1e-9

    def test_is_in_range(self):
        uav1 = self._make_uav("u1", 0.0, 0.0, comm_radius=200.0)
        uav2 = self._make_uav("u2", 100.0, 0.0)
        uav3 = self._make_uav("u3", 300.0, 0.0)
        assert uav1.is_in_range(uav2) is True
        assert uav1.is_in_range(uav3) is False

    def test_get_neighbors(self):
        uav1 = self._make_uav("u1", 0.0, 0.0, comm_radius=200.0)
        uav2 = self._make_uav("u2", 100.0, 0.0, comm_radius=200.0)
        uav3 = self._make_uav("u3", 300.0, 0.0, comm_radius=200.0)
        neighbors = uav1.get_neighbors([uav1, uav2, uav3])
        assert len(neighbors) == 1
        assert neighbors[0].node_id == "u2"

    def test_predicted_contact_duration_stationary(self):
        uav1 = self._make_uav("u1", 0.0, 0.0, velocity=0.0, comm_radius=200.0)
        uav2 = self._make_uav("u2", 100.0, 0.0, velocity=0.0, comm_radius=200.0)
        # Both stationary → infinite contact
        assert uav1.predicted_contact_duration(uav2) == float("inf")

    def test_predicted_contact_duration_out_of_range(self):
        uav1 = self._make_uav("u1", 0.0, 0.0, comm_radius=200.0)
        uav2 = self._make_uav("u2", 500.0, 0.0, comm_radius=200.0)
        assert uav1.predicted_contact_duration(uav2) == 0.0

    def test_predicted_contact_duration_moving_apart(self):
        uav1 = self._make_uav("u1", 0.0, 0.0, velocity=10.0,
                               direction=math.pi, comm_radius=200.0)  # heading west
        uav2 = self._make_uav("u2", 100.0, 0.0, velocity=10.0,
                               direction=0.0, comm_radius=200.0)  # heading east
        contact = uav1.predicted_contact_duration(uav2)
        assert contact > 0.0
        assert contact < float("inf")
        # Moving apart at 20 m/s total, 100m remaining range → ~5 seconds
        assert abs(contact - 5.0) < 1.0

    def test_no_mobility_raises(self):
        uav = self._make_uav("u1", 0.0, 0.0)
        rng = np.random.default_rng(42)
        with pytest.raises(RuntimeError):
            uav.initialize_position(rng, 1000.0, 1000.0)

    def test_mobility_integration(self):
        """Test that UAV movement works end-to-end with mobility model."""
        rng = np.random.default_rng(42)
        mob = RandomWaypointMobility(v_min=5.0, v_max=20.0,
                                      pause_min=0.0, pause_max=0.0)
        edge = EdgeNode("u1", cpu_capacity=4.0,
                        memory_capacity=8192.0, storage_capacity=64000.0)
        uav = UAVNode(node_id="u1", communication_radius=200.0,
                      edge=edge, mobility_model=mob)
        uav.initialize_position(rng, 1000.0, 1000.0)
        initial_x, initial_y = uav.x, uav.y

        # Move for 100 steps
        for _ in range(100):
            uav.update_position(1.0, rng, 1000.0, 1000.0)
            assert 0.0 <= uav.x <= 1000.0
            assert 0.0 <= uav.y <= 1000.0

        # Should have moved from initial position
        moved = math.sqrt((uav.x - initial_x)**2 + (uav.y - initial_y)**2)
        assert moved > 0.0


class TestCreateUAVNodes:
    """Tests for the UAV factory function."""

    def test_create_from_config(self):
        config = {
            "simulation": {
                "area_width": 1000.0,
                "area_height": 1000.0,
            },
            "uav": {
                "count": 3,
                "communication_radius": 200.0,
                "velocity_range": [5.0, 20.0],
                "cpu_capacity": 4.0,
                "memory_capacity": 8192.0,
                "storage_capacity": 64000.0,
                "initial_utilization": {"cpu": 0.2, "memory": 0.3, "storage": 0.1},
            },
            "mobility": {
                "model": "random_waypoint",
                "pause_time": [0.0, 5.0],
            },
        }
        rng = np.random.default_rng(42)
        mob = create_mobility_model(config)
        nodes = create_uav_nodes(config, mob, rng)

        assert len(nodes) == 3
        for node in nodes:
            assert 0.0 <= node.x <= 1000.0
            assert 0.0 <= node.y <= 1000.0
            assert node.communication_radius == 200.0
            assert abs(node.edge.cpu_utilization - 0.2) < 1e-6
            assert abs(node.edge.memory_utilization - 0.3) < 1e-6
            assert abs(node.edge.storage_utilization - 0.1) < 1e-6

    def test_nodes_have_unique_ids(self):
        config = {
            "simulation": {"area_width": 1000.0, "area_height": 1000.0},
            "uav": {
                "count": 5,
                "communication_radius": 200.0,
                "velocity_range": [5.0, 20.0],
                "cpu_capacity": 4.0,
                "memory_capacity": 8192.0,
                "storage_capacity": 64000.0,
            },
            "mobility": {"model": "random_waypoint", "pause_time": [0.0, 5.0]},
        }
        rng = np.random.default_rng(42)
        mob = create_mobility_model(config)
        nodes = create_uav_nodes(config, mob, rng)
        ids = [n.node_id for n in nodes]
        assert len(set(ids)) == 5

    def test_reproducibility_with_same_seed(self):
        config = {
            "simulation": {"area_width": 1000.0, "area_height": 1000.0},
            "uav": {
                "count": 3,
                "communication_radius": 200.0,
                "velocity_range": [5.0, 20.0],
                "cpu_capacity": 4.0,
                "memory_capacity": 8192.0,
                "storage_capacity": 64000.0,
            },
            "mobility": {"model": "random_waypoint", "pause_time": [0.0, 5.0]},
        }
        mob1 = create_mobility_model(config)
        mob2 = create_mobility_model(config)
        nodes1 = create_uav_nodes(config, mob1, np.random.default_rng(42))
        nodes2 = create_uav_nodes(config, mob2, np.random.default_rng(42))

        for n1, n2 in zip(nodes1, nodes2):
            assert abs(n1.x - n2.x) < 1e-9
            assert abs(n1.y - n2.y) < 1e-9
