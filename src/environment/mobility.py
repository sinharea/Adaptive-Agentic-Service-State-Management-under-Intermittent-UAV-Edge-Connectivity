"""
Mobility models for UAV simulation.

Supports:
  - RandomWaypointMobility: UAV picks random destination, flies to it, pauses, repeat.
  - RandomWalkMobility: UAV picks random direction/speed, moves for an interval, repeat.

All models operate in a 2D bounded area [0, width] × [0, height].
Additional models can be added by subclassing MobilityModel.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


class MobilityModel(ABC):
    """Abstract base class for UAV mobility models."""

    @abstractmethod
    def initialize(self, rng: np.random.Generator,
                   area_width: float, area_height: float) -> Tuple[float, float]:
        """Return initial (x, y) position."""
        ...

    @abstractmethod
    def update(self, x: float, y: float, velocity: float, direction: float,
               dt: float, rng: np.random.Generator,
               area_width: float, area_height: float) -> Tuple[float, float, float, float]:
        """Advance one time step.

        Args:
            x, y: Current position.
            velocity: Current speed (m/s).
            direction: Current heading (radians).
            dt: Time step duration (seconds).
            rng: Numpy random generator.
            area_width, area_height: Simulation area bounds.

        Returns:
            (new_x, new_y, new_velocity, new_direction)
        """
        ...


class RandomWaypointMobility(MobilityModel):
    """Random Waypoint Mobility Model.

    The UAV selects a random destination within the area, flies toward it
    at a random speed within [v_min, v_max], pauses for a random duration
    within [pause_min, pause_max], then selects a new destination.

    Reference:
        Johnson, D. B., & Maltz, D. A. (1996). Dynamic source routing
        in ad hoc wireless networks. In Mobile Computing, pp. 153-181.
    """

    def __init__(self, v_min: float, v_max: float,
                 pause_min: float = 0.0, pause_max: float = 5.0):
        self.v_min = v_min
        self.v_max = v_max
        self.pause_min = pause_min
        self.pause_max = pause_max

        # Per-UAV internal state (set during initialize / update)
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._pause_remaining: float = 0.0
        self._current_velocity: float = 0.0

    def initialize(self, rng: np.random.Generator,
                   area_width: float, area_height: float) -> Tuple[float, float]:
        x = rng.uniform(0, area_width)
        y = rng.uniform(0, area_height)
        self._pick_new_target(rng, area_width, area_height)
        self._pause_remaining = 0.0
        return x, y

    def update(self, x: float, y: float, velocity: float, direction: float,
               dt: float, rng: np.random.Generator,
               area_width: float, area_height: float) -> Tuple[float, float, float, float]:
        # If pausing, decrement pause timer
        if self._pause_remaining > 0:
            self._pause_remaining -= dt
            return x, y, 0.0, direction

        # Compute distance to target
        dx = self._target_x - x
        dy = self._target_y - y
        dist_to_target = math.sqrt(dx * dx + dy * dy)

        # If close enough to target, start pause and pick new target
        step_dist = self._current_velocity * dt
        if dist_to_target <= step_dist + 1e-6:
            new_x = self._target_x
            new_y = self._target_y
            self._pause_remaining = rng.uniform(self.pause_min, self.pause_max)
            self._pick_new_target(rng, area_width, area_height)
            return new_x, new_y, 0.0, direction

        # Move toward target
        new_direction = math.atan2(dy, dx)
        new_x = x + self._current_velocity * dt * math.cos(new_direction)
        new_y = y + self._current_velocity * dt * math.sin(new_direction)

        # Clamp to area bounds
        new_x = max(0.0, min(new_x, area_width))
        new_y = max(0.0, min(new_y, area_height))

        return new_x, new_y, self._current_velocity, new_direction

    def _pick_new_target(self, rng: np.random.Generator,
                         area_width: float, area_height: float) -> None:
        self._target_x = rng.uniform(0, area_width)
        self._target_y = rng.uniform(0, area_height)
        self._current_velocity = rng.uniform(self.v_min, self.v_max)

    def clone(self) -> "RandomWaypointMobility":
        """Create an independent copy (useful for per-UAV instances)."""
        m = RandomWaypointMobility(self.v_min, self.v_max,
                                   self.pause_min, self.pause_max)
        return m


class RandomWalkMobility(MobilityModel):
    """Random Walk (Random Direction) Mobility Model.

    The UAV picks a random direction and speed, moves for a fixed interval,
    then picks a new direction and speed. At boundaries, the UAV reflects.

    Reference:
        Camp, T., Boleng, J., & Davies, V. (2002). A survey of mobility
        models for ad hoc network research. WCMC, 2(5), 483-502.
    """

    def __init__(self, v_min: float, v_max: float,
                 direction_change_interval: int = 10):
        self.v_min = v_min
        self.v_max = v_max
        self.direction_change_interval = direction_change_interval
        self._steps_since_change: int = 0
        self._current_velocity: float = 0.0
        self._current_direction: float = 0.0

    def initialize(self, rng: np.random.Generator,
                   area_width: float, area_height: float) -> Tuple[float, float]:
        x = rng.uniform(0, area_width)
        y = rng.uniform(0, area_height)
        self._current_velocity = rng.uniform(self.v_min, self.v_max)
        self._current_direction = rng.uniform(0, 2 * math.pi)
        self._steps_since_change = 0
        return x, y

    def update(self, x: float, y: float, velocity: float, direction: float,
               dt: float, rng: np.random.Generator,
               area_width: float, area_height: float) -> Tuple[float, float, float, float]:
        self._steps_since_change += 1

        # Pick new direction/speed after interval
        if self._steps_since_change >= self.direction_change_interval:
            self._current_velocity = rng.uniform(self.v_min, self.v_max)
            self._current_direction = rng.uniform(0, 2 * math.pi)
            self._steps_since_change = 0

        # Move
        new_x = x + self._current_velocity * dt * math.cos(self._current_direction)
        new_y = y + self._current_velocity * dt * math.sin(self._current_direction)

        # Reflect at boundaries
        if new_x < 0:
            new_x = -new_x
            self._current_direction = math.pi - self._current_direction
        elif new_x > area_width:
            new_x = 2 * area_width - new_x
            self._current_direction = math.pi - self._current_direction

        if new_y < 0:
            new_y = -new_y
            self._current_direction = -self._current_direction
        elif new_y > area_height:
            new_y = 2 * area_height - new_y
            self._current_direction = -self._current_direction

        # Clamp (safety)
        new_x = max(0.0, min(new_x, area_width))
        new_y = max(0.0, min(new_y, area_height))

        return new_x, new_y, self._current_velocity, self._current_direction

    def clone(self) -> "RandomWalkMobility":
        """Create an independent copy."""
        m = RandomWalkMobility(self.v_min, self.v_max,
                               self.direction_change_interval)
        return m


def create_mobility_model(config: dict) -> MobilityModel:
    """Factory function to create a mobility model from configuration.

    Args:
        config: Full configuration dict (must contain 'mobility' and 'uav' sections).

    Returns:
        A MobilityModel instance (to be cloned per UAV).
    """
    mobility_cfg = config["mobility"]
    uav_cfg = config["uav"]
    model_name = mobility_cfg["model"]
    v_min, v_max = uav_cfg["velocity_range"]

    if model_name == "random_waypoint":
        pause_min, pause_max = mobility_cfg.get("pause_time", [0.0, 5.0])
        return RandomWaypointMobility(v_min, v_max, pause_min, pause_max)
    elif model_name == "random_walk":
        interval = mobility_cfg.get("direction_change_interval", 10)
        return RandomWalkMobility(v_min, v_max, interval)
    else:
        raise ValueError(f"Unknown mobility model: {model_name}")
