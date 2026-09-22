"""
Simulation engine.

Integrates all components into a single simulation loop:
UAV mobility → network → service → agent → action execution → metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import numpy as np
import yaml

from src.environment.uav import UAVNode, create_uav_nodes
from src.environment.mobility import create_mobility_model
from src.environment.network import NetworkModel, ConnectivityState
from src.service.state import ServiceState, create_service
from src.service.checkpoint import CheckpointManager
from src.service.replication import ReplicationManager
from src.service.synchronization import SynchronizationManager
from src.service.migration import MigrationManager
from src.service.recovery import RecoveryManager
from src.agents.base_agent import (
    BaseAgent, Action, Observation, Decision, build_observation,
)
from src.agents.local_agent import LocalAgent
from src.agents.threshold_agent import ThresholdAgent
from src.agents.mobility_agent import MobilityAgent
from src.agents.rl_agent import RLAgent
from src.agents.agentic_manager import AgenticManager
from src.evaluation.metrics import StepMetrics, SimulationMetrics, compute_objective_cost

logger = logging.getLogger(__name__)


AGENT_REGISTRY: Dict[str, type] = {
    "local_execution": LocalAgent,
    "threshold_based": ThresholdAgent,
    "mobility_aware": MobilityAgent,
    "rl_based": RLAgent,
    "agentic_manager": AgenticManager,
}


def create_agent(strategy: str, config: dict) -> BaseAgent:
    """Create an agent by strategy name."""
    cls = AGENT_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: {list(AGENT_REGISTRY.keys())}")
    return cls(config)


class Simulator:
    """Discrete-event simulation of UAV-edge state management.

    Runs one simulation episode: initializes environment, then loops
    through time steps executing the agent's decisions and collecting
    metrics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.sim_cfg = config["simulation"]
        self.duration = self.sim_cfg["duration"]

    def run(self, strategy: str, seed: int,
            config_overrides: Optional[dict] = None) -> SimulationMetrics:
        """Run a single simulation episode.

        Args:
            strategy: Name of the agent strategy to use.
            seed: Random seed for reproducibility.
            config_overrides: Optional parameter overrides.

        Returns:
            SimulationMetrics with per-step data and summary.
        """
        config = dict(self.config)
        if config_overrides:
            for key, value in config_overrides.items():
                if isinstance(value, dict) and key in config:
                    config[key] = {**config[key], **value}
                else:
                    config[key] = value

        rng = np.random.default_rng(seed)

        # --- Initialize components ---
        mobility = create_mobility_model(config)
        uavs = create_uav_nodes(config, mobility, rng)
        uav_map = {u.node_id: u for u in uavs}
        network = NetworkModel(config)

        # Set failure probability from config
        exp_cfg = config.get("experiments", {})
        failure_prob = config.get("_failure_probability", 0.0)
        network.set_failure_probability(failure_prob)

        # Service runs on first UAV
        service = create_service(config, uavs[0].node_id)

        # State management components
        ckpt_mgr = CheckpointManager(config)
        rep_mgr = ReplicationManager(config)
        sync_mgr = SynchronizationManager(config)
        mig_mgr = MigrationManager(config)
        rec_mgr = RecoveryManager(config)

        # Agent
        agent = create_agent(strategy, config)
        obj_weights = config.get("objective", {})

        # Metrics
        sim_metrics = SimulationMetrics(strategy=strategy, seed=seed)
        # --- Tracking state for disruption/recovery model ---
        recovery_countdown = 0  # steps remaining until service recovers
        last_disruption_step = -100

        # --- Simulation loop ---
        for step in range(1, self.duration + 1):
            step_metrics = StepMetrics(step=step)

            # 1. Update UAV positions
            for uav in uavs:
                uav.update_position(1.0, rng,
                                   self.sim_cfg["area_width"],
                                   self.sim_cfg["area_height"])

            # 2. Inject network failures
            network.inject_failures(uavs, rng)

            # 2.5 Auto-recovery: if service is down and countdown expires
            if not service.is_running and recovery_countdown > 0:
                recovery_countdown -= 1
                if recovery_countdown == 0:
                    # Auto-recover: end the interruption
                    duration = service.end_interruption(step)
                    step_metrics.recovery_time = duration

            # 2.6 Connectivity-driven service disruption
            # Models: UAV services depend on connectivity to ground station
            # or peer nodes. Isolation + network failure → service disruption.
            source_uav = uav_map[service.current_node]
            if service.is_running and not service.is_migrating:
                neighbors = source_uav.get_neighbors(uavs)
                connected_neighbors = 0
                for neighbor in neighbors:
                    link = network.evaluate_link(source_uav, neighbor)
                    if link.state != ConnectivityState.DISCONNECTED:
                        connected_neighbors += 1

                # Disruption probability: isolated nodes are much more
                # vulnerable; connected nodes still face some risk under
                # heavy failure conditions.
                if connected_neighbors == 0:
                    disruption_prob = min(failure_prob * 1.5, 0.6)
                elif connected_neighbors == 1:
                    disruption_prob = failure_prob * 0.4
                else:
                    disruption_prob = failure_prob * 0.1

                # Cooldown: don't disrupt again for 5 steps after recovery
                if (step - last_disruption_step) < 5:
                    disruption_prob = 0.0

                if disruption_prob > 0 and rng.random() < disruption_prob:
                    service.begin_interruption(step)
                    last_disruption_step = step

                    # State loss depends on checkpoint/replica freshness
                    if service.has_checkpoint:
                        step_metrics.state_loss = min(
                            service.checkpoint_staleness, 50)
                    elif service.num_replicas > 0:
                        step_metrics.state_loss = min(
                            service.replica_staleness, 50)
                    else:
                        # No protection → lose all accumulated state
                        step_metrics.state_loss = min(
                            service.state_version, 50)

                    service.total_state_loss += step_metrics.state_loss
                    step_metrics.interruption = True

                    # Recovery time depends on available recovery sources:
                    # - Fresh checkpoint → fast recovery (2-5 steps)
                    # - Stale checkpoint → moderate (5-15 steps)
                    # - Replica only → moderate (3-8 steps)
                    # - Cold restart (nothing) → slow (10-30 steps)
                    if service.has_checkpoint and service.checkpoint_staleness < 20:
                        recovery_countdown = int(2 + service.state_size / 200)
                    elif service.has_checkpoint:
                        recovery_countdown = int(
                            5 + service.checkpoint_staleness / 10)
                    elif service.num_replicas > 0:
                        recovery_countdown = int(3 + service.state_size / 150)
                    else:
                        recovery_countdown = int(
                            10 + service.state_size / 50)

                    # Cap recovery time
                    recovery_countdown = min(recovery_countdown, 30)

            # 3. Update service state
            service.update_state(step)

            # 4. Build observation
            source_uav = uav_map[service.current_node]
            obs = build_observation(service, source_uav, uavs, network, step)

            # 5. Agent decides
            decision = agent.decide(obs, step)
            step_metrics.action = decision.action.value

            # 6. Execute action
            reward = 0.0
            self._execute_action(
                decision, service, source_uav, uavs, uav_map,
                network, ckpt_mgr, rep_mgr, sync_mgr, mig_mgr, rec_mgr,
                step, step_metrics,
            )

            # 6.5 If agent recovered the service, cancel auto-recovery
            if service.is_running:
                recovery_countdown = 0

            # 7. Collect metrics
            step_metrics.service_running = service.is_running
            if not service.is_running:
                step_metrics.interruption = True
            # SLA check for ongoing interruption
            if (service.interruption_start is not None and
                    (step - service.interruption_start) > service.sla_max_interruption):
                step_metrics.sla_violation = True

            step_metrics.bandwidth = obs.bandwidth
            step_metrics.latency = obs.latency
            step_metrics.packet_loss = obs.packet_loss
            step_metrics.connectivity_state = obs.connectivity_state
            step_metrics.cpu_utilization = source_uav.edge.cpu_utilization
            step_metrics.memory_utilization = source_uav.edge.memory_utilization
            step_metrics.storage_utilization = source_uav.edge.storage_utilization
            step_metrics.state_size = service.state_size
            step_metrics.state_version = service.state_version
            step_metrics.num_replicas = service.num_replicas
            step_metrics.checkpoint_staleness = service.checkpoint_staleness

            sim_metrics.steps.append(step_metrics)

            # 8. Compute reward and record for learning agents
            if service.is_running:
                reward = 1.0
            else:
                reward = -1.0
            if step_metrics.state_loss > 0:
                reward -= step_metrics.state_loss * 0.1
            if step_metrics.sla_violation:
                reward -= 1.0

            next_obs = build_observation(service, uav_map[service.current_node],
                                         uavs, network, step)
            agent.record_outcome(obs, decision.action, reward, next_obs)

        return sim_metrics

    def _execute_action(self, decision: Decision,
                        service: ServiceState,
                        source_uav: UAVNode,
                        uavs: List[UAVNode],
                        uav_map: Dict[str, UAVNode],
                        network: NetworkModel,
                        ckpt_mgr: CheckpointManager,
                        rep_mgr: ReplicationManager,
                        sync_mgr: SynchronizationManager,
                        mig_mgr: MigrationManager,
                        rec_mgr: RecoveryManager,
                        step: int,
                        metrics: StepMetrics) -> None:
        """Execute the agent's decision and update metrics."""

        action = decision.action

        if action == Action.LOCAL_EXECUTION:
            metrics.action_success = True

        elif action == Action.CHECKPOINT:
            result = ckpt_mgr.create_checkpoint(service, source_uav.edge, step)
            metrics.checkpoint_created = result.success
            metrics.checkpoint_cost_cpu = result.cpu_cost
            metrics.checkpoint_cost_storage = result.storage_cost
            metrics.action_success = result.success

        elif action == Action.REPLICATE:
            target_id = decision.target_node
            if target_id and target_id in uav_map:
                target = uav_map[target_id]
                result = rep_mgr.replicate(service, source_uav, target, network)
                metrics.replication_performed = result.success
                metrics.replication_cost_bw = result.bandwidth_cost
                metrics.action_success = result.success
            else:
                metrics.action_success = False

        elif action == Action.SYNCHRONIZE:
            results = sync_mgr.synchronize_all(service, source_uav,
                                                uav_map, network, step)
            metrics.sync_performed = any(r.success for r in results)
            metrics.sync_data = sum(r.data_transferred for r in results)
            metrics.action_success = metrics.sync_performed

        elif action == Action.MIGRATE:
            target_id = decision.target_node
            if target_id and target_id in uav_map:
                target = uav_map[target_id]
                result = mig_mgr.migrate(service, source_uav, target, network, step)
                metrics.migration_attempted = True
                metrics.migration_success = result.success
                metrics.migration_time = result.migration_time
                metrics.data_transferred = result.data_transferred
                metrics.action_success = result.success
                if not result.success:
                    metrics.interruption = True
            else:
                metrics.migration_attempted = True
                metrics.migration_success = False
                metrics.action_success = False

        elif action == Action.DEGRADE:
            # Reduce service update rate temporarily
            service.state_update_rate *= 0.5
            metrics.action_success = True

        elif action == Action.RECOVER:
            result = rec_mgr.attempt_recovery(service, ckpt_mgr, step)
            metrics.recovery_time = result.recovery_time
            metrics.state_loss = result.state_loss
            metrics.action_success = result.success
            if result.recovery_time > service.sla_max_recovery_time:
                metrics.sla_violation = True


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
