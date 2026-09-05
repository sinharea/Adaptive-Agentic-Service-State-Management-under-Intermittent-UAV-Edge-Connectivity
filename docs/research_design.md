# Research Design

## Service State Management in Edge Devices using Agentic AI for UAV-Enabled Intermittent Edge Networks

---

## 1. Research Problem

UAV-enabled edge computing extends computational resources to areas beyond fixed infrastructure.
However, UAV mobility causes **intermittent connectivity** between edge nodes: communication
links form and break as UAVs move, transmission ranges change, bandwidth fluctuates, and
contact durations are unpredictable.

Stateful services running on edge devices depend on consistent connectivity for state
management operations such as checkpointing, replication, synchronization, and migration.
When connectivity is intermittent, these operations may fail, resulting in service
interruption, state loss, or wasted resources.

Current approaches typically rely on **static policies** (e.g., threshold-based migration
triggers) that do not adapt to changing conditions, or require **continuous connectivity**
that is not guaranteed in UAV-edge environments.

---

## 2. Research Question

**RQ:** Can an adaptive, context-aware agentic approach to service state management
reduce service interruption and state loss under intermittent UAV-edge connectivity,
compared to fixed migration policies and learning-based baselines?

### Sub-questions

- **RQ1:** How does connectivity intermittence affect service continuity under different
  state management strategies?
- **RQ2:** Under what conditions does proactive state management (checkpointing, replication)
  outperform reactive migration?
- **RQ3:** Does incorporating UAV mobility prediction and historical outcomes into
  decision-making improve state management quality?
- **RQ4:** What are the trade-offs between service continuity and state-transfer overhead
  across different strategies?

---

## 3. Research Hypotheses

### H1: Adaptive vs. Fixed Policies
Adaptive state management reduces service interruption under intermittent connectivity
compared with fixed migration policies.

### H2: Proactive Management and State Size
Checkpointing and replication become more beneficial as state size increases or connectivity
becomes less predictable.

### H3: Mobility Awareness
Incorporating UAV mobility and contact-duration awareness reduces unnecessary migrations.

### H4: Agentic Trade-offs
An adaptive agent can achieve better trade-offs between service continuity and
state-transfer overhead than simple threshold policies.

**Note:** These are hypotheses, not conclusions. The experiments will determine whether
they hold.

---

## 4. Assumptions

See [assumptions.md](assumptions.md) for a detailed list. Key assumptions include:

- 2D simulation area with discrete time steps (1 step = 1 second)
- UAV movement follows random waypoint or random walk models
- Connectivity is determined by Euclidean distance and communication radius
- Bandwidth degrades with distance following a decay function
- State sizes and update rates are configurable but synthetic
- Migration, checkpointing, and replication costs are modeled, not measured on real hardware
- Energy consumption is approximated (CPU-time based proxy)

---

## 5. System Model

### 5.1 Environment

- **Simulation area:** 1000m × 1000m (configurable)
- **UAV nodes:** Mobile edge devices with CPU, memory, storage resources
- **Mobility:** Random waypoint / random walk models
- **Time:** Discrete steps, 1 step = 1 second

### 5.2 Network

- **Connectivity:** Distance-based, with connected / degraded / disconnected states
- **Bandwidth:** Decreases with distance, subject to variability
- **Latency:** Base latency + distance-proportional component
- **Packet loss:** Base rate + distance-proportional component
- **Contact duration:** Predicted based on relative velocity and communication range

### 5.3 Edge Resources

Each UAV node has:
- CPU capacity (GHz) with current utilization
- Memory capacity (MB) with current utilization
- Storage capacity (MB) with current utilization

### 5.4 Stateful Service

- Continuously updates its state at a configurable rate
- Has a state size, version counter, and priority level
- Runs on a single node at a time (primary location)
- May have checkpoints and replicas on other nodes

---

## 6. Simulation Model

### 6.1 Simulation Loop

```
For each time step t = 1, 2, ..., T:
    1. Update UAV positions (mobility model)
    2. Update network connectivity (distance-based)
    3. Update service state (increment version, grow state)
    4. Agent observes environment
    5. Agent selects action
    6. Execute action (checkpoint / replicate / migrate / etc.)
    7. Record metrics
    8. Inject failures (if configured)
```

### 6.2 Stochastic Elements

- UAV direction and speed (within configured ranges)
- Network variability (noise on bandwidth/latency)
- Link/node failures (configurable probability)
- Initial UAV positions

### 6.3 Reproducibility

- Controlled via random seeds
- Multiple seeds per experiment for statistical validity
- Configuration files capture all parameters

---

## 7. State Representation (for Decision Layer)

The agent observes a state vector containing:

| Category | Features |
|----------|----------|
| **Network** | bandwidth, latency, packet_loss, connectivity_state, predicted_contact_duration |
| **UAV** | position (x, y), velocity, direction, num_neighbors |
| **Edge** | cpu_utilization, memory_utilization, storage_utilization |
| **Service** | state_size, state_version, update_rate, priority, has_checkpoint, num_replicas, time_since_last_sync |

---

## 8. Action Space

The agent selects from:

| Action | Description |
|--------|-------------|
| `LOCAL_EXECUTION` | Continue running on current node, no state management action |
| `CHECKPOINT` | Save state to local persistent storage |
| `REPLICATE` | Copy state to a connected neighboring node |
| `SYNCHRONIZE` | Update existing replicas with latest state |
| `MIGRATE` | Transfer service entirely to a different node |
| `DEGRADE` | Reduce service quality to conserve resources |
| `RECOVER` | Restore service from checkpoint or replica |

---

## 9. Baseline Methods

### 9.1 Local Execution (Baseline 1)
Never migrates. Serves as a lower bound for migration strategies. Demonstrates the
cost of inaction when connectivity degrades.

### 9.2 Threshold-based Migration (Baseline 2)
Migrates when any metric exceeds a threshold:
- bandwidth < threshold, OR
- latency > threshold, OR
- CPU utilization > threshold

Thresholds are configurable. This represents a common reactive approach.

### 9.3 Mobility-aware Migration (Baseline 3)
Uses predicted UAV contact duration. Migrates proactively when predicted contact
duration falls below a threshold, provided sufficient bandwidth exists for state
transfer. This represents a more sophisticated reactive approach.

### 9.4 RL-based (Baseline 4)
Lightweight reinforcement learning agent (DQN or Q-learning) trained on the
simulation. Uses the same state representation and action space as the agentic
manager. This tests whether data-driven learning outperforms engineered policies.

**Note:** The RL agent may not converge fully within limited training. If so,
this will be reported honestly as a finding rather than discarded.

### 9.5 Proposed Agentic Manager
Context-aware agent with:
- Multi-source observation (network, UAV, edge, service)
- Context building from recent history
- Explicit reasoning over risk, cost, and benefit
- Action selection based on composite scoring
- Outcome memory for learning from past decisions
- Replaceable decision engine (rule-based, RL, LLM, hybrid)

---

## 10. Evaluation Metrics

### 10.1 Service Metrics
- **Service availability:** Fraction of time the service is operational
- **Interruption time:** Total time service is unavailable
- **Recovery time:** Time to restore service after disruption
- **SLA violations:** Count of events exceeding max interruption or recovery thresholds
- **State loss:** Amount of state data lost (measured in versions or MB)

### 10.2 Migration Metrics
- **Migration time:** Duration of each migration
- **Migration success rate:** Fraction of attempted migrations that complete
- **Migration frequency:** Number of migrations per simulation
- **Unnecessary migrations:** Migrations that were not beneficial in hindsight

### 10.3 State Management Metrics
- **Checkpoint overhead:** CPU and storage cost of checkpointing
- **Replication overhead:** Bandwidth and storage cost of replication
- **Synchronization lag:** Time since last successful replica sync
- **State-transfer volume:** Total data transferred for state management

### 10.4 Network Metrics
- **Bandwidth consumption:** Total bandwidth used for state management
- **Latency:** Average and peak latency during operations
- **Packet loss:** Observed packet loss rates

### 10.5 Resource Metrics
- **CPU utilization:** Including state management overhead
- **Memory utilization:** Including state management overhead
- **Storage utilization:** Including checkpoints and replicas
- **Energy proxy:** CPU-time based approximation (clearly labeled as estimate)

### 10.6 Composite Cost
```
Total Cost = w1 × interruption + w2 × state_transfer + w3 × state_loss
           + w4 × migration_cost + w5 × SLA_violation + w6 × resource_cost
```
All weights configurable via YAML.

---

## 11. Experimental Variables

| Variable | Values |
|----------|--------|
| Connectivity reliability | stable (5%), moderate (20%), high intermittence (50%) |
| State size | 50, 100, 250, 500, 1000 MB |
| UAV velocity | 5, 10, 15, 20, 30 m/s |
| Bandwidth | 10, 25, 50, 100, 200 Mbps |
| Number of UAVs | 2, 3, 5, 10, 20 |
| Service priority | low, medium, high |
| Failure probability | 0.0, 0.1, 0.2, 0.3, 0.5 |

Each experiment configuration varies one or two variables while holding others at
default values, to isolate effects.

---

## 12. Required Visualizations

1. Service interruption vs. connectivity reliability
2. State loss vs. connectivity reliability
3. Migration overhead vs. state size
4. Recovery time vs. state size
5. Service availability vs. UAV velocity
6. Bandwidth overhead by strategy
7. Number of migrations by strategy
8. SLA violations by strategy
9. Overall cost by strategy
10. Performance comparison across all methods

All plots saved in both PNG and PDF format with consistent labels, units, legends,
and figure numbering.

---

## 13. Limitations

- **Simulation fidelity:** The 2D discrete-time simulation does not capture all aspects
  of real UAV communication (e.g., multipath fading, antenna patterns, 3D terrain).
- **State abstraction:** Service state is modeled as a size value, not actual application
  data. Transfer times are computed from size and bandwidth.
- **Single service:** The current model focuses on a single stateful service. Multi-service
  contention is not modeled.
- **Homogeneous UAVs:** All UAVs have identical resource capacities in the default configuration.
- **No energy model:** Energy consumption is approximated as a CPU-time proxy. Real energy
  measurements would require hardware experiments.
- **Simplified mobility:** Random waypoint and random walk models are standard in the
  literature but may not capture all real UAV flight patterns (e.g., mission-specific
  trajectories, formation flight).

---

## 14. Scope

This prototype investigates whether adaptive state management provides measurable benefit
under controlled simulation conditions. It does not claim to solve the general problem of
edge service management, nor does it claim to be the first system addressing UAV-edge
state management.

The contribution is an **empirical comparison** of multiple state management strategies
under systematically varied intermittent connectivity conditions, with a focus on whether
context-awareness and outcome memory improve decision quality.
