# Research Assumptions

This document explicitly lists the assumptions underlying the simulation model.
Reviewers and readers should be aware of these when interpreting experimental results.

---

## 1. Network Model

| Assumption | Justification |
|------------|---------------|
| Connectivity is determined by Euclidean distance between UAVs | Standard simplification in mobile ad-hoc network simulations; avoids need for full RF propagation modeling |
| Bandwidth decays with distance (inverse relationship within communication range) | Approximates free-space path loss effects on throughput |
| Latency increases linearly with distance | Simplified model; real latency depends on routing, queuing, processing |
| Packet loss increases with distance | Approximates increased error probability at range boundaries |
| Three connectivity states: connected, degraded, disconnected | Provides discrete categories for decision-making while based on continuous bandwidth ratio |
| No multi-hop routing | Each UAV communicates only with directly reachable neighbors; simplifies the network model |
| No interference or contention modeling | Bandwidth is modeled per-link, not accounting for shared medium effects |

---

## 2. UAV Mobility Model

| Assumption | Justification |
|------------|---------------|
| 2D movement only (no altitude changes) | Simplifies the model; altitude could be added as future work |
| Random waypoint or random walk models | Well-studied models in the mobile networking literature (Camp et al., 2002) |
| Constant speed between waypoints | Simplification; real UAVs accelerate and decelerate |
| UAVs remain within the simulation boundary | Wraparound or reflection at boundaries to maintain UAV count |
| All UAVs are identical in capability | Simplifies resource analysis; heterogeneous UAVs are future work |

---

## 3. Service State Model

| Assumption | Justification |
|------------|---------------|
| State is modeled as a single block of data with a size in MB | Abstracts application-level state; sufficient for studying transfer dynamics |
| State grows at a constant rate (configurable) | Represents continuous computation (e.g., sensor data aggregation, model updates) |
| State is versioned with monotonically increasing counters | Enables detection of stale replicas and lost updates |
| Only one primary instance of the service runs at a time | Avoids need for distributed consistency protocols (out of scope) |
| Service priority is static (does not change during simulation) | Simplification; dynamic priority could be added |

---

## 4. Migration Mechanism

| Assumption | Justification |
|------------|---------------|
| Migration transfers the entire state to the destination node | Models live migration (not partial transfer) |
| Migration time = state_size / (bandwidth × bandwidth_fraction) | Standard bandwidth-based transfer time estimation |
| During migration, the service experiences interruption | Represents stop-and-copy migration; live migration with minimal downtime is future work |
| Migration can fail if connectivity drops during transfer | Realistic — incomplete transfers waste resources |
| Only one migration can be in progress at a time | Simplifies the model and avoids race conditions |

---

## 5. Checkpoint Model

| Assumption | Justification |
|------------|---------------|
| Checkpoint captures the full state at a point in time | Standard checkpoint semantics |
| Checkpoint size ≈ state_size × storage_multiplier | Accounts for metadata overhead |
| Checkpointing consumes CPU resources (configurable fraction) | Realistic — serialization and writing require compute |
| A limited number of checkpoints are stored (oldest evicted) | Prevents unbounded storage growth |
| Recovery from checkpoint restores state to checkpoint version (later updates lost) | Standard checkpoint-restart semantics |

---

## 6. Replication Model

| Assumption | Justification |
|------------|---------------|
| Replicas store a copy of the service state on other nodes | Standard replication semantics |
| Synchronization requires bandwidth and time proportional to delta or full state | Models the cost of keeping replicas current |
| Replicas may become stale if synchronization is delayed | Reflects real-world eventual consistency challenges |
| Recovery from replica may involve some state loss depending on staleness | Realistic — stale replicas yield older state |

---

## 7. Resource Model

| Assumption | Justification |
|------------|---------------|
| CPU, memory, and storage are modeled as continuous values with utilization fractions | Standard resource modeling in cloud/edge computing research |
| Resource consumption from state management operations is additive | Simplification; real resource interactions may be more complex |
| No resource contention between services | Only one service is modeled per experiment |

---

## 8. Timing Model

| Assumption | Justification |
|------------|---------------|
| 1 simulation step = 1 second of real time | Provides intuitive interpretation of velocities (m/s) and bandwidths (Mbps) |
| Operations (checkpoint, replicate, migrate) may span multiple steps | Realistic — large state transfers take time |
| Actions take effect at the beginning of the step | Discrete-event approximation |

---

## 9. Failure Model

| Assumption | Justification |
|------------|---------------|
| Link failures are injected with a configurable probability per step | Allows controlled study of intermittence effects |
| Node failures are not modeled in the current prototype | Focus is on connectivity intermittence, not node crashes |
| Failed links recover automatically (based on UAV movement restoring range) | Transient failures, consistent with UAV mobility |

---

## 10. Energy Model

| Assumption | Justification |
|------------|---------------|
| Energy is approximated as CPU-time × constant factor | Without real hardware, direct measurement is not possible |
| Energy cost is **explicitly labeled as an estimate** in all results | Prevents over-claiming based on approximate values |
| Communication energy is not separately modeled | Simplification; could be added with a transmission power model |

---

## 11. Decision Timing

| Assumption | Justification |
|------------|---------------|
| The agent makes one decision per simulation step | Simplifies the control loop; sub-step decisions would add complexity without clear benefit for this study |
| Decision computation time is assumed negligible | The agent logic (rule evaluation, scoring) is lightweight compared to data transfer |
| The agent has full observability of its local node and direct neighbors | No information delay for local state; network information limited to direct links |

---

## References

- Camp, T., Boleng, J., & Davies, V. (2002). A survey of mobility models for ad hoc network research. *Wireless Communications and Mobile Computing*, 2(5), 483-502.
- Clark, C., et al. (2005). Live migration of virtual machines. *NSDI*.
- Machen, A., et al. (2018). Live service migration in mobile edge clouds. *IEEE Wireless Communications*.
