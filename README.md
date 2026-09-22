# Adaptive Agentic Service State Management under Intermittent UAV-Edge Connectivity

> **Research Prototype** for the paper:  
> *"Service State Management in Edge Devices using Agentic AI for UAV-Enabled Intermittent Edge Networks"*

## Overview

This prototype investigates whether **agentic AI-driven service state management** can reduce service disruptions, minimize state loss, and improve recovery efficiency in UAV-enabled edge computing environments with intermittent network connectivity.

The system simulates a fleet of UAV-mounted edge nodes running stateful services. As UAVs move, network links form and break. An **agentic decision layer** observes the environment, builds context from recent observations, assesses risk using multiple factors, and selects actions (checkpoint, replicate, synchronize, migrate, or continue locally) to preserve service state.

### Key Research Questions

1. **RQ1**: Can an agentic approach reduce total service interruption time compared to baseline strategies under intermittent connectivity?
2. **RQ2**: How does state size affect migration overhead and recovery efficiency across strategies?
3. **RQ3**: Under what connectivity conditions does proactive state management outweigh its costs?
4. **RQ4**: What is the trade-off between migration frequency and state loss?

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Simulation Loop                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │   UAV    │→ │ Network  │→ │ Service State    │  │
│  │ Mobility │  │  Model   │  │ Management       │  │
│  └──────────┘  └──────────┘  │ ┌──────────────┐ │  │
│                              │ │ Checkpoint   │ │  │
│  ┌──────────────────────┐   │ │ Replication  │ │  │
│  │   Agent (Decision    │   │ │ Sync         │ │  │
│  │   Layer)             │←──│ │ Migration    │ │  │
│  │                      │──→│ │ Recovery     │ │  │
│  │  Observe → Context   │   │ └──────────────┘ │  │
│  │  → Reason → Act      │   └──────────────────┘  │
│  │  → Learn             │                          │
│  └──────────────────────┘   ┌──────────────────┐  │
│                              │ Metrics &        │  │
│                              │ Analysis         │  │
│                              └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Strategies Compared

| Strategy | Type | Description |
|----------|------|-------------|
| `local_execution` | Baseline 1 | Never migrates; periodic checkpoints only |
| `threshold_based` | Baseline 2 | Migrates when bandwidth/latency/CPU exceed static thresholds |
| `mobility_aware` | Baseline 3 | Proactively migrates based on predicted contact duration |
| `rl_based` | Baseline 4 | Tabular Q-learning with discretized state space |
| `agentic_manager` | **Proposed** | Context-aware agent with risk assessment, cost-benefit analysis, outcome memory, and adaptive thresholds |

## Project Structure

```
├── config/
│   └── default.yaml              # Simulation and agent configuration
├── src/
│   ├── environment/
│   │   ├── edge_node.py           # Edge node resource model (CPU, Mem, Storage)
│   │   ├── mobility.py            # Mobility models (RandomWaypoint, RandomWalk)
│   │   ├── uav.py                 # UAV node (position, velocity, resources)
│   │   ├── network.py             # Network model (bandwidth, latency, connectivity)
│   │   └── simulator.py           # Core discrete-event simulation engine
│   ├── service/
│   │   ├── state.py               # Stateful service model with versioned state
│   │   ├── checkpoint.py          # Checkpoint creation, eviction, storage tracking
│   │   ├── recovery.py            # Recovery from checkpoints or replicas
│   │   ├── replication.py         # State replication to neighboring nodes
│   │   ├── synchronization.py     # Delta-based replica synchronization
│   │   └── migration.py           # Service migration with feasibility checks
│   ├── agents/
│   │   ├── base_agent.py          # Base interface, Action enum, Observation struct
│   │   ├── local_agent.py         # Baseline 1: Local execution
│   │   ├── threshold_agent.py     # Baseline 2: Threshold-based migration
│   │   ├── mobility_agent.py      # Baseline 3: Mobility-aware migration
│   │   ├── rl_agent.py            # Baseline 4: Q-learning agent
│   │   └── agentic_manager.py     # Proposed: Agentic state manager
│   └── evaluation/
│       ├── metrics.py             # Per-step and aggregate metrics collection
│       ├── analysis.py            # Statistical analysis (mean, std, CI)
│       └── plots.py               # Publication-quality visualization
├── tests/                         # Unit and integration tests (149+ tests)
├── results/                       # Output: CSV results and plots
├── main.py                        # CLI experiment runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/<user>/Adaptive-Agentic-Service-State-Management-under-Intermittent-UAV-Edge-Connectivity.git
cd Adaptive-Agentic-Service-State-Management-under-Intermittent-UAV-Edge-Connectivity
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Default Experiment

```bash
python main.py --config config/default.yaml --experiment default
```

### Run Specific Experiments

```bash
# Vary connectivity reliability
python main.py --experiment connectivity

# Vary service state size
python main.py --experiment state_size

# Vary UAV velocity
python main.py --experiment velocity

# Run all experiments
python main.py --experiment all

# Run specific strategies only
python main.py --strategies local_execution agentic_manager --experiment default
```

### Output

Results are saved to `results/`:
- `results/*.csv` — Raw and aggregated metric tables
- `results/plots/` — Publication-quality figures (PNG + PDF)

## Configuration

All parameters are in [`config/default.yaml`](config/default.yaml):

| Section | Key Parameters |
|---------|---------------|
| `simulation` | `duration` (steps), `num_seeds`, `area_width/height` |
| `uav` | `count`, `communication_radius`, `velocity_range`, resource capacities |
| `mobility` | `model` (random_waypoint / random_walk), `pause_time` |
| `network` | `max_bandwidth`, `latency_base`, connectivity thresholds |
| `service` | `state_size`, `state_update_rate`, `priority`, SLA constraints |
| `checkpoint` | `interval`, `cpu_overhead_fraction`, `max_checkpoints` |
| `replication` | `max_replicas`, `sync_interval`, `bandwidth_fraction` |
| `migration` | `bandwidth_fraction`, `min_bandwidth_required`, `min_contact_duration` |
| `objective` | Weighted cost function: interruption, state_transfer, state_loss, migration, SLA, resource |

## Evaluation Metrics

### Service Metrics
- **Service Availability**: Fraction of steps with service running
- **Interruption Time**: Total steps where service was interrupted
- **State Loss**: Accumulated version deltas lost during disruptions
- **SLA Violations**: Count of interruptions exceeding SLA thresholds

### Migration Metrics
- **Migration Success Rate**: Successful migrations / attempted
- **Migration Frequency**: Total migration attempts
- **Migration Time**: Time spent migrating

### Resource Metrics
- **Total Data Transferred**: Bandwidth consumed by all operations
- **Checkpoint Cost**: CPU and storage overhead
- **Energy Proxy**: Accumulated CPU utilization

### Composite Objective
```
Cost = w₁·Interruption + w₂·StateTransfer + w₃·StateLoss
     + w₄·MigrationCost + w₅·SLAViolation + w₆·ResourceCost
```

## Design Decisions

1. **Discrete-event simulation** (1 step = 1 second) for reproducibility and transparency.
2. **Distance-based network model** with bandwidth decay, latency growth, packet loss, and configurable failure injection.
3. **Stop-and-copy migration** — service is interrupted during transfer (conservative model).
4. **Tabular Q-learning** for RL baseline — intentionally simple for transparency. If it underperforms, that is a valid finding.
5. **No external LLM APIs** — the agentic manager uses rule-based reasoning with context and memory, not language model calls.
6. **All randomness is seeded** — every experiment is fully reproducible.

## Reproducibility

- All experiments use fixed random seeds (configurable via `simulation.random_seed`)
- Each experiment is run across `num_seeds` (default: 5) independent seeds
- Results include mean, standard deviation, and 95% confidence intervals
- All figures are generated deterministically from CSV results

## License

This project is part of academic research. See LICENSE for details.
