# Adaptive Agentic Service State Management under Intermittent UAV-Edge Connectivity

> **Status: In Development** — This is a research prototype under active development.

## Research Motivation

UAV-enabled edge computing extends cloud services to remote or mobile environments. However, UAV mobility introduces **intermittent connectivity** between edge devices: communication links form and break as UAVs move, bandwidth fluctuates, and contact durations are unpredictable.

Stateful services running on edge devices cannot assume that migration is always possible or optimal. A service must dynamically decide whether to continue local execution, checkpoint its state, replicate to nearby nodes, synchronize with replicas, migrate entirely, or gracefully degrade — based on current and predicted network/resource conditions.

## Problem Statement

This prototype investigates whether an **adaptive, agentic state-management approach** can improve service continuity under intermittent UAV-edge connectivity, compared to fixed migration policies (threshold-based, mobility-aware) and learning-based approaches (RL).

## Architecture

```
                     Stateful Service
                            │
                            ▼
                  ┌──────────────────┐
                  │  State Manager   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Decision Layer  │
                  │                  │
                  │  Observe         │
                  │  Context Build   │
                  │  Reason          │
                  │  Act             │
                  │  Learn           │
                  └────────┬─────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Checkpoint     Replication    Migration
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                UAV-enabled Edge Nodes
```

### Strategies Compared

| Strategy | Description |
|----------|-------------|
| **Local Execution** | Never migrate (baseline) |
| **Threshold-based** | Migrate when metrics exceed static thresholds |
| **Mobility-aware** | Use predicted UAV contact duration for decisions |
| **RL-based** | Lightweight reinforcement learning agent |
| **Agentic (proposed)** | Context-aware agent with memory and adaptive reasoning |

## Installation

```bash
# Clone the repository
git clone https://github.com/sinharea/Adaptive-Agentic-Service-State-Management-under-Intermittent-UAV-Edge-Connectivity.git
cd Adaptive-Agentic-Service-State-Management-under-Intermittent-UAV-Edge-Connectivity

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default configuration
python main.py --config config/default.yaml

# Run specific experiment
python main.py --config config/experiments/connectivity.yaml

# Run tests
pytest tests/ -v
```

> **Note**: The experiment runner and full configuration system are under development.

## Project Structure

```
├── config/              # YAML configuration files
│   ├── default.yaml
│   └── experiments/     # Per-experiment configs
├── docs/                # Research documentation
│   ├── research_design.md
│   └── assumptions.md
├── src/
│   ├── environment/     # UAV, edge nodes, network, mobility
│   ├── service/         # State, checkpoint, replication, migration
│   ├── agents/          # Decision strategies (baselines + proposed)
│   └── evaluation/      # Metrics, analysis, plotting
├── experiments/         # Experiment scripts and logs
├── tests/               # Automated test suite
└── results/             # Experimental results (CSVs, plots)
```

## Current Limitations

- This is a simulation-based prototype, not a deployment on real UAV hardware.
- The network model uses distance-based connectivity, not a full wireless channel model.
- Energy consumption is approximated, not measured.
- The RL baseline may not converge fully within limited training episodes.

See [docs/assumptions.md](docs/assumptions.md) for a complete list of modeling assumptions.

## License

This project is developed for academic research purposes.

## Citation

If you use this prototype in your research, please cite the associated paper (citation details to be added upon publication).
