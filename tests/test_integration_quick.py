"""Quick integration test for the simulation pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.simulator import Simulator, load_config
from src.evaluation.metrics import compute_objective_cost

config = load_config("config/default.yaml")
config["simulation"]["duration"] = 100
sim = Simulator(config)
weights = config.get("objective", {})

strategies = ["local_execution", "threshold_based", "mobility_aware", "agentic_manager"]

for strategy in strategies:
    m = sim.run(strategy, seed=42)
    s = m.compute_summary()
    cost = compute_objective_cost(s, weights)
    print(f"{strategy:20s} | avail={s['service_availability']:.3f} | "
          f"interruptions={s['total_interruption_time']:3.0f} | "
          f"migrations={s['migrations_attempted']:2.0f} | "
          f"state_loss={s['total_state_loss']:3.0f} | "
          f"cost={cost:.2f}")

print("\nIntegration test PASSED")
