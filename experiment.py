"""
Unified experiment runner for QAOA tasks.
Runs Task 2-7 experiments with configurable parameters.
"""

import json
import time
import os
import numpy as np
import networkx as nx

from hamiltonian import maxcut_cost, maxcut_exact, graph_c4, graph_g6, graph_g9
from circuit import qaoa_circuit, qaoa_circuit_sampling
from init_strategies import INIT_STRATEGIES
from optimizer import OPTIMIZERS


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_single_experiment(
    graph: nx.Graph,
    graph_name: str,
    n_layers: int = 1,
    init_strategy: str = "zero",
    optimizer_name: str = "adam",
    n_shots: int = None,  # None = exact expectation
    max_iter: int = 300,
    seed: int = 42,
) -> dict:
    """Run a single QAOA experiment and return results."""
    np.random.seed(seed)

    # Build energy function
    if n_shots is None:
        energy_fn = qaoa_circuit(graph=graph, n_layers=n_layers)
    else:
        def energy_fn(params):
            return qaoa_circuit_sampling(params, graph, n_layers, n_shots)

    # Initialize parameters
    initial_params = INIT_STRATEGIES[init_strategy](n_layers)
    initial_params = initial_params.astype(np.float64) if n_shots else initial_params

    # Optimize
    t0 = time.time()
    opt_params, final_energy, history = OPTIMIZERS[optimizer_name](
        energy_fn, initial_params, max_iter=max_iter
    )
    elapsed = time.time() - t0

    # Compute approximation ratio
    exact_cut, _ = maxcut_exact(graph)
    approx_ratio = final_energy / exact_cut if exact_cut > 0 else 0.0

    result = {
        "graph": graph_name,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_layers": n_layers,
        "init_strategy": init_strategy,
        "optimizer": optimizer_name,
        "n_shots": n_shots,
        "initial_params": initial_params.tolist(),
        "optimized_params": opt_params.tolist(),
        "final_energy": float(final_energy),
        "exact_maxcut": int(exact_cut),
        "approximation_ratio": float(approx_ratio),
        "n_iterations": len(history),
        "elapsed_time": elapsed,
        "history": [float(h) for h in history],
        "seed": seed,
    }
    return result


def task2_p1_baseline() -> list[dict]:
    """Task 2: p=1 baseline on C4, G6, G9."""
    results = []
    graphs = [("C4", graph_c4()), ("G6", graph_g6()), ("G9", graph_g9())]
    for name, g in graphs:
        print(f"  Running {name}...")
        r = run_single_experiment(g, name, n_layers=1)
        results.append(r)
    return results


def task3_p_scan(max_p: int = 5) -> list[dict]:
    """Task 3: scan depth p = 1..max_p on all three graphs."""
    results = []
    graphs = [("C4", graph_c4()), ("G6", graph_g6()), ("G9", graph_g9())]
    for name, g in graphs:
        for p in range(1, max_p + 1):
            print(f"  {name} p={p}...")
            r = run_single_experiment(g, name, n_layers=p)
            results.append(r)
    return results


def task4_init_comparison() -> list[dict]:
    """Task 4: compare initialization strategies."""
    results = []
    strategies = ["zero", "random", "linear_ramp"]
    graphs = [("C4", graph_c4()), ("G6", graph_g6()), ("G9", graph_g9())]
    for name, g in graphs:
        for strat in strategies:
            print(f"  {name} init={strat}...")
            for seed in range(5):  # 5 random restarts
                r = run_single_experiment(
                    g, name, n_layers=2, init_strategy=strat, seed=seed
                )
                results.append(r)
    return results


def task5_optimizer_comparison() -> list[dict]:
    """Task 5: compare classical optimizers."""
    results = []
    optimizers = ["adam", "bfgs", "cobyla", "spsa"]
    graphs = [("C4", graph_c4()), ("G6", graph_g6()), ("G9", graph_g9())]
    for name, g in graphs:
        for opt in optimizers:
            print(f"  {name} opt={opt}...")
            r = run_single_experiment(
                g, name, n_layers=2, optimizer_name=opt, max_iter=300
            )
            results.append(r)
    return results


def task6_shot_comparison() -> list[dict]:
    """Task 6: compare different shot numbers."""
    results = []
    shot_list = [100, 500, 1000, 5000]
    graphs = [("C4", graph_c4()), ("G6", graph_g6())]
    for name, g in graphs:
        for shots in shot_list:
            print(f"  {name} shots={shots}...")
            r = run_single_experiment(
                g, name, n_layers=1, n_shots=shots, optimizer_name="spsa"
            )
            results.append(r)
    return results


def save_results(results: list[dict], filename: str):
    """Save experiment results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def run_all():
    """Run all baseline experiments for Tasks 2-7."""
    print("=" * 50)
    print("Task 2: p=1 Baseline")
    print("=" * 50)
    r2 = task2_p1_baseline()
    save_results(r2, "task2_p1_baseline.json")

    print("\n" + "=" * 50)
    print("Task 3: p-Scan (p=1..5)")
    print("=" * 50)
    r3 = task3_p_scan(max_p=5)
    save_results(r3, "task3_p_scan.json")

    print("\n" + "=" * 50)
    print("Task 4: Init Strategy Comparison")
    print("=" * 50)
    r4 = task4_init_comparison()
    save_results(r4, "task4_init_comparison.json")

    print("\n" + "=" * 50)
    print("Task 5: Optimizer Comparison")
    print("=" * 50)
    r5 = task5_optimizer_comparison()
    save_results(r5, "task5_optimizer_comparison.json")

    print("\n" + "=" * 50)
    print("Task 6: Shot Number Comparison")
    print("=" * 50)
    r6 = task6_shot_comparison()
    save_results(r6, "task6_shot_comparison.json")


if __name__ == "__main__":
    run_all()
