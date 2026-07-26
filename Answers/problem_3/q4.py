"""Problem 3, Question 4: QAOA parameter initialization strategies."""

import numpy as np

from q1 import GRAPHS
from q2 import expectation, optimize_exact


def initial_parameters(p, strategy, seed=7):
    if strategy == "zero":
        return np.zeros(2 * p)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        gammas = rng.uniform(0.0, 2.0 * np.pi, p)
        betas = rng.uniform(0.0, np.pi, p)
        return np.r_[gammas, betas]

    if strategy == "linear-ramp":
        # Annealing-inspired: problem strength rises, mixer strength falls.
        gammas = np.linspace(0.1, 0.8, p)
        betas = np.linspace(0.8, 0.1, p)
        return np.r_[gammas, betas]

    raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    P = 3
    STRATEGIES = ["zero", "random", "linear-ramp"]

    print("graph  initialization  evaluations  final <C>")
    for name, (n, edges) in GRAPHS.items():
        for strategy in STRATEGIES:
            x0 = initial_parameters(P, strategy)
            result = optimize_exact(
                n,
                edges,
                P,
                x0=x0,
                method="COBYLA",
                maxiter=100,
            )
            print(
                f"{name:5s} {strategy:14s} "
                f"{result.nfev:11d} {expectation(n, edges, result.x):10.4f}"
            )

    print(
        "\nZero initialization is simple but highly symmetric. Random "
        "initialization explores more basins but varies with the seed. "
        "The linear ramp uses the QAOA/annealing interpretation and is "
        "usually a stronger deterministic starting point."
    )
