"""Problem 3, Question 5: compare classical optimizers."""

from time import perf_counter

import numpy as np

from q1 import GRAPHS
from q2 import expectation, optimize_exact


def grid_search_p1(n, edges):
    """Small exact grid search for p=1."""
    best_x = None
    best_value = -1.0
    evaluations = 0

    for gamma in np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False):
        for beta in np.linspace(0.0, np.pi, 9, endpoint=False):
            x = np.array([gamma, beta])
            value = expectation(n, edges, x)
            evaluations += 1
            if value > best_value:
                best_x, best_value = x, value

    return best_x, best_value, evaluations


if __name__ == "__main__":
    P = 1
    X0 = np.array([0.5, 0.3])
    METHODS = ["BFGS", "L-BFGS-B", "COBYLA"]

    print("graph  optimizer  runtime(s)  evaluations  final <C>")
    for name, (n, edges) in GRAPHS.items():
        start = perf_counter()
        _, value, evaluations = grid_search_p1(n, edges)
        print(
            f"{name:5s} {'grid':9s} {perf_counter()-start:10.3f} "
            f"{evaluations:12d} {value:10.4f}"
        )

        for method in METHODS:
            start = perf_counter()
            result = optimize_exact(
                n,
                edges,
                P,
                x0=X0,
                method=method,
                maxiter=100,
            )
            runtime = perf_counter() - start
            evaluations = result.nfev
            if method in {"BFGS", "L-BFGS-B"}:
                evaluations += 2 * len(result.x) * result.njev
            print(
                f"{name:5s} {method:9s} {runtime:10.3f} "
                f"{evaluations:12d} "
                f"{expectation(n, edges, result.x):10.4f}"
            )

    print(
        "\nGrid search is robust for p=1 but scales exponentially with "
        "the number of parameters. BFGS and L-BFGS-B are efficient with "
        "smooth exact expectations; COBYLA is gradient-free and is often "
        "more tolerant of finite-shot noise."
    )
