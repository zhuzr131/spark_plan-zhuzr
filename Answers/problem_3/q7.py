"""Problem 3, Question 7: combined, improved QAOA-MaxCut."""

from time import perf_counter

import numpy as np

from q1 import GRAPHS, exact_solution
from q2 import expectation, optimize_exact, sample_qaoa
from q4 import initial_parameters


def best_multistart_result(n, edges, p):
    """Combine ramp initialization, random restarts, and COBYLA."""
    starts = [initial_parameters(p, "linear-ramp")]
    starts += [
        initial_parameters(p, "random", seed)
        for seed in (3, 11)
    ]

    results = [
        optimize_exact(
            n,
            edges,
            p,
            x0=x0,
            method="COBYLA",
            maxiter=80,
        )
        for x0 in starts
    ]
    return min(results, key=lambda result: result.fun)


if __name__ == "__main__":
    P = 3
    SHOTS = 2000

    print("graph  runtime(s)  iterations  evaluations  <C>    best  ratio")
    for name, (n, edges) in GRAPHS.items():
        start = perf_counter()
        result = best_multistart_result(n, edges, P)
        runtime = perf_counter() - start
        evaluations = result.nfev

        exact_average = expectation(n, edges, result.x)
        _, _, best_z, best_cut = sample_qaoa(
            n, edges, result.x, shots=SHOTS
        )
        optimum = exact_solution(n, edges)[1]

        print(
            f"{name:5s} {runtime:10.3f} {result.nit:11d} "
            f"{evaluations:12d} "
            f"{exact_average:6.3f} {best_cut:3d} "
            f"{best_cut/optimum:6.3f}  z={best_z}"
        )

    print(
        "\nThis version combines p=3, annealing-inspired initialization, "
        "two random restarts, noise-tolerant COBYLA optimization, and a "
        "larger final sample. The trade-off is greater classical and "
        "quantum runtime than the p=1 warm-up."
    )
