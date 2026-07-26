"""Problem 3, Question 3: effect of QAOA depth p."""

from time import perf_counter

from q1 import GRAPHS, exact_solution
from q2 import expectation, optimize_exact, sample_qaoa


if __name__ == "__main__":
    SHOTS = 100
    DEPTHS = [1, 2, 3]

    print("graph  p  runtime(s)  iterations  evaluations  <C>     best/optimal")
    for name, (n, edges) in GRAPHS.items():
        optimum = exact_solution(n, edges)[1]

        for p in DEPTHS:
            start = perf_counter()
            result = optimize_exact(n, edges, p, maxiter=80)
            runtime = perf_counter() - start
            exact_average = expectation(n, edges, result.x)
            _, _, _, best_cut = sample_qaoa(
                n, edges, result.x, shots=SHOTS
            )

            print(
                f"{name:5s} {p:2d} {runtime:10.3f} "
                f"{result.nit:11d} {result.nfev:12d} "
                f"{exact_average:7.3f} "
                f"{best_cut:2d}/{optimum}"
            )

    print(
        "\nIncreasing p adds 2 parameters per layer and increases circuit "
        "depth, runtime, and optimization difficulty.  It can also raise "
        "the optimized expectation and the probability of an optimal cut."
    )
