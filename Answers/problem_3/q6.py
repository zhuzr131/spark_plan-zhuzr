"""Problem 3, Question 6: effect of the number of measurement shots."""

import numpy as np

from q1 import GRAPHS
from q2 import optimize_exact, sample_qaoa


if __name__ == "__main__":
    SHOT_NUMBERS = [10, 100, 1000, 10000]
    REPEATS = 10

    print("graph  shots  mean F  std(F)  mean best cut")
    for name, (n, edges) in GRAPHS.items():
        # Optimize once, then isolate the statistical effect of shot count.
        params = optimize_exact(n, edges, p=1, maxiter=80).x

        for shots in SHOT_NUMBERS:
            averages = []
            best_cuts = []
            for _ in range(REPEATS):
                _, average, _, best_cut = sample_qaoa(
                    n, edges, params, shots=shots
                )
                averages.append(average)
                best_cuts.append(best_cut)

            print(
                f"{name:5s} {shots:5d} "
                f"{np.mean(averages):7.3f} "
                f"{np.std(averages, ddof=1):7.3f} "
                f"{np.mean(best_cuts):13.3f}"
            )

    print(
        "\nThe estimator remains unbiased, while its standard error "
        "decreases approximately as 1/sqrt(Nshot). More shots also make "
        "it more likely that a high-value bitstring is observed."
    )
