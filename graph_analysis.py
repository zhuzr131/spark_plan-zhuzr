import gc
import os
from itertools import product
from time import perf_counter

import jax
import numpy as np
import pandas as pd
import tensorcircuit as tc
from IPython.display import display
from scipy.optimize import OptimizeResult, minimize

K = tc.set_backend("jax")

GRAPHS = {
    # TensorCircuit uses zero-based qubit indices: 0, ..., n - 1.
    "Petersen": (
        10,
        [
            (0, 2), (2, 4), (4, 6), (6, 8), (0, 8),
            (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
            (1, 5), (3, 7), (5, 9), (1, 7), (3, 9),
        ],
    ),
    "1 * 4 rectangle": (
        10,
        [
            (1, 2), (1, 4), (2, 3), (3, 4), (3, 6),
            (4, 5), (5, 6), (5, 8), (6, 7), (7, 8),
            (0, 7), (8, 9), (0, 9),
        ],
    ),
    "Complete bipartite": (
        10,
        [
            (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 6),
            (1, 7), (1, 8), (1, 9), (2, 5), (2, 6), (2, 7), (2, 8),
            (2, 9), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 5), 
            (4, 6), (4, 7), (4, 8), (4, 9),
        ],
    ),
    "Complete":(
        10,
        [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 3),
            (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6),
            (3, 7), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 6),
            (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
        ],
    ),
    "Random1": (
        10,
        [
            (0, 2), (1, 2), (0, 3), (0, 8), (1, 4),
            (3, 5), (3, 7), (4, 6), (6, 8), (7, 9),
            (8, 9),
        ],
    ),
    "Random2": (
        10,
        [
            (0, 1), (0, 2), (0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 8),
            (1, 9), (2, 3), (2, 5), (2, 6), (2, 8), (3, 4), (3, 5), (3, 6),
            (3, 7), (3, 9), (4, 5), (4, 8), (5, 6), (5, 8), (5, 9), (6, 7),
            (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
        ],
    ),
}

print("TensorCircuit backend:", tc.backend.name)


def validate_graph(n, edges):
    """Check that every endpoint is a valid zero-based qubit index."""
    invalid = [(i, j) for i, j in edges if not (0 <= i < n and 0 <= j < n)]
    if invalid:
        raise ValueError(
            f"For n={n}, vertices must be numbered from 0 to {n - 1}; "
            f"invalid edges: {invalid}"
        )


def maxcut_value(z, edges):
    """Number of edges crossing the cut described by z."""
    return sum(z[i] != z[j] for i, j in edges)


def exact_solution(n, edges):
    """Find an exact MaxCut solution by checking all 2^n strings."""
    validate_graph(n, edges)
    best_z, best_value = None, -1
    for z in product([0, 1], repeat=n):
        value = maxcut_value(z, edges)
        if value > best_value:
            best_z, best_value = z, value
    return best_z, best_value


def qaoa_circuit(n, edges, gammas, betas):
    validate_graph(n, edges)
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)  # |s> = |+>^n

    for gamma, beta in zip(gammas, betas):
        # U(C, gamma), up to an irrelevant global phase.
        for i, j in edges:
            c.cnot(i, j)
            c.rz(j, theta=-gamma)
            c.cnot(i, j)

        # U(B, beta) = product_i RX_i(2 beta).
        for i in range(n):
            c.rx(i, theta=2.0 * beta)
    return c


def expectation(n, edges, params):
    p = len(params) // 2
    c = qaoa_circuit(n, edges, params[:p], params[p:])
    value = 0.0
    for i, j in edges:
        zz = c.expectation(
            [tc.gates.z(), [i]],
            [tc.gates.z(), [j]],
        )
        value += (1.0 - K.real(zz)) / 2.0
    return float(K.numpy(value))


def numerical_gradient(n, edges, x, step=1e-3):
    """Central finite-difference gradient of -<C>."""
    gradient = np.zeros_like(x, dtype=float)
    for k in range(len(x)):
        forward, backward = x.copy(), x.copy()
        forward[k] += step
        backward[k] -= step
        gradient[k] = (
            -expectation(n, edges, forward)
            + expectation(n, edges, backward)
        ) / (2 * step)
    return gradient


def sample_qaoa(n, edges, params, shots=100):
    p = len(params) // 2
    c = qaoa_circuit(n, edges, params[:p], params[p:])
    counts = c.sample(
        batch=shots,
        allow_state=True,
        format="count_dict_bin",
    )
    average = sum(
        maxcut_value(z, edges) * count for z, count in counts.items()
    ) / shots
    best_z = max(counts, key=lambda z: maxcut_value(z, edges))
    return counts, average, best_z, maxcut_value(best_z, edges)


def optimize_exact(
    n, edges, p, x0=None, method="COBYLA", maxiter=80,
    learning_rate=None,
):
    if x0 is None:
        x0 = np.r_[
            np.linspace(0.2, 0.8, p),
            np.linspace(0.7, 0.2, p),
        ]

    def loss(x):
        return -expectation(n, edges, x)

    method = method.upper()

    if method == "GD":
        x = np.asarray(x0, dtype=float).copy()
        learning_rate = 0.04 if learning_rate is None else learning_rate
        for _ in range(maxiter):
            x -= learning_rate * numerical_gradient(n, edges, x)
        return OptimizeResult(
            x=x, fun=loss(x), nit=maxiter,
            nfev=2 * len(x) * maxiter + 1,
            success=True, message="Gradient Descent completed.",
        )

    if method == "ADAM":
        x = np.asarray(x0, dtype=float).copy()
        learning_rate = 0.08 if learning_rate is None else learning_rate
        beta1, beta2 = 0.9, 0.999
        first_moment = np.zeros_like(x)
        second_moment = np.zeros_like(x)

        for iteration in range(1, maxiter + 1):
            gradient = numerical_gradient(n, edges, x)
            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
            m_hat = first_moment / (1 - beta1**iteration)
            v_hat = second_moment / (1 - beta2**iteration)
            x -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        return OptimizeResult(
            x=x, fun=loss(x), nit=maxiter,
            nfev=2 * len(x) * maxiter + 1,
            success=True, message="Adam completed.",
        )

    bounds = None
    if method == "L-BFGS-B":
        bounds = [(0, 2 * np.pi)] * p + [(0, np.pi)] * p
    jac = None
    if method in {"BFGS", "L-BFGS-B"}:
        jac = lambda x: numerical_gradient(n, edges, x)

    result = minimize(
        loss,
        np.asarray(x0, dtype=float),
        method=method,
        jac=jac,
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    if method in {"BFGS", "L-BFGS-B"}:
        result.nfev += 2 * len(result.x) * result.njev
    if not hasattr(result, "nit"):
        result.nit = result.nfev
    return result


def initial_parameters(p, strategy, seed=7):
    if strategy == "zero":
        return np.zeros(2 * p)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.r_[
            rng.uniform(0, 2 * np.pi, p),
            rng.uniform(0, np.pi, p),
        ]
    if strategy == "linear-ramp":
        return np.r_[
            np.linspace(0.1, 0.8, p),
            np.linspace(0.8, 0.1, p),
        ]
    raise ValueError(strategy)


def best_multistart_result(n, edges, p=4):
    starts = [initial_parameters(p, "linear-ramp")]
    starts += [initial_parameters(p, "random", seed) for seed in (3, 11)]
    results = [
        optimize_exact(
            n, edges, p, x0=x0,
            method="COBYLA", maxiter=80,
        )
        for x0 in starts
    ]
    return min(results, key=lambda result: result.fun)


if __name__ == "__main__":
    for name, (n, edges) in GRAPHS.items():
        start = perf_counter()
        result = best_multistart_result(n, edges)
        runtime = perf_counter() - start
        average = expectation(n, edges, result.x)
        _, _, best_z, best_cut = sample_qaoa(n, edges, result.x, 2000)
        optimum = exact_solution(n, edges)[1]
        print(
            f"{name}: runtime={runtime:.3f}s, "
            f"iterations={getattr(result, 'nit', result.nfev)}, "
            f"<C>={average:.3f}, z={best_z}, "
            f"cut={best_cut}, ratio={best_cut/optimum:.3f}"
        )
        gc.collect()
        jax.clear_caches()

print(
    "\nConclusion 1: the runtime of the algorithm is highly influenced by the amount of edges."
)
