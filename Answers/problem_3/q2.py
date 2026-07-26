"""Problem 3, Question 2: p=1 QAOA-MaxCut warm-up."""

import numpy as np
import tensorcircuit as tc
from scipy.optimize import minimize

from q1 import GRAPHS, K, exact_solution, maxcut_value


def qaoa_circuit(n, edges, gammas, betas):
    """Prepare |gamma,beta> with TensorCircuit."""
    c = tc.Circuit(n)

    # Initial state |s> = |+>^n.
    for i in range(n):
        c.H(i)

    for gamma, beta in zip(gammas, betas):
        # U(C,gamma).  RZZ(-gamma) differs only by a global phase.
        for i, j in edges:
            c.cnot(i, j)
            c.rz(j, theta=-gamma)
            c.cnot(i, j)

        # U(B,beta) = product_i RX_i(2 beta).
        for i in range(n):
            c.rx(i, theta=2.0 * beta)

    return c


def expectation(n, edges, params):
    """Exact statevector value <C> for p=len(params)/2."""
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


def sample_qaoa(n, edges, params, shots=100):
    """Measure a QAOA state and return counts, average, and best sample."""
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


def optimize_exact(n, edges, p, x0=None, method="COBYLA", maxiter=100):
    """Optimize the exact expectation; useful in Questions 3-8."""
    if x0 is None:
        x0 = np.r_[
            np.linspace(0.2, 0.8, p),
            np.linspace(0.7, 0.2, p),
        ]

    bounds = None
    if method == "L-BFGS-B":
        bounds = [(0.0, 2.0 * np.pi)] * p + [(0.0, np.pi)] * p

    def loss(x):
        return -expectation(n, edges, x)

    def central_gradient(x):
        """Stable finite differences for JAX's default single precision."""
        step = 1e-3
        gradient = np.zeros_like(x)
        for k in range(len(x)):
            forward = x.copy()
            backward = x.copy()
            forward[k] += step
            backward[k] -= step
            gradient[k] = (loss(forward) - loss(backward)) / (2.0 * step)
        return gradient

    jac = central_gradient if method in {"BFGS", "L-BFGS-B"} else None

    return minimize(
        loss,
        np.asarray(x0, dtype=float),
        method=method,
        jac=jac,
        bounds=bounds,
        options={"maxiter": maxiter},
    )


def optimize_p1_with_shots(n, edges, shots=100):
    """Finite-shot grid optimizer for the two p=1 parameters."""
    best_params = None
    best_average = -1.0

    gammas = np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)
    betas = np.linspace(0.0, np.pi, 7, endpoint=False)

    for gamma in gammas:
        for beta in betas:
            params = np.array([gamma, beta])
            _, average, _, _ = sample_qaoa(n, edges, params, shots)
            if average > best_average:
                best_params = params
                best_average = average

    return best_params, best_average


if __name__ == "__main__":
    SHOTS = 100

    for name, (n, edges) in GRAPHS.items():
        params, training_average = optimize_p1_with_shots(n, edges, SHOTS)
        counts, average, best_z, best_cut = sample_qaoa(
            n, edges, params, SHOTS
        )
        optimum = exact_solution(n, edges)[1]

        print(f"\n{name}")
        print(f"gamma={params[0]:.4f}, beta={params[1]:.4f}")
        print(f"training estimate F1={training_average:.3f}")
        print(f"final estimate F1={average:.3f}")
        print(f"best sample={best_z}, C(z)={best_cut}/{optimum}")
        print(f"distinct measured strings={len(counts)}")
