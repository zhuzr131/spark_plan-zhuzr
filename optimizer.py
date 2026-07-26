"""
Classical optimizer wrappers for QAOA parameter training.
Supports: Adam, BFGS, COBYLA, SPSA.
"""

import numpy as np
from scipy.optimize import minimize


def optimize_adam(
    energy_fn, initial_params: np.ndarray,
    lr: float = 0.01, max_iter: int = 500, tol: float = 1e-5
) -> tuple[np.ndarray, float, list]:
    """Adam optimizer (gradient-based)."""
    import tensorcircuit as tc

    energy_vg = tc.backend.value_and_grad(energy_fn)
    params = np.array(initial_params, dtype=np.float32)
    history = []

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for it in range(max_iter):
        val, grad = energy_vg(params)
        grad = np.array(grad)
        history.append(float(val))

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        if it > 0 and abs(history[-1] - history[-2]) < tol:
            break

    return params, history[-1], history


def optimize_bfgs(
    energy_fn, initial_params: np.ndarray, max_iter: int = 500
) -> tuple[np.ndarray, float, list]:
    """L-BFGS-B optimizer (quasi-Newton)."""
    import tensorcircuit as tc
    energy_vg = tc.backend.value_and_grad(energy_fn)

    history = []

    def fun(x):
        x_tc = tc.array_to_tensor(x)
        val, _ = energy_vg(x_tc)
        return float(val)

    def jac(x):
        x_tc = tc.array_to_tensor(x)
        _, grad = energy_vg(x_tc)
        return np.array(grad, dtype=np.float64)

    res = minimize(
        fun, np.array(initial_params), method="L-BFGS-B",
        jac=jac, options={"maxiter": max_iter, "disp": False},
        callback=lambda x: history.append(fun(x)),
    )
    if not history:
        history.append(fun(res.x))
    return res.x, history[-1], history


def optimize_cobyla(
    energy_fn, initial_params: np.ndarray, max_iter: int = 500
) -> tuple[np.ndarray, float, list]:
    """COBYLA optimizer (derivative-free, constrained)."""
    history = []

    def fun(x):
        val = energy_fn(x)
        history.append(float(val))
        return -float(val)  # COBYLA minimizes, we want max cut

    res = minimize(
        fun, np.array(initial_params), method="COBYLA",
        options={"maxiter": max_iter, "disp": False},
    )
    if not history:
        history.append(-fun(res.x))
    return res.x, -history[-1], history


def optimize_spsa(
    energy_fn, initial_params: np.ndarray,
    max_iter: int = 500, a: float = 0.1, c: float = 0.1
) -> tuple[np.ndarray, float, list]:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation).
    Derivative-free, robust to noise.
    """
    params = np.array(initial_params, dtype=np.float64)
    history = [float(energy_fn(params))]

    A = max_iter / 10
    for k in range(1, max_iter + 1):
        ak = a / (k + A) ** 0.602
        ck = c / k ** 0.101
        delta = np.random.choice([-1, 1], size=params.shape)

        params_plus = params + ck * delta
        params_minus = params - ck * delta

        f_plus = float(energy_fn(params_plus))
        f_minus = float(energy_fn(params_minus))

        g_hat = (f_plus - f_minus) / (2 * ck * delta)
        params = params + ak * g_hat
        history.append(float(energy_fn(params)))

    return params, history[-1], history


OPTIMIZERS = {
    "adam": optimize_adam,
    "bfgs": optimize_bfgs,
    "cobyla": optimize_cobyla,
    "spsa": optimize_spsa,
}
