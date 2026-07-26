"""
Initialization strategies for QAOA parameters.
"""

import numpy as np


def init_zero(n_layers: int) -> np.ndarray:
    """All parameters initialized to 0."""
    return np.zeros(2 * n_layers)


def init_random(n_layers: int, seed: int = None) -> np.ndarray:
    """
    Uniform random initialization.
    gamma ~ U[0, pi], beta ~ U[0, pi/2] (reasonable ranges).
    """
    rng = np.random.RandomState(seed)
    params = np.zeros(2 * n_layers)
    for layer in range(n_layers):
        params[2 * layer] = rng.uniform(0, np.pi)       # gamma
        params[2 * layer + 1] = rng.uniform(0, np.pi / 2)  # beta
    return params


def init_linear_ramp(n_layers: int) -> np.ndarray:
    """
    Linear ramp initialization (annealing-inspired).
    gamma_l = (l + 0.5) / p * dt, beta_l = (1 - (l + 0.5) / p) * dt
    where dt ~ 0.5-1.0.
    """
    params = np.zeros(2 * n_layers)
    dt = 0.7  # total "time"
    for layer in range(n_layers):
        s = (layer + 0.5) / n_layers
        params[2 * layer] = s * dt        # gamma: 0 -> dt
        params[2 * layer + 1] = (1 - s) * dt  # beta: dt -> 0
    return params


def init_trotter(n_layers: int, T: float = 1.0) -> np.ndarray:
    """
    Trotterized adiabatic initialization.
    Uniform step sizes over total time T.
    """
    params = np.zeros(2 * n_layers)
    dt = T / n_layers
    for layer in range(n_layers):
        params[2 * layer] = dt
        params[2 * layer + 1] = dt
    return params


INIT_STRATEGIES = {
    "zero": init_zero,
    "random": init_random,
    "linear_ramp": init_linear_ramp,
    "trotter": init_trotter,
}
