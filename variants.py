"""
QAOA variants for Task 9b.
Supports: multi-angle QAOA, warm-start QAOA.
"""

import tensorcircuit as tc
import numpy as np
import networkx as nx

tc.set_backend("jax")
tc.set_dtype("complex64")


def ma_qaoa_circuit(params: np.ndarray, graph: nx.Graph):
    """
    Multi-angle QAOA: each edge and each qubit gets its own parameter.
    params = [gamma_0..gamma_{|E|-1}, beta_0..beta_{n-1}]  (for p=1)
    For p > 1, params shape is p * (|E| + n).
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    m = len(edges)
    n_layers = len(params) // (m + n)

    def energy(params_arr):
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)

        for layer in range(n_layers):
            offset = layer * (m + n)
            # Problem unitary with per-edge gammas
            for idx, (u, v) in enumerate(edges):
                gamma = params_arr[offset + idx]
                c.rzz(u, v, theta=2 * gamma)
            # Mixer unitary with per-qubit betas
            for i in range(n):
                beta = params_arr[offset + m + i]
                c.rx(i, theta=2 * beta)

        # Compute <C>
        exp_val = 0.0
        for u, v in edges:
            zz = c.expectation_ps(z=[u, v])
            exp_val += 0.5 * (1.0 - tc.backend.real(zz))
        return exp_val

    return energy


def warmstart_qaoa_circuit(params: np.ndarray, graph: nx.Graph, init_cut: list[int]):
    """
    Warm-start QAOA: initialize qubits based on a classical solution.
    init_cut: list of 0/1 for each qubit indicating partition.

    Each qubit is prepared in |init_theta_i> = RY(theta_i)|0>,
    where theta_i ~ pi/3 if qubit is in partition 0, or 2pi/3 if in partition 1.
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    n_layers = len(params) // 2

    def energy(params_arr):
        c = tc.Circuit(n)

        # Warm-start initialization
        for i in range(n):
            theta = np.pi / 3 if init_cut[i] == 0 else 2 * np.pi / 3
            c.ry(i, theta=theta)

        for layer in range(n_layers):
            gamma = params_arr[2 * layer]
            beta = params_arr[2 * layer + 1]
            for u, v in edges:
                c.rzz(u, v, theta=2 * gamma)
            for i in range(n):
                c.rx(i, theta=2 * beta)

        exp_val = 0.0
        for u, v in edges:
            zz = c.expectation_ps(z=[u, v])
            exp_val += 0.5 * (1.0 - tc.backend.real(zz))
        return exp_val

    return energy


def fourier_qaoa_circuit(params: np.ndarray, graph: nx.Graph, n_layers: int = 1):
    """
    Fourier parameterization: express gammas and betas as Fourier series.
    Reduces parameter count for large p.
    params: q_fourier coefficients for gamma, q_fourier for beta.
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    n_fourier = len(params) // 2

    def energy(params_arr):
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)

        gamma_coeffs = params_arr[:n_fourier]
        beta_coeffs = params_arr[n_fourier:]

        for layer in range(n_layers):
            s = (layer + 0.5) / n_layers
            # gamma_l = sum_k gamma_k * sin(k*pi*s)
            gamma = sum(gamma_coeffs[k] * np.sin((k + 1) * np.pi * s) for k in range(n_fourier))
            # beta_l = sum_k beta_k * cos(k*pi*s)
            beta = sum(beta_coeffs[k] * np.cos((k + 1) * np.pi * s) for k in range(n_fourier))

            for u, v in edges:
                c.rzz(u, v, theta=2 * gamma)
            for i in range(n):
                c.rx(i, theta=2 * beta)

        exp_val = 0.0
        for u, v in edges:
            zz = c.expectation_ps(z=[u, v])
            exp_val += 0.5 * (1.0 - tc.backend.real(zz))
        return exp_val

    return energy
