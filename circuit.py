"""
QAOA circuit construction using TensorCircuit with JAX backend.
"""

import tensorcircuit as tc
import numpy as np

# Force JAX backend for automatic differentiation
tc.set_backend("jax")
tc.set_dtype("complex64")


def qaoa_circuit(graph, n_layers: int = 1):
    """
    Build QAOA circuit with p = n_layers.
    Returns a function energy(params) that computes <C>.
    params: array of shape (2 * n_layers,) -> [gamma_0, beta_0, gamma_1, beta_1, ...]
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())

    def energy(params):
        c = tc.Circuit(n)
        # Initial state: |+>^{⊗ n}
        for i in range(n):
            c.h(i)

        for layer in range(n_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]

            # Problem unitary: apply ZZ interactions
            for u, v in edges:
                c.rzz(u, v, theta=2 * gamma)
            # Mixer unitary: apply RX rotations
            for i in range(n):
                c.rx(i, theta=2 * beta)

        # Measure <C> via sampling or exact expectation
        return compute_expectation(c, edges, n)

    return energy


def compute_expectation(circuit: tc.Circuit, edges: list, n: int) -> float:
    """Compute <C> = sum (1 - <Z_i Z_j>)/2 from the circuit."""
    exp_val = 0.0
    for u, v in edges:
        zz = circuit.expectation_ps(z=[u, v])
        exp_val += 0.5 * (1.0 - tc.backend.real(zz))
    return exp_val


def qaoa_circuit_sampling(params: np.ndarray, graph, n_layers: int = 1, n_shots: int = 1024):
    """
    QAOA circuit with shot-based sampling.
    Returns the estimated <C> from measurement samples.
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())

    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)

    for layer in range(n_layers):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        for u, v in edges:
            c.rzz(u, v, theta=2 * gamma)
        for i in range(n):
            c.rx(i, theta=2 * beta)

    # Measure all qubits
    samples = c.sample(batch=n_shots, allow_state=True, format_="count_dict_bin")
    # Estimate <C> from samples
    energy = 0.0
    total = sum(samples.values())
    for bitstring, count in samples.items():
        # bitstring is like "010...", MSB = qubit 0
        for u, v in edges:
            if bitstring[n - 1 - u] != bitstring[n - 1 - v]:
                energy += count / total
    return energy
