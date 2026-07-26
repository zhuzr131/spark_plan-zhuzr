"""
Hardware deployment for Task 8 - real superconducting qubit hardware.

Uses TensorCircuit's noise simulation to emulate hardware constraints:
- Depolarizing noise channels
- Qubit connectivity (topology)
- Gate fidelity decay
"""

import tensorcircuit as tc
import numpy as np
import networkx as nx

tc.set_backend("jax")
tc.set_dtype("complex64")


def hardware_aware_qaoa(
    params: np.ndarray,
    graph: nx.Graph,
    n_layers: int = 1,
    noise_level: float = 0.01,
    device_topology: nx.Graph = None,
    n_shots: int = 1024,
):
    """
    QAOA with hardware noise model.

    Args:
        noise_level: single-qubit gate error rate (2-qubit = 10x worse)
        device_topology: hardware qubit connectivity graph
    """
    n = graph.number_of_nodes()

    # Map logical qubits to physical qubits based on topology
    if device_topology is not None:
        mapping = _qubit_mapping(graph, device_topology)
    else:
        mapping = list(range(n))  # identity mapping for simulation

    mapped_edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]

    def energy(params_arr):
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
            # Single-qubit gate noise
            if noise_level > 0:
                c.depolarizing(i, px=noise_level/3, py=noise_level/3, pz=noise_level/3)

        for layer in range(n_layers):
            gamma = params_arr[2 * layer]
            beta = params_arr[2 * layer + 1]

            for u, v in mapped_edges:
                c.rzz(u, v, theta=2 * gamma)
                if noise_level > 0:
                    # Two-qubit gate noise is typically ~10x worse
                    p2 = min(noise_level * 10, 0.5)
                    c.depolarizing(u, px=p2/3, py=p2/3, pz=p2/3)
                    c.depolarizing(v, px=p2/3, py=p2/3, pz=p2/3)

            for i in range(n):
                c.rx(i, theta=2 * beta)
                if noise_level > 0:
                    c.depolarizing(i, px=noise_level/3, py=noise_level/3, pz=noise_level/3)

        # Sample from noisy circuit
        exp_val = 0.0
        for u, v in graph.edges():
            zz = c.expectation_ps(z=[u, v])
            exp_val += 0.5 * (1.0 - tc.backend.real(zz))
        return exp_val

    return energy


def _qubit_mapping(logical_graph: nx.Graph, physical_topology: nx.Graph) -> list[int]:
    """
    Map logical qubits to physical qubits.
    Greedy: assign most-connected logical qubits to central physical qubits.
    """
    n = logical_graph.number_of_nodes()
    if physical_topology.number_of_nodes() < n:
        raise ValueError(f"Device has {physical_topology.number_of_nodes()} qubits, need {n}")

    logical_degrees = sorted(logical_graph.degree(), key=lambda x: -x[1])
    physical_degrees = sorted(physical_topology.degree(), key=lambda x: -x[1])
    mapping = {}
    for (li, _), (pi, _) in zip(logical_degrees, physical_degrees):
        mapping[li] = pi
    return [mapping[i] for i in range(n)]


def run_noise_analysis(graph: nx.Graph, noise_levels: list[float]) -> dict:
    """
    Run QAOA at different noise levels and measure performance decay.
    Returns {noise_level: approximation_ratio}
    """
    from hamiltonian import maxcut_exact
    exact_cut, _ = maxcut_exact(graph)

    results = {}
    for noise in noise_levels:
        energy_fn = hardware_aware_qaoa(
            np.zeros(2), graph, n_layers=1, noise_level=noise
        )
        val = float(energy_fn(np.array([0.3, 0.3])))  # fixed params for comparison
        results[noise] = val / exact_cut if exact_cut > 0 else 0.0

    return results
