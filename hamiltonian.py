"""
MaxCut Hamiltonian construction.
C = sum_{(i,j) in E} (I - Z_i Z_j) / 2
"""

import networkx as nx
import numpy as np


def maxcut_cost(z: np.ndarray, graph: nx.Graph) -> float:
    """Classical MaxCut cost for bitstring z in {0,1}^n."""
    cost = 0.0
    for u, v in graph.edges():
        if z[u] != z[v]:
            cost += 1.0
    return cost


def maxcut_hamiltonian_terms(graph: nx.Graph) -> list[tuple[float, list[tuple[int, str]]]]:
    """
    Return C as list of (coefficient, Pauli terms).
    Each Pauli term is [(qubit_idx, 'Z'), ...].
    For MaxCut: C = sum_{(i,j)}(0.5*I - 0.5*Z_i Z_j)
    """
    n = graph.number_of_nodes()
    terms = []
    # Constant term |E|/2 * I (we can track or ignore for optimization)
    coeff_i = graph.number_of_edges() / 2.0
    terms.append((coeff_i, []))  # identity
    for u, v in graph.edges():
        terms.append((-0.5, [(int(u), 'Z'), (int(v), 'Z')]))
    return terms


def build_graph_from_edges(n: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """Create a NetworkX graph from edge list."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


# Predefined graphs from Problem 3
def graph_c4():
    """C4: cycle of 4 vertices."""
    return build_graph_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])


def graph_g6():
    """G6: 6 vertices, 7 edges (as specified in the problem)."""
    return build_graph_from_edges(6, [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (1, 4)
    ])


def graph_g9():
    """G9: 9 vertices, 12 edges (as specified in the problem)."""
    return build_graph_from_edges(9, [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (0, 4), (2, 6),
        (1, 5), (3, 7)
    ])


def maxcut_exact(graph: nx.Graph) -> tuple[int, list[int]]:
    """Brute-force exact MaxCut for small graphs."""
    n = graph.number_of_nodes()
    best_cut = -1
    best_z = None
    for x in range(1 << n):
        z = [(x >> i) & 1 for i in range(n)]
        cut = maxcut_cost(np.array(z), graph)
        if cut > best_cut:
            best_cut = cut
            best_z = z
    return best_cut, best_z
