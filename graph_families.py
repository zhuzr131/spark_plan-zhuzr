"""
Graph family generators for Task 9a.
"""

import networkx as nx
import numpy as np


def path_graph(n: int) -> nx.Graph:
    """Path graph P_n."""
    return nx.path_graph(n)


def cycle_graph(n: int) -> nx.Graph:
    """Cycle graph C_n."""
    return nx.cycle_graph(n)


def random_regular_graph(n: int, d: int, seed: int = None) -> nx.Graph:
    """
    Random d-regular graph on n vertices.
    Uses NetworkX's generator.
    """
    return nx.random_regular_graph(d, n, seed=seed)


def erdos_renyi_graph(n: int, p: float, seed: int = None) -> nx.Graph:
    """Erdos-Renyi G(n, p) random graph."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def complete_bipartite_graph(a: int, b: int) -> nx.Graph:
    """Complete bipartite graph K_{a,b}."""
    return nx.complete_bipartite_graph(a, b)


def weighted_graph(graph: nx.Graph, seed: int = None) -> nx.Graph:
    """Add random integer weights in [1, 10] to a graph."""
    rng = np.random.RandomState(seed)
    G = graph.copy()
    for u, v in G.edges():
        G[u][v]['weight'] = rng.randint(1, 11)
    return G


def regular_graph_family(n: int, d_list: list[int], n_samples: int = 5) -> list[tuple[str, nx.Graph]]:
    """Generate family of d-regular graphs."""
    graphs = []
    for d in d_list:
        for s in range(n_samples):
            seed = 100 * d + s
            g = random_regular_graph(n, d, seed=seed)
            graphs.append((f"RG_n{n}_d{d}_s{s}", g))
    return graphs


def er_graph_family(n: int, p_list: list[float], n_samples: int = 10) -> list[tuple[str, nx.Graph]]:
    """Generate family of ER random graphs."""
    graphs = []
    for p in p_list:
        for s in range(n_samples):
            seed = 1000 * int(p * 100) + s
            g = erdos_renyi_graph(n, p, seed=seed)
            graphs.append((f"ER_n{n}_p{p}_s{s}", g))
    return graphs
