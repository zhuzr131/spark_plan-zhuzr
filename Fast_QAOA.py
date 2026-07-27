from time import perf_counter

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import tensorcircuit as tc
from scipy.optimize import minimize

K = tc.set_backend("jax")


# ---------------------------------------------------------------------------
# Change only these settings for a quick experiment.
# ---------------------------------------------------------------------------
ENGINE = "tensorcircuit"  # "tensorcircuit" or "qiskit"
GRAPH = "random"       # cycle, complete-bipartite, complete, random
N = 16
P = 4
MAXITER = 80
SHOTS = 2000
SEED = 7


def make_graph(name, n=N, seed=7):
    """Create correctly numbered graphs with NetworkX."""
    if name == "cycle":
        graph = nx.cycle_graph(n)
    elif name == "complete-bipartite":
        graph = nx.complete_bipartite_graph(n // 2, n - n // 2)
    elif name == "complete":
        graph = nx.complete_graph(n)
    elif name == "random":
        graph = nx.erdos_renyi_graph(n, 0.3, seed=seed)
    else:
        raise ValueError(f"Unknown graph: {name}")

    # TensorCircuit needs consecutive, zero-based qubit indices.
    return nx.convert_node_labels_to_integers(graph)


def cut_value(bits, edges):
    return sum(bits[i] != bits[j] for i, j in edges)


def reference_cut(graph, exact_limit=16, seed=7):
    """Exact for small graphs; fast classical approximation for larger ones."""
    n = graph.number_of_nodes()
    edges = tuple(graph.edges())
    if n <= exact_limit:
        best = -1
        for integer in range(1 << n):
            bits = [(integer >> (n - 1 - i)) & 1 for i in range(n)]
            best = max(best, cut_value(bits, edges))
        return best, "exact"

    value, _ = nx.algorithms.approximation.maxcut.one_exchange(
        graph,
        seed=seed,
    )
    return value, "NetworkX one_exchange reference"


def classical_cost_vector(n, edges):
    """C(z) for all basis states; basis ordering matches tc.Circuit.state()."""
    values = np.empty(1 << n, dtype=np.float32)
    for integer in range(1 << n):
        bits = [(integer >> (n - 1 - i)) & 1 for i in range(n)]
        values[integer] = cut_value(bits, edges)
    return jnp.asarray(values)


def compile_tensorcircuit_objective(graph, p):
    """
    Compile one loss-and-gradient function for a fixed graph and depth.

    This is the main speed improvement over repeatedly constructing the
    circuit for every edge and estimating finite-difference gradients.
    """
    n = graph.number_of_nodes()
    edges = tuple(graph.edges())
    costs = classical_cost_vector(n, edges)

    def probabilities(params):
        circuit = tc.Circuit(n)
        for qubit in range(n):
            circuit.h(qubit)

        gammas, betas = params[:p], params[p:]
        for gamma, beta in zip(gammas, betas):
            for i, j in edges:
                circuit.cnot(i, j)
                circuit.rz(j, theta=-gamma)
                circuit.cnot(i, j)
            for qubit in range(n):
                circuit.rx(qubit, theta=2.0 * beta)

        state = circuit.state()
        return jnp.real(jnp.conj(state) * state)

    def loss(params):
        # One dot product replaces a Python loop of edge expectation calls.
        return -jnp.dot(probabilities(params), costs)

    return jax.jit(jax.value_and_grad(loss)), jax.jit(probabilities)


def solve_with_tensorcircuit(
    graph,
    p=4,
    maxiter=80,
    shots=2000,
    seed=7,
):
    n = graph.number_of_nodes()
    if n > 22:
        raise ValueError(
            "A local statevector needs 2**n amplitudes. For n > 22, use the "
            "Qiskit route with real hardware or install an approximate MPS "
            "simulator instead of increasing local statevector memory."
        )

    compile_start = perf_counter()
    value_and_gradient, probability_function = compile_tensorcircuit_objective(
        graph,
        p,
    )

    initial_point = np.r_[
        np.linspace(0.1, 0.8, p),
        np.linspace(0.8, 0.1, p),
    ]

    # The first call performs JAX compilation. Later calls reuse it.
    value_and_gradient(jnp.asarray(initial_point))
    compile_seconds = perf_counter() - compile_start

    def scipy_value_and_gradient(params):
        value, gradient = value_and_gradient(jnp.asarray(params))
        return float(value), np.asarray(gradient, dtype=float)

    optimize_start = perf_counter()
    result = minimize(
        scipy_value_and_gradient,
        initial_point,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, 2 * np.pi)] * p + [(0, np.pi)] * p,
        options={"maxiter": maxiter},
    )
    optimize_seconds = perf_counter() - optimize_start

    probabilities = np.asarray(probability_function(jnp.asarray(result.x)))
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probabilities), shots, p=probabilities)
    best_integer = max(samples, key=lambda x: classical_cost_vector_value(x, n, graph))
    best_bits = format(int(best_integer), f"0{n}b")
    best_cut = cut_value(best_bits, tuple(graph.edges()))

    return {
        "engine": "TensorCircuit/JAX",
        "params": result.x,
        "expectation": -float(result.fun),
        "best_bits": best_bits,
        "best_cut": best_cut,
        "evaluations": result.nfev,
        "iterations": result.nit,
        "compile_seconds": compile_seconds,
        "optimize_seconds": optimize_seconds,
        "cnot_count": 2 * p * graph.number_of_edges(),
    }


def classical_cost_vector_value(integer, n, graph):
    bits = format(int(integer), f"0{n}b")
    return cut_value(bits, tuple(graph.edges()))


def maxcut_operator_qiskit(graph):
    """
    Return H = 1/2 sum_(i,j) Zi Zj.

    Qiskit QAOA minimizes H. Since
        C = |E|/2 - 1/2 sum_(i,j) Zi Zj,
    minimizing H is equivalent to maximizing the MaxCut objective C.
    """
    from qiskit.quantum_info import SparsePauliOp

    n = graph.number_of_nodes()
    pauli_labels = []
    for i, j in graph.edges():
        label = ["I"] * n
        # Qiskit Pauli strings display qubit 0 on the right.
        label[n - 1 - i] = "Z"
        label[n - 1 - j] = "Z"
        pauli_labels.append("".join(label))
    return SparsePauliOp(pauli_labels, coeffs=[0.5] * len(pauli_labels))


def local_qiskit_sampler(shots, seed):
    """
    Support both the current tutorial API and the older installed API.

    Tutorial: Qiskit 2.x -> StatevectorSampler
    This environment: Qiskit 0.46 -> Sampler
    """
    try:
        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler(default_shots=shots, seed=seed)
    except ImportError:
        from qiskit.primitives import Sampler

        return Sampler(options={"shots": shots, "seed": seed})


def most_likely_bits(distribution, n):
    if hasattr(distribution, "binary_probabilities"):
        distribution = distribution.binary_probabilities(num_bits=n)

    key = max(distribution, key=distribution.get)
    if isinstance(key, int):
        qiskit_order = format(key, f"0{n}b")
    else:
        qiskit_order = str(key).replace(" ", "").zfill(n)

    # Convert Qiskit's q_(n-1)...q_0 display to bits[qubit_index].
    return qiskit_order[::-1]


def solve_with_qiskit(
    graph,
    p=4,
    maxiter=80,
    shots=2000,
    seed=7,
    optimizer_name="COBYLA",
):
    """
    Qiskit implementation following the official QAOA tutorial.

    For shot noise or hardware, try optimizer_name="SPSA".
    The local sampler remains a statevector-based simulator.
    """
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = seed
    optimizer = (
        SPSA(maxiter=maxiter)
        if optimizer_name.upper() == "SPSA"
        else COBYLA(maxiter=maxiter)
    )
    sampler = local_qiskit_sampler(shots, seed)
    initial_point = np.r_[
        np.linspace(0.1, 0.8, p),
        np.linspace(0.8, 0.1, p),
    ]

    start = perf_counter()
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=p,
        initial_point=initial_point,
    )
    result = qaoa.compute_minimum_eigenvalue(maxcut_operator_qiskit(graph))
    runtime = perf_counter() - start

    bits_by_qubit = most_likely_bits(result.eigenstate, graph.number_of_nodes())
    best_cut = cut_value(bits_by_qubit, tuple(graph.edges()))

    return {
        "engine": f"Qiskit local Sampler + {optimizer_name.upper()}",
        "params": result.optimal_point,
        "expectation": len(graph.edges()) / 2 - float(result.eigenvalue.real),
        "best_bits": bits_by_qubit,
        "best_cut": best_cut,
        "evaluations": result.cost_function_evals,
        "iterations": "reported through evaluations",
        "compile_seconds": "included below",
        "optimize_seconds": runtime,
        "cnot_count": 2 * p * graph.number_of_edges(),
    }


def print_result(graph, result, reference, reference_kind):
    print("\nQAOA result")
    print("-" * 64)
    print(f"engine          : {result['engine']}")
    print(
        f"graph           : n={graph.number_of_nodes()}, "
        f"|E|={graph.number_of_edges()}"
    )
    print(f"expected cut    : {result['expectation']:.3f}")
    print(f"best sampled    : {result['best_cut']}  ({result['best_bits']})")
    print(f"reference       : {reference}  ({reference_kind})")
    print(f"best/reference  : {result['best_cut'] / reference:.3f}")
    print(f"evaluations     : {result['evaluations']}")
    print(f"JIT compile (s) : {result['compile_seconds']}")
    print(f"optimization(s) : {result['optimize_seconds']:.3f}")
    print(f"approx. CNOTs   : {result['cnot_count']}")
    print("-" * 64)
    print(
        "Note: increasing |E| raises gate count; increasing n raises local "
        "statevector memory as 2**n."
    )


if __name__ == "__main__":
    selected_graph = make_graph(GRAPH, N, SEED)
    reference, reference_kind = reference_cut(selected_graph, seed=SEED)

    if ENGINE == "qiskit":
        answer = solve_with_qiskit(
            selected_graph,
            p=P,
            maxiter=MAXITER,
            shots=SHOTS,
            seed=SEED,
            optimizer_name="COBYLA",
        )
    else:
        answer = solve_with_tensorcircuit(
            selected_graph,
            p=P,
            maxiter=MAXITER,
            shots=SHOTS,
            seed=SEED,
        )

    print_result(selected_graph, answer, reference, reference_kind)
