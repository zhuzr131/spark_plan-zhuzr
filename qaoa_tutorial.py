"""
Quantum Approximate Optimization Algorithm (QAOA) 教程整合脚本

来源: https://qiskit-community.github.io/qiskit-algorithms/tutorials/05_qaoa.html
版权: (c) Copyright IBM 2017, 2025, Qiskit Algorithms Development Team

本脚本将官方 QAOA 教程中的所有 Python 代码整合到一起，演示如何用 QAOA
求解图分区（Graph Partitioning）问题，并与暴力法、经典特征值求解器、
SamplingVQE 等方法进行对比，最后演示自定义转译器（PassManager）的用法。

依赖环境（参考版本）:
    qiskit==2.0.3
    qiskit-aer==0.17.1
    qiskit-algorithms==0.4.0
    networkx
    numpy
    matplotlib

运行:
    python qaoa_tutorial.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import n_local
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def objective_value(x, w):
    """Compute the value of a cut.

    Args:
        x: Binary string as numpy array.
        w: Adjacency matrix.

    Returns:
        Value of the cut.
    """
    X = np.outer(x, (1 - x))
    w_01 = np.where(w != 0, 1, 0)
    return np.sum(w_01 * X)


def bitfield(n, L):
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]


def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.

    Args:
        state_vector: Quasi-distribution.

    Returns:
        Array of bits.
    """
    return np.array([int(bit) for bit in max(state_vector.items(), key=lambda x: x[1])[0][::-1]])


def get_operator(weight_matrix):
    r"""Generate Hamiltonian for the graph partitioning.

    Notes:
        Goals:
            1 Separate the vertices into two set of the same size.
            2 Make sure the number of edges between the two set is minimized.
        Hamiltonian:
            H = H_A + H_B
            H_A = sum\_{(i,j)\in E}{(1-ZiZj)/2}
            H_B = (sum_{i}{Zi})^2 = sum_{i}{Zi^2}+sum_{i!=j}{ZiZj}
            H_A is for achieving goal 2 and H_B is for achieving goal 1.

    Args:
        weight_matrix: Adjacency matrix.

    Returns:
        Operator for the Hamiltonian.
        A constant shift for the obj function.
    """
    num_nodes = len(weight_matrix)
    pauli_list = []
    coeffs = []
    shift = 0

    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append(Pauli((z_p, x_p)))
                coeffs.append(-0.5)
                shift += 0.5

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append(Pauli((z_p, x_p)))
                coeffs.append(1.0)
            else:
                shift += 1

    return SparsePauliOp(pauli_list, coeffs=coeffs), shift


# ---------------------------------------------------------------------------
# 步骤 1: 定义图（邻接矩阵）
# ---------------------------------------------------------------------------
def build_graph():
    num_nodes = 4
    w = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )
    G = nx.from_numpy_array(w)
    return num_nodes, w, G


# ---------------------------------------------------------------------------
# 步骤 2: 可视化图
# ---------------------------------------------------------------------------
def draw_graph(G, save_path="qaoa_graph.png"):
    layout = nx.random_layout(G, seed=10)
    colors = ["r", "g", "b", "y"]
    nx.draw(G, layout, node_color=colors)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    plt.savefig(save_path)
    plt.close()
    print(f"图已保存至 {save_path}")


# ---------------------------------------------------------------------------
# 步骤 3: 暴力求解法
# ---------------------------------------------------------------------------
def brute_force(num_nodes, w):
    L = num_nodes
    maximum = 2**L
    sol = np.inf
    for i in range(maximum):
        cur = bitfield(i, L)

        how_many_nonzero = np.count_nonzero(cur)
        if how_many_nonzero * 2 != L:  # not balanced
            continue

        cur_v = objective_value(np.array(cur), w)
        if cur_v < sol:
            sol = cur_v

    print(f"Objective value computed by the brute-force method is {sol}")
    return sol


# ---------------------------------------------------------------------------
# 步骤 5: 使用 QAOA 算法求解
# ---------------------------------------------------------------------------
def solve_with_qaoa(qubit_op, w, sampler):
    algorithm_globals.random_seed = 10598

    optimizer = COBYLA()
    qaoa = QAOA(sampler, optimizer, reps=2)

    result = qaoa.compute_minimum_eigenvalue(qubit_op)

    x = sample_most_likely(result.eigenstate)

    print(x)
    print(f"Objective value computed by QAOA is {objective_value(x, w)}")
    return x


# ---------------------------------------------------------------------------
# 步骤 6: 使用经典 NumPyMinimumEigensolver 作为参考
# ---------------------------------------------------------------------------
def solve_with_numpy(qubit_op, w):
    npme = NumPyMinimumEigensolver()
    result = npme.compute_minimum_eigenvalue(Operator(qubit_op))

    x = sample_most_likely(result.eigenstate.probabilities_dict())

    print(x)
    print(f"Objective value computed by the NumPyMinimumEigensolver is {objective_value(x, w)}")
    return x


# ---------------------------------------------------------------------------
# 步骤 7: 使用 SamplingVQE 作为替代方案
# ---------------------------------------------------------------------------
def solve_with_sampling_vqe(qubit_op, w, sampler):
    algorithm_globals.random_seed = 13345

    optimizer = COBYLA()
    ansatz = n_local(qubit_op.num_qubits, "ry", "cz", reps=2, entanglement="linear")
    sampling_vqe = SamplingVQE(sampler, ansatz, optimizer)

    result = sampling_vqe.compute_minimum_eigenvalue(qubit_op)

    x = sample_most_likely(result.eigenstate)

    print(x)
    print(f"Objective value computed by SamplingVQE is {objective_value(x, w)}")
    return x


# ---------------------------------------------------------------------------
# 步骤 8-11: 自定义转译器（PassManager）演示
# ---------------------------------------------------------------------------
def callback(**kwargs):
    if kwargs["count"] == 0:
        print("Callback function has been called!")


def solve_with_custom_transpiler(qubit_op, w, sampler):
    # 步骤 8: 定义自定义后端
    coupling_map = [(0, 1), (1, 2), (2, 3)]
    backend = GenericBackendV2(num_qubits=4, coupling_map=coupling_map, seed=54)

    # 步骤 9: 为后端定义 PassManager
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

    # 步骤 11: 将 PassManager 传给 QAOA 并运行
    optimizer = COBYLA()
    qaoa = QAOA(
        sampler,
        optimizer,
        reps=2,
        transpiler=pm,
        transpiler_options={"callback": callback},
    )
    result = qaoa.compute_minimum_eigenvalue(qubit_op)

    x = sample_most_likely(result.eigenstate)

    print(x)
    print(f"Objective value computed by QAOA is {objective_value(x, w)}")
    return x


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # 采样器（共享）
    sampler = StatevectorSampler(seed=42)

    print("=" * 60)
    print("步骤 1-2: 定义并可视化图")
    print("=" * 60)
    num_nodes, w, G = build_graph()
    draw_graph(G)

    print("\n" + "=" * 60)
    print("步骤 3: 暴力求解法")
    print("=" * 60)
    brute_force(num_nodes, w)

    print("\n" + "=" * 60)
    print("步骤 4: 将图分区问题转换为 Ising 哈密顿量")
    print("=" * 60)
    qubit_op, offset = get_operator(w)
    print(f"哈密顿量算符构造完成, offset = {offset}")

    print("\n" + "=" * 60)
    print("步骤 5: 使用 QAOA 算法求解")
    print("=" * 60)
    solve_with_qaoa(qubit_op, w, sampler)

    print("\n" + "=" * 60)
    print("步骤 6: 使用经典 NumPyMinimumEigensolver 作为参考")
    print("=" * 60)
    solve_with_numpy(qubit_op, w)

    print("\n" + "=" * 60)
    print("步骤 7: 使用 SamplingVQE 作为替代方案")
    print("=" * 60)
    solve_with_sampling_vqe(qubit_op, w, sampler)

    print("\n" + "=" * 60)
    print("步骤 8-11: 自定义转译器（PassManager）演示")
    print("=" * 60)
    solve_with_custom_transpiler(qubit_op, w, sampler)


if __name__ == "__main__":
    main()
