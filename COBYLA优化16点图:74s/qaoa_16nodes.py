"""
QAOA 求解 Max-Cut —— 16 节点 / 6 层 / COBYLA / 单次正式测试

配置:
    - 节点数: 16   (态空间 2^16 = 65536)
    - 层数 reps: 6 (共 12 个待优化参数)
    - 优化器: COBYLA
    - 初始化: 线性绝热 (gamma 递增, beta 递减)
    - 采样器: qiskit-aer SamplerV2 (C++ 后端, 快)
    - 图: 无权 Erdos-Renyi 随机图

依赖:
    qiskit, qiskit-aer, qiskit-algorithms, networkx, numpy, matplotlib

运行:
    python3 qaoa_16nodes.py
"""

import time
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler  # C++ 后端, 比 StatevectorSampler 快很多

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

# 屏蔽 scipy 稀疏矩阵效率警告（不影响结果，只是让输出干净）
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 配置项
# ---------------------------------------------------------------------------
CONFIG = {
    "num_nodes": 16,       # 节点数
    "edge_prob": 0.4,      # 随机图连边概率（Erdos-Renyi）
    "graph_seed": 7,       # 图生成随机种子（保证图可复现）
    "reps": 6,             # QAOA 层数 p
    "maxiter": 300,        # COBYLA 最大迭代次数
    "seed": 10598,         # 全局随机种子
    "sampler_seed": 42,
    "shots": 2048,         # aer 采样次数（越大越精确、越慢）
    "color_a": "#e74c3c",  # A 组（比特=1）红
    "color_b": "#3498db",  # B 组（比特=0）蓝
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def cut_value(x, w):
    """计算给定分组 x 的切割边数（无权图: 横跨两组的边数）。"""
    edges = np.triu(np.where(w != 0, 1, 0))   # 上三角，避免重复计边
    diff = x[:, None] != x[None, :]           # 两端是否不同组
    return int(np.sum(edges * diff))


def sample_most_likely(state_vector):
    """从概率分布里挑出概率最高的比特串，并修正 Qiskit 的反序。"""
    best = max(state_vector.items(), key=lambda kv: kv[1])[0]
    return np.array([int(b) for b in best[::-1]])


def _zz_pauli(num_nodes, i, j):
    """构造作用在第 i、j 位上的 Z_i·Z_j 泡利算符。"""
    z_p = np.zeros(num_nodes, dtype=bool)
    x_p = np.zeros(num_nodes, dtype=bool)
    z_p[i] = True
    z_p[j] = True
    return Pauli((z_p, x_p))


def build_maxcut_operator(weight_matrix):
    r"""把 Max-Cut 问题编码成 Ising 哈密顿量（无权版本）。

    每条边 (i, j) 贡献一项 (Z_i Z_j - 1) / 2:
        - 同组:  Z_i Z_j = +1 -> 贡献  0
        - 异组:  Z_i Z_j = -1 -> 贡献 -1
    能量 = -(切割边数)，切割数 = offset - 能量。

    注: 若将来改加权图，只需把下面的 0.5 改为 0.5 * weight_matrix[i, j]。
    """
    num_nodes = len(weight_matrix)
    pauli_list = []
    coeffs = []
    offset = 0.0

    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                pauli_list.append(_zz_pauli(num_nodes, i, j))
                coeffs.append(0.5)
                offset += 0.5

    op = SparsePauliOp(pauli_list, coeffs=np.array(coeffs))
    return op, offset


# ---------------------------------------------------------------------------
# 步骤 1: 生成 16 节点随机图
# ---------------------------------------------------------------------------
def build_graph():
    """生成一个 16 节点的 Erdos-Renyi 随机图（无权）。"""
    n = CONFIG["num_nodes"]
    G = nx.erdos_renyi_graph(n=n, p=CONFIG["edge_prob"], seed=CONFIG["graph_seed"])
    w = nx.to_numpy_array(G, dtype=float)
    return w, G


# ---------------------------------------------------------------------------
# 线性绝热初始化
# ---------------------------------------------------------------------------
def linear_ramp_init(reps, gamma_max=np.pi, beta_max=np.pi):
    r"""线性绝热初始化: gamma 线性递增, beta 线性递减。

    Qiskit QAOA 参数顺序为每层交替 [gamma_1, beta_1, gamma_2, beta_2, ...]。
    """
    point = []
    for k in range(1, reps + 1):
        s = k / (reps + 1)
        point.append(gamma_max * s)        # gamma（递增）
        point.append(beta_max * (1 - s))   # beta（递减）
    return np.array(point)


# ---------------------------------------------------------------------------
# 步骤 2: 用 QAOA + COBYLA 求解（单次）
# ---------------------------------------------------------------------------
def solve_with_qaoa(qubit_op, offset, w):
    """运行一次 QAOA(COBYLA)，返回 (最优分组 x, 切割数曲线 history, 末态分布)。"""
    reps = CONFIG["reps"]
    algorithm_globals.random_seed = CONFIG["seed"]

    history = []
    print_every = 5           # 每多少次评估打印一次进度
    t_start = [None]          # 用列表包装, 便于在闭包里赋值

    def store_intermediate(eval_count, params, mean, metadata):
        cut = offset - mean   # 真实切割数（= offset - 能量）
        history.append(cut)
        n = len(history)
        if n % print_every == 0:
            best = max(history)
            elapsed = time.time() - t_start[0]
            print(
                f"  [第 {n:4d} 次] 当前切割数 = {cut:6.3f} | "
                f"目前最优 = {best:6.3f} | 已用时 {elapsed:6.1f}s",
                flush=True,   # 强制立即刷新输出, 避免被缓冲
            )

    initial_point = linear_ramp_init(reps)
    print(f"层数 reps = {reps}, 待优化参数个数 = {len(initial_point)}")
    print(f"绝热初始化 gamma (递增): {np.round(initial_point[0::2], 3)}")
    print(f"绝热初始化 beta  (递减): {np.round(initial_point[1::2], 3)}")

    sampler = AerSampler(
        default_shots=CONFIG["shots"],
        seed=CONFIG["sampler_seed"],
    )
    optimizer = COBYLA(maxiter=CONFIG["maxiter"])
    # aer 的 C++ 后端不认识高层 QAOA gate, 需先转译分解成基础门
    pm = generate_preset_pass_manager(
        optimization_level=1, basis_gates=["rz", "rx", "ry", "h", "cx"]
    )
    qaoa = QAOA(
        sampler,
        optimizer,
        reps=reps,
        initial_point=initial_point,
        callback=store_intermediate,
        transpiler=pm,
    )

    print(f"\n开始 QAOA 优化（COBYLA, 最多 {CONFIG['maxiter']} 次迭代, 每 {print_every} 次打印）...")
    t0 = time.time()
    t_start[0] = t0
    result = qaoa.compute_minimum_eigenvalue(qubit_op)
    elapsed = time.time() - t0

    x = sample_most_likely(result.eigenstate)
    n_a = int(np.count_nonzero(x))
    print(f"QAOA 最优分组: {x}")
    print(f"切割边数 (越大越好): {cut_value(x, w)}")
    print(f"两组节点数: A={n_a}, B={len(x) - n_a}")
    print(f"优化迭代次数: {len(history)}")
    print(f">>> 单次运行耗时: {elapsed:.2f} 秒 <<<")
    return x, history, result.eigenstate


# ---------------------------------------------------------------------------
# 可视化 (a): 分组结果图
# ---------------------------------------------------------------------------
def draw_partition(G, x, save_path="qaoa_16nodes_partition.png"):
    pos = nx.spring_layout(G, seed=10)
    node_colors = [CONFIG["color_a"] if b == 1 else CONFIG["color_b"] for b in x]

    cut_edges = [(u, v) for u, v in G.edges() if x[u] != x[v]]
    normal_edges = [(u, v) for u, v in G.edges() if x[u] == x[v]]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#bdc3c7", width=1.2)
    nx.draw_networkx_edges(
        G, pos, edgelist=cut_edges, edge_color=CONFIG["color_a"], width=2.2, style="dashed"
    )
    plt.title(
        f"QAOA Max-Cut (16 nodes) | Cut edges (dashed) = {len(cut_edges)}",
        fontsize=12,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"分组结果图已保存: {save_path}")


# ---------------------------------------------------------------------------
# 可视化 (b): 优化收敛曲线
# ---------------------------------------------------------------------------
def draw_convergence(history, save_path="qaoa_16nodes_convergence.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, color="#2c3e50", linewidth=1.6)
    plt.scatter(range(len(history)), history, color=CONFIG["color_a"], s=10)
    plt.xlabel("Optimization iteration")
    plt.ylabel("Cut value (higher is better)")
    plt.title("QAOA Optimization Convergence (16 nodes, 6 layers, COBYLA)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"优化收敛曲线已保存: {save_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("步骤 1: 生成 16 节点随机图")
    print("=" * 60)
    w, G = build_graph()
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    print("\n" + "=" * 60)
    print("步骤 2: 构造 Max-Cut 哈密顿量")
    print("=" * 60)
    qubit_op, offset = build_maxcut_operator(w)
    print(f"量子比特数: {qubit_op.num_qubits}, 哈密顿量项数: {len(qubit_op)}, offset = {offset}")

    print("\n" + "=" * 60)
    print("步骤 3: 使用 QAOA + COBYLA 求解（单次）")
    print("=" * 60)
    x, history, _ = solve_with_qaoa(qubit_op, offset, w)

    print("\n" + "=" * 60)
    print("步骤 4: 可视化")
    print("=" * 60)
    draw_partition(G, x)
    draw_convergence(history)
    plt.show()


if __name__ == "__main__":
    main()
