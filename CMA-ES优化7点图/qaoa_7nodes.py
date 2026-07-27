"""
QAOA 求解 Max-Cut（最大割）—— 7 节点版本

基于 Qiskit QAOA 教程改写:
    https://qiskit-community.github.io/qiskit-algorithms/tutorials/05_qaoa.html

与原教程的区别:
    1. 只保留 QAOA 方法。
    2. 节点数扩展为 7 个。
    3. 求解的是 Max-Cut（最大割）: 只追求"切割边数最多"，
       不要求两组节点数相等（已去掉平衡惩罚项）。
    4. 4 层线路 + 线性绝热初始化（gamma 递增, beta 递减）。
    5. 可视化: (a) 分组结果  (b) 优化收敛曲线  (c) 切割边数概率分布。

依赖:
    qiskit, qiskit-algorithms, networkx, numpy, matplotlib

运行:
    python3 qaoa_7nodes.py
"""

import time

import cma
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.primitives import StatevectorSampler

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, OptimizerSupportLevel
from qiskit_algorithms.utils import algorithm_globals


# ---------------------------------------------------------------------------
# 配置项（集中管理，方便调整）
# ---------------------------------------------------------------------------
CONFIG = {
    "reps": 5,            # QAOA 层数 p
    "maxiter": 250,       # 优化器最大迭代（代数）上限
    "seed": 10598,        # 随机种子（保证可复现）
    "sampler_seed": 42,
    "cma_sigma0": 0.5,    # CMA-ES 初始步长（探索范围, 越大探索越广）
    "color_a": "#e74c3c",  # A 组（比特=1）红
    "color_b": "#3498db",  # B 组（比特=0）蓝
}


# ---------------------------------------------------------------------------
# CMA-ES 优化器（封装 cma 库为 Qiskit 兼容的 Optimizer）
# ---------------------------------------------------------------------------
class CMAESOptimizer(Optimizer):
    """把 cma 库封装成 qiskit-algorithms 的 Optimizer 接口。

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 是一种
    免梯度的进化策略全局优化器: 每一代从一个多元正态分布中采样一批候选解,
    根据它们的好坏更新分布的均值和协方差矩阵, 逐步收缩到最优区域。
    相比 COBYLA, 它更擅长跳出局部最优、探索复杂参数空间。
    """

    def __init__(self, maxiter=250, sigma0=0.5, seed=None):
        super().__init__()
        self._maxiter = maxiter
        self._sigma0 = sigma0
        self._seed = seed

    def get_support_level(self):
        # CMA-ES 免梯度, 支持边界, 不需要梯度
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }

    def minimize(self, fun, x0, jac=None, bounds=None):
        opts = {
            "maxiter": self._maxiter,
            "verbose": -9,  # 关闭 cma 自身的冗长打印
        }
        if self._seed is not None:
            opts["seed"] = self._seed
        if bounds is not None:
            lower = [b[0] if b[0] is not None else None for b in bounds]
            upper = [b[1] if b[1] is not None else None for b in bounds]
            opts["bounds"] = [lower, upper]

        es = cma.CMAEvolutionStrategy(list(x0), self._sigma0, opts)
        es.optimize(fun)  # fun: 参数向量 -> 标量能量

        result = OptimizerResult()
        result.x = np.array(es.result.xbest)
        result.fun = float(es.result.fbest)
        result.nfev = int(es.result.evaluations)
        result.nit = int(es.result.iterations)
        return result


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def cut_value(x, w):
    """计算给定分组 x 的切割边数（横跨两组的边数）。

    Args:
        x: 0/1 分组数组。
        w: 邻接矩阵。

    Returns:
        跨组边数（int）。
    """
    edges = np.triu(np.where(w != 0, 1, 0))          # 上三角，避免重复计边
    diff = x[:, None] != x[None, :]                  # 两端是否不同组
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
    r"""把 Max-Cut 问题编码成 Ising 哈密顿量（无平衡约束）。

    每条边 (i, j) 贡献一项 (Z_i Z_j - 1) / 2:
        - 两端同组:  Z_i Z_j = +1 -> 贡献  0
        - 两端异组:  Z_i Z_j = -1 -> 贡献 -1
    因此能量 = -(切割边数)。QAOA 找最小能量 <=> 找最大切割。

    Returns:
        (哈密顿量 SparsePauliOp, 常数偏移 offset)
        满足关系:  切割边数 = offset - 能量
    """
    num_nodes = len(weight_matrix)
    pauli_list = []
    coeffs = []
    offset = 0.0

    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                pauli_list.append(_zz_pauli(num_nodes, i, j))
                coeffs.append(0.5)   # (Z_i Z_j - 1)/2 中的 0.5·Z_iZ_j 部分
                offset += 0.5        # 常数 -0.5，累加到 offset（切割 = offset - 能量）

    op = SparsePauliOp(pauli_list, coeffs=np.array(coeffs))
    return op, offset


# ---------------------------------------------------------------------------
# 步骤 1: 定义 7 节点图
# ---------------------------------------------------------------------------
def build_graph():
    """定义一个 7 节点无向图（邻接矩阵）。1 表示两节点间有边。"""
    w = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0],
        ],
        dtype=float,
    )
    return w, nx.from_numpy_array(w)


# ---------------------------------------------------------------------------
# 线性绝热初始化
# ---------------------------------------------------------------------------
def linear_ramp_init(reps, gamma_max=np.pi, beta_max=np.pi):
    r"""线性绝热（linear ramp）初始化: gamma 线性递增, beta 线性递减。

    第 k 层 (k = 1..reps):
        s = k / (reps + 1)
        gamma_k = gamma_max * s        # 逐步打开问题哈密顿量
        beta_k  = beta_max  * (1 - s)  # 逐步关闭混合哈密顿量

    Qiskit QAOA 参数顺序为每层交替 [gamma_1, beta_1, gamma_2, beta_2, ...]。
    """
    point = []
    for k in range(1, reps + 1):
        s = k / (reps + 1)
        point.append(gamma_max * s)        # gamma（递增）
        point.append(beta_max * (1 - s))   # beta（递减）
    return np.array(point)


# ---------------------------------------------------------------------------
# 步骤 2: 用 QAOA 求解
# ---------------------------------------------------------------------------
def solve_with_qaoa(qubit_op, offset, w):
    """运行 QAOA，返回 (最优分组 x, 真实切割数曲线 history, 末态分布)。"""
    reps = CONFIG["reps"]
    algorithm_globals.random_seed = CONFIG["seed"]

    # 回调记录每次评估的"真实切割数"(= offset - 能量)，用于画收敛曲线
    history = []

    def store_intermediate(eval_count, params, mean, metadata):
        history.append(offset - mean)

    initial_point = linear_ramp_init(reps)
    print(f"层数 reps = {reps}, 待优化参数个数 = {len(initial_point)}")
    print(f"绝热初始化 gamma (递增): {np.round(initial_point[0::2], 3)}")
    print(f"绝热初始化 beta  (递减): {np.round(initial_point[1::2], 3)}")

    sampler = StatevectorSampler(seed=CONFIG["sampler_seed"])
    optimizer = CMAESOptimizer(
        maxiter=CONFIG["maxiter"],
        sigma0=CONFIG["cma_sigma0"],
        seed=CONFIG["seed"],
    )
    qaoa = QAOA(
        sampler,
        optimizer,
        reps=reps,
        initial_point=initial_point,
        callback=store_intermediate,
    )

    t0 = time.time()
    result = qaoa.compute_minimum_eigenvalue(qubit_op)
    elapsed = time.time() - t0

    x = sample_most_likely(result.eigenstate)
    n_a = int(np.count_nonzero(x))
    print(f"QAOA 最优分组: {x}")
    print(f"切割边数 (越大越好): {cut_value(x, w)}")
    print(f"两组节点数: A={n_a}, B={len(x) - n_a} (Max-Cut 不要求相等)")
    print(f"优化迭代次数: {len(history)}, 耗时: {elapsed:.2f}s")
    return x, history, result.eigenstate


# ---------------------------------------------------------------------------
# 可视化 (a): 分组结果图
# ---------------------------------------------------------------------------
def draw_partition(G, x, save_path="qaoa_7nodes_partition.png"):
    """按 QAOA 结果给节点上色，被切断的跨组边用红色虚线标出。"""
    pos = nx.spring_layout(G, seed=10)
    node_colors = [CONFIG["color_a"] if b == 1 else CONFIG["color_b"] for b in x]

    cut_edges = [(u, v) for u, v in G.edges() if x[u] != x[v]]
    normal_edges = [(u, v) for u, v in G.edges() if x[u] == x[v]]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#bdc3c7", width=1.5)
    nx.draw_networkx_edges(
        G, pos, edgelist=cut_edges, edge_color=CONFIG["color_a"], width=2.5, style="dashed"
    )
    plt.title(
        f"QAOA Max-Cut (7 nodes) | Cut edges (dashed) = {len(cut_edges)}",
        fontsize=12,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"分组结果图已保存: {save_path}")


# ---------------------------------------------------------------------------
# 可视化 (b): 优化收敛曲线（显示真实切割数）
# ---------------------------------------------------------------------------
def draw_convergence(history, save_path="qaoa_7nodes_convergence.png"):
    """绘制 QAOA 优化过程中"切割边数"随迭代的变化。"""
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, color="#2c3e50", linewidth=1.8)
    plt.scatter(range(len(history)), history, color=CONFIG["color_a"], s=12)
    plt.xlabel("Optimization iteration")
    plt.ylabel("Cut value (higher is better)")
    plt.title("QAOA Optimization Convergence (Max-Cut)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"优化收敛曲线已保存: {save_path}")


# ---------------------------------------------------------------------------
# 可视化 (c): 切割边数的概率分布
# ---------------------------------------------------------------------------
def draw_cut_distribution(eigenstate, w, save_path="qaoa_7nodes_cut_distribution.png"):
    """统计 QAOA 末态中每种"切割边数"出现的概率并画直方图。"""
    num_nodes = len(w)
    cut_prob = {}
    for bitstring, prob in eigenstate.items():
        x = np.array([int(b) for b in bitstring[::-1]])
        if len(x) < num_nodes:
            x = np.pad(x, (0, num_nodes - len(x)))
        c = cut_value(x, w)
        cut_prob[c] = cut_prob.get(c, 0.0) + float(prob)

    cuts = sorted(cut_prob.keys())
    probs = [cut_prob[c] for c in cuts]

    print("\n切割边数的概率分布:")
    for c, p in zip(cuts, probs):
        print(f"  切割边数 = {c:2d}  ->  概率 = {p:.4f}")

    # Max-Cut 目标是"切得越多越好"，故最大切割值用红色高亮
    best_cut = max(cuts)
    colors = [CONFIG["color_a"] if c == best_cut else CONFIG["color_b"] for c in cuts]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(c) for c in cuts], probs, color=colors, edgecolor="black")
    for bar, p in zip(bars, probs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xlabel("Number of cut edges")
    plt.ylabel("Probability")
    plt.title("QAOA: Probability Distribution over Cut Sizes (Max-Cut, 7 nodes)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"切割边数概率分布图已保存: {save_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("步骤 1: 定义 7 节点图")
    print("=" * 60)
    w, G = build_graph()
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    print("\n" + "=" * 60)
    print("步骤 2: 构造 Max-Cut 哈密顿量（无平衡约束）")
    print("=" * 60)
    qubit_op, offset = build_maxcut_operator(w)
    print(f"量子比特数: {qubit_op.num_qubits}, offset = {offset} (切割数 = offset - 能量)")

    print("\n" + "=" * 60)
    print("步骤 3: 使用 QAOA 求解")
    print("=" * 60)
    x, history, eigenstate = solve_with_qaoa(qubit_op, offset, w)

    print("\n" + "=" * 60)
    print("步骤 4: 可视化")
    print("=" * 60)
    draw_partition(G, x)
    draw_convergence(history)
    draw_cut_distribution(eigenstate, w)


if __name__ == "__main__":
    main()
