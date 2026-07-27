"""
QAOA 求解 Max-Cut —— 混合优化器 (CMA-ES 全局探索 -> COBYLA 局部精修)

策略 (方案 A: 全局 -> 局部, 串行两阶段):
    阶段1 (CMA-ES): 从绝热初始点出发, 用进化策略做全局探索,
                    跳出局部最优, 找到一个"好的区域/方向"。
    阶段2 (COBYLA): 从 CMA-ES 的最优点出发, 快速局部下降精修到谷底。

    两阶段共享同一个能量函数, 收敛历史连续拼接, 便于画出
    "前段 CMA-ES 抖动探索 + 后段 COBYLA 平滑精修" 的完整曲线。

配置:
    - 节点数: 16, 层数 reps: 6 (12 参数)
    - 采样器: qiskit-aer SamplerV2 (C++ 后端, 快)
    - 图: 无权 Erdos-Renyi 随机图

依赖:
    qiskit, qiskit-aer, qiskit-algorithms, cma, networkx, numpy, matplotlib

运行:
    python3 qaoa_hybrid.py
"""

import time
import warnings

import cma
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, OptimizerSupportLevel
from qiskit_algorithms.utils import algorithm_globals

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 配置项
# ---------------------------------------------------------------------------
CONFIG = {
    "num_nodes": 16,
    "edge_prob": 0.4,
    "graph_seed": 7,
    "reps": 6,
    "seed": 10598,
    "sampler_seed": 42,
    "shots": 2048,
    # --- 混合优化器参数 ---
    "cma_maxiter": 25,     # CMA-ES 阶段: 最多进化代数 (全局探索)
    "cma_sigma0": 0.4,     # CMA-ES 初始步长 (探索范围)
    "cobyla_maxiter": 120,  # COBYLA 阶段: 最多迭代次数 (局部精修)
    # --- 绘图 ---
    "color_a": "#e74c3c",
    "color_b": "#3498db",
}


# ---------------------------------------------------------------------------
# 混合优化器: CMA-ES (全局) -> COBYLA (局部)
# ---------------------------------------------------------------------------
class HybridOptimizer(Optimizer):
    """先用 CMA-ES 全局探索, 再用 COBYLA 局部精修的两阶段混合优化器。

    boundary 记录两阶段的分界评估序号, 供画图时区分前后段。
    """

    def __init__(self, cma_maxiter=25, cma_sigma0=0.4, cobyla_maxiter=120, seed=None):
        super().__init__()
        self._cma_maxiter = cma_maxiter
        self._cma_sigma0 = cma_sigma0
        self._cobyla_maxiter = cobyla_maxiter
        self._seed = seed
        self.boundary = None   # 阶段一结束时的累计评估次数

    def get_support_level(self):
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def minimize(self, fun, x0, jac=None, bounds=None):
        eval_counter = {"n": 0}

        # 包一层计数器, 让两阶段共享同一个评估计数
        def wrapped(params):
            eval_counter["n"] += 1
            return fun(params)

        # ---- 阶段 1: CMA-ES 全局探索 ----
        print("  >> 阶段1 [CMA-ES] 全局探索...", flush=True)
        opts = {"maxiter": self._cma_maxiter, "verbose": -9}
        if self._seed is not None:
            opts["seed"] = self._seed
        es = cma.CMAEvolutionStrategy(list(x0), self._cma_sigma0, opts)
        es.optimize(wrapped)
        x_cma = np.array(es.result.xbest)
        self.boundary = eval_counter["n"]
        print(
            f"  >> 阶段1 结束: CMA-ES 评估 {self.boundary} 次, "
            f"最优能量 = {es.result.fbest:.4f}",
            flush=True,
        )

        # ---- 阶段 2: COBYLA 局部精修 (从 CMA-ES 最优点出发) ----
        print("  >> 阶段2 [COBYLA] 局部精修...", flush=True)
        res = scipy_minimize(
            wrapped,
            x_cma,
            method="COBYLA",
            options={"maxiter": self._cobyla_maxiter},
        )
        print(
            f"  >> 阶段2 结束: 总评估 {eval_counter['n']} 次, "
            f"最终能量 = {res.fun:.4f}",
            flush=True,
        )

        result = OptimizerResult()
        result.x = np.array(res.x)
        result.fun = float(res.fun)
        result.nfev = eval_counter["n"]
        result.nit = eval_counter["n"]
        return result


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def cut_value(x, w):
    edges = np.triu(np.where(w != 0, 1, 0))
    diff = x[:, None] != x[None, :]
    return int(np.sum(edges * diff))


def sample_most_likely(state_vector):
    best = max(state_vector.items(), key=lambda kv: kv[1])[0]
    return np.array([int(b) for b in best[::-1]])


def _zz_pauli(num_nodes, i, j):
    z_p = np.zeros(num_nodes, dtype=bool)
    x_p = np.zeros(num_nodes, dtype=bool)
    z_p[i] = True
    z_p[j] = True
    return Pauli((z_p, x_p))


def build_maxcut_operator(weight_matrix):
    r"""Max-Cut 哈密顿量 (无权): 每条边贡献 (Z_i Z_j - 1)/2。切割数 = offset - 能量。"""
    num_nodes = len(weight_matrix)
    pauli_list, coeffs, offset = [], [], 0.0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                pauli_list.append(_zz_pauli(num_nodes, i, j))
                coeffs.append(0.5)
                offset += 0.5
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs)), offset


def build_graph():
    n = CONFIG["num_nodes"]
    G = nx.erdos_renyi_graph(n=n, p=CONFIG["edge_prob"], seed=CONFIG["graph_seed"])
    return nx.to_numpy_array(G, dtype=float), G


def linear_ramp_init(reps, gamma_max=np.pi, beta_max=np.pi):
    """线性绝热初始化: gamma 递增, beta 递减。"""
    point = []
    for k in range(1, reps + 1):
        s = k / (reps + 1)
        point.append(gamma_max * s)
        point.append(beta_max * (1 - s))
    return np.array(point)


# ---------------------------------------------------------------------------
# 用 QAOA + 混合优化器求解
# ---------------------------------------------------------------------------
def solve_with_hybrid(qubit_op, offset, w):
    """运行 QAOA(混合优化器)，返回 (x, 切割数曲线 history, 阶段分界 boundary, 末态)。"""
    reps = CONFIG["reps"]
    algorithm_globals.random_seed = CONFIG["seed"]

    history = []
    t_start = [None]

    def store_intermediate(eval_count, params, mean, metadata):
        history.append(offset - mean)
        n = len(history)
        if n % 10 == 0:
            elapsed = time.time() - t_start[0]
            print(
                f"     [评估 {n:4d}] 当前切割 = {offset - mean:6.3f} | "
                f"最优 = {max(history):6.3f} | {elapsed:6.1f}s",
                flush=True,
            )

    initial_point = linear_ramp_init(reps)
    print(f"层数 reps = {reps}, 参数个数 = {len(initial_point)}")

    sampler = AerSampler(default_shots=CONFIG["shots"], seed=CONFIG["sampler_seed"])
    optimizer = HybridOptimizer(
        cma_maxiter=CONFIG["cma_maxiter"],
        cma_sigma0=CONFIG["cma_sigma0"],
        cobyla_maxiter=CONFIG["cobyla_maxiter"],
        seed=CONFIG["seed"],
    )
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

    print("\n开始 QAOA 优化（混合: CMA-ES -> COBYLA）...")
    t0 = time.time()
    t_start[0] = t0
    result = qaoa.compute_minimum_eigenvalue(qubit_op)
    elapsed = time.time() - t0

    x = sample_most_likely(result.eigenstate)
    n_a = int(np.count_nonzero(x))
    print(f"\nQAOA 最优分组: {x}")
    print(f"切割边数 (越大越好): {cut_value(x, w)}")
    print(f"两组节点数: A={n_a}, B={len(x) - n_a}")
    print(f"总评估次数: {len(history)}, CMA-ES/COBYLA 分界: {optimizer.boundary}")
    print(f">>> 总耗时: {elapsed:.2f} 秒 <<<")
    return x, history, optimizer.boundary, result.eigenstate


# ---------------------------------------------------------------------------
# 可视化 (a): 分组结果图
# ---------------------------------------------------------------------------
def draw_partition(G, x, save_path="qaoa_hybrid_partition.png"):
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
    plt.title(f"QAOA Max-Cut (Hybrid, 16 nodes) | Cut edges = {len(cut_edges)}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"分组结果图已保存: {save_path}")


# ---------------------------------------------------------------------------
# 可视化 (b): 两阶段收敛曲线 (CMA-ES 段 + COBYLA 段)
# ---------------------------------------------------------------------------
def draw_convergence(history, boundary, save_path="qaoa_hybrid_convergence.png"):
    plt.figure(figsize=(9, 5))
    idx = range(len(history))
    plt.plot(idx, history, color="#2c3e50", linewidth=1.5, zorder=1)

    # 用竖线和背景色区分两个阶段
    if boundary is not None and 0 < boundary < len(history):
        plt.axvspan(0, boundary, color="#f39c12", alpha=0.12, label="Stage 1: CMA-ES (global)")
        plt.axvspan(
            boundary, len(history) - 1, color="#2ecc71", alpha=0.12,
            label="Stage 2: COBYLA (local)",
        )
        plt.axvline(boundary, color="gray", linestyle="--", linewidth=1)

    plt.scatter(idx, history, color=CONFIG["color_a"], s=10, zorder=2)
    plt.xlabel("Energy evaluation count")
    plt.ylabel("Cut value (higher is better)")
    plt.title("Hybrid QAOA Convergence: CMA-ES (global) -> COBYLA (local)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"两阶段收敛曲线已保存: {save_path}")


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
    print("步骤 3: QAOA + 混合优化器 (CMA-ES -> COBYLA)")
    print("=" * 60)
    x, history, boundary, _ = solve_with_hybrid(qubit_op, offset, w)

    print("\n" + "=" * 60)
    print("步骤 4: 可视化")
    print("=" * 60)
    draw_partition(G, x)
    draw_convergence(history, boundary)
    plt.show()


if __name__ == "__main__":
    main()
