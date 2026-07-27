"""
QAOA 加权 Max-Cut（18 节点难图）—— SPSA + Adam 优化器

优化方法:
    SPSA 采样估计梯度 + Adam 更新参数
    - SPSA: 用随机扰动 ±cΔ 做两次测量估计梯度, 每步仅 2 次能量评估, 抗噪声
    - Adam: 一阶/二阶动量, 自适应学习率
    这是 NISQ 真实量子硬件上运行 QAOA 的推荐搭配之一。

配置:
    - 节点数: 18, 层数 reps: 6 (12 参数)
    - 加权 Erdos-Renyi 社区结构难图
    - 采样器: qiskit-aer SamplerV2 (C++ 后端)

依赖: qiskit, qiskit-aer, qiskit-algorithms, networkx, numpy, matplotlib
运行: python3 qaoa_spsa_adam.py
"""

import time
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
    "num_nodes": 18,
    "n_clusters": 3,
    "p_in": 0.75,
    "p_out": 0.08,
    "w_low": 0.5,
    "w_high": 5.0,
    "graph_seed": 20,
    "reps": 6,
    "seed": 10598,
    "sampler_seed": 42,
    "shots": 8192,           # 提高采样次数, 降低期望估计的统计噪声
    # SPSA + Adam（正规版: c/a 随迭代衰减 + 梯度平均 + 多起点重启）
    "spsa_maxiter": 120,     # 每次重启的迭代步数
    "spsa_c0": 0.4,          # SPSA 初始扰动幅度 (调大, 让早期梯度信号盖过采样噪声)
    "spsa_gamma": 0.101,     # c 衰减指数: c_t = c0 / t^gamma
    "spsa_a0": 0.2,          # Adam 初始学习率
    "spsa_alpha": 0.602,     # a 衰减指数: a_t = a0 / (t+A)^alpha
    "spsa_A": 10.0,          # 稳定常数 A (防止早期步长过大)
    "spsa_grad_avg": 4,      # 梯度平均次数 k: 每步估 k 次梯度取平均, 降噪 √k 倍
    "adam_b1": 0.9,          # Adam 一阶动量系数
    "adam_b2": 0.999,        # Adam 二阶动量系数
    "adam_eps": 1e-8,
    # 多起点重启
    "n_restarts": 4,         # 随机重启次数 (含 1 次绝热初始化 + 若干随机起点), 取最优
    "restart_jitter": 0.5,   # 随机起点在绝热初始点上的扰动幅度
    # 绘图
    "color_a": "#e74c3c",
    "color_b": "#3498db",
}


# ---------------------------------------------------------------------------
# 优化器: SPSA 估梯度 + Adam 更新（正规版, 含 c/a 衰减）
# ---------------------------------------------------------------------------
class SPSAAdamOptimizer(Optimizer):
    r"""正规版 SPSA 采样估计梯度 + Adam 更新参数（含 c/a 衰减）。

    每次迭代 t:
        1. 衰减系数 (Spall 经典公式):
               c_t = c0 / t^gamma           (扰动幅度, 逐步减小)
               a_t = a0 / (t + A)^alpha      (学习率, 逐步减小)
        2. 随机扰动 Δ (各分量 ±1)
        3. 梯度估计 g = [f(θ+c_t Δ) - f(θ-c_t Δ)] / (2 c_t) * Δ   (仅 2 次评估)
        4. Adam 更新 θ (学习率用 a_t)

    早期 c 大 -> 梯度信号盖过采样噪声; 后期 c、a 减小 -> 精细稳定收敛。
    每步仅 2 次能量评估, 抗噪声, 适合真实量子硬件。
    """

    def __init__(self, maxiter=150, c0=0.4, gamma=0.101, a0=0.2, alpha=0.602,
                 A=10.0, grad_avg=1, b1=0.9, b2=0.999, eps=1e-8, seed=None):
        super().__init__()
        self._maxiter = maxiter
        self._c0 = c0
        self._gamma = gamma
        self._a0 = a0
        self._alpha = alpha
        self._A = A
        self._grad_avg = grad_avg   # 每步梯度平均次数 k
        self._b1 = b1
        self._b2 = b2
        self._eps = eps
        self._rng = np.random.default_rng(seed)

    def get_support_level(self):
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def minimize(self, fun, x0, jac=None, bounds=None):
        theta = np.array(x0, dtype=float)
        dim = len(theta)
        m = np.zeros(dim)   # Adam 一阶动量
        v = np.zeros(dim)   # Adam 二阶动量
        counter = {"n": 0}
        last_f = [0.0]

        def f(p):
            counter["n"] += 1
            return fun(p)

        for t in range(1, self._maxiter + 1):
            # 1. 衰减系数
            c_t = self._c0 / (t ** self._gamma)
            a_t = self._a0 / ((t + self._A) ** self._alpha)

            # 2-3. SPSA 梯度估计 + 梯度平均 (估 k 次取平均, 降噪 √k 倍)
            g = np.zeros(dim)
            f_acc = 0.0
            for _ in range(self._grad_avg):
                delta = self._rng.choice([-1.0, 1.0], size=dim)   # 随机扰动 ±1
                f_plus = f(theta + c_t * delta)
                f_minus = f(theta - c_t * delta)
                f_acc += 0.5 * (f_plus + f_minus)
                g += (f_plus - f_minus) / (2.0 * c_t) * delta     # delta 分量 ±1, 1/delta = delta
            g /= self._grad_avg
            last_f[0] = f_acc / self._grad_avg

            # 4. Adam 更新 (学习率用衰减后的 a_t)
            m = self._b1 * m + (1 - self._b1) * g
            v = self._b2 * v + (1 - self._b2) * (g * g)
            m_hat = m / (1 - self._b1 ** t)
            v_hat = v / (1 - self._b2 ** t)
            theta = theta - a_t * m_hat / (np.sqrt(v_hat) + self._eps)

        # 返回最终参数(衰减后已精细收敛)
        result = OptimizerResult()
        result.x = theta
        result.fun = float(last_f[0])
        result.nfev = counter["n"]
        result.nit = self._maxiter
        return result


# ---------------------------------------------------------------------------
# 工具函数（加权）
# ---------------------------------------------------------------------------
def cut_value(x, w):
    triu = np.triu(w, k=1)
    diff = (x[:, None] != x[None, :]).astype(float)
    return float(np.sum(triu * diff))


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
    r"""加权 Max-Cut 哈密顿量: 每条边 (i,j) 贡献 w_ij * (Z_i Z_j - 1)/2。切割 = offset - 能量。"""
    num_nodes = len(weight_matrix)
    pauli_list, coeffs, offset = [], [], 0.0
    for i in range(num_nodes):
        for j in range(i):
            wij = weight_matrix[i, j]
            if wij != 0:
                pauli_list.append(_zz_pauli(num_nodes, i, j))
                coeffs.append(0.5 * wij)
                offset += 0.5 * wij
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs)), offset


def build_graph():
    n = CONFIG["num_nodes"]
    k = CONFIG["n_clusters"]
    rng = np.random.default_rng(CONFIG["graph_seed"])
    labels = np.array([i % k for i in range(n)])
    rng.shuffle(labels)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = CONFIG["p_in"] if labels[i] == labels[j] else CONFIG["p_out"]
            if rng.random() < p:
                wij = round(rng.uniform(CONFIG["w_low"], CONFIG["w_high"]), 2)
                w[i, j] = w[j, i] = wij
    return w, nx.from_numpy_array(w)


def linear_ramp_init(reps, gamma_max=np.pi, beta_max=np.pi):
    point = []
    for k in range(1, reps + 1):
        s = k / (reps + 1)
        point.append(gamma_max * s)
        point.append(beta_max * (1 - s))
    return np.array(point)


# ---------------------------------------------------------------------------
# 用 QAOA + SPSA+Adam 求解
# ---------------------------------------------------------------------------
def _make_start_points(reps, n_restarts, jitter, seed):
    """生成 n_restarts 个初始点: 第 1 个为绝热初始化, 其余在其基础上加随机扰动。"""
    base = linear_ramp_init(reps)
    rng = np.random.default_rng(seed)
    points = [base]
    for _ in range(n_restarts - 1):
        points.append(base + rng.uniform(-jitter, jitter, size=len(base)))
    return points


def solve_with_spsa_adam(qubit_op, offset, w):
    """SPSA+Adam (梯度平均) + 多起点重启, 返回最优 (x, 最优重启的 history, 所有重启 history, 末态)。"""
    reps = CONFIG["reps"]
    algorithm_globals.random_seed = CONFIG["seed"]
    n_restarts = CONFIG["n_restarts"]

    pm = generate_preset_pass_manager(
        optimization_level=1, basis_gates=["rz", "rx", "ry", "h", "cx"]
    )
    start_points = _make_start_points(
        reps, n_restarts, CONFIG["restart_jitter"], CONFIG["seed"]
    )

    print(f"层数 reps = {reps}, 参数个数 = {len(start_points[0])}")
    print(f"SPSA c0={CONFIG['spsa_c0']}(衰减), a0={CONFIG['spsa_a0']}(衰减), "
          f"梯度平均 k={CONFIG['spsa_grad_avg']}, 每次重启 {CONFIG['spsa_maxiter']} 步")
    print(f"多起点重启次数: {n_restarts} (第1个绝热初始化 + {n_restarts-1}个随机起点), 取最优")

    all_histories = []       # 每次重启的收敛历史
    best = {"cut": -np.inf, "x": None, "hist": None, "state": None, "idx": -1}
    t_global = time.time()

    for r in range(n_restarts):
        history = []
        t_start = [time.time()]

        def store_intermediate(eval_count, params, mean, metadata):
            history.append(offset - mean)

        optimizer = SPSAAdamOptimizer(
            maxiter=CONFIG["spsa_maxiter"],
            c0=CONFIG["spsa_c0"], gamma=CONFIG["spsa_gamma"],
            a0=CONFIG["spsa_a0"], alpha=CONFIG["spsa_alpha"], A=CONFIG["spsa_A"],
            grad_avg=CONFIG["spsa_grad_avg"],
            b1=CONFIG["adam_b1"], b2=CONFIG["adam_b2"], eps=CONFIG["adam_eps"],
            seed=CONFIG["seed"] + r,   # 每次重启不同随机种子
        )
        qaoa = QAOA(
            sampler=AerSampler(default_shots=CONFIG["shots"], seed=CONFIG["sampler_seed"]),
            optimizer=optimizer,
            reps=reps,
            initial_point=start_points[r],
            callback=store_intermediate,
            transpiler=pm,
        )

        print(f"\n--- 重启 {r + 1}/{n_restarts} ---", flush=True)
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        x = sample_most_likely(result.eigenstate)
        cv = cut_value(x, w)
        all_histories.append(history)
        print(f"    重启 {r + 1} 结果: 加权切割 = {cv:.3f} | "
              f"评估 {len(history)} 次 | 累计 {time.time() - t_global:.1f}s", flush=True)

        if cv > best["cut"]:
            best.update({"cut": cv, "x": x, "hist": history,
                         "state": result.eigenstate, "idx": r})

    elapsed = time.time() - t_global
    x = best["x"]
    n_a = int(np.count_nonzero(x))
    print(f"\n>>> 最优来自重启 {best['idx'] + 1}/{n_restarts}")
    print(f"QAOA 最优分组: {x}")
    print(f"加权切割值 (越大越好): {best['cut']:.3f}")
    print(f"两组节点数: A={n_a}, B={len(x) - n_a}")
    print(f">>> 总耗时: {elapsed:.2f} 秒 <<<")
    return x, best["hist"], all_histories, best["state"]


# ---------------------------------------------------------------------------
# 可视化 (a): 分组结果图（边宽反映权重）
# ---------------------------------------------------------------------------
def draw_partition(G, x, w, save_path="qaoa_spsa_adam_partition.png"):
    pos = nx.spring_layout(G, seed=10)
    node_colors = [CONFIG["color_a"] if b == 1 else CONFIG["color_b"] for b in x]
    cut_edges = [(u, v) for u, v in G.edges() if x[u] != x[v]]
    normal_edges = [(u, v) for u, v in G.edges() if x[u] == x[v]]
    wmax = w.max() if w.max() > 0 else 1.0

    def widths(edges):
        return [0.5 + 2.5 * w[u, v] / wmax for u, v in edges]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=550)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=9)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#bdc3c7",
                           width=widths(normal_edges))
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color=CONFIG["color_a"],
                           width=widths(cut_edges), style="dashed")
    plt.title(f"SPSA+Adam Weighted Max-Cut (18 nodes) | Cut weight = {cut_value(x, w):.2f}",
              fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"分组结果图已保存: {save_path}")


# ---------------------------------------------------------------------------
# 可视化 (b): 收敛曲线
# ---------------------------------------------------------------------------
def draw_convergence(all_histories, best_idx, save_path="qaoa_spsa_adam_convergence.png"):
    """画出所有重启的收敛曲线, 最优的那条高亮。"""
    plt.figure(figsize=(9, 5))
    for i, hist in enumerate(all_histories):
        if i == best_idx:
            plt.plot(range(len(hist)), hist, color="#e74c3c", linewidth=2.2,
                     label=f"Restart {i + 1} (best)", zorder=3)
        else:
            plt.plot(range(len(hist)), hist, linewidth=1.0, alpha=0.6,
                     label=f"Restart {i + 1}")
    plt.xlabel("Energy evaluation count")
    plt.ylabel("Cut value (expectation, higher is better)")
    plt.title("SPSA+Adam Multi-restart Convergence (18-node weighted)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"多起点收敛曲线已保存: {save_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("步骤 1: 生成 18 节点加权难图")
    print("=" * 60)
    w, G = build_graph()
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    print("\n" + "=" * 60)
    print("步骤 2: 构造加权 Max-Cut 哈密顿量")
    print("=" * 60)
    qubit_op, offset = build_maxcut_operator(w)
    print(f"量子比特数: {qubit_op.num_qubits}, 哈密顿量项数: {len(qubit_op)}, offset = {offset:.3f}")

    print("\n" + "=" * 60)
    print("步骤 3: QAOA + SPSA + Adam (梯度平均 + 多起点重启) 求解")
    print("=" * 60)
    x, best_hist, all_histories, _ = solve_with_spsa_adam(qubit_op, offset, w)
    # 找出最优重启的下标(其 history 即 best_hist)
    best_idx = next((i for i, h in enumerate(all_histories) if h is best_hist), 0)

    print("\n" + "=" * 60)
    print("步骤 4: 可视化")
    print("=" * 60)
    draw_partition(G, x, w)
    draw_convergence(all_histories, best_idx)
    plt.show()


if __name__ == "__main__":
    main()
