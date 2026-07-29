"""
随机 10 节点无权图的 QAOA MaxCut 层数扫描

流程：
1. 固定随机种子生成连通的 10 节点无权图。
2. 穷举 2^10 个划分，得到 exact MaxCut。
3. 用 Adam 分别优化 p=1,2,3,4,5 的 QAOA 电路。
4. 记录每个 p 的运行时间、期望割值、采样最优割值和近似比。
5. 生成 p_vs_time.png 与 p_vs_approximation_ratio.png。
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
import tensorcircuit as tc

# ============================== 配置 ========================================
N_NODES = 10
EDGE_PROBABILITY = 0.35
GRAPH_SEED = 42
QAOA_SEED = 2026
P_VALUES = range(1, 6)
ADAM_STEPS = 250
LEARNING_RATE = 0.05
N_SHOTS = 4096

K = tc.set_backend("tensorflow")
tc.set_dtype("complex64")
np.random.seed(QAOA_SEED)
tf.random.set_seed(QAOA_SEED)


# ============================== 随机图 ======================================
def generate_connected_graph(n, edge_probability, seed):
    """生成可复现的随机连通无权图。"""
    for current_seed in range(seed, seed + 10_000):
        graph = nx.gnp_random_graph(n, edge_probability, seed=current_seed)
        if nx.is_connected(graph):
            return graph, current_seed
    raise RuntimeError("无法生成连通随机图")


GRAPH, USED_GRAPH_SEED = generate_connected_graph(
    N_NODES, EDGE_PROBABILITY, GRAPH_SEED
)
EDGES = sorted(tuple(sorted(edge)) for edge in GRAPH.edges())
N_EDGES = len(EDGES)
N_STATES = 1 << N_NODES


# ======================= 代价表与 exact MaxCut ==============================
def build_cut_table(n, edges):
    """计算每个计算基态对应的割边数。

    TensorCircuit 状态向量中 qubit 0 是最高有效位，因此顶点 u 对应
    整数索引的第 n-1-u 位。
    """
    states = np.arange(1 << n, dtype=np.int32)
    cut_values = np.zeros(1 << n, dtype=np.float32)
    for u, v in edges:
        bit_u = (states >> (n - 1 - u)) & 1
        bit_v = (states >> (n - 1 - v)) & 1
        cut_values += bit_u ^ bit_v
    return cut_values


CUT_VALUES = build_cut_table(N_NODES, EDGES)
CUT_VALUES_TF = tf.constant(CUT_VALUES, dtype=tf.float32)
EXACT_MAXCUT = int(np.max(CUT_VALUES))
EXACT_INDEX = int(np.argmax(CUT_VALUES))
EXACT_BITS = format(EXACT_INDEX, f"0{N_NODES}b")


# ============================== QAOA ========================================
def build_qaoa_circuit(params, p):
    """构造 p 层标准 QAOA 电路。参数顺序为 [gamma..., beta...]。"""
    gammas = params[:p]
    betas = params[p:]

    circuit = tc.Circuit(N_NODES)
    for qubit in range(N_NODES):
        circuit.h(qubit)

    for layer in range(p):
        # exp(-i gamma C) 中与参数有关的部分对应 RZZ(-gamma)
        for u, v in EDGES:
            circuit.rzz(u, v, theta=-gammas[layer])
        # exp(-i beta sum X)
        for qubit in range(N_NODES):
            circuit.rx(qubit, theta=2.0 * betas[layer])

    return circuit


def qaoa_expectation(params, p):
    """由完整状态向量精确计算 QAOA 的期望割值。"""
    state = build_qaoa_circuit(params, p).state()
    probabilities = tf.math.real(state * tf.math.conj(state))
    return tf.reduce_sum(probabilities * CUT_VALUES_TF)


def initial_parameters(p):
    """使用带少量随机扰动的线性斜坡初始化。"""
    rng = np.random.default_rng(QAOA_SEED + p)
    gammas = np.linspace(0.2, 0.9, p)
    betas = np.linspace(0.7, 0.15, p)
    params = np.concatenate([gammas, betas])
    params += rng.normal(0.0, 0.03, size=2 * p)
    return params.astype(np.float32)


def optimize_with_adam(p, steps=ADAM_STEPS, learning_rate=LEARNING_RATE):
    """使用 Adam 最大化 QAOA 期望割值，并返回该层数的完整结果。"""
    params = tf.Variable(initial_parameters(p), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    best_energy = -np.inf
    best_params = params.numpy().copy()
    history = []

    start = time.perf_counter()
    for step in range(1, steps + 1):
        with tf.GradientTape() as tape:
            energy = qaoa_expectation(params, p)
            loss = -energy

        # energy 对应更新前的参数，因此先保存本轮参数，再执行 Adam 更新。
        energy_value = float(energy.numpy())
        current_params = params.numpy().copy()
        history.append(energy_value)
        if energy_value > best_energy:
            best_energy = energy_value
            best_params = current_params

        gradient = tape.gradient(loss, params)
        optimizer.apply_gradients([(gradient, params)])

        if step == 1 or step % 50 == 0 or step == steps:
            print(
                f"    Adam {step:3d}/{steps}: "
                f"<C>={energy_value:.6f}, ratio={energy_value / EXACT_MAXCUT:.6f}"
            )

    # 最后一次 Adam 更新后的参数还未参与上面的能量比较，这里补充评估。
    final_energy = float(qaoa_expectation(params, p).numpy())
    if final_energy > best_energy:
        best_energy = final_energy
        best_params = params.numpy().copy()
    history.append(final_energy)

    # 用优化期间记录的最优参数重新计算最终态并采样。
    final_state = build_qaoa_circuit(
        tf.constant(best_params, dtype=tf.float32), p
    ).state().numpy()
    probabilities = np.abs(final_state) ** 2
    probabilities /= probabilities.sum()

    rng = np.random.default_rng(QAOA_SEED + 100 + p)
    sampled_indices = rng.choice(N_STATES, size=N_SHOTS, p=probabilities)
    sampled_cuts = CUT_VALUES[sampled_indices]
    best_sample_position = int(np.argmax(sampled_cuts))
    best_sample_index = int(sampled_indices[best_sample_position])
    best_sample_cut = int(sampled_cuts[best_sample_position])
    best_sample_bits = format(best_sample_index, f"0{N_NODES}b")

    elapsed = time.perf_counter() - start
    return {
        "p": p,
        "time": elapsed,
        "expectation": best_energy,
        "approximation_ratio": best_energy / EXACT_MAXCUT,
        "best_sample_cut": best_sample_cut,
        "best_sample_bits": best_sample_bits,
        "parameters": best_params,
        "history": history,
    }


# ============================== 绘图 ========================================
def save_plots(results):
    """分别保存层数-时间图和层数-近似比图。"""
    layers = [result["p"] for result in results]
    times = [result["time"] for result in results]
    ratios = [result["approximation_ratio"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, times, "o-", color="#2E86AB", linewidth=2, markersize=7)
    for p, elapsed in zip(layers, times):
        ax.annotate(f"{elapsed:.2f}s", (p, elapsed), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(layers)
    ax.set_xlabel("QAOA Depth p")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("QAOA Depth vs Runtime")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("p_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, ratios, "o-", color="#E76F51", linewidth=2, markersize=7)
    ax.axhline(1.0, color="#264653", linestyle="--", label="Exact optimum")
    for p, ratio in zip(layers, ratios):
        ax.annotate(f"{ratio:.4f}", (p, ratio), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(layers)
    ax.set_xlabel("QAOA Depth p")
    ax.set_ylabel(r"Approximation Ratio $\langle C \rangle / C_{max}$")
    ax.set_title("QAOA Depth vs Approximation Ratio")
    ax.set_ylim(max(0.0, min(ratios) - 0.08), 1.04)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("p_vs_approximation_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================== 主程序 ======================================
def main():
    print("=" * 72)
    print("Random 10-Node Graph — Exact MaxCut and Adam QAOA p-Scan")
    print("=" * 72)
    print(f"Graph seed: {USED_GRAPH_SEED}")
    print(f"Vertices: {N_NODES}, edges: {N_EDGES}")
    print(f"Edges: {EDGES}")
    print(f"Exact MaxCut: {EXACT_MAXCUT}/{N_EDGES}")
    print(f"One exact partition: {EXACT_BITS}")

    results = []
    for p in P_VALUES:
        print(f"\n[p={p}] Adam optimization ({ADAM_STEPS} steps)")
        result = optimize_with_adam(p)
        results.append(result)
        print(
            f"  Finished p={p}: time={result['time']:.3f}s, "
            f"<C>={result['expectation']:.6f}, "
            f"ratio={result['approximation_ratio']:.6f}, "
            f"best sampled={result['best_sample_cut']}/{N_EDGES} "
            f"({result['best_sample_bits']})"
        )

    save_plots(results)

    print("\n" + "=" * 72)
    print(f"{'p':>3} {'time(s)':>12} {'<C>':>12} {'ratio':>12} {'sample cut':>12}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['p']:>3d} {result['time']:>12.3f} "
            f"{result['expectation']:>12.6f} "
            f"{result['approximation_ratio']:>12.6f} "
            f"{result['best_sample_cut']:>7d}/{N_EDGES:<4d}"
        )
    print("=" * 72)
    print("Saved: p_vs_time.png")
    print("Saved: p_vs_approximation_ratio.png")


if __name__ == "__main__":
    main()
