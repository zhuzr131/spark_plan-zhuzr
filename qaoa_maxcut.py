"""
QAOA-MaxCut 热身练习  (TensorCircuit + TensorFlow)
p=1, N_shot=100, 手动梯度上升优化
"""
import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

tc.set_backend("tensorflow")

# ============================================================
# 图定义
# ============================================================
C4 = {"n": 4, "edges": [(0, 1), (1, 2), (2, 3), (3, 0)]}
# G6: 2行×3列 正方格
#   0 --- 1 --- 2
#   |     |     |
#   3 --- 4 --- 5
G6 = {"n": 6, "edges": [
    (0, 1), (1, 2), (3, 4), (4, 5),  # 水平边
    (0, 3), (1, 4), (2, 5),            # 垂直边
]}
# G9: 3行×3列 正方格
#   0 --- 1 --- 2
#   |     |     |
#   3 --- 4 --- 5
#   |     |     |
#   6 --- 7 --- 8
G9 = {"n": 9, "edges": [
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),  # 水平边
    (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),  # 垂直边
]}

# ============================================================
# 步骤 (a)(b) —— 制备量子电路
# ============================================================

def build_circuit(n, edges, gamma, beta):
    """
    (a) 制备初态 |s> = |+>^⊗n
    (b) 构建 U(B,β₁)U(C,γ₁) 并制备 |ψ₁>
    """
    c = tc.Circuit(n)

    # (a) 初态：每个 qubit 作用 Hadamard 门
    for i in range(n):
        c.h(i)

    # (b) U(C, γ₁): Cost Hamiltonian 演化
    # RZZ(θ) = exp(-iθ Z_u Z_v / 2)，取 θ = -γ₁
    for u, v in edges:
        c.rzz(u, v, theta=-gamma)

    # (b) U(B, β₁): Mixer Hamiltonian 演化
    # RX(θ) = exp(-iθ X / 2)，取 θ = 2β₁
    for i in range(n):
        c.rx(i, theta=2 * beta)

    return c


# ============================================================
# 期望值计算
# ============================================================

def compute_expectation(n, edges, gamma, beta):
    """
    精确计算 <C> = <ψ₁| C |ψ₁>
    C = Σ_{(u,v)∈E} (I - Z_u Z_v)/2
    """
    c = build_circuit(n, edges, gamma, beta)
    total = 0.0
    for u, v in edges:
        # <Z_u Z_v>
        zz = tf.math.real(c.expectation_ps(z=[u, v]))
        total += (1.0 - zz) / 2.0
    return float(total)


def sample_and_measure(n, edges, gamma, beta, n_shots=100):
    """
    (c) 制备 N_shot 份 |ψ₁>，测量，得到比特串
    返回：<C>_sample, z'', C(z''), 所有割边数列表
    """
    c = build_circuit(n, edges, gamma, beta)

    total = 0
    best_z = None
    best_cut = -1
    all_costs = []

    for bit_arr, _ in c.sample(batch=n_shots):
        bit = bit_arr.numpy().astype(int)

        # 计算这个比特串的割边数
        cut_size = 0
        for u, v in edges:
            if bit[u] != bit[v]:
                cut_size += 1

        total += cut_size
        all_costs.append(cut_size)

        if cut_size > best_cut:
            best_cut = cut_size
            best_z = "".join(map(str, bit))

    avg_c = total / n_shots
    return avg_c, best_z, best_cut, all_costs


# ============================================================
# 步骤 (d) —— 经典优化器（手动梯度上升）
# ============================================================

def run_qaoa(name, g, n_shots=100, n_iter=100, lr=0.05, eps=1e-4):
    """
    (d) 使用经典优化器更新参数 γ₁, β₁

    优化方法：数值梯度 + 手动梯度上升
    1. 计算当前期望值 E(γ, β)
    2. 用有限差分近似梯度：
       dE/dγ ≈ [E(γ+ε, β) - E(γ-ε, β)] / (2ε)
       dE/dβ ≈ [E(γ, β+ε) - E(γ, β-ε)] / (2ε)
    3. 沿梯度方向更新：
       γ ← γ + lr × dE/dγ
       β ← β + lr × dE/dβ
    """
    n = g["n"]
    edges = g["edges"]
    n_edges = len(edges)

    print(f"\n{'='*55}")
    print(f"  QAOA: {name}  n={n}, |E|={n_edges}, p=1")
    print(f"  n_iter={n_iter}, lr={lr}, n_shots={n_shots}")
    print(f"{'='*55}")

    # 随机初始化参数
    gamma = np.random.uniform(0, np.pi)
    beta  = np.random.uniform(0, np.pi)

    # 记录优化历史（每步都记，方便画优化器性能图）
    history = {
        "iter": [], "gamma": [], "beta": [],
        "exact": [], "sample": [], "best": [],
        "grad_norm": [], "grad_gamma": [], "grad_beta": [],
    }
    best_z_global = None
    best_cut_global = -1

    for it in range(n_iter):
        # --- 用数值差分计算梯度 ---
        # dE/dγ ≈ [E(γ+ε) - E(γ-ε)] / (2ε)
        e_plus_g  = compute_expectation(n, edges, gamma + eps, beta)
        e_minus_g = compute_expectation(n, edges, gamma - eps, beta)
        grad_gamma = (e_plus_g - e_minus_g) / (2 * eps)

        # dE/dβ ≈ [E(β+ε) - E(β-ε)] / (2ε)
        e_plus_b  = compute_expectation(n, edges, gamma, beta + eps)
        e_minus_b = compute_expectation(n, edges, gamma, beta - eps)
        grad_beta = (e_plus_b - e_minus_b) / (2 * eps)

        # 记录梯度信息（每步都记）
        grad_norm = np.sqrt(grad_gamma**2 + grad_beta**2)
        history["grad_norm"].append(grad_norm)
        history["grad_gamma"].append(grad_gamma)
        history["grad_beta"].append(grad_beta)
        history["gamma"].append(gamma)
        history["beta"].append(beta)

        # --- 梯度上升：沿梯度方向更新参数 ---
        # 我们想最大化 <C>，所以是上升（+号）
        gamma = gamma + lr * grad_gamma
        beta  = beta  + lr * grad_beta

        # --- 每 5 步输出并采样 ---
        if it % 5 == 0 or it == n_iter - 1:
            e_exact = compute_expectation(n, edges, gamma, beta)
            e_sample, best_z, best_cut, _ = sample_and_measure(
                n, edges, gamma, beta, n_shots
            )

            history["iter"].append(it)
            history["exact"].append(e_exact)
            history["sample"].append(e_sample)
            history["best"].append(best_cut)

            if best_cut > best_cut_global:
                best_cut_global = best_cut
                best_z_global = best_z

            print(f"  iter{it:3d}  γ={gamma:.4f}  β={beta:.4f}  "
                  f"<C>_ex={e_exact:.4f}  <C>_sa={e_sample:.3f}  "
                  f"C(z'')={best_cut}")

    # 最终结果
    e_exact_final = compute_expectation(n, edges, gamma, beta)
    e_sample_final, best_z_final, best_cut_final, costs_final = \
        sample_and_measure(n, edges, gamma, beta, n_shots)

    ratio = e_exact_final / n_edges
    print(f"\n  ★ 最优参数:  γ*={gamma:.4f},  β*={beta:.4f}")
    print(f"  ★ <C>_exact = {e_exact_final:.4f}")
    print(f"  ★ C(z'') = {best_cut_final}/{n_edges}")
    print(f"  ★ 近似比 = {ratio:.4f}")

    return {
        "name": name, "g": g,
        "history": history,
        "gamma": gamma, "beta": beta,
        "exact": e_exact_final,
        "sample": e_sample_final,
        "best_cut": best_cut_final,
        "best_z": best_z_final,
        "costs": costs_final,
        "n_shots": n_shots,
        "lr": lr,
    }


# ============================================================
# 可视化
# ============================================================

def plot_cut(name, g, best_z):
    """画出图的最优分割方案"""
    n = g["n"]
    edges = g["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    # 分离割边和非割边
    cut_edges = []
    for u, v in edges:
        if best_z[u] != best_z[v]:
            cut_edges.append((u, v))

    plt.figure(figsize=(5, 4), facecolor="white")

    # 节点颜色：分区 0 = 红色，分区 1 = 青色
    node_colors = []
    text_colors = []
    for i in range(n):
        if best_z[i] == "0":
            node_colors.append("#FF6B6B")   # 红色
            text_colors.append("white")      # 红底白字
        else:
            node_colors.append("#4ECDC4")   # 青色
            text_colors.append("black")      # 青底黑字

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=500, edgecolors="black", linewidths=1)

    # 逐个画标签（支持不同颜色）
    for i, (x, y) in pos.items():
        plt.text(x, y, str(i), fontsize=12, fontweight="bold",
                 color=text_colors[i], ha="center", va="center")

    # 非割边：灰色虚线
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color="gray", style="dashed", alpha=0.4)
    # 割边：黑色实线
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color="#333333", width=3)

    plt.title(f"{name} — MaxCut: {len(cut_edges)}/{len(edges)} cuts",
              fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"qaoa_{name}_cut.png", dpi=150,
                facecolor="white", edgecolor="none")


def plot_all_results(results):
    """综合可视化：收敛曲线 + 参数轨迹 + 近似比 + 测量分布"""

    # ========== 图 1：优化历史（4 子图）==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图 1：精确 <C> 收敛
    for r in results:
        h = r["history"]
        axes[0, 0].plot(h["iter"], h["exact"], "o-", ms=3, label=r["name"])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("<C> (exact)")
    axes[0, 0].set_title("Exact <C> vs Iteration")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 子图 2：采样 <C>
    for r in results:
        h = r["history"]
        axes[0, 1].plot(h["iter"], h["sample"], "s--", ms=3, label=r["name"])
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("<C> (sample)")
    axes[0, 1].set_title(f"Sample <C> vs Iteration (N_shot={results[0]['n_shots']})")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 子图 3：参数轨迹 (γ, β)
    for r in results:
        h = r["history"]
        axes[1, 0].plot(h["gamma"], h["beta"], "o-", ms=3, label=r["name"])
        axes[1, 0].scatter(h["gamma"][-1], h["beta"][-1],
                           marker="*", s=200, zorder=5)
    axes[1, 0].set_xlabel("γ₁")
    axes[1, 0].set_ylabel("β₁")
    axes[1, 0].set_title("Parameter Trajectory (γ₁, β₁)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 子图 4：近似比柱状图
    names = [r["name"] for r in results]
    n_edges_list = [len(r["g"]["edges"]) for r in results]
    x = np.arange(len(names))
    w = 0.25

    exact_ratios = [r["exact"] / ne for r, ne in zip(results, n_edges_list)]
    sample_ratios = [r["sample"] / ne for r, ne in zip(results, n_edges_list)]
    best_ratios = [r["best_cut"] / ne for r, ne in zip(results, n_edges_list)]

    axes[1, 1].bar(x - w, exact_ratios, w, label="<C>_exact/|E|", color="#4ECDC4")
    axes[1, 1].bar(x, sample_ratios, w, label="<C>_sample/|E|", color="#FFE66D")
    axes[1, 1].bar(x + w, best_ratios, w, label="C(z'')/|E|", color="#FF6B6B")

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_ylabel("Approximation Ratio")
    axes[1, 1].set_title("Final Approximation Ratios")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("qaoa_optimization_history.png", dpi=150,
                facecolor="white", edgecolor="none")

    # ========== 图 2：测量分布 ==========
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for i, r in enumerate(results):
        counter = Counter(r["costs"])
        n_edges = len(r["g"]["edges"])

        all_cut_values = list(range(n_edges + 1))
        counts = [counter.get(v, 0) for v in all_cut_values]

        colors = []
        for v in all_cut_values:
            if v == r["best_cut"]:
                colors.append("#FF6B6B")
            else:
                colors.append("#4ECDC4")

        axes[i].bar(all_cut_values, counts, color=colors, edgecolor="white")
        axes[i].set_xlabel("Cut size C(z)")
        axes[i].set_ylabel("Counts")
        axes[i].set_title(f"{r['name']} — best = {r['best_cut']}/{n_edges}")
        axes[i].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("qaoa_counts.png", dpi=150,
                facecolor="white", edgecolor="none")


def plot_optimizer_performance(results):
    """优化器性能可视化：梯度衰减 + 参数收敛 + 步长变化"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for r in results:
        h = r["history"]
        iters = list(range(len(h["grad_norm"])))

        # 子图 1：梯度范数随迭代衰减（对数坐标看趋势）
        axes[0, 0].semilogy(iters, h["grad_norm"], linewidth=1.2, label=r["name"])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("||∇E||  (log scale)")
    axes[0, 0].set_title("Gradient Norm Decay (smaller = converged)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        iters = list(range(len(h["grad_norm"])))
        # 子图 2：dE/dγ 和 dE/dβ 分别画
        axes[0, 1].plot(iters, h["grad_gamma"], alpha=0.7, linewidth=0.8,
                        label=f"{r['name']} ∂E/∂γ")
        axes[0, 1].plot(iters, h["grad_beta"], alpha=0.7, linewidth=0.8,
                        label=f"{r['name']} ∂E/∂β")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Gradient Value")
    axes[0, 1].set_title("Per-Parameter Gradients ∂E/∂γ, ∂E/∂β")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        iters = list(range(len(h["gamma"])))
        # 子图 3：参数收敛曲线
        axes[1, 0].plot(iters, h["gamma"], "o-", ms=2, label=f"{r['name']} γ")
        axes[1, 0].plot(iters, h["beta"], "s-", ms=2, label=f"{r['name']} β")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Parameter Value")
    axes[1, 0].set_title("Parameter Convergence γ(iter), β(iter)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        lr = r["lr"]
        # 子图 4：参数更新步长 |Δγ|, |Δβ| = lr * |grad|
        deltas_g = [lr * abs(g) for g in h["grad_gamma"]]
        deltas_b = [lr * abs(g) for g in h["grad_beta"]]
        iters = list(range(len(deltas_g)))
        axes[1, 1].semilogy(iters, deltas_g, alpha=0.7, linewidth=0.8,
                            label=f"{r['name']} |Δγ|")
        axes[1, 1].semilogy(iters, deltas_b, alpha=0.7, linewidth=0.8,
                            label=f"{r['name']} |Δβ|")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("|Update Step|  (log scale)")
    axes[1, 1].set_title("Parameter Update Size |lr × grad|")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("qaoa_optimizer_performance.png", dpi=150,
                facecolor="white", edgecolor="none")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 不设 Agg，使用默认后端以支持 plt.show() 弹出窗口
    np.random.seed(42)

    # 只跑 C4
    results = [run_qaoa("C4", C4)]

    # 打印汇总
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Graph':<6} {'γ₁*':<8} {'β₁*':<8} "
          f"{'<C>_ex':<10} {'<C>_sa':<10} {'C(z'')':<8} {'Ratio':<8}")
    print(f"  " + "-" * 55)
    for r in results:
        ne = len(r["g"]["edges"])
        print(f"  {r['name']:<6} {r['gamma']:<8.4f} {r['beta']:<8.4f} "
              f"{r['exact']:<10.4f} {r['sample']:<10.4f} "
              f"{r['best_cut']}/{ne:<6} {r['exact']/ne:<8.4f}")

    # 画图（保存 + 弹出）
    for r in results:
        plot_cut(r["name"], r["g"], r["best_z"])
    plot_all_results(results)
    plot_optimizer_performance(results)

    print("\n  Done! 图片已保存，窗口弹出中...")
    plt.show()
