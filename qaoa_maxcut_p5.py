"""
QAOA-MaxCut p=5 层  (TensorCircuit + TensorFlow)
手动数值梯度 + 梯度上升，无黑盒优化器
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

# QAOA 层数
P = 5

# ============================================================
# 步骤 (a)(b) —— 制备量子电路
# ============================================================

def build_circuit(n, edges, gammas, betas):
    """
    (a) 初态 |s> = |+>^⊗n
    (b) 重复 p 层: U(B,β_k) U(C,γ_k)
        U(C,γ_k) = Π RZZ(-γ_k)  作用在所有边上
        U(B,β_k) = Π RX(2β_k)   作用在所有 qubit 上
    """
    c = tc.Circuit(n)

    # (a) 初态
    for i in range(n):
        c.h(i)

    # (b) p 层交替演化
    for k in range(P):
        gamma = gammas[k]
        beta  = betas[k]

        # Cost 层: U(C, γ_k)
        for u, v in edges:
            c.rzz(u, v, theta=-gamma)

        # Mixer 层: U(B, β_k)
        for i in range(n):
            c.rx(i, theta=2 * beta)

    return c


# ============================================================
# 期望值计算
# ============================================================

def compute_expectation(n, edges, gammas, betas):
    """
    精确 <C> = <ψ_p| C |ψ_p>
    C = Σ_{(u,v)∈E} (I - Z_u Z_v)/2
    """
    c = build_circuit(n, edges, gammas, betas)
    total = 0.0
    for u, v in edges:
        zz = tf.math.real(c.expectation_ps(z=[u, v]))
        total += (1.0 - zz) / 2.0
    return float(total)


def sample_and_measure(n, edges, gammas, betas, n_shots=100):
    """
    (c) 测量 N_shot 次，返回采样估计
    """
    c = build_circuit(n, edges, gammas, betas)

    total = 0
    best_z = None
    best_cut = -1
    all_costs = []

    for bit_arr, _ in c.sample(batch=n_shots):
        bit = bit_arr.numpy().astype(int)
        cut_size = 0
        for u, v in edges:
            if bit[u] != bit[v]:
                cut_size += 1
        total += cut_size
        all_costs.append(cut_size)
        if cut_size > best_cut:
            best_cut = cut_size
            best_z = "".join(map(str, bit))

    return total / n_shots, best_z, best_cut, all_costs


# ============================================================
# 步骤 (d) —— 手动数值梯度 + 梯度上升
# ============================================================

def run_qaoa(name, g, n_shots=100, n_iter=100, lr=0.03, eps=1e-4):
    """
    p 层 QAOA，手动梯度上升优化 2p 个参数

    对每个参数 θ_i (i=0..2p-1):
      grad_i ≈ [E(θ_i+ε) - E(θ_i-ε)] / (2ε)
      θ_i ← θ_i + lr × grad_i
    """
    n = g["n"]
    edges = g["edges"]
    n_edges = len(edges)

    print(f"\n{'='*55}")
    print(f"  QAOA: {name}  n={n}, |E|={n_edges}, p={P}")
    print(f"  n_iter={n_iter}, lr={lr}")
    print(f"{'='*55}")

    # 随机初始化 2p 个参数
    # gammas[0..P-1] = γ₁, γ₂, ..., γₚ
    # betas[0..P-1]  = β₁, β₂, ..., βₚ
    gammas = np.random.uniform(0, np.pi, P)
    betas  = np.random.uniform(0, np.pi, P)

    history = {
        "iter": [], "exact": [], "sample": [], "best": [],
        "gamma_0": [], "beta_0": [],
        "grad_norm": [], "grad_gammas": [], "grad_betas": [],
    }
    best_z_global = None
    best_cut_global = -1

    for it in range(n_iter):
        # ===== 计算所有 2P 个参数的数值梯度 =====
        grad_gammas = np.zeros(P)
        grad_betas  = np.zeros(P)

        for k in range(P):
            # --- ∂E/∂γ_k ---
            gammas_plus  = gammas.copy()
            gammas_minus = gammas.copy()
            gammas_plus[k]  += eps
            gammas_minus[k] -= eps

            e_plus  = compute_expectation(n, edges, gammas_plus, betas)
            e_minus = compute_expectation(n, edges, gammas_minus, betas)
            grad_gammas[k] = (e_plus - e_minus) / (2 * eps)

            # --- ∂E/∂β_k ---
            betas_plus  = betas.copy()
            betas_minus = betas.copy()
            betas_plus[k]  += eps
            betas_minus[k] -= eps

            e_plus  = compute_expectation(n, edges, gammas, betas_plus)
            e_minus = compute_expectation(n, edges, gammas, betas_minus)
            grad_betas[k] = (e_plus - e_minus) / (2 * eps)

        # 记录梯度信息（每步都记）
        g_flat = np.concatenate([grad_gammas, grad_betas])
        history["grad_norm"].append(float(np.sqrt(np.sum(g_flat**2))))
        history["grad_gammas"].append(grad_gammas.copy())
        history["grad_betas"].append(grad_betas.copy())

        # ===== 梯度上升更新所有参数 =====
        for k in range(P):
            gammas[k] += lr * grad_gammas[k]
            betas[k]  += lr * grad_betas[k]

        # ===== 每 5 步输出 =====
        if it % 5 == 0 or it == n_iter - 1:
            e_exact = compute_expectation(n, edges, gammas, betas)
            e_sample, best_z, best_cut, _ = sample_and_measure(
                n, edges, gammas, betas, n_shots
            )

            history["iter"].append(it)
            history["exact"].append(e_exact)
            history["sample"].append(e_sample)
            history["best"].append(best_cut)
            history["gamma_0"].append(gammas[0])
            history["beta_0"].append(betas[0])

            if best_cut > best_cut_global:
                best_cut_global = best_cut
                best_z_global = best_z

            print(f"  iter{it:3d}  <C>_ex={e_exact:.4f}  <C>_sa={e_sample:.3f}  "
                  f"C(z'')={best_cut}")

            if it % 20 == 0 or it == n_iter - 1:
                g_str = ", ".join([f"{g:.3f}" for g in gammas])
                b_str = ", ".join([f"{b:.3f}" for b in betas])
                print(f"         γ=[{g_str}]")
                print(f"         β=[{b_str}]")

    # 最终结果
    e_exact_final = compute_expectation(n, edges, gammas, betas)
    e_sample_final, best_z_final, best_cut_final, costs_final = \
        sample_and_measure(n, edges, gammas, betas, n_shots)

    ratio = e_exact_final / n_edges
    print(f"\n  ★ γ* = {[round(g, 4) for g in gammas]}")
    print(f"  ★ β* = {[round(b, 4) for b in betas]}")
    print(f"  ★ <C>_exact = {e_exact_final:.4f}")
    print(f"  ★ C(z'') = {best_cut_final}/{n_edges}")
    print(f"  ★ 近似比 = {ratio:.4f}")

    return {
        "name": name, "g": g,
        "history": history,
        "gammas": list(gammas), "betas": list(betas),
        "exact": e_exact_final,
        "sample": e_sample_final,
        "best_cut": best_cut_final,
        "best_z": best_z_final,
        "costs": costs_final,
        "n_shots": n_shots,
        "p": P,
        "lr": lr,
    }


# ============================================================
# 可视化
# ============================================================

def plot_cut(name, g, best_z, p):
    """画图的最优分割"""
    n = g["n"]
    edges = g["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    cut_edges = [(u, v) for u, v in edges if best_z[u] != best_z[v]]

    plt.figure(figsize=(5, 4), facecolor="white")

    node_colors = []
    text_colors = []
    for i in range(n):
        if best_z[i] == "0":
            node_colors.append("#FF6B6B")
            text_colors.append("white")
        else:
            node_colors.append("#4ECDC4")
            text_colors.append("black")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=500, edgecolors="black", linewidths=1)
    for i, (x, y) in pos.items():
        plt.text(x, y, str(i), fontsize=12, fontweight="bold",
                 color=text_colors[i], ha="center", va="center")
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color="gray", style="dashed", alpha=0.4)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color="#333333", width=3)

    plt.title(f"{name}  p={p}  — MaxCut: {len(cut_edges)}/{len(edges)} cuts",
              fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"qaoa_p{p}_{name}_cut.png", dpi=150,
                facecolor="white", edgecolor="none")


def plot_all_results(results):
    """综合可视化"""
    p = results[0]["p"]

    # ===== 图 1：优化历史 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for r in results:
        h = r["history"]
        axes[0, 0].plot(h["iter"], h["exact"], "o-", ms=3, label=r["name"])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("<C> (exact)")
    axes[0, 0].set_title(f"Exact <C> vs Iteration (p={p})")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        axes[0, 1].plot(h["iter"], h["sample"], "s--", ms=3, label=r["name"])
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("<C> (sample)")
    axes[0, 1].set_title(f"Sample <C> vs Iteration (N_shot={results[0]['n_shots']})")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        axes[1, 0].plot(h["gamma_0"], h["beta_0"], "o-", ms=3, label=r["name"])
        axes[1, 0].scatter(h["gamma_0"][-1], h["beta_0"][-1],
                           marker="*", s=200, zorder=5)
    axes[1, 0].set_xlabel("γ₁")
    axes[1, 0].set_ylabel("β₁")
    axes[1, 0].set_title(f"Trajectory of (γ₁, β₁) [first of {p} layers]")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    names = [r["name"] for r in results]
    ne_list = [len(r["g"]["edges"]) for r in results]
    x = np.arange(len(names))
    w = 0.25
    axes[1, 1].bar(x - w, [r["exact"]/ne for r, ne in zip(results, ne_list)],
                   w, label="<C>_exact/|E|", color="#4ECDC4")
    axes[1, 1].bar(x, [r["sample"]/ne for r, ne in zip(results, ne_list)],
                   w, label="<C>_sample/|E|", color="#FFE66D")
    axes[1, 1].bar(x + w, [r["best_cut"]/ne for r, ne in zip(results, ne_list)],
                   w, label="C(z'')/|E|", color="#FF6B6B")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_ylabel("Approximation Ratio")
    axes[1, 1].set_title(f"Final Approximation Ratios (p={p})")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"qaoa_p{p}_optimization_history.png", dpi=150,
                facecolor="white", edgecolor="none")

    # ===== 图 2：测量分布 =====
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for i, r in enumerate(results):
        counter = Counter(r["costs"])
        ne = len(r["g"]["edges"])
        vals = list(range(ne + 1))
        counts = [counter.get(v, 0) for v in vals]
        colors = ["#FF6B6B" if v == r["best_cut"] else "#4ECDC4" for v in vals]
        axes[i].bar(vals, counts, color=colors, edgecolor="white")
        axes[i].set_xlabel("Cut size C(z)")
        axes[i].set_ylabel("Counts")
        axes[i].set_title(f"{r['name']}  p={p}  — best = {r['best_cut']}/{ne}")
        axes[i].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"qaoa_p{p}_counts.png", dpi=150,
                facecolor="white", edgecolor="none")


def plot_optimizer_performance(results):
    """优化器性能可视化：梯度衰减 + 参数更新步长"""
    p = results[0]["p"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for r in results:
        h = r["history"]
        iters = list(range(len(h["grad_norm"])))

        # 子图 1：梯度总范数（对数坐标）
        axes[0, 0].semilogy(iters, h["grad_norm"], linewidth=1.2, label=r["name"])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("||∇E||  (log scale)")
    axes[0, 0].set_title(f"Gradient Norm Decay — p={p} (smaller = converged)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        lr = r["lr"]
        iters = list(range(len(h["grad_norm"])))
        # 子图 2：每层 γ_k 的梯度范数（看哪层贡献大）
        for k in range(min(3, p)):  # 只画前 3 层，避免太乱
            gk = [abs(g[k]) for g in h["grad_gammas"]]
            axes[0, 1].semilogy(iters, gk, linewidth=0.8,
                                alpha=0.7, label=f"{r['name']} |∂E/∂γ_{{{k+1}}}|")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("|∂E/∂γ_k|  (log scale)")
    axes[0, 1].set_title(f"Gradient Magnitude Per Layer γ_k (first {min(3,p)} layers)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        lr = r["lr"]
        iters = list(range(len(h["grad_norm"])))
        # 子图 3：参数更新总步长
        total_step = [lr * h["grad_norm"][i] for i in range(len(h["grad_norm"]))]
        axes[1, 0].semilogy(iters, total_step, linewidth=1.2, label=r["name"])
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Total Update Step  (log scale)")
    axes[1, 0].set_title(f"Parameter Update Size |lr × ∇E| — p={p}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    for r in results:
        h = r["history"]
        lr = r["lr"]
        # 子图 4：期望值增量（每步提高了多少）
        iters = list(range(len(h["grad_norm"])))
        # 采样期望值每 5 步才记录，这里用梯度范数间接表示改善速度
        axes[1, 1].plot(iters, h["grad_norm"], linewidth=0.8,
                        label=f"{r['name']} ∇ norm")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("||∇E||")
    axes[1, 1].set_title(f"Gradient Norm Over Iterations — p={p}")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"qaoa_p{p}_optimizer_performance.png", dpi=150,
                facecolor="white", edgecolor="none")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 不设 Agg，使用默认后端以支持 plt.show() 弹出窗口
    np.random.seed(42)

    results = [
        run_qaoa("C4", C4, n_iter=100, lr=0.03),
        run_qaoa("G6", G6, n_iter=150, lr=0.02),
        run_qaoa("G9", G9, n_iter=200, lr=0.02),
    ]

    # 汇总
    print(f"\n{'='*70}")
    print(f"  SUMMARY  p={P}")
    print(f"{'='*70}")
    print(f"  {'Graph':<6} {'<C>_ex':<10} {'<C>_sa':<10} {'C(z'')':<10} {'Ratio':<8}")
    print(f"  " + "-" * 45)
    for r in results:
        ne = len(r["g"]["edges"])
        print(f"  {r['name']:<6} {r['exact']:<10.4f} {r['sample']:<10.4f} "
              f"{r['best_cut']}/{ne:<7} {r['exact']/ne:<8.4f}")

    print(f"\n  Optimal parameters:")
    for r in results:
        print(f"  {r['name']}:")
        print(f"    γ = {[round(g,4) for g in r['gammas']]}")
        print(f"    β = {[round(b,4) for b in r['betas']]}")

    # 画图（保存 + 弹出）
    for r in results:
        plot_cut(r["name"], r["g"], r["best_z"], p=P)
    plot_all_results(results)
    plot_optimizer_performance(results)

    print(f"\n  Done! 图片已保存，窗口弹出中...")
    plt.show()
