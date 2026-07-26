"""
QAOA-MaxCut 热身练习  (TensorCircuit + TensorFlow)
p=1, N_shot=100, 优化器: Adam
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
G6 = {"n": 6, "edges": [
    (0, 1), (0, 2), (0, 5), (1, 2), (1, 3),
    (2, 3), (2, 4), (3, 4), (3, 5), (4, 5),
]}
G9 = {"n": 9, "edges": [
    (0, 1), (0, 3), (0, 8), (1, 2), (1, 4),
    (2, 3), (2, 5), (3, 6), (4, 5), (4, 7),
    (5, 6), (5, 8), (6, 7), (7, 8),
]}


# ============================================================
# QAOA 核心
# ============================================================
def cut(bit, edges):
    """MaxCut 割边数"""
    return sum(1 for u, v in edges if bit[u] != bit[v])


def qaoa_circuit(n, edges, gamma, beta):
    """制备 |ψ₁> = U(B,β₁)U(C,γ₁)|+>^⊗n"""
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)                           # (a) 初态 |+>^⊗n
    for u, v in edges:
        c.rzz(u, v, theta=-gamma)        # U(C,γ): RZZ(-γ)
    for i in range(n):
        c.rx(i, theta=2 * beta)          # U(B,β): RX(2β)
    return c


def exact_exp(n, edges, gamma, beta):
    """<C> = Σ (1 - <Z_u Z_v>)/2"""
    c = qaoa_circuit(n, edges, gamma, beta)
    return sum((1.0 - tf.math.real(c.expectation_ps(z=[u, v]))) / 2.0
               for u, v in edges)


def sample_exp(n, edges, gamma, beta, n_shots=100):
    """(c) N_shot 次采样 → <C>, z'', C(z'')"""
    c = qaoa_circuit(n, edges, gamma, beta)
    total, best_z, best_cut, costs = 0, None, -1, []
    for bit_arr, _ in c.sample(batch=n_shots):
        bit = bit_arr.numpy().astype(int)
        cc = cut(bit, edges)
        total += cc; costs.append(cc)
        if cc > best_cut:
            best_cut, best_z = cc, "".join(map(str, bit))
    return total / n_shots, best_z, best_cut, costs


def run_qaoa(name, g, n_shots=100, n_iter=100, lr=0.05):
    """(d) 完整 QAOA 流程"""
    n, edges = g["n"], g["edges"]
    print(f"\n{'='*55}\n  QAOA: {name}  n={n}, |E|={len(edges)}\n{'='*55}")

    gamma = tf.Variable(np.random.uniform(0, np.pi), tf.float32)
    beta = tf.Variable(np.random.uniform(0, np.pi), tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    hist = {"iter": [], "g": [], "b": [], "exact": [], "sample": [], "best": []}
    best_z_global, best_cut_global = None, -1

    for it in range(n_iter):
        with tf.GradientTape() as tape:
            loss = -exact_exp(n, edges, gamma, beta)
        opt.apply_gradients(zip(tape.gradient(loss, [gamma, beta]), [gamma, beta]))

        if it % 5 == 0 or it == n_iter - 1:
            e = float(exact_exp(n, edges, gamma, beta))
            s, bz, bc, _ = sample_exp(n, edges, gamma, beta, n_shots)
            hist["iter"].append(it); hist["g"].append(float(gamma))
            hist["b"].append(float(beta)); hist["exact"].append(e)
            hist["sample"].append(s); hist["best"].append(bc)
            if bc > best_cut_global:
                best_cut_global, best_z_global = bc, bz
            print(f"  iter{it:3d}  γ={gamma.numpy():.4f} β={beta.numpy():.4f}  "
                  f"<C>_ex={e:.3f}  <C>_sa={s:.3f}  best={bc}  z''={bz}")

    s_final, bz_final, bc_final, costs_final = sample_exp(n, edges, gamma, beta, n_shots)
    e_final = float(exact_exp(n, edges, gamma, beta))
    ratio = e_final / len(edges)
    print(f"\n  ★ γ*={gamma.numpy():.4f}  β*={beta.numpy():.4f}")
    print(f"  ★ <C>={e_final:.4f}  C(z'')={bc_final}/{len(edges)}  ratio={ratio:.4f}")

    return {"name": name, "g": g, "hist": hist, "gamma": float(gamma), "beta": float(beta),
            "exact": e_final, "sample": s_final, "best_cut": bc_final, "best_z": bz_final,
            "costs": costs_final, "n_shots": n_shots}


# ============================================================
# 可视化
# ============================================================
def plot_cut(name, g, bs):
    """图的最优 MaxCut 分割"""
    n = g["n"]; edges = g["edges"]
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    cut_es = [(u, v) for u, v in edges if bs[u] != bs[v]]
    plt.figure(figsize=(5, 4), facecolor="white")
    # 分区 0 = 红色, 分区 1 = 青色 (浅色)
    node_c = ["#FF6B6B" if bs[i] == "0" else "#4ECDC4" for i in range(n)]
    txt_c = ["white" if bs[i] == "0" else "black" for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_color=node_c, node_size=500,
                           edgecolors="black", linewidths=1)
    # 逐个画 label 以支持不同颜色
    for i, (x, y) in pos.items():
        plt.text(x, y, str(i), fontsize=12, fontweight="bold",
                 color=txt_c[i], ha="center", va="center")
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="gray",
                           style="dashed", alpha=0.4)
    nx.draw_networkx_edges(G, pos, edgelist=cut_es, edge_color="#333333", width=3)
    plt.title(f"{name} — MaxCut: {len(cut_es)}/{len(edges)} cuts", fontsize=14)
    plt.axis("off")
    plt.tight_layout(); plt.savefig(f"qaoa_{name}_cut.png", dpi=150,
                                     facecolor="white", edgecolor="none"); plt.close()


def plot_results(results):
    """优化历史 + 最终近似比 + 测量分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # (1) <C> exact 收敛
    for r in results:
        axes[0, 0].plot(r["hist"]["iter"], r["hist"]["exact"], "o-", ms=3, label=r["name"])
    axes[0, 0].set(xlabel="Iteration", ylabel="<C> (exact)", title="Exact <C> vs Iteration")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # (2) <C> sample
    for r in results:
        axes[0, 1].plot(r["hist"]["iter"], r["hist"]["sample"], "s--", ms=3, label=r["name"])
    axes[0, 1].set(xlabel="Iteration", ylabel="<C> (sample)",
                   title=f"Sample <C> vs Iteration (N_shot={results[0]['n_shots']})")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    # (3) 参数轨迹
    for r in results:
        axes[1, 0].plot(r["hist"]["g"], r["hist"]["b"], "o-", ms=3, label=r["name"])
        axes[1, 0].scatter(r["hist"]["g"][-1], r["hist"]["b"][-1], marker="*", s=200)
    axes[1, 0].set(xlabel="γ₁", ylabel="β₁", title="Parameter Trajectory (γ₁, β₁)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    # (4) 近似比
    names = [r["name"] for r in results]; ne = [len(r["g"]["edges"]) for r in results]
    x = np.arange(len(names)); w = 0.25
    axes[1, 1].bar(x - w, [r["exact"]/e for r, e in zip(results, ne)], w, label="<C>_exact/|E|", color="#4ECDC4")
    axes[1, 1].bar(x, [r["sample"]/e for r, e in zip(results, ne)], w, label="<C>_sample/|E|", color="#FFE66D")
    axes[1, 1].bar(x + w, [r["best_cut"]/e for r, e in zip(results, ne)], w, label="C(z'')/|E|", color="#FF6B6B")
    axes[1, 1].set(xticks=x, xticklabels=names, ylabel="Approximation Ratio",
                   title="Final Approximation Ratios", ylim=(0, 1.15))
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("qaoa_optimization_history.png", dpi=150,
                                     facecolor="white", edgecolor="none"); plt.close()

    # 测量分布
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1: axes = [axes]
    for i, r in enumerate(results):
        cnt = Counter(r["costs"]); ne = len(r["g"]["edges"])
        costs = list(range(ne + 1))
        freqs = [cnt.get(c, 0) for c in costs]
        clr = ["#FF6B6B" if c == r["best_cut"] else "#4ECDC4" for c in costs]
        axes[i].bar(costs, freqs, color=clr, edgecolor="white")
        axes[i].set(xlabel="Cut size C(z)", ylabel="Counts",
                    title=f"{r['name']} — best={r['best_cut']}/{ne}")
        axes[i].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("qaoa_counts.png", dpi=150,
                                     facecolor="white", edgecolor="none"); plt.close()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    plt.switch_backend("Agg")
    np.random.seed(42)

    results = [run_qaoa("C4", C4)]

    # 汇总表
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Graph':<6} {'γ₁*':<8} {'β₁*':<8} {'<C>_ex':<10} {'<C>_sa':<10} {'C(z'')':<8} {'Ratio':<8}")
    print("  " + "-" * 55)
    for r in results:
        ne = len(r["g"]["edges"])
        print(f"  {r['name']:<6} {r['gamma']:<8.4f} {r['beta']:<8.4f} "
              f"{r['exact']:<10.4f} {r['sample']:<10.4f} "
              f"{r['best_cut']:<8} {r['exact']/ne:<8.4f}")

    # 出图
    for r in results:
        plot_cut(r["name"], r["g"], r["best_z"])
    plot_results(results)
    print("\n  Done!")
