"""
QAOA 层数扫描实验 — 固定 17 顶点随机图, 扫描 QAOA 层数 P=1..P_MAX
研究:
  (1) 层数 vs 运行时间 (以及层数 vs 最终近似比)
  (2) 各层数的优化收敛曲线对比
纯 NumPy 态矢量引擎, SPSA 随机梯度 + Adam 自适应步长 (每步仅 2 次评估)
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 17 顶点随机图 (Erdős–Rényi, seed=17)
#   35 条边, 连通, 最大割 = 29 (暴力枚举)
# ============================================================
N_QUBITS = 17
EDGES = [
    (0,2), (0,5), (0,10), (0,11), (0,13), (0,16), (1,4), (1,5),
    (1,6), (1,13), (2,5), (2,14), (2,16), (3,5), (3,15), (3,16),
    (4,5), (4,6), (4,7), (4,15), (4,16), (5,12), (5,15), (6,8),
    (6,11), (6,12), (7,12), (8,10), (8,16), (9,13), (9,16),
    (12,13), (12,14), (13,15), (13,16),
]
N_EDGES = len(EDGES)   # 35
MAX_CUT = 29           # 真实最大割 (暴力枚举)

P_LIST = [1, 2, 3, 4, 5, 6, 8]   # 要扫描的层数
N_ITER = 60                       # 每个 P 的优化迭代数 (17 顶点较慢, 适度降低)
LR = 0.05

# ============================================================
# 纯 NumPy QAOA 引擎 (与层数 P 解耦, P 作为参数传入)
# ============================================================
_DIM = 2 ** N_QUBITS
_idx = np.arange(_DIM)
_bits = (_idx[:, None] >> np.arange(N_QUBITS)[None, :]) & 1
_EDGE_ZZ = np.zeros(_DIM)
_CUT_TABLE = np.zeros(_DIM)
for _u, _v in EDGES:
    _s = (1 - 2 * _bits[:, _u]) * (1 - 2 * _bits[:, _v])
    _EDGE_ZZ += _s
    _CUT_TABLE += (_s < 0)
_PLUS = np.ones(_DIM, dtype=complex) / np.sqrt(_DIM)


def _rx_all(state, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    st = state.reshape([2] * N_QUBITS)
    for q in range(N_QUBITS):
        st = np.moveaxis(st, q, 0)
        a0, a1 = st[0].copy(), st[1].copy()
        st[0], st[1] = c * a0 + s * a1, s * a0 + c * a1
        st = np.moveaxis(st, 0, q)
    return st.reshape(_DIM)


def qaoa_probs(gammas, betas, P):
    state = _PLUS.copy()
    for k in range(P):
        state *= np.exp(0.5j * gammas[k] * _EDGE_ZZ)
        state = _rx_all(state, betas[k])
    return np.abs(state) ** 2


def expectation_exact(gammas, betas, P):
    return float(np.sum(qaoa_probs(gammas, betas, P) * _CUT_TABLE))


# ============================================================
# 单个层数 P 的 QAOA 优化 (SPSA + Adam, 精确 <C>)
# ============================================================
def optimize_for_P(P, n_iter=N_ITER, lr=LR, a_spsa=0.1, alpha=0.101):
    n_params = 2 * P

    def energy(theta):
        return expectation_exact(theta[:P], theta[P:], P)

    params = np.random.uniform(0, np.pi, n_params)
    m = np.zeros(n_params); v = np.zeros(n_params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    t0 = time.time()
    for it in range(n_iter):
        history.append(energy(params))          # 记录当前 <C>
        ck = a_spsa / (it + 1) ** alpha
        delta = np.random.choice([1.0, -1.0], size=n_params)
        grad = (energy(params + ck * delta) - energy(params - ck * delta)) / (2 * ck * delta)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        params += lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + eps)
    elapsed = time.time() - t0

    final_exact = energy(params)
    history.append(final_exact)
    ratio = final_exact / MAX_CUT
    print(f"  P={P:2d}  final<C>={final_exact:6.3f}  ratio={ratio:.4f}  "
          f"time={elapsed:5.2f}s  ({n_iter} iters, {n_params} params)", flush=True)
    return {"P": P, "history": history, "time": elapsed,
            "final": final_exact, "ratio": ratio, "n_params": n_params}


# ============================================================
# 可视化
# ============================================================
def plot_scan(results):
    Ps = [r["P"] for r in results]
    times = [r["time"] for r in results]
    ratios = [r["ratio"] for r in results]

    # ===== 图 1: 层数 vs 运行时间 (+ 近似比) =====
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#B2182B"
    ax1.plot(Ps, times, "o-", color=color1, linewidth=2, markersize=7, label="Runtime")
    ax1.set_xlabel("QAOA Layers  P")
    ax1.set_ylabel("Runtime (seconds)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)
    for p, t in zip(Ps, times):
        ax1.annotate(f"{t:.1f}s", (p, t), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=color1)

    ax2 = ax1.twinx()
    color2 = "#2166AC"
    ax2.plot(Ps, ratios, "s--", color=color2, linewidth=2, markersize=7, label="Approx. Ratio")
    ax2.set_ylabel("Approximation Ratio  <C>/MaxCut", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(min(ratios) - 0.05, 1.02)
    for p, rr in zip(Ps, ratios):
        ax2.annotate(f"{rr:.3f}", (p, rr), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=color2)

    plt.title(f"QAOA Layers vs Runtime & Quality  ({N_QUBITS}-vertex graph, MaxCut={MAX_CUT})",
              fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig("qaoa_layers_time_vs_quality.png", dpi=130, facecolor="white")
    print("  saved: qaoa_layers_time_vs_quality.png", flush=True)

    # ===== 图 2: 各层数收敛曲线对比 =====
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for r, col in zip(results, cmap):
        h = r["history"]
        ax.plot(range(len(h)), h, "-", color=col, linewidth=1.8,
                label=f"P={r['P']}  (→{r['final']:.2f}, {r['time']:.1f}s)")
    ax.axhline(y=MAX_CUT, color="gray", linestyle=":", alpha=0.6, label=f"MaxCut={MAX_CUT}")
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("<C> (exact expectation)")
    ax.set_title(f"Convergence Curves for Different QAOA Layers  P  ({N_QUBITS}-vertex, {N_EDGES} edges)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig("qaoa_layers_convergence.png", dpi=130, facecolor="white")
    print("  saved: qaoa_layers_convergence.png", flush=True)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    print(f"QAOA layer scan on {N_QUBITS}-vertex random graph "
          f"({N_EDGES} edges, MaxCut={MAX_CUT})", flush=True)
    print(f"Scanning P = {P_LIST},  {N_ITER} iters each\n", flush=True)

    results = [optimize_for_P(P) for P in P_LIST]

    print(f"\n  {'P':<4} {'params':<8} {'final<C>':<10} {'ratio':<8} {'time(s)':<8}")
    print("  " + "-" * 40)
    for r in results:
        print(f"  {r['P']:<4} {r['n_params']:<8} {r['final']:<10.4f} "
              f"{r['ratio']:<8.4f} {r['time']:<8.2f}")

    plot_scan(results)
    print(f"\n  Done!")
    plt.show()
