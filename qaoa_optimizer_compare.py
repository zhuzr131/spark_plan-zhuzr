"""
QAOA p=3, 20 顶点随机图 (Erdős–Rényi) — Adam vs SPSA 优化器对比
纯 NumPy 态矢量引擎 (2^20≈100万维, 单次评估 ~1s)
模拟有限采样投影噪声, 两个优化器都只需反复调用带噪声的 <C> 估计
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 20 顶点随机图 (Erdős–Rényi, seed=7, p≈0.22)
#   41 条边, 连通, 最大割 = 33 (暴力枚举, 非平凡)
# ============================================================
N_QUBITS = 20
EDGES = [
    (0,7), (1,3), (1,4), (1,6), (1,7), (1,15), (1,16), (1,18),
    (2,3), (2,5), (2,12), (2,18), (3,5), (3,13), (3,17), (3,19),
    (4,9), (4,14), (4,15), (4,18), (5,11), (5,17), (5,19), (6,7),
    (7,8), (7,10), (7,13), (7,16), (8,11), (8,13), (10,12), (11,16),
    (11,17), (11,18), (12,14), (13,18), (13,19), (14,19), (16,18),
    (16,19), (17,18),
]
P = 3
N_PARAMS = 2 * P
N_EDGES = len(EDGES)   # 41
MAX_CUT = 33           # 真实最大割 (暴力枚举)
SHOTS_NOISY = 1024

# ============================================================
# 纯 NumPy QAOA 引擎
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


def qaoa_probs(gammas, betas):
    state = _PLUS.copy()
    for k in range(P):
        state *= np.exp(0.5j * gammas[k] * _EDGE_ZZ)
        state = _rx_all(state, betas[k])
    return np.abs(state) ** 2


def expectation_exact(gammas, betas):
    return float(np.sum(qaoa_probs(gammas, betas) * _CUT_TABLE))


def expectation(gammas, betas, n_shots=None):
    """有限采样估计 <C>, 模拟投影噪声"""
    n = n_shots or SHOTS_NOISY
    probs = qaoa_probs(gammas, betas)
    samp = np.random.choice(_DIM, size=n, p=probs / probs.sum())
    return float(_CUT_TABLE[samp].mean())


def sample_distribution(gammas, betas, n_shots=2000):
    probs = qaoa_probs(gammas, betas)
    samp = np.random.choice(_DIM, size=n_shots, p=probs / probs.sum())
    vals, counts = np.unique(_CUT_TABLE[samp].astype(int), return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


# ============================================================
# 优化器 1: Adam (SPSA 随机梯度 + Adam 自适应步长)
# ============================================================
def run_adam(n_iter=120, lr=0.05):
    params = np.random.uniform(0, np.pi / 2, N_PARAMS)
    m = np.zeros(N_PARAMS); v = np.zeros(N_PARAMS)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []

    def f(p):
        return expectation(p[:P], p[P:])

    print(f"  [ADAM] {n_iter} iters, lr={lr}", flush=True)
    t0 = time.time()
    for it in range(1, n_iter + 1):
        history.append(f(params))
        ck = 0.1 / it ** 0.101
        delta = np.random.choice([1.0, -1.0], size=N_PARAMS)
        g = (f(params + ck * delta) - f(params - ck * delta)) / (2 * ck * delta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        params += lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
        if it % 30 == 0 or it == n_iter:
            print(f"  [ADAM] {it:3d}/{n_iter}  <C>={history[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    exact = expectation_exact(params[:P], params[P:])
    return {"label": "Adam", "history": history, "time": time.time() - t0,
            "cuts": sample_distribution(params[:P], params[P:]), "exact_final": exact}


# ============================================================
# 优化器 2: SPSA (同步扰动随机近似 + 普通梯度上升)
# ============================================================
def run_spsa(n_iter=120, a=0.15, c=0.1, alpha=0.602, gamma=0.101, A=20):
    params = np.random.uniform(0, np.pi / 2, N_PARAMS)
    history = []

    def f(p):
        return expectation(p[:P], p[P:])

    print(f"  [SPSA] {n_iter} iters", flush=True)
    t0 = time.time()
    for k in range(1, n_iter + 1):
        history.append(f(params))
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = np.random.choice([1.0, -1.0], size=N_PARAMS)
        g = (f(params + ck * delta) - f(params - ck * delta)) / (2 * ck * delta)
        params += ak * g
        if k % 30 == 0 or k == n_iter:
            print(f"  [SPSA] {k:3d}/{n_iter}  <C>={history[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    exact = expectation_exact(params[:P], params[P:])
    return {"label": "SPSA", "history": history, "time": time.time() - t0,
            "cuts": sample_distribution(params[:P], params[P:]), "exact_final": exact}


# ============================================================
# 可视化
# ============================================================
COLORS = {"Adam": "#B2182B", "SPSA": "#4DAF4A"}
STYLES = {"Adam": "-", "SPSA": "--"}


def plot_distributions(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_shots = sum(results[0]["cuts"].values())
    for i, r in enumerate(results):
        cuts = r["cuts"]
        xs = sorted(cuts.keys())
        ys = [cuts[x] / n_shots for x in xs]
        bar_colors = ["#D41159" if x == MAX_CUT else COLORS[r["label"]] for x in xs]
        axes[i].bar(xs, ys, color=bar_colors, edgecolor="white", width=0.6)
        if MAX_CUT in cuts:
            axes[i].text(0.95, 0.92, f"P(C={MAX_CUT})={cuts[MAX_CUT]/n_shots:.3f}",
                         transform=axes[i].transAxes, ha="right", fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        axes[i].set(xlabel="Cut Size", ylabel="Probability", title=r["label"],
                    xlim=(-0.5, N_EDGES + 0.5))
        axes[i].grid(alpha=0.3, axis="y")
    plt.suptitle(f"Cut-Size Distribution ({n_shots} shots)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("qaoa_p6_distributions.png", dpi=130, facecolor="white")
    print("  saved: qaoa_p6_distributions.png", flush=True)


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    noise_std = np.sqrt(N_EDGES * (N_EDGES / 4) / SHOTS_NOISY)

    # 收敛曲线
    ax = axes[0]
    for r in results:
        it = list(range(len(r["history"])))
        h = np.array(r["history"])
        ax.plot(it, h, STYLES[r["label"]], color=COLORS[r["label"]], linewidth=1.5,
                label=f"{r['label']} ({r['time']:.0f}s)")
        ax.fill_between(it, h - noise_std, h + noise_std, color=COLORS[r["label"]], alpha=0.08)
    ax.axhline(y=MAX_CUT, color="gray", linestyle=":", alpha=0.5, label=f"MaxCut={MAX_CUT}")
    ax.set(xlabel="Iteration", ylabel="<C>", title="Convergence (noisy)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 近似比
    ax = axes[1]
    bw = 0.35
    for i, r in enumerate(results):
        rn, re = r["history"][-1] / MAX_CUT, r["exact_final"] / MAX_CUT
        ax.bar(i - bw/2, rn, bw, color=COLORS[r["label"]], alpha=0.6, label="noisy" if i == 0 else "")
        ax.bar(i + bw/2, re, bw, color=COLORS[r["label"]], edgecolor="black", linewidth=1.2,
               label="exact" if i == 0 else "")
        ax.text(i - bw/2, rn + 0.01, f"{rn:.3f}", ha="center", fontsize=8)
        ax.text(i + bw/2, re + 0.01, f"{re:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(results))); ax.set_xticklabels([r["label"] for r in results])
    ax.set(ylabel="Approximation Ratio", title=f"Noisy vs Exact (shots={SHOTS_NOISY})", ylim=(0, 1.15))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 耗时
    ax = axes[2]
    times = [r["time"] for r in results]
    ax.bar(range(len(results)), times, color=[COLORS[r["label"]] for r in results], width=0.5)
    for i, t in enumerate(times):
        ax.text(i, t + max(times)*0.02, f"{t:.1f}s", ha="center", fontsize=10)
    ax.set_xticks(range(len(results))); ax.set_xticklabels([r["label"] for r in results])
    ax.set(ylabel="Time (s)", title="Wall Time"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("qaoa_p6_optimizer_compare.png", dpi=130, facecolor="white")
    print("  saved: qaoa_p6_optimizer_compare.png", flush=True)

    n_shots = sum(results[0]["cuts"].values())
    print(f"\n  {'Optimizer':<12} {'noisy<C>':<10} {'exact<C>':<10} {'Ratio':<8} {'Time':<8} {'P(C='+str(MAX_CUT)+')':<10}")
    print("  " + "-" * 58)
    for r in results:
        p_max = r["cuts"].get(MAX_CUT, 0) / n_shots
        print(f"  {r['label']:<12} {r['history'][-1]:<10.4f} {r['exact_final']:<10.4f} "
              f"{r['exact_final']/MAX_CUT:<8.4f} {r['time']:<8.1f} {p_max:<10.4f}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    print(f"QAOA p={P}, 20-vertex random graph ({N_EDGES} edges, MaxCut={MAX_CUT})", flush=True)
    print(f"Projection noise: shots={SHOTS_NOISY}  (单次评估 ~1s, 20 qubit)\n", flush=True)

    results = [run_adam(n_iter=60), run_spsa(n_iter=60)]

    plot_results(results)
    plot_distributions(results)
    plt.show()
