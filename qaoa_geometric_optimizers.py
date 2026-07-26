"""
QAOA p=3, 不规则 12 顶点图 — 无梯度几何优化器对比
  CMA-ES (协方差矩阵自适应演化策略)
  Nelder-Mead (单纯形几何搜索, 带重采样的噪声鲁棒版本)
纯 NumPy 态矢量引擎 (比 tensorcircuit 快约 150×), 投影噪声 1024 shots
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 不规则 12 顶点图 (含对角/交叉边, 非二分图)
#   总边数 22, 最大割 = 17 (非平凡, 因含奇环)
# ============================================================
N_QUBITS = 12
EDGES = [
    (0,1), (1,2), (2,3), (4,5), (5,6), (6,7), (8,9), (9,10), (10,11),
    (0,4), (4,8), (1,5), (5,9), (2,6), (6,10), (3,7), (7,11),
    (0,5), (2,5), (6,9), (1,6), (3,6),
]
P = 3
N_PARAMS = 2 * P
N_EDGES = len(EDGES)   # 22
MAX_CUT = 17           # 真实最大割
SHOTS_NOISY = 1024
MAX_EVAL = 800         # 每个优化器的函数评估预算

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
    n = n_shots or SHOTS_NOISY
    probs = qaoa_probs(gammas, betas)
    samp = np.random.choice(_DIM, size=n, p=probs / probs.sum())
    return float(_CUT_TABLE[samp].mean())


def sample_distribution(gammas, betas, n_shots=2000):
    probs = qaoa_probs(gammas, betas)
    samp = np.random.choice(_DIM, size=n_shots, p=probs / probs.sum())
    vals, counts = np.unique(_CUT_TABLE[samp].astype(int), return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def params_obj(params, n_shots):
    """最小化目标 f = -<C>"""
    return -expectation(params[:P], params[P:], n_shots)


# ============================================================
# 优化器 1: CMA-ES (协方差矩阵自适应演化策略)
# ============================================================
def run_cma_es(max_evals=MAX_EVAL):
    N = N_PARAMS
    xmean = np.random.uniform(0, np.pi / 2, N)
    sigma = 0.3
    lam = 4 + int(3 * np.log(N))      # 种群规模
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1 / np.sum(weights ** 2)

    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    pc = np.zeros(N); ps = np.zeros(N)
    C = np.eye(N); B = np.eye(N); D = np.ones(N)

    history, eval_hist = [], []
    t0 = time.time(); evals = 0
    print(f"  [CMA-ES] λ={lam} μ={mu}  max_evals={max_evals}", flush=True)

    while evals < max_evals:
        # 采样
        arz = np.random.randn(lam, N)
        ary = arz @ np.diag(D) @ B.T
        arx = xmean + sigma * ary
        arx = np.clip(arx, 0, np.pi)
        fitness = np.array([params_obj(x, SHOTS_NOISY) for x in arx])
        evals += lam

        order = np.argsort(fitness)
        arx, ary = arx[order], ary[order]

        xold = xmean.copy()
        xmean = weights @ arx[:mu]
        ymean = weights @ ary[:mu]

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (ymean / D))
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * evals / lam)) / chiN < 1.4 + 2 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean

        artmp = ary[:mu]
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * weights) @ artmp)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        C = np.triu(C) + np.triu(C, 1).T
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-20))

        history.append(-fitness[order[0]])
        eval_hist.append(evals)
        if len(eval_hist) % 5 == 0 or evals >= max_evals:
            print(f"  [CMA-ES] eval {evals:4d}/{max_evals}  best<C>={history[-1]:.3f}"
                  f"  σ={sigma:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    best = xmean
    exact = expectation_exact(best[:P], best[P:])
    print(f"  [CMA-ES] noisy<C>={history[-1]:.4f}  exact<C>={exact:.4f}  {time.time()-t0:.1f}s", flush=True)
    return {"label": "CMA-ES", "history": history, "eval_hist": eval_hist,
            "time": time.time() - t0, "cuts": sample_distribution(best[:P], best[P:]),
            "exact_final": exact}


# ============================================================
# 优化器 2: Nelder-Mead (带重采样的噪声鲁棒版本)
#   每轮重新评估最优点, 消除"陈旧有利噪声"锁死比较的问题
# ============================================================
def run_nelder_mead(max_evals=MAX_EVAL):
    N = N_PARAMS
    n_s = SHOTS_NOISY
    x0 = np.random.uniform(0, np.pi / 2, N)
    simplex = [x0.copy()]
    for i in range(N):
        pt = x0.copy()
        pt[i] += 0.3 if pt[i] + 0.3 < np.pi else -0.3
        simplex.append(pt)
    fvals = np.array([params_obj(p, n_s) for p in simplex])
    a, gm, rho, sig = 1.0, 2.0, 0.5, 0.5

    history = []; t0 = time.time(); evals = N + 1
    print(f"  [NM] simplex={N+1} vertices  max_evals={max_evals}", flush=True)

    while evals < max_evals:
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]; fvals = fvals[order]
        fvals[0] = params_obj(simplex[0], n_s); evals += 1   # 重采样最优点
        history.append(-fvals[0])

        if len(history) % 40 == 0 or evals >= max_evals:
            print(f"  [NM] eval {evals:4d}/{max_evals}  best<C>={history[-1]:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        xbar = np.mean(simplex[:-1], axis=0)
        xr = xbar + a * (xbar - simplex[-1]); fr = params_obj(xr, n_s); evals += 1
        if fr < fvals[0]:
            xe = xbar + gm * (xr - xbar); fe = params_obj(xe, n_s); evals += 1
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = xbar + rho * ((xr if fr < fvals[-1] else simplex[-1]) - xbar)
            fc = params_obj(xc, n_s); evals += 1
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, N + 1):
                    simplex[i] = simplex[0] + sig * (simplex[i] - simplex[0])
                    fvals[i] = params_obj(simplex[i], n_s); evals += 1
                    if evals >= max_evals:
                        break

    best = simplex[0]
    exact = expectation_exact(best[:P], best[P:])
    print(f"  [NM] noisy<C>={history[-1]:.4f}  exact<C>={exact:.4f}  {time.time()-t0:.1f}s", flush=True)
    return {"label": "Nelder-Mead", "history": history, "time": time.time() - t0,
            "cuts": sample_distribution(best[:P], best[P:]), "exact_final": exact}


# ============================================================
# 可视化
# ============================================================
COLORS = {"CMA-ES": "#E69F00", "Nelder-Mead": "#009E73"}
STYLES = {"CMA-ES": "-", "Nelder-Mead": "--"}


def plot_summary(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 收敛
    ax = axes[0]
    for r in results:
        x = r.get("eval_hist", np.arange(1, len(r["history"]) + 1) * (MAX_EVAL / len(r["history"])))
        ax.plot(x, r["history"], STYLES[r["label"]], color=COLORS[r["label"]],
                linewidth=1.5, label=f"{r['label']} ({r['time']:.0f}s)")
    ax.axhline(y=MAX_CUT, color="gray", linestyle=":", alpha=0.5, label=f"MaxCut={MAX_CUT}")
    ax.set(xlabel="Function Evaluations", ylabel="<C>", title="Convergence Curves")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 近似比
    ax = axes[1]
    bw = 0.35
    for i, r in enumerate(results):
        rn, re = r["history"][-1] / MAX_CUT, r["exact_final"] / MAX_CUT
        ax.bar(i - bw/2, rn, bw, color=COLORS[r["label"]], alpha=0.5, label="noisy" if i == 0 else "")
        ax.bar(i + bw/2, re, bw, color=COLORS[r["label"]], edgecolor="black", linewidth=1.2,
               label="exact" if i == 0 else "")
        ax.text(i - bw/2, rn + 0.015, f"{rn:.3f}", ha="center", fontsize=8)
        ax.text(i + bw/2, re + 0.015, f"{re:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(results))); ax.set_xticklabels([r["label"] for r in results])
    ax.set(ylabel="Approximation Ratio", title=f"Noisy vs Exact (shots={SHOTS_NOISY})", ylim=(0, 1.20))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 耗时
    ax = axes[2]
    times = [r["time"] for r in results]
    ax.bar(range(len(results)), times, color=[COLORS[r["label"]] for r in results], width=0.5)
    for i, t in enumerate(times):
        ax.text(i, t + max(times)*0.03, f"{t:.0f}s", ha="center", fontsize=10)
    ax.set_xticks(range(len(results))); ax.set_xticklabels([r["label"] for r in results])
    ax.set(ylabel="Seconds", title="Wall Time"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("qaoa_geo_optimizer_compare.png", dpi=130, facecolor="white")
    print("  saved: qaoa_geo_optimizer_compare.png", flush=True)


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
    plt.suptitle(f"Cut-Size Distribution (2000 shots)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("qaoa_geo_distributions.png", dpi=130, facecolor="white")
    print("  saved: qaoa_geo_distributions.png", flush=True)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    print(f"\nQAOA p={P}, irregular 12-vertex graph ({N_EDGES} edges, MaxCut={MAX_CUT})", flush=True)
    print(f"Projection noise: {SHOTS_NOISY} shots  |  Eval budget: {MAX_EVAL}\n", flush=True)

    results = [run_cma_es(), run_nelder_mead()]

    print(f"\n{'Optimizer':<14} {'noisy<C>':<10} {'exact<C>':<10} {'Ratio':<8} {'Time':<8} {'P(C='+str(MAX_CUT)+')':<10}")
    print("-" * 60)
    n_shots = sum(results[0]["cuts"].values())
    for r in results:
        p_max = r["cuts"].get(MAX_CUT, 0) / n_shots
        print(f"{r['label']:<14} {r['history'][-1]:<10.4f} {r['exact_final']:<10.4f} "
              f"{r['exact_final']/MAX_CUT:<8.4f} {r['time']:<8.1f} {p_max:<10.4f}")

    plot_summary(results)
    plot_distributions(results)
    plt.show()
