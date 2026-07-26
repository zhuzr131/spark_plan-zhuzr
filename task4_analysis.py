"""
Task 4: Initialization Strategy Comparison — Full Analysis
Produces: statistical table, box plot, convergence curves, LaTeX table.
"""
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Setup ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

for fname in fm.findSystemFonts():
    try:
        if any(kw in fname.lower() for kw in ['pingfang', 'heiti', 'songti', 'noto sans cjk', 'microsoft yahei']):
            prop = fm.FontProperties(fname=fname)
            plt.rcParams['font.family'] = prop.get_name()
            break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# --- Import project modules ---
from hamiltonian import graph_c4, graph_g6, graph_g9, maxcut_exact
from circuit import qaoa_circuit
from init_strategies import init_zero, init_random, init_linear_ramp, init_trotter
from optimizer import optimize_adam, optimize_bfgs

# ============================================================
# STEP 1: Run optimized experiments
# ============================================================
def run_task4_experiments():
    """Run Task 4 with better optimization settings."""
    graphs = [("C4", graph_c4()), ("G6", graph_g6()), ("G9", graph_g9())]
    strategies = {
        "zero": init_zero,
        "random": init_random,
        "linear_ramp": init_linear_ramp,
        "trotter": init_trotter,
    }
    results = []

    for gname, g in graphs:
        exact, _ = maxcut_exact(g)
        energy_fn = qaoa_circuit(graph=g, n_layers=1)  # p=1 for clean comparison

        for sname, init_fn in strategies.items():
            for seed in range(3):  # 3 restarts
                np.random.seed(seed)
                if sname == "random":
                    init_params = init_fn(1, seed=seed).astype(np.float32)
                else:
                    init_params = init_fn(1).astype(np.float32)

                # Use BFGS for reliable convergence
                opt_params, final_e, history = optimize_bfgs(
                    energy_fn, init_params, max_iter=500
                )

                results.append({
                    "graph": gname,
                    "init_strategy": sname,
                    "n_layers": 1,
                    "final_energy": float(final_e),
                    "exact_maxcut": int(exact),
                    "approximation_ratio": float(final_e / exact) if exact > 0 else 0,
                    "n_iterations": len(history),
                    "history": [float(h) for h in history],
                    "seed": seed,
                })
                print(f"  {gname} {sname} seed={seed}: AR={final_e/exact:.4f}, iters={len(history)}")

    path = os.path.join(RESULTS_DIR, "task4_clean.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# STEP 2: Statistical Summary
# ============================================================
def print_stats_table(results):
    agg = defaultdict(list)
    for d in results:
        agg[(d["graph"], d["init_strategy"])].append({
            "ar": d["approximation_ratio"],
            "iters": d["n_iterations"],
        })

    graphs = ["C4", "G6", "G9"]
    strats = ["zero", "random", "linear_ramp", "trotter"]

    print("\n" + "=" * 75)
    print("  Task 4 统计摘要：p=1 QAOA 初始化策略对比")
    print("=" * 75)
    header = f"{'图':<6} {'策略':<13} {'AR mean':>8} {'AR std':>8} {'AR min':>8} {'AR max':>8} {'Avg iters':>10}"
    print(header)
    print("-" * 75)
    for g in graphs:
        for s in strats:
            runs = agg[(g, s)]
            ars = [r["ar"] for r in runs]
            iters = [r["iters"] for r in runs]
            print(f"{g:<6} {s:<13} {np.mean(ars):>8.4f} {np.std(ars):>8.4f} "
                  f"{np.min(ars):>8.4f} {np.max(ars):>8.4f} {np.mean(iters):>10.1f}")

    return agg


# ============================================================
# STEP 3: Generate LaTeX table
# ============================================================
def generate_latex_table(results):
    agg = defaultdict(list)
    for d in results:
        agg[(d["graph"], d["init_strategy"])].append({
            "ar": d["approximation_ratio"],
            "iters": d["n_iterations"],
        })

    graphs = ["C4", "G6", "G9"]
    strats = ["zero", "random", "linear_ramp", "trotter"]
    strat_labels = {"zero": "Zero", "random": "Random", "linear_ramp": "Linear Ramp", "trotter": "Trotterized"}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{QAOA $p=1$ 初始化策略对比：近似比 $\pm$ 标准差}")
    lines.append(r"  \label{tab:init_comparison}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \textbf{Zero} & \textbf{Random} & \textbf{Linear Ramp} & \textbf{Trotterized} \\")
    lines.append(r"    \midrule")

    for g in graphs:
        vals = []
        for s in strats:
            ars = [r["ar"] for r in agg[(g, s)]]
            vals.append(f"${np.mean(ars):.3f} \\pm {np.std(ars):.3f}$")
        lines.append(f"    {g} & {' & '.join(vals)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "task4_latex_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX 表格已保存: results/task4_latex_table.tex")
    return latex


# ============================================================
# STEP 4: Visualization — Box Plot + Convergence Curves
# ============================================================
def plot_task4_visualizations(results):
    agg = defaultdict(list)
    for d in results:
        agg[(d["graph"], d["init_strategy"])].append(d)

    graphs = ["C4", "G6", "G9"]
    strats = ["zero", "random", "linear_ramp", "trotter"]
    colors = {"zero": "#E8E8E8", "random": "#534AB7", "linear_ramp": "#7B6FC8", "trotter": "#3C3489"}
    strat_cn = {"zero": "零初始化", "random": "随机初始化",
                "linear_ramp": "线性渐变", "trotter": "Trotter化"}

    # --- Figure 1: Box plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for gi, (gname, ax) in enumerate(zip(graphs, axes)):
        data_groups = []
        labels = []
        for s in strats:
            ars = [r["approximation_ratio"] for r in agg[(gname, s)]]
            data_groups.append(ars)
            labels.append(strat_cn[s])

        bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, s in zip(bp['boxes'], strats):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.8)
        for median in bp['medians']:
            median.set_color('#26215C')
            median.set_linewidth(2)

        # Mark exact optimum
        exact = agg[(gname, "zero")][0]["exact_maxcut"]
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f"{gname} (MaxCut={exact})", fontsize=13, fontweight='bold')
        if gi == 0:
            ax.set_ylabel("近似比 (Approximation Ratio)", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=9)

    fig.suptitle("Task 4: 初始化策略对比 (p=1 QAOA, 3 restarts)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGS_DIR, f"task4_boxplot.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Convergence curves (one subplot per graph) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for gi, (gname, ax) in enumerate(zip(graphs, axes)):
        for s in strats:
            # Plot all restarts for each strategy
            all_hist = [r["history"] for r in agg[(gname, s)]]
            max_len = max(len(h) for h in all_hist)
            # Pad shorter histories with their final value
            padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in all_hist])
            mean_hist = np.mean(padded, axis=0)
            std_hist = np.std(padded, axis=0)
            x = np.arange(len(mean_hist))
            ax.plot(x, mean_hist, color=colors[s], label=strat_cn[s], linewidth=1.8)
            ax.fill_between(x, mean_hist - std_hist, mean_hist + std_hist,
                            color=colors[s], alpha=0.15)

        exact = agg[(gname, "zero")][0]["exact_maxcut"]
        ax.axhline(y=exact, color='gray', linestyle='--', alpha=0.4, label=f'MaxCut={exact}')
        ax.set_title(f"{gname}", fontsize=13, fontweight='bold')
        ax.set_xlabel("迭代次数", fontsize=10)
        if gi == 0:
            ax.set_ylabel("能量期望值 ⟨C⟩", fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle("收敛曲线对比：不同初始化策略的优化轨迹", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGS_DIR, f"task4_convergence.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 3: Bar chart with error bars ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(graphs))
    w = 0.2
    for i, s in enumerate(strats):
        means = [np.mean([r["approximation_ratio"] for r in agg[(g, s)]]) for g in graphs]
        stds = [np.std([r["approximation_ratio"] for r in agg[(g, s)]]) for g in graphs]
        bars = ax.bar(x + i * w, means, w, label=strat_cn[s], color=colors[s],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.errorbar(x + i * w, means, yerr=stds, fmt='none', ecolor='#26215C',
                    capsize=4, linewidth=1)

        # Add value labels
        for bar, mean in zip(bars, means):
            if mean > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{mean:.3f}", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(graphs, fontsize=12)
    ax.set_ylabel("近似比 (Approximation Ratio)", fontsize=11)
    ax.set_title("Task 4: 初始化策略对比 (p=1 QAOA, 3 restarts)", fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='最优值')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGS_DIR, f"task4_barchart.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n可视化文件已保存到 results/figures/")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=== Task 4: 初始化策略分析 ===")
    print("正在运行实验 (p=1, BFGS, 3 restarts)...")

    results = run_task4_experiments()

    print_stats_table(results)
    latex = generate_latex_table(results)
    plot_task4_visualizations(results)

    print("\n=== 输出清单 ===")
    print("  results/task4_clean.json         — 原始数据")
    print("  results/task4_latex_table.tex     — LaTeX 表格")
    print("  results/figures/task4_boxplot.*   — 箱线图")
    print("  results/figures/task4_convergence.* — 收敛曲线")
    print("  results/figures/task4_barchart.*  — 柱状图")
    print("\nDone!")
