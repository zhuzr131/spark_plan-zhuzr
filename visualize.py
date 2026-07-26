"""
Visualization for QAOA experiment results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGS_DIR = os.path.join(os.path.dirname(__file__), "results", "figures")


def setup_chinese_font():
    """Try to set up a Chinese-capable font."""
    for fname in fm.findSystemFonts():
        try:
            prop = fm.FontProperties(fname=fname)
            if any(kw in fname.lower() for kw in ['pingfang', 'heiti', 'songti', 'noto sans cjk', 'microsoft yahei']):
                plt.rcParams['font.family'] = prop.get_name()
                return
        except Exception:
            continue
    # Fallback: use sans-serif (Chinese may not render, but plots are still functional)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_font()
os.makedirs(FIGS_DIR, exist_ok=True)


def plot_task2_p1_baseline():
    """Plot Task 2: p=1 baseline results."""
    path = os.path.join(RESULTS_DIR, "task2_p1_baseline.json")
    if not os.path.exists(path):
        print("task2_p1_baseline.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    names = [d["graph"] for d in data]
    approx_ratios = [d["approximation_ratio"] for d in data]
    exact_cuts = [d["exact_maxcut"] for d in data]
    final_energies = [d["final_energy"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: approximation ratio
    colors = ['#534AB7', '#3C3489', '#7B6FC8']
    axes[0].bar(names, approx_ratios, color=colors, alpha=0.85)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Approximation Ratio")
    axes[0].set_title("QAOA p=1: Approximation Ratio")
    axes[0].set_ylim(0, 1.1)
    for i, (bar, ar) in enumerate(zip(axes[0].patches, approx_ratios)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{ar:.3f}", ha='center', va='bottom', fontsize=9)

    # Right: Energy vs Exact
    x = np.arange(len(names))
    w = 0.35
    axes[1].bar(x - w/2, final_energies, w, label="QAOA Energy", color='#534AB7', alpha=0.85)
    axes[1].bar(x + w/2, exact_cuts, w, label="Exact MaxCut", color='#CCCCCC', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("MaxCut Value")
    axes[1].set_title("QAOA vs Exact")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "task2_p1_baseline.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "task2_p1_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task2 plot")


def plot_task3_p_scan():
    """Plot Task 3: performance vs depth p."""
    path = os.path.join(RESULTS_DIR, "task3_p_scan.json")
    if not os.path.exists(path):
        print("task3_p_scan.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Group by graph
    graphs = {}
    for d in data:
        name = d["graph"]
        if name not in graphs:
            graphs[name] = {"p": [], "ratio": [], "energy": []}
        graphs[name]["p"].append(d["n_layers"])
        graphs[name]["ratio"].append(d["approximation_ratio"])
        graphs[name]["energy"].append(d["final_energy"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {"C4": "#534AB7", "G6": "#3C3489", "G9": "#7B6FC8"}
    markers = {"C4": "o", "G6": "s", "G9": "^"}

    for name, gdata in graphs.items():
        axes[0].plot(gdata["p"], gdata["ratio"], marker=markers[name], color=colors[name],
                     label=name, linewidth=2, markersize=8)
        axes[1].plot(gdata["p"], gdata["energy"], marker=markers[name], color=colors[name],
                     label=name, linewidth=2, markersize=8)

    axes[0].set_xlabel("Depth p")
    axes[0].set_ylabel("Approximation Ratio")
    axes[0].set_title("Approximation Ratio vs Depth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Depth p")
    axes[1].set_ylabel("MaxCut Energy")
    axes[1].set_title("Energy vs Depth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "task3_p_scan.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "task3_p_scan.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task3 plot")


def plot_task4_init_comparison():
    """Plot Task 4: initialization strategy comparison."""
    path = os.path.join(RESULTS_DIR, "task4_init_comparison.json")
    if not os.path.exists(path):
        print("task4_init_comparison.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Aggregate by (graph, strategy)
    from collections import defaultdict
    agg = defaultdict(list)
    for d in data:
        key = (d["graph"], d["init_strategy"])
        agg[key].append(d["approximation_ratio"])

    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["zero", "random", "linear_ramp"]
    graph_names = sorted(set(d["graph"] for d in data))
    colors = {"zero": "#534AB7", "random": "#3C3489", "linear_ramp": "#7B6FC8"}
    x = np.arange(len(graph_names))
    w = 0.25

    for i, strat in enumerate(strategies):
        means = []
        stds = []
        for gname in graph_names:
            vals = agg.get((gname, strat), [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + i * w, means, w, yerr=stds, label=strat, color=colors[strat],
               alpha=0.85, capsize=4)

    ax.set_xticks(x + w)
    ax.set_xticklabels(graph_names)
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Init Strategy Comparison (p=2)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "task4_init_comparison.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "task4_init_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task4 plot")


def plot_task5_optimizer_comparison():
    """Plot Task 5: optimizer comparison."""
    path = os.path.join(RESULTS_DIR, "task5_optimizer_comparison.json")
    if not os.path.exists(path):
        print("task5_optimizer_comparison.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    from collections import defaultdict
    agg = defaultdict(list)
    conv = defaultdict(list)
    for d in data:
        key = (d["graph"], d["optimizer"])
        agg[key].append(d["approximation_ratio"])
        conv[key].append(d["n_iterations"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    optimizers = ["adam", "bfgs", "cobyla", "spsa"]
    graph_names = sorted(set(d["graph"] for d in data))
    colors = {"adam": "#534AB7", "bfgs": "#3C3489", "cobyla": "#7B6FC8", "spsa": "#A89DE0"}
    x = np.arange(len(graph_names))
    w = 0.2

    for i, opt in enumerate(optimizers):
        means = [np.mean(agg.get((g, opt), [0])) for g in graph_names]
        axes[0].bar(x + i * w, means, w, label=opt, color=colors[opt], alpha=0.85)
        iters = [np.mean(conv.get((g, opt), [0])) for g in graph_names]
        axes[1].bar(x + i * w, iters, w, label=opt, color=colors[opt], alpha=0.85)

    axes[0].set_xticks(x + 1.5 * w)
    axes[0].set_xticklabels(graph_names)
    axes[0].set_ylabel("Approximation Ratio")
    axes[0].set_title("Optimizer: Solution Quality")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].set_xticks(x + 1.5 * w)
    axes[1].set_xticklabels(graph_names)
    axes[1].set_ylabel("Iterations to Converge")
    axes[1].set_title("Optimizer: Convergence Speed")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "task5_optimizer_comparison.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "task5_optimizer_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved task5 plot")


def plot_dla_results(dla_data: list[dict], filename: str = "dla_analysis"):
    """Plot DLA dimension vs graph properties."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    names = [d["name"] for d in dla_data]
    dims = [d["dla_dim"] for d in dla_data]
    n_nodes = [d["n"] for d in dla_data]
    n_edges = [d["edges"] for d in dla_data]
    max_dim = [4**n for n in n_nodes]

    # Left: DLA dim vs n (with 4^n reference)
    axes[0].scatter(n_nodes, dims, s=80, color='#534AB7', alpha=0.8)
    axes[0].plot(sorted(n_nodes), [4**n for n in sorted(n_nodes)],
                 '--', color='gray', alpha=0.5, label=r'$4^n$ (max)')
    axes[0].set_xlabel("Number of Qubits n")
    axes[0].set_ylabel("DLA Dimension")
    axes[0].set_title("DLA Dimension vs Qubit Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Right: DLA dim vs edges
    axes[1].scatter(n_edges, dims, s=80, color='#3C3489', alpha=0.8)
    for i, name in enumerate(names):
        axes[1].annotate(name, (n_edges[i], dims[i]), fontsize=7,
                         xytext=(5, 5), textcoords='offset points', alpha=0.7)
    axes[1].set_xlabel("Number of Edges")
    axes[1].set_ylabel("DLA Dimension")
    axes[1].set_title("DLA Dimension vs Graph Density")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{filename}.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, f"{filename}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved DLA plot: {filename}")


def generate_all_plots():
    """Generate all plots from saved experiment results."""
    plot_task2_p1_baseline()
    plot_task3_p_scan()
    plot_task4_init_comparison()
    plot_task5_optimizer_comparison()


if __name__ == "__main__":
    generate_all_plots()
