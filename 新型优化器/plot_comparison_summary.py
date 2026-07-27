#!/usr/bin/env python3
"""从多尺度测试结果生成汇总对比图表。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 数据 ──────────────────────────────────────────────────────────────
# 格式: (节点数, {方法: (ratio_mean, ratio_std, time_mean)})

# 14 节点（无 Plain Adam 的旧数据）
data_14 = {
    "Original Valley":     (0.7864, 0.0090, 14.24),
    "Layer-by-Layer":      (0.7835, 0.0206, 14.47),
    "Noise-aware V+P":     (0.8001, 0.0064,  6.20),
}

# 16 节点（有 Plain Adam）
data_16 = {
    "Plain Adam":          (0.7711, 0.0186, 28.35),
    "Original Valley":     (0.7693, 0.0182, 27.66),
    "Layer-by-Layer":      (0.7813, 0.0151, 28.12),
    "Noise-aware V+P":     (0.7879, 0.0136, 12.01),
}

# 18 节点（无 Plain Adam）
data_18 = {
    "Original Valley":     (0.7637, 0.0184,  89.95),
    "Layer-by-Layer":      (0.7797, 0.0144,  86.91),
    "Noise-aware V+P":     (0.7972, 0.0055,  42.52),
}

# 19 节点（有 Plain Adam）
data_19 = {
    "Plain Adam":          (0.7416, 0.0253, 171.18),
    "Original Valley":     (0.7394, 0.0157, 165.16),
    "Layer-by-Layer":      (0.7477, 0.0176, 150.61),
    "Noise-aware V+P":     (0.7647, 0.0173,  62.75),
}

ALL_DATA = {14: data_14, 16: data_16, 18: data_18, 19: data_19}
NODES = sorted(ALL_DATA.keys())  # [14, 16, 18, 19]

METHODS = ["Plain Adam", "Original Valley", "Layer-by-Layer", "Noise-aware V+P"]
COLORS = {
    "Plain Adam":          "#2ecc71",
    "Original Valley":     "#5d9cec",
    "Layer-by-Layer":      "#f39c12",
    "Noise-aware V+P":     "#e74c3c",
}
MARKERS = {
    "Plain Adam":          "s",
    "Original Valley":     "o",
    "Layer-by-Layer":      "D",
    "Noise-aware V+P":     "p",
}
LINESTYLES = {
    "Plain Adam":          "--",
    "Original Valley":     "-.",
    "Layer-by-Layer":      ":",
    "Noise-aware V+P":     "-",
}

# ── 图 1：近似比 vs 节点数 ──────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
for method in METHODS:
    xs, ys, es = [], [], []
    for n in NODES:
        if method in ALL_DATA[n]:
            r, s, _ = ALL_DATA[n][method]
            xs.append(n)
            ys.append(r)
            es.append(s)
    if xs:
        ax1.errorbar(xs, ys, yerr=es, marker=MARKERS[method],
                     color=COLORS[method], linewidth=2.0, capsize=5,
                     linestyle=LINESTYLES[method], label=method,
                     markersize=9)

ax1.set_xlabel("Number of nodes (qubits)", fontsize=12)
ax1.set_ylabel("Approximation ratio", fontsize=12)
ax1.set_title("Solution Quality vs Problem Size\n"
              r"(QAOA p=2, $\sigma$=0.01, 5 trials per point)",
              fontsize=13)
ax1.legend(fontsize=10, loc="lower left")
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.set_xticks(NODES)
fig1.tight_layout()
fig1.savefig("/Users/zhuzhengrong/Desktop/comparison_ratio_vs_nodes.png",
             dpi=200)
plt.close(fig1)

# ── 图 2：耗时 vs 节点数 ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
for method in METHODS:
    xs, ys = [], []
    for n in NODES:
        if method in ALL_DATA[n]:
            _, _, t = ALL_DATA[n][method]
            xs.append(n)
            ys.append(t)
    if xs:
        ax2.plot(xs, ys, marker=MARKERS[method],
                 color=COLORS[method], linewidth=2.0,
                 linestyle=LINESTYLES[method], label=method,
                 markersize=9)

ax2.set_xlabel("Number of nodes (qubits)", fontsize=12)
ax2.set_ylabel("Runtime (seconds)", fontsize=12)
ax2.set_title("Runtime vs Problem Size\n"
              r"(QAOA p=2, $\sigma$=0.01, statevector simulation, ~3300 noisy evals)",
              fontsize=13)
ax2.legend(fontsize=10, loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.set_xticks(NODES)
fig2.tight_layout()
fig2.savefig("/Users/zhuzhengrong/Desktop/comparison_runtime_vs_nodes.png",
             dpi=200)
plt.close(fig2)

# ── 图 3：稳定性（std）vs 节点数 ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5.5))
for method in METHODS:
    xs, ys = [], []
    for n in NODES:
        if method in ALL_DATA[n]:
            _, s, _ = ALL_DATA[n][method]
            xs.append(n)
            ys.append(s)
    if xs:
        ax3.plot(xs, ys, marker=MARKERS[method],
                 color=COLORS[method], linewidth=2.0,
                 linestyle=LINESTYLES[method], label=method,
                 markersize=9)

ax3.set_xlabel("Number of nodes (qubits)", fontsize=12)
ax3.set_ylabel("Standard deviation of ratio", fontsize=12)
ax3.set_title("Optimization Stability vs Problem Size\n"
              r"(lower std = more consistent across trials, $\sigma$=0.01)",
              fontsize=13)
ax3.legend(fontsize=10, loc="upper left")
ax3.grid(True, linestyle="--", alpha=0.35)
ax3.set_xticks(NODES)
fig3.tight_layout()
fig3.savefig("/Users/zhuzhengrong/Desktop/comparison_stability_vs_nodes.png",
             dpi=200)
plt.close(fig3)

# ── 图 4：16-节点全景仪表盘（2×2，最完整数据）─────────────────────
data_16_all = {
    "Plain Adam":      (0.7711, 0.0186, 28.35, 761),
    "Original Valley": (0.7693, 0.0182, 27.66, 710),
    "Layer-by-Layer":  (0.7813, 0.0151, 28.12, 1161),
    "Noise-aware V+P": (0.7879, 0.0136, 12.01, 483),
}
names_16 = list(data_16_all.keys())
x16 = np.arange(len(names_16))

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上：近似比
ratios_16 = [data_16_all[m][0] for m in names_16]
stds_16   = [data_16_all[m][1] for m in names_16]
bars = axes[0, 0].bar(x16, ratios_16, yerr=stds_16, capsize=6,
                      color=[COLORS[m.replace("V+P", "V+P")] for m in names_16],
                      edgecolor="white", linewidth=0.8)
# 标注数值
for i, (r, s) in enumerate(zip(ratios_16, stds_16)):
    axes[0, 0].text(i, r + s + 0.002, f"{r:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
axes[0, 0].set_xticks(x16, [n.replace("V+P", "V+P").replace(" ", "\n")
                            for n in names_16], fontsize=9)
axes[0, 0].set_ylabel("Approximation ratio", fontsize=11)
axes[0, 0].set_title("Solution quality (16 nodes)", fontsize=12, fontweight="bold")
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

# 右上：耗时
times_16 = [data_16_all[m][2] for m in names_16]
bars = axes[0, 1].bar(x16, times_16,
                      color=[COLORS[m.replace("V+P", "V+P")] for m in names_16],
                      edgecolor="white", linewidth=0.8)
for i, t in enumerate(times_16):
    axes[0, 1].text(i, t + 0.3, f"{t:.1f}s", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
axes[0, 1].set_xticks(x16, [n.replace(" ", "\n") for n in names_16], fontsize=9)
axes[0, 1].set_ylabel("Runtime (s)", fontsize=11)
axes[0, 1].set_title("Runtime (16 nodes)", fontsize=12, fontweight="bold")
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.35)

# 左下：Aer 作业次数
aer_16 = [data_16_all[m][3] for m in names_16]
bars = axes[1, 0].bar(x16, aer_16,
                      color=[COLORS[m.replace("V+P", "V+P")] for m in names_16],
                      edgecolor="white", linewidth=0.8)
for i, a in enumerate(aer_16):
    axes[1, 0].text(i, a + 5, str(a), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
axes[1, 0].set_xticks(x16, [n.replace(" ", "\n") for n in names_16], fontsize=9)
axes[1, 0].set_ylabel("Aer estimator.run calls", fontsize=11)
axes[1, 0].set_title("Backend job calls (16 nodes)", fontsize=12, fontweight="bold")
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.35)

# 右下：稳定性
stds_16_vals = [data_16_all[m][1] for m in names_16]
bars = axes[1, 1].bar(x16, stds_16_vals,
                      color=[COLORS[m.replace("V+P", "V+P")] for m in names_16],
                      edgecolor="white", linewidth=0.8)
for i, s in enumerate(stds_16_vals):
    axes[1, 1].text(i, s + 0.0003, f"{s:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
axes[1, 1].set_xticks(x16, [n.replace(" ", "\n") for n in names_16], fontsize=9)
axes[1, 1].set_ylabel("Std dev of ratio", fontsize=11)
axes[1, 1].set_title("Stability across trials (16 nodes)", fontsize=12, fontweight="bold")
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)

fig4.suptitle("16-Node Max-Cut QAOA: Four-Optimizer Full Comparison\n"
              r"p=2, $\sigma$=0.01, 5 trials, ~3300 noisy evaluations budget each",
              fontsize=14, fontweight="bold", y=1.01)
fig4.tight_layout()
fig4.savefig("/Users/zhuzhengrong/Desktop/comparison_16node_dashboard.png",
             dpi=200)
plt.close(fig4)

# ── 图 5：多规模总览（2×1）─────────────────────────────────────────
fig5, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))

# 左：近似比
for method in METHODS:
    xs, ys, es = [], [], []
    for n in NODES:
        if method in ALL_DATA[n]:
            r, s, _ = ALL_DATA[n][method]
            xs.append(n)
            ys.append(r)
            es.append(s)
    if xs:
        ax_a.errorbar(xs, ys, yerr=es, marker=MARKERS[method],
                      color=COLORS[method], linewidth=2.2, capsize=5,
                      linestyle=LINESTYLES[method], label=method,
                      markersize=10)
ax_a.set_xlabel("Nodes (qubits)", fontsize=12)
ax_a.set_ylabel("Approximation ratio", fontsize=12)
ax_a.set_title("Quality vs scale", fontsize=13, fontweight="bold")
ax_a.legend(fontsize=9, loc="lower left")
ax_a.grid(True, linestyle="--", alpha=0.35)
ax_a.set_xticks(NODES)

# 右：运行时
for method in METHODS:
    xs, ys = [], []
    for n in NODES:
        if method in ALL_DATA[n]:
            _, _, t = ALL_DATA[n][method]
            xs.append(n)
            ys.append(t)
    if xs:
        ax_b.plot(xs, ys, marker=MARKERS[method],
                  color=COLORS[method], linewidth=2.2,
                  linestyle=LINESTYLES[method], label=method,
                  markersize=10)
ax_b.set_xlabel("Nodes (qubits)", fontsize=12)
ax_b.set_ylabel("Runtime (s)", fontsize=12)
ax_b.set_title("Speed vs scale", fontsize=13, fontweight="bold")
ax_b.legend(fontsize=9, loc="upper left")
ax_b.grid(True, linestyle="--", alpha=0.35)
ax_b.set_xticks(NODES)

fig5.suptitle("Multi-Scale Summary: 14 / 16 / 18 / 19 Nodes\n"
              r"QAOA p=2, $\sigma$=0.01, ~3300 noisy evals, statevector simulation, 5 trials each",
              fontsize=14, fontweight="bold")
fig5.tight_layout()
fig5.savefig("/Users/zhuzhengrong/Desktop/comparison_multiscale_overview.png",
             dpi=200)
plt.close(fig5)

# ── 图 6：速度放大图（16 与 19 节点并排，突出 Noise-aware 速度优势）─
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, n_nodes, title in [
    (ax6a, 16, "16 nodes"), 
    (ax6b, 19, "19 nodes")
]:
    data = ALL_DATA[n_nodes]
    methods_here = [m for m in METHODS if m in data]
    xs = np.arange(len(methods_here))
    times_here = [data[m][2] for m in methods_here]
    ratios_here = [data[m][0] for m in methods_here]
    colors_here = [COLORS[m] for m in methods_here]
    
    ax.bar(xs, times_here, color=colors_here, edgecolor="white", linewidth=0.8)
    for i, (t, r) in enumerate(zip(times_here, ratios_here)):
        ax.text(i, t + max(times_here)*0.02, 
                f"{t:.1f}s\nratio={r:.4f}", 
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs, [m.replace("V+P", "V+P").replace("Noise-aware", "N-aware\n")
                       .replace("Original", "Orig\n").replace("Layer-by", "LbL\n")
                       for m in methods_here], fontsize=9)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

fig6.suptitle("Speed Advantage: Noise-aware Valley + Planck vs Baselines\n"
              r"$\sigma$=0.01, ~3300 noisy evals budget each",
              fontsize=13, fontweight="bold")
fig6.tight_layout()
fig6.savefig("/Users/zhuzhengrong/Desktop/comparison_speed_advantage.png",
             dpi=200)
plt.close(fig6)

print("Done. Generated 6 figures:")
print("  1. comparison_ratio_vs_nodes.png      — 近似比-规模趋势")
print("  2. comparison_runtime_vs_nodes.png     — 耗时-规模趋势")
print("  3. comparison_stability_vs_nodes.png   — 稳定性-规模趋势")
print("  4. comparison_16node_dashboard.png     — 16 节点全景仪表盘")
print("  5. comparison_multiscale_overview.png  — 多规模质量+速度总览")
print("  6. comparison_speed_advantage.png      — 速度优势特写")
