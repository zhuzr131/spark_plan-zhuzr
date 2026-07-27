#!/usr/bin/env python3
"""
同一加权 Max-Cut QAOA 问题上的三架构抗噪对比：

1. Original Valley：规则粗网格 → 只诊断投影方差 → 多起点 Adam → Adam polish。
2. LBL：p=1 → p=2 分层训练 → 全参数 Adam polish。
3. Noise-aware Valley + Planck-Adam：Sobol + LCB → 噪声校正投影候选 →
   同方向重复 SPSA + Planck 置信度滤波 → 高重复确认。

三者使用同一张图、同一 QAOA 深度和相近的带噪目标评估预算。
无噪声精确期望值只用于记录最终性能，不参与前两个基线的决策。

依赖同目录文件：maxcut_noise_aware_valley_planck.py

运行：
    /usr/bin/python3 maxcut_three_architectures_comparison.py
快速验证：
    /usr/bin/python3 maxcut_three_architectures_comparison.py --fast
"""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import maxcut_noise_aware_valley_planck as navp

warnings.filterwarnings("ignore")


CONFIG: dict[str, Any] = {
    "trials": 5,
    "noise_sigma": 0.020,
    "base_seed": 44021,
    # 问题规模：16 节点 → 16 量子比特（statevector 仿真）。
    "num_nodes": 17,
    # Plain Adam，约 3420 次带噪评估（4 起点×70 + polish 100，每迭代 9 次）。
    "adam_restarts": 4,
    "adam_iterations": 70,
    "adam_polish_iterations": 100,
    "adam_learning_rate": 0.05,
    # Original Valley，约 3303 次带噪评估。
    "valley_coarse_per_dim": 3,
    "valley_top": 4,
    "valley_scan_directions": 8,
    "valley_scan_samples": 9,
    "valley_scan_radius": 0.75,
    "valley_refine_iterations": 70,
    "valley_polish_iterations": 70,
    "valley_learning_rate": 0.05,
    "coordinate_h": 0.02,
    # LBL，约 3300 次带噪评估。
    "lbl_restarts": 4,
    "lbl_layer_iterations": 60,
    "lbl_polish_iterations": 100,
    "lbl_learning_rate": 0.06,
    # 设为 50 后抗噪方法约 3336 次评估，与两个约 3300 次的基线接近。
    "noise_aware_local_iterations": 50,
    "output_dir": "maxcut_three_architectures_results",
}

METHODS = (
    "Plain Adam",
    "Original Valley",
    "Layer-by-Layer",
    "Noise-aware Valley + Planck",
)
COLORS = {
    "Plain Adam": "#2ecc71",
    "Original Valley": "#5d9cec",
    "Layer-by-Layer": "#f39c12",
    "Noise-aware Valley + Planck": "#e74c3c",
}
SHORT_LABELS = {
    "Plain Adam": "Plain\nAdam",
    "Original Valley": "Original\nValley",
    "Layer-by-Layer": "LBL",
    "Noise-aware Valley + Planck": "Noise-aware\nValley+Planck",
}


def noisy_coordinate_gradient(oracle, theta, active_indices, h, reps):
    """用单次带噪中心差分估计指定坐标梯度；目标为最大化切割值。"""
    theta = np.asarray(theta, dtype=float)
    points = []
    for index in active_indices:
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += h
        minus[index] -= h
        points.extend((plus, minus))
    means, _, _, _ = oracle.sample(np.asarray(points), reps)
    gradient = np.zeros_like(theta)
    for position, index in enumerate(active_indices):
        gradient[index] = (
            means[2 * position] - means[2 * position + 1]
        ) / (2.0 * h)
    return gradient


def naive_adam_ascent(
    oracle, initial_point, active_indices, iterations, learning_rate, reps,
    start_step=0,
):
    """原始基线使用的带噪 Adam；按单次带噪观测保存最优参数。"""
    theta = np.asarray(initial_point, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best_point = theta.copy()
    best_noisy = -np.inf
    history = []

    for iteration in range(1, iterations + 1):
        gradient = noisy_coordinate_gradient(
            oracle,
            theta,
            active_indices,
            float(CONFIG["coordinate_h"]),
            reps,
        )
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * gradient ** 2
        m_hat = m / (1.0 - beta1 ** iteration)
        v_hat = v / (1.0 - beta2 ** iteration)
        theta += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        theta = navp.wrap_parameters(theta, oracle.exact.reps)

        means, _, true_values, _ = oracle.sample(theta[None, :], reps)
        noisy_value = float(means[0])
        if noisy_value > best_noisy:
            best_noisy = noisy_value
            best_point = theta.copy()
        history.append({
            "step": start_step + iteration,
            "noisy": noisy_value,
            "true": float(true_values[0]),
            "evals": oracle.noisy_evaluations,
        })
    return best_point, best_noisy, history


def plain_adam_search(exact_objective, sigma, seed):
    """纯 Adam 基线：随机多起点，全参数带噪 Adam 上升，取最优后 polish。"""
    start_time = time.time()
    oracle = navp.NoisyCutOracle(exact_objective, sigma, seed)
    calls_before = exact_objective.estimator_calls
    points_before = exact_objective.point_evaluations
    reps = exact_objective.reps
    dimensions = exact_objective.num_parameters
    rng = np.random.default_rng(seed + 71)

    restart_points = []
    restart_scores = []
    history = []
    for restart in range(int(CONFIG["adam_restarts"])):
        betas = rng.uniform(0.0, np.pi, size=reps)
        gammas = rng.uniform(0.0, 2.0 * np.pi, size=reps)
        point = np.concatenate((betas, gammas))
        trained, score, local_history = naive_adam_ascent(
            oracle,
            point,
            np.arange(dimensions),
            int(CONFIG["adam_iterations"]),
            float(CONFIG["adam_learning_rate"]),
            1,
        )
        for row in local_history:
            row["phase"] = f"adam-restart-{restart}"
        history.extend(local_history)
        restart_points.append(trained)
        restart_scores.append(score)

    best_index = int(np.argmax(restart_scores))
    polished, noisy_score, polish_history = naive_adam_ascent(
        oracle,
        restart_points[best_index],
        np.arange(dimensions),
        int(CONFIG["adam_polish_iterations"]),
        float(CONFIG["adam_learning_rate"]),
        1,
    )
    for row in polish_history:
        row["phase"] = "adam-polish"
    history.extend(polish_history)
    true_value = float(exact_objective(polished)[0])
    return {
        "point": polished,
        "true_cut_fraction": true_value,
        "noisy_score": noisy_score,
        "evals": oracle.noisy_evaluations,
        "estimator_calls": exact_objective.estimator_calls - calls_before,
        "point_evaluations": exact_objective.point_evaluations - points_before,
        "runtime": time.time() - start_time,
        "history": history,
        "diagnostic": {},
    }


def original_valley_search(exact_objective, sigma, seed):
    """复现原文件的 Valley 架构；投影方差仅诊断，不生成候选。"""
    start_time = time.time()
    oracle = navp.NoisyCutOracle(exact_objective, sigma, seed)
    calls_before = exact_objective.estimator_calls
    points_before = exact_objective.point_evaluations
    reps = exact_objective.reps
    dimensions = exact_objective.num_parameters
    grid_size = int(CONFIG["valley_coarse_per_dim"])

    beta_grid = np.linspace(0.0, np.pi, grid_size + 2)[1:-1]
    gamma_grid = np.linspace(0.0, 2.0 * np.pi, grid_size + 2)[1:-1]
    coordinate_grids = [beta_grid] * reps + [gamma_grid] * reps
    mesh = np.meshgrid(*coordinate_grids, indexing="ij")
    coarse_points = np.stack([axis.ravel() for axis in mesh], axis=1)
    coarse_means, _, coarse_true, _ = oracle.sample(coarse_points, 1)
    top_indices = np.argsort(coarse_means)[-int(CONFIG["valley_top"]):][::-1]
    top_points = coarse_points[top_indices]

    # 原始思想：在 noisy 最佳点周围扫描方差，但不使用方向最佳位移。
    rng = np.random.default_rng(seed + 11)
    directions = navp.random_orthogonal_directions(
        dimensions, int(CONFIG["valley_scan_directions"]), rng
    )
    offsets = np.linspace(
        -float(CONFIG["valley_scan_radius"]),
        float(CONFIG["valley_scan_radius"]),
        int(CONFIG["valley_scan_samples"]),
    )
    projection_variances = []
    for direction in directions:
        line_points = navp.wrap_parameters(
            top_points[0][None, :] + offsets[:, None] * direction[None, :], reps
        )
        means, _, _, _ = oracle.sample(line_points, 1)
        projection_variances.append(float(np.var(means)))

    refined_points = []
    refined_scores = []
    history = []
    for start_index, point in enumerate(top_points):
        refined, score, local_history = naive_adam_ascent(
            oracle,
            point,
            np.arange(dimensions),
            int(CONFIG["valley_refine_iterations"]),
            float(CONFIG["valley_learning_rate"]),
            1,
        )
        refined_points.append(refined)
        refined_scores.append(score)
        for row in local_history:
            row["phase"] = f"refine-{start_index}"
        history.extend(local_history)

    best_index = int(np.argmax(refined_scores))
    polished, noisy_score, polish_history = naive_adam_ascent(
        oracle,
        refined_points[best_index],
        np.arange(dimensions),
        int(CONFIG["valley_polish_iterations"]),
        float(CONFIG["valley_learning_rate"]),
        1,
    )
    for row in polish_history:
        row["phase"] = "polish"
    history.extend(polish_history)
    true_value = float(exact_objective(polished)[0])
    return {
        "point": polished,
        "true_cut_fraction": true_value,
        "noisy_score": noisy_score,
        "evals": oracle.noisy_evaluations,
        "estimator_calls": exact_objective.estimator_calls - calls_before,
        "point_evaluations": exact_objective.point_evaluations - points_before,
        "runtime": time.time() - start_time,
        "history": history,
        "diagnostic": {
            "coarse_best_true": float(np.max(coarse_true[top_indices])),
            "max_projection_variance": float(np.max(projection_variances)),
        },
    }


def layer_by_layer_search(operator, sigma, seed):
    """复现传统 LBL：逐层只训练新参数，最后全参数 polish。"""
    start_time = time.time()
    rng = np.random.default_rng(seed + 31)
    p1_objective = navp.ExactQAOACut(operator, 1)
    p2_objective = navp.ExactQAOACut(operator, 2)
    p1_oracle = navp.NoisyCutOracle(p1_objective, sigma, seed)
    p2_oracle = navp.NoisyCutOracle(p2_objective, sigma, seed + 1)
    fixed_betas = []
    fixed_gammas = []
    history = []

    for layer in (1, 2):
        objective = p1_objective if layer == 1 else p2_objective
        oracle = p1_oracle if layer == 1 else p2_oracle
        layer_best_score = -np.inf
        layer_best_point = None
        for restart in range(int(CONFIG["lbl_restarts"])):
            betas = np.asarray(fixed_betas + [rng.uniform(0.0, np.pi)])
            gammas = np.asarray(fixed_gammas + [rng.uniform(0.0, 2.0 * np.pi)])
            point = np.concatenate((betas, gammas))
            active_indices = np.asarray([layer - 1, 2 * layer - 1])
            trained, score, local_history = naive_adam_ascent(
                oracle,
                point,
                active_indices,
                int(CONFIG["lbl_layer_iterations"]),
                float(CONFIG["lbl_learning_rate"]),
                1,
            )
            for row in local_history:
                row["phase"] = f"layer-{layer}-restart-{restart}"
            history.extend(local_history)
            if score > layer_best_score:
                layer_best_score = score
                layer_best_point = trained.copy()
        fixed_betas = list(layer_best_point[:layer])
        fixed_gammas = list(layer_best_point[layer:])

    full_point = np.concatenate((np.asarray(fixed_betas), np.asarray(fixed_gammas)))
    polished, noisy_score, polish_history = naive_adam_ascent(
        p2_oracle,
        full_point,
        np.arange(4),
        int(CONFIG["lbl_polish_iterations"]),
        float(CONFIG["lbl_learning_rate"]),
        1,
    )
    for row in polish_history:
        row["phase"] = "full-polish"
    history.extend(polish_history)
    true_value = float(p2_objective(polished)[0])
    return {
        "point": polished,
        "true_cut_fraction": true_value,
        "noisy_score": noisy_score,
        "evals": p1_oracle.noisy_evaluations + p2_oracle.noisy_evaluations,
        "estimator_calls": (
            p1_objective.estimator_calls + p2_objective.estimator_calls
        ),
        "point_evaluations": (
            p1_objective.point_evaluations + p2_objective.point_evaluations
        ),
        "runtime": time.time() - start_time,
        "history": history,
        "diagnostic": {},
    }


def noise_aware_valley_search(exact_objective, sigma, seed):
    """运行抗噪声 Valley + Planck 架构，不在内部写文件。"""
    start_time = time.time()
    navp.CONFIG["seed"] = seed
    navp.CONFIG["noise_sigma"] = sigma
    oracle = navp.NoisyCutOracle(exact_objective, sigma, seed)
    calls_before = exact_objective.estimator_calls
    points_before = exact_objective.point_evaluations

    elite_points, global_records = navp.sobol_global_scan(
        oracle, exact_objective.num_parameters, exact_objective.reps
    )
    candidate_points, projection_candidates, profiles = navp.robust_projection_search(
        oracle, elite_points, exact_objective.reps
    )
    means, errors, _, _ = oracle.sample(
        candidate_points, int(navp.CONFIG["validation_repeats"])
    )
    scores = navp.lower_confidence_bound(means, errors)
    start_indices = np.argsort(scores)[-int(navp.CONFIG["local_starts"]):][::-1]

    refined_points = []
    history = []
    for index, point in enumerate(candidate_points[start_indices]):
        refined, _, local_history = navp.planck_adam_refine(
            oracle, point, exact_objective.reps, index
        )
        refined_points.append(refined)
        for row in local_history:
            history.append({
                "step": int(row["iteration"]),
                "noisy": float(row["mean"]),
                "true": float(row["true"]),
                "evals": oracle.noisy_evaluations,
                "phase": f"planck-start-{index}",
            })

    final_point, final_summary = navp.final_confirmation(
        oracle, np.asarray(refined_points)
    )
    return {
        "point": final_point,
        "true_cut_fraction": float(final_summary["true"]),
        "noisy_score": float(final_summary["lcb"]),
        "evals": oracle.noisy_evaluations,
        "estimator_calls": exact_objective.estimator_calls - calls_before,
        "point_evaluations": exact_objective.point_evaluations - points_before,
        "runtime": time.time() - start_time,
        "history": history,
        "diagnostic": {
            "global_best_true": float(max(row["true"] for row in global_records)),
            "projection_best_true": float(
                max(row["true"] for row in projection_candidates)
            ),
            "max_geometry_snr": float(
                max(profile["geometry_snr"] for profile in profiles)
            ),
        },
    }


def summarize_and_plot(
    output_dir, rows, exact_cut, total_weight, graph, exact_bits
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(output_dir / "raw_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        ratios = np.asarray([row["approximation_ratio"] for row in selected])
        evaluations = np.asarray([row["evals"] for row in selected])
        aer_calls = np.asarray([row["estimator_calls"] for row in selected])
        runtimes = np.asarray([row["runtime"] for row in selected])
        summary[method] = {
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "evals_mean": float(np.mean(evaluations)),
            "estimator_calls_mean": float(np.mean(aer_calls)),
            "runtime_mean": float(np.mean(runtimes)),
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "comparison_config": CONFIG,
            "problem_config": navp.CONFIG,
            "exact_maxcut": exact_cut,
            "total_edge_weight": total_weight,
            "methods": summary,
        }, file, ensure_ascii=False, indent=2)

    x = np.arange(len(METHODS))
    means = [summary[method]["ratio_mean"] for method in METHODS]
    errors = [summary[method]["ratio_std"] for method in METHODS]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(
        x, means, yerr=errors, capsize=5,
        color=[COLORS[method] for method in METHODS],
    )
    axes[0].set_xticks(x, [SHORT_LABELS[m] for m in METHODS])
    axes[0].set_ylabel("true expected approximation ratio")
    axes[0].set_title("Solution quality")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(
        x, [summary[m]["estimator_calls_mean"] for m in METHODS],
        color=[COLORS[m] for m in METHODS],
    )
    axes[1].set_xticks(x, [SHORT_LABELS[m] for m in METHODS])
    axes[1].set_ylabel("mean Aer estimator.run calls")
    axes[1].set_title("Backend job calls (runtime driver)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].bar(
        x, [summary[m]["runtime_mean"] for m in METHODS],
        color=[COLORS[m] for m in METHODS],
    )
    axes[2].set_xticks(x, [SHORT_LABELS[m] for m in METHODS])
    axes[2].set_ylabel("mean runtime (s)")
    axes[2].set_title("Runtime")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)
    figure.suptitle(
        f"Three QAOA architectures under Gaussian noise σ={CONFIG['noise_sigma']}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "three_architectures_comparison.png", dpi=180)
    plt.close(figure)

    # 每个 trial 的配对结果。
    plt.figure(figsize=(9, 5))
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        plt.plot(
            [row["trial"] for row in selected],
            [row["approximation_ratio"] for row in selected],
            marker="o", linewidth=1.7, color=COLORS[method], label=method,
        )
    plt.xlabel("trial")
    plt.ylabel("true expected approximation ratio")
    plt.title("Paired trial results")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "paired_trials.png", dpi=180)
    plt.close()

    # 三种方法共同使用的图及精确最大割，确保问题实例完全一致。
    position = nx.spring_layout(graph, seed=19)
    node_colors = ["#e74c3c" if bit else "#3498db" for bit in exact_bits]
    cut_edges = [(u, v) for u, v in graph.edges() if exact_bits[u] != exact_bits[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if exact_bits[u] == exact_bits[v]]
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, position, node_color=node_colors, node_size=520)
    nx.draw_networkx_labels(
        graph, position, font_color="white", font_weight="bold"
    )
    nx.draw_networkx_edges(
        graph, position, edgelist=uncut_edges, edge_color="#bdc3c7"
    )
    nx.draw_networkx_edges(
        graph, position, edgelist=cut_edges, edge_color="#2c3e50",
        style="dashed", width=2.0,
    )
    plt.title(f"Common weighted graph | exact Max-Cut = {exact_cut:.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "common_graph_exact_partition.png", dpi=180)
    plt.close()
    return summary


def apply_fast_mode():
    CONFIG.update({
        "trials": 1,
        "adam_restarts": 2,
        "adam_iterations": 8,
        "adam_polish_iterations": 10,
        "valley_coarse_per_dim": 2,
        "valley_top": 2,
        "valley_scan_directions": 3,
        "valley_scan_samples": 5,
        "valley_refine_iterations": 8,
        "valley_polish_iterations": 8,
        "lbl_restarts": 2,
        "lbl_layer_iterations": 8,
        "lbl_polish_iterations": 10,
        "noise_aware_local_iterations": 10,
        "output_dir": "maxcut_three_architectures_results_fast",
    })
    navp.apply_fast_mode()


def main():
    parser = argparse.ArgumentParser(description="Three QAOA architecture comparison")
    parser.add_argument("--fast", action="store_true", help="快速验证模式")
    parser.add_argument("--trials", type=int, default=None, help="覆盖重复次数")
    parser.add_argument("--sigma", type=float, default=None, help="覆盖高斯噪声标准差")
    args = parser.parse_args()
    if args.fast:
        apply_fast_mode()
    if args.trials is not None:
        CONFIG["trials"] = args.trials
    if args.sigma is not None:
        CONFIG["noise_sigma"] = args.sigma
    if int(CONFIG["trials"]) < 1 or float(CONFIG["noise_sigma"]) < 0.0:
        raise ValueError("trials 必须大于 0，sigma 不能为负")

    navp.CONFIG["noise_sigma"] = float(CONFIG["noise_sigma"])
    navp.CONFIG["num_nodes"] = int(CONFIG["num_nodes"])
    navp.CONFIG["local_iterations"] = int(
        CONFIG["noise_aware_local_iterations"]
    )
    weights, graph, labels = navp.build_weighted_graph()
    exact_bits, exact_cut = navp.exact_maxcut(weights)
    operator, total_weight = navp.build_normalized_operator(weights)
    exact_fraction = exact_cut / total_weight
    p2_objective = navp.ExactQAOACut(operator, 2)

    print("=" * 78)
    print("同一噪声下三种 QAOA 优化架构对比")
    print("=" * 78)
    print(
        f"图: {len(weights)} 节点, {graph.number_of_edges()} 条边, "
        f"精确最大割={exact_cut:.3f}, sigma={CONFIG['noise_sigma']}"
    )

    rows = []
    for trial in range(int(CONFIG["trials"])):
        trial_seed = int(CONFIG["base_seed"]) + trial * 100
        print(f"\nTrial {trial + 1}/{CONFIG['trials']}")
        results = {
            "Plain Adam": plain_adam_search(
                p2_objective, float(CONFIG["noise_sigma"]), trial_seed + 4
            ),
            "Original Valley": original_valley_search(
                p2_objective, float(CONFIG["noise_sigma"]), trial_seed + 1
            ),
            "Layer-by-Layer": layer_by_layer_search(
                operator, float(CONFIG["noise_sigma"]), trial_seed + 2
            ),
            "Noise-aware Valley + Planck": noise_aware_valley_search(
                p2_objective, float(CONFIG["noise_sigma"]), trial_seed + 3
            ),
        }
        for method, result in results.items():
            ratio = result["true_cut_fraction"] / exact_fraction
            expected_cut = result["true_cut_fraction"] * total_weight
            rows.append({
                "trial": trial + 1,
                "seed": trial_seed,
                "method": method,
                "true_cut_fraction": float(result["true_cut_fraction"]),
                "expected_cut": float(expected_cut),
                "approximation_ratio": float(ratio),
                "evals": int(result["evals"]),
                "estimator_calls": int(result["estimator_calls"]),
                "point_evaluations": int(result["point_evaluations"]),
                "runtime": float(result["runtime"]),
            })
            print(
                f"  {method:30s} ratio={ratio:.4f}, "
                f"expected cut={expected_cut:.3f}, "
                f"evals={result['evals']}, "
                f"aer_calls={result['estimator_calls']}, "
                f"time={result['runtime']:.2f}s"
            )

    output_dir = Path(str(CONFIG["output_dir"]))
    summary = summarize_and_plot(
        output_dir, rows, exact_cut, total_weight, graph, exact_bits
    )

    print("\n汇总:")
    for method in METHODS:
        item = summary[method]
        print(
            f"  {method:30s} "
            f"ratio={item['ratio_mean']:.4f}±{item['ratio_std']:.4f}, "
            f"evals={item['evals_mean']:.0f}, "
            f"aer_calls={item['estimator_calls_mean']:.0f}, "
            f"time={item['runtime_mean']:.2f}s"
        )
    print(f"结果目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
