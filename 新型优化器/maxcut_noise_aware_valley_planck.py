#!/usr/bin/env python3
"""
抗噪声 Valley Search + Planck-Adam 求解加权 Max-Cut QAOA。

核心架构：
1. Sobol 全局扫描，以置信下界而非单次带噪值筛选精英点。
2. 在精英点附近做随机正交投影扫描；从投影方差中扣除测量噪声方差。
3. 投影扫描不仅诊断谷地，还把每条可靠方向上的最佳点加入候选池。
4. 使用同方向重复 SPSA 估计纯噪声，并用软死区 Planck 置信度滤波 Adam 一阶方向。
5. 用高重复验证和置信下界选择最终参数，避免把偶然低噪声当成最优解。

噪声模型为归一化期望切割值上的高斯噪声，适合快速研究优化器抗噪性。
运行：
    /usr/bin/python3 maxcut_noise_aware_valley_planck.py
快速测试：
    /usr/bin/python3 maxcut_noise_aware_valley_planck.py --fast
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
from scipy.stats import qmc
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

warnings.filterwarnings("ignore")


CONFIG: dict[str, Any] = {
    # 新图；默认 14 节点以兼顾地形扫描速度，可改成 20。
    "num_nodes": 18,
    "n_clusters": 3,
    "p_in": 0.68,
    "p_out": 0.14,
    "w_low": 0.5,
    "w_high": 4.0,
    "graph_seed": 20260702,
    # QAOA 和归一化目标噪声。
    "reps": 2,
    "noise_sigma": 0.010,
    "seed": 91573,
    # 阶段 1：Sobol 全局扫描。
    "global_points": 64,
    "global_repeats": 4,
    "elite_count": 4,
    "selection_z": 1.645,
    # 阶段 2：抗噪声投影谷地搜索。
    "projection_directions": 8,
    "projection_samples": 9,
    "projection_radius": 0.75,
    "projection_repeats": 3,
    "projection_candidates": 8,
    # 阶段 3：Planck-Adam 局部精修。
    "local_starts": 4,
    "local_iterations": 60,
    "spsa_directions": 2,
    "direction_repeats": 2,
    "spsa_c0": 0.10,
    "spsa_gamma": 0.101,
    "learning_rate": 0.08,
    "learning_rate_decay": 0.10,
    "adam_beta1": 0.90,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-8,
    "planck_temperature": 1.0,
    "planck_deadzone": 0.30,
    "planck_gate_sharpness": 5.0,
    "planck_w_min": 0.50,
    "planck_delta": 1e-8,
    "planck_x_max": 8.0,
    # 验证与输出。
    "validation_interval": 5,
    "validation_repeats": 8,
    "final_repeats": 32,
    "output_dir": "noise_aware_valley_planck_results",
}


def build_weighted_graph():
    """生成连通的加权社区图。"""
    n = int(CONFIG["num_nodes"])
    clusters = int(CONFIG["n_clusters"])
    rng = np.random.default_rng(int(CONFIG["graph_seed"]))
    labels = np.asarray([i % clusters for i in range(n)], dtype=int)
    rng.shuffle(labels)

    for _ in range(100):
        weights = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                probability = (
                    float(CONFIG["p_in"])
                    if labels[i] == labels[j]
                    else float(CONFIG["p_out"])
                )
                if rng.random() < probability:
                    weight = round(
                        rng.uniform(float(CONFIG["w_low"]), float(CONFIG["w_high"])),
                        2,
                    )
                    weights[i, j] = weights[j, i] = weight
        graph = nx.from_numpy_array(weights)
        if nx.is_connected(graph):
            return weights, graph, labels
    raise RuntimeError("无法生成连通图，请更换 graph_seed")


def cut_value(bits, weights):
    bits = np.asarray(bits, dtype=int)
    return float(
        np.sum(np.triu(weights, 1) * (bits[:, None] != bits[None, :]))
    )


def exact_maxcut(weights):
    """枚举互补不重复的 2^(n-1) 个分组。"""
    n = len(weights)
    best_value = -np.inf
    best_bits = None
    for state in range(1 << (n - 1)):
        bits = np.zeros(n, dtype=int)
        bits[1:] = (state >> np.arange(n - 1)) & 1
        value = cut_value(bits, weights)
        if value > best_value:
            best_value = value
            best_bits = bits.copy()
    return best_bits, float(best_value)


def zz_pauli(num_qubits, i, j):
    z = np.zeros(num_qubits, dtype=bool)
    x = np.zeros(num_qubits, dtype=bool)
    z[i] = True
    z[j] = True
    return Pauli((z, x))


def build_normalized_operator(weights):
    """构造归一化 H=sum w_ij Z_iZ_j/(2W)，切割比例=1/2-<H>。"""
    n = len(weights)
    total_weight = float(np.sum(np.triu(weights, 1)))
    if total_weight <= 0.0:
        raise ValueError("图中没有正权边")
    paulis = []
    coefficients = []
    for i in range(n):
        for j in range(i):
            if weights[i, j] > 0.0:
                paulis.append(zz_pauli(n, i, j))
                coefficients.append(float(weights[i, j]) / (2.0 * total_weight))
    return SparsePauliOp(paulis, coeffs=np.asarray(coefficients)), total_weight


def wrap_parameters(points, reps):
    """QAOAAnsatz 参数顺序为 beta 后 gamma；利用周期性包裹参数。"""
    result = np.asarray(points, dtype=float).copy()
    one_dimensional = result.ndim == 1
    result = np.atleast_2d(result)
    result[:, :reps] %= np.pi
    result[:, reps:] %= 2.0 * np.pi
    return result[0] if one_dimensional else result


class ExactQAOACut:
    """用 Aer statevector 批量计算精确归一化期望切割值。"""

    def __init__(self, operator, reps):
        self.reps = reps
        ansatz = QAOAAnsatz(operator, reps=reps, flatten=True)
        pass_manager = generate_preset_pass_manager(
            optimization_level=1,
            basis_gates=["rz", "rx", "h", "cx"],
        )
        self.circuit = pass_manager.run(ansatz)
        self.observable = operator
        self.estimator: Any = AerEstimator(
            options={"backend_options": {"method": "statevector"}}
        )
        self.num_parameters = self.circuit.num_parameters
        # 统计后端调用情况：estimator.run 次数与累计参数点数。
        self.estimator_calls = 0
        self.point_evaluations = 0

    def __call__(self, parameter_points):
        points = wrap_parameters(parameter_points, self.reps)
        points = np.atleast_2d(points)
        publication = (self.circuit, self.observable, points)
        result = self.estimator.run([publication], precision=0.0).result()[0]
        energies = np.asarray(result.data.evs, dtype=float).reshape(-1)
        self.estimator_calls += 1
        self.point_evaluations += len(points)
        return 0.5 - energies


class NoisyCutOracle:
    """向精确归一化期望切割值加入可控高斯噪声并统计评估预算。"""

    def __init__(self, exact_objective, sigma, seed):
        self.exact = exact_objective
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.noisy_evaluations = 0

    def sample(self, points, repeats):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        true_values = self.exact(points)
        noise = self.rng.normal(0.0, self.sigma, size=(len(points), repeats))
        samples = true_values[:, None] + noise
        self.noisy_evaluations += len(points) * repeats
        means = np.mean(samples, axis=1)
        if repeats > 1:
            standard_errors = np.std(samples, axis=1, ddof=1) / np.sqrt(repeats)
        else:
            standard_errors = np.full(len(points), np.inf)
        return means, standard_errors, true_values, samples


def lower_confidence_bound(means, standard_errors):
    return means - float(CONFIG["selection_z"]) * standard_errors


def sobol_global_scan(oracle, dimensions, reps):
    """用低差异 Sobol 点覆盖全局参数空间并按置信下界选精英点。"""
    count = int(CONFIG["global_points"])
    sampler = qmc.Sobol(
        d=dimensions, scramble=True, seed=int(CONFIG["seed"])
    )
    power = int(np.ceil(np.log2(max(2, count))))
    unit_points = sampler.random_base2(power)[:count]
    upper_bounds = np.concatenate((np.full(reps, np.pi), np.full(reps, 2.0 * np.pi)))
    points = unit_points * upper_bounds

    # 加入结构化线性初始化和零角初态附近点。
    beta = np.linspace(0.42, 0.12, reps)
    gamma = np.linspace(0.15, 0.70, reps)
    points = np.vstack((points, np.concatenate((beta, gamma)), np.zeros(dimensions)))

    means, errors, true_values, _ = oracle.sample(
        points, int(CONFIG["global_repeats"])
    )
    scores = lower_confidence_bound(means, errors)
    elite_indices = np.argsort(scores)[-int(CONFIG["elite_count"]):][::-1]
    records = [
        {
            "stage": "global",
            "candidate": index,
            "mean": float(means[index]),
            "se": float(errors[index]),
            "lcb": float(scores[index]),
            "true": float(true_values[index]),
        }
        for index in range(len(points))
    ]
    return points[elite_indices], records


def random_orthogonal_directions(dimensions, count, rng):
    """分块生成近似均匀的随机正交方向。"""
    directions = []
    while len(directions) < count:
        matrix = rng.normal(size=(dimensions, dimensions))
        orthogonal, _ = np.linalg.qr(matrix)
        directions.extend(orthogonal.T)
    return np.asarray(directions[:count])


def robust_projection_search(oracle, elite_points, reps):
    """抗噪投影扫描：扣除噪声方差，并把方向上的可靠最佳点加入候选池。"""
    rng = np.random.default_rng(int(CONFIG["seed"]) + 101)
    dimensions = elite_points.shape[1]
    directions = random_orthogonal_directions(
        dimensions, int(CONFIG["projection_directions"]), rng
    )
    offsets = np.linspace(
        -float(CONFIG["projection_radius"]),
        float(CONFIG["projection_radius"]),
        int(CONFIG["projection_samples"]),
    )
    candidates = []
    profiles = []

    for elite_index, center in enumerate(elite_points):
        for direction_index, direction in enumerate(directions):
            line_points = wrap_parameters(
                center[None, :] + offsets[:, None] * direction[None, :], reps
            )
            means, errors, true_values, _ = oracle.sample(
                line_points, int(CONFIG["projection_repeats"])
            )
            scores = lower_confidence_bound(means, errors)
            best_index = int(np.argmax(scores))

            observed_variance = float(np.var(means, ddof=1))
            noise_variance = float(np.mean(errors ** 2))
            signal_variance = max(observed_variance - noise_variance, 0.0)
            geometry_snr = np.sqrt(signal_variance) / (
                float(np.mean(errors)) + 1e-12
            )
            improvement_lcb = float(scores[best_index] - scores[len(offsets) // 2])
            candidates.append({
                "point": line_points[best_index].copy(),
                "lcb": float(scores[best_index]),
                "mean": float(means[best_index]),
                "se": float(errors[best_index]),
                "true": float(true_values[best_index]),
                "geometry_snr": float(geometry_snr),
                "improvement_lcb": improvement_lcb,
            })
            profiles.append({
                "elite": elite_index,
                "direction": direction_index,
                "offsets": offsets.copy(),
                "means": means.copy(),
                "errors": errors.copy(),
                "true_values": true_values.copy(),
                "signal_variance": signal_variance,
                "geometry_snr": float(geometry_snr),
            })

    candidates.sort(
        key=lambda item: (item["lcb"], item["geometry_snr"]), reverse=True
    )
    limit = int(CONFIG["projection_candidates"])
    candidate_points = [item["point"] for item in candidates[:limit]]
    # 精英中心也保留，防止噪声使投影移动反而变差。
    candidate_points.extend(point.copy() for point in elite_points)
    return np.asarray(candidate_points), candidates, profiles


def planck_confidence(noise_to_signal):
    """带软死区的 Planck 置信度。"""
    x = float(np.clip(noise_to_signal, 0.0, float(CONFIG["planck_x_max"])))
    temperature = float(CONFIG["planck_temperature"])
    scaled = x / temperature
    if scaled < 1e-4:
        base = 1.0 - scaled / 2.0 + scaled ** 2 / 12.0
    else:
        base = scaled / np.expm1(scaled)
    gate_argument = np.clip(
        float(CONFIG["planck_gate_sharpness"])
        * (x - float(CONFIG["planck_deadzone"])),
        -60.0,
        60.0,
    )
    gate = 1.0 / (1.0 + np.exp(-gate_argument))
    confidence = 1.0 - gate * (1.0 - base)
    return max(confidence, float(CONFIG["planck_w_min"]))


def planck_adam_refine(oracle, initial_point, reps, start_index):
    """同方向重复 SPSA + Planck 置信度滤波 Adam，一阶滤波、二阶保留原噪声。"""
    theta = wrap_parameters(initial_point, reps)
    dimensions = len(theta)
    rng = np.random.default_rng(int(CONFIG["seed"]) + 1000 + start_index)
    m = np.zeros(dimensions)
    v = np.zeros(dimensions)
    directions = int(CONFIG["spsa_directions"])
    repeats = int(CONFIG["direction_repeats"])
    beta1 = float(CONFIG["adam_beta1"])
    beta2 = float(CONFIG["adam_beta2"])
    epsilon = float(CONFIG["adam_epsilon"])

    best_point = theta.copy()
    means, errors, true_values, _ = oracle.sample(
        theta[None, :], int(CONFIG["validation_repeats"])
    )
    best_lcb = float(lower_confidence_bound(means, errors)[0])
    history = [{
        "start": start_index,
        "iteration": 0,
        "mean": float(means[0]),
        "se": float(errors[0]),
        "lcb": best_lcb,
        "true": float(true_values[0]),
        "confidence": 1.0,
        "noise_to_signal": 0.0,
    }]

    for iteration in range(1, int(CONFIG["local_iterations"]) + 1):
        c_t = float(CONFIG["spsa_c0"]) / (
            iteration ** float(CONFIG["spsa_gamma"])
        )
        deltas = rng.choice([-1.0, 1.0], size=(directions, dimensions))
        plus_points = wrap_parameters(theta + c_t * deltas, reps)
        minus_points = wrap_parameters(theta - c_t * deltas, reps)

        _, _, _, plus_samples = oracle.sample(plus_points, repeats)
        _, _, _, minus_samples = oracle.sample(minus_points, repeats)
        slopes = (plus_samples - minus_samples) / (2.0 * c_t)
        gradient_samples = slopes[:, :, None] * deltas[:, None, :]
        raw_gradient = np.mean(gradient_samples, axis=(0, 1))

        within_direction_variance = np.var(gradient_samples, axis=1, ddof=1)
        gradient_standard_error = np.sqrt(
            np.sum(within_direction_variance, axis=0)
            / (directions ** 2 * repeats)
        )
        noise_to_signal = np.linalg.norm(gradient_standard_error) / (
            np.linalg.norm(raw_gradient) + float(CONFIG["planck_delta"])
        )
        confidence = planck_confidence(noise_to_signal)

        if iteration == 1:
            filtered_gradient = raw_gradient
        else:
            previous_direction = m / (1.0 - beta1 ** (iteration - 1))
            filtered_gradient = (
                confidence * raw_gradient
                + (1.0 - confidence) * previous_direction
            )

        m = beta1 * m + (1.0 - beta1) * filtered_gradient
        # 二阶矩使用原始梯度，保留噪声尺度。
        v = beta2 * v + (1.0 - beta2) * raw_gradient ** 2
        m_hat = m / (1.0 - beta1 ** iteration)
        v_hat = v / (1.0 - beta2 ** iteration)
        learning_rate = float(CONFIG["learning_rate"]) / (
            iteration ** float(CONFIG["learning_rate_decay"])
        )
        theta = wrap_parameters(
            theta + learning_rate * m_hat / (np.sqrt(v_hat) + epsilon), reps
        )

        if (
            iteration == 1
            or iteration % int(CONFIG["validation_interval"]) == 0
            or iteration == int(CONFIG["local_iterations"])
        ):
            means, errors, true_values, _ = oracle.sample(
                theta[None, :], int(CONFIG["validation_repeats"])
            )
            lcb = float(lower_confidence_bound(means, errors)[0])
            if lcb > best_lcb:
                best_lcb = lcb
                best_point = theta.copy()
            history.append({
                "start": start_index,
                "iteration": iteration,
                "mean": float(means[0]),
                "se": float(errors[0]),
                "lcb": lcb,
                "true": float(true_values[0]),
                "confidence": float(confidence),
                "noise_to_signal": float(noise_to_signal),
            })
    return best_point, best_lcb, history


def final_confirmation(oracle, candidate_points):
    """用较高重复数统一复测局部候选，并按置信下界选择最终参数。"""
    means, errors, true_values, _ = oracle.sample(
        candidate_points, int(CONFIG["final_repeats"])
    )
    scores = lower_confidence_bound(means, errors)
    index = int(np.argmax(scores))
    return candidate_points[index], {
        "mean": float(means[index]),
        "se": float(errors[index]),
        "lcb": float(scores[index]),
        "true": float(true_values[index]),
        "all_true": true_values,
        "all_lcb": scores,
    }


def save_outputs(
    output_dir, graph, weights, labels, exact_bits, exact_cut, total_weight,
    global_records, projection_candidates, profiles, local_history, final_summary,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "graph_weights.csv", weights, delimiter=",", fmt="%.2f")
    with open(output_dir / "config_and_result.json", "w", encoding="utf-8") as file:
        json.dump({
            "config": CONFIG,
            "cluster_labels": labels.tolist(),
            "exact_bits": exact_bits.tolist(),
            "exact_maxcut": exact_cut,
            "total_edge_weight": total_weight,
            "final": {key: value for key, value in final_summary.items()
                      if not isinstance(value, np.ndarray)},
        }, file, ensure_ascii=False, indent=2)

    with open(output_dir / "global_scan.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(global_records[0].keys()))
        writer.writeheader()
        writer.writerows(global_records)
    with open(output_dir / "local_history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(local_history[0].keys()))
        writer.writeheader()
        writer.writerows(local_history)

    # 新图及精确分组。
    position = nx.spring_layout(graph, seed=19)
    node_colors = ["#e74c3c" if bit else "#3498db" for bit in exact_bits]
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph, position, node_color=node_colors, node_size=520,
        font_color="white", font_weight="bold", edge_color="#95a5a6",
    )
    plt.title(f"Weighted graph | exact Max-Cut = {exact_cut:.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "graph_exact_partition.png", dpi=180)
    plt.close()

    # 投影曲线：绘制几何 SNR 最高的四条。
    selected_profiles = sorted(
        profiles, key=lambda item: item["geometry_snr"], reverse=True
    )[:4]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for axis, profile in zip(axes.flat, selected_profiles):
        axis.errorbar(
            profile["offsets"], profile["means"], yerr=profile["errors"],
            marker="o", capsize=3, color="#3498db", label="noisy mean ± SE",
        )
        axis.plot(
            profile["offsets"], profile["true_values"], color="#e74c3c",
            linestyle="--", label="true landscape",
        )
        axis.set_title(
            f"elite {profile['elite']}, dir {profile['direction']} | "
            f"geometry SNR={profile['geometry_snr']:.2f}"
        )
        axis.set_xlabel("projection offset")
        axis.set_ylabel("normalized expected cut")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "noise_corrected_valley_profiles.png", dpi=180)
    plt.close(figure)

    # 局部精修收敛与 Planck 置信度。
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    starts = sorted(set(int(row["start"]) for row in local_history))
    for start in starts:
        rows = [row for row in local_history if int(row["start"]) == start]
        iterations = [row["iteration"] for row in rows]
        left.plot(iterations, [row["true"] for row in rows], marker="o", label=f"start {start}")
        right.plot(iterations, [row["confidence"] for row in rows], marker="o", label=f"start {start}")
    left.set_title("Local refinement: true expected cut")
    left.set_xlabel("iteration")
    left.set_ylabel("normalized expected cut")
    right.set_title("Planck confidence")
    right.set_xlabel("iteration")
    right.set_ylabel("confidence")
    right.set_ylim(0.45, 1.02)
    for axis in (left, right):
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "local_refinement.png", dpi=180)
    plt.close(figure)

    # 候选的 LCB 与真实值，检查噪声排序是否可靠。
    candidate_true = np.asarray([item["true"] for item in projection_candidates])
    candidate_lcb = np.asarray([item["lcb"] for item in projection_candidates])
    plt.figure(figsize=(7, 5))
    plt.scatter(candidate_lcb, candidate_true, color="#8e44ad", alpha=0.75)
    plt.xlabel("projection candidate LCB")
    plt.ylabel("true normalized expected cut")
    plt.title("Noise-aware candidate ranking")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_ranking.png", dpi=180)
    plt.close()


def apply_fast_mode():
    CONFIG.update({
        "global_points": 16,
        "global_repeats": 2,
        "elite_count": 2,
        "projection_directions": 3,
        "projection_samples": 5,
        "projection_repeats": 2,
        "projection_candidates": 3,
        "local_starts": 2,
        "local_iterations": 10,
        "validation_interval": 2,
        "validation_repeats": 3,
        "final_repeats": 6,
        "output_dir": "noise_aware_valley_planck_results_fast",
    })


def main():
    parser = argparse.ArgumentParser(
        description="Noise-aware Valley Search + Planck-Adam for QAOA Max-Cut"
    )
    parser.add_argument("--fast", action="store_true", help="快速验证模式")
    parser.add_argument("--sigma", type=float, default=None, help="覆盖高斯噪声标准差")
    args = parser.parse_args()
    if args.fast:
        apply_fast_mode()
    if args.sigma is not None:
        if args.sigma < 0.0:
            raise ValueError("sigma 不能小于 0")
        CONFIG["noise_sigma"] = args.sigma

    start_time = time.time()
    output_dir = Path(str(CONFIG["output_dir"]))
    weights, graph, labels = build_weighted_graph()
    exact_bits, exact_cut = exact_maxcut(weights)
    operator, total_weight = build_normalized_operator(weights)
    exact_cut_fraction = exact_cut / total_weight
    exact_objective = ExactQAOACut(operator, int(CONFIG["reps"]))
    oracle = NoisyCutOracle(
        exact_objective, float(CONFIG["noise_sigma"]), int(CONFIG["seed"])
    )

    print("=" * 72)
    print("抗噪声 Valley Search + Planck-Adam")
    print("=" * 72)
    print(
        f"图: {len(weights)} 节点, {graph.number_of_edges()} 条边, "
        f"精确最大割={exact_cut:.3f}"
    )
    print(
        f"QAOA p={CONFIG['reps']}, 归一化高斯噪声 sigma={CONFIG['noise_sigma']}"
    )

    print("\n[1/4] Sobol 全局扫描与置信下界筛选")
    elite_points, global_records = sobol_global_scan(
        oracle, exact_objective.num_parameters, int(CONFIG["reps"])
    )
    global_best = max(global_records, key=lambda item: item["lcb"])
    print(
        f"  全局最佳 LCB={global_best['lcb']:.4f}, "
        f"真实期望切割={global_best['true']:.4f}"
    )

    print("\n[2/4] 抗噪声随机投影谷地搜索")
    candidate_points, projection_candidates, profiles = robust_projection_search(
        oracle, elite_points, int(CONFIG["reps"])
    )
    best_projection = projection_candidates[0]
    print(
        f"  投影最佳 LCB={best_projection['lcb']:.4f}, "
        f"真实值={best_projection['true']:.4f}, "
        f"geometry SNR={best_projection['geometry_snr']:.2f}"
    )

    # 统一预验证，避免仅按投影阶段的少量样本决定局部起点。
    means, errors, _, _ = oracle.sample(
        candidate_points, int(CONFIG["validation_repeats"])
    )
    pre_scores = lower_confidence_bound(means, errors)
    start_indices = np.argsort(pre_scores)[-int(CONFIG["local_starts"]):][::-1]
    local_starts = candidate_points[start_indices]

    print("\n[3/4] 同方向重复 SPSA + Planck-Adam 多起点精修")
    refined_points = []
    all_local_history = []
    for index, point in enumerate(local_starts):
        refined, best_lcb, history = planck_adam_refine(
            oracle, point, int(CONFIG["reps"]), index
        )
        refined_points.append(refined)
        all_local_history.extend(history)
        best_true_seen = max(row["true"] for row in history)
        mean_confidence = float(np.mean([row["confidence"] for row in history[1:]]))
        print(
            f"  start {index}: best LCB={best_lcb:.4f}, "
            f"best true={best_true_seen:.4f}, mean confidence={mean_confidence:.3f}"
        )

    print("\n[4/4] 高重复统一确认")
    final_point, final_summary = final_confirmation(
        oracle, np.asarray(refined_points)
    )
    expected_cut = final_summary["true"] * total_weight
    approximation_ratio = expected_cut / exact_cut
    print(
        f"  最终 noisy mean={final_summary['mean']:.4f} ± {final_summary['se']:.4f}"
    )
    print(f"  最终真实期望切割={expected_cut:.3f}")
    print(f"  相对精确最大割的期望近似比={approximation_ratio:.4f}")
    print(f"  带噪目标评估数={oracle.noisy_evaluations}")

    final_summary["expected_cut"] = expected_cut
    final_summary["approximation_ratio"] = approximation_ratio
    final_summary["noisy_evaluations"] = oracle.noisy_evaluations
    final_summary["parameters"] = final_point
    save_outputs(
        output_dir, graph, weights, labels, exact_bits, exact_cut, total_weight,
        global_records, projection_candidates, profiles, all_local_history,
        final_summary,
    )
    print(f"  总耗时={time.time() - start_time:.1f}s")
    print(f"  结果目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
