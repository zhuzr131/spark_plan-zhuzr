"""Noise-aware Valley + Planck QAOA 优化器 — TensorCircuit / 腾讯量子云真机版。

策略：Sobol 全局扫描 → LCB 精英选取 → 投影搜索 → Planck-SPSA 精修 → holdout 精选。

关键改进（适用于真机）：
  - Planck 置信度做幅度缩放而非方向混合
  - 中位数 SPSA 梯度抗离群噪声
  - 动量重置防陷入坏盆地
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import qmc
from tc_qaoa_utils import (
    random_weighted_graph, noisy_evaluate,
)

CONFIG = {
    "num_nodes": 8,
    "graph_seed": 42,
    "reps": 3,
    "noise_model": "depolarizing",
    "noise_strength": 0.02,
    "shots": 1024,
    "eval_repeats": 2,
    # Sobol 全局扫描
    "global_points": 64,
    "global_repeats": 2,
    "elite_count": 4,
    "min_elite_distance": 0.18,
    # 投影搜索
    "projection_directions": 6,
    "projection_samples": 7,
    "projection_radius": 0.45,
    "projection_repeats": 2,
    "projection_snr_min": 0.8,
    # SPSA + Planck
    "local_iterations": 50,
    "spsa_directions": 2,
    "spsa_c0": 0.12,
    "spsa_gamma": 0.101,
    "learning_rate": 0.07,
    "learning_rate_decay": 0.12,
    "planck_temperature": 0.9,
    "planck_deadzone": 0.25,
    "planck_gate_sharpness": 6.0,
    "planck_w_min": 0.10,
    "validation_interval": 5,
    "validation_repeats": 3,
    # 输出
    "output_dir": "/Users/zhuzhengrong/Desktop/新型优化器/tc_noise_aware_results",
}


def lcb(means: np.ndarray, errors: np.ndarray, z: float = 1.645) -> np.ndarray:
    """下置信界：L = μ − zσ，值越大越优（MaxCut 是最大化问题）。"""
    return means - z * errors


def planck_confidence(noise_to_signal: float) -> float:
    """Planck 置信度阻尼：噪声大→置信度低→步幅自动缩小。

    weight = 1 / (1 + exp((noise/signal - deadzone) * sharpness))
    映射到 [w_min, 1.0]。
    """
    T = float(CONFIG["planck_temperature"])
    dz = float(CONFIG["planck_deadzone"])
    k = float(CONFIG["planck_gate_sharpness"])
    w_min = float(CONFIG["planck_w_min"])
    if noise_to_signal <= dz:
        return 1.0
    raw = 1.0 / (1.0 + np.exp((noise_to_signal - dz) * k / T))
    return w_min + (1.0 - w_min) * raw


def sobol_scan(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Sobol 低差异序列全局扫描。"""
    reps = int(CONFIG["reps"])
    n_points = int(CONFIG["global_points"])
    n_dims = 2 * reps

    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=int(CONFIG["graph_seed"]))
    points = sampler.random(n_points)

    # beta ∈ [0, π), gamma ∈ [0, 2π)
    all_b = points[:, :reps] * np.pi
    all_g = points[:, reps:] * 2.0 * np.pi
    n_r = int(CONFIG["global_repeats"])

    means, errors = [], []
    for i in range(n_points):
        m, e = noisy_evaluate(
            edges, n_qubits, total_weight, all_b[i], all_g[i],
            shots=int(CONFIG["shots"]), repeats=n_r,
)
        means.append(m)
        errors.append(e)
    return all_b, all_g, means, errors


def select_elites(
    betas: np.ndarray, gammas: np.ndarray,
    means: List[float], errors: List[float],
) -> List[int]:
    """LCB 排序 + 最小距离过滤，返回精英索引。"""
    scores = lcb(np.array(means), np.array(errors))
    order = np.argsort(scores)[::-1]
    elite_count = int(CONFIG["elite_count"])
    min_dist = float(CONFIG["min_elite_distance"])
    reps = int(CONFIG["reps"])

    selected = []
    for idx in order:
        if len(selected) >= elite_count:
            break
        point = np.concatenate([betas[idx], gammas[idx]])
        too_close = False
        for s in selected:
            spoint = np.concatenate([betas[s], gammas[s]])
            dist = np.sqrt(np.sum((point - spoint) ** 2))
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append(int(idx))
    return selected


def projection_search(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    base_betas: np.ndarray, base_gammas: np.ndarray,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """从基点出发沿随机方向投影搜索，返回 (β, γ, cut) 列表。"""
    reps = len(base_betas)
    n_dirs = int(CONFIG["projection_directions"])
    n_samples = int(CONFIG["projection_samples"])
    radius = float(CONFIG["projection_radius"])

    results = []
    for _ in range(n_dirs):
        dir_all = rng.normal(0, 1, 2 * reps)
        dir_all /= np.linalg.norm(dir_all)
        for s in range(n_samples):
            t = (s / (n_samples - 1) - 0.5) * 2.0 * radius if n_samples > 1 else 0.0
            tb = t * dir_all[:reps]
            tg = t * dir_all[reps:]
            test_b = (base_betas + tb) % np.pi
            test_g = (base_gammas + tg) % (2.0 * np.pi)
            cut, _ = noisy_evaluate(
                edges, n_qubits, total_weight, test_b, test_g,
                shots=int(CONFIG["shots"]),
                repeats=int(CONFIG["projection_repeats"]),
                noise_model=CONFIG["noise_model"],
                noise_strength=float(CONFIG["noise_strength"]),
                rng=rng,
            )
            results.append((test_b.copy(), test_g.copy(), cut))
    return results


def planck_spsa_refine(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas: np.ndarray, gammas: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, list]:
    """中位数 SPSA + Planck 置信度阻尼 + 动量重置。"""
    reps = len(betas)
    n_iters = int(CONFIG["local_iterations"])
    beta1, beta2, eps = 0.9, 0.98, 1e-8
    lr0 = float(CONFIG["learning_rate"])
    decay = float(CONFIG["learning_rate_decay"])
    c0 = float(CONFIG["spsa_c0"])
    gamma = float(CONFIG["spsa_gamma"])
    n_dirs = int(CONFIG["spsa_directions"])

    theta = np.concatenate([betas, gammas])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    best_theta = theta.copy()
    best_lcb = -np.inf
    prev_noise = 0.0
    stagnation = 0
    prev_score = best_lcb
    history = []

    rng = np.random.default_rng(seed)

    for iteration in range(1, n_iters + 1):
        c_base = c0 / (iteration ** gamma)
        c_t = c_base * max(1.0, 1.0 + min(prev_noise, 2.0))

        # SPSA 多方向梯度
        dim = len(theta)
        deltas = rng.choice([-1, 1], size=(n_dirs, dim))
        plus_pts = np.array([theta + c_t * d for d in deltas])
        minus_pts = np.array([theta - c_t * d for d in deltas])

        plus_vals, minus_vals = [], []
        for i in range(n_dirs):
            pv, _ = noisy_evaluate(
                edges, n_qubits, total_weight,
                plus_pts[i, :reps], plus_pts[i, reps:],
                shots=int(CONFIG["shots"]), repeats=2,
                noise_model=CONFIG["noise_model"],
                noise_strength=float(CONFIG["noise_strength"]),
                rng=rng,
            )
            mv, _ = noisy_evaluate(
                edges, n_qubits, total_weight,
                minus_pts[i, :reps], minus_pts[i, reps:],
                shots=int(CONFIG["shots"]), repeats=2,
                noise_model=CONFIG["noise_model"],
                noise_strength=float(CONFIG["noise_strength"]),
                rng=rng,
            )
            plus_vals.append(pv)
            minus_vals.append(mv)
        plus_vals = np.array(plus_vals)
        minus_vals = np.array(minus_vals)

        slopes = (plus_vals - minus_vals) / (2.0 * c_t)
        gradient_samples = slopes[:, None] * deltas  # (n_dirs, dim)
        raw_gradient = np.median(gradient_samples, axis=0)  # 中位数抗离群
        flat = gradient_samples.reshape(-1)
        se = np.std(flat, ddof=1) / np.sqrt(len(flat)) if len(flat) > 1 else 0.0
        noise_to_signal = float(se / (np.linalg.norm(raw_gradient) + 1e-12))

        confidence = planck_confidence(noise_to_signal)
        prev_noise = noise_to_signal

        # 动量重置
        if confidence < 0.25:
            stagnation += 1
        else:
            stagnation = max(0, stagnation - 1)
        if stagnation >= 3:
            m = np.zeros_like(m)
            v = np.zeros_like(v)
            stagnation = 0

        filtered = confidence * raw_gradient
        m = beta1 * m + (1.0 - beta1) * filtered
        v = beta2 * v + (1.0 - beta2) * filtered ** 2
        m_hat = m / (1.0 - beta1 ** iteration)
        v_hat = v / (1.0 - beta2 ** iteration)

        lr = lr0 / (iteration ** decay)
        theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)
        theta[:reps] %= np.pi
        theta[reps:] %= (2.0 * np.pi)

        # 验证
        if iteration % int(CONFIG["validation_interval"]) == 0 or iteration == n_iters:
            cur_betas = theta[:reps]
            cur_gammas = theta[reps:]
            m_vals, e_vals = noisy_evaluate(
                edges, n_qubits, total_weight, cur_betas, cur_gammas,
                shots=int(CONFIG["shots"]),
                repeats=int(CONFIG["validation_repeats"]),
                noise_model=CONFIG["noise_model"],
                noise_strength=float(CONFIG["noise_strength"]),
                rng=rng,
            )
            score = float(lcb(np.array([m_vals]), np.array([e_vals]))[0])
            history.append({"iteration": iteration, "lcb": score, "confidence": confidence,
                            "noise_to_signal": noise_to_signal})
            if score > best_lcb:
                best_lcb = score
                best_theta = theta.copy()
                stagnation = 0
            elif score <= prev_score:
                stagnation += 1
                if stagnation >= 3:
                    m = np.zeros_like(m)
                    v = np.zeros_like(v)
                    stagnation = 0
            prev_score = score

    return best_theta[:reps], best_theta[reps:], best_lcb, history


def main():
    import os, json, time
    from pathlib import Path

    rng = np.random.default_rng(int(CONFIG["graph_seed"]))
    _, edges_arr, total_weight = random_weighted_graph(
        int(CONFIG["num_nodes"]), int(CONFIG["graph_seed"]),
    )
    n_qubits = int(CONFIG["num_nodes"])
    reps = int(CONFIG["reps"])
    print(f"图: {n_qubits} 节点, {len(edges_arr)} 条边")

    start = time.time()

    # 1) Sobol 扫描
    print("Sobol 全局扫描...")
    all_b, all_g, means, errors = sobol_scan(edges_arr, n_qubits, total_weight, rng)

    # 2) 精英选取
    elites = select_elites(all_b, all_g, means, errors)
    print(f"  精英: {len(elites)} 个")

    # 3) 投影搜索
    print("投影搜索...")
    proj_results = []
    for idx in elites:
        res = projection_search(
            edges_arr, n_qubits, total_weight,
            all_b[idx], all_g[idx], rng,
        )
        proj_results.extend(res)

    # 选 LCB 最优的几个起点
    proj_results.sort(key=lambda x: x[2], reverse=True)
    starts = proj_results[:3]

    # 4) Planck-SPSA 精修
    print("Planck-SPSA 精修...")
    best_bt, best_gm, best_l = None, None, -np.inf
    for i, (sb, sg, _) in enumerate(starts):
        bt, gm, l, _ = planck_spsa_refine(
            edges_arr, n_qubits, total_weight,
            sb, sg, int(CONFIG["graph_seed"]) + 1000 + i,
        )
        cut_fn, _ = noisy_evaluate(
            edges_arr, n_qubits, total_weight, bt, gm,
            shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
)
        if cut_fn > best_l:
            best_l = cut_fn
            best_bt = bt
            best_gm = gm

    elapsed = time.time() - start
    print(f"\n最终近似比: {best_l:.4f}")
    print(f"耗时: {elapsed:.1f}s")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Noise-aware Valley + Planck (TC)",
        "num_nodes": n_qubits, "reps": reps,
        "ratio": float(best_l),
        "gammas": best_gm.tolist(), "betas": best_bt.tolist(),
        "time": elapsed,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
