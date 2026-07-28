"""Depth-continued Noise-aware QAOA — TensorCircuit / 腾讯量子云真机版。

策略：p=2 高效搜索 → 参数插值延续到 p=3 → Planck-SPSA 精修 → LCB holdout 精选。

核心优势：
  - p=2 搜索节省真机评估次数
  - 插值 + 精修以低成本扩展到 p=3
  - 零填充安全地板保证不退化
  - 中位数 SPSA + 动量重置防陷阱
"""

import numpy as np
from typing import Tuple, List
from scipy.stats import qmc
from tc_qaoa_utils import (
    random_weighted_graph, noisy_evaluate,
)
from tc_noise_aware_valley_planck import (
    sobol_scan, select_elites, projection_search,
    planck_spsa_refine, lcb,
)

CONFIG = {
    "num_nodes": 8,
    "graph_seed": 42,
    "reps": 3,                    # 目标层数
    "noise_model": "depolarizing",
    "noise_strength": 0.02,
    "shots": 1024,
    "eval_repeats": 2,
    # p=2 源搜索
    "source_global_points": 48,
    "source_global_repeats": 2,
    "source_elite_count": 4,
    "source_min_elite_distance": 0.18,
    "source_projection_directions": 5,
    "source_projection_samples": 6,
    "source_projection_radius": 0.45,
    "source_projection_repeats": 2,
    "source_local_iterations": 50,
    "source_spsa_directions": 2,
    # p=3 延续
    "continuation_candidates": 2,
    "continuation_iterations": 25,
    "final_repeats": 10,
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
    "output_dir": "/Users/zhuzhengrong/Desktop/新型优化器/tc_depth_continued_results",
}


def interpolate_depth(point_2: np.ndarray, target_reps: int) -> np.ndarray:
    """线性插值将 p=2 参数扩展到 p=target_reps。

    point_2 结构：[β₁, β₂, γ₁, γ₂]。
    """
    source_reps = 2
    source_axis = np.linspace(0.0, 1.0, source_reps)
    target_axis = np.linspace(0.0, 1.0, target_reps)
    betas = np.interp(target_axis, source_axis, point_2[:source_reps])
    gammas = np.interp(target_axis, source_axis, point_2[source_reps:])
    return np.concatenate([betas, gammas])


def p2_source_search(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    rng: np.random.Generator, seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """在 p=2 上运行完整搜索管线：Sobol → 精英 → 投影 → SPSA。"""
    # 临时覆盖参数
    saved = {k: CONFIG[k] for k in (
        "global_points", "global_repeats", "elite_count",
        "min_elite_distance", "projection_directions",
        "projection_samples", "projection_radius",
        "projection_repeats", "local_iterations",
        "spsa_directions",
    )}
    CONFIG.update({
        "global_points": int(CONFIG["source_global_points"]),
        "global_repeats": int(CONFIG["source_global_repeats"]),
        "elite_count": int(CONFIG["source_elite_count"]),
        "min_elite_distance": float(CONFIG["source_min_elite_distance"]),
        "projection_directions": int(CONFIG["source_projection_directions"]),
        "projection_samples": int(CONFIG["source_projection_samples"]),
        "projection_radius": float(CONFIG["source_projection_radius"]),
        "projection_repeats": int(CONFIG["source_projection_repeats"]),
        "local_iterations": int(CONFIG["source_local_iterations"]),
        "spsa_directions": int(CONFIG["source_spsa_directions"]),
    })
    # Sobol scan
    all_b, all_g, means, errors = sobol_scan(edges, n_qubits, total_weight, rng)
    # 精英选取
    elites = select_elites(all_b, all_g, means, errors)
    # 投影搜索
    proj_results = []
    for idx in elites:
        res = projection_search(edges, n_qubits, total_weight, all_b[idx], all_g[idx], rng)
        proj_results.extend(res)
    proj_results.sort(key=lambda x: x[2], reverse=True)
    starts = proj_results[:3]
    # SPSA 精修
    refinements = []
    for i, (sb, sg, _) in enumerate(starts):
        bt, gm, l, _ = planck_spsa_refine(
            edges, n_qubits, total_weight, sb, sg, seed + 100 + i,
        )
        refinements.append((bt, gm, l))
    refinements.sort(key=lambda x: x[2], reverse=True)
    # 恢复
    CONFIG.update(saved)
    final_betas = np.array([r[0] for r in refinements])
    final_gammas = np.array([r[1] for r in refinements])
    return final_betas, final_gammas


def continuation_candidates(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    source_betas: np.ndarray, source_gammas: np.ndarray,
    rng: np.random.Generator, seed: int,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """将 p=2 候选延续到 p=3：插值 → SPSA 精修 → 零填充安全地板。"""
    target_reps = int(CONFIG["reps"])
    n_cont = int(CONFIG["continuation_candidates"])
    n_cont = min(n_cont, len(source_betas))

    # 安全地板：零填充最优 p=2 候选
    source_best = np.concatenate([source_betas[0], source_gammas[0]])
    zero_padded = np.concatenate([
        source_best[:2], np.zeros(target_reps - 2),
        source_best[2:4], np.zeros(target_reps - 2),
    ])

    # 插值 + SPSA 精修
    candidates = []
    for idx in range(n_cont):
        p2_point = np.concatenate([source_betas[idx], source_gammas[idx]])
        p3_transferred = interpolate_depth(p2_point, target_reps)
        p3_betas = p3_transferred[:target_reps]
        p3_gammas = p3_transferred[target_reps:]

        # SPSA 精修
        saved_iter = int(CONFIG["local_iterations"])
        CONFIG["local_iterations"] = int(CONFIG["continuation_iterations"])
        bt, gm, _, _ = planck_spsa_refine(
            edges, n_qubits, total_weight, p3_betas, p3_gammas,
            seed + 200 + idx,
        )
        CONFIG["local_iterations"] = saved_iter
        cut, _ = noisy_evaluate(
            edges, n_qubits, total_weight, bt, gm,
            shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
)
        candidates.append((bt, gm, cut))

    # 添加零填充
    z_betas = zero_padded[:target_reps]
    z_gammas = zero_padded[target_reps:]
    z_cut, _ = noisy_evaluate(
        edges, n_qubits, total_weight, z_betas, z_gammas,
        shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
        noise_model=CONFIG["noise_model"],
        noise_strength=float(CONFIG["noise_strength"]),
        rng=rng,
    )
    candidates.append((z_betas, z_gammas, z_cut))

    return candidates


def lcb_holdout_select(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    candidates: List[Tuple[np.ndarray, np.ndarray, float]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """LCB holdout 精选最终候选。"""
    n_reps = int(CONFIG["final_repeats"])
    best_bt, best_gm, best_lcb = None, None, -np.inf
    for bt, gm, _ in candidates:
        means, errors = noisy_evaluate(
            edges, n_qubits, total_weight, bt, gm,
            shots=int(CONFIG["shots"]), repeats=n_reps,
)
        score = float(lcb(np.array([means]), np.array([errors]))[0])
        if score > best_lcb:
            best_lcb = score
            best_bt = bt
            best_gm = gm
    return best_bt, best_gm, best_lcb


def main():
    import os, json, time
    from pathlib import Path

    rng = np.random.default_rng(int(CONFIG["graph_seed"]))
    _, edges_arr, total_weight = random_weighted_graph(
        int(CONFIG["num_nodes"]), int(CONFIG["graph_seed"]),
    )
    n_qubits = int(CONFIG["num_nodes"])
    target_reps = int(CONFIG["reps"])
    seed = int(CONFIG["graph_seed"])
    print(f"图: {n_qubits} 节点, {len(edges_arr)} 条边, 目标 p={target_reps}")

    start = time.time()

    # 1) p=2 源搜索
    print("p=2 源搜索...")
    source_betas, source_gammas = p2_source_search(
        edges_arr, n_qubits, total_weight, rng, seed,
    )
    print(f"  获得 {len(source_betas)} 个 p=2 候选")

    # 2) 深度延续
    print("深度延续 p=2 → 3...")
    candidates = continuation_candidates(
        edges_arr, n_qubits, total_weight,
        source_betas, source_gammas, rng, seed,
    )
    print(f"  生成 {len(candidates)} 个 p=3 候选（含安全地板）")

    # 3) LCB holdout 精选
    print("LCB holdout 精选...")
    bt, gm, l = lcb_holdout_select(
        edges_arr, n_qubits, total_weight, candidates, rng,
    )

    elapsed = time.time() - start
    print(f"\n最终近似比: {l:.4f}")
    print(f"耗时: {elapsed:.1f}s")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Depth-continued Noise-aware (TC)",
        "num_nodes": n_qubits, "reps": target_reps,
        "ratio": float(l),
        "gammas": gm.tolist(), "betas": bt.tolist(),
        "time": elapsed,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
