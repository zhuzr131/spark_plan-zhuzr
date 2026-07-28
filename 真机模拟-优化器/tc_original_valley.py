"""Original Valley QAOA 优化器 — TensorCircuit / 腾讯量子云真机版。

策略：粗网格扫描 → 多方向投影扫描 → Adam 坐标梯度精修。
"""

import numpy as np
from typing import Tuple, List
from tc_qaoa_utils import (
    random_weighted_graph, noisy_evaluate,
)

CONFIG = {
    "num_nodes": 8,
    "graph_seed": 42,
    "reps": 3,                    # QAOA 层数
    "noise_model": "depolarizing",
    "noise_strength": 0.02,
    "shots": 1024,
    "eval_repeats": 2,
    "coarse_per_dim": 2,          # 粗网格每维点数（2^6=64 点）
    "top_k": 4,                   # 粗筛保留数量
    "scan_directions": 8,         # 投影扫描方向数
    "scan_samples": 9,            # 每方向采样点数
    "scan_radius": 0.75,          # 扫描半径
    "refine_iterations": 30,      # 精修 Adam 迭代数
    "polish_iterations": 20,
    "learning_rate": 0.05,
    "coordinate_h": 0.02,
    "output_dir": "/Users/zhuzhengrong/Desktop/新型优化器/tc_valley_results",
}


def coarse_grid_scan(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """均匀网格扫描，返回 (所有点 β, 所有点 γ, cut 列表)。"""
    reps = int(CONFIG["reps"])
    n = int(CONFIG["coarse_per_dim"])
    n_dims = 2 * reps

    # beta ∈ [0, π), gamma ∈ [0, 2π)，各取 n 个点
    beta_grid = np.linspace(0.0, np.pi, n + 2)[1:-1]   # 去掉端点 0 和 π
    gamma_grid = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]

    grids = [beta_grid] * reps + [gamma_grid] * reps
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=1)  # (n^dims, dims)

    all_betas = points[:, :reps]
    all_gammas = points[:, reps:]
    cuts = []
    for i in range(len(points)):
        cut, _ = noisy_evaluate(
            edges, n_qubits, total_weight,
            all_betas[i].reshape(reps), all_gammas[i].reshape(reps),
            shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
)
        cuts.append(cut)
    return all_betas, all_gammas, cuts


def project_scan(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    base_betas: np.ndarray, base_gammas: np.ndarray,
    directions: int, samples: int, radius: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """从基点出发，沿随机方向扫描。返回最优 (β, γ, cut)。"""
    reps = len(base_betas)
    best_betas = base_betas.copy()
    best_gammas = base_gammas.copy()
    best_cut, _ = noisy_evaluate(
        edges, n_qubits, total_weight, best_betas, best_gammas,
        shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
        noise_model=CONFIG["noise_model"],
        noise_strength=float(CONFIG["noise_strength"]),
        rng=rng,
    )

    for _ in range(directions):
        # 随机方向（在高维参数空间）
        dir_betas = rng.normal(0, 1, reps)
        dir_gammas = rng.normal(0, 1, reps)
        norm = np.sqrt(np.sum(dir_betas**2) + np.sum(dir_gammas**2))
        dir_betas /= norm
        dir_gammas /= norm

        for s in range(samples):
            t = (s / (samples - 1) - 0.5) * 2.0 * radius if samples > 1 else 0.0
            test_betas = (base_betas + t * dir_betas) % np.pi
            test_gammas = (base_gammas + t * dir_gammas) % (2.0 * np.pi)
            cut, _ = noisy_evaluate(
                edges, n_qubits, total_weight, test_betas, test_gammas,
                shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
                noise_model=CONFIG["noise_model"],
                noise_strength=float(CONFIG["noise_strength"]),
                rng=rng,
            )
            if cut > best_cut:
                best_cut = cut
                best_betas = test_betas.copy()
                best_gammas = test_gammas.copy()

    return best_betas, best_gammas, best_cut


def coordinate_adam_refine(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas: np.ndarray, gammas: np.ndarray,
    iterations: int, lr: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """坐标梯度 Adam 精修（与 Plain Adam 相同的核心优化器）。"""
    reps = len(betas)
    h = float(CONFIG["coordinate_h"])
    m_b, v_b = np.zeros(reps), np.zeros(reps)
    m_g, v_g = np.zeros(reps), np.zeros(reps)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_betas = betas.copy()
    best_gammas = gammas.copy()
    best_cut, _ = noisy_evaluate(
        edges, n_qubits, total_weight, best_betas, best_gammas,
        shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
        noise_model=CONFIG["noise_model"],
        noise_strength=float(CONFIG["noise_strength"]),
        rng=rng,
    )

    for t in range(1, iterations + 1):
        g_b, g_g = _coordinate_gradient(
            edges, n_qubits, total_weight, betas, gammas, h, rng,
        )
        m_b = beta1 * m_b + (1.0 - beta1) * g_b
        v_b = beta2 * v_b + (1.0 - beta2) * g_b**2
        m_g = beta1 * m_g + (1.0 - beta1) * g_g
        v_g = beta2 * v_g + (1.0 - beta2) * g_g**2
        m_b_hat = m_b / (1.0 - beta1**t)
        v_b_hat = v_b / (1.0 - beta2**t)
        m_g_hat = m_g / (1.0 - beta1**t)
        v_g_hat = v_g / (1.0 - beta2**t)

        betas += lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        gammas += lr * m_g_hat / (np.sqrt(v_g_hat) + eps)
        betas %= np.pi
        gammas %= (2.0 * np.pi)

        cut, _ = noisy_evaluate(
            edges, n_qubits, total_weight, betas, gammas,
            shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
)
        if cut > best_cut:
            best_cut = cut
            best_betas = betas.copy()
            best_gammas = gammas.copy()

    return best_betas, best_gammas, best_cut


def _coordinate_gradient(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas: np.ndarray, gammas: np.ndarray, h: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    reps = len(betas)
    grad_b = np.zeros(reps)
    grad_g = np.zeros(reps)
    r = max(int(CONFIG["eval_repeats"]), 1)
    cfg = {
        "shots": int(CONFIG["shots"]), "repeats": r,
        "noise_model": CONFIG["noise_model"],
        "noise_strength": float(CONFIG["noise_strength"]),
    }
    for i in range(reps):
        bp, bm = betas.copy(), betas.copy()
        bp[i] += h; bm[i] -= h
        fp, _ = noisy_evaluate(edges, n_qubits, total_weight, bp, gammas, rng=rng, **cfg)
        fm, _ = noisy_evaluate(edges, n_qubits, total_weight, bm, gammas, rng=rng, **cfg)
        grad_b[i] = (fp - fm) / (2.0 * h)

        gp, gm = gammas.copy(), gammas.copy()
        gp[i] += h; gm[i] -= h
        fp, _ = noisy_evaluate(edges, n_qubits, total_weight, betas, gp, rng=rng, **cfg)
        fm, _ = noisy_evaluate(edges, n_qubits, total_weight, betas, gm, rng=rng, **cfg)
        grad_g[i] = (fp - fm) / (2.0 * h)
    return grad_b, grad_g


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

    # 1) 粗网格扫描
    print("粗网格扫描...")
    all_b, all_g, cuts = coarse_grid_scan(
        edges_arr, n_qubits, total_weight, rng,
    )
    order = np.argsort(cuts)[::-1][:int(CONFIG["top_k"])]
    print(f"  top-{len(order)}: {np.round(np.array(cuts)[order], 4)}")

    # 2) 投影扫描
    best_bt, best_gm, best_cut = None, None, -np.inf
    for idx in order:
        bt, gm, cut = project_scan(
            edges_arr, n_qubits, total_weight,
            all_b[idx].reshape(reps), all_g[idx].reshape(reps),
            int(CONFIG["scan_directions"]), int(CONFIG["scan_samples"]),
            float(CONFIG["scan_radius"]), rng,
        )
        if cut > best_cut:
            best_cut = cut
            best_bt = bt
            best_gm = gm

    # 3) Adam 精修
    print("Adam 精修...")
    bt, gm, cut = coordinate_adam_refine(
        edges_arr, n_qubits, total_weight, best_bt, best_gm,
        int(CONFIG["refine_iterations"]), float(CONFIG["learning_rate"]),
        rng,
    )
    # 4) 最终 polish
    bt, gm, cut = coordinate_adam_refine(
        edges_arr, n_qubits, total_weight, bt, gm,
        int(CONFIG["polish_iterations"]), float(CONFIG["learning_rate"]) * 0.5,
        rng,
    )
    elapsed = time.time() - start

    print(f"\n最终近似比: {cut:.4f}")
    print(f"耗时: {elapsed:.1f}s")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Original Valley (TC)",
        "num_nodes": n_qubits, "reps": reps,
        "ratio": float(cut),
        "gammas": gm.tolist(), "betas": bt.tolist(),
        "time": elapsed,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
