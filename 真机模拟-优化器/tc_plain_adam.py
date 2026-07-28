"""Plain Adam QAOA 优化器 — TensorCircuit / 腾讯量子云真机版。

策略：多起点坐标梯度 Adam 上升 + 最终全参数 polish。
每轮用一个含噪评估的真实 quantum circuit 采样。
"""

import numpy as np
from typing import Tuple, Optional
from tc_qaoa_utils import (
    random_weighted_graph, noisy_evaluate,
)

CONFIG = {
    "num_nodes": 8,
    "graph_seed": 42,
    "reps": 3,                    # QAOA 层数
    "noise_model": "depolarizing",
    "noise_strength": 0.02,
    "shots": 1024,                # 每轮采样数（真机上可调）
    "eval_repeats": 2,            # 每点重复评估次数
    "restarts": 3,                # 随机起点数
    "iterations": 50,             # 每起点 Adam 迭代数
    "polish_iterations": 30,      # 最终 polish 迭代数
    "learning_rate": 0.05,
    "coordinate_h": 0.02,         # 坐标梯度扰动步长
    "output_dir": "/Users/zhuzhengrong/Desktop/新型优化器/tc_plain_adam_results",
}


def coordinate_gradient(
    edges: np.ndarray,
    n_qubits: int,
    total_weight: float,
    betas: np.ndarray,
    gammas: np.ndarray,
    h: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """坐标方向有限差分梯度。

    对每个 β_i, γ_i 分别做 ±h 扰动，计算含噪 cut 差值。
    """
    n_params = len(betas)
    grad_betas = np.zeros(n_params)
    grad_gammas = np.zeros(n_params)
    reps_eval = max(int(CONFIG["eval_repeats"]), 1)

    for i in range(n_params):
        # β 方向的扰动
        bp = betas.copy()
        bm = betas.copy()
        bp[i] += h
        bm[i] -= h
        fp, _ = noisy_evaluate(
            edges, n_qubits, total_weight, bp, gammas,
            shots=int(CONFIG["shots"]), repeats=reps_eval,
)
        fm, _ = noisy_evaluate(
            edges, n_qubits, total_weight, bm, gammas,
            shots=int(CONFIG["shots"]), repeats=reps_eval,
)
        grad_betas[i] = (fp - fm) / (2.0 * h)

        # γ 方向的扰动
        gp = gammas.copy()
        gm = gammas.copy()
        gp[i] += h
        gm[i] -= h
        fp, _ = noisy_evaluate(
            edges, n_qubits, total_weight, betas, gp,
            shots=int(CONFIG["shots"]), repeats=reps_eval,
)
        fm, _ = noisy_evaluate(
            edges, n_qubits, total_weight, betas, gm,
            shots=int(CONFIG["shots"]), repeats=reps_eval,
)
        grad_gammas[i] = (fp - fm) / (2.0 * h)

    return grad_betas, grad_gammas


def adam_optimize(
    edges: np.ndarray,
    n_qubits: int,
    total_weight: float,
    betas: np.ndarray,
    gammas: np.ndarray,
    iterations: int,
    lr: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, list]:
    """含噪坐标 Adam 上升。返回 (最优 β, 最优 γ, 最优 cut, 历史)。"""
    reps = len(betas)
    h = float(CONFIG["coordinate_h"])

    m_b = np.zeros(reps)
    v_b = np.zeros(reps)
    m_g = np.zeros(reps)
    v_g = np.zeros(reps)
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
    history = [best_cut]

    for t in range(1, iterations + 1):
        g_b, g_g = coordinate_gradient(
            edges, n_qubits, total_weight, betas, gammas, h, rng,
        )
        m_b = beta1 * m_b + (1.0 - beta1) * g_b
        v_b = beta2 * v_b + (1.0 - beta2) * g_b ** 2
        m_g = beta1 * m_g + (1.0 - beta1) * g_g
        v_g = beta2 * v_g + (1.0 - beta2) * g_g ** 2
        m_b_hat = m_b / (1.0 - beta1 ** t)
        v_b_hat = v_b / (1.0 - beta2 ** t)
        m_g_hat = m_g / (1.0 - beta1 ** t)
        v_g_hat = v_g / (1.0 - beta2 ** t)

        betas += lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        gammas += lr * m_g_hat / (np.sqrt(v_g_hat) + eps)
        betas %= np.pi
        gammas %= (2.0 * np.pi)

        cut, _ = noisy_evaluate(
            edges, n_qubits, total_weight, betas, gammas,
            shots=int(CONFIG["shots"]), repeats=int(CONFIG["eval_repeats"]),
)
        history.append(cut)
        if cut > best_cut:
            best_cut = cut
            best_betas = betas.copy()
            best_gammas = gammas.copy()

    return best_betas, best_gammas, best_cut, history


def main():
    import os, json, time
    from pathlib import Path

    rng = np.random.default_rng(int(CONFIG["graph_seed"]))
    edges_arr, n_qubits, total_weight = random_weighted_graph(
        int(CONFIG["num_nodes"]), int(CONFIG["graph_seed"]),
    )
    reps = int(CONFIG["reps"])
    print(f"图: {n_qubits} 节点, {len(edges_arr)} 条边, 总权重 {total_weight:.2f}")

    start = time.time()
    best_overall_betas = None
    best_overall_gammas = None
    best_overall_cut = -np.inf
    all_histories = []

    for restart in range(int(CONFIG["restarts"])):
        rng_restart = np.random.default_rng(int(CONFIG["graph_seed"]) + restart * 100)
        init_betas = rng_restart.uniform(0, np.pi, size=reps)
        init_gammas = rng_restart.uniform(0, 2 * np.pi, size=reps)

        betas, gammas, cut, hist = adam_optimize(
            edges_arr, n_qubits, total_weight,
            init_betas, init_gammas,
            int(CONFIG["iterations"]), float(CONFIG["learning_rate"]),
            rng_restart,
        )
        all_histories.append(hist)
        if cut > best_overall_cut:
            best_overall_cut = cut
            best_overall_betas = betas
            best_overall_gammas = gammas
        print(f"  restart {restart}: best cut frac = {cut:.4f}")

    # 最终 polish
    final_betas, final_gammas, final_cut, _ = adam_optimize(
        edges_arr, n_qubits, total_weight,
        best_overall_betas, best_overall_gammas,
        int(CONFIG["polish_iterations"]), float(CONFIG["learning_rate"]) * 0.5,
        rng,
    )
    elapsed = time.time() - start

    print(f"\n最终近似比: {final_cut:.4f}")
    print(f"γ: {np.round(final_gammas, 3)}")
    print(f"β: {np.round(final_betas, 3)}")
    print(f"耗时: {elapsed:.1f}s")

    # 保存结果
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Plain Adam (TC)",
        "num_nodes": n_qubits,
        "reps": reps,
        "ratio": float(final_cut),
        "gammas": final_gammas.tolist(),
        "betas": final_betas.tolist(),
        "time": elapsed,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果保存到 {out_dir}")


if __name__ == "__main__":
    main()
