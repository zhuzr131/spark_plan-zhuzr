"""TensorCircuit QAOA 工具函数 — 腾讯量子云真机版。

电路构造、真机提交、含噪/理想采样、MaxCut 期望值计算。

后端模式（通过 BACKEND_MODE 切换）：
  - "local"      → 本地 statevector 理想采样（无需 token，快速调试）
  - "local_noisy" → 本地退极化噪声模拟
  - "cloud_sim"  → 腾讯云模拟器 "simulator:tc"
  - "real"       → 腾讯云真机（需设备名 + token）
"""

import time
import numpy as np
from typing import Tuple, List, Sequence, Optional
import tensorcircuit as tc

# ── 后端配置 ──
BACKEND_MODE = "local"           # "local" | "local_noisy" | "cloud_sim" | "real"
CLOUD_DEVICE_NAME = "simulator:tc"  # 真机时替换为 "tianxuan_s1" 等
NOISE_STRENGTH = 0.02             # local_noisy 模式的噪声强度
# 首次连接腾讯云时取消注释：
# tc.cloud.apis.set_token("你的 token")
# tc.cloud.apis.set_provider("tencent")

# 全局设备缓存（仅云端模式使用）
_cloud_device = None


def _get_cloud_device() -> Optional[object]:
    """懒加载获取云端设备对象。"""
    global _cloud_device
    if _cloud_device is None and BACKEND_MODE in ("cloud_sim", "real"):
        try:
            _cloud_device = tc.cloud.apis.get_device(CLOUD_DEVICE_NAME)
            print(f"[tc] 已连接设备: {CLOUD_DEVICE_NAME}")
        except Exception as e:
            print(f"[tc] 连接设备失败: {e}，回退到本地模式")
            return None
    return _cloud_device


def random_weighted_graph(
    num_nodes: int, graph_seed: int, edge_prob: float = 0.4,
) -> Tuple[np.ndarray, int, float]:
    """生成随机加权图，返回 (邻接表, n_qubits, total_weight)。"""
    rng = np.random.default_rng(graph_seed)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                w = rng.uniform(0.5, 3.0)
                edges.append((i, j, float(w)))
    total_weight = sum(w for _, _, w in edges)
    return np.asarray(edges, dtype=object), num_nodes, total_weight


def build_qaoa_circuit(
    edges: np.ndarray, n_qubits: int, betas: np.ndarray, gammas: np.ndarray,
) -> tc.Circuit:
    """构建 p 层 QAOA 电路。

    cost 层 → mixer 层 交替。每个 ZZ 旋转角 = 2 * γ * w_ij。
    """
    p = len(betas)
    c = tc.Circuit(n_qubits)
    for q in range(n_qubits):
        c.h(q)

    for layer in range(p):
        # cost 层：exp(-i γ_l w_ij Z_i Z_j) → 2 CNOT + 1 RZ
        for i, j, w in edges:
            c.cnot(int(i), int(j))
            c.rz(int(j), theta=2.0 * gammas[layer] * w)
            c.cnot(int(i), int(j))
        # mixer 层：exp(-i β_l X_i)
        for q in range(n_qubits):
            c.rx(q, theta=2.0 * betas[layer])

    return c


# ── 后端无关的 Cut 计算 ──

def _compute_cut_from_counts(counts: dict, edges: np.ndarray,
                             total_weight: float, n_qubits: int,
                             shots: int) -> float:
    """从测量计数计算 MaxCut 期望值。

    cut = Σ w_ij * P(bit_i ≠ bit_j) = Σ w_ij * (1 - <Z_i Z_j>) / 2
    """
    cut_val = 0.0
    for i, j, w in edges:
        diff = 0
        for bitstring, cnt in counts.items():
            # 注意 bitstring 索引：bitstring[0] 是最高位量子比特
            if bitstring[n_qubits - 1 - int(i)] != bitstring[n_qubits - 1 - int(j)]:
                diff += cnt
        cut_val += w * diff / shots
    return cut_val / total_weight


# ── 后端实现 ──

def _evaluate_local(circuit: tc.Circuit, edges: np.ndarray,
                    total_weight: float, shots: int) -> float:
    """本地理想采样。"""
    n_qubits = circuit._nqubits
    state = circuit.wavefunction()
    probs = np.abs(np.asarray(state)) ** 2
    indices = np.random.choice(2**n_qubits, size=shots, p=probs)
    counts = {}
    for idx in indices:
        bs = format(idx, f"0{n_qubits}b")
        counts[bs] = counts.get(bs, 0) + 1
    return _compute_cut_from_counts(counts, edges, total_weight, n_qubits, shots)


def _evaluate_local_noisy(circuit: tc.Circuit, edges: np.ndarray,
                          total_weight: float, shots: int) -> float:
    """本地退极化噪声模拟。"""
    n_qubits = circuit._nqubits
    state = circuit.wavefunction()
    ideal_probs = np.abs(np.asarray(state)) ** 2
    uniform = np.ones(2**n_qubits) / (2**n_qubits)
    mixed = (1.0 - NOISE_STRENGTH) * ideal_probs + NOISE_STRENGTH * uniform
    mixed /= mixed.sum()
    indices = np.random.choice(2**n_qubits, size=shots, p=mixed)
    counts = {}
    for idx in indices:
        bs = format(idx, f"0{n_qubits}b")
        counts[bs] = counts.get(bs, 0) + 1
    return _compute_cut_from_counts(counts, edges, total_weight, n_qubits, shots)


def _evaluate_cloud(circuit: tc.Circuit, edges: np.ndarray,
                    total_weight: float, shots: int) -> float:
    """提交到腾讯云（模拟器或真机）。"""
    device = _get_cloud_device()
    if device is None:
        # 回退到本地
        return _evaluate_local_noisy(circuit, edges, total_weight, shots)

    t = tc.cloud.apis.submit_task(device=device, circuit=circuit, shots=shots)
    counts = t.results()  # dict: {'0110': 245, '1001': 231, ...}
    if not isinstance(counts, dict):
        raise RuntimeError(f"云返回结果格式异常: {type(counts)}")
    return _compute_cut_from_counts(counts, edges, total_weight,
                                    circuit._nqubits, shots)


# ── 公共接口 ──

def evaluate(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas: np.ndarray, gammas: np.ndarray,
    shots: int = 1024,
) -> float:
    """单次评估 QAOA 电路的 MaxCut cut fraction。

    根据 BACKEND_MODE 自动选择后端路径。
    """
    circuit = build_qaoa_circuit(edges, n_qubits, betas, gammas)

    if BACKEND_MODE == "local":
        return _evaluate_local(circuit, edges, total_weight, shots)
    elif BACKEND_MODE == "local_noisy":
        return _evaluate_local_noisy(circuit, edges, total_weight, shots)
    elif BACKEND_MODE in ("cloud_sim", "real"):
        return _evaluate_cloud(circuit, edges, total_weight, shots)
    else:
        raise ValueError(f"未知 BACKEND_MODE: {BACKEND_MODE}")


def noisy_evaluate(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas: np.ndarray, gammas: np.ndarray,
    shots: int = 1024,
    repeats: int = 1,
) -> Tuple[float, float]:
    """多次评估，返回 (均值, 标准误)。"""
    values = []
    for _ in range(repeats):
        v = evaluate(edges, n_qubits, total_weight, betas, gammas, shots)
        values.append(v)
    arr = np.asarray(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


# ── 批量评估（真机并行提交） ──

def batched_evaluate(
    edges: np.ndarray, n_qubits: int, total_weight: float,
    betas_list: np.ndarray, gammas_list: np.ndarray,
    shots: int = 1024,
) -> np.ndarray:
    """批量评估多组参数的 cut fraction。

    云端模式下批量提交任务以节省等待时间。
    """
    circuits = []
    for i in range(len(betas_list)):
        c = build_qaoa_circuit(edges, n_qubits, betas_list[i], gammas_list[i])
        circuits.append(c)

    if BACKEND_MODE in ("cloud_sim", "real"):
        device = _get_cloud_device()
        if device is not None:
            tasks = tc.cloud.apis.submit_task(
                device=device, circuit=circuits, shots=shots,
            )
            results = np.array([
                _compute_cut_from_counts(t.results(), edges, total_weight,
                                         n_qubits, shots)
                for t in tasks
            ])
            return results

    # 本地逐条评估
    results = []
    for c in circuits:
        v = _evaluate_local(c, edges, total_weight, shots)
        results.append(v)
    return np.array(results)


# ── 设备信息 ──

def list_available_devices():
    """列出当前可用的腾讯云设备。"""
    try:
        providers = tc.cloud.apis.list_providers()
        print(f"Providers: {providers}")
        for p in providers:
            devices = tc.cloud.apis.list_devices(p, state="on")
            print(f"  {p} online: {devices}")
    except Exception as e:
        print(f"无法获取设备列表: {e}")


if __name__ == "__main__":
    # 快速冒烟测试
    print("=== 快速测试 ===")
    edges, n_qubits, total_weight = random_weighted_graph(6, 42)
    print(f"图: {n_qubits} 节点, {len(edges)} 条边, 总权重 {total_weight:.2f}")

    betas = np.array([0.3, 0.8, 1.2])
    gammas = np.array([0.5, 1.1, 0.7])
    cut, se = noisy_evaluate(edges, n_qubits, total_weight, betas, gammas,
                             shots=512, repeats=3)
    print(f"cut fraction = {cut:.4f} ± {se:.4f}")
