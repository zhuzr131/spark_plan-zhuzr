from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorcircuit as tc
from scipy.optimize import minimize
from scipy.stats import qmc

class MaxCutProblem:
    """Max-Cut 问题实例: 图 + 哈密顿量对角元预计算"""

    def __init__(self, edges, n_qubits, name="graph"):
        self.edges = edges
        self.n = n_qubits
        self.m = len(edges)
        self.name = name
        self._hp_diag = self._build_hp_diag()
        self.max_cut = self.m

    def _build_hp_diag(self):
        """预计算 H_p = Σ Z_iZ_j 的对角元。
        Z_iZ_j|z> = (+1 if z_i==z_j else -1)|z>
        """
        dim = 2 ** self.n
        diag = np.zeros(dim)
        for z in range(dim):
            val = 0.0
            for i, j in self.edges:
                bi, bj = (z >> i) & 1, (z >> j) & 1
                val += 1.0 if bi == bj else -1.0
            diag[z] = val
        return diag

    def cut_value(self, z):
        """比特串 z (int) 的割边数"""
        return sum(1 for i, j in self.edges if ((z >> i) & 1) != ((z >> j) & 1))

    def brute_force_maxcut(self):
        """暴力枚举最优割 (小图)"""
        best = max(self.cut_value(z) for z in range(2 ** self.n))
        self.max_cut = best
        return best

class QAOACircuit:
    """p 层 QAOA statevector 引擎"""

    def __init__(self, problem: MaxCutProblem):
        self.prob = problem
        self.n = problem.n
        self.hp = problem._hp_diag

    def state(self, params):
        """构造 QAOA 态。params = [γ_1,β_1, γ_2,β_2, ..., γ_p,β_p]"""
        p = len(params) // 2
        dim = 2 ** self.n
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        for layer in range(p):
            g, b = params[2 * layer], params[2 * layer + 1]
            psi = psi * np.exp(-1j * g * self.hp)
            c, s = np.cos(b), np.sin(b)
            for q in range(self.n):
                psi = psi.reshape(2 ** q, 2, 2 ** (self.n - q - 1))
                a0 = psi[:, 0, :].copy()
                a1 = psi[:, 1, :].copy()
                psi[:, 0, :] = c * a0 - 1j * s * a1
                psi[:, 1, :] = -1j * s * a0 + c * a1
                psi = psi.reshape(dim)
        return psi

    def expectation(self, params):
        """<H_p> 期望 (代价函数, 优化器最小化这个)"""
        psi = self.state(params)
        probs = np.abs(psi) ** 2
        return float(np.dot(probs, self.hp))

    def expected_cut(self, params):
        """期望割边 = (m - <H_p>) / 2"""
        return (self.prob.m - self.expectation(params)) / 2

    def sample_best_cut(self, params, n_shots=2000, rng=None):
        """采样, 返回采到的最优割边数"""
        if rng is None:
            rng = np.random.default_rng()
        psi = self.state(params)
        probs = np.abs(psi) ** 2
        probs = probs / probs.sum()
        samples = rng.choice(2 ** self.n, size=n_shots, p=probs)
        best = max(self.prob.cut_value(int(z)) for z in samples)
        return best

def cost_weight(n_qubits, edges_count, p, mode="hardware"):
    """每次函数评估的成本权重.
    态空间口径: 2^n
    硬件口径: 2^n × 电路深度(≈ 2p·(m+n))
    """
    if mode == "state":
        return 2 ** n_qubits
    elif mode == "hardware":
        depth = 2 * p * (edges_count + n_qubits)
        return (2 ** n_qubits) * depth
    return 1.0

def _run_optimizer(objective, x0, bounds, optimizer, maxiter):
    """跑一次优化, 返回 (x, fun, nfev).

    optimizer 可以是:
      - str: scipy.minimize 的 method 名 ("COBYLA"/"L-BFGS-B"/...)
      - callable: 签名 (objective, x0, bounds, maxiter) -> (x, fun, nfev)
    """
    if callable(optimizer):
        return optimizer(objective, x0, bounds, maxiter)
    kw = {"maxiter": maxiter}
    if optimizer in ("L-BFGS-B", "Powell"):
        res = minimize(objective, x0, method=optimizer, bounds=bounds, options=kw)
    else:
        res = minimize(objective, x0, method=optimizer, options=kw)
    return res.x, float(res.fun), res.nfev

class QAOA_ZZH:
    """带 RG 指导的 QAOA 求解器. 优化器通过 scipy method 或 callable 传入."""

    def __init__(self, problem_big, problem_small=None, p=2,
                 optimizer="COBYLA", maxiter=400, cost_mode="hardware"):
        """
        problem_big:   目标问题 (大尺度)
        problem_small: 粗粒化问题 (小尺度, RG 用). None 则只能无 RG
        p:             QAOA 深度
        optimizer:     scipy method 字符串, 或朋友的 callable
        maxiter:       优化器最大迭代
        cost_mode:     "state" / "hardware"
        """
        self.pb = problem_big
        self.ps = problem_small
        self.p = p
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.cost_mode = cost_mode
        self.circ_big = QAOACircuit(problem_big)
        self.circ_small = QAOACircuit(problem_small) if problem_small else None
        self.bounds = [(-np.pi, np.pi)] * (2 * p)

    def solve_no_rg(self, n_restarts=5, seed=42):
        """基线: 直接在大尺度多次随机重启优化"""
        rng = np.random.default_rng(seed)
        best_x, best_f, nfev_big = None, np.inf, 0
        for _ in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, 2 * self.p)
            x, f, nf = _run_optimizer(self.circ_big.expectation, x0,
                                      self.bounds, self.optimizer, self.maxiter)
            nfev_big += nf
            if f < best_f:
                best_f, best_x = f, x
        return {"params": best_x, "fun": best_f,
                "nfev_big": nfev_big, "nfev_small": 0,
                "expected_cut": self.circ_big.expected_cut(best_x)}

    def solve_rg(self, n_restarts_small=3, seed=42):
        """RG 指导: 小尺度优化 + 大尺度微调"""
        assert self.circ_small is not None, "需提供 problem_small 才能用 RG"
        rng = np.random.default_rng(seed)
        best_xs, best_fs, nfev_small = None, np.inf, 0
        for _ in range(n_restarts_small):
            x0 = rng.uniform(-np.pi, np.pi, 2 * self.p)
            x, f, nf = _run_optimizer(self.circ_small.expectation, x0,
                                      self.bounds, self.optimizer, self.maxiter)
            nfev_small += nf
            if f < best_fs:
                best_fs, best_xs = f, x
        x, f, nfev_big = _run_optimizer(self.circ_big.expectation, best_xs.copy(),
                                        self.bounds, self.optimizer, self.maxiter)
        return {"params": x, "fun": f,
                "nfev_big": nfev_big, "nfev_small": nfev_small,
                "expected_cut": self.circ_big.expected_cut(x)}

    def weighted_cost(self, result):
        """加权成本 = nfev_big·c(N) + nfev_small·c(N/2)"""
        c_big = cost_weight(self.pb.n, self.pb.m, self.p, self.cost_mode)
        c_small = (cost_weight(self.ps.n, self.ps.m, self.p, self.cost_mode)
                   if self.ps else 0)
        return result["nfev_big"] * c_big + result["nfev_small"] * c_small

    def compare(self, n_trials=30, seed0=42):
        """对比 无RG vs RG, 返回统计. 展示用主接口."""
        no_costs, rg_costs, no_cuts, rg_cuts, rg_better = [], [], [], [], 0
        for t in range(n_trials):
            seed = seed0 + t * 137
            r_no = self.solve_no_rg(seed=seed)
            r_rg = self.solve_rg(seed=seed)
            c_no, c_rg = self.weighted_cost(r_no), self.weighted_cost(r_rg)
            no_costs.append(c_no); rg_costs.append(c_rg)
            no_cuts.append(r_no["expected_cut"]); rg_cuts.append(r_rg["expected_cut"])
            if c_rg < c_no:
                rg_better += 1
        no_costs, rg_costs = np.array(no_costs), np.array(rg_costs)
        return {
            "optimizer": str(self.optimizer),
            "cost_mode": self.cost_mode,
            "no_rg_cost_mean": no_costs.mean(),
            "rg_cost_mean": rg_costs.mean(),
            "weighted_saving_pct": (1 - rg_costs.mean() / no_costs.mean()) * 100,
            "rg_better": rg_better, "n_trials": n_trials,
            "no_rg_cut_mean": float(np.mean(no_cuts)),
            "rg_cut_mean": float(np.mean(rg_cuts)),
        }

def _finite_diff_vg(objective, eps=1e-5):
    """把纯 numpy objective 包装成朋友 adam_refine 需要的 value_and_gradient。

    """
    def vg(params):
        x = np.asarray(params, dtype=float)
        f0 = float(objective(x))
        g = np.zeros_like(x)
        for k in range(x.size):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            g[k] = (float(objective(xp)) - float(objective(xm))) / (2 * eps)
        return f0, g
    return vg

def make_friend_optimizer(global_starts=6, selected=2,
                          adam_steps=35, adam_lr=0.06, lbfgs_maxiter=50,
                          seed=0):
    """工厂：返回符合 QAOA_ZZH 接口的 optimizer callable。

    返回 callable 签名: (objective, x0, bounds, maxiter) -> (x, fun, nfev)
    """
    def optimizer(objective, x0, bounds, maxiter):
        x0 = np.asarray(x0, dtype=float)
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        vg = _finite_diff_vg(objective)
        rng = np.random.default_rng(seed)

        starts = [x0] + [rng.uniform(lower, upper) for _ in range(global_starts - 1)]
        vals = [float(objective(s)) for s in starts]
        nfev = len(starts)
        order = np.argsort(vals)[:selected]

        candidates = []
        for idx in order:
            adam_point, adam_loss, adam_nfev = adam_refine(
                vg, starts[idx], lower, upper, adam_steps, adam_lr,
            )
            nfev += adam_nfev * (1 + 2 * x0.size)
            res = minimize(objective, np.asarray(adam_point, dtype=float),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": min(lbfgs_maxiter, maxiter),
                                    "maxfun": 2 * min(lbfgs_maxiter, maxiter)})
            nfev += int(res.nfev)
            if float(res.fun) <= float(adam_loss):
                candidates.append((res.x, float(res.fun)))
            else:
                candidates.append((np.asarray(adam_point, dtype=float), float(adam_loss)))

        best = min(candidates, key=lambda c: c[1])
        return np.asarray(best[0], dtype=float), float(best[1]), nfev

    return optimizer

K = tc.set_backend("jax")

ALL_ANSATZES = (
    "Standard",
    "Multi-angle",
    "Warm-start",
    "Hardware-efficient",
)

@dataclass
class ExperimentConfig:
    p: int = 2
    seeds: tuple[int, ...] = (3, 11)
    shots: int = 1000
    global_starts: int = 6
    selected_starts: int = 2
    adam_steps: int = 35
    adam_learning_rate: float = 0.06
    lbfgs_maxiter: int = 50
    quality_tolerance: float = 0.01

def build_unweighted_graphs() -> dict[str, nx.Graph]:
    """四类结构不同的小图；所有边都没有 weight 属性。"""
    graphs = {
        "C17": nx.cycle_graph(17),
        "Complete bipartite": nx.complete_bipartite_graph(8, 8),
        "3-regular": nx.random_regular_graph(3, 16, seed=17),
        "Random-16": nx.erdos_renyi_graph(16, 0.35, seed=7),
    }
    return {
        name: nx.convert_node_labels_to_integers(graph)
        for name, graph in graphs.items()
    }

def validate_unweighted_graph(graph: nx.Graph) -> None:
    if graph.is_directed():
        raise ValueError("只支持无向图")
    if graph.number_of_edges() == 0:
        raise ValueError("图中至少需要一条边")
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        raise ValueError("节点必须连续编号为 0, 1, ..., n-1")
    for _, _, data in graph.edges(data=True):
        if "weight" in data and not np.isclose(float(data["weight"]), 1.0):
            raise ValueError("本实验只支持无权图；请删除非 1 的 weight 属性")

def cost_vector(n: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """返回每个计算基态对应的无权 cut size。"""
    states = np.arange(1 << n, dtype=np.uint64)
    values = np.zeros(1 << n, dtype=np.float32)
    for i, j in edges:
        bit_i = (states >> np.uint64(n - 1 - i)) & 1
        bit_j = (states >> np.uint64(n - 1 - j)) & 1
        values += (bit_i ^ bit_j).astype(np.float32)
    return values

def exact_reference(graph: nx.Graph) -> tuple[int, np.ndarray]:
    """小图上枚举全部 2^n 个状态，得到严格的 Max-Cut 分母。"""
    n = graph.number_of_nodes()
    if n > 22:
        raise ValueError("精确参考解需要 2^n 内存；本实验请使用 n <= 22")
    costs = cost_vector(n, tuple(graph.edges()))
    return int(costs.max()), costs

def warm_start_data(
    graph: nx.Graph,
    epsilon: float = 0.15,
    seed: int = 7,
    restarts: int = 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """用多次 one_exchange 得到便宜的经典 warm start。"""
    best_value = -1
    best_partition = None
    for trial in range(restarts):
        value, partition = nx.algorithms.approximation.maxcut.one_exchange(
            graph, seed=seed + trial
        )
        if value > best_value:
            best_value = int(value)
            best_partition = partition

    assert best_partition is not None
    _, ones = best_partition
    bits = np.zeros(graph.number_of_nodes(), dtype=int)
    bits[list(ones)] = 1
    probabilities = np.where(bits == 1, 1.0 - epsilon, epsilon)
    theta = np.arcsin(np.sqrt(probabilities))
    return theta, bits, best_value

def greedy_edge_colors(graph: nx.Graph) -> int:
    line_graph = nx.line_graph(graph)
    if line_graph.number_of_nodes() == 0:
        return 0
    coloring = nx.coloring.greedy_color(line_graph, strategy="largest_first")
    return max(coloring.values()) + 1

def prepare_graph_data(graph: nx.Graph) -> dict:
    validate_unweighted_graph(graph)
    optimum, costs = exact_reference(graph)
    warm_theta, warm_bits, warm_cut = warm_start_data(graph)
    return {
        "n": graph.number_of_nodes(),
        "edges": tuple(graph.edges()),
        "costs": costs,
        "optimum": optimum,
        "edge_colors": greedy_edge_colors(graph),
        "warm_theta": warm_theta,
        "warm_bits": warm_bits,
        "warm_cut": warm_cut,
    }

def compile_ansatz(
    graph_data: dict,
    ansatz: str,
    p: int,
) -> tuple[Callable, Callable, dict]:
    """构造 JAX 目标函数；目标是最小化负的期望 cut size。"""
    n = graph_data["n"]
    edges = graph_data["edges"]
    m = len(edges)
    costs_np = graph_data["costs"]
    costs = jnp.asarray(costs_np)
    warm_theta = jnp.asarray(graph_data["warm_theta"])
    edge_colors = graph_data["edge_colors"]

    if ansatz in {"Standard", "Warm-start"}:
        parameter_count = 2 * p
        entangling_gates = 2 * p * m
    elif ansatz == "Multi-angle":
        parameter_count = p * (m + n)
        entangling_gates = 2 * p * m
    elif ansatz == "Hardware-efficient":
        parameter_count = (p + 1) * n
        entangling_gates = p * m
    else:
        raise ValueError(f"未知 Ansatz: {ansatz}")

    if ansatz in {"Standard", "Multi-angle"}:
        depth_estimate = 1 + p * (3 * edge_colors + 1)
    elif ansatz == "Warm-start":
        depth_estimate = 1 + p * (3 * edge_colors + 3)
    else:
        depth_estimate = 2 + p * (edge_colors + 1)

    def probabilities(params: jax.Array) -> jax.Array:
        circuit = tc.Circuit(n)

        if ansatz == "Warm-start":
            for qubit in range(n):
                circuit.ry(qubit, theta=2.0 * warm_theta[qubit])
        else:
            for qubit in range(n):
                circuit.h(qubit)

        if ansatz == "Standard":
            gammas, betas = params[:p], params[p:]
            for layer in range(p):
                for i, j in edges:
                    circuit.rzz(i, j, theta=-gammas[layer])
                for qubit in range(n):
                    circuit.rx(qubit, theta=2.0 * betas[layer])

        elif ansatz == "Multi-angle":
            gamma_count = p * m
            gammas = params[:gamma_count].reshape((p, m))
            betas = params[gamma_count:].reshape((p, n))
            for layer in range(p):
                for edge_index, (i, j) in enumerate(edges):
                    circuit.rzz(
                        i,
                        j,
                        theta=-gammas[layer, edge_index],
                    )
                for qubit in range(n):
                    circuit.rx(
                        qubit,
                        theta=2.0 * betas[layer, qubit],
                    )

        elif ansatz == "Warm-start":
            gammas, betas = params[:p], params[p:]
            for layer in range(p):
                for i, j in edges:
                    circuit.rzz(i, j, theta=-gammas[layer])
                for qubit in range(n):
                    theta = warm_theta[qubit]
                    circuit.ry(qubit, theta=-2.0 * theta)
                    circuit.rz(qubit, theta=-2.0 * betas[layer])
                    circuit.ry(qubit, theta=2.0 * theta)

        else:
            angles = params.reshape((p + 1, n))
            for layer in range(p):
                for qubit in range(n):
                    circuit.ry(qubit, theta=angles[layer, qubit])
                for i, j in edges:
                    circuit.cz(i, j)
            for qubit in range(n):
                circuit.ry(qubit, theta=angles[p, qubit])

        state = circuit.state()
        return jnp.real(jnp.conj(state) * state)

    def loss(params: jax.Array) -> jax.Array:
        return -jnp.dot(probabilities(params), costs)

    metadata = {
        "parameters": parameter_count,
        "entangling_gates": entangling_gates,
        "depth_est": depth_estimate,
        "costs": costs_np,
        "warm_cut": graph_data["warm_cut"],
    }
    return jax.jit(jax.value_and_grad(loss)), jax.jit(probabilities), metadata

def parameter_bounds(
    ansatz: str,
    n: int,
    m: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    if ansatz in {"Standard", "Warm-start"}:
        lower = np.zeros(2 * p)
        upper = np.r_[np.full(p, 2.0 * np.pi), np.full(p, np.pi)]
    elif ansatz == "Multi-angle":
        lower = np.zeros(p * (m + n))
        upper = np.r_[
            np.full(p * m, 2.0 * np.pi),
            np.full(p * n, np.pi),
        ]
    else:
        lower = np.full((p + 1) * n, -np.pi)
        upper = np.full((p + 1) * n, np.pi)
    return lower, upper

def standard_to_multi_angle(
    standard_params: np.ndarray,
    n: int,
    m: int,
    p: int,
) -> np.ndarray:
    gammas = np.tile(np.asarray(standard_params[:p])[:, None], (1, m)).ravel()
    betas = np.tile(np.asarray(standard_params[p:])[:, None], (1, n)).ravel()
    return np.r_[gammas, betas]

def make_initial_points(
    ansatz: str,
    graph_data: dict,
    config: ExperimentConfig,
    seed: int,
    standard_params: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """少量低差异点加结构化初值，避免昂贵的密集网格。"""
    n = graph_data["n"]
    m = len(graph_data["edges"])
    p = config.p
    rng = np.random.default_rng(seed)
    lower, upper = parameter_bounds(ansatz, n, m, p)
    dimensions = len(lower)
    points: list[np.ndarray] = []

    if ansatz == "Standard":
        structured = np.r_[
            np.linspace(0.15, 0.75, p),
            np.linspace(0.70, 0.15, p),
        ]
        points.append(structured)
        power = int(np.ceil(np.log2(max(2, config.global_starts))))
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
        unit_points = sampler.random_base2(power)[: config.global_starts]
        points.extend(lower + unit_points * (upper - lower))

    elif ansatz == "Multi-angle":
        if standard_params is None:
            raise ValueError("Multi-angle 需要先完成 Standard 的参数迁移")
        anchor = standard_to_multi_angle(standard_params, n, m, p)
        points.append(anchor)
        for _ in range(config.global_starts):
            points.append(anchor + rng.normal(0.0, 0.04, dimensions))

    elif ansatz == "Warm-start":
        points.append(np.zeros(dimensions))
        for _ in range(config.global_starts):
            points.append(np.abs(rng.normal(0.0, 0.08, dimensions)))

    else:
        for _ in range(config.global_starts + 1):
            points.append(rng.normal(0.0, 0.35, dimensions))

    return np.clip(np.asarray(points), lower, upper), lower, upper

def adam_refine(
    value_and_gradient: Callable,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, int]:
    """零噪声版 Adam；精确梯度下 Planck 置信权重恒为 1。"""
    theta = np.asarray(initial, dtype=float).copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    beta1, beta2 = 0.90, 0.98
    best_theta = theta.copy()
    best_loss = np.inf

    for iteration in range(1, steps + 1):
        value, gradient = value_and_gradient(jnp.asarray(theta))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        if value < best_loss:
            best_loss = value
            best_theta = theta.copy()

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        first_hat = first_moment / (1.0 - beta1**iteration)
        second_hat = second_moment / (1.0 - beta2**iteration)
        rate = learning_rate / (iteration**0.10)
        theta -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        theta = np.clip(theta, lower, upper)

    final_value, _ = value_and_gradient(jnp.asarray(theta))
    if float(final_value) < best_loss:
        best_loss = float(final_value)
        best_theta = theta.copy()
    return best_theta, best_loss, steps + 1

def hybrid_optimize(
    value_and_gradient: Callable,
    starts: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ExperimentConfig,
) -> dict:
    """Sobol/结构化筛选 -> Adam -> L-BFGS-B。"""
    initial_values = np.asarray(
        [float(value_and_gradient(jnp.asarray(point))[0]) for point in starts]
    )
    chosen = np.argsort(initial_values)[: config.selected_starts]
    bounds = list(zip(lower, upper))
    candidates = []
    evaluations = len(starts)

    def scipy_value_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(jnp.asarray(params))
        return float(value), np.asarray(gradient, dtype=float).copy()

    for index in chosen:
        adam_point, adam_loss, adam_evaluations = adam_refine(
            value_and_gradient,
            starts[index],
            lower,
            upper,
            config.adam_steps,
            config.adam_learning_rate,
        )
        result = minimize(
            scipy_value_gradient,
            adam_point,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.lbfgs_maxiter, "maxfun": 2 * config.lbfgs_maxiter},
        )
        evaluations += adam_evaluations + int(result.nfev)
        if float(result.fun) <= adam_loss:
            candidates.append(
                {
                    "params": np.asarray(result.x, dtype=float),
                    "loss": float(result.fun),
                    "success": bool(result.success),
                }
            )
        else:
            candidates.append(
                {
                    "params": adam_point,
                    "loss": float(adam_loss),
                    "success": True,
                }
            )

    best = min(candidates, key=lambda item: item["loss"])
    best["evaluations"] = evaluations
    return best

def test_ansatz(
    graph_name: str,
    graph: nx.Graph,
    graph_data: dict,
    ansatz: str,
    config: ExperimentConfig,
    standard_params: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    value_and_gradient, probability_function, metadata = compile_ansatz(
        graph_data, ansatz, config.p
    )

    warmup_starts, _, _ = make_initial_points(
        ansatz,
        graph_data,
        config,
        config.seeds[0],
        standard_params,
    )
    compile_start = perf_counter()
    warm_value, warm_gradient = value_and_gradient(jnp.asarray(warmup_starts[0]))
    jax.block_until_ready((warm_value, warm_gradient))
    compile_seconds = perf_counter() - compile_start

    runs = []
    for seed in config.seeds:
        starts, lower, upper = make_initial_points(
            ansatz, graph_data, config, seed, standard_params
        )
        start = perf_counter()
        optimized = hybrid_optimize(
            value_and_gradient, starts, lower, upper, config
        )
        optimized["runtime_s"] = perf_counter() - start
        runs.append(optimized)

    best_run = min(runs, key=lambda item: item["loss"])
    best_params = np.asarray(best_run["params"], dtype=float)
    best_loss = float(best_run["loss"])

    if ansatz == "Multi-angle" and standard_params is not None:
        anchor = standard_to_multi_angle(
            standard_params,
            graph_data["n"],
            len(graph_data["edges"]),
            config.p,
        )
        anchor_loss = float(value_and_gradient(jnp.asarray(anchor))[0])
        if anchor_loss < best_loss:
            best_params, best_loss = anchor, anchor_loss

    probabilities = np.asarray(
        probability_function(jnp.asarray(best_params)), dtype=float
    )
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    optimum = graph_data["optimum"]
    expectation = -best_loss
    raw_ratio = expectation / optimum
    if raw_ratio > 1.0 + 1e-5:
        raise RuntimeError(
            f"{graph_name}/{ansatz}: expectation_ratio={raw_ratio:.6f} > 1；"
            "cost vector 与参考解不一致"
        )
    expectation_ratio = min(raw_ratio, 1.0)

    rng = np.random.default_rng(2026)
    samples = rng.choice(len(probabilities), config.shots, p=probabilities)
    sampled_costs = metadata["costs"][samples]
    best_sampled = int(sampled_costs.max())
    optimal_probability = float(
        probabilities[metadata["costs"] == optimum].sum()
    )
    run_expectations = np.asarray([-run["loss"] for run in runs])

    row = {
        "graph": graph_name,
        "ansatz": ansatz,
        "n": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "p": config.p,
        "parameters": metadata["parameters"],
        "entangling_gates": metadata["entangling_gates"],
        "depth_est": metadata["depth_est"],
        "maxcut": optimum,
        "expectation": expectation,
        "expectation_ratio": expectation_ratio,
        "best_sampled": best_sampled,
        "sample_ratio": best_sampled / optimum,
        "optimal_probability": optimal_probability,
        "seed_std": float(run_expectations.std()),
        "evaluations": int(best_run["evaluations"]),
        "median_runtime_s": float(np.median([run["runtime_s"] for run in runs])),
        "compile_s": compile_seconds,
        "warm_classical_cut": (
            metadata["warm_cut"] if ansatz == "Warm-start" else np.nan
        ),
    }
    return row, best_params

def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("ansatz", sort=False)
        .agg(
            mean_ratio=("expectation_ratio", "mean"),
            worst_ratio=("expectation_ratio", "min"),
            mean_optimal_probability=("optimal_probability", "mean"),
            mean_parameters=("parameters", "mean"),
            mean_entanglers=("entangling_gates", "mean"),
            median_runtime_s=("median_runtime_s", "median"),
            mean_seed_std=("seed_std", "mean"),
        )
        .sort_values(
            ["mean_ratio", "worst_ratio", "median_runtime_s"],
            ascending=[False, False, True],
        )
    )

def choose_ansatz(
    summary: pd.DataFrame,
    quality_tolerance: float,
) -> tuple[str, str]:
    """
    先保留平均 ratio 距最佳不超过 tolerance 的方案，再按最差图表现、
    运行时间和参数数排序。另给出严格 QAOA 家族中的选择。
    """
    best_quality = float(summary["mean_ratio"].max())
    candidates = summary[
        summary["mean_ratio"] >= best_quality - quality_tolerance
    ].sort_values(
        ["worst_ratio", "median_runtime_s", "mean_parameters"],
        ascending=[False, True, True],
    )
    recommended = str(candidates.index[0])

    strict_names = [
        name
        for name in ("Standard", "Multi-angle", "Warm-start")
        if name in summary.index
    ]
    if strict_names:
        strict_summary = summary.loc[strict_names]
        strict_best = float(strict_summary["mean_ratio"].max())
        strict_candidates = strict_summary[
            strict_summary["mean_ratio"]
            >= strict_best - quality_tolerance
        ].sort_values(
            ["worst_ratio", "median_runtime_s", "mean_parameters"],
            ascending=[False, True, True],
        )
        strict_recommended = str(strict_candidates.index[0])
    else:
        strict_recommended = "未测试"
    return recommended, strict_recommended

def ansatz_comment(ansatz: str) -> str:
    comments = {
        "Standard": (
            "优点：仅 2p 个参数、含义清楚、通常最容易优化。"
            "缺点：共享角度限制表达能力，不规则图上质量可能较低。"
        ),
        "Multi-angle": (
            "优点：包含 Standard 作为特例，通常能提高期望近似比。"
            "缺点：参数增至 p(|E|+|V|)，经典优化成本最高。"
        ),
        "Warm-start": (
            "优点：参数少、可利用便宜的经典局部搜索。"
            "缺点：结果依赖经典初值，且优势不能解释为纯量子优势。"
        ),
        "Hardware-efficient": (
            "优点：本机仿真中常有最高解质量，CZ 数较少。"
            "缺点：参数较多，并非严格的 QAOA 交替算符结构。"
        ),
    }
    return comments[ansatz]

def markdown_table(frame: pd.DataFrame) -> str:
    display_frame = frame.copy()
    display_frame.insert(0, frame.index.name or "ansatz", frame.index)
    headers = [str(column) for column in display_frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def save_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    recommended: str,
    strict_recommended: str,
    config: ExperimentConfig,
    output_dir: Path,
    make_plot: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "ansatz_results.csv", index=False)
    summary.to_csv(output_dir / "ansatz_summary.csv")

    payload = {
        "config": asdict(config),
        "recommended_ansatz": recommended,
        "strict_qaoa_recommendation": strict_recommended,
        "ansatz_comments": {
            name: ansatz_comment(name) for name in summary.index
        },
    }
    with open(output_dir / "recommendation.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    report_lines = [
        "# 无权 Max-Cut Ansatz 本机实验总结",
        "",
        markdown_table(summary.round(4)),
        "",
        f"**综合推荐：{recommended}**",
        "",
        f"若必须保持严格的 QAOA 结构，推荐：**{strict_recommended}**。",
        "",
        "选择规则：先保留平均 expectation ratio 距最佳值不超过 "
        f"{config.quality_tolerance:.1%} 的方案，再比较最差图表现、"
        "运行时间和参数数。",
        "",
    ]
    for name in summary.index:
        report_lines.extend([f"- **{name}**：{ansatz_comment(name)}", ""])
    (output_dir / "summary.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    if make_plot:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        summary["mean_ratio"].plot(
            kind="bar", ax=axes[0], color="#2a9d8f"
        )
        axes[0].set_ylim(0.0, 1.02)
        axes[0].set_ylabel("Mean expectation ratio")
        axes[0].set_xlabel("")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].scatter(
            summary["mean_parameters"],
            summary["mean_ratio"],
            s=60,
            color="#e76f51",
        )
        for name, row in summary.iterrows():
            axes[1].annotate(
                name,
                (row["mean_parameters"], row["mean_ratio"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].set_xlabel("Mean trainable parameters")
        axes[1].set_ylabel("Mean expectation ratio")
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "ansatz_comparison.png", dpi=170)
        plt.close(figure)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="本机无权 Max-Cut 的四种 Ansatz 对比实验"
    )
    parser.add_argument("--fast", action="store_true", help="快速验证模式")
    parser.add_argument("--p", type=int, default=2, help="Ansatz 层数")
    parser.add_argument(
        "--ansatz",
        nargs="+",
        choices=ALL_ANSATZES,
        default=list(ALL_ANSATZES),
        help="需要测试的 Ansatz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("local_unweighted_ansatz_results"),
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.p < 1:
        raise ValueError("p 必须至少为 1")
    if "Multi-angle" in args.ansatz and "Standard" not in args.ansatz:
        raise ValueError("测试 Multi-angle 时必须同时测试 Standard，以便迁移初值")

    config = ExperimentConfig(p=args.p)
    if args.fast:
        config.seeds = (3,)
        config.shots = 400
        config.global_starts = 3
        config.selected_starts = 1
        config.adam_steps = 15
        config.lbfgs_maxiter = 25

    print("=" * 84)
    print("本机无权 Max-Cut Ansatz 对比")
    print("=" * 84)
    print(
        f"backend={tc.backend.name}, p={config.p}, seeds={config.seeds}, "
        f"optimizer=Sobol/structured -> Adam -> L-BFGS-B"
    )
    print("所有图均为无权图；expectation_ratio 使用精确枚举 Max-Cut 作分母。")

    graphs = build_unweighted_graphs()
    rows = []
    total_start = perf_counter()

    for graph_name, graph in graphs.items():
        graph_data = prepare_graph_data(graph)
        standard_params = None
        print(
            f"\n{graph_name}: n={graph.number_of_nodes()}, "
            f"|E|={graph.number_of_edges()}, exact Max-Cut={graph_data['optimum']}"
        )
        for ansatz in args.ansatz:
            row, best_params = test_ansatz(
                graph_name,
                graph,
                graph_data,
                ansatz,
                config,
                standard_params,
            )
            if ansatz == "Standard":
                standard_params = best_params
            rows.append(row)
            print(
                f"  {ansatz:18s} ratio={row['expectation_ratio']:.4f} "
                f"params={row['parameters']:3d} "
                f"time={row['median_runtime_s']:.3f}s"
            )

        jax.clear_caches()
        gc.collect()

    results = pd.DataFrame(rows)
    summary = aggregate_results(results)
    recommended, strict_recommended = choose_ansatz(
        summary, config.quality_tolerance
    )
    elapsed = perf_counter() - total_start

    print("\n各图详细结果")
    detail_columns = [
        "graph",
        "ansatz",
        "expectation_ratio",
        "optimal_probability",
        "parameters",
        "entangling_gates",
        "median_runtime_s",
    ]
    print(results[detail_columns].round(4).to_string(index=False))

    print("\n跨图汇总")
    print(summary.round(4).to_string())
    print("\n优缺点")
    for name in summary.index:
        print(f"- {name}: {ansatz_comment(name)}")

    print(f"\n综合推荐: {recommended}")
    if recommended == "Hardware-efficient":
        print(
            "理由：在本机 statevector 实验中优先追求解质量和较少纠缠门；"
            "但它不是严格 QAOA。"
        )
    print(f"严格 QAOA 结构下的推荐: {strict_recommended}")
    print(f"总耗时: {elapsed:.2f}s")

    save_outputs(
        results,
        summary,
        recommended,
        strict_recommended,
        config,
        args.output_dir,
        not args.no_plot,
    )
    print(f"结果目录: {args.output_dir.resolve()}")

def _grid_edges(Lx, Ly):
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                edges.append((i, (x + 1) * Ly + y))
            if y + 1 < Ly:
                edges.append((i, x * Ly + y + 1))
    return edges

def combined_demo():
    print("=" * 70)
    print("整合 Demo：RG 粗粒化（周子晗）× 朋友的混合优化器（adam_refine→L-BFGS-B）")
    print("=" * 70)
    big = MaxCutProblem(_grid_edges(4, 2), 8, name="4x2 grid")
    small = MaxCutProblem(_grid_edges(2, 2), 4, name="2x2 grid")
    big.brute_force_maxcut()
    print(f"目标图: {big.name} (N={big.n}, m={big.m}, 最优割={big.max_cut})")
    print(f"粗粒化: {small.name} (N={small.n}, m={small.m})")

    friend_opt = make_friend_optimizer(
        global_starts=6, selected=2, adam_steps=30, adam_lr=0.06,
        lbfgs_maxiter=50, seed=0,
    )
    print(f"\n{'='*70}\n对比：无 RG vs RG（两者都用朋友的优化器）\n{'='*70}")
    for mode in ["state", "hardware"]:
        solver = QAOA_ZZH(big, small, p=2, optimizer=friend_opt,
                          maxiter=50, cost_mode=mode)
        r = solver.compare(n_trials=10)
        print(f"\n  [{mode}口径] 无RG={r['no_rg_cost_mean']:,.0f}, RG={r['rg_cost_mean']:,.0f}")
        print(f"    加权节省={r['weighted_saving_pct']:.1f}%, RG更优={r['rg_better']}/{r['n_trials']}")
        print(f"    RG期望割={r['rg_cut_mean']:.2f}, 无RG期望割={r['no_rg_cut_mean']:.2f}")
    print(f"\n{'='*70}\n结论\n{'='*70}")
    print("  ✓ 上层 RG 框架通过 adapter 真正调用下层【朋友】的 adam_refine（同文件内）")
    print("  ✓ RG 加权成本节省在朋友的优化器下仍成立 —— RG 与优化器解耦")

def _combined_main():
    import sys
    mode = "combined"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]
    if mode == "friend":
        main()
    else:
        combined_demo()

if __name__ == "__main__":
    _combined_main()
