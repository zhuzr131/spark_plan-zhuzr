#!/usr/bin/env python3
"""
Combined_QAOA.py 的 TensorCircuit QCloud / 真实量子设备版本。

与本机 statevector 版本的主要区别
----------------------------------
1. 不调用 circuit.state()、probability() 或 expectation()。
2. Max-Cut 目标值来自多次 Z 基测量：

       E[C] = sum_z count(z) * C(z) / shots

   因为无权 Max-Cut 的所有 Z_i Z_j 项彼此对易，一次全量测量即可同时
   计算所有边的贡献。
3. 不使用 JAX 精确梯度。优化器改为 SPSA-Adam；每次梯度估计只需要
   theta+c*delta 和 theta-c*delta 两个电路，与参数数量无关。
4. 在云端模式下，正负扰动电路用 submit_task 批量提交，使它们尽量处于
   相近的设备噪声条件。
5. 每次提交后立即保存 task id；程序中断后可用 --recover-task 找回任务。
6. 当前 Qiskit 已移除旧的 QuantumCircuit.qasm()。本文件使用
   qiskit.qasm2.dumps() 兼容导出，并以 source=... 提交，避开旧接口。

安全默认值
----------
不带参数运行时只进行本地 shots dry-run，不会提交云任务：

    python Combined_QAOA_hardware.py --fast

查看当前在线设备：

    export TC_TOKEN="从腾讯量子云网页复制的 token"
    python Combined_QAOA_hardware.py --list-devices

先在云模拟器检查完整提交流程：

    python Combined_QAOA_hardware.py --mode cloud --device simulator:tc --fast

提交到真实设备（DEVICE_NAME 请从 --list-devices 的实时输出中选择）：

    python Combined_QAOA_hardware.py \
        --mode cloud \
        --device DEVICE_NAME \
        --graph C8 \
        --ansatz Standard

默认从环境变量 TC_TOKEN 读取 token，代码和输出文件均不会保存 token。

硬件版默认使用 Standard QAOA：它只有 2p 个参数，比本机精确梯度实验中
表现很好的 Hardware-efficient Ansatz 更适合有限 shots 的小预算优化。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import networkx as nx
import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")

ANSATZES = (
    "Standard",
    "Multi-angle",
    "Warm-start",
    "Hardware-efficient",
)

GRAPH_NAMES = ("C8", "K4,4", "3-regular-8", "Random-8")


@dataclass
class RunConfig:
    graph: str = "C8"
    ansatz: str = "Standard"
    p: int = 2
    seed: int = 7
    shots: int = 1024
    final_shots: int = 4096
    pretrain_iterations: int = 16
    cloud_iterations: int = 8
    validation_interval: int = 2
    learning_rate: float = 0.08
    spsa_c: float = 0.12


def build_graph(name: str) -> nx.Graph:
    """只生成无权图；默认图均不超过教程示例芯片的 9 个量子比特。"""
    graphs = {
        "C8": nx.cycle_graph(8),
        "K4,4": nx.complete_bipartite_graph(4, 4),
        "3-regular-8": nx.random_regular_graph(3, 8, seed=17),
        "Random-8": nx.erdos_renyi_graph(8, 0.35, seed=7),
    }
    graph = nx.convert_node_labels_to_integers(graphs[name])
    if graph.number_of_edges() == 0:
        raise ValueError("图中至少需要一条边")
    return graph


def cut_value(bitstring: str, edges: Iterable[tuple[int, int]]) -> int:
    """TensorCircuit count key 的第 i 位对应逻辑量子比特 i。"""
    return sum(bitstring[i] != bitstring[j] for i, j in edges)


def normalize_count_key(key: Any, n: int) -> str:
    clean = "".join(character for character in str(key) if character in "01")
    if len(clean) != n:
        raise ValueError(
            f"测量结果 {key!r} 含 {len(clean)} 位，但电路需要 {n} 位。"
            "请检查测量指令和设备映射。"
        )
    return clean


def expected_cut_from_counts(
    counts: dict[Any, Any],
    edges: tuple[tuple[int, int], ...],
    n: int,
) -> tuple[float, str, int, float]:
    """
    将真实测量计数转为期望 cut。

    readout mitigation 可能返回浮点 quasi-count，因此这里用总权重归一化，
    而不是假定分母严格等于 shots。
    """
    clean_counts: dict[str, float] = {}
    for raw_key, raw_count in counts.items():
        key = normalize_count_key(raw_key, n)
        clean_counts[key] = clean_counts.get(key, 0.0) + float(raw_count)

    total = float(sum(clean_counts.values()))
    if total <= 0.0:
        raise RuntimeError("测量计数之和不为正，无法计算期望值")

    expectation = sum(
        weight * cut_value(key, edges)
        for key, weight in clean_counts.items()
    ) / total
    positive = {
        key: weight for key, weight in clean_counts.items() if weight > 0.0
    }
    if not positive:
        raise RuntimeError("没有正的测量计数")
    best_bitstring = max(
        positive,
        key=lambda key: (cut_value(key, edges), positive[key]),
    )
    best_cut = cut_value(best_bitstring, edges)
    best_probability = positive[best_bitstring] / sum(positive.values())
    return float(expectation), best_bitstring, int(best_cut), float(best_probability)


def exact_maxcut(graph: nx.Graph) -> tuple[int, str]:
    """仅用于小图的经典参考分母，不参与量子目标函数计算。"""
    n = graph.number_of_nodes()
    edges = tuple(graph.edges())
    best_cut, best_string = -1, ""
    for state in range(1 << n):
        bitstring = format(state, f"0{n}b")
        value = cut_value(bitstring, edges)
        if value > best_cut:
            best_cut, best_string = value, bitstring
    return best_cut, best_string


def warm_start_angles(
    graph: nx.Graph,
    epsilon: float = 0.15,
    seed: int = 7,
    restarts: int = 16,
) -> np.ndarray:
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
    return np.arcsin(np.sqrt(probabilities))


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


def initial_parameters(
    ansatz: str,
    n: int,
    m: int,
    p: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gamma = np.linspace(0.15, 0.75, p)
    beta = np.linspace(0.70, 0.15, p)

    if ansatz == "Standard":
        params = np.r_[gamma, beta]
    elif ansatz == "Multi-angle":
        params = np.r_[
            np.tile(gamma[:, None], (1, m)).ravel(),
            np.tile(beta[:, None], (1, n)).ravel(),
        ]
    elif ansatz == "Warm-start":
        params = np.zeros(2 * p)
    else:
        params = rng.normal(0.0, 0.25, (p + 1) * n)

    lower, upper = parameter_bounds(ansatz, n, m, p)
    return np.clip(params, lower, upper)


def build_ansatz_circuit(
    graph: nx.Graph,
    ansatz: str,
    params: np.ndarray,
    p: int,
    add_measurements: bool,
) -> tc.Circuit:
    """仅构建门电路；不读取 state、probability 或 expectation。"""
    n = graph.number_of_nodes()
    edges = tuple(graph.edges())
    m = len(edges)
    params = np.asarray(params, dtype=float)
    circuit = tc.Circuit(n)

    if ansatz == "Warm-start":
        theta = warm_start_angles(graph)
        for qubit in range(n):
            circuit.ry(qubit, theta=2.0 * theta[qubit])
    else:
        for qubit in range(n):
            circuit.h(qubit)

    if ansatz == "Standard":
        gammas, betas = params[:p], params[p:]
        for layer in range(p):
            for i, j in edges:
                apply_zz_rotation(circuit, i, j, -gammas[layer])
            for qubit in range(n):
                circuit.rx(qubit, theta=2.0 * betas[layer])

    elif ansatz == "Multi-angle":
        gamma_count = p * m
        gammas = params[:gamma_count].reshape(p, m)
        betas = params[gamma_count:].reshape(p, n)
        for layer in range(p):
            for edge_index, (i, j) in enumerate(edges):
                apply_zz_rotation(
                    circuit,
                    i,
                    j,
                    -gammas[layer, edge_index],
                )
            for qubit in range(n):
                circuit.rx(qubit, theta=2.0 * betas[layer, qubit])

    elif ansatz == "Warm-start":
        theta = warm_start_angles(graph)
        gammas, betas = params[:p], params[p:]
        for layer in range(p):
            for i, j in edges:
                apply_zz_rotation(circuit, i, j, -gammas[layer])
            for qubit in range(n):
                circuit.ry(qubit, theta=-2.0 * theta[qubit])
                circuit.rz(qubit, theta=-2.0 * betas[layer])
                circuit.ry(qubit, theta=2.0 * theta[qubit])

    elif ansatz == "Hardware-efficient":
        angles = params.reshape(p + 1, n)
        for layer in range(p):
            for qubit in range(n):
                circuit.ry(qubit, theta=angles[layer, qubit])
            for i, j in edges:
                circuit.cz(i, j)
        for qubit in range(n):
            circuit.ry(qubit, theta=angles[p, qubit])
    else:
        raise ValueError(f"未知 Ansatz: {ansatz}")

    if add_measurements:
        # 教程说明云任务会默认测量，但涉及 qubit mapping 时建议显式添加。
        circuit.measure_instruction(*range(n))
    return circuit


def apply_zz_rotation(
    circuit: tc.Circuit,
    qubit_a: int,
    qubit_b: int,
    theta: float,
) -> None:
    """RZZ(theta) 的硬件友好分解：CX - RZ(theta) - CX。"""
    circuit.cnot(qubit_a, qubit_b)
    circuit.rz(qubit_b, theta=theta)
    circuit.cnot(qubit_a, qubit_b)


def circuit_to_qasm2(circuit: tc.Circuit) -> str:
    """
    兼容 Qiskit 1/2 的 OpenQASM 2 导出。

    TensorCircuit 当前版本的 to_openqasm() 仍依赖已经删除的
    QuantumCircuit.qasm()，所以优先使用新版 qiskit.qasm2.dumps()。
    """
    try:
        from qiskit import qasm2

        qiskit_circuit = circuit.to_qiskit(enable_instruction=True)
        return qasm2.dumps(qiskit_circuit)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "无法导出 OpenQASM 2。请安装兼容的 qiskit（需提供 qiskit.qasm2.dumps）。"
        ) from error


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class ShotExecutor:
    """同一目标函数的本地 shots 与 QCloud 两种执行后端。"""

    def __init__(
        self,
        mode: str,
        graph: nx.Graph,
        ansatz: str,
        p: int,
        device: Any = None,
        with_rem: bool = False,
        group: str = "Combined_QAOA",
        task_log: Path | None = None,
    ):
        self.mode = mode
        self.graph = graph
        self.ansatz = ansatz
        self.p = p
        self.device = device
        self.with_rem = with_rem
        self.group = group
        self.task_log = task_log
        self.edges = tuple(graph.edges())
        self.n = graph.number_of_nodes()
        self.circuit_evaluations = 0
        self.task_ids: list[str] = []

    def evaluate_many(
        self,
        parameter_points: Iterable[np.ndarray],
        shots: int,
        stage: str,
    ) -> list[dict[str, Any]]:
        points = [np.asarray(point, dtype=float) for point in parameter_points]
        if self.mode == "dry-run":
            counts_list = []
            for point in points:
                circuit = build_ansatz_circuit(
                    self.graph,
                    self.ansatz,
                    point,
                    self.p,
                    add_measurements=False,
                )
                counts = circuit.sample(
                    batch=shots,
                    allow_state=True,
                    format="count_dict_bin",
                )
                counts_list.append(counts)
        else:
            circuits = [
                build_ansatz_circuit(
                    self.graph,
                    self.ansatz,
                    point,
                    self.p,
                    add_measurements=True,
                )
                for point in points
            ]
            sources = [circuit_to_qasm2(circuit) for circuit in circuits]
            submitted = tc.cloud.apis.submit_task(
                device=self.device,
                source=sources if len(sources) > 1 else sources[0],
                shots=shots,
                group=self.group,
                enable_qos_gate_decomposition=True,
                enable_qos_qubit_mapping=True,
            )
            tasks = submitted if isinstance(submitted, list) else [submitted]
            if len(tasks) != len(circuits):
                raise RuntimeError(
                    f"提交了 {len(circuits)} 个电路，但只返回 {len(tasks)} 个任务"
                )

            for task in tasks:
                self.task_ids.append(task.id_)
                if self.task_log is not None:
                    append_json_line(
                        self.task_log,
                        {
                            "time_utc": datetime.now(timezone.utc).isoformat(),
                            "stage": stage,
                            "task_id": task.id_,
                            "device": str(self.device),
                            "shots": shots,
                        },
                    )
            counts_list = [
                task.results(blocked=True, mitigated=self.with_rem)
                for task in tasks
            ]

        self.circuit_evaluations += len(points)
        evaluations = []
        for counts in counts_list:
            expectation, best_bits, best_cut, best_probability = (
                expected_cut_from_counts(counts, self.edges, self.n)
            )
            evaluations.append(
                {
                    "expected_cut": expectation,
                    "best_bitstring": best_bits,
                    "best_cut": best_cut,
                    "best_probability": best_probability,
                    "counts": {
                        str(key): float(value) for key, value in counts.items()
                    },
                }
            )
        return evaluations

    def evaluate(
        self,
        params: np.ndarray,
        shots: int,
        stage: str,
    ) -> dict[str, Any]:
        return self.evaluate_many([params], shots, stage)[0]


def spsa_adam_optimize(
    executor: ShotExecutor,
    initial: np.ndarray,
    config: RunConfig,
    iterations: int,
    shots: int,
    stage: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """
    面向 shots/QPU 的优化器。

    每轮用同一个随机方向的正负扰动估计梯度。对于任意参数维数，
    梯度估计始终只需两个电路。
    """
    n = executor.graph.number_of_nodes()
    m_edges = executor.graph.number_of_edges()
    lower, upper = parameter_bounds(config.ansatz, n, m_edges, config.p)
    rng = np.random.default_rng(config.seed + (1000 if stage == "cloud" else 0))
    theta = np.asarray(initial, dtype=float).copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    beta1, beta2 = 0.90, 0.98

    best_evaluation = executor.evaluate(
        theta, shots, f"{stage}-initial"
    )
    best_theta = theta.copy()
    history = [
        {
            "iteration": 0,
            "expected_cut": best_evaluation["expected_cut"],
            "best_cut": best_evaluation["best_cut"],
        }
    ]

    for iteration in range(1, iterations + 1):
        delta = rng.choice([-1.0, 1.0], size=len(theta))
        perturbation = config.spsa_c / (iteration**0.101)
        plus = np.clip(theta + perturbation * delta, lower, upper)
        minus = np.clip(theta - perturbation * delta, lower, upper)

        plus_eval, minus_eval = executor.evaluate_many(
            [plus, minus],
            shots,
            f"{stage}-spsa-{iteration}",
        )
        slope = (
            plus_eval["expected_cut"] - minus_eval["expected_cut"]
        ) / (2.0 * perturbation)
        gradient = slope * delta

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        first_hat = first_moment / (1.0 - beta1**iteration)
        second_hat = second_moment / (1.0 - beta2**iteration)
        rate = config.learning_rate / (iteration**0.15)
        theta += rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        theta = np.clip(theta, lower, upper)

        should_validate = (
            iteration % config.validation_interval == 0
            or iteration == iterations
        )
        if should_validate:
            current = executor.evaluate(
                theta, shots, f"{stage}-validation-{iteration}"
            )
            if current["expected_cut"] > best_evaluation["expected_cut"]:
                best_evaluation = current
                best_theta = theta.copy()
            history.append(
                {
                    "iteration": iteration,
                    "expected_cut": current["expected_cut"],
                    "best_cut": current["best_cut"],
                }
            )

        print(
            f"  {stage} iter {iteration:02d}/{iterations}: "
            f"plus={plus_eval['expected_cut']:.3f}, "
            f"minus={minus_eval['expected_cut']:.3f}, "
            f"best={best_evaluation['expected_cut']:.3f}"
        )

    return best_theta, best_evaluation, history


def configure_cloud(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """设置 provider/token，取得设备并检查图大小。"""
    tc.cloud.apis.set_provider(args.provider)
    token = os.environ.get(args.token_env)
    if token:
        tc.cloud.apis.set_token(
            token,
            provider=args.provider,
            cached=args.cache_token,
        )
    elif tc.cloud.apis.get_token(args.provider) is None:
        raise RuntimeError(
            f"没有找到云 token。请先执行 export {args.token_env}='你的 token'，"
            "或按 TensorCircuit 教程设置缓存 token。"
        )

    device = tc.cloud.apis.get_device(
        provider=args.provider,
        device=args.device,
    )
    properties = device.list_properties()
    return device, properties


def print_device_summary(device: Any, properties: dict[str, Any]) -> None:
    print("\n云设备信息")
    print(f"  device       : {device}")
    print(f"  state        : {properties.get('state', 'unknown')}")
    print(f"  type         : {properties.get('type', 'unknown')}")
    print(f"  qubits       : {properties.get('qubits', 'unknown')}")
    print(f"  native gates : {device.native_gates()}")
    print(f"  topology     : {device.topology()}")


def list_online_devices(args: argparse.Namespace) -> None:
    tc.cloud.apis.set_provider(args.provider)
    token = os.environ.get(args.token_env)
    if token:
        tc.cloud.apis.set_token(
            token,
            provider=args.provider,
            cached=args.cache_token,
        )
    devices = tc.cloud.apis.list_devices(args.provider, state="on")
    print("当前在线设备：")
    for device in devices:
        print(f"  {device}")


def recover_tasks(args: argparse.Namespace) -> None:
    device, properties = configure_cloud(args)
    print_device_summary(device, properties)
    for task_id in args.recover_task:
        task = tc.cloud.apis.get_task(task_id, device=device)
        status = task.status()
        print(f"\ntask {task_id}: {status}")
        if status == "completed" or args.wait:
            counts = task.results(
                blocked=args.wait,
                mitigated=args.with_rem,
            )
            print(json.dumps(counts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 shots 与 TensorCircuit QCloud 的无权 Max-Cut QAOA"
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "cloud"),
        default="dry-run",
        help="dry-run 不联网；cloud 会真实提交任务",
    )
    parser.add_argument("--graph", choices=GRAPH_NAMES, default="C8")
    parser.add_argument("--ansatz", choices=ANSATZES, default="Standard")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--final-shots", type=int, default=4096)
    parser.add_argument("--pretrain-iterations", type=int, default=16)
    parser.add_argument("--cloud-iterations", type=int, default=8)
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="cloud 模式下跳过本地 shots 预训练",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")

    parser.add_argument("--provider", default="tencent")
    parser.add_argument(
        "--device",
        default="simulator:tc",
        help="先用 --list-devices 查询；真实 QPU 名称不要写死在代码中",
    )
    parser.add_argument(
        "--token-env",
        default="TC_TOKEN",
        help="存放 token 的环境变量名称",
    )
    parser.add_argument(
        "--cache-token",
        action="store_true",
        help="显式允许 TensorCircuit 将 token 缓存在用户目录",
    )
    parser.add_argument(
        "--with-rem",
        action="store_true",
        help="启用 TensorCircuit 自动 readout error mitigation",
    )
    parser.add_argument("--group", default="Combined_QAOA")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--recover-task", nargs="+")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="恢复任务时等待其完成",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("Combined_QAOA_hardware_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_online_devices(args)
        return
    if args.recover_task:
        recover_tasks(args)
        return
    if args.p < 1:
        raise ValueError("p 必须至少为 1")
    if args.shots < 1 or args.final_shots < 1:
        raise ValueError("shots 必须为正整数")

    config = RunConfig(
        graph=args.graph,
        ansatz=args.ansatz,
        p=args.p,
        seed=args.seed,
        shots=args.shots,
        final_shots=args.final_shots,
        pretrain_iterations=args.pretrain_iterations,
        cloud_iterations=args.cloud_iterations,
    )
    if args.fast:
        config.shots = min(config.shots, 256)
        config.final_shots = min(config.final_shots, 512)
        config.pretrain_iterations = 3
        config.cloud_iterations = 2

    graph = build_graph(config.graph)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    exact_cut, exact_bits = exact_maxcut(graph)
    params = initial_parameters(
        config.ansatz,
        n,
        m,
        config.p,
        config.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_log = args.output_dir / "task_ids.jsonl"
    run_start = perf_counter()
    all_history: dict[str, Any] = {}

    print("=" * 78)
    print("TensorCircuit shots/QCloud Max-Cut")
    print("=" * 78)
    print(
        f"mode={args.mode}, graph={config.graph}, n={n}, |E|={m}, "
        f"ansatz={config.ansatz}, p={config.p}"
    )
    print(f"classical exact Max-Cut={exact_cut}, one optimum={exact_bits}")
    print("量子目标值来自 bitstring 测量平均，不使用 expectation/state/probability。")

    device = None
    device_properties: dict[str, Any] = {}
    if args.mode == "cloud":
        device, device_properties = configure_cloud(args)
        print_device_summary(device, device_properties)
        capacity = device_properties.get("qubits")
        if isinstance(capacity, int) and n > capacity:
            raise ValueError(
                f"图需要 {n} 个量子比特，但设备只有 {capacity} 个"
            )
        if device_properties.get("state") not in (None, "on"):
            raise RuntimeError(
                f"设备当前状态为 {device_properties.get('state')!r}，不是 on"
            )

    local_executor = None
    if args.mode == "dry-run" or not args.skip_pretrain:
        print("\n[1] 本地 shots 预训练")
        local_executor = ShotExecutor(
            "dry-run",
            graph,
            config.ansatz,
            config.p,
        )
        params, _, local_history = spsa_adam_optimize(
            local_executor,
            params,
            config,
            config.pretrain_iterations,
            config.shots,
            "local",
        )
        all_history["local"] = local_history

    if args.mode == "cloud":
        print("\n[2] QCloud/QPU shots 精修")
        active_executor = ShotExecutor(
            "cloud",
            graph,
            config.ansatz,
            config.p,
            device=device,
            with_rem=args.with_rem,
            group=args.group,
            task_log=task_log,
        )
        params, _, cloud_history = spsa_adam_optimize(
            active_executor,
            params,
            config,
            config.cloud_iterations,
            config.shots,
            "cloud",
        )
        all_history["cloud"] = cloud_history
    else:
        assert local_executor is not None
        active_executor = local_executor

    print("\n[3] 最终高 shots 测量")
    final = active_executor.evaluate(
        params,
        config.final_shots,
        "final",
    )
    ratio = final["expected_cut"] / exact_cut
    elapsed = perf_counter() - run_start
    print(f"  expected cut     : {final['expected_cut']:.4f}")
    print(f"  expectation ratio: {ratio:.4f}")
    print(
        f"  best bitstring   : {final['best_bitstring']} "
        f"(cut={final['best_cut']})"
    )
    print(f"  its frequency    : {final['best_probability']:.4f}")
    total_circuit_evaluations = active_executor.circuit_evaluations
    if local_executor is not None and local_executor is not active_executor:
        total_circuit_evaluations += local_executor.circuit_evaluations
    print(f"  circuit runs     : {total_circuit_evaluations}")
    print(f"  elapsed          : {elapsed:.2f}s")

    device_summary = {}
    if device is not None:
        device_summary = {
            "name": str(device),
            "state": device_properties.get("state"),
            "type": device_properties.get("type"),
            "qubits": device_properties.get("qubits"),
            "native_gates": device.native_gates(),
            "topology": device.topology(),
        }

    output = {
        "config": asdict(config),
        "mode": args.mode,
        "device": device_summary,
        "readout_mitigation": args.with_rem,
        "exact_maxcut": exact_cut,
        "exact_bitstring": exact_bits,
        "final_parameters": params.tolist(),
        "final": {
            key: value for key, value in final.items() if key != "counts"
        },
        "expectation_ratio": ratio,
        "final_counts": final["counts"],
        "task_ids": active_executor.task_ids,
        "circuit_evaluations": total_circuit_evaluations,
        "history": all_history,
        "elapsed_s": elapsed,
    }
    hardware_ready_circuit = build_ansatz_circuit(
        graph,
        config.ansatz,
        params,
        config.p,
        add_measurements=True,
    )
    qasm_path = args.output_dir / "circuit.qasm"
    qasm_path.write_text(
        circuit_to_qasm2(hardware_ready_circuit),
        encoding="utf-8",
    )
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  result           : {result_path.resolve()}")
    print(f"  measured circuit : {qasm_path.resolve()}")
    if active_executor.task_ids:
        print(f"  task id log      : {task_log.resolve()}")


if __name__ == "__main__":
    main()
