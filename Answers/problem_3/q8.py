"""Problem 3, Question 8: hardware-ready QAOA-MaxCut on C4.

The default run is an ideal local simulation. To submit to Tencent
superconducting hardware, set TC_TOKEN and TC_DEVICE, then run

    python q8.py --hardware

The cloud compiler is asked to map onto connected, high-quality physical
qubits and to decompose gates into the device basis.
"""

import argparse
import os

import tensorcircuit as tc

from q1 import GRAPHS
from q2 import optimize_exact, qaoa_circuit, sample_qaoa


def build_c4_circuit():
    n, edges = GRAPHS["C4"]
    result = optimize_exact(n, edges, p=1, maxiter=80)
    c = qaoa_circuit(n, edges, result.x[:1], result.x[1:])
    c.measure_instruction(*range(n))
    return c, result.x


def submit_to_hardware(circuit, shots):
    """Submit through TensorCircuit cloud when credentials are available."""
    from tensorcircuit.cloud import apis

    token = os.getenv("TC_TOKEN")
    device_name = os.getenv("TC_DEVICE")
    if not token or not device_name:
        raise RuntimeError(
            "Set TC_TOKEN and TC_DEVICE before using --hardware."
        )

    device = apis.get_device(provider="tencent", device=device_name)
    tasks = apis.submit_task(
        provider="tencent",
        device=device,
        token=token,
        circuit=circuit,
        shots=shots,
        compiling=False,
        enable_qos_qubit_mapping=True,
        enable_qos_gate_decomposition=True,
        enable_qos_initial_mapping=True,
        remarks="Problem 3 Q8: QAOA-MaxCut on C4",
    )

    for task in tasks:
        print(task.details())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()

    c, params = build_c4_circuit()
    print(f"optimized gamma={params[0]:.6f}, beta={params[1]:.6f}")

    if args.hardware:
        submit_to_hardware(c, args.shots)
    else:
        n, edges = GRAPHS["C4"]
        _, average, best_z, best_cut = sample_qaoa(
            n, edges, params, shots=args.shots
        )
        print(f"local simulation: <C>={average:.3f}")
        print(f"best sample: z={best_z}, C(z)={best_cut}")
        print("Use --hardware after setting TC_TOKEN and TC_DEVICE.")
