"""Problem 3, Question 1: MaxCut objective and Hamiltonian.

For a graph G=(V,E),

    C(z) = sum_{(i,j) in E} (z_i XOR z_j),
    C    = sum_{(i,j) in E} (I - Z_i Z_j) / 2.

Proof:
Z_i Z_j |z> = (-1)^(z_i+z_j)|z>.  Therefore each edge term has
eigenvalue [1-(-1)^(z_i+z_j)]/2, which is 1 exactly when z_i != z_j.
Summing the edge terms proves C|z> = C(z)|z>.
"""

from itertools import product

import tensorcircuit as tc


# Select the installed JAX backend.
K = tc.set_backend("jax")


# Graphs given in Problem 3.
GRAPHS = {
    "C4": (
        4,
        [(0, 1), (1, 2), (2, 3), (0, 3)],
    ),
    "G6": (
        6,
        [(0, 1), (3, 4), (2, 5), (0, 3), (4, 5), (1, 2), (1, 4)],
    ),
    "G9": (
        9,
        [
            (0, 1),
            (3, 4),
            (7, 8),
            (2, 5),
            (0, 3),
            (4, 5),
            (6, 7),
            (1, 2),
            (4, 7),
            (5, 8),
            (3, 6),
            (1, 4),
        ],
    ),
}


def maxcut_value(z, edges):
    """Classical objective C(z)."""
    return sum(z[i] != z[j] for i, j in edges)


def quantum_value(z, edges):
    """Compute <z|C|z> with a TensorCircuit circuit object."""
    c = tc.Circuit(len(z))

    # The default input is |0...0>; X gates prepare |z>.
    for i, bit in enumerate(z):
        if bit:
            c.X(i)

    value = 0.0
    for i, j in edges:
        zz = c.expectation(
            [tc.gates.z(), [i]],
            [tc.gates.z(), [j]],
        )
        value += (1.0 - K.real(zz)) / 2.0

    return float(K.numpy(value))


def exact_solution(n, edges):
    """Find a maximum cut by checking all 2^n bitstrings."""
    best_z = None
    best_value = -1

    for z in product([0, 1], repeat=n):
        value = maxcut_value(z, edges)
        if value > best_value:
            best_z = z
            best_value = value

    return best_z, best_value


if __name__ == "__main__":
    for name, (n, edges) in GRAPHS.items():
        z, classical = exact_solution(n, edges)
        quantum = quantum_value(z, edges)

        print(
            f"{name}: z={''.join(map(str, z))}, "
            f"C(z)={classical}, <z|C|z>={quantum:.0f}"
        )
