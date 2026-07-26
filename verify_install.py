"""
Verify that all dependencies are installed and core modules load correctly.
Run: python qaoa_project/verify_install.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

passed = 0
failed = 0


def check(module_name, import_name=None):
    global passed, failed
    name = import_name or module_name
    try:
        __import__(name)
        print(f"  [OK] {module_name}")
        passed += 1
    except ImportError as e:
        print(f"  [FAIL] {module_name}: {e}")
        failed += 1


print("=== Checking Python ===")
print(f"  Python: {sys.version}")

print("\n=== Core Dependencies ===")
check("numpy")
check("scipy")
check("networkx", "networkx")
check("matplotlib")
check("seaborn")

print("\n=== ML / Autodiff ===")
check("jax")
check("jaxlib")

print("\n=== Quantum ===")
check("tensorcircuit")

print("\n=== Project Modules ===")
try:
    from hamiltonian import graph_c4, graph_g6, graph_g9, maxcut_exact
    g = graph_c4()
    cut, _ = maxcut_exact(g)
    assert cut == 4, f"Expected maxcut=4 for C4 (bipartite), got {cut}"
    print("  [OK] hamiltonian (C4 maxcut verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] hamiltonian: {e}")
    failed += 1

try:
    from init_strategies import INIT_STRATEGIES
    for name in ["zero", "random", "linear_ramp"]:
        params = INIT_STRATEGIES[name](3)
        assert len(params) == 6, f"Expected 6 params, got {len(params)}"
    print("  [OK] init_strategies (all strategies)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] init_strategies: {e}")
    failed += 1

try:
    from graph_families import path_graph, cycle_graph, random_regular_graph, erdos_renyi_graph
    g = random_regular_graph(6, 3, seed=42)
    assert g.number_of_nodes() == 6
    print("  [OK] graph_families")
    passed += 1
except Exception as e:
    print(f"  [FAIL] graph_families: {e}")
    failed += 1

try:
    from dla import dla_basis
    from hamiltonian import graph_c4
    g = graph_c4()
    dim, basis = dla_basis(g, verbose=False)
    print(f"  [OK] dla (C4 dim={dim})")
    passed += 1
except Exception as e:
    print(f"  [FAIL] dla: {e}")
    failed += 1

print(f"\n=== Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed > 0 else 0)
