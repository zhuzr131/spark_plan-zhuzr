import tensorcircuit as tc
import tencirchem
import qiskit
import cirq
import pyscf
from rdkit import Chem
print("✅ tc 环境全部 import 成功")
print(f"   tensorcircuit: {tc.__version__}")
print(f"   qiskit:        {qiskit.__version__}")
print(f"   JAX backend:   {tc.backend.name}")