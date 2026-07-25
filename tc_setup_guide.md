# Miniconda + Mamba + 量子计算环境（tc）配置指南

> 适用：Windows (WSL2 Ubuntu) / macOS  
> 目标：构建名为 **`tc`** 的 conda 虚拟环境，含 TensorCircuit / TencirChem / Qiskit / RDKit / JAX / PySCF 等  
> 编辑器：VS Code / PyCharm

---

## 一、Windows：启用 WSL2（macOS 用户跳到第二节）

### 1.1 一键安装（管理员 PowerShell）
```powershell
wsl --install
```
默认装 **Ubuntu**，装完重启。

### 1.2 验证
```powershell
wsl --list --verbose
```

### 1.3 进入 WSL
```powershell
wsl
```
> 以下所有命令在 **WSL Ubuntu** 或 **macOS 终端**执行。

---

## 二、下载并安装 Miniconda

### 2.1 下载（Python 3.12 安装包）

**Linux / WSL：**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

**macOS (Apple Silicon)：**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

**macOS (Intel)：**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

### 2.2 安装
```bash
bash Miniconda3-latest-Linux-x86_64.sh   # macOS 对应替换文件名
```
一路回车，`yes` 同意 License，同意 conda init。装完**重开终端**。

### 2.3 验证
```bash
conda --version
# 期望输出：conda 24.x 或 25.x
```

---

## 三、安装 Mamba（替代 conda 做依赖解析，快很多）

```bash
conda install -n base -c conda-forge mamba -y
```

验证：
```bash
mamba --version
```

---

## 四、创建环境 `tc`（Python 3.12）

```bash
mamba create -n tc python=3.12 -y
conda activate tc
```

---

## 五、Conda 渠道包（RDKit 等带 C 扩展的）

```bash
mamba install -c conda-forge \
  "rdkit=2025.03.*" \
  openfermionpyscf=0.5 \
  -y
```

> `openfermionpyscf` 会自动把 PySCF 的 conda 部分也拉齐，后面 pip 的 pyscf 是对齐版本用的。

---

## 六、Pip 包安装

### 6.1 先升 pip / setuptools / wheel
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6.2 一次性装完全部 pip 依赖
```bash
pip install \
  tensorcircuit==0.12.0 \
  tencirchem==2024.11 \
  openfermion==1.6.1 \
  pyscf==2.7.0 \
  qiskit==0.46.2 \
  qiskit-terra==0.46.2 \
  qiskit-nature==0.7.2 \
  qiskit-algorithms==0.3.1 \
  cirq==1.6.1 \
  jax==0.7.0 \
  jaxlib==0.7.0 \
  optax==0.2.8 \
  opt_einsum==3.4.0 \
  "numpy>=1.24,<2" \
  "scipy>=1.11,<2" \
  networkx==3.6.1 \
  sympy==1.14.0 \
  symengine==0.13.0 \
  PyYAML==6.0.2 \
  python-dotenv==1.2.2 \
  tqdm==4.67.3 \
  matplotlib==3.10.5 \
  scikit-learn==1.7.1 \
  scikit-optimize==0.10.2 \
  pennylane==0.44.1 \
  pennylane_lightning==0.45.0 \
  autoray==0.8.2 \
  noisyopt==0.2.2 \
  tensornetwork-ng==0.5.1 \
  ase==3.28.0 \
  PubChemPy==1.0.5 \
  torch==2.6.0
```

> ⚠️ `numpy<2` 是硬性约束——PySCF / JAX / RDKit 这一代还没全适配 NumPy 2.x，锁住防翻车。

---

## 七、验活

```bash
python - << 'EOF'
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
EOF
```

没报错就 OK。

---

## 八、VS Code 绑定 `tc` 环境

1. 装插件：**Python**、**Pylance**（WSL 用户再装 **WSL**）
2. `Ctrl+Shift+P` → `Python: Select Interpreter`
3. 选路径：
   ```
   ~/miniconda3/envs/tc/bin/python
   ```
   （Windows 本地 conda 的话是 `C:\Users\<你>\miniconda3\envs\tc\python.exe`）

WSL 用户：在 WSL 窗口里 `code .`，VS Code 会自动用 WSL 里的 conda。

---

## 九、PyCharm 绑定 `tc` 环境

1. `Settings/Preferences` → `Python Interpreter`
2. ⚙️ → `Add Interpreter` → `Conda Environment`
3. 选 **Existing environment**
4. Interpreter 指到：
   ```
   ~/miniconda3/envs/tc/bin/python
   ```

---

## 十、环境导出 / 复现

```bash
# 导出（conda 部分 + pip 部分都记着）
conda env export --no-builds -n tc > tc_env.yaml

# 别人复现
mamba env create -f tc_env.yaml
```

---

## 十一、常见坑

| 现象 | 解法 |
|---|---|
| `ImportError: libstdc++.so.6: version GLIBCXX_… not found` | `mamba install -c conda-forge libstdcxx-ng -y` |
| JAX 走 CPU 想换 GPU | `pip install "jax[cuda12]==0.7.0"`（需 NVIDIA 驱动 + CUDA 12） |
| PySCF 找不到 `libxc` | conda 装的 `openfermionpyscf` 已带，别混装 pip 版 pyscf 覆盖 |
| RDKit import 慢 | 正常，第一次约 2–3 s |

---

## 附：完整 dependencies 快照（yaml 风格备份用）

```yaml
dependencies:
  - python=3.12
  - rdkit=2025.03.*
  - openfermionpyscf=0.5
  - pip
  - pip:
      - tensorcircuit==0.12.0
      - tencirchem==2024.11
      - openfermion==1.6.1
      - pyscf==2.7.0
      - qiskit==0.46.2
      - qiskit-terra==0.46.2
      - qiskit-nature==0.7.2
      - qiskit-algorithms==0.3.1
      - cirq==1.6.1
      - jax==0.7.0
      - jaxlib==0.7.0
      - optax==0.2.8
      - opt_einsum==3.4.0
      - "numpy>=1.24,<2"
      - "scipy>=1.11,<2"
      - networkx==3.6.1
      - sympy==1.14.0
      - symengine==0.13.0
      - PyYAML==6.0.2
      - python-dotenv==1.2.2
      - tqdm==4.67.3
      - matplotlib==3.10.5
      - scikit-learn==1.7.1
      - scikit-optimize==0.10.2
      - pennylane==0.44.1
      - pennylane_lightning==0.45.0
      - autoray==0.8.2
      - noisyopt==0.2.2
      - tensornetwork-ng==0.5.1
      - ase==3.28.0
      - PubChemPy==1.0.5
      - torch==2.6.0
```

---

> 📌 这份 MD 直接 `save as tc_setup.md` 就能用。装完如果 `tencirchem` 或 `pyscf` 报底层库问题，把报错贴出来我再帮你定位——这两个在 Apple Silicon 上最容易卡。
