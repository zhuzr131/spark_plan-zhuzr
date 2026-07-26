#!/bin/bash
# QAOA Project 一键环境安装脚本
# 在项目根目录（qaoa_project/）下运行: bash setup.sh

set -e

echo "=== QAOA 环境安装 ==="

# 1. 检查 Python 版本（需要 3.10+）
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
echo "使用 Python: $PYTHON"
$PYTHON --version

# 2. 创建虚拟环境
if [ -d "venv" ]; then
    echo "venv 已存在，跳过创建。如需重建请先 rm -rf venv"
else
    echo "创建虚拟环境..."
    $PYTHON -m venv venv
fi

# 3. 激活并安装依赖
source venv/bin/activate
echo "安装依赖包..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. 验证
echo ""
echo "=== 运行环境验证 ==="
python verify_install.py

echo ""
echo "=== 安装完成！==="
echo "后续使用前先激活: source venv/bin/activate"
echo "一键跑全部实验: python run_all.py"
