#!/bin/bash
# 云实例开箱脚本(AutoDL/优云智算,Ubuntu 22.04 + Python 3.10 镜像)
# 用法: bash cloud_setup.sh main   # 主力机(JAX 训练)
#       bash cloud_setup.sh aux    # 辅机(bc 训练 env + eval 评测 env)
# 版本钉死自已验证环境(服务器 mahjax_env + 本地 Mortal .venv),勿升级。
set -e
ROLE=${1:-main}

JAX_PKGS="jax[cuda12]==0.6.2 flax==0.10.7 optax==0.2.8 distrax==0.1.5 chex==0.1.90 \
numpy==2.2.6 pydantic==2.13.4 omegaconf==2.3.1 svgwrite"

if [ "$ROLE" = "main" ]; then
  pip install -q $JAX_PKGS
  python - << "EOF"
import jax, flax
print("main OK:", jax.__version__, flax.__version__, jax.devices())
EOF
else
  conda create -y -n bc python=3.10 >/dev/null
  conda run -n bc pip install -q $JAX_PKGS
  conda create -y -n eval python=3.10 >/dev/null
  conda run -n eval pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  conda run -n eval pip install -q jax==0.6.2 flax==0.10.7 numpy==2.2.6 toml tqdm
  echo "eval env: 另需拷入 libriichi.so(py3.10/ubuntu22.04 版)与 Mortal/mortal 源码目录"
  conda run -n bc python -c "import jax; print(\"bc OK:\", jax.devices())"
  conda run -n eval python -c "import torch; print(\"eval OK:\", torch.cuda.is_available())"
fi
echo "SETUP_DONE role=$ROLE"
