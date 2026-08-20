# 环境搭建

本仓库只包含**本项目自己的代码**。评测桥依赖上游 [Mortal](https://github.com/Equim-chan/Mortal)
的 `libriichi`(AGPL-3.0),按其许可条款自行克隆与构建,不随本仓库分发。

## 1. 依赖版本(钉死,勿升级)

```
python 3.10 · jax[cuda12]==0.6.2 · flax==0.10.7 · optax==0.2.8
distrax==0.1.5 · chex==0.1.90 · numpy==2.2.6 · pydantic==2.13.4 · omegaconf==2.3.1
```

环境安装脚本见 [`jax_rl/cloud_setup.sh`](jax_rl/cloud_setup.sh)(云机/新机一键起栈)。

## 2. 麻将环境(Mahjax)

```bash
git clone https://github.com/nissymori/mahjax ~/mahjax
export PYTHONPATH=~/mahjax
```

## 3. 评测桥依赖(libriichi,AGPL-3.0)

```bash
git clone https://github.com/Equim-chan/Mortal ~/Mortal   # 仓库根同级亦可
cd ~/Mortal/libriichi && cargo build --release            # 产出 libriichi.so
```

评测脚本需要 `Mortal/mortal` 在 `PYTHONPATH` 上,并需要一份对手权重放在
`baseline/`(权重不随本仓库分发,申领方式见上游说明)。

## 4. 训练数据

**不分发**:BC 数据集由天凤凤凰卓牌谱构建,受天凤条款约束,本项目不再分发原始牌谱
或其派生数据集。请自备牌谱后用 [`jax_rl/data_bridge/make_bc_dataset.py`](jax_rl/data_bridge/make_bc_dataset.py)
自行构建:

```bash
PYTHONPATH=~/mahjax python make_bc_dataset.py "<牌谱glob>" <局数> <输出目录> v2
```

## 5. 冒烟自检

```bash
python jax_rl/mjai_bot/test_tracker_diff.py "<牌谱glob>" 100 v2   # 观测差分应全等
```
