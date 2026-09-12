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

### 2.1 必打的牌山 RNG 补丁(v0.1.3 之前的版本)

**训练前必须确认这个补丁在位,否则训练数据会被静默破坏。**
`v0.1.3`(2026-09-04,上游 commit `fade6de` / #73)之前的 mahjax 里,
`red_mahjong/env.py::_init` 调 `_make_state` 时没有传 `rng_key`,该字段保持 dataclass
默认的 `PRNGKey(0)`;而第 2 局起的牌山取自 `split(round_state.rng_key)`,`Env.step`
拿到 key 后第一行就 `del key`。后果是**每个半庄的第 2..9 局都是同一组固定的 8 副牌**,
跨 seed、跨并行环境、跨 episode、跨 run 完全一样。`round_mode="single"` 不受影响,
**`round_mode="half"` 下 88.9% 的训练步落在常数牌山里**(实测)。

它不崩、不报错,牌数守恒/零和/mask 合法性/终局结算全部照常通过,只会安静地把
训练数据变成 8 副牌的重复;而评测走 `libriichi.arena` 不碰 mahjax,于是表现为
"内部指标好看、外部强度不动"。本项目 2026-08-23 至 09-07 的约 29.8 亿步 RL
(≈131 GPU-h)就栽在这里。

若用的是打补丁前的版本,改 `_init` 两处即可(**不必**升级到 v0.1.3 ——
它带 `step` 强制要 key、观测 schema 重写等 API 破坏,对本项目无收益):

```python
def _init(rng: PRNGKey, game_config=None) -> State:
    dealer_key, wall_key, round_key = jax.random.split(rng, 3)      # 原为 rng, subkey = split(rng)
    current_player = jnp.int8(jax.random.randint(dealer_key, (), 0, 4))   # 原用 rng
    ...
    deck = Tile.from_tile_id_to_tile(
        jax.random.permutation(wall_key, jnp.arange(136))).astype(jnp.int8)   # 原用 rng
    ...
    state = _make_state(..., rng_key=round_key)                     # 原来没有这一行
```

(第二处顺带修掉"庄家与牌山共用同一个 `rng`",即庄家是牌山的确定性函数。)

**改完必须验**,不要只看 diff:

```bash
python - <<'EOF'
import jax, numpy as np, mahjax
env = mahjax.make("red_mahjong", round_mode="half", observe_type="dict")
a = np.asarray(jax.jit(env.init)(jax.random.PRNGKey(1)).round_state.rng_key)
b = np.asarray(jax.jit(env.init)(jax.random.PRNGKey(999)).round_state.rng_key)
assert not np.array_equal(a, b), "rng_key 仍未播种,补丁没生效"
print("OK", a, b)
EOF
```

## 3. 评测桥依赖(libriichi,AGPL-3.0)

```bash
git clone https://github.com/Equim-chan/Mortal ~/Mortal   # 仓库根同级亦可
cd ~/Mortal/libriichi && cargo build --release            # 产出 libriichi.so
```

评测脚本需要 `Mortal/mortal` 在 `PYTHONPATH` 上,并需要一份对手权重放在
`baseline/`(权重不随本仓库分发,申领方式见上游说明)。

### 3.1 锚定值微调补丁(价值线 C',可选)

本项目目前最强的模型(12k 复式 `-0.75 ± 1.31` vs mortal_v4)靠的是给上游 trainer
加的一处**锚定损失**:在线训练时把**全部合法动作**的 Q 与冻结基座的 Q 取平方误差,
抑制"未采取动作的 Q 无监督漂移 → 动作排序被侵蚀"这条慢速病灶。补丁对上游是惰性的
—— 配置里没有 `[anchor]` 节或 `weight = 0` 时,行为与上游完全一致。

```bash
cd ~/Mortal && patch -p1 < /path/to/TanyaoDojo/patches/anchor_train.patch
```

λ(即 `[anchor] weight`)是这套方案的主旋钮,不是可有可无的小数:λ=0.5 时锚把策略
钉死,约 40 小时的 12k 配对只有 `+0.19`(z=0.33);放到 0.2 之后 9 小时就拿到
`+1.56`(z=2.37)。配置见 [`configs/online_selfplay.toml`](configs/online_selfplay.toml),
判读见 [RESULTS.md](RESULTS.md)。

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
