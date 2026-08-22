"""
R2 轻量网络:34 列 1D-CNN 残差主干(Mortal 式)+ 共享干双头。

对照上游 examples/networks/red_network.py 的两点结构性省流:
  1) 免除 200-token 历史注意力(obs_lean 已把牌局状态结构化);
  2) policy/critic 共享主干(上游是两套独立 FeatureExtractor,纯 2x 浪费)。
规模档位由 channels/blocks 控制,默认 128ch x 6blk ~= 1.9M 参数。
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

NUM_ACTIONS = 87


def orthogonal_init(scale: float = 2.0 ** 0.5):
    return nn.initializers.orthogonal(scale)


class ResBlock1D(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):  # (B, 34, C)
        y = nn.LayerNorm()(x)
        y = nn.relu(y)
        y = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(y)
        y = nn.LayerNorm()(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(y)
        return x + y


class LeanCriticNet(nn.Module):
    """独立 value-only 网络,供非对称 AC:输入 oracle obs(37 平面/29 标量),
    结构与 LeanACNet 主干一致但只出 value。训练期专用,eval 时整套丢弃。"""

    channels: int = 128
    blocks: int = 6
    head_dim: int = 256

    @nn.compact
    def __call__(self, obs: dict):
        planes = jnp.asarray(obs["planes"], jnp.float32)
        scalars = jnp.asarray(obs["scalars"], jnp.float32)
        if planes.ndim == 2:
            planes, scalars = planes[None], scalars[None]
        s = nn.Dense(32, kernel_init=orthogonal_init())(scalars)
        s = nn.relu(s)
        s_tiled = jnp.repeat(s[:, None, :], planes.shape[1], axis=1)
        x = jnp.concatenate([planes, s_tiled], axis=-1)
        x = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(x)
        for _ in range(self.blocks):
            x = ResBlock1D(self.channels)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        flat = x.reshape(x.shape[0], -1)
        trunk = nn.Dense(self.head_dim, kernel_init=orthogonal_init())(flat)
        trunk = nn.relu(trunk)
        return nn.Dense(1, kernel_init=orthogonal_init())(trunk).squeeze(-1)


class LeanQCriticNet(nn.Module):
    """中心化 Q-critic:输入 oracle obs,输出每动作 Q 值(NUM_ACTIONS 维)。
    配合 Expected SARSA(λ) —— backup 用 sum_a pi(a|s)Q(s,a) 的策略加权期望,
    不采样下一动作,消除自博弈随机策略下 GAE 的采样方差(arXiv 2605.19235)。"""

    channels: int = 128
    blocks: int = 6
    head_dim: int = 256

    @nn.compact
    def __call__(self, obs: dict):
        planes = jnp.asarray(obs["planes"], jnp.float32)
        scalars = jnp.asarray(obs["scalars"], jnp.float32)
        if planes.ndim == 2:
            planes, scalars = planes[None], scalars[None]
        s = nn.Dense(32, kernel_init=orthogonal_init())(scalars)
        s = nn.relu(s)
        s_tiled = jnp.repeat(s[:, None, :], planes.shape[1], axis=1)
        x = jnp.concatenate([planes, s_tiled], axis=-1)
        x = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(x)
        for _ in range(self.blocks):
            x = ResBlock1D(self.channels)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        flat = x.reshape(x.shape[0], -1)
        trunk = nn.Dense(self.head_dim, kernel_init=orthogonal_init())(flat)
        trunk = nn.relu(trunk)
        return nn.Dense(NUM_ACTIONS, kernel_init=orthogonal_init())(trunk)


class LeanACNet(nn.Module):
    channels: int = 128
    blocks: int = 6
    head_dim: int = 256

    @nn.compact
    def __call__(self, obs: dict):
        planes = jnp.asarray(obs["planes"], jnp.float32)  # (B, 34, P)
        scalars = jnp.asarray(obs["scalars"], jnp.float32)  # (B, S)
        if planes.ndim == 2:
            planes, scalars = planes[None], scalars[None]

        # 标量走 Dense 后铺到 34 列并入通道(FiLM-lite)
        s = nn.Dense(32, kernel_init=orthogonal_init())(scalars)
        s = nn.relu(s)
        s_tiled = jnp.repeat(s[:, None, :], planes.shape[1], axis=1)  # (B,34,32)
        x = jnp.concatenate([planes, s_tiled], axis=-1)
        x = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(x)
        for _ in range(self.blocks):
            x = ResBlock1D(self.channels)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        flat = x.reshape(x.shape[0], -1)  # (B, 34*C)

        trunk = nn.Dense(self.head_dim, kernel_init=orthogonal_init())(flat)
        trunk = nn.relu(trunk)
        logits = nn.Dense(NUM_ACTIONS, kernel_init=orthogonal_init(0.01))(trunk)
        value = nn.Dense(1, kernel_init=orthogonal_init())(trunk).squeeze(-1)
        return logits, value
