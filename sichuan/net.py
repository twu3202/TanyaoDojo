"""
川麻 AC 网络(P2)。与立直线的 `net_lean.LeanACNet` 同构,差别只在三处形状:
27 列(万筒条各 9,无字牌)、22 平面 + 26 标量、61 动作。

规模默认 128ch × 6blk ≈ 1.3M 参数——川麻状态空间比立直小(无字牌/无役/无宝牌),
先用小网跑通 P2 的判据,容量留到 P3 再加。
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from sichuan.env_jax import NUM_ACTIONS
from sichuan.obs import NUM_PLANES, NUM_SCALARS

NUM_COLS = 27


def orthogonal_init(scale: float = 2.0 ** 0.5):
    return nn.initializers.orthogonal(scale)


class ResBlock1D(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.relu(y)
        y = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(y)
        y = nn.LayerNorm()(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, kernel_size=(3,), kernel_init=orthogonal_init())(y)
        return x + y


class SichuanACNet(nn.Module):
    channels: int = 128
    blocks: int = 6
    head_dim: int = 256

    @nn.compact
    def __call__(self, obs: dict):
        planes = jnp.asarray(obs["planes"], jnp.float32)      # (B, 27, 22)
        scalars = jnp.asarray(obs["scalars"], jnp.float32)    # (B, 26)
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
        logits = nn.Dense(NUM_ACTIONS, kernel_init=orthogonal_init(0.01))(trunk)
        value = nn.Dense(1, kernel_init=orthogonal_init())(trunk).squeeze(-1)
        return logits, value
