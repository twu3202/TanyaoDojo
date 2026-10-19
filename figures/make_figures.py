"""Draws the README figures from the summary numbers below (all from this repository's own evaluations).
    python figures/make_figures.py      -> figures/*.png
Riichi numbers: avg placement points per hanchan vs mortal_v4, duplicate 1v3, seed_key 20260711, pt [90,45,0,-135];
the 95% CIs are over seed groups (4 rotations of one wall). Sources: RESULTS.md, HISTORY.md, patches/*/README.md."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C_ON, C_OFF, C_JAX, C_REF, C_GRAY = "#1f6feb", "#8957e5", "#d29922", "#2da44e", "#8c959f"

# 1. riichi scoreboard: every model with the full sacred 100k (seeds 10000-34999, the same 25,000 groups)
board = [  # label, mean, ci, kind
    ("tpt 408k: online RL, eval-proportional reward (best own)", -0.637, 0.458, "on"),
    ("Stage 1 448k: targeted exploration", -0.765, 0.458, "on"),
    ("v4 opponent pool 448k", -0.863, 0.462, "on"),
    ("C' line 368k: anchor λ 0.1, old reward", -0.886, 0.461, "on"),
    ("tpt 448k: 40k steps further", -0.932, 0.459, "on"),
    ("M1-sel 448k: calibrated anchor", -0.996, 0.455, "on"),
    ("M1 448k: global anchor margin 0.3", -1.038, 0.460, "on"),
    ("q25: best 25% of games by player quality (best offline)", -1.947, 0.451, "off"),
    ("q25one: same selection, one pass", -2.047, 0.443, "off"),
    ("symaug: 18-way symmetry augmentation", -2.494, 0.459, "off"),
    ("v11: offline baseline (MC + CQL)", -2.498, 0.455, "off"),
    ("B1: v11 recipe at v4's size (256×54)", -2.680, 0.464, "off"),
    ("v11long: v11 trained 1.8× longer", -3.758, 0.466, "off"),
    ("Mahjax line: BC base v2 g186 (JAX stack)", -4.66, 0.535, "jax"),
]
fig, ax = plt.subplots(figsize=(9.6, 6.2))
col = {"on": C_ON, "off": C_OFF, "jax": C_JAX}
y = np.arange(len(board))[::-1]
for yi, (lab, mu, ci, k) in zip(y, board):
    ax.errorbar(mu, yi, xerr=ci, fmt="o", color=col[k], ms=5, capsize=3, lw=1.4)
ax.errorbar(0.123, len(board), xerr=0.101, fmt="D", color=C_REF, ms=5, capsize=3, lw=1.4)
ax.set_yticks(list(y) + [len(board)])
ax.set_yticklabels([b[0] for b in board] + ["A²: fine-tuned from v4 itself (1.1M games; reference only)"])
ax.axvline(0, color="k", lw=1)
ax.text(-0.08, -1.2, "parity with v4 →", fontsize=8, va="center", ha="right")
ax.set_ylim(-1.6, len(board) + 0.6)
ax.set_xlabel("avg placement points per hanchan vs mortal_v4 (100k games, 95% CI)")
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                   ((C_ON, "own pipeline: offline base + online RL"), (C_OFF, "own pipeline: offline base only"),
                    (C_JAX, "from-scratch JAX line"), (C_REF, "starts from v4's weights"))],
          loc="upper left", fontsize=8, frameon=False)
fig.suptitle("Riichi: every model evaluated at 100k games on the same walls", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/riichi_scoreboard.png")

# 2. milestones of the two riichi lines (own pipelines only)
value = [("v1_best\noffline", -3.99), ("v11\noffline", -2.50), ("rl1_best\nonline RL", -1.96),
         ("C' final\nanchored RL", -0.75), ("tpt 408k\n(100k)", -0.637)]
mahjax = [("BC\n20k games", -14.29), ("BC\n353k games", -12.52), ("BC wide\n192×8", -8.87), ("BC 256×10\n10-yr pool", -6.95),
          ("14-yr pool\ng1080", -5.76), ("refined\ng402", -5.07), ("obs v2\ng186", -4.86), ("obs v2 g186\n(100k)", -4.66)]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [5, 8]})
for ax, ms, c, t in ((axs[0], value, C_ON, "Value line (Mortal stack)"), (axs[1], mahjax, C_JAX, "Mahjax line (JAX, from scratch)")):
    xs = np.arange(len(ms))
    ax.plot(xs, [v for _, v in ms], "-o", color=c, ms=5)
    for x, (_, v) in zip(xs, ms):
        ax.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _ in ms], fontsize=7.5)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_title(t, loc="left", fontsize=10)
axs[0].set_ylabel("avg pt vs mortal_v4")
axs[0].set_ylim(-5, 0.8)
axs[1].set_ylim(-16, 1.5)
fig.suptitle("Milestones in order. Value line: 100k readings except C' final (12k). Mahjax line: 4k-12k screens, last point 100k.", x=0.01, ha="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/milestones.png")

# 3. where human experts lose to the AI: true-wall rollouts of human-vs-AI disagreements (Tenhou Phoenix)
kinds = [("discard A → discard B", 0.133, 0.268), ("riichi → dama", 0.660, 0.511), ("call → pass", 0.622, 0.180),
         ("pass → call", 0.919, 0.220), ("dama → riichi", 2.382, 0.452)]
fig, ax = plt.subplots(figsize=(8.0, 3.1))
yy = np.arange(len(kinds))
ax.barh(yy, [k[1] for k in kinds], xerr=[k[2] for k in kinds], color=C_ON, alpha=0.85, capsize=3)
ax.set_yticks(yy)
ax.set_yticklabels([f"human {k[0].split(' → ')[0]}, AI {k[0].split(' → ')[1]}" for k in kinds])
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("AI's choice minus the human's (placement pt, 95% CI clustered by game)")
fig.suptitle("Human-vs-AI disagreements priced on the true walls (36,583 decisions, 3,991 Phoenix games)", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/rollout_disagreements.png")

# 4. Sichuan pure RL: head-to-head vs the strongest snapshot (b2272), 300 walls
steps = [4.19, 12.59, 25.17, 50.33, 105, 185, 298]
h2h = [(-5.216, 0.344), (-5.124, 0.365), (-1.422, 0.411), (-0.245, 0.369), (-0.058, 0.323), (-0.085, 0.333), (0.0, 0.0)]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.errorbar(steps, [h[0] for h in h2h], yerr=[h[1] for h in h2h], fmt="-o", color=C_REF, capsize=3, ms=5)
ax.axhline(-2.062, color=C_GRAY, ls="--", lw=1)
ax.text(4.3, -1.85, "hand-written rule bot L1", color=C_GRAY, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("environment steps (millions)")
ax.set_ylabel("pt per hand vs 3× b2272")
ax.set_title("Sichuan (Xuezhan) from scratch: learning curve, direct head-to-head", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sichuan_curve.png")

# 5. evaluation economics: current 12k gate + 100k vs a sequential test (simulated from our own 100k dumps)
d = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
cur_p, cur_n = [0.00, 0.00, 0.02, 0.11, 0.34, 0.65, 0.84], [23, 35, 55, 66, 77, 85, 90]
sp_p, sp_n = [0.00, 0.00, 0.02, 0.08, 0.26, 0.54, 0.80], [12, 17, 30, 42, 54, 54, 49]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].plot(d, cur_p, "-o", color=C_GRAY, label="12k gate, then 100k")
axs[0].plot(d, sp_p, "-o", color=C_ON, label="SPRT 0 vs +1 (α 0.025, β 0.2)")
axs[0].set_xlabel("true effect, candidate − control (pt)")
axs[0].set_ylabel("P(promote)")
axs[0].legend(frameon=False, fontsize=8)
axs[1].plot(d, cur_n, "-o", color=C_GRAY)
axs[1].plot(d, sp_n, "-o", color=C_ON)
axs[1].set_xlabel("true effect, candidate − control (pt)")
axs[1].set_ylabel("games played (thousands)")
fig.suptitle("Same decisions for about half the games (simulated from 11 real paired 100k comparisons)", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sequential_eval.png")
print("figures written to", OUT)
