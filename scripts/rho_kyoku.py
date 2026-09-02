"""
复核诊断报告主张③:"单局(kyoku)尺度与最终顺位的相关 ρ < 0.15,故组相对路线不可用"。

做法:用 GRP 数据集(每行 = 一局开局时的局面 X + 该半庄最终顺位点 Y)重建对局分段,
取同一对局相邻两行的点数差作为"该局四家得失",与最终顺位点求相关。

分段判据:round 回落,或 (东1 & 0本场 & 四家 25000)。已用 "Y 变化处必被判为新局"
反向校验:漏检 0 例。注意每局最后一局(南4/终局)没有后继行,故其得失未计入
—— 那恰是最决定顺位的一局,因此本估计偏保守(真值只会更高)。

用法: python3 scripts/rho_kyoku.py [grp_data_dir]
"""
import sys, glob
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "/home/r/grp_data"
Xs, Ys = [], []
for f in sorted(glob.glob(f"{d}/grp_*.npz")):
    z = np.load(f); Xs.append(z["X"]); Ys.append(z["Y"])
X = np.concatenate(Xs); Y = np.concatenate(Ys) * 135.0   # 还原顺位点 [90,45,0,-135]
rnd, hon, sc = X[:, 0], X[:, 1], X[:, 3:7]

newg = np.zeros(len(X), bool); newg[0] = True
newg[1:] = (rnd[1:] < rnd[:-1]) | ((rnd[1:] == 0) & (hon[1:] == 0) &
                                   (np.abs(sc[1:] - 25).sum(1) < 1e-6))
gid = np.cumsum(newg) - 1
ychg = np.zeros(len(X), bool); ychg[1:] = (Y[1:] != Y[:-1]).any(1)
print(f"[校验] 分段漏检(Y 变了却未判新局) = {(ychg & ~newg).sum()}  对局数 = {gid.max()+1:,}")

same = gid[1:] == gid[:-1]
d_pt = (sc[1:] - sc[:-1])[same] * 1000.0      # 该局四家得失点
pt = Y[:-1][same]                              # 该半庄最终顺位点
dd, pp = d_pt.ravel(), pt.ravel()
rho = float(np.corrcoef(dd, pp)[0, 1])
print(f"样本(局×家) = {dd.size:,}")
print(f"[整体] rho = {rho:.4f}   R^2 = {rho**2:.4f}   (单局得失 sd={dd.std():.0f}点)")

names = ["东1", "东2", "东3", "东4", "南1", "南2", "南3", "南4"]
kk = np.broadcast_to(rnd[:-1][same][:, None], d_pt.shape).ravel()
for k in range(8):
    m = kk == k
    if m.sum() < 5000:
        continue
    r = float(np.corrcoef(dd[m], pp[m])[0, 1])
    print(f"  {names[k]}: n={m.sum():>10,}  rho={r:.4f}")
print(f"\n[判据] 报告主张 rho<0.15 → 实测 {rho:.4f} → "
      f"{支持 if abs(rho) < 0.15 else 不支持}")
