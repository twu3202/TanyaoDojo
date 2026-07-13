# 评测结果记录

## Phase 1 baseline:v1 对 Mortal v4(10 万局复式 1v3)

评测方法:固定牌山种子(seed_key=20260711),challenger 轮换 4 座位消除运气,
jun_pt=[90,45,0,-135],champion = 本地 Mortal v4 全量权重(256ch×54blk,离线+在线RL的完全体)。

| checkpoint | 步数 | avg_rank | avg_pt (95% CI) | 判定 | 评测环境 |
|---|---|---|---|---|---|
| **v1_best** | 540k | **2.5434 ± 0.007** | **-3.99 ± 0.53** | 略低于 v4 | AutoDL 4090 |
| v1_final | 1.21M(满 epoch) | 2.5613 ± 0.007 | -5.10 ± 0.53 | 略低于 v4 | 本地 RTX 6000 Ada |

四位率:best 27.27% / final 27.60%(均衡为 25%)——失分主因是四位偏多。

### 关键发现

1. **v1_best(54万步)显著强于 v1_final(121万步满训)**,差约 1.1pt,两者置信区间
   不重叠(best [-4.53,-3.46] vs final [-5.64,-4.57])。**训满整个 epoch 反而变弱了。**
   - 训练中 1000 局的 test_play 噪声太大(全程在 -2~-9 抖),没能反映这个差异;
     10 万局才把它压出来。
   - 疑因:当前 LR schedule 是**恒定 1e-4 无衰减**(config 里 warm_up=max_steps=1000,
     peak=final=1e-4),长训后期策略震荡/退化;加余弦衰减到更低 final LR 很可能救回来。
   - **Phase 2 待办**:① LR 加正经衰减;② 认真做 checkpoint 选择(best 靠 test_play 选出,
     但 test_play 太噪,应改成定期跑更大规模评测选点)。

2. **-4pt 是"半成品对成品"的合理差距**:v1 只做了离线 CQL(Phase 1),
   v4 是离线+在线 RL 的完全体。纯离线模型落后在线微调模型 4pt 属正常范围
   (Mortal 官方版本差 ~1.2pt/代)。**在线 RL(Phase 2b)正是补这段差距的手段。**

3. **管线验证通过** ✅:从零用自有数据+自有环境训出逼近 v4 的模型,无翻车。
   Phase 1 目标达成,可进 Phase 2。

### v1 交付结论

**采用 v1_best(54万步)作为 Phase 1 交付权重**——它是两者中更强的,-4pt 距 v4。
备份:`weights_backup/v1_best.pth`(交付版)、`weights_backup/v1_final.pth`(对比留档)。
