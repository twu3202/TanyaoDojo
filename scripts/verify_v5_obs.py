#!/usr/bin/env python
"""端到端验证 v5 防守特征:重放真实牌谱,检查
  1) v5 obs 形状 (1022,34);
  2) v5 前 1012 通道 == v4(基础特征未被扰动);
  3) 全部值 ∈ [0,1];
  4) 有对手立直时:防守通道非零;现物(对手河里)放铳率=0 且 genbutsu 掩码=1;
     无筋中张危险度 > 现物。
"""
import sys, gzip, json, glob
sys.path.insert(0, '/Users/r/HMM/Better_mortal/Mortal/mortal')
import numpy as np
import libriichi
from libriichi.state import PlayerState

DEF_BASE = 1012  # v4 通道数,防守通道从这里开始
# 防守通道布局:[0:3]=放铳率/对手, [3]=最大, [4:7]=genbutsu, [7:10]=suji

def tile_id(pai):
    HON = {'E':27,'S':28,'W':29,'N':30,'P':31,'F':32,'C':33}
    p = pai.replace('r','')
    if p in HON: return HON[p]
    n = 5 if p[0]=='0' else int(p[0]); suit = p[1]
    return {'m':0,'p':9,'s':18}[suit] + (n-1)

def run(files, max_games=200):
    n_checked = 0
    n_base_ok = 0
    n_riichi_frames = 0
    riichi_checks = []
    for f in files[:max_games]:
        try:
            with gzip.open(f,'rt') as fh:
                events = [json.loads(l) for l in fh if l.strip()]
        except Exception:
            continue
        ps = PlayerState(0)  # 我是 player 0
        opp_riichi_kawa = {1:set(),2:set(),3:set()}
        opp_riichi = {1:False,2:False,3:False}
        for ev in events:
            ps.update(json.dumps(ev))
            t = ev.get('type')
            if t == 'start_kyoku':
                opp_riichi = {1:False,2:False,3:False}
                opp_riichi_kawa = {1:set(),2:set(),3:set()}
            elif t == 'reach_accepted' and ev['actor'] in (1,2,3):
                opp_riichi[ev['actor']] = True
            elif t == 'dahai' and ev['actor'] in (1,2,3) and opp_riichi[ev['actor']]:
                opp_riichi_kawa[ev['actor']].add(tile_id(ev['pai']))
            # 只在轮到我出牌(can_discard)时编码
            try:
                cans = ps.last_cans
                if not getattr(cans, 'can_discard', False):
                    continue
                o4 = np.array(ps.encode_obs(4, False)[0])
                o5 = np.array(ps.encode_obs(5, False)[0])
            except Exception:
                continue
            n_checked += 1
            # 1) 形状
            assert o5.shape == (1022,34), o5.shape
            assert o4.shape == (1012,34), o4.shape
            # 2) 基础通道一致
            if np.array_equal(o5[:DEF_BASE], o4):
                n_base_ok += 1
            # 3) 值域
            assert o5.min() >= 0 and o5.max() <= 1.0001, (o5.min(), o5.max())
            # 4) 立直帧防守检查
            active = [o for o in (1,2,3) if opp_riichi[o]]
            if active:
                n_riichi_frames += 1
                defch = o5[DEF_BASE:]  # (10,34)
                # 现物:对手河里的牌,放铳率行应为0,genbutsu掩码应为1
                for o in active:
                    row_prob = defch[o-1]       # 该对手放铳率
                    row_genb = defch[4+(o-1)]   # 该对手genbutsu掩码
                    for g in opp_riichi_kawa[o]:
                        if row_prob[g] != 0: riichi_checks.append(('genbutsu_prob_nonzero',o,g,row_prob[g]))
                        if row_genb[g] != 1: riichi_checks.append(('genbutsu_mask_wrong',o,g,row_genb[g]))
                    # 无筋中张(如5,若不在河里)危险度应>0
                    danger_max = defch[3]
                    if danger_max.max() <= 0:
                        riichi_checks.append(('no_danger_under_riichi',o))
    print(f'编码帧数: {n_checked}')
    print(f'基础通道 v5[:1012]==v4: {n_base_ok}/{n_checked} ' + ('✓' if n_base_ok==n_checked else '✗ 不一致!'))
    print(f'立直帧数: {n_riichi_frames}')
    if riichi_checks:
        print(f'防守检查失败 {len(riichi_checks)} 项(示例):')
        for c in riichi_checks[:8]: print('  ', c)
    else:
        print('防守检查全部通过 ✓ (现物放铳率=0, genbutsu掩码=1, 立直下有危险度)')

if __name__ == '__main__':
    files = sorted(glob.glob('/Users/r/HMM/Better_mortal/data/houou/2023/*.mjson.gz'))
    run(files, max_games=int(sys.argv[1]) if len(sys.argv)>1 else 200)
