#!/usr/bin/env python
"""从 houou mjai 语料统计"对立直的放铳率表"(Track B / B1 第3步)。

对每个"在有对手立直状态下的舍牌",按 (牌类别, 现物/筋/无筋, 巡目桶) 分桶,
统计放铳率 = 该桶内被荣和的次数 / 该桶内舍牌总数。
先小样本验证方法正确性(现物≈0、19字最安全、无筋中张最危险)。

用法: python build_deal_in_table.py <glob> [--limit N] [--out table.json]
"""
import sys, gzip, json, glob, argparse
from collections import defaultdict

# 牌 id: 0-8=m1-9, 9-17=p1-9, 18-26=s1-9, 27-33=字(E S W N P F C)
HONORS = {'E':27,'S':28,'W':29,'N':30,'P':31,'F':32,'C':33}
def tile_id(pai):
    p = pai.replace('r','')  # 去赤
    if p in HONORS: return HONORS[p]
    n = int(p[0]); suit = p[1]
    if p[0] == '0': n = 5  # 赤5
    base = {'m':0,'p':9,'s':18}[suit]
    return base + (n-1)

def value_cat(tid):
    if tid >= 27: return 'honor'
    v = tid % 9 + 1  # 1..9
    return {1:'1/9',9:'1/9',2:'2/8',8:'2/8',3:'3/7',7:'3/7',4:'4/6',6:'4/6',5:'5'}[v]

def is_number(tid): return tid < 27
def suit_of(tid): return tid // 9
def num_of(tid): return tid % 9 + 1  # 1..9

def junme_bucket(j):
    if j <= 6: return '1-6'
    if j <= 12: return '7-12'
    return '13+'

def process_game(events, stats):
    """单局:重放事件,统计"对立直舍牌"的放铳。"""
    riichi_accepted = [False]*4      # 已成立立直
    riichi_declared_pending = [None]*4
    kawa = [set() for _ in range(4)]  # 各家舍牌(牌id集合),用于现物/筋判定
    junme = [0]*4
    last_dahai = None  # (actor, tid)
    for ev in events:
        t = ev.get('type')
        if t == 'start_kyoku':
            riichi_accepted = [False]*4
            kawa = [set() for _ in range(4)]
            junme = [0]*4
            last_dahai = None
        elif t == 'reach':
            riichi_declared_pending[ev['actor']] = True
        elif t == 'reach_accepted':
            riichi_accepted[ev['actor']] = True
        elif t == 'tsumo':
            junme[ev['actor']] += 1
        elif t == 'dahai':
            actor = ev['actor']; tid = tile_id(ev['pai'])
            # 是否存在别家已成立立直
            riichi_opps = [o for o in range(4) if o != actor and riichi_accepted[o]]
            if riichi_opps:
                # 对每个立直对手判定该牌类别,取"最危险"口径(min 安全度):
                #   现物 = 在该对手河里;筋 = 数牌且 ±3 在其河里
                cats = []
                for o in riichi_opps:
                    if tid in kawa[o]:
                        cats.append('genbutsu')
                    elif is_number(tid):
                        s = suit_of(tid); n = num_of(tid)
                        suji = any((num_of(k)==n-3 or num_of(k)==n+3) and suit_of(k)==s
                                   for k in kawa[o] if is_number(k))
                        cats.append('suji' if suji else 'nosuji')
                    else:
                        cats.append('nosuji')  # 字牌无筋概念,并入(后面按 value_cat 分开)
                # 综合:有任一现物则算现物(整体安全);否则若全是筋算筋;否则无筋
                if 'genbutsu' in cats: safety = 'genbutsu'
                elif all(c == 'suji' for c in cats): safety = 'suji'
                else: safety = 'nosuji'
                key = (value_cat(tid), safety, junme_bucket(max(junme)))
                stats[key]['n'] += 1
                last_dahai = (actor, tid, key)
            else:
                last_dahai = None
            kawa[actor].add(tid)
        elif t == 'hora':
            # 荣和:target 是放铳者;若刚好是我们记录的对立直舍牌,则计一次放铳
            if last_dahai is not None and ev.get('target') == last_dahai[0] \
               and ev.get('actor') != ev.get('target'):
                stats[last_dahai[2]]['deal_in'] += 1
            last_dahai = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('globs', nargs='+')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    files = []
    for g in args.globs: files.extend(glob.glob(g))
    if args.limit: files = files[:args.limit]
    stats = defaultdict(lambda: {'n':0,'deal_in':0})
    for i, f in enumerate(files):
        try:
            with gzip.open(f, 'rt') as fh:
                events = [json.loads(l) for l in fh if l.strip()]
            process_game(events, stats)
        except Exception as e:
            continue
        if (i+1) % 2000 == 0:
            print(f'  {i+1}/{len(files)} games', file=sys.stderr)
    # 汇总打印(按放铳率排序)
    rows = []
    for (cat, safety, jb), c in stats.items():
        if c['n'] >= 30:
            rate = c['deal_in']/c['n']
            rows.append((rate, cat, safety, jb, c['n'], c['deal_in']))
    rows.sort(reverse=True)
    print(f'\n{len(files)} games, {sum(c["n"] for c in stats.values())} 对立直舍牌')
    print(f'{"牌类":>6} {"安全度":>9} {"巡目":>6} {"放铳率":>8} {"样本":>8}')
    for rate, cat, safety, jb, n, di in rows:
        print(f'{cat:>6} {safety:>9} {jb:>6} {rate*100:>7.2f}% {n:>8}')
    if args.out:
        table = {f'{cat}|{safety}|{jb}': {'rate': c['deal_in']/c['n'] if c['n'] else 0,
                 'n': c['n']} for (cat,safety,jb),c in stats.items()}
        json.dump(table, open(args.out,'w'), ensure_ascii=False, indent=1)
        print(f'saved -> {args.out}', file=sys.stderr)

if __name__ == '__main__':
    main()
