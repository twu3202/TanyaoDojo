"""续跑 C'(λ=0.2)的崩溃保险 —— 重建当年丢失的哨兵规则:连续两次 test_play avg_pt < -8 -> 优雅停机。

test_play 与神圣协议没有固定偏移,不能用来判断"是否追平";但判断"崩了"足够(价值线首夜从 +1.3
单调崩到 -32,进程始终活着)。只读续跑开始之后生成的 TensorBoard 事件文件,避免每轮扫描 4000+ 个旧文件。
"""
import datetime, glob, os, subprocess, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

B = os.path.expanduser("~/Projects/better_mortal")
TB = f"{B}/runs/selfplay/tb"
LOG = f"{B}/runs/selfplay/collapse_guard.log"
STOP = f"{B}/scripts/stop_selfplay.sh"
START = datetime.datetime(2026, 9, 12, 3, 30).timestamp()
THRESH = -8.0

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d_%H:%M:%S ") + msg + "\n")

log(f"GUARD START(规则:连续两次 test_play avg_pt < {THRESH} -> stop_selfplay.sh)")
seen = 0
while True:
    time.sleep(1800)
    pts = []
    for f in glob.glob(TB + "/**/events.out.tfevents.*", recursive=True):
        if os.path.getmtime(f) < START:
            continue
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "test_play/avg_pt" in ea.Tags().get("scalars", []):
            pts += [(e.wall_time, e.value) for e in ea.Scalars("test_play/avg_pt") if e.wall_time >= START]
    vals = [v for _, v in sorted(pts)]
    if len(vals) > seen:
        log(f"test_play n={len(vals)} 最近: " + " ".join(f"{v:+.2f}" for v in vals[-4:]))
        seen = len(vals)
    if len(vals) >= 2 and vals[-1] < THRESH and vals[-2] < THRESH:
        log("ALERT: 连续两次 < -8,执行 stop_selfplay.sh(state.pth 保留)")
        subprocess.run(["bash", STOP])
        break
