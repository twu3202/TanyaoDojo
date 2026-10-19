"""Online transfer test: the collapse guard of the online runs, with the run directory as a parameter.
    python xfer_collapse_guard.py SP
Two test_play readings in a row below -8 (800 games vs v4 each; noisy, but a collapse goes far below: the first night
of the value line fell from +1.3 to -32 with every process alive) -> stop the run, state.pth kept. Reads only the
TensorBoard events of this run written after the guard started."""
import glob, os, subprocess, sys, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SP = sys.argv[1]
TB, LOG = f"{SP}/tb", f"{SP}/collapse_guard.log"
STOP = ["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "xfer_helpers.sh"), "stop", SP]
START = time.time() - 60
THRESH = -8.0


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d_%H:%M:%S ") + msg + "\n")


log(f"GUARD START (two test_play avg_pt < {THRESH} in a row -> stop)")
seen = 0
while True:
    time.sleep(900)
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
        log(f"test_play n={len(vals)} latest: " + " ".join(f"{v:+.2f}" for v in vals[-4:]))
        seen = len(vals)
    if len(vals) >= 2 and vals[-1] < THRESH and vals[-2] < THRESH:
        log("ALERT: two in a row below -8, stopping the run (state.pth kept)")
        subprocess.run(STOP)
        break
