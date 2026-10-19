"""Online transfer test: how often the trainer saw each fresh sample.
    python xfer_reuse.py SP
Each trainer subprocess drains the buffer once and reads that file list (num_epochs = 1e9) until the next test_play
boundary, then exits. Per drain: files, batches trained on it, and reuse = batches x 512 / trainee samples, with the
trainee samples per file counted by the trainer's own reader on up to 300 files of the last drain left in SP/drain."""
import os, random, re, sys
import numpy as np

SP = sys.argv[1]
BATCH = 512
lines = open(f"{SP}/logs/trainer.log", encoding="utf-8", errors="replace").read().splitlines()
drains = []
for i, ln in enumerate(lines):
    m = re.search(r"file list size: ([\d,]+)", ln)
    if m:
        nxt = next((re.search(r"total steps: ([\d,]+)", x) for x in lines[i + 1:i + 3] if "total steps" in x), None)
        drains.append([int(m.group(1).replace(",", "")), int(nxt.group(1).replace(",", "")) if nxt else None])
last = max((int(x.replace(",", "")) for x in re.findall(r"total steps: ([\d,]+)", "\n".join(lines))), default=0)
for j, d in enumerate(drains):
    d.append((drains[j + 1][1] if j + 1 < len(drains) else last) - d[1])

spf = float("nan")
files = sorted(os.listdir(f"{SP}/drain"))
if files:
    sys.path.insert(0, os.path.expanduser("~/Projects/better_mortal/Mortal/mortal"))   # libriichi.so of the trainer
    from libriichi.dataset import GameplayLoader
    loader = GameplayLoader(version=4, oracle=False, player_names=["trainee"], excludes=[])
    random.seed(0)
    pick = random.sample(files, min(300, len(files)))
    n = [sum(len(g.take_actions()) for g in loader.load_gz_log_files([os.path.join(SP, "drain", f)])[0]) for f in pick]
    spf = float(np.mean(n))
    print(f"trainee samples per file: {spf:.1f} (+- {1.96 * np.std(n, ddof=1) / np.sqrt(len(n)):.1f}, {len(n)} files of the last drain)")
print(f"{len(drains)} drains, {last:,} steps")
for files_, start, batches in drains:
    print(f"  drain at {start:>7,}: {files_:>6,} files, {batches:>5,} batches, reuse {batches * BATCH / (files_ * spf):.2f}")
tot_b = sum(d[2] for d in drains)
tot_s = sum(d[0] for d in drains) * spf
print(f"overall: {tot_b:,} batches = {tot_b * BATCH:,} samples trained on {tot_s:,.0f} fresh trainee samples, reuse {tot_b * BATCH / tot_s:.2f}")
