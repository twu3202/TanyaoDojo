"""C' replay on trx with July's topology (arm ct): writes the arm's config from the July config embedded in C392k's
state and checks it key by key.
    python ct_setup.py SRC SP CFG PORT CPRIME_FINAL
- paths under July's run directory (runs/selfplay) are moved to SP, and the port 5577 becomes PORT;
- LR peak/final are set to 5e-6, the LR that ran in July: the scheduler restored from the state kept 5e-6, while the
  config said 1e-5 (the LR trap). The state's own scheduler says 5e-6 either way.
The written file is read back and compared with the configs embedded in C392k (lambda 0.5) and C' final (lambda 0.2).
The only differences allowed are the moved paths, the port and peak/final, plus the anchor weight against C' final.
Everything else stays July's, including loader num_workers 0 and test_play 2,000 games."""
import sys

import toml
import torch

SRC, SP, CFG, PORT, CPF = sys.argv[1:6]
OLD = "/home/twu/Projects/better_mortal/runs/selfplay"


def conv(x):
    if isinstance(x, dict):
        return {k: conv(v) for k, v in x.items()}
    if isinstance(x, list):
        return [conv(v) for v in x]
    if isinstance(x, str):
        return x.replace(OLD, SP)
    if type(x) is int and x == 5577:
        return int(PORT)
    return x


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


july = torch.load(SRC, weights_only=True, map_location="cpu")["config"]
final = torch.load(CPF, weights_only=True, map_location="cpu")["config"]
c = conv(july)
c["optim"]["scheduler"]["peak"] = 5e-6
c["optim"]["scheduler"]["final"] = 5e-6
assert c["anchor"]["weight"] == 0.5, c["anchor"]
with open(CFG + ".new", "w") as f:
    f.write("# C' replay on trx, July topology (patches/cprime_replay/README.md): written by ct_setup.py from the config\n"
            "# embedded in C392k's July state; paths moved to the arm's directory, port %s, LR peak/final 5e-6.\n" % PORT)
    toml.dump(c, f)
back = flat(toml.load(CFG + ".new"))
assert [k for k in back if k == "weight" or k.endswith(".weight")] == ["anchor.weight"], "more than one weight key"
ok = True
for name, ref in (("C392k", flat(july)), ("C' final", flat(final))):
    if set(back) != set(ref):
        print(f"!! key sets differ from {name}: {sorted(set(back) ^ set(ref))}")
        ok = False
        continue
    diffs = []
    for k in sorted(ref):
        a, b = ref[k], back[k]
        if a == b and type(a) is type(b):
            continue
        moved = isinstance(a, str) and OLD in a and b == a.replace(OLD, SP)
        port = type(a) is int and a == 5577 and b == int(PORT)
        lr = k in ("optim.scheduler.peak", "optim.scheduler.final") and b == 5e-6
        lam = name == "C' final" and k == "anchor.weight" and (a, b) == (0.2, 0.5)
        tag = "path" if moved else "port" if port else "lr" if lr else "lambda" if lam else "UNEXPECTED"
        ok &= tag != "UNEXPECTED"
        diffs.append(f"{k}: {a!r} -> {b!r} [{tag}]")
    print(f"config vs {name}: {len(diffs)} differences")
    for d in diffs:
        if not d.endswith("[path]"):
            print("   ", d)
    print(f"    ({sum(d.endswith('[path]') for d in diffs)} moved paths)")
print("loader num_workers", back["dataset.num_workers"], "| test_play games", back["test_play.games"],
      "| pts", back["env.pts"] if "env.pts" in back else [v for k, v in back.items() if k.endswith(".pts")],
      "| boltzmann", back.get("train_play.default.boltzmann_temp"), back.get("train_play.default.boltzmann_epsilon"))
if not ok:
    sys.exit("!! config check FAILED")
import os
os.replace(CFG + ".new", CFG)
print("config written:", CFG)
