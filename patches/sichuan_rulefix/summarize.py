"""Final summaries only after all five fixed evaluations completed."""
import json
import sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
engine = 'response_priority_drawn_tile_fix_20261118'
seeds = np.arange(10000, 13000)
challengers = [int(i) for i in sys.argv[2:]] or [101, 102]
names = ['legacy_vs_L1'] + [f'seed{i}_vs_{side}' for i in challengers for side in ('L1', 'legacy')]
reports, scores = {}, {}
for name in names:
    output = run/'eval'/name
    report = json.loads(output.with_suffix('.json').read_text())
    with np.load(output.with_suffix('.npz')) as data:
        np.testing.assert_array_equal(data['seed'], seeds)
        raw = data['raw']
    if (raw.shape != (len(seeds), 4) or not np.isfinite(raw).all()
            or report['status'] != 'complete' or report['engine'] != engine
            or report['games'] != 12000 or report['shadow_checks']['mask_mismatch']
            or report['shadow_checks']['fallback']
            or [(s['start'], s['groups']) for s in report['segments']] != [(10000, 1000), (11000, 2000)]):
        raise RuntimeError(f'invalid fixed evaluation {name}')
    scores[name] = raw.mean(axis=1)
    np.testing.assert_allclose(report['mean'], scores[name].mean())
    reports[name] = report

paired = {}
for i in challengers:
    delta = scores[f'seed{i}_vs_L1'] - scores['legacy_vs_L1']
    paired[str(i)] = dict(groups=len(delta), mean=float(delta.mean()),
                          ci95_halfwidth=float(1.96*delta.std(ddof=1)/np.sqrt(len(delta))))
summary = dict(status='complete', engine=engine, paired_against_legacy_on_L1=paired, evaluations=reports)
text = json.dumps(summary, indent=2)
tmp = run/'results.json.new'
tmp.write_text(text)
tmp.replace(run/'results.json')
print(text, flush=True)
