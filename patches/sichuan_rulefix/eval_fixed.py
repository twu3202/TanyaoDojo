"""Strict fixed two-segment Sichuan evaluation, isolated outputs, no record selection."""
import argparse
import json
import math
import statistics
from pathlib import Path
import numpy as np
from sichuan import eval_h2h as E

ENGINE = 'response_priority_drawn_tile_fix_20261118'

p = argparse.ArgumentParser()
p.add_argument('candidate')
p.add_argument('opponent')
p.add_argument('--output', required=True)
p.add_argument('--start', type=int, default=10000)
p.add_argument('--groups', type=int, default=3000)
p.add_argument('--batch-groups', type=int, default=0)
p.add_argument('--verify-serial', action='store_true')
a = p.parse_args()
if a.groups < 2 or a.batch_groups < 0 or (a.verify_serial and not a.batch_groups):
    p.error('invalid group sizes or serial verification without batching')
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
A, B = E.make_pick(a.candidate), E.make_pick(a.opponent)


def serial_group(seed, traces=None):
    return [E.play_deal(seed, seat, A, B, rng_seed=seed*4+seat,
                        trace=None if traces is None else traces[seat]) for seat in range(4)]


rows = []
if a.batch_groups:
    from eval_batched import BatchedMatches
    batch = BatchedMatches(A, B, a.batch_groups)
chunk = a.batch_groups or 50
split = min(1000, a.groups)
segments = [(a.start, a.start + split)]
if split < a.groups:
    segments.append((a.start + split, a.start + a.groups))
batches = ((start, min(start + chunk, end)) for begin, end in segments for start in range(begin, end, chunk))
for start, stop in batches:
    if a.batch_groups:
        traces = [[] for _ in range(4 * (stop - start))] if a.verify_serial else None
        scores = batch.play(range(start, stop), traces)
        if a.verify_serial:
            want_traces = [[] for _ in traces]
            want_scores = [serial_group(seed, want_traces[4*i:4*i+4])
                           for i, seed in enumerate(range(start, stop))]
            np.testing.assert_array_equal(scores, want_scores)
            for index, (got, want) in enumerate(zip(traces, want_traces)):
                if got != want:
                    raise RuntimeError(f'batch/serial action history mismatch for seed {start + index//4}, seat {index%4}')
    else:
        scores = np.asarray([serial_group(seed) for seed in range(start, stop)])
    vals = scores.diagonal(axis1=1, axis2=2)
    if E.STATS['fallback'] or E.STATS['mask_mismatch']:
        raise RuntimeError(f'shadow check failed: {E.STATS}')
    if not np.isfinite(vals).all():
        raise RuntimeError('non-finite result')
    rows.extend(vals)
    means = np.asarray(rows).mean(axis=1)
    np.savez(out.with_suffix('.partial.npz'), seed=np.arange(a.start, stop), raw=np.asarray(rows), mean=means)
    print(f'{len(rows)}/{a.groups} groups mean={means.mean():+.4f}', flush=True)
raw = np.asarray(rows, dtype=float)
means = raw.mean(axis=1)
ci = 1.96*statistics.stdev(means)/math.sqrt(len(means))
np.savez(out.with_suffix('.npz'), seed=np.arange(a.start, a.start+a.groups), raw=raw, mean=means)
result = dict(candidate=a.candidate, opponent=a.opponent, start=a.start, groups=a.groups, games=4*a.groups,
              mean=float(means.mean()), ci95_halfwidth=ci, shadow_checks=E.STATS,
              segments=[dict(start=begin, groups=end-begin, mean=float(means[begin-a.start:end-a.start].mean()))
                        for begin, end in segments],
              engine=ENGINE, batch_groups=a.batch_groups, serial_trace_verified=a.verify_serial, status='complete')
out.with_suffix('.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
