import re, sys, numpy as np
counts = np.zeros(4)
for line in open(sys.argv[1]):
    m = re.search(r"challenger rankings: \[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]", line)
    if m: counts += np.array([int(x) for x in m.groups()])
n = counts.sum()
if n == 0:
    print("no rankings parsed"); sys.exit(1)
p = counts/n; ranks = np.arange(1,5); pts = np.array([90,45,0,-135])
avg_rank=(p*ranks).sum(); avg_pt=(p*pts).sum()
se_pt=((p*(pts-avg_pt)**2).sum()/n)**0.5; se_rank=((p*(ranks-avg_rank)**2).sum()/n)**0.5
print(f"=== AGGREGATE over {int(n)} games ===")
print(f"rank dist: 1st={counts[0]:.0f} 2nd={counts[1]:.0f} 3rd={counts[2]:.0f} 4th={counts[3]:.0f}")
print(f"avg_rank = {avg_rank:.4f} +/- {1.96*se_rank:.4f}")
print(f"avg_pt   = {avg_pt:.3f} +/- {1.96*se_pt:.3f} pt/game (95% CI)")
print("verdict:", "v1_final BEATS v4" if avg_pt-1.96*se_pt>0 else ("MATCHES v4" if avg_pt+1.96*se_pt>0 else "below v4"))
