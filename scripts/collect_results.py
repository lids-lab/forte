#!/usr/bin/env python
"""Aggregate FORTE run logs into per-(db,task) tables grouped by config tag.

Parses logs/forte_forte-<tag>-<short>_<jobid>.log: tag+db from filename, and the
test metrics from the 'Best test metrics' block. Prints a tidy table.
"""
import glob, os, re, sys
from collections import defaultdict

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "logs")

# (tag, db, task) -> metric
res = {}
for f in glob.glob(os.path.join(LOGDIR, "forte_forte-*.log")):
    base = os.path.basename(f)[len("forte_forte-"):]
    base = re.sub(r"_\d+\.log$", "", base)         # strip _<jobid>.log
    parts = base.rsplit("-", 1)
    if len(parts) == 2:
        tag, short = parts
    else:
        tag, short = "full", parts[0]
    if tag in ("smoke", "linkpred", "lpall"):
        continue
    txt = open(f, errors="ignore").read()
    if "Best test metrics" not in txt:
        continue
    for m in re.finditer(r"(rel-[\w-]+)/([\w-]+)/test:\s*([-\d.]+)", txt):
        db, task, val = m.group(1), m.group(2), float(m.group(3))
        res[(tag, db, task)] = val

# organize
tasks = sorted({(db, task) for (_, db, task) in res})
tags = sorted({tag for (tag, _, _) in res})
clf = {"driver-dnf","driver-top3","study-outcome","user-badge","user-engagement",
       "user-visits","user-clicks","user-churn","item-churn"}

print(f"configs found: {tags}\n")
hdr = f"{'db':10s} {'task':16s} " + " ".join(f"{t:>10s}" for t in tags)
print(hdr); print("-"*len(hdr))
for db, task in tasks:
    row = f"{db:10s} {task:16s} " + " ".join(
        (f"{res[(t,db,task)]:10.4f}" if (t,db,task) in res else f"{'-':>10s}") for t in tags)
    print(row)

# best-per-task across CLP tags and across PSP tags (for ablation columns)
def best(prefix, db, task):
    vals = [v for (t,d,k),v in res.items() if d==db and k==task and t.startswith(prefix)]
    return max(vals) if vals else None
print("\nbest +CLP (over lambda) and best +PSP (over alpha,sched) per task:")
for db, task in tasks:
    bc, bp = best("clp", db, task), best("psp", db, task)
    base = res.get(("base", db, task))
    full = res.get(("full", db, task))
    print(f"  {db:10s} {task:16s}  base={base}  bestCLP={bc}  bestPSP={bp}  full={full}")
