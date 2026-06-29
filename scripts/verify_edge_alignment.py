#!/usr/bin/env python
"""Validate the FORTE slot->role reconstruction in rt/data.py.   [Stage A/B check]

For sampled batches of a single DB, recomputes f2p_role_idxs and checks, for every
resolved slot (role > 0), that the role's expected parent table (from fk_roles.json)
equals the ACTUAL parent node's table (resolved via table_info intervals).

  contradiction = role assigned but parent table mismatch  (should be 0)
  resolved      = fraction of real f2p slots given a nonzero role

Usage:
  HOME=/home/<you> pixi run python scripts/verify_edge_alignment.py rel-f1 [n_batches]
"""
import sys
import numpy as np

from forte.data import (
    RelationalDataset,
    _global_fk_role_registry,
    _resolve_tables,
)
from forte import tasks as T

DB = sys.argv[1] if len(sys.argv) > 1 else "rel-f1"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# pick a forecast task for this DB
task = next(t for t in T.forecast_tasks if t[0] == DB)
db, table, target, drop = task
print(f"[verify_edge_alignment] {db}/{table}  n_batches={N_BATCHES}")

reg = _global_fk_role_registry()
# reverse map: global_role -> expected parent table
role_to_parent: dict[int, str] = {}
for (d, tbl), lst in reg["table_fklist"].items():
    if d != db:
        continue
    for parent_table, grole in lst:
        role_to_parent[grole] = parent_table

ds = RelationalDataset(
    tasks=[(db, table, target, "train", drop)],
    batch_size=16, seq_len=1024, rank=0, world_size=1,
    max_bfs_width=256, embedding_model="all-MiniLM-L12-v2", d_text=384, seed=0,
)

total_slots = 0      # real (non-null) f2p slots
resolved = 0         # nonzero role
contradictions = 0   # nonzero role but wrong parent table
role_set = set()

for bi in range(N_BATCHES):
    b = ds[bi % len(ds)]
    node_idxs = b["node_idxs"].numpy()
    f2p = b["f2p_nbr_idxs"].numpy()
    roles = b["f2p_role_idxs"].numpy()
    dsx = b["dataset_idxs"].numpy()
    B, S, _ = f2p.shape
    for r in range(B):
        if int(dsx[r, 0]) < 0:
            continue
        parent_tabs = _resolve_tables(db, f2p[r])  # (S,5)
        for s in range(S):
            for j in range(5):
                p = int(f2p[r, s, j])
                if p < 0:
                    continue
                total_slots += 1
                g = int(roles[r, s, j])
                if g > 0:
                    resolved += 1
                    role_set.add(g)
                    expected = role_to_parent.get(g, "?")
                    actual = parent_tabs[s, j]
                    if expected != actual:
                        contradictions += 1

print(f"  real f2p slots checked : {total_slots:,}")
print(f"  resolved (role>0)      : {resolved:,}  ({100*resolved/max(total_slots,1):.1f}%)")
print(f"  contradictions         : {contradictions:,}")
print(f"  distinct global roles  : {sorted(role_set)}")
print("  RESULT:", "PASS" if contradictions == 0 and resolved > 0 else "FAIL")
