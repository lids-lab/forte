#!/usr/bin/env python
"""Generate fk_roles.json for each preprocessed DB.   [FORTE Stage A, edge RoRA]

Enumerates every FK column across all tables (db tables + task tables), in
PARQUET COLUMN ORDER, assigning a local role index >= 1 (0 reserved for
"no edge"). This order MUST match how rustler/src/pre.rs builds f2p_nbr_idxs
(it iterates df columns in schema order and keeps the ones that are FKs), so
that the j-th non-null f2p slot of a node corresponds to the j-th FK column of
its table in this same order.

Output: scratch/pre/<db>/fk_roles.json
  { "<fk_col> of <table>": [local_role_idx, parent_table], ... }

The string key "<fk_col> of <table>" is identical to the col-name template used
in text.json, so gen_fk_role_embs.py can embed it into the same MiniLM space as
col_name_values.

Per-DB hacks mirrored from pre.rs:
  - rel-avito: FK col "UserID" -> parent "UserInfo", "AdID" -> parent "AdsInfo".

Usage:
  HOME=/home/<you> pixi run python scripts/gen_fk_roles.py [db ...]
"""
import json
import os
import sys
from glob import glob
from pathlib import Path

import polars as pl

SCRATCH = os.environ.get("FORTE_SCRATCH_DIR", os.path.join(os.environ["HOME"], "scratch"))
DBS = ["rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"]

# expected role counts from the handoff (sanity signal, not enforced)
EXPECTED = {"rel-amazon": 14, "rel-avito": 18, "rel-event": 13, "rel-f1": 22,
            "rel-hm": 7, "rel-stack": 20, "rel-trial": 26}


def _avito_rename(db_name: str, fk_col: str, parent_table: str) -> str:
    if db_name == "rel-avito":
        if fk_col == "UserID":
            return "UserInfo"
        if fk_col == "AdID":
            return "AdsInfo"
    return parent_table


def gen_fk_roles(db_name: str) -> None:
    db_path = f"{SCRATCH}/relbench/{db_name}"
    pre_path = f"{SCRATCH}/pre/{db_name}"

    parquet_files = sorted(glob(f"{db_path}/db/*.parquet")) + sorted(
        glob(f"{db_path}/tasks/*/*.parquet")
    )

    fk_roles: dict[str, list] = {}
    role_idx = 1
    processed: set[str] = set()
    for pq_path in parquet_files:
        p = Path(pq_path)
        table_name = p.stem if "db" in p.parts else p.parent.name
        if table_name in processed:
            continue
        processed.add(table_name)
        try:
            meta = pl.read_parquet_metadata(pq_path)
            schema = pl.read_parquet_schema(pq_path)  # ordered {col: dtype}
        except Exception as e:
            print(f"  warn: read failed for {pq_path}: {e}")
            continue
        fkey_cols: dict = json.loads(meta.get("fkey_col_to_pkey_table", "{}"))
        if not fkey_cols:
            continue
        # iterate columns in parquet schema order; keep the FK ones (matches pre.rs)
        for col_name in schema.keys():
            if col_name not in fkey_cols:
                continue
            parent_table = _avito_rename(db_name, col_name, fkey_cols[col_name])
            key = f"{col_name} of {table_name}"
            fk_roles[key] = [role_idx, parent_table]
            role_idx += 1

    out_path = f"{pre_path}/fk_roles.json"
    with open(out_path, "w") as f:
        json.dump(fk_roles, f, indent=0)
    n = len(fk_roles)
    exp = EXPECTED.get(db_name)
    flag = "" if exp is None else (f"  (handoff expects {exp})" if n != exp else "  (matches handoff)")
    print(f"  {db_name}: {n} roles -> {out_path}{flag}")


if __name__ == "__main__":
    dbs = sys.argv[1:] or DBS
    print(f"[gen_fk_roles] SCRATCH={SCRATCH}")
    for db in dbs:
        gen_fk_roles(db)
