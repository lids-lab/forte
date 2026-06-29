#!/usr/bin/env python
"""Generate col_kind_index.json for each preprocessed DB.   [FORTE Stage A]

Maps col_idx (as string) -> structural kind integer:
  0 = feature   (regular column)
  1 = pk        (primary key)
  2 = fk        (foreign key)
  4 = timestamp (time column)
  (kind 3 = label is assigned at runtime from the target column, in data.py)

Reads parquet KV metadata (pkey_col, fkey_col_to_pkey_table, time_col) and the
preprocessed column_index.json ("<col> of <table>" -> idx).

Usage:
  HOME=/home/<you> pixi run python scripts/gen_col_kinds.py [db ...]
  (no args -> all 7 forecast DBs)
"""
import json
import os
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import polars as pl

SCRATCH = os.environ.get("FORTE_SCRATCH_DIR", os.path.join(os.environ["HOME"], "scratch"))
DBS = ["rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"]

KIND_FEATURE, KIND_PK, KIND_FK, KIND_TIMESTAMP = 0, 1, 2, 4
KIND_NAMES = {0: "feature", 1: "pk", 2: "fk", 3: "label", 4: "timestamp"}


def gen_col_kinds(db_name: str) -> None:
    db_path = f"{SCRATCH}/relbench/{db_name}"
    pre_path = f"{SCRATCH}/pre/{db_name}"
    col_index_path = f"{pre_path}/column_index.json"
    if not os.path.exists(col_index_path):
        print(f"  skip {db_name}: {col_index_path} not found")
        return

    col_index: dict[str, int] = json.load(open(col_index_path))

    schema_kinds: dict[tuple[str, str], int] = {}
    processed: set[str] = set()
    parquet_files = sorted(glob(f"{db_path}/db/*.parquet")) + sorted(
        glob(f"{db_path}/tasks/*/*.parquet")
    )
    for pq_path in parquet_files:
        p = Path(pq_path)
        table_name = p.stem if "db" in p.parts else p.parent.name
        if table_name in processed:
            continue
        processed.add(table_name)
        try:
            meta = pl.read_parquet_metadata(pq_path)
        except Exception as e:
            print(f"  warn: metadata read failed for {pq_path}: {e}")
            continue
        pkey_col = json.loads(meta.get("pkey_col", "null"))
        fkey_cols: dict = json.loads(meta.get("fkey_col_to_pkey_table", "{}"))
        time_col = json.loads(meta.get("time_col", "null"))
        if pkey_col:
            schema_kinds[(table_name, pkey_col)] = KIND_PK
        for fk_col in fkey_cols:
            schema_kinds[(table_name, fk_col)] = KIND_FK
        if time_col:
            schema_kinds[(table_name, time_col)] = KIND_TIMESTAMP

    col_kind_index: dict[str, int] = {}
    for col_str, col_idx in col_index.items():
        parts = col_str.split(" of ", 1)
        if len(parts) != 2:
            continue
        col_name, table_name = parts
        col_kind_index[str(col_idx)] = schema_kinds.get((table_name, col_name), KIND_FEATURE)

    out_path = f"{pre_path}/col_kind_index.json"
    with open(out_path, "w") as f:
        json.dump(col_kind_index, f)
    counts = Counter(col_kind_index.values())
    summary = {KIND_NAMES[k]: v for k, v in sorted(counts.items())}
    print(f"  {db_name}: {summary} -> {out_path}")


if __name__ == "__main__":
    dbs = sys.argv[1:] or DBS
    print(f"[gen_col_kinds] SCRATCH={SCRATCH}")
    for db in dbs:
        gen_col_kinds(db)
