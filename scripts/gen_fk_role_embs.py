#!/usr/bin/env python
"""Generate fk_role_embs.npy for each preprocessed DB.   [FORTE Stage A, edge RoRA]

For each FK role in fk_roles.json, embed its "<col> of <table>" string with the
SAME MiniLM model used for col_name_values, so the frozen role table C lives in
the same space as the schema embeddings the model already sees.

Output: scratch/pre/<db>/fk_role_embs.npy   shape (R+1, 384) float32
  row 0      = zeros            (the "no edge" role)
  row i (>=1) = MiniLM("<col> of <table>")  for the role with local index i

Usage:
  HOME=/home/<you> pixi run python scripts/gen_fk_role_embs.py [db ...]
"""
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRATCH = os.environ.get("FORTE_SCRATCH_DIR", os.path.join(os.environ["HOME"], "scratch"))
DBS = ["rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"]
EMB_MODEL = "all-MiniLM-L12-v2"
D_TEXT = 384


def gen_fk_role_embs(db_name: str, model: SentenceTransformer) -> None:
    pre_path = f"{SCRATCH}/pre/{db_name}"
    fk_roles_path = f"{pre_path}/fk_roles.json"
    if not os.path.exists(fk_roles_path):
        print(f"  skip {db_name}: {fk_roles_path} not found (run gen_fk_roles first)")
        return
    fk_roles: dict[str, list] = json.load(open(fk_roles_path))

    n_roles = max((v[0] for v in fk_roles.values()), default=0)
    # ordered role strings, indexed by local role idx (1..n_roles)
    role_str_by_idx: dict[int, str] = {v[0]: k for k, v in fk_roles.items()}
    strings = [role_str_by_idx[i] for i in range(1, n_roles + 1)]

    C = np.zeros((n_roles + 1, D_TEXT), dtype=np.float32)
    if strings:
        embs = model.encode(strings, convert_to_numpy=True, show_progress_bar=False)
        C[1:] = embs.astype(np.float32)

    out_path = f"{pre_path}/fk_role_embs.npy"
    np.save(out_path, C)
    print(f"  {db_name}: C={C.shape} (n_roles={n_roles}) -> {out_path}")


if __name__ == "__main__":
    dbs = sys.argv[1:] or DBS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gen_fk_role_embs] SCRATCH={SCRATCH} device={device}")
    model = SentenceTransformer(f"sentence-transformers/{EMB_MODEL}", device=device)
    for db in dbs:
        gen_fk_role_embs(db, model)
