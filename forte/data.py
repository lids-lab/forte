import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import json
import os
from functools import cache

import ml_dtypes
import numpy as np
import torch
from rustler import Sampler
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# FORTE structural column kinds (must match rt/model.py)
# ---------------------------------------------------------------------------
COL_KIND_FEATURE = 0
COL_KIND_PK = 1
COL_KIND_FK = 2
COL_KIND_LABEL = 3
COL_KIND_TIMESTAMP = 4

# Canonical, FIXED DB order for the global FK-role registry. Using a fixed list
# (independent of which DBs a given run loads) guarantees that a role's GLOBAL
# index is identical across processes and across leave-one-out runs, and that the
# held-out DB's roles are present in C for zero-shot eval.
FORTE_DBS = [
    "rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial",
]

D_TEXT = 384


def _pre_dir(db_name: str) -> str:
    home = os.environ.get("HOME", ".")
    scratch = os.environ.get("FORTE_SCRATCH_DIR", os.path.join(home, "scratch"))
    return os.path.join(scratch, "pre", db_name)


@cache
def _load_column_index(db_name: str) -> dict:
    with open(os.path.join(_pre_dir(db_name), "column_index.json")) as f:
        return json.load(f)


def get_column_index(column_name: str, table_name: str, db_name: str) -> int:
    column_index = _load_column_index(db_name)
    target = f"{column_name} of {table_name}"
    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in column_index.json for dataset {db_name}.'
        )
    return column_index[target]


@cache
def _load_col_kind_array(db_name: str) -> np.ndarray | None:
    """col_kind_index.json -> int32 array indexed by col_idx (None if absent)."""
    path = os.path.join(_pre_dir(db_name), "col_kind_index.json")
    if not os.path.exists(path):
        return None
    raw: dict[str, int] = json.load(open(path))
    if not raw:
        return None
    arr = np.zeros(max(int(k) for k in raw) + 1, dtype=np.int32)
    for k, v in raw.items():
        arr[int(k)] = v
    return arr


@cache
def _load_table_intervals(db_name: str):
    """Return (bounds, names) so a node_idx resolves to its table name via
    searchsorted. Intervals come from table_info.json ("<table>:<type>")."""
    with open(os.path.join(_pre_dir(db_name), "table_info.json")) as f:
        table_info = json.load(f)
    spans = []
    for key, info in table_info.items():
        table = key.split(":", 1)[0]
        lo = int(info["node_idx_offset"])
        hi = lo + int(info["num_nodes"])
        spans.append((lo, hi, table))
    spans.sort()
    bounds = np.array([lo for lo, _, _ in spans], dtype=np.int64)
    his = np.array([hi for _, hi, _ in spans], dtype=np.int64)
    names = [t for _, _, t in spans]
    return bounds, his, names


def _resolve_tables(db_name: str, node_idxs: np.ndarray) -> np.ndarray:
    """Vectorized node_idx -> table-name (object array). -1 / out-of-range -> ''."""
    bounds, his, names = _load_table_intervals(db_name)
    out = np.empty(node_idxs.shape, dtype=object)
    out[...] = ""
    flat = node_idxs.reshape(-1)
    valid = flat >= 0
    pos = np.searchsorted(bounds, flat, side="right") - 1
    res = np.empty(flat.shape, dtype=object)
    res[...] = ""
    ok = valid & (pos >= 0)
    idx_ok = np.where(ok)[0]
    for i in idx_ok:
        p = pos[i]
        if flat[i] < his[p]:
            res[i] = names[p]
    return res.reshape(node_idxs.shape)


@cache
def _global_fk_role_registry():
    """Concatenate all FORTE_DBS' FK-role vocabularies into ONE global index
    space and stack their fk_role_embs.npy into a single frozen C.

    Returns dict with:
      C            : (R_total+1, 384) float32   (row 0 = zeros = "no edge")
      offset       : {db: int}  local role r (>=1) -> global r-1+offset[db]
      table_fklist : {(db, table): [(parent_table, global_role), ...]}  schema order
    """
    rows = [np.zeros((1, D_TEXT), dtype=np.float32)]  # global row 0 = no-edge
    offset: dict[str, int] = {}
    table_fklist: dict[tuple, list] = {}
    g = 1  # next free global index
    for db in FORTE_DBS:
        pre = _pre_dir(db)
        fk_roles_path = os.path.join(pre, "fk_roles.json")
        embs_path = os.path.join(pre, "fk_role_embs.npy")
        if not (os.path.exists(fk_roles_path) and os.path.exists(embs_path)):
            offset[db] = g
            continue
        fk_roles: dict[str, list] = json.load(open(fk_roles_path))
        C_db = np.load(embs_path)  # (R_db+1, 384), row 0 zeros
        R_db = C_db.shape[0] - 1
        offset[db] = g  # local r -> global (g + r - 1)
        if R_db > 0:
            rows.append(C_db[1:].astype(np.float32))
        # per-table FK list in schema order (sorted by local role idx)
        by_table: dict[str, list] = {}
        for key, (local_role, parent_table) in fk_roles.items():
            # key == "<col> of <table>"
            table = key.split(" of ", 1)[1]
            global_role = g + int(local_role) - 1
            by_table.setdefault(table, []).append((int(local_role), parent_table, global_role))
        for table, lst in by_table.items():
            lst.sort()  # by local_role -> schema order
            table_fklist[(db, table)] = [(pt, gr) for (_, pt, gr) in lst]
        g += R_db
    C = np.concatenate(rows, axis=0)
    return {"C": C, "offset": offset, "table_fklist": table_fklist}


class RelationalDataset(Dataset):
    def __init__(
        self,
        tasks,
        batch_size,
        seq_len,
        rank,
        world_size,
        max_bfs_width,
        embedding_model,
        d_text,
        seed,
    ):
        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []

        # dataset_idx -> db_name (for per-row DB resolution in __getitem__)
        self._db_by_dataset_idx: list[str] = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            if split == "train":
                split = "Train"
            elif split == "val":
                split = "Val"
            elif split == "test":
                split = "Test"

            with open(os.path.join(_pre_dir(db_name), "table_info.json")) as f:
                table_info = json.load(f)

            table_info_key = (
                f"{table_name}:Db"
                if f"{table_name}:Db" in table_info
                else f"{table_name}:{split}"
            )
            info = table_info[table_info_key]
            node_idx_offset = info["node_idx_offset"]
            num_nodes = info["num_nodes"]

            target_idx = get_column_index(target_column, table_name, db_name)
            target_column_indices.append(target_idx)
            drop_indices = [
                get_column_index(col, table_name, db_name) for col in columns_to_drop
            ]
            drop_column_indices.append(drop_indices)

            dataset_tuples.append((db_name, node_idx_offset, num_nodes))
            self._db_by_dataset_idx.append(db_name)

        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=rank,
            world_size=world_size,
            max_bfs_width=max_bfs_width,
            embedding_model=embedding_model,
            d_text=d_text,
            seed=seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
        )

        self.seq_len = seq_len
        self.d_text = d_text

        # --- FORTE: global role registry + frozen C (shared, fixed indices) ---
        reg = _global_fk_role_registry()
        self.fk_role_C = torch.from_numpy(reg["C"])           # (R_total+1, 384) f32
        self._role_offset = reg["offset"]
        self._table_fklist = reg["table_fklist"]
        self._role_cache: dict[tuple, tuple] = {}

    def __len__(self):
        return self.sampler.len_py()

    # ------------------------------------------------------------------ #
    #  FORTE edge-role reconstruction (handoff Section 6.2)
    # ------------------------------------------------------------------ #
    def _roles_for_pattern(self, db: str, table: str, parent_tables: tuple) -> tuple:
        """Greedily match a node's non-null f2p parent tables (schema order)
        against table's FK list to recover each slot's GLOBAL role.
        Cached by (db, table, parent_tables)."""
        key = (db, table, parent_tables)
        cached = self._role_cache.get(key)
        if cached is not None:
            return cached
        fklist = self._table_fklist.get((db, table), [])
        roles = []
        cursor = 0
        for P in parent_tables:
            role = 0
            while cursor < len(fklist):
                ptab, grole = fklist[cursor]
                cursor += 1
                if ptab == P:
                    role = grole
                    break
            roles.append(role)
        out = tuple(roles)
        self._role_cache[key] = out
        return out

    def _build_f2p_role_idxs(self, db, node_idxs_b, f2p_b):
        """node_idxs_b: (S,) int; f2p_b: (S,5) int -> (S,5) int32 global roles."""
        S = node_idxs_b.shape[0]
        roles = np.zeros((S, 5), dtype=np.int32)
        if db not in self._role_offset:
            return roles
        cell_tables = _resolve_tables(db, node_idxs_b)            # (S,) table of each cell's node
        parent_tables = _resolve_tables(db, f2p_b)                # (S,5) table of each parent
        # dedupe by node: all cells of the same node share f2p slots
        uniq, first_idx = np.unique(node_idxs_b, return_index=True)
        node_to_roles: dict[int, tuple] = {}
        for n, fi in zip(uniq.tolist(), first_idx.tolist()):
            if n < 0:
                continue
            T = cell_tables[fi]
            if not T:
                continue
            slots = f2p_b[fi]                                     # (5,)
            ptabs = tuple(
                parent_tables[fi, j] for j in range(5) if slots[j] >= 0
            )
            if not ptabs:
                continue
            node_to_roles[n] = self._roles_for_pattern(db, T, ptabs)
        # scatter back to all cells of each node
        for s in range(S):
            n = int(node_idxs_b[s])
            r = node_to_roles.get(n)
            if r is None:
                continue
            for j, rr in enumerate(r):
                roles[s, j] = rr
        return roles

    def __getitem__(self, batch_idx):
        tup = self.sampler.batch_py(batch_idx)
        out = dict(tup)
        for k, v in out.items():
            if k in [
                "number_values",
                "datetime_values",
                "text_values",
                "col_name_values",
                "boolean_values",
            ]:
                out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
            elif k == "true_batch_size":
                pass
            else:
                out[k] = torch.from_numpy(v)

        out["node_idxs"] = out["node_idxs"].view(-1, self.seq_len)
        out["sem_types"] = out["sem_types"].view(-1, self.seq_len)
        out["masks"] = out["masks"].view(-1, self.seq_len)
        out["is_targets"] = out["is_targets"].view(-1, self.seq_len)
        out["is_task_nodes"] = out["is_task_nodes"].view(-1, self.seq_len)
        out["is_padding"] = out["is_padding"].view(-1, self.seq_len)
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, self.seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, self.seq_len)
        out["class_value_idxs"] = out["class_value_idxs"].view(-1, self.seq_len)
        out["dataset_idxs"] = out["dataset_idxs"].view(-1, self.seq_len)

        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, self.seq_len, 5)
        out["number_values"] = out["number_values"].view(-1, self.seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, self.seq_len, 1)
        out["boolean_values"] = (
            out["boolean_values"].view(-1, self.seq_len, 1).bfloat16()
        )
        out["text_values"] = out["text_values"].view(-1, self.seq_len, self.d_text)
        out["col_name_values"] = out["col_name_values"].view(
            -1, self.seq_len, self.d_text
        )

        B, S = out["col_name_idxs"].shape
        col_name_idxs_np = out["col_name_idxs"].numpy()
        is_targets_np = out["is_targets"].numpy()
        node_idxs_np = out["node_idxs"].numpy()
        f2p_np = out["f2p_nbr_idxs"].numpy()
        dataset_idxs_np = out["dataset_idxs"].numpy()

        col_kinds = np.zeros((B, S), dtype=np.int32)
        f2p_role_idxs = np.zeros((B, S, 5), dtype=np.int32)

        for b in range(B):
            # per-row DB from the sampler's dataset tag (robust for multi-DB)
            ds_idx = int(dataset_idxs_np[b, 0])
            if ds_idx < 0 or ds_idx >= len(self._db_by_dataset_idx):
                continue
            db = self._db_by_dataset_idx[ds_idx]

            # --- col_kinds (B,S): kind per cell, target overridden to LABEL ---
            kind_arr = _load_col_kind_array(db)
            if kind_arr is not None:
                ci = np.clip(col_name_idxs_np[b], 0, len(kind_arr) - 1)
                kb = kind_arr[ci].copy()
            else:
                kb = np.zeros(S, dtype=np.int32)
            kb[is_targets_np[b]] = COL_KIND_LABEL
            col_kinds[b] = kb

            # --- f2p_role_idxs (B,S,5): global FK role per parent slot ---
            f2p_role_idxs[b] = self._build_f2p_role_idxs(db, node_idxs_np[b], f2p_np[b])

        out["col_kinds"] = torch.from_numpy(col_kinds)
        out["f2p_role_idxs"] = torch.from_numpy(f2p_role_idxs)
        return out
