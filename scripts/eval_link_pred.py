#!/usr/bin/env python
"""FORTE link-prediction eval (option A: CLP-based, post-hoc on a checkpoint).

Measures how well the trained model's node representations + LinkDecoder recover
the true FK parent of a child row, on the held-out DB (zero-shot). For each child
node with an F->P parent present in the sampled neighborhood, rank all candidate
nodes by LinkDecoder score and check the rank of the true parent.

Metrics: Hits@1, Hits@10, MRR  (averaged over query nodes).

Usage:
  HOME=/home/<you> pixi run python scripts/eval_link_pred.py \
      --db rel-f1 --ckpt ckpts/forte_edge_f1_b6_s0/rel-f1_driver-dnf_best.pt [--n_batches 20]
"""
import os
os.environ.setdefault("WANDB_MODE", "disabled")
import torch
import strictfire

from forte.data import RelationalDataset, COL_KIND_FEATURE
from forte.model import Forte
from forte import tasks as T


def run(db, ckpt, split="test", n_batches=20, seq_len=1024, batch_size=16,
        num_blocks=6, d_model=256, num_heads=8, d_ff=1024, d_text=384):
    task = next(t for t in T.forecast_tasks if t[0] == db)
    _, table, target, drop = task
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    ds = RelationalDataset(
        tasks=[(db, table, target, split, drop)],
        batch_size=batch_size, seq_len=seq_len, rank=0, world_size=1,
        max_bfs_width=256, embedding_model="all-MiniLM-L12-v2", d_text=d_text, seed=0,
    )
    ds.sampler.shuffle_py(0)

    net = Forte(num_blocks=num_blocks, d_model=d_model, d_text=d_text,
                               num_heads=num_heads, d_ff=d_ff, use_clp=True)
    sd = torch.load(os.path.expanduser(ckpt), map_location="cpu")
    net.load_state_dict(sd, strict=True)
    net = net.to(device).to(dtype)
    net.set_fk_role_C(ds.fk_role_C)
    net.eval()

    hits1 = hits10 = mrr = 0.0
    n_q = 0
    with torch.inference_mode():
        for bi in range(min(n_batches, len(ds))):
            b = ds[bi]
            b.pop("true_batch_size", None)
            for k in b:
                if torch.is_tensor(b[k]):
                    b[k] = b[k].to(device)
                    if b[k].dtype in (torch.float16, torch.bfloat16, torch.float32) and "values" in k:
                        b[k] = b[k].to(dtype)
            x = net(b, return_x=True).float()                       # (B,S,D)
            ni = b["node_idxs"]; f2p = b["f2p_nbr_idxs"]
            pad = b["is_padding"].bool(); ck = b["col_kinds"]
            is_feat = (ck == COL_KIND_FEATURE) & ~pad
            tau = net.link_decoder.log_temp.exp().float()
            B, S, _ = x.shape
            for bb in range(B):
                nis = ni[bb]
                uniq = torch.unique(nis[(~pad[bb]) & (nis >= 0)])
                if uniq.numel() < 3:
                    continue
                idx_of = {int(u): i for i, u in enumerate(uniq.tolist())}
                # node embedding = mean of feature-cell outputs per node
                embs = []
                for u in uniq.tolist():
                    cells = (nis == u) & is_feat[bb]
                    if cells.any():
                        embs.append(x[bb][cells].mean(0))
                    else:                                            # fall back to all cells
                        embs.append(x[bb][(nis == u) & ~pad[bb]].mean(0))
                node_emb = torch.stack(embs, 0)                      # (M,D)
                z = net.link_decoder.proj(node_emb.to(dtype)).float()
                scores = (z @ node_emb.t()) / tau                   # (M,M)
                scores.fill_diagonal_(float("-inf"))
                # for each child node, rank its true parents
                for u in uniq.tolist():
                    qi = idx_of[u]
                    # parents = u's f2p neighbors that are present in this item
                    cell = (nis == u).nonzero(as_tuple=True)[0]
                    if cell.numel() == 0:
                        continue
                    pars = f2p[bb][cell[0]]
                    par_idx = [idx_of[int(p)] for p in pars.tolist() if int(p) in idx_of and int(p) != u]
                    if not par_idx:
                        continue
                    order = torch.argsort(scores[qi], descending=True)
                    ranks = {int(o): r + 1 for r, o in enumerate(order.tolist())}
                    best = min(ranks[p] for p in par_idx)
                    hits1 += float(best == 1)
                    hits10 += float(best <= 10)
                    mrr += 1.0 / best
                    n_q += 1

    n_q = max(n_q, 1)
    print(f"[link-pred] {db}/{split}  ckpt={os.path.basename(ckpt)}  queries={n_q}")
    print(f"  Hits@1 = {hits1/n_q:.4f}   Hits@10 = {hits10/n_q:.4f}   MRR = {mrr/n_q:.4f}")


if __name__ == "__main__":
    strictfire.StrictFire(run)
