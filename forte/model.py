import json
import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from ml_dtypes import bfloat16
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint as grad_checkpoint

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)

# structural column kinds (must match rt/data.py)
COL_KIND_FEATURE = 0
COL_KIND_PK = 1
COL_KIND_FK = 2
COL_KIND_LABEL = 3
COL_KIND_TIMESTAMP = 4
N_COL_KINDS = 5

# RoRA direction ids
DIR_SAME = 0
DIR_F2P = 1
DIR_P2F = 2


class MaskedAttention(nn.Module):
    """Dense masked attention. Uses flex_attention with a BlockMask on CUDA;
    falls back to a manual masked SDPA when given a dense bool mask (CPU/eager)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.num_heads)

        if mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        elif isinstance(mask, torch.Tensor):
            # dense bool fallback (B,S,S): manual masked attention
            scale = 1.0 / math.sqrt(q.shape[-1])
            a = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
            a = a.masked_fill(~mask[:, None], float("-inf"))
            a = torch.softmax(a, dim=-1)
            a = torch.nan_to_num(a, 0.0)
            x = torch.einsum("bhqk,bhkd->bhqd", a, v.float()).to(v.dtype)
        else:
            x = flex_attention(q, k, v, block_mask=mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        return self.wo(x)


class RoRA(nn.Module):
    """Role-conditioned Relational Attention (edge level).  [handoff Section 8]

    Cross-row attention with a per-head bias conditioned on the SPECIFIC FK
    column connecting two rows. Bias is factored to avoid an SxSxR tensor:
        P1[b,h,r,k] = (W_role([C[r] || D_f2p]) reshaped to heads) . K[b,h,k]
        rs1[b,h,q,k] = P1[b,h, E[q,k], k]       (pure gather over the role dim)
    and analogously rs2 for P->F, plus a single FK-independent same-row bias rs0.
    Manual float32 attention (Triton bf16 flex backward bug, handoff Gotcha #3).
    """

    def __init__(self, d_model, num_heads, d_schema=384, d_dir=32, use_edge_roles=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_schema = d_schema
        self.use_edge_roles = use_edge_roles
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dir_emb = nn.Embedding(3, d_dir)                       # same / f2p / p2f
        self.same_row = nn.Parameter(torch.zeros(d_schema))         # sigma
        self.w_role = nn.Linear(d_schema + d_dir, d_model, bias=False)

    def forward(self, x, C, E, same_node, kv_in_f2p, q_in_f2p, attn_mask):
        B, S, _ = x.shape
        H, dh = self.num_heads, self.head_dim
        scale = 1.0 / math.sqrt(dh)

        q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=H).float()
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=H).float()
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=H).float()

        # content attention logits
        a = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale          # (B,H,S,S)

        # everything in float32 (manual fp32 RoRA): the model is cast to bf16.
        W_role = self.w_role.weight.float()                       # (d_model, d_schema+d_dir)
        d_same = self.dir_emb.weight[DIR_SAME].float()             # (d_dir,)
        d_f2p = self.dir_emb.weight[DIR_F2P].float()
        d_p2f = self.dir_emb.weight[DIR_P2F].float()

        # FK-independent per-direction bias (uses the learned same-row vector sigma) -> (B,H,S)
        def dir_bias(dir_vec):
            s = F.linear(torch.cat([self.same_row.float(), dir_vec], dim=-1), W_role).view(H, dh)
            return torch.einsum("hd,bhkd->bhk", s, k) * scale
        rs0 = dir_bias(d_same)                                     # same-row bias (B,H,S)

        if self.use_edge_roles:
            # edge-level: per-FK-column bias gathered from the frozen role table C by E
            C = C.float()                                          # (R+1, d_schema)
            R1 = C.shape[0]

            def proj_K(dir_vec):
                inp = torch.cat([C, dir_vec.expand(R1, -1)], dim=-1)
                P = F.linear(inp, W_role).view(R1, H, dh)         # (R+1, H, dh)
                return torch.einsum("rhd,bhkd->bhrk", P, k) * scale  # (B,H,R+1,S)

            E_exp = E[:, None, :, :].expand(B, H, S, S)            # (B,H,S,S) int64
            rs1 = torch.gather(proj_K(d_f2p), 2, E_exp)           # (B,H,S,S)
            rs2 = torch.gather(proj_K(d_p2f), 2, E_exp)
        else:
            # direction-only (FORTE-Dir): FK-independent f2p / p2f biases, broadcast over q
            rs1 = dir_bias(d_f2p)[:, :, None, :]                  # (B,H,1,S)
            rs2 = dir_bias(d_p2f)[:, :, None, :]

        m_same = same_node[:, None].float()                        # (B,1,S,S)
        m_f2p = kv_in_f2p[:, None].float()
        m_p2f = q_in_f2p[:, None].float()
        role_bias = m_same * rs0[:, :, None, :] + m_f2p * rs1 + m_p2f * rs2

        a = a + role_bias
        a = a.masked_fill(~attn_mask[:, None], float("-inf"))
        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a, 0.0)                               # all-masked rows -> 0
        out = torch.einsum("bhqk,bhkd->bhqd", a, v)
        out = rearrange(out, "b h s d -> b s (h d)").to(x.dtype)
        return self.wo(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LinkDecoder(nn.Module):
    """CLP head: project a node embedding and score links by inner product
    with a learnable temperature.  [handoff Section 9]"""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.log_temp = nn.Parameter(torch.tensor(0.07).log())


class RelationalBlock(nn.Module):
    """FORTE block: Column -> RoRA -> Full -> FFN  (pre-norm RMSNorm + residual)."""

    def __init__(self, d_model, num_heads, d_ff, d_schema, use_edge_roles=True):
        super().__init__()
        self.norms = nn.ModuleDict(
            {l: nn.RMSNorm(d_model) for l in ["col", "rora", "full", "ffn"]}
        )
        self.col_attn = MaskedAttention(d_model, num_heads)
        self.rora = RoRA(d_model, num_heads, d_schema, use_edge_roles=use_edge_roles)
        self.full_attn = MaskedAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks, C, E, same_node, kv_in_f2p, q_in_f2p, rora_mask):
        x = x + self.col_attn(self.norms["col"](x), block_masks["col"])
        x = x + self.rora(self.norms["rora"](x), C, E, same_node, kv_in_f2p, q_in_f2p, rora_mask)
        x = x + self.full_attn(self.norms["full"](x), block_masks["full"])
        x = x + self.ffn(self.norms["ffn"](x))
        return x


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, _compile=True,
    )


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_text,
        num_heads,
        d_ff,
        use_clp=True,
        use_edge_roles=True,
        grad_ckpt=False,
        clp_fk_mask_prob=0.10,
    ):
        super().__init__()
        self.use_clp = use_clp
        self.grad_ckpt = grad_ckpt
        self.clp_fk_mask_prob = clp_fk_mask_prob

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                t: nn.RMSNorm(d_model)
                for t in ["number", "text", "datetime", "col_name", "boolean"]
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        # FORTE: column-kind embedding
        self.col_kind_emb = nn.Embedding(N_COL_KINDS, d_model)

        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, num_heads, d_ff, d_schema=d_text, use_edge_roles=use_edge_roles)
             for _ in range(num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model

        # CLP head (always constructed so it is anchored in the DDP graph)
        self.link_decoder = LinkDecoder(d_model)

        # FORTE: frozen global FK-role table C, supplied from the dataset.
        # Non-persistent so checkpoints stay portable across DB sets.
        self.register_buffer("fk_role_C", torch.zeros(1, d_text), persistent=False)

    def set_fk_role_C(self, C: torch.Tensor):
        self.fk_role_C = C.to(self.fk_role_C.device).to(torch.float32)

    def forward(self, batch, skip_clp=False, C_override=None, return_x=False):
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        f2p_role_idxs = batch["f2p_role_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        sem_types = batch["sem_types"]
        col_kinds = batch["col_kinds"]
        masks = batch["masks"]
        B, S = node_idxs.shape
        device = node_idxs.device

        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])           # (B,S,S)
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        # --- edge-identity matrix E (B,S,S): role connecting q,k, symmetrized ---
        match = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]) & (
            f2p_nbr_idxs[:, :, None, :] >= 0
        )                                                                    # (B,S,S,5)
        rolej = f2p_role_idxs[:, :, None, :].to(torch.int64)                 # (B,S,1,5)
        E_f2p = torch.where(match, rolej, torch.zeros_like(rolej)).amax(-1)  # (B,S,S)
        E = torch.maximum(E_f2p, E_f2p.transpose(1, 2))                      # (B,S,S) int64

        # --- CLP: mask ~p of FK children; sever their parent edges in RoRA masks ---
        fk_mask = None
        kv_in_f2p_orig = kv_in_f2p
        if self.use_clp and self.training and not skip_clp:
            has_f2p = f2p_nbr_idxs[:, :, 0] >= 0                              # (B,S)
            draw = torch.rand(B, S, device=device)
            fk_mask = has_f2p & (draw < self.clp_fk_mask_prob)               # (B,S)
            if fk_mask.any():
                kv_in_f2p = kv_in_f2p & ~fk_mask.unsqueeze(2)
                q_in_f2p = q_in_f2p & ~fk_mask.unsqueeze(1)

        rora_mask = ((same_node | kv_in_f2p | q_in_f2p) & pad).contiguous()
        col_mask = (same_col_table & pad).contiguous()
        full_mask = pad.contiguous()

        # CUDA -> flex block masks; CPU/eager -> dense bool fallback
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            mbm = partial(_make_block_mask, batch_size=B, seq_len=S, device=device)
            block_masks = {"col": mbm(col_mask), "full": mbm(full_mask)}
        else:
            block_masks = {"col": col_mask, "full": full_mask}

        # frozen role table C (buffer), or a PSP-perturbed override (handoff Section 10)
        C = self.fk_role_C if C_override is None else C_override

        # --- cell token construction ---
        x = 0
        x = x + (
            self.norm_dict["col_name"](self.enc_dict["col_name"](batch["col_name_values"]))
            * (~is_padding)[..., None]
        )
        x = x + self.col_kind_emb(col_kinds) * (~is_padding)[..., None]      # col-kind
        is_fk = (col_kinds == COL_KIND_FK)
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((sem_types == i) & ~masks & ~is_padding & ~is_fk)[..., None]   # skip FK
            )
            x = x + (
                self.mask_embs[t]
                * ((sem_types == i) & masks & ~is_padding)[..., None]
            )

        for block in self.blocks:
            if self.grad_ckpt and self.training:
                x = grad_checkpoint(block, x, block_masks, C, E, same_node,
                                    kv_in_f2p, q_in_f2p, rora_mask, use_reentrant=False)
            else:
                x = block(x, block_masks, C, E, same_node, kv_in_f2p, q_in_f2p, rora_mask)

        x = self.norm_out(x)

        if return_x:
            return x  # encoder output, for the post-hoc link-prediction eval

        # --- MCP loss + predictions ---
        loss_mcp = x.new_zeros(())
        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}
        masks_b = masks.bool()
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)
            y = batch[f"{t}_values"]
            sem_type_mask = (sem_types == i) & masks_b
            if not sem_type_mask.any():
                loss_mcp = loss_mcp + (yhat.sum() * 0.0)
                yhat_out[t] = yhat
                continue
            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")
            loss_mcp = loss_mcp + (loss_t * sem_type_mask).sum()
            yhat_out[t] = yhat
        loss_mcp = loss_mcp / masks_b.sum().clamp(min=1)

        # --- CLP loss (InfoNCE over node links) ---
        loss_clp = self._clp_loss(x, fk_mask, kv_in_f2p_orig, same_node, is_padding, col_kinds, skip_clp)

        return loss_mcp, loss_clp, yhat_out

    def _clp_loss(self, x, fk_mask, kv_in_f2p_orig, same_node, is_padding, col_kinds, skip_clp):
        # DDP anchor: keep ALL link_decoder params (proj + log_temp) in the graph
        # every step, so find_unused_parameters=False never hangs when a batch
        # happens to sample no FK positions.
        anchor = sum(p.sum() for p in self.link_decoder.parameters()) * 0.0
        if not (self.use_clp and self.training and not skip_clp) or fk_mask is None or not fk_mask.any():
            return anchor

        B, S, D = x.shape
        not_pad = ~is_padding
        is_feat = col_kinds == COL_KIND_FEATURE
        valid = not_pad & is_feat                                            # (B,S) bool

        # node embedding = mean of feature-cell outputs over same-node cells
        # (keep in x's dtype -- the model trains in bf16; CLP has no Triton issue)
        same_valid = (same_node & valid[:, None, :]).to(x.dtype)             # (B,S,S)
        node_sum = torch.bmm(same_valid, x)                                  # (B,S,D)
        node_cnt = same_valid.sum(-1, keepdim=True).clamp(min=1.0)
        node_emb = node_sum / node_cnt                                       # (B,S,D)

        z = self.link_decoder.proj(node_emb)                                 # (B,S,D)
        tau = self.link_decoder.log_temp.exp()
        scores = torch.bmm(z, node_emb.transpose(1, 2)) / tau                # (B,S,S)

        key_valid = (~same_node & not_pad[:, None, :]).bool()                # (B,S,S)
        scores = scores.masked_fill(~key_valid, float("-inf"))
        pos_mask = kv_in_f2p_orig & key_valid                                # parents (pre-sever)
        has_pos = pos_mask.any(-1)                                           # (B,S)
        query_mask = fk_mask & has_pos
        if not query_mask.any():
            return anchor

        scores_pos = scores.masked_fill(~pos_mask, float("-inf"))
        lse_all = torch.logsumexp(scores, dim=-1)                            # (B,S)
        lse_pos = torch.logsumexp(scores_pos, dim=-1)
        per_q = lse_all - lse_pos                                            # (B,S)
        return per_q[query_mask].mean() + anchor

# Backwards-compatible alias
Forte = RelationalTransformer
