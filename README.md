# FORTE: A Relational Foundation Model with Foreign-Key-Role Awareness

**FORTE** (*FOreign key Role aware Transformer Encoder*) is a relational
foundation model: a single transformer is pretrained over the structure of
several relational databases and transfers to an **entirely unseen database with
no fine-tuning**, producing classification, regression, and foreign-key
link-prediction outputs zero-shot.

FORTE represents a database as a **graph of cells** — every value in every row is
one token — and processes it with attention that is **aware of the relational
schema**: which column a cell belongs to, what structural role that column plays
(feature, primary key, foreign key, label, timestamp), and *which specific
foreign key* connects any two rows. Making attention conditional on foreign-key
**role** is the core of FORTE, and is what lets one model generalize across
databases with different tables, columns, and domains.

Prior relational foundation models use foreign keys only to decide *which* rows
are related, not *how*. We call this limitation **foreign-key role blindness** and
show it is a structural limit: a role-blind model maps databases that differ only
in their relationships to the same representation (Prop. 2.1 in the paper). FORTE
removes it.

> **Paper:** *FORTE: A Relational Foundation Model with Foreign-Key-Role
> Awareness*, SIGMOD/PODS 2027. FORTE is built on the **Relational Transformer
> (RT)** ([paper](https://arxiv.org/abs/2510.06377),
> [code](https://github.com/snap-stanford/relational-transformer)): we reuse and
> adapt RT's cell-graph representation, data preprocessing, and breadth-first
> sampler, and add foreign-key-role-aware attention plus two relational training
> objectives on top.

---

## Key ideas

A database is sampled into a sequence of `S = 1024` cell tokens by a **temporal
breadth-first walk** over foreign-key edges from a seed row (no future leakage).
Each cell token is the sum of embeddings:

```
token = column-name embedding    (frozen MiniLM embedding of "<table>.<column>")
      + column-kind embedding     (feature | primary-key | foreign-key | label | timestamp)
      + row-position embedding
      + value embedding           (per semantic type; DROPPED for foreign-key cells)
```

A foreign-key value is a pointer — its meaning lives in the row it points to, and
its integer id does not transfer across databases — so FORTE **drops the value
embedding for foreign-key cells** and resolves the reference through role instead
(paper §3.1).

The model is a stack of **FORTE blocks** (pre-norm RMSNorm + residual):

```
FORTE block:  Column → RoRA → Full → FFN          (3 attention layers)
RT block:     Column → Feature → Neighbor → Full → FFN   (4 attention layers)
```

RoRA replaces RT's two direction-specific foreign-key layers (Feature, Neighbor)
with a **single** role-conditioned layer, so **FORTE has fewer parameters than RT
at the same depth**.

* **Column attention** — cells in the same column attend across rows.
* **RoRA — Role-conditioned Relational Attention** (paper §3.2) is the heart of
  FORTE. It adds to each attention score a bias determined by (i) the **direction**
  of the connecting foreign key (same-row / child→parent / parent→child) and
  (ii) the **identity** of the specific FK column joining the two rows. The
  identity is read from the frozen text embedding of the column's qualified name
  (e.g. `results.driverId`), so a child that references parents through `buyer_id`
  vs. `seller_id` produces a *different* bias toward each — which role-blind
  attention cannot express. Because roles live in a shared text-embedding space,
  what FORTE learns about a column transfers to a similarly named column in an
  unseen schema.
* **Full attention** — dense pass over all cells the earlier layers left unscored.

**FORTE-Dir** is a control variant that keeps FK *direction* but drops FK
*identity* (`use_edge_roles=False`); the gap between it and full FORTE isolates
the contribution of foreign-key identity (paper §3.3).

### Training

FORTE is pretrained with **Masked Cell Prediction (MCP)**, augmented with two
techniques that teach the schema-aware parameters what relationships mean:

* **CLP — Contrastive Link Prediction** (paper §4.1). A training-time InfoNCE
  objective: mask a fraction `p_wh = 0.1` of foreign-key edges and train a small
  link decoder to recover each masked parent among candidate rows. This directly
  supervises RoRA's role parameters (MCP trains them only indirectly), and the
  same decoder answers link-prediction queries at inference. Loss weight
  **λ = 1.0**.
* **PSP — Partial Schema Perturbation** (paper §4.2). An **input-embedding
  perturbation** (no loss term): with probability `p_psp`, a cell's column-name
  embedding is swapped for that of a *same-kind* column from another database in
  the batch, so a name cannot serve as a reliable label for a role. A curriculum
  keeps `p_psp = 0` for the first 1/5 of training, then ramps to **α = 0.3**
  (linear schedule) by the midpoint and holds.

The pretraining loss is:

```
L = L_MCP + λ · L_CLP        (λ = 1.0)      # PSP acts on inputs, adds no loss term
```

Both CLP and PSP are removed at inference; the link decoder is kept only for
link-prediction tasks.

### Evaluation — leave-one-database-out

Strict zero-shot protocol: **pretrain on all databases but one, evaluate on the
held-out database with no fine-tuning**, reporting **test-at-best-validation**
(per task). Classification → ROC-AUC, regression → R², foreign-key link
prediction → MRR / MAP@k.

---

## Model variants

| Model | Blocks | Attn/block | Parameters |
|-------|:---:|:---:|:---:|
| RT-6      | 6  | 4 | 11.3 M |
| **FORTE-6**  | 6  | 3 | **10.5 M** |
| RT-12     | 12 | 4 | 22.3 M |
| FORTE-12  | 12 | 3 | 20.6 M |

FORTE is smaller than RT at every depth. FORTE-6 is the strongest and smallest
model overall; it trains in **2.5–3.0 h** on one H100 (vs. 5–6 h for 12 blocks)
and is the default.

---

## Repository layout

```
forte/                     # model + training code
├── model.py               # RoRA, FORTE block, MCP + CLP heads, forward pass
├── data.py                # cell tokens, column-kinds, edge-identity E, frozen role table C
├── main.py                # training / evaluation loop (bf16, OneCycle, DDP-ready)
├── tasks.py               # benchmark task list
└── embed.py               # frozen MiniLM embedding of schema strings
rustler/src/{pre,fly,common}.rs   # Rust preprocessing (parquet → cell-graph) + temporal-BFS sampler (PyO3)
scripts/
├── download_relbench.py   # fetch the RelBench databases
├── gen_col_kinds.py       # column-kind index (feature / PK / FK / label / timestamp)
├── gen_fk_roles.py        # foreign-key role vocabulary per database
├── gen_fk_role_embs.py    # frozen role-embedding table C
├── verify_edge_alignment.py   # sanity-check edge-role reconstruction
├── train.py               # leave-one-database-out training driver
├── eval_link_pred.py      # foreign-key link recovery (MRR / Hits)
└── collect_results.py     # aggregate run logs into result tables
run_train.sh               # SLURM launcher for one leave-one-out run
```

---

## Installation

FORTE uses [pixi](https://pixi.sh) for environment management and builds a Rust
sampler via maturin.

```bash
pixi install
cd rustler && pixi run maturin develop --uv --release && cd ..
```

Requirements: Python 3.12, one CUDA GPU (the reference config fits on a single
80 GB card), and a Rust toolchain (pulled in by pixi).

---

## Data preparation

Done once per database. Six databases are evaluation targets — `rel-f1`,
`rel-trial`, `rel-stack`, `rel-avito`, `rel-hm`, `rel-amazon` — and (following RT)
`rel-event` is kept in the pretraining pool but never evaluated (it has temporal
leakage that is harmless for pretraining but fatal for evaluation). Preprocess all
seven.

```bash
# 1. download the RelBench databases
pixi run python scripts/download_relbench.py
# 2. point the data root at the download cache
mkdir -p ~/scratch && ln -s ~/.cache/relbench ~/scratch/relbench
# 3. build the cell-graph + FK adjacency (Rust), per database
cd rustler && pixi run cargo run --release -- pre rel-f1 && cd ..
# 4. embed schema strings with the frozen sentence encoder, per database
pixi run python -m forte.embed rel-f1
# 5. structural indices, per database
pixi run python scripts/gen_col_kinds.py    rel-f1
pixi run python scripts/gen_fk_roles.py     rel-f1
pixi run python scripts/gen_fk_role_embs.py rel-f1
# 6. (optional) sanity-check edge-role reconstruction
pixi run python scripts/verify_edge_alignment.py rel-f1
```

Repeat steps 3–5 for each of the seven databases (the `gen_*` scripts also accept
several databases at once).

---

## Training — reproducing the paper

Each **leave-one-database-out run is an independent single-GPU job**; the held-out
database is evaluated zero-shot throughout training. Reference configuration:
**6 blocks, `d_model=256`, 8 heads, `d_ff=1024`, S=1024, batch size 32,
30K steps**, AdamW (lr `1e-3`, weight decay `0.1`, one-cycle `pct_start=0.2`),
gradient clipping `1.0`, bf16 weights with the RoRA bias in float32, MiniLM
`all-MiniLM-L12-v2` (`d_text=384`).

The paper's main results (Tables 2–3) use a **single uniform configuration for all
databases: `clp_weight=1.0`, `psp_max_weight=0.3`, `psp_schedule=linear`,
`p_wh=0.1`.** Reproduce it with:

```bash
# one held-out database
pixi run python scripts/train.py --leaveout rel-f1 \
    --clp_weight 1.0 --psp_max_weight 0.3 --psp_schedule linear

# all six, one GPU each, in parallel
dbs=(rel-f1 rel-trial rel-stack rel-avito rel-hm rel-amazon)
for i in "${!dbs[@]}"; do
  CUDA_VISIBLE_DEVICES=$i pixi run python scripts/train.py --leaveout "${dbs[$i]}" \
      --clp_weight 1.0 --psp_max_weight 0.3 --psp_schedule linear &
done; wait
```

SLURM (one GPU per job):

```bash
for db in rel-f1 rel-trial rel-stack rel-avito rel-hm rel-amazon; do
  sbatch -p <partition> --gres=gpu:1 --mem=32G --time=12:00:00 --cpus-per-task=8 \
      --job-name=forte-${db#rel-} -o logs/%x_%j.log \
      --export=ALL,LEAVEOUT=$db,NW=4,MAXSTEPS=30000,\
"EXTRA_ARGS=--clp_weight 1.0 --psp_max_weight 0.3 --psp_schedule linear" run_train.sh
done
```

Best-on-validation checkpoints are written to `ckpts/`, per-step metrics to
`logs/`. W&B logging is off by default (`WANDB_MODE=disabled`).

### Ablations

Toggle components with flags to `scripts/train.py` (they pass through to
`forte.main.main`):

| Configuration | Flags |
|---|---|
| Architecture only (no CLP/PSP) | `--use_clp False --use_psp False` |
| + PSP only | `--use_clp False --use_psp True` |
| + CLP only | `--use_clp True  --use_psp False` |
| Full FORTE | `--use_clp True  --use_psp True` |
| **FORTE-Dir** (direction only, no FK identity) | `--use_edge_roles False` |
| Sensitivity sweeps | `--clp_weight {0.1,0.5,1.0,2.0,3.0}`, `--psp_max_weight {0.1,0.3,0.5}`, `--psp_schedule {linear,step}` |
| Depth | `--num_blocks {6,12}` |

---

## Link prediction

**Recovering existing foreign-key links** (paper §5.4, Table 8): hide a foreign
key and rank its true parent among candidate rows on the held-out database,
zero-shot, using the CLP link decoder. Evaluation is **leak-free** — the target
parent is removed from the child's forward pass and vice versa — so a high score
can only come from the role representation (RT scores at chance here).

```bash
pixi run python scripts/eval_link_pred.py --db rel-f1 \
    --ckpt ckpts/<run>/rel-f1_<task>_best.pt --n_batches 30
```

Reports Hits@1, Hits@10, and MRR.

*Forecasting future links* (RelBench recommendation, MAP@k; paper §5.4, Table 9)
uses the official RelBench recommendation tasks with optional fine-tuning; see the
paper for the protocol.

---

## Representative results (paper, FORTE-6, zero-shot, leave-one-DB-out)

**Classification (ROC-AUC×100) and regression (R²×100), means over tasks:**

| Model | Classification (mean) | Regression (mean) |
|---|:---:|:---:|
| RT-6      | 69.2 | 21.8 |
| **FORTE-6**  | **73.2** | **24.8** |
| RT-12     | 69.7 | 21.8 |
| FORTE-12  | 72.3 | 24.3 |

A FORTE variant is the best model on **every** classification and regression task
in the paper, and FORTE-6 — the smallest model — has the best means on both.

**Foreign-key link prediction (FORTE-6):**

| Task | Metric | FORTE |
|---|---|:---:|
| Recovering existing links (leak-free) | mean MRR | 34.9 |
| Forecasting future links (RelBench recommendation), zero-shot | mean MAP@k | 8.1 |
| Forecasting future links, fine-tuned | mean MAP@k | 11.8 |

RT cannot perform link prediction at all (it has no representation of *which* FK
connects two rows). See paper Tables 2–9 for full per-task numbers.

---

## Citation

```bibtex
@inproceedings{forte2027,
  title     = {FORTE: A Relational Foundation Model with Foreign-Key-Role Awareness},
  booktitle = {Proceedings of the ACM SIGMOD/PODS International Conference on Management of Data (SIGMOD/PODS '27)},
  year      = {2027}
}
```

## Acknowledgments

FORTE is built on the **Relational Transformer**
([code](https://github.com/snap-stanford/relational-transformer),
[paper](https://arxiv.org/abs/2510.06377)): we reuse and adapt its data
preprocessing and breadth-first cell-graph sampler. Databases and tasks come from
the [RelBench](https://relbench.stanford.edu) benchmark; schema strings are
embedded with the [MiniLM](https://huggingface.co/sentence-transformers) sentence
encoder.
