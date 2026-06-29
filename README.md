# FORTE: Foreign-key-role-aware Transformer Encoding for Relational Deep Learning

FORTE is a transformer that learns directly from **multi-table relational
databases** and transfers to **entirely unseen databases with no fine-tuning**.
A single pretrained model predicts over the tables, columns, and foreign-key
relationships of a database it has never seen, producing classification,
regression, and foreign-key link-prediction outputs zero-shot.

FORTE represents a database as a **graph of cells** — every value in every row
becomes one token — and processes it with attention that is **aware of the
relational schema**: which column a cell belongs to, what structural role that
column plays (feature, key, timestamp, label), and *which specific foreign key*
connects any two rows. This last piece — making attention conditional on
foreign-key role — is the core of FORTE, and is what lets one model generalize
across databases with different tables, columns, and domains.

> FORTE builds on the cell-graph representation and breadth-first sampler of the
> **Relational Transformer (RT)**
> ([paper](https://arxiv.org/abs/2510.06377),
> [code](https://github.com/snap-stanford/relational-transformer)): we use and
> adapt RT's data-preprocessing and sampling code, and add foreign-key-role-aware
> attention together with two relational training objectives on top.

---

## Key ideas

A database is sampled into a sequence of cell tokens by a temporal breadth-first
walk over foreign-key edges (no future leakage), and each cell token is

```
token = column-name embedding   (frozen MiniLM embedding of "<column> of <table>")
      + column-kind embedding    (feature / primary-key / foreign-key / label / timestamp)
      + value embedding          (per semantic type; skipped for foreign-key cells)
```

The model is a stack of **FORTE blocks**, each of which is

```
ForteBlock = Column-Attention  →  RoRA  →  Full-Attention  →  SwiGLU FFN
             (pre-norm RMSNorm + residual around each sub-layer)
```

* **Column-Attention** lets cells in the same column attend across rows.
* **RoRA — Role-conditioned Relational Attention** is the heart of FORTE. It is a
  cross-row attention whose per-pair bias is conditioned on the **relational
  direction** (child→parent, parent→child, same row) *and on the identity of the
  specific foreign-key column that connects the two rows*. A child row that
  references parents through, say, `buyer_id` and `seller_id` produces a
  *different* bias toward each parent — something attention that is blind to
  foreign-key identity cannot express. Role identities live in the same frozen
  text-embedding space as column names, so they transfer to unseen schemas.
* **Full-Attention** is a dense pass over all cells in the sequence.

Generalization to unseen schemas comes from embedding every schema string
(column names, foreign-key roles) with a **frozen sentence encoder**: a
never-before-seen column such as `"horsepower"` still has a meaningful vector,
so the model can reason about it without ever having trained on it.

### Training

FORTE is pretrained by **Masked Cell Prediction (MCP)** — mask a target cell,
predict its value — augmented with two objectives that teach the schema-aware
parameters what relationships mean:

* **CLP — Contrastive Link Prediction.** A training-time InfoNCE objective that
  masks a fraction of foreign-key edges and asks the model to recover which
  parent row a child links to. This directly supervises the role representation.
* **PSP — Partial Schema Perturbation.** A curriculum augmentation that randomly
  swaps column-name (and foreign-key role) embeddings during training, so the
  model relies on *relational structure* rather than memorizing schema strings.

```
L = L_MCP  +  λ · L_CLP  +  w(t) · L_PSP
```

### Evaluation — leave-one-database-out

FORTE is evaluated under a strict zero-shot protocol: **pretrain on all
databases but one, then evaluate on the held-out database with no fine-tuning**.
Classification is scored by ROC-AUC, regression by R², and foreign-key link
prediction by MRR / Hits@k.

---

## Repository layout

```
forte/                     # the model + training code
├── model.py               # FORTE model: RoRA, ForteBlock, CLP, MCP, forward
├── data.py                # cell tokens + column-kinds + edge roles + frozen role table
├── main.py                # training / evaluation loop (bf16, OneCycle, DDP-ready)
├── tasks.py               # the benchmark task list
└── embed.py               # MiniLM text embedding of schema strings
rustler/                   # high-performance Rust preprocessing + on-the-fly sampler (PyO3)
└── src/{pre,fly,common}.rs # parquet → cell-graph, then temporal-BFS → flat token tensors
scripts/
├── download_relbench.py   # fetch the benchmark databases
├── gen_col_kinds.py        # column-kind index (feature / PK / FK / timestamp)
├── gen_fk_roles.py         # foreign-key role vocabulary per database
├── gen_fk_role_embs.py     # frozen role-embedding table
├── verify_edge_alignment.py# validates the edge-role reconstruction
├── train.py                # leave-one-database-out training driver
├── eval_link_pred.py       # foreign-key link-prediction evaluation
└── collect_results.py      # aggregate run logs into result tables
run_train.sh                # SLURM launcher for one leave-one-out run
```

---

## Installation

FORTE uses [pixi](https://pixi.sh) for environment management and builds a Rust
sampler via maturin.

```bash
pixi install
# compile and install the Rust data sampler
cd rustler && pixi run maturin develop --uv --release && cd ..
```

Requirements: Python 3.12, a CUDA GPU (the reference configuration fits
comfortably on one 80 GB card), and a Rust toolchain (pulled in by pixi).

---

## Data preparation

Done once per database. Six databases are evaluation targets — `rel-f1`,
`rel-trial`, `rel-stack`, `rel-avito`, `rel-hm`, `rel-amazon` — and (following RT)
`rel-event` is additionally kept in the pretraining pool but is never evaluated.
Preprocess all seven.

```bash
# 1. download the benchmark databases (RelBench)
pixi run python scripts/download_relbench.py

# 2. point the data root at the download cache
mkdir -p ~/scratch && ln -s ~/.cache/relbench ~/scratch/relbench

# 3. build the cell-graph + foreign-key adjacency (Rust), per database
cd rustler && pixi run cargo run --release -- pre rel-f1 && cd ..

# 4. embed schema/text strings with the frozen sentence encoder, per database
pixi run python -m forte.embed rel-f1

# 5. structural indices used by FORTE, per database
pixi run python scripts/gen_col_kinds.py     rel-f1   # column kinds
pixi run python scripts/gen_fk_roles.py      rel-f1   # foreign-key role vocabulary
pixi run python scripts/gen_fk_role_embs.py  rel-f1   # frozen role-embedding table

# 6. (optional) sanity-check the edge-role reconstruction
pixi run python scripts/verify_edge_alignment.py rel-f1
```

Repeat steps 3–5 for each database. The `gen_*` scripts also accept several
databases at once, or all of them when called with no argument.

---

## Training

Each **leave-one-database-out run is an independent single-GPU job** — the
held-out database is evaluated zero-shot throughout training. The reference
configuration is **6 blocks, `d_model=256`, 8 heads, `d_ff=1024`, sequence length
1024, batch size 32, 30K steps** (~10.5M parameters), and takes about 3 hours on
one H100.

**Single GPU** — train one held-out database at a time:
```bash
pixi run python scripts/train.py --leaveout rel-f1
```

**Multi-GPU machine (e.g. 8 GPUs)** — the leave-one-out runs are independent, so
launch one per GPU in parallel:
```bash
dbs=(rel-f1 rel-trial rel-stack rel-avito rel-hm rel-amazon)
for i in "${!dbs[@]}"; do
  CUDA_VISIBLE_DEVICES=$i pixi run python scripts/train.py --leaveout "${dbs[$i]}" &
done
wait
```

**SLURM cluster** — one GPU per job:
```bash
for db in rel-f1 rel-trial rel-stack rel-avito rel-hm rel-amazon; do
  sbatch -p <partition> --gres=gpu:1 --mem=32G --time=12:00:00 --cpus-per-task=8 \
      --job-name=forte-${db#rel-} -o logs/%x_%j.log \
      --export=ALL,LEAVEOUT=$db,NW=4,MAXSTEPS=30000 run_train.sh
done
```

Useful flags (`scripts/train.py`): `--num-blocks`, `--max-steps`,
`--clp-weight λ`, `--psp-max-weight α`, `--psp-schedule {linear,step}`,
`--use-clp / --use-psp` (toggle components for ablations), `--tag` (names the
checkpoint directory). Best-on-validation checkpoints are written to `ckpts/`,
per-step metrics to `logs/`. Logging uses Weights & Biases; disable it with
`pixi run wandb disabled`.

---

## Link prediction

Foreign-key link prediction reuses FORTE's CLP link decoder to rank candidate
parent rows for a child, on the held-out database, zero-shot:

```bash
pixi run python scripts/eval_link_pred.py --db rel-f1 \
    --ckpt ckpts/<run>/rel-f1_<task>_best.pt --n_batches 30
```

Reports Hits@1, Hits@10, and MRR.

---

## Representative zero-shot results

Leave-one-database-out, test split, 6-block reference configuration. FORTE
transfers across schemas and domains with no fine-tuning. Gains over
schema-agnostic baselines are largest on databases with the densest foreign-key
structure (e.g. `rel-avito`), consistent with role-awareness mattering most
where a row participates in many relationships.

| Database  | Classification (ROC-AUC) | Regression (R²) |
|-----------|:---:|:---:|
| rel-f1     | 0.79 – 0.85 | 0.45 |
| rel-trial  | 0.62        | 0.01 – 0.16 |
| rel-stack  | 0.85 – 0.89 | 0.21 |
| rel-avito  | 0.50 – 0.59 | 0.05 |
| rel-hm     | 0.66        | 0.09 |
| rel-amazon | 0.64 – 0.70 | 0.03 – 0.24 |

---

## Acknowledgments

FORTE is built on the **Relational Transformer**
([code](https://github.com/snap-stanford/relational-transformer),
[paper](https://arxiv.org/abs/2510.06377)): we use and adapt its data
preprocessing and breadth-first cell-graph sampler. The databases and tasks come
from the [RelBench](https://relbench.stanford.edu) benchmark, and schema strings
are embedded with the [MiniLM](https://huggingface.co/sentence-transformers)
sentence encoder.
