#!/usr/bin/env python
"""FORTE leave-one-database-out driver (edge RoRA + col-kind + CLP + PSP).

Pretrains on every database except the held-out one (rel-event stays in the
pretraining pool, as in RT, and is never evaluated), then zero-shot evaluates on
the forecast tasks of the held-out DB.

Example (1 GPU):
  pixi run python scripts/train.py --leaveout rel-f1

A full leave-one-out fleet is launched via run_train.sh.
"""
import os

os.environ.setdefault("WANDB_MODE", "disabled")

import strictfire

from forte.main import main
from forte.tasks import all_tasks, forecast_tasks

# Per-DB best HP from the sensitivity grids (handoff Section 12), keyed by held-out DB.
BEST_HP = {
    "rel-f1":     dict(clp_weight=0.5, psp_max_weight=0.3, psp_schedule="linear"),
    "rel-trial":  dict(clp_weight=0.1, psp_max_weight=0.3, psp_schedule="step"),
    "rel-stack":  dict(clp_weight=0.1, psp_max_weight=0.1, psp_schedule="linear"),
    "rel-avito":  dict(clp_weight=0.1, psp_max_weight=0.1, psp_schedule="linear"),
    "rel-hm":     dict(clp_weight=0.1, psp_max_weight=0.1, psp_schedule="linear"),
    "rel-amazon": dict(clp_weight=0.1, psp_max_weight=0.5, psp_schedule="step"),
}


def run(
    leaveout: str,
    num_blocks: int = 6,
    max_steps: int = 30_000,
    batch_size: int = 32,
    seq_len: int = 1024,
    num_workers: int = 4,
    eval_freq: int = 1000,
    eval_pow2: bool = True,
    max_eval_steps: int = 40,
    # FORTE HP (default to per-DB best; override on the CLI)
    clp_weight: float = None,
    psp_max_weight: float = None,
    psp_schedule: str = None,
    psp_perturb_roles: bool = True,
    use_clp: bool = True,
    use_psp: bool = True,
    use_edge_roles: bool = True,
    compile_: bool = True,
    seed: int = 0,
    tag: str = "full",
    dry_run: bool = False,
):
    assert leaveout in {t[0] for t in all_tasks}, f"unknown leaveout {leaveout}"
    hp = BEST_HP.get(leaveout, dict(clp_weight=0.5, psp_max_weight=0.3, psp_schedule="linear"))
    clp_weight = hp["clp_weight"] if clp_weight is None else clp_weight
    psp_max_weight = hp["psp_max_weight"] if psp_max_weight is None else psp_max_weight
    psp_schedule = hp["psp_schedule"] if psp_schedule is None else psp_schedule

    # Leave-one-database-out (as in RT): pretrain on all tasks of every database
    # except the held-out one -- this keeps rel-event in the pretraining pool;
    # it is never an evaluation target. (musicbrainz is not in the task list.)
    train_tasks = [t for t in all_tasks if t[0] != leaveout]
    eval_tasks = [t for t in forecast_tasks if t[0] == leaveout]

    here = os.path.dirname(__file__)
    short = leaveout.replace("rel-", "")
    name = f"forte_edge_{short}_{tag}_b{num_blocks}_s{seed}"
    log_dir = os.path.join(here, "..", "logs")
    ckpt_dir = os.path.join(here, "..", "ckpts", name)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(f"[FORTE edge] leave-one-out = {leaveout}")
    print(f"  train tasks : {len(train_tasks)} (DBs: {sorted({t[0] for t in train_tasks})})")
    print(f"  eval tasks  : {len(eval_tasks)} -> {[t[1] for t in eval_tasks]}")
    print(f"  config      : blocks={num_blocks} steps={max_steps} bs={batch_size} seq={seq_len}")
    print(f"  FORTE HP    : clp_weight={clp_weight} psp_max_weight={psp_max_weight} "
          f"psp_schedule={psp_schedule} psp_perturb_roles={psp_perturb_roles}")
    print(f"  ckpt_dir    : {ckpt_dir}")
    print("=" * 70)
    if dry_run:
        print("[dry_run] not training.")
        return

    main(
        project="forte_edge",
        eval_splits=["val", "test"],
        eval_freq=eval_freq,
        eval_pow2=eval_pow2,
        max_eval_steps=max_eval_steps,
        load_ckpt_path=None,
        save_ckpt_dir=ckpt_dir,
        compile_=compile_,
        seed=seed,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        batch_size=batch_size,
        num_workers=num_workers,
        max_bfs_width=256,
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=max_steps,
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=seq_len,
        num_blocks=num_blocks,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        # FORTE
        use_clp=use_clp,
        use_psp=use_psp,
        use_edge_roles=True,
        clp_weight=clp_weight,
        psp_max_weight=psp_max_weight,
        psp_schedule=psp_schedule,
        psp_perturb_roles=psp_perturb_roles,
    )


if __name__ == "__main__":
    strictfire.StrictFire(run)
