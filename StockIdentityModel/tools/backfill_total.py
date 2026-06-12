"""Offline backfill of the stationary-loss indicator for any run, past or future.

Replays the EMA-path normalizers in float64 from the raws logged in
train_log.jsonl and emits, per step, both scalars:

  total_ema    — the historical (live EMA + kappa-floor) total, replayed; used
                 as the fail-loud validation target against the logged `total`
  total_frozen — the stationary total under norm_freeze_step = T: identical to
                 total_ema through step T, constant denominators thereafter

This is how legacy and flag-off runs (r1..r14, ...) get the indicator without
touching training. Replay fidelity measured at <= 2e-7 relative error against
the logged totals (float32 logging quantization); the tool ABORTS if the error
exceeds 1e-4 — a loud failure means the log does not match the run's config
(e.g. spliced objectives, edited logs, or a code drift this tool predates).

    python -m StockIdentityModel.tools.backfill_total \
        --run-dir StockIdentityModel/runs/r11 [--freeze-step 1000]

Output: {run_dir}/total_frozen.jsonl, lines {"step", "total_frozen", "total_ema"}.
Resume-duplicated steps in the input are deduped keep-last (the r14 precedent).
Read the indicator as centered 250-step rolling medians (per-step draw noise
sd ~0.15); levels compare within a run, not across runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for direct execution

from StockIdentityModel.config import Config
from StockIdentityModel.losses import _fixed_scale, _lambda_of


def backfill(run_dir: str | Path, freeze_step: int = 1000, abort_rel_err: float = 1e-4) -> Path:
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.json")  # legacy configs: absent fields -> defaults
    if cfg.norm_freeze_step > 0:
        # the run itself trained on the frozen path: its log already IS the
        # indicator; replay with the run's own T and validate against the
        # frozen replay (validating vs the EMA replay would abort post-T)
        if freeze_step != cfg.norm_freeze_step:
            print(f"[backfill] run trained with norm_freeze_step={cfg.norm_freeze_step}; "
                  f"using it (requested --freeze-step {freeze_step} ignored)")
        freeze_step = cfg.norm_freeze_step
    validate_frozen = cfg.norm_freeze_step > 0

    # keep-last dedup of step records (crash-resume replays duplicate steps)
    recs: dict[int, dict] = {}
    for line in open(run_dir / "train_log.jsonl"):
        r = json.loads(line)
        if "total" in r and "raw" in r:
            recs[r["step"]] = r
    if not recs:
        raise SystemExit(f"no step records in {run_dir}/train_log.jsonl")

    kappa, eps, beta = cfg.kappa_floor, cfg.eps, cfg.beta
    ema: dict[str, float] = {}
    first: dict[str, float] = {}
    frozen: dict[str, float] = {}
    max_err = 0.0
    err_step = None
    out_path = run_dir / "total_frozen.jsonl"
    with open(out_path, "w") as out:
        for step in sorted(recs):
            raw = recs[step]["raw"]
            t_ema = 0.0
            t_frz = 0.0
            for name, r in raw.items():
                if r is None:
                    continue
                lam = _lambda_of(cfg, name)
                fs = _fixed_scale(cfg, name)
                if fs is not None:  # xsep/psep/util/anc: constant in both modes
                    t_ema += lam * r / fs
                    t_frz += lam * r / fs
                    continue
                # EMA path (sc/tc/syn) — mirror EmaNormalizer.normalize exactly
                prev = ema.get(name, r)
                if name not in first:
                    first[name] = r
                denom_hist = max(prev, kappa * first[name]) + eps
                t_ema += lam * r / denom_hist
                if step > freeze_step:
                    if name not in frozen:  # post-freeze debut: freeze at first-seen value
                        frozen[name] = max(r, kappa * r) + eps
                    t_frz += lam * r / frozen[name]
                else:
                    t_frz += lam * r / denom_hist
                # historical EMA update (use-then-update), always — the frozen
                # constants are snapshotted from it at the boundary below
                ema[name] = r if name not in ema else beta * ema[name] + (1 - beta) * r
                first.setdefault(name, r)
            if step == freeze_step:  # snapshot AFTER folding step T (use-then-update)
                frozen = {k: max(v, kappa * first[k]) + eps for k, v in ema.items()}
            logged = recs[step]["total"]
            replayed = t_frz if validate_frozen else t_ema
            rel = abs(replayed - logged) / max(abs(logged), 1e-12)
            if rel > max_err:
                max_err, err_step = rel, step
            if rel > abort_rel_err:
                raise SystemExit(
                    f"ABORT: replayed total diverges from logged total at step {step} "
                    f"(rel err {rel:.3e} > {abort_rel_err:.0e}) — the log does not match "
                    "this config/code (spliced objectives, edited log, or code drift)"
                )
            out.write(json.dumps({"step": step, "total_frozen": round(t_frz, 8), "total_ema": round(t_ema, 8)}) + "\n")
    print(f"[backfill] {out_path}: {len(recs)} steps, freeze_step={freeze_step}, "
          f"max replay rel err {max_err:.2e} @ step {err_step} (abort threshold {abort_rel_err:.0e})")
    if frozen:
        print(f"[backfill] frozen denominators: {json.dumps({k: round(v, 6) for k, v in sorted(frozen.items())})}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Backfill the stationary-loss indicator from a run's train_log")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--freeze-step", type=int, default=1000)
    args = ap.parse_args()
    backfill(args.run_dir, args.freeze_step)


if __name__ == "__main__":
    main()
