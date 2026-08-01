"""Promote a finished run's consumer artifacts into the served model home.

Runs archive side by side under models/v5/runs/<name> and are never read by
consumers; run_live_signals_v5.py, run_backtest_v5.py, and the default
tools/score_checkpoints.py invocation read the promoted copies in models/v5 and
forecasts/. Promotion is the explicit, recorded step between the two:

    python3 tools/promote_run.py --run-dir models/v5/runs/r1

Copies v5_meta.json, v5_norm.json, config.json, run_manifest.json and every
member_{i}.pt into models/v5/, mirrors the run's forecast dir into forecasts/,
and stamps models/v5/promoted_from.json with the source run, nonce, git
revisions, and UTC time so the served artifacts always name the run they came
from. Refuses unfinished runs (missing finalize artifacts or member
checkpoints).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_MODEL_DIR = PROJECT_ROOT / "models" / "v5"
FORECAST_DIR = PROJECT_ROOT / "forecasts"
ARTIFACTS = ("v5_meta.json", "v5_norm.json", "config.json", "run_manifest.json")
FORECAST_SIDECARS = ("split_info.json", "oos_start_date.txt", "surface_manifest.json")


def promote(run_dir, model_dir=V5_MODEL_DIR, forecast_dir=FORECAST_DIR):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"no such run dir: {run_dir}")
    missing = [a for a in ARTIFACTS if not (run_dir / a).exists()]
    if missing:
        raise SystemExit(f"{run_dir.name}: missing {missing} -- the finalize did "
                         "not complete; refusing to promote an unfinished run")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    n_members = int(manifest["args"]["members"])
    members = [run_dir / f"member_{i}.pt" for i in range(n_members)]
    absent = [m.name for m in members if not m.exists()]
    if absent:
        raise SystemExit(f"{run_dir.name}: missing checkpoints {absent}")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for a in ARTIFACTS:
        shutil.copy2(run_dir / a, model_dir / a)
    for m in members:
        shutil.copy2(m, model_dir / m.name)

    src_fc = Path(forecast_dir) / run_dir.name
    copied_fc = 0
    if src_fc.is_dir():
        for f in sorted(src_fc.iterdir()):
            if f.suffix == ".csv" or f.name in FORECAST_SIDECARS:
                shutil.copy2(f, Path(forecast_dir) / f.name)
                copied_fc += 1
    else:
        print(f"[promote] no forecast dir at {src_fc}; skipping forecast promotion")

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        rev = "unknown"
    rel = (run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT)
           else run_dir)
    stamp = {
        "run_dir": str(rel),
        "run_nonce": manifest.get("run_nonce"),
        "git_revision_at_run": manifest.get("git_revision"),
        "git_revision_at_promotion": rev,
        "promoted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "members": n_members,
        "forecast_files": copied_fc,
    }
    (model_dir / "promoted_from.json").write_text(json.dumps(stamp, indent=2),
                                                  encoding="utf-8")
    print(f"[promote] {run_dir.name} -> {model_dir} ({n_members} members, "
          f"{copied_fc} forecast files)")
    return stamp


def main():
    ap = argparse.ArgumentParser(
        description="Promote a finished run's artifacts into models/v5")
    ap.add_argument("--run-dir", required=True,
                    help="finished run directory, e.g. models/v5/runs/r1")
    args = ap.parse_args()
    promote(args.run_dir)


if __name__ == "__main__":
    main()
