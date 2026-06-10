"""Artifact export: encoder weights + configuration, context-fetch recipe,
and the canonical embedding table recomputed through the inference recipe.

    python -m StockIdentityModel.export --checkpoint StockIdentityModel/runs/r1/best.pt \
        --out StockIdentityModel/artifacts/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config, REPO_ROOT
from .data import StockData
from .evaluate import window_embeddings
from .model import IdentityEncoder


def export_artifact(checkpoint: str | Path, out_dir: str | Path, device: str | None = None) -> Path:
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    if device is not None:
        cfg.device = device
    torch.set_num_threads(cfg.num_threads)
    model = IdentityEncoder(cfg)
    model.load_state_dict(ck["model"])
    model.to(torch.device(cfg.resolved_device()))
    model.eval()

    ds = StockData(cfg)  # deterministic rebuild: same holdout, same clip thresholds
    meta = ds.summary()
    ck_meta = ck.get("data_meta")
    if ck_meta:
        # data identity: the rebuilt dataset must match the checkpoint's snapshot,
        # else the recipe/table would not reflect what the weights were trained on
        bad = sorted(k for k in set(ck_meta) | set(meta) if ck_meta.get(k) != meta.get(k))
        if bad:
            detail = "\n".join(f"  {k}: checkpoint={ck_meta.get(k)!r} rebuild={meta.get(k)!r}" for k in bad)
            raise ValueError(
                "refusing to export: the dataset no longer matches the checkpoint's data_meta "
                "(raw data changed since training):\n" + detail
            )
    out = REPO_ROOT / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. weights + configuration (saved on CPU: the artifact is machine-portable)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "weights.pt")
    config_doc = {
        "config": ck["config"],
        "trained_steps": ck["step"],
        "normalization": {
            "features": "log O/C_prev, log H/C_prev, log L/C_prev, log C/C_prev, log V/median_w(V); day 0 base = own open",
            "clip_lo": ds.clip_lo,
            "clip_hi": ds.clip_hi,
            "calendar": cfg.calendar,
            # halt-tolerant markets: carried/halted days carry zero price returns and
            # volume sentinel 0.0, with the volume median over traded days only; the
            # Embedder reads the operative values from `config`, this block documents
            **(
                {
                    "halt_markets": sorted(cfg.halt_markets),
                    "halt_minfrac": cfg.halt_minfrac,
                    "max_ffill_days": cfg.max_ffill_days,
                }
                if cfg.halt_markets
                else {}
            ),
        },
        "universe": [ds.tickers[i] for i in ds.train_idx],
        "holdout": meta["holdout"],
        "data_meta": {k: v for k, v in meta.items() if k != "holdout"},
    }
    (out / "config.json").write_text(json.dumps(config_doc, indent=2))

    # 2. context-fetch recipe: the last K_inf windows and their context tickers —
    # per market in cross-market mode (each market's windows live on its own calendar)
    def _market_windows(mkt) -> list[dict]:
        return [
            {
                "window_id": int(w),
                "start": mkt.window_dates(w)[0],
                "end": mkt.window_dates(w)[1],
                "dates": mkt.window_date_list(w),
                "context_tickers": [ds.tickers[i] for i in mkt.train_universe(w)],
            }
            for w in mkt.usable_windows[-cfg.K_inf :]
        ]

    recipe = {"K_inf": cfg.K_inf, "windows": _market_windows(ds.us)}
    if cfg.cross_market:
        recipe["markets"] = {m.name: {"windows": _market_windows(m)} for m in ds.secondary}
    (out / "context_recipe.json").write_text(json.dumps(recipe, indent=2))

    # 3. canonical embedding table, recomputed through the inference recipe
    # (deterministic full view, single group, averaged over the last K_inf windows)
    # rather than copied from training anchors, so training tickers and unseen
    # tickers land in directly comparable coordinates. Cross-market: every
    # market's tickers, each embedded on its own calendar, in one table.
    acc: dict[int, list[np.ndarray]] = {}
    for mkt in ds.markets.values():
        for w in mkt.usable_windows[-cfg.K_inf :]:
            uni, Z, _ = window_embeddings(model, ds, w, mkt)
            for k, t in enumerate(uni):
                acc.setdefault(int(t), []).append(Z[k])
    rows = []
    for t in sorted(acc.keys()):
        z = np.stack(acc[t]).mean(0)
        rows.append(ds.tickers[t] + "," + ",".join(f"{x:.6f}" for x in z))
    header = "ticker," + ",".join(f"z{i}" for i in range(cfg.D))
    (out / "embedding_table.csv").write_text("\n".join([header] + rows) + "\n")

    print(f"[export] {out}: weights.pt, config.json, context_recipe.json, embedding_table.csv ({len(rows)} tickers)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Export the Stock Identity artifact")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help='"auto", "cpu", "cuda", or "cuda:N"')
    args = ap.parse_args()
    export_artifact(args.checkpoint, args.out, device=args.device)


if __name__ == "__main__":
    main()
