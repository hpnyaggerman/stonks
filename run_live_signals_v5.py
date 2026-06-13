"""Emit live buy/no-buy signals from the trained ensemble.

Loads the v5 artifacts (feature/normalization metadata, config, and the member
checkpoints), builds the most recent window per ticker through the shared feature
builder so the inputs are bit-identical to training, runs the deep-ensemble
predictor, and writes two CSVs per run: per-ticker scores and a single buy decision.

The window builder enforces the same ``min_real_rows`` floor as training; a ticker
with too little history is reported with the ``insufficient_window`` status rather
than scored. Tickers whose data lags the requested decision date are listed so a
stale snapshot is visible instead of silently scored as fresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import features_v5 as fx
from features_v5 import HORIZON_DAYS
from v5_backbone import V5Backbone, V5Config, assert_theta_on_bin_edges, ensemble_predict
from v5.forecast import HORIZON_LABELS, build_forecast_columns

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
V5_MODEL_DIR = MODELS_DIR / "v5"
SIGNALS_DIR = PROJECT_ROOT / "signals"
STOCK_LIST_PATH = PROJECT_ROOT / "TrainingData" / "stockList.csv"


def load_artifacts(device):
    meta = json.loads((MODELS_DIR / "v5_meta.json").read_text(encoding="utf-8"))
    if meta["feature_names_hash"] != fx.feature_names_hash():
        raise ValueError("feature spec has changed since training; retrain before scoring")
    cfg = V5Config(**json.loads((V5_MODEL_DIR / "config.json").read_text(encoding="utf-8")))
    assert_theta_on_bin_edges(cfg)
    scaler = fx.RobustScaler.load(MODELS_DIR / "v5_norm.json")
    members = []
    for i in range(meta["n_members"]):
        model = V5Backbone(cfg)
        state = torch.load(V5_MODEL_DIR / f"member_{i}.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        members.append(model)
    return meta, cfg, scaler, members, meta.get("temperatures")


def load_universe(tickers_arg):
    if tickers_arg:
        return [t.upper() for t in tickers_arg]
    if STOCK_LIST_PATH.exists():
        col = pd.read_csv(STOCK_LIST_PATH, header=None).iloc[:, 0].astype(str).str.strip().str.upper()
        return [t for t in col if t and t not in {"SYMBOL", "TICKER"}]
    raise SystemExit("no universe: pass --tickers or provide TrainingData/stockList.csv")


@torch.no_grad()
def score_ticker(ticker, ohlcv, fear_greed, scaler, cfg, members, temps, as_of, device):
    """Return per-horizon prediction columns for the latest window, or a status string."""
    ff = fx.build_feature_frame(ticker, ohlcv, fear_greed)
    in_window = ff.dates <= np.datetime64(as_of)
    if not in_window.any():
        return "insufficient_window", None
    anchor = int(np.where(in_window)[0][-1])
    norm = scaler.transform(ff.features)
    win = fx.assemble_window(norm, anchor, cfg.window, cfg.min_real_rows)
    if win is None:
        return "insufficient_window", None
    x = torch.from_numpy(win[None]).to(device)
    hist, cls3, up_std, score_std = ensemble_predict(members, x, cfg, temps=temps)
    to_np = lambda t: t.cpu().numpy()
    cols = build_forecast_columns(
        dates=ff.dates[[anchor]], close=ff.close[[anchor]], sigma_hat=ff.sigma_hat[[anchor]],
        horizon_days=cfg.horizons,
        up=to_np(cls3[..., 2]), up_std=to_np(up_std), down=to_np(cls3[..., 0]),
        neutral=to_np(cls3[..., 1]), score=to_np(cls3[..., 2] - cls3[..., 0]),
        score_std=to_np(score_std), hist=to_np(hist),
        bin_width=cfg.bin_width, n_bins=cfg.n_bins)
    return "scored", {k: (v[0] if hasattr(v, "__len__") and len(v) == 1 else v) for k, v in cols.items()}


def best_horizon(row_cols, std_factor):
    """Pick the horizon maximizing P(up) - std_factor * Std(P(up))."""
    best = None
    for label in HORIZON_LABELS:
        prob = float(row_cols[f"Pred_Prob_{label}"])
        std = max(float(row_cols[f"Pred_Prob_Std_{label}"]), 1e-8)
        adj = prob - std_factor * std
        cand = {
            "horizon": label, "pred_prob": prob, "pred_std": std, "adj_prob": adj,
            "pred_down": float(row_cols[f"Pred_Prob_Down_{label}"]),
            "pred_neutral": float(row_cols[f"Pred_Prob_Neutral_{label}"]),
            "score": float(row_cols[f"Score_{label}"]),
            "score_std": float(row_cols[f"Score_Std_{label}"]),
            "q10": float(row_cols[f"Pred_Q10_{label}"]),
            "q50": float(row_cols[f"Pred_Q50_{label}"]),
            "q90": float(row_cols[f"Pred_Q90_{label}"]),
        }
        if best is None or cand["adj_prob"] > best["adj_prob"]:
            best = cand
    return best


def main():
    parser = argparse.ArgumentParser(description="Generate live buy/no-buy signals (v5).")
    parser.add_argument("--min-accepted", type=float, default=0.2)
    parser.add_argument("--std-factor", type=float, default=0.0)
    parser.add_argument("--mc-samples", type=int, default=25,
                        help="Reserved for the MC-dropout fallback; the ensemble is the default.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Reserved hook for an incremental Tiingo tail fetch of stale tickers.")
    parser.add_argument("--as-of-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    SIGNALS_DIR.mkdir(exist_ok=True)
    meta, cfg, scaler, members, temps = load_artifacts(args.device)

    universe = load_universe(args.tickers)
    ohlcv = fx.load_us_ohlcv(tickers=universe)
    fear_greed = fx.load_fear_greed()
    as_of = (pd.Timestamp(args.as_of_date).normalize() if args.as_of_date
             else (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)))
    run_date = str(pd.Timestamp.today().normalize().date())
    print(f"[v5-live] as_of={as_of.date()} tickers={len(universe)} members={len(members)}")

    rows, candidates, stale = [], [], []
    for ticker in universe:
        df = ohlcv.get(ticker)
        if df is None or df.empty:
            rows.append({"run_date": run_date, "ticker": ticker, "status": "missing_data"})
            continue
        data_max = pd.Timestamp(df["date"].max())
        if data_max < as_of:
            stale.append((ticker, data_max.date()))
        try:
            status, cols = score_ticker(ticker, df, fear_greed, scaler, cfg, members, temps,
                                        as_of, args.device)
        except Exception as exc:                       # one bad ticker must not abort the run
            rows.append({"run_date": run_date, "ticker": ticker, "status": f"error:{type(exc).__name__}"})
            continue
        if status != "scored":
            rows.append({"run_date": run_date, "ticker": ticker, "status": status})
            continue
        best = best_horizon(cols, args.std_factor)
        is_buy = best["adj_prob"] > args.min_accepted
        rows.append({
            "run_date": run_date, "date": run_date,
            "as_of_close_date": str(pd.Timestamp(cols["Date"]).date()),
            "ticker": ticker, "status": "buy_candidate" if is_buy else "no_buy",
            "close": float(cols["Close"]),
            "best_horizon": best["horizon"], "best_pred_prob": best["pred_prob"],
            "best_pred_std": best["pred_std"], "best_adj_prob": best["adj_prob"],
            "best_pred_down": best["pred_down"], "best_pred_neutral": best["pred_neutral"],
            "best_score": best["score"], "best_score_std": best["score_std"],
            "best_q10": best["q10"], "best_q50": best["q50"], "best_q90": best["q90"],
        })
        if is_buy:
            candidates.append({"ticker": ticker, **best,
                               "as_of_close_date": str(pd.Timestamp(cols["Date"]).date())})

    if stale:
        print(f"[v5-live] {len(stale)} ticker(s) lag {as_of.date()}: "
              + ", ".join(f"{t}@{d}" for t, d in stale[:20]) + (" ..." if len(stale) > 20 else ""))

    scores_df = pd.DataFrame(rows).sort_values(
        ["status", "best_adj_prob"], ascending=[True, False], na_position="last", ignore_index=True)
    scores_path = SIGNALS_DIR / f"live_scores_{run_date}.csv"
    scores_df.to_csv(scores_path, index=False)

    if candidates:
        top = max(candidates, key=lambda c: c["adj_prob"])
        decision = {"run_date": run_date, "date": run_date,
                    "as_of_close_date": top["as_of_close_date"], "decision": "BUY",
                    "reason": f"Best adjusted P(up) above threshold {args.min_accepted}",
                    "ticker": top["ticker"], "horizon": top["horizon"],
                    "adj_prob": top["adj_prob"], "pred_prob": top["pred_prob"],
                    "pred_std": top["pred_std"], "score": top["score"]}
    else:
        decision = {"run_date": run_date, "date": run_date, "as_of_close_date": "",
                    "decision": "NO_BUY",
                    "reason": f"No ticker above min_accepted={args.min_accepted}",
                    "ticker": "", "horizon": "", "adj_prob": np.nan,
                    "pred_prob": np.nan, "pred_std": np.nan, "score": np.nan}
    decision_path = SIGNALS_DIR / f"live_decision_{run_date}.csv"
    pd.DataFrame([decision]).to_csv(decision_path, index=False)
    print(f"[v5-live] wrote {scores_path}\n[v5-live] wrote {decision_path}")
    print(pd.DataFrame([decision]).to_string(index=False))


if __name__ == "__main__":
    main()
