"""Emit live buy/no-buy signals from the trained ensemble.

Loads the v5 artifacts (feature/normalization metadata, config, and the member
checkpoints) from the run directory, builds the most recent window per ticker
through the shared feature builder so the inputs are bit-identical to training,
runs the deep-ensemble predictor, and writes two CSVs per run: per-ticker scores
and a single buy decision.

Decision semantics (post label-centering): candidates are ranked and gated on the
consumed score ``Score_h = P(up) - P(down)`` against the per-horizon floors the
finalize wrote into ``v5_meta.json`` (quantiles of the tradability-filtered emitted
score). An absolute P(up) floor is meaningless once labels are market-relative, so
there is none -- the cash decision lives outside the model by design. Buy candidacy
additionally requires the causal tradability rule at the anchor, mirroring the
backtest's candidate gate: the floors are calibrated on the tradability-filtered
pool, and an unfiltered decision pool would recreate exactly the
calibration-pool/decision-pool mismatch. Raw P(up) stays in the output as the
second score, never gated on.

The window builder enforces the same ``min_real_rows`` floor as training; a ticker
with too little history is reported with the ``insufficient_window`` status rather
than scored. A feed whose freshest session lags the decision date by more than the
staleness bound hard-fails the whole run -- a silently stale snapshot scored as
fresh is worse than no signal.
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
DEFAULT_RUN_DIR = PROJECT_ROOT / "models" / "v5"
SIGNALS_DIR = PROJECT_ROOT / "signals"
STOCK_LIST_PATH = PROJECT_ROOT / "TrainingData" / "stockList.csv"
FG_STALENESS_SESSIONS = 5


def load_artifacts(device, run_dir=DEFAULT_RUN_DIR):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "v5_meta.json").read_text(encoding="utf-8"))
    if meta["feature_names_hash"] != fx.feature_names_hash():
        raise ValueError("feature spec has changed since training; retrain before scoring")
    cfg = V5Config(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    assert_theta_on_bin_edges(cfg)
    scaler = fx.RobustScaler.load(run_dir / "v5_norm.json")
    members = []
    for i in range(meta["n_members"]):
        model = V5Backbone(cfg)
        state = torch.load(run_dir / f"member_{i}.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        members.append(model)
    return meta, cfg, scaler, members, meta.get("temperatures")


def load_universe(tickers_arg, use_stocklist=False):
    """Default universe = the trained parquet universe: the score floors are
    quantiles of a ~5k-name argmax and do not transfer to a 350-name list (which
    also carries symbols absent from the US-filtered feed as permanent
    missing_data rows). ``--stocklist`` restores the legacy list explicitly."""
    if tickers_arg:
        return [t.upper() for t in tickers_arg]
    if use_stocklist:
        if not STOCK_LIST_PATH.exists():
            raise SystemExit(f"--stocklist requested but {STOCK_LIST_PATH} is missing")
        col = pd.read_csv(STOCK_LIST_PATH, header=None).iloc[:, 0].astype(str).str.strip().str.upper()
        return [t for t in col if t and t not in {"SYMBOL", "TICKER"}]
    return fx.list_us_tickers()


def resolve_score_floors(meta, args):
    floors = {}
    stored = meta.get("score_floors") or {}
    for h in HORIZON_LABELS:
        cli = getattr(args, f"min_score_{h}", None)
        if cli is not None:
            floors[h] = cli
        elif stored.get(h) is not None:
            floors[h] = float(stored[h])
        else:
            floors[h] = 0.0
    return floors


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
        bin_width=cfg.bin_width, n_bins=cfg.n_bins,
        volume=ff.volume[[anchor]], tradable=ff.tradable[[anchor]],
        vol_med63=ff.vol_med63[[anchor]])
    return "scored", {k: (v[0] if hasattr(v, "__len__") and len(v) == 1 else v) for k, v in cols.items()}


def best_horizon(row_cols, std_factor, horizons, floors):
    """Rank on adj Score = Score - std_factor * Std(Score) over the included
    horizons; a horizon qualifies for buy candidacy iff its adj Score clears that
    horizon's floor. Returns (best, qualifying): the argmax row (for reporting even
    when nothing qualifies) and the qualifying subset winner or None."""
    best, best_q = None, None
    for label in horizons:
        score = float(row_cols[f"Score_{label}"])
        sstd = max(float(row_cols[f"Score_Std_{label}"]), 1e-8)
        adj = score - std_factor * sstd
        cand = {
            "horizon": label, "score": score, "score_std": sstd, "adj_score": adj,
            "floor": floors[label],
            "pred_prob": float(row_cols[f"Pred_Prob_{label}"]),
            "pred_std": float(row_cols[f"Pred_Prob_Std_{label}"]),
            "pred_down": float(row_cols[f"Pred_Prob_Down_{label}"]),
            "pred_neutral": float(row_cols[f"Pred_Prob_Neutral_{label}"]),
            "q10": float(row_cols[f"Pred_Q10_{label}"]),
            "q50": float(row_cols[f"Pred_Q50_{label}"]),
            "q90": float(row_cols[f"Pred_Q90_{label}"]),
        }
        if best is None or cand["adj_score"] > best["adj_score"]:
            best = cand
        if adj > floors[label] and (best_q is None or adj > best_q["adj_score"]):
            best_q = cand
    return best, best_q


def decide_ticker(cols, std_factor, horizons, floors):
    """Per-ticker decision: (is_buy, chosen, tradable). Untradable tickers are
    scored and reported but can never be buy candidates -- the floors are
    calibrated on the tradability-filtered pool, and an unfiltered decision pool
    would recreate the calibration-pool/decision-pool mismatch."""
    tradable = int(cols.get("Tradable", 0))
    best, best_q = best_horizon(cols, std_factor, horizons, floors)
    is_buy = tradable == 1 and best_q is not None
    return is_buy, (best_q if is_buy else best), tradable


def check_feed_staleness(freshest, as_of, staleness_k):
    """SystemExit when the feed's freshest session lags the decision date by more
    than ``staleness_k`` business days -- scoring a silently stale snapshot as
    fresh is the failure mode this guard exists for."""
    if freshest is None:
        raise SystemExit("no OHLCV rows loaded for the requested universe")
    lag_sessions = int(np.busday_count(pd.Timestamp(freshest).date(),
                                       pd.Timestamp(as_of).date()))
    if lag_sessions > staleness_k:
        raise SystemExit(
            f"feed stale: freshest session {pd.Timestamp(freshest).date()} lags as_of "
            f"{pd.Timestamp(as_of).date()} by {lag_sessions} business days "
            f"(> staleness_k={staleness_k}); refresh the feed (and regenerate the "
            "session census in the same operation)")
    return lag_sessions


def main():
    parser = argparse.ArgumentParser(description="Generate live buy/no-buy signals (v5).")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                        help="Run directory holding v5_meta.json / v5_norm.json / "
                             "config.json / member checkpoints.")
    parser.add_argument("--std-factor", type=float, default=0.0)
    for h in HORIZON_LABELS:
        parser.add_argument(f"--min-score-{h}", type=float, default=None,
                            help=f"Override the stored Score floor for {h}.")
    parser.add_argument("--include-6m", action="store_true",
                        help="Let 6m compete in the argmax (report-only by default "
                             "until the pooled gate evidence exists).")
    parser.add_argument("--stocklist", action="store_true",
                        help="Use the legacy TrainingData/stockList.csv universe "
                             "instead of the trained parquet universe.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Reserved: incremental Tiingo tail fetch + census + "
                             "fear/greed + SPY refresh (the feed refresh contract). "
                             "Until implemented, the staleness hard-fail is the guard.")
    parser.add_argument("--as-of-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    SIGNALS_DIR.mkdir(exist_ok=True)
    meta, cfg, scaler, members, temps = load_artifacts(args.device, args.run_dir)
    staleness_k = int(meta.get("staleness_k", 63))
    floors = resolve_score_floors(meta, args)
    horizons = list(HORIZON_LABELS) if args.include_6m else ["1d", "1w", "1m"]
    if args.std_factor != 0.0:
        print(f"[v5-live] CAVEAT: --std-factor {args.std_factor} with n={len(members)} "
              "members -- the member-std has ~40% sampling CV at 4 members; the "
              "uncertainty-quality check must establish usability before this knob "
              "means much.")
    print(f"[v5-live] floors: " + " ".join(f"{h}:{floors[h]:+.4f}" for h in HORIZON_LABELS)
          + f" | horizons in argmax: {horizons} (1d is cost-conditioned at the gates)")

    universe = load_universe(args.tickers, args.stocklist)
    ohlcv = fx.load_us_ohlcv(tickers=universe)
    fear_greed = fx.load_fear_greed()
    as_of = (pd.Timestamp(args.as_of_date).normalize() if args.as_of_date
             else (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)))
    run_date = str(pd.Timestamp.today().normalize().date())
    print(f"[v5-live] as_of={as_of.date()} tickers={len(universe)} members={len(members)}")

    # --force-refresh is a stub until the feed refresh contract is implemented; the
    # staleness hard-fail below is the guard in the interim.
    freshest = max((pd.Timestamp(df["date"].max()) for df in ohlcv.values() if len(df)),
                   default=None)
    check_feed_staleness(freshest, as_of, staleness_k)
    fg_last = fear_greed["date"].max() if len(fear_greed) else None
    if fg_last is not None:
        fg_lag = int(np.busday_count(pd.Timestamp(fg_last).date(), as_of.date()))
        if fg_lag > FG_STALENESS_SESSIONS:
            print(f"[v5-live] WARNING: fear_greed.csv ends {pd.Timestamp(fg_last).date()}, "
                  f"{fg_lag} business days behind as_of -- a frozen fg progressively "
                  "degrades the fg_corr channel; refresh it with the feed.")

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
        is_buy, chosen, tradable = decide_ticker(cols, args.std_factor, horizons, floors)
        rows.append({
            "run_date": run_date, "date": run_date,
            "as_of_close_date": str(pd.Timestamp(cols["Date"]).date()),
            "ticker": ticker, "status": "buy_candidate" if is_buy else "no_buy",
            "close": float(cols["Close"]), "tradable": tradable,
            "best_horizon": chosen["horizon"], "best_score": chosen["score"],
            "best_score_std": chosen["score_std"], "best_adj_score": chosen["adj_score"],
            "floor": chosen["floor"],
            "best_pred_prob": chosen["pred_prob"], "best_pred_std": chosen["pred_std"],
            "best_pred_down": chosen["pred_down"], "best_pred_neutral": chosen["pred_neutral"],
            "best_q10": chosen["q10"], "best_q50": chosen["q50"], "best_q90": chosen["q90"],
        })
        if is_buy:
            candidates.append({"ticker": ticker, **best_q,
                               "as_of_close_date": str(pd.Timestamp(cols["Date"]).date())})

    if stale:
        print(f"[v5-live] {len(stale)} ticker(s) lag {as_of.date()}: "
              + ", ".join(f"{t}@{d}" for t, d in stale[:20]) + (" ..." if len(stale) > 20 else ""))

    scores_df = pd.DataFrame(rows).sort_values(
        ["status", "best_adj_score"], ascending=[True, False], na_position="last",
        ignore_index=True)
    scores_path = SIGNALS_DIR / f"live_scores_{run_date}.csv"
    scores_df.to_csv(scores_path, index=False)

    if candidates:
        top = max(candidates, key=lambda c: c["adj_score"])
        decision = {"run_date": run_date, "date": run_date,
                    "as_of_close_date": top["as_of_close_date"], "decision": "BUY",
                    "reason": f"Best adjusted Score above the per-horizon floor "
                              f"({top['adj_score']:+.4f} > {top['floor']:+.4f})",
                    "ticker": top["ticker"], "horizon": top["horizon"],
                    "score": top["score"], "adj_score": top["adj_score"],
                    "pred_prob": top["pred_prob"], "pred_std": top["pred_std"]}
    else:
        decision = {"run_date": run_date, "date": run_date, "as_of_close_date": "",
                    "decision": "NO_BUY",
                    "reason": "No tradable ticker's adjusted Score clears its floor",
                    "ticker": "", "horizon": "", "score": np.nan, "adj_score": np.nan,
                    "pred_prob": np.nan, "pred_std": np.nan}
    decision_path = SIGNALS_DIR / f"live_decision_{run_date}.csv"
    pd.DataFrame([decision]).to_csv(decision_path, index=False)
    print(f"[v5-live] wrote {scores_path}\n[v5-live] wrote {decision_path}")
    print(pd.DataFrame([decision]).to_string(index=False))


if __name__ == "__main__":
    main()
