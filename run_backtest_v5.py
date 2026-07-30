"""Descriptive OOS backtest for the v5 pipeline (single position, next-close fills).

DESCRIPTIVE ONLY: the decision gates adjudicate on the IC/CE metric suite; a single
~254-session single-position path cannot. This script exists to sanity-check the
signal -> execution chain under honest accounting: gap-aware split detection with
volume corroboration, t+1 entry fills matching the re-anchored labels, per-ticker
exit calendars, forced delisting exits with an explicit haircut sensitivity, a
round-trip cost grid, and a null distribution replaying the identical protocol.

Inputs: ``forecasts/*_forecast.csv`` (Score_{h} plus the transported Volume /
Tradable / VolMed63 columns), ``forecasts/split_info.json`` (required),
``forecasts/surface_manifest.json`` (absent -> all "time-only"), and the per-horizon
score floors from ``models/v5/v5_meta.json`` (CLI-overridable).

Split handling: a same-direction snap-set match with volume corroboration
back-adjusts all PRIOR closes by the exact ratio -- the ticker and its genuine
history stay in P&L (v4 whole-ticker-rejected on any large print, deleting real
crashes and preferentially deleting future losers via reverse splits). Candidate
selection, the tradability gate, and the corroboration statistic read ONLY
as-printed values dated <= t; back-adjusted closes exist only inside return booking
and equity marks, where a ratio detected at session s cancels out of any ratio of
two sessions both earlier than s -- so a mid-hold split adjusts the trade's return
at exit and never re-ranks any earlier day's candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FORECAST_DIR = PROJECT_ROOT / "forecasts"
V5_META_PATH = PROJECT_ROOT / "models" / "v5" / "v5_meta.json"
SPY_CSV = (PROJECT_ROOT / "TrainingData" / "indicators_data" / "processed"
           / "SPY-VIX" / "SPY_daily_processed.csv")
VIDEOS_DIR = PROJECT_ROOT / "videos"

HORIZON_LABELS = ("1d", "1w", "1m", "6m")
HORIZON_DAYS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}
# Split-snap ratios, rho > 1 only -- NO reciprocals: an exists-search over
# reciprocal rho would let a genuine -50% crash match rho=1/2 on the reverse
# branch, whose corroboration bound flat volume vacuously satisfies.
SNAP_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0)
SNAP_TOL = 0.02
SPLIT_MAX_SPAN_DAYS = 7          # returns spanning more calendar days are never candidates
ENTRY_MAX_SPAN_DAYS = 5          # no next print within this window -> skip the fill
COST_GRID_BP = (0, 10, 25)
DEATH_SLACK_SESSIONS = 5         # last print this close to the OOS end = clamp, not death


# ------------------------------------------------------------ split detector

def detect_splits(dates, close, volume, vol_med63):
    """Per-print split detection on the as-printed series.

    Returns ``(adjust, events)``: ``adjust[j]`` multiplies close[j] onto the FINAL
    basis (the product over splits detected at sessions s > j of 1/rho for forward
    and rho for reverse splits), and ``events`` is the detection log. Only booking
    and equity marks may consume ``adjust``; ratios cancel out of any close ratio
    whose sessions both precede the split, which is what keeps the adjustment causal
    for trade returns.

    Branch sign-locking: a down print can only be a forward split (volume >=
    (rho/2) * median -- split days trade heavier) and an up print only a reverse
    split (volume <= (2/rho) * median -- consolidations shrink share count).
    Zero/absent volume or NaN median skips the corroboration test (snap-only,
    counted separately). Non-snapping large prints stay in P&L: the majority are
    genuine moves, and excluding them would delete real crashes.
    """
    n = len(close)
    day = pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[D]").astype(np.int64)
    events = []
    ratio_at = np.ones(n, dtype=np.float64)      # multiplier applied to closes < s
    for j in range(1, n):
        if day[j] - day[j - 1] > SPLIT_MAX_SPAN_DAYS:
            continue
        if close[j] <= 0 or close[j - 1] <= 0:
            continue
        lnr = float(np.log(close[j] / close[j - 1]))
        med = vol_med63[j]
        vol = volume[j]
        corroborate = np.isfinite(med) and med > 0 and np.isfinite(vol) and vol > 0
        hit = None
        if lnr < 0:                              # forward-split branch
            for rho in SNAP_RATIOS:
                if abs(lnr + np.log(rho)) <= SNAP_TOL:
                    if not corroborate or vol >= (rho / 2.0) * med:
                        hit = ("forward", rho, 1.0 / rho, not corroborate)
                    break
        elif lnr > 0:                            # reverse-split branch
            for rho in SNAP_RATIOS:
                if abs(lnr - np.log(rho)) <= SNAP_TOL:
                    if not corroborate or vol <= (2.0 / rho) * med:
                        hit = ("reverse", rho, rho, not corroborate)
                    break
        if hit:
            kind, rho, mult, snap_only = hit
            ratio_at[j] = mult
            events.append({"index": int(j), "date": str(pd.Timestamp(dates[j]).date()),
                           "kind": kind, "rho": rho, "snap_only": bool(snap_only)})
    # adjust[j] = product of mult over detections at s > j.
    adjust = np.ones(n, dtype=np.float64)
    running = 1.0
    for j in range(n - 1, -1, -1):
        adjust[j] = running
        running *= ratio_at[j]                   # detection AT j applies to closes < j
    return adjust, events


# ----------------------------------------------------------------- data load

def load_forecasts(forecast_dir, oos_start, exclude_missing_transport=False):
    frames = {}
    missing_transport = []
    for path in sorted(Path(forecast_dir).glob("*_forecast.csv")):
        ticker = path.name[:-len("_forecast.csv")]
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df[df["Date"] >= pd.Timestamp(oos_start)].reset_index(drop=True)
        if df.empty:
            continue
        if "Tradable" not in df.columns:
            missing_transport.append(ticker)
            if exclude_missing_transport:
                continue
            df["Tradable"] = 0                   # never a candidate without the gate
            df["Volume"] = np.nan
            df["VolMed63"] = np.nan
        adjust, events = detect_splits(df["Date"].to_numpy(), df["Close"].to_numpy(),
                                       df["Volume"].to_numpy(dtype=np.float64),
                                       df["VolMed63"].to_numpy(dtype=np.float64))
        df["AdjClose"] = df["Close"].to_numpy() * adjust
        frames[ticker] = {"df": df, "split_events": events}
    if missing_transport:
        print(f"[bt5] WARNING: {len(missing_transport)} forecast CSV(s) lack the "
              "Volume/Tradable/VolMed63 transport columns; those tickers can never "
              "be candidates")
    return frames


def load_score_floors(args):
    floors = {h: 0.0 for h in HORIZON_LABELS}
    src = "0.0 fallback"
    if V5_META_PATH.exists():
        meta = json.loads(V5_META_PATH.read_text(encoding="utf-8"))
        stored = meta.get("score_floors")
        if stored and all(stored.get(h) is not None for h in HORIZON_LABELS):
            floors = {h: float(stored[h]) for h in HORIZON_LABELS}
            src = str(V5_META_PATH)
    else:
        print(f"[bt5] WARNING: {V5_META_PATH} missing; score floors default to 0.0")
    for h in HORIZON_LABELS:
        cli = getattr(args, f"min_score_{h}")
        if cli is not None:
            floors[h] = cli
            src += f" (+CLI {h})"
    print(f"[bt5] score floors [{src}]: "
          + " ".join(f"{h}:{floors[h]:+.4f}" for h in HORIZON_LABELS))
    return floors


# ------------------------------------------------------------------ strategy

def build_candidates(frames, floors, horizons):
    """Per union date: the gated candidate list [(ticker, horizon, score), ...].
    Reads ONLY as-printed values dated at that session (Score, Tradable)."""
    by_date = {}
    date_index = {}
    for ticker, obj in frames.items():
        df = obj["df"]
        dates = df["Date"].to_numpy()
        date_index[ticker] = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        tradable = df["Tradable"].to_numpy()
        for h in horizons:
            scores = df[f"Score_{h}"].to_numpy(dtype=np.float64)
            floor = floors[h]
            ok = (tradable == 1) & np.isfinite(scores) & (scores > floor)
            for i in np.flatnonzero(ok):
                by_date.setdefault(pd.Timestamp(dates[i]), []).append(
                    (ticker, h, float(scores[i])))
    return by_date, date_index


def run_strategy(frames, by_date, date_index, union_dates, cost_bp, haircut,
                 pick="argmax", rng=None, collect_trades=False, surface=None):
    """One replay of the protocol. ``pick`` is ``argmax`` (the strategy) or
    ``uniform`` (a null draw from the identical gated candidate set). Returns the
    equity series on the union calendar plus counters (and the trade log when
    ``collect_trades``)."""
    equity = 1.0
    curve = np.empty(len(union_dates))
    trades = []
    counters = {"trades": 0, "skipped_no_fill": 0, "forced_exits": 0,
                "haircuts_applied": 0, "clamped_exits": 0}
    last_union = union_dates[-1]
    slack_cut = union_dates[max(0, len(union_dates) - 1 - DEATH_SLACK_SESSIONS)]

    pos = None            # dict(ticker, df, adj, entry_idx, exit_idx, forced, entry_equity)
    blackout_until = -1   # union index; re-entry earliest the FOLLOWING union date

    for ui, today in enumerate(union_dates):
        if pos is not None:
            df = pos["df"]
            # Mark to market at the ticker's own print if one lands today.
            i = pos["dindex"].get(today)
            if i is not None and i <= pos["exit_idx"]:
                equity = pos["entry_equity"] * (df["AdjClose"].iloc[i]
                                                / df["AdjClose"].iloc[pos["entry_idx"]])
                if i == pos["exit_idx"]:
                    equity *= (1.0 - cost_bp * 1e-4)
                    if pos["forced"] and pos["apply_haircut"]:
                        equity *= (1.0 - haircut)
                        counters["haircuts_applied"] += 1
                    if collect_trades:
                        trades.append({
                            "Ticker": pos["ticker"], "Horizon": pos["h"],
                            "SignalDate": str(pos["signal_date"].date()),
                            "EntryDate": str(df["Date"].iloc[pos["entry_idx"]].date()),
                            "ExitDate": str(df["Date"].iloc[i].date()),
                            "EntryClose": float(df["Close"].iloc[pos["entry_idx"]]),
                            "ExitClose": float(df["Close"].iloc[i]),
                            "LogR": float(np.log(df["AdjClose"].iloc[i]
                                                 / df["AdjClose"].iloc[pos["entry_idx"]])),
                            "CostBps": cost_bp,
                            "ForcedExit": int(pos["forced"]),
                            "HaircutApplied": int(pos["forced"] and pos["apply_haircut"]),
                            "ClampedExit": int(pos["clamped"]),
                            "SurfaceTag": (surface or {}).get(pos["ticker"], "time-only"),
                        })
                    counters["trades"] += 1
                    if pos["forced"]:
                        counters["forced_exits"] += 1
                    if pos["clamped"]:
                        counters["clamped_exits"] += 1
                    pos = None
                    blackout_until = ui + 1     # exit day admits no new candidate
        if pos is None and ui >= blackout_until:
            cands = by_date.get(today)
            if cands:
                if pick == "argmax":
                    ticker, h, _ = max(cands, key=lambda c: c[2])
                else:
                    ticker, h, _ = cands[rng.integers(0, len(cands))]
                obj = frames[ticker]
                df = obj["df"]
                dindex = date_index[ticker]
                sig_i = dindex[today]
                if sig_i + 1 >= len(df):
                    counters["skipped_no_fill"] += 1
                else:
                    entry_i = sig_i + 1
                    span = (df["Date"].iloc[entry_i] - today).days
                    if span > ENTRY_MAX_SPAN_DAYS:
                        counters["skipped_no_fill"] += 1
                    else:
                        d = HORIZON_DAYS[h]
                        target_i = entry_i + d
                        clamped = forced = False
                        apply_haircut = False
                        if target_i >= len(df):
                            exit_i = len(df) - 1
                            last_print = pd.Timestamp(df["Date"].iloc[-1])
                            if last_print < slack_cut:
                                forced = True       # death well before the window end
                                apply_haircut = haircut > 0
                            else:
                                clamped = True      # horizon crosses the OOS end
                        else:
                            exit_i = target_i
                        if exit_i <= entry_i:
                            counters["skipped_no_fill"] += 1
                        else:
                            pos = {"ticker": ticker, "h": h, "df": df,
                                   "dindex": dindex, "entry_idx": entry_i,
                                   "exit_idx": exit_i, "forced": forced,
                                   "clamped": clamped, "apply_haircut": apply_haircut,
                                   "entry_equity": equity, "signal_date": today}
        curve[ui] = equity
    # A still-open position at the calendar end: book it at its last mark (already
    # reflected in equity); count it.
    if pos is not None:
        counters["clamped_exits"] += 1
    return curve, counters, trades


def curve_metrics(curve):
    total = float(curve[-1] / curve[0] - 1.0)
    n = len(curve)
    cagr = float((curve[-1] / curve[0]) ** (252.0 / max(n, 1)) - 1.0)
    peak = np.maximum.accumulate(curve)
    mdd = float((curve / peak - 1.0).min())
    return {"total_return": total, "cagr": cagr, "max_drawdown": mdd,
            "sessions": int(n)}


# ---------------------------------------------------------------- benchmark

def spy_benchmark(union_dates, spy_tr_csv=None):
    src = Path(spy_tr_csv) if spy_tr_csv else SPY_CSV
    if not src.exists():
        print(f"[bt5] SPY benchmark skipped: {src} missing")
        return None
    df = pd.read_csv(src)
    date_col = "date" if "date" in df.columns else "Date"
    close_col = "close" if "close" in df.columns else "Close"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
    last_real = df[date_col].max()
    # Forward-fill only within the real span: the comparison truncates at
    # min(OOS end, last real SPY date) rather than flat-filling a stale tail.
    end = min(pd.Timestamp(union_dates[-1]), last_real)
    dates = [d for d in union_dates if pd.Timestamp(d) <= end]
    if not dates:
        return None
    s = df.set_index(date_col)[close_col]
    s = s[~s.index.duplicated(keep="last")]
    aligned = s.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=np.float64)
    ok = np.isfinite(aligned)
    if ok.sum() < 2:
        return None
    curve = aligned[ok] / aligned[ok][0]
    out = curve_metrics(curve)
    out["truncated_at"] = str(pd.Timestamp(end).date())
    out["source"] = str(src)
    out["note"] = ("price-only SPY (no dividends): understates buy-and-hold by "
                   "~1-1.5%/yr" if spy_tr_csv is None else "total-return series supplied")
    return out


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="v5 OOS backtest (descriptive only).")
    p.add_argument("--forecast-dir", default=str(FORECAST_DIR))
    p.add_argument("--cost-bps", type=float, default=None,
                   help="Extra cost grid point beside the standard {0, 10, 25} bp; "
                        "also selects the trade-log configuration.")
    p.add_argument("--delist-haircut", type=float, default=0.0,
                   help="Haircut applied on forced delisting exits (an extra report "
                        "row at 0.3 is always produced -- the bankruptcy-vs-"
                        "acquisition sensitivity).")
    p.add_argument("--exclude-horizons", default="6m",
                   help="Comma-separated horizons excluded from candidacy "
                        "(default 6m until the pooled evidence exists).")
    for h in HORIZON_LABELS:
        p.add_argument(f"--min-score-{h}", type=float, default=None)
    p.add_argument("--null-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spy-tr-csv", default=None,
                   help="Optional total-return SPY series (preferred when supplied; "
                        "no TR series exists in the repo).")
    args = p.parse_args()

    print("[bt5] DESCRIPTIVE ONLY: the pre-registered IC/CE gates adjudicate the "
          "run; a single ~254-session single-position path cannot.")

    forecast_dir = Path(args.forecast_dir)
    split_path = forecast_dir / "split_info.json"
    if not split_path.exists():
        raise SystemExit(f"{split_path} missing -- refuse to backtest without the "
                         "split contract (train/finalize first)")
    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    oos_start = split_info.get("oos_start")
    if not oos_start:
        raise SystemExit("split_info.json carries no oos_start -- refuse to backtest")
    print(f"[bt5] OOS start {oos_start} (from {split_path})")

    surface_path = forecast_dir / "surface_manifest.json"
    surface = (json.loads(surface_path.read_text(encoding="utf-8"))
               if surface_path.exists() else {})
    if not surface:
        print("[bt5] surface_manifest.json absent -> all tickers treated time-only")

    horizons = [h for h in HORIZON_LABELS
                if h not in set(x.strip() for x in args.exclude_horizons.split(",") if x)]
    print(f"[bt5] horizons in candidacy: {horizons} "
          f"(1d is cost-conditioned at the gates; 6m report-only by default)")

    floors = load_score_floors(args)
    frames = load_forecasts(forecast_dir, oos_start)
    if not frames:
        raise SystemExit("no forecast CSVs with OOS rows found")
    n_events = sum(len(o["split_events"]) for o in frames.values())
    n_snap_only = sum(1 for o in frames.values() for e in o["split_events"]
                      if e["snap_only"])
    print(f"[bt5] {len(frames)} tickers | split detections: {n_events} "
          f"({n_snap_only} snap-only, no volume corroboration available)")

    union_dates = sorted({pd.Timestamp(d) for o in frames.values()
                          for d in o["df"]["Date"]})
    print(f"[bt5] union calendar: {len(union_dates)} sessions "
          f"{union_dates[0].date()} .. {union_dates[-1].date()}")

    by_date, date_index = build_candidates(frames, floors, horizons)
    n_cand = sum(len(v) for v in by_date.values())
    print(f"[bt5] gated candidates: {n_cand} over {len(by_date)} dates")

    cost_grid = sorted(set(COST_GRID_BP) | ({args.cost_bps} if args.cost_bps is not None
                                            else set()))
    haircuts = sorted({args.delist_haircut, 0.3})
    primary_cost = args.cost_bps if args.cost_bps is not None else 0
    report = {"descriptive_only": True, "oos_start": oos_start,
              "horizons": horizons, "score_floors": floors,
              "n_tickers": len(frames), "split_detections": n_events,
              "split_detections_snap_only": n_snap_only, "grid": {}}

    trade_log = None
    for cost in cost_grid:
        for hc in haircuts:
            key = f"cost{int(cost)}bp_haircut{hc:g}"
            collect = (cost == primary_cost and hc == args.delist_haircut)
            curve, counters, trades = run_strategy(
                frames, by_date, date_index, union_dates, cost, hc,
                pick="argmax", collect_trades=collect, surface=surface)
            entry = {"strategy": curve_metrics(curve), "counters": counters}
            if collect:
                trade_log = trades
                entry["primary"] = True
            # Null distribution: uniform picks from the identical gated sets.
            if args.null_runs > 0:
                rng = np.random.default_rng(args.seed)
                nulls = {"total_return": [], "cagr": [], "max_drawdown": []}
                for _ in range(args.null_runs):
                    ncurve, _, _ = run_strategy(frames, by_date, date_index,
                                                union_dates, cost, hc,
                                                pick="uniform", rng=rng)
                    nm = curve_metrics(ncurve)
                    for k in nulls:
                        nulls[k].append(nm[k])
                entry["null"] = {}
                for k, vals in nulls.items():
                    vals = np.asarray(vals)
                    strat_v = entry["strategy"][k]
                    sd = float(vals.std())
                    entry["null"][k] = {
                        "mean": float(vals.mean()), "std": sd,
                        "percentile": float((vals < strat_v).mean()),
                        "z": float((strat_v - vals.mean()) / sd) if sd > 0 else float("nan"),
                    }
            report["grid"][key] = entry
            s = entry["strategy"]
            nz = entry.get("null", {}).get("total_return", {})
            print(f"[bt5] [{key}] total {s['total_return']:+.2%} | CAGR {s['cagr']:+.2%} "
                  f"| maxDD {s['max_drawdown']:+.2%} | trades {counters['trades']} "
                  f"(forced {counters['forced_exits']}, clamped {counters['clamped_exits']}, "
                  f"no-fill {counters['skipped_no_fill']})"
                  + (f" | null z {nz.get('z', float('nan')):+.2f} "
                     f"pct {nz.get('percentile', float('nan')):.2f}" if nz else ""))

    bench = spy_benchmark(union_dates, args.spy_tr_csv)
    if bench:
        report["spy_benchmark"] = bench
        print(f"[bt5] SPY benchmark: total {bench['total_return']:+.2%} | "
              f"CAGR {bench['cagr']:+.2%} | truncated at {bench['truncated_at']} | "
              f"{bench['note']}")

    VIDEOS_DIR.mkdir(exist_ok=True)
    out_json = VIDEOS_DIR / "backtest_metrics_v5.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bt5] wrote {out_json}")
    if trade_log is not None:
        out_csv = PROJECT_ROOT / "trade_summary_v5.csv"
        pd.DataFrame(trade_log).to_csv(out_csv, index=False)
        print(f"[bt5] wrote {out_csv} ({len(trade_log)} trades)")

    # Best-effort equity plot (matplotlib optional).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve, _, _ = run_strategy(frames, by_date, date_index, union_dates,
                                   primary_cost, args.delist_haircut, pick="argmax")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
        ax.set_facecolor("black")
        ax.plot(union_dates, curve, color="#39ff14", lw=1.5)
        ax.set_title("v5 backtest equity (descriptive only)", color="#39ff14")
        ax.tick_params(colors="#39ff14")
        for spine in ax.spines.values():
            spine.set_color("#39ff14")
        out_png = PROJECT_ROOT / "output_plots" / "backtest_equity_v5.png"
        out_png.parent.mkdir(exist_ok=True)
        fig.savefig(out_png, facecolor="black", dpi=120)
        plt.close(fig)
        print(f"[bt5] wrote {out_png}")
    except Exception as exc:
        print(f"[bt5] equity plot skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
