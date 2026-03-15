import os
import sys
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(os.getcwd())
video_dir       = PROJECT_ROOT / "videos"
output_plots_dir = PROJECT_ROOT / "output_plots"
forecast_dir    = PROJECT_ROOT / "forecasts"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(output_plots_dir, exist_ok=True)

# Config
initial_value   = 1000.0
random_runs     = 10
SPIKE_THRESHOLD = 0.80      # Reject stocks with daily abs change > 80%
MIN_ACCEPTED    = 0.2       # Minimum adjusted probability required
STD_FACTOR      = 0.00       # AdjustedProb = PredProb - STD_FACTOR * StdDev

# Out-of-sample backtest: only dates >= this are used. Strategy is evaluated ONLY on
# the test period (after training and validation). Prefer split_info.json from the
# notebook so backtest and notebook stay in sync.
def _get_oos_start():
    env = os.environ.get("BACKTEST_OOS_START", None)
    if env:
        return env
    split_path = PROJECT_ROOT / "forecasts" / "split_info.json"
    if split_path.exists():
        try:
            info = json.loads(split_path.read_text())
            oos = info.get("oos_start")
            if oos:
                return oos
        except Exception:
            pass
    oos_file = PROJECT_ROOT / "forecasts" / "oos_start_date.txt"
    if oos_file.exists():
        try:
            return oos_file.read_text().strip()
        except Exception:
            pass
    return None

OOS_START_DATE = _get_oos_start()

# Backtest horizons
PERIODS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}

# Load only columns needed for backtest to reduce memory
REQUIRED_COLS = ["Date", "Close"] + [f"Pred_Prob_{l}" for l in PERIODS] + [f"Pred_Prob_Std_{l}" for l in PERIODS]

all_forecasts = {}
rejected_tickers = []

for filename in os.listdir(forecast_dir):

    if not filename.endswith("_forecast.csv"):
        continue

    filepath = forecast_dir / filename
    ticker   = filename.split("_forecast")[0]

    df = pd.read_csv(filepath, parse_dates=["Date"], usecols=lambda c: c in REQUIRED_COLS)
    df = df.set_index("Date").sort_index()
    df["Ticker"] = ticker

    # Spike filter (I need to build something better for stock splits in the future lol)
    if "Close" in df.columns:
        df["pct_change"] = df["Close"].pct_change()
        max_change       = df["pct_change"].abs().max()

        if max_change > SPIKE_THRESHOLD:
            rejected_tickers.append((ticker, max_change))
            continue

        # Compute realized log returns for horizons
        for label, days in PERIODS.items():
            df[f"Actual_LogR_{label}"] = np.log(df["Close"].shift(-days) / df["Close"])

    all_forecasts[ticker] = df


# Report rejected tickers
if rejected_tickers:
    print("\nRejected tickers due to unrealistic daily spikes (>30%):")
    for t, m in rejected_tickers:
        print(f"  - {t}: {m:.2%}")
else:
    print("\nNo stocks rejected for excessive daily spikes.")

if not all_forecasts:
    print("\nNo forecast files found in forecasts/ (need *_forecast.csv). Exiting.")
    sys.exit(0)

# Trading dates: use union so we can trade whenever any ticker has data.
# (Intersection would require every ticker to have every date → often empty.)
all_dates     = [set(df.index) for df in all_forecasts.values()]
common_dates  = sorted(set.union(*all_dates))

# Restrict to test period only: strategy must run ONLY on data the model was not trained on
# (and not used for validation). oos_start = first date of test period.
if OOS_START_DATE:
    oos_cutoff = pd.Timestamp(OOS_START_DATE)
    common_dates = [d for d in common_dates if d >= oos_cutoff]
    print(f"\n[OOS] Backtest starts at {OOS_START_DATE} (test period only; no training or validation dates).")
    print(f"      {len(common_dates)} trading days in backtest window.")
else:
    print("\n[WARNING] No OOS start date. Run the notebook to create forecasts/split_info.json, or set BACKTEST_OOS_START.")
    print("          Backtest would include in-sample dates → inflated returns. Exiting to avoid accidental use of training data.")
    sys.exit(1)

# Print training/validation/backtest windows from notebook (if split_info.json exists)
split_info_path = PROJECT_ROOT / "forecasts" / "split_info.json"
if split_info_path.exists():
    try:
        split_info = json.loads(split_info_path.read_text())
        print("\n--- Date windows (from run_forecast_v4.ipynb) ---")
        print("  Training window:  ", split_info.get("train_start", "?"), "to", split_info.get("train_end", "?"))
        print("  Validation window:", split_info.get("val_start", "?"), "to", split_info.get("val_end", "?"))
        print("  Backtest window:  ", split_info.get("oos_start", "?"), "to", split_info.get("data_end", "?"), "  (test only; model not trained on this)")
    except Exception as e:
        print("\n[INFO] Could not read split_info.json:", e)
if common_dates:
    print("\n--- Backtest date range (actual) ---")
    print("  First date:", common_dates[0].date() if hasattr(common_dates[0], "date") else common_dates[0])
    print("  Last date: ", common_dates[-1].date() if hasattr(common_dates[-1], "date") else common_dates[-1])
    print("  Total:    ", len(common_dates), "trading days")

print(f"\nUsing {len(common_dates)} trading dates across {len(all_forecasts)} tickers")

if not common_dates:
    print("No trading dates after OOS filter. Set BACKTEST_OOS_START later or leave unset.")
    sys.exit(0)


# Backtesting strategy:
# Rules:
#   For each date, compute adj_prob = Pred_Prob - STD_FACTOR * Pred_Prob_Std
#   Reject candidates with adj_prob <= MIN_ACCEPTED
#   Out of all candidates, choose the one with the highest adj_prob
#   Only hold 1 position at a time

strategy_value  = initial_value
strategy_history = []
trade_log        = []
buy_points       = []

current_hold = None
successful_buys = 0
total_buys      = 0

for i, date in enumerate(common_dates):
    # Exit if horizon has expired
    if current_hold is not None and i == current_hold["exit_idx"]:

        realized_logr = current_hold["Actual_LogR"]
        realized_pct  = np.exp(realized_logr) - 1 if not np.isnan(realized_logr) else np.nan

        trade_log.append({
            "BuyDate"        : current_hold["BuyDate"],
            "SellDate"       : date,
            "Ticker"         : current_hold["Ticker"],
            "Horizon"        : current_hold["Period"],
            "DaysHeld"       : current_hold["Days"],
            "Pred_Prob"      : current_hold["Pred_Prob"],
            "Pred_Prob_Std"  : current_hold["Pred_Prob_Std"],
            "Adj_Prob"       : current_hold["Adj_Prob"],
            "Actual_LogR"    : realized_logr,
            "Actual_Return%" : realized_pct * 100 if not np.isnan(realized_pct) else np.nan,
        })

        if not np.isnan(realized_logr):
            strategy_value *= np.exp(realized_logr)
            if realized_logr > 0:
                successful_buys += 1

        current_hold = None

    # If still holding a position, skip new buys
    if current_hold is not None:
        strategy_history.append(strategy_value)
        continue

    # Build buy candidates for today
    candidates = []

    for ticker, df in all_forecasts.items():

        if date not in df.index:
            continue

        row = df.loc[date]

        for label, days in PERIODS.items():

            prob_col = f"Pred_Prob_{label}"
            std_col  = f"Pred_Prob_Std_{label}"
            act_col  = f"Actual_LogR_{label}"

            if prob_col not in row or pd.isna(row[prob_col]):
                continue

            pred_prob = float(row[prob_col])
            pred_std  = float(row.get(std_col, 0.0))
            pred_std  = max(pred_std, 1e-6)

            adj_prob = pred_prob - STD_FACTOR * pred_std

            if adj_prob <= MIN_ACCEPTED:
                continue

            actual_logr = float(row.get(act_col, np.nan))

            candidates.append({
                "Ticker"        : ticker,
                "Period"        : label,
                "Days"          : days,
                "Pred_Prob"     : pred_prob,
                "Pred_Prob_Std" : pred_std,
                "Adj_Prob"      : adj_prob,
                "Actual_LogR"   : actual_logr,
            })

    # No candidates, then nothing to buy
    if not candidates:
        strategy_history.append(strategy_value)
        continue

    # Selecting best candidates
    best = max(candidates, key=lambda x: x["Adj_Prob"])

    entry_idx = i
    exit_idx  = min(i + best["Days"], len(common_dates) - 1)

    best["entry_idx"] = entry_idx
    best["exit_idx"]  = exit_idx
    best["BuyDate"]   = date

    current_hold = best
    total_buys  += 1

    buy_points.append({
        "Date"        : date,
        "Value"       : strategy_value,
        "Ticker"      : best["Ticker"],
        "Horizon"     : best["Period"],
        "Pred_Prob"   : best["Pred_Prob"],
        "Adj_Prob"    : best["Adj_Prob"],
    })

    strategy_history.append(strategy_value)


# save trade summary
summary_df = pd.DataFrame(trade_log)
summary_df.to_csv("trade_summary_prob_strategy.csv", index=False)

print("\nSaved trade summary to trade_summary_prob_strategy.csv")
print(summary_df.head())

if total_buys > 0:
    print(f"\nBuy success rate: {successful_buys}/{total_buys} = {(successful_buys/total_buys):.2%}")
else:
    print("\nNo completed trades.")


# Random Baseline (1-day random picks)

returns_by_date = {}

for date in common_dates:
    vals = []
    for ticker, df in all_forecasts.items():
        if date in df.index and "Actual_LogR_1d" in df.columns:
            v = df.loc[date, "Actual_LogR_1d"]
            if not pd.isna(v):
                vals.append(v)
    returns_by_date[date] = vals

random_results = np.zeros((len(common_dates), random_runs))

for run in range(random_runs):
    value = initial_value
    for i, date in enumerate(common_dates):
        if returns_by_date[date]:
            pick = random.choice(returns_by_date[date])
            value *= np.exp(pick)
        random_results[i, run] = value

random_mean = np.mean(random_results, axis=1)
random_std  = np.std(random_results, axis=1)

# SPY BUY & HOLD: start at same date as backtest (fair comparison — both lines start at initial_value)
spy_path = PROJECT_ROOT / "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"
strategy_history_arr = np.array(strategy_history)
n_dates = len(common_dates)
spy_values_arr = np.full(n_dates, initial_value)
if n_dates > 0 and spy_path.exists():
    spy_df = pd.read_csv(spy_path, parse_dates=["date"])
    spy_df = spy_df.rename(columns={"date": "Date", "close": "Close"})
    spy_df = spy_df.sort_values("Date")
    spy_close = spy_df.set_index("Date")["Close"]
    # Align SPY close to backtest dates (ffill so we have a price for every common_date)
    spy_close_aligned = spy_close.reindex(pd.DatetimeIndex(common_dates)).ffill()
    # Normalize so SPY = initial_value on first backtest date (same as strategy)
    close_at_start = spy_close_aligned.iloc[0]
    if pd.notna(close_at_start) and close_at_start > 0:
        spy_values_arr = (initial_value * (spy_close_aligned / close_at_start)).values
    else:
        spy_values_arr = np.full(n_dates, initial_value)
elif not spy_path.exists():
    print(f"\nSPY file not found at {spy_path}; using flat baseline for plot.")

# Plotting: defer matplotlib/PIL so script runs even if they fail (e.g. PIL DLL)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    plt.rcParams["image.cmap"] = "viridis"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["savefig.facecolor"] = "black"
    plt.rcParams["savefig.edgecolor"] = "black"
    try:
        from PIL import Image
        def force_png_rgb(path):
            img = Image.open(path).convert("RGB")
            img.save(path)
    except Exception:
        def force_png_rgb(path):
            pass

    def save_final_plot(
        dates,
        random_results,
        strat_values,
        spy_values,
        random_mean=None,
        random_std=None,
        show_uncertainty=False,
        filename="final_plot.png",
    ):
        # YouTube 1080p (1920x1080), 16:9
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        for r in range(random_results.shape[1]):
            ax.plot(dates, random_results[:, r], alpha=0.12, lw=1.5, color="white")
        if show_uncertainty and random_mean is not None:
            ax.fill_between(
                dates, random_mean - 3 * random_std, random_mean + 3 * random_std,
                color="gray", alpha=0.08,
            )
            ax.fill_between(
                dates, random_mean - random_std, random_mean + random_std,
                color="gray", alpha=0.20,
            )
        ax.plot(dates, spy_values, color="white", lw=4, label="SPY")
        ax.plot(dates, strat_values, color="#39FF14", lw=4, label="Prob Strategy")
        ymin = 0.0
        ymax = 1.2 * max(np.nanmax(spy_values), np.nanmax(strat_values))
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_title("AI Probability Strategy vs Random vs SPY", color="white", fontsize=22)
        ax.set_xlabel("Date", color="white", fontsize=16)
        ax.set_ylabel("Portfolio Value", color="white", fontsize=16)
        ax.tick_params(colors="white", labelsize=14)
        legend = ax.legend(facecolor="black", edgecolor="white", fontsize=14)
        for text in legend.get_texts():
            text.set_color("white")
        png_path = video_dir / filename
        plt.savefig(png_path, dpi=100, facecolor="black")
        force_png_rgb(png_path)
        plt.close(fig)
        print(f"Saved FULL STATIC PNG: {png_path}")

    # 2K = 2560x1440 (QHD), 16:9; 5 seconds at 30 fps = 150 frames
    W_2K, H_2K = 25.6, 14.4
    FPS_VIDEO = 30
    DURATION_5S = 5

    def save_evolving_mp4(
        dates,
        random_results,
        strat_values,
        spy_values,
        random_mean=None,
        random_std=None,
        show_uncertainty=False,
        filename_mp4="evolving_plot.mp4",
        duration_sec=10,
        fps=30,
        figsize=(19.2, 10.8),
    ):
        """Produce MP4 with data evolving in time; y-axis tracks max of strategy line."""
        n = len(dates)
        if n == 0:
            return
        n_frames = int(duration_sec * fps)
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_xlim(dates[0], dates[-1])
        ax.set_title("AI Probability Strategy vs Random vs SPY", color="white", fontsize=22)
        ax.set_xlabel("Date", color="white", fontsize=16)
        ax.set_ylabel("Portfolio Value", color="white", fontsize=16)
        ax.tick_params(colors="white", labelsize=14)
        lines_random = [ax.plot([], [], alpha=0.12, lw=1.5, color="white")[0] for _ in range(random_results.shape[1])]
        line_spy, = ax.plot([], [], color="white", lw=4, label="SPY")
        line_strat, = ax.plot([], [], color="#39FF14", lw=4, label="Prob Strategy")
        line_spy.set_zorder(10)
        line_strat.set_zorder(10)
        legend = ax.legend(facecolor="black", edgecolor="white", fontsize=14)
        for text in legend.get_texts():
            text.set_color("white")
        fill_arts = []  # collect fill_between artists to remove each frame

        def init():
            for l in lines_random:
                l.set_data([], [])
            line_spy.set_data([], [])
            line_strat.set_data([], [])
            return []

        def update(frame):
            end_idx = min(n - 1, max(0, int((frame + 1) / n_frames * n)))
            t = end_idx + 1
            d = dates[:t]
            for r, l in enumerate(lines_random):
                l.set_data(d, random_results[:t, r])
            line_spy.set_data(d, spy_values[:t])
            line_strat.set_data(d, strat_values[:t])
            # Redraw uncertainty bands for current range
            for art in fill_arts:
                art.remove()
            fill_arts.clear()
            if show_uncertainty and random_mean is not None and t > 0:
                fill_arts.append(ax.fill_between(d, random_mean[:t] - 3 * random_std[:t], random_mean[:t] + 3 * random_std[:t], color="gray", alpha=0.08))
                fill_arts.append(ax.fill_between(d, random_mean[:t] - random_std[:t], random_mean[:t] + random_std[:t], color="gray", alpha=0.20))
            # Y-axis: min 0, max 1.2 * max(SPY, neon green strategy)
            visible_strat = strat_values[:t]
            visible_spy = spy_values[:t]
            y_min = 0.0
            y_max = 1.2 * max(np.nanmax(visible_spy), np.nanmax(visible_strat))
            ax.set_ylim(y_min, y_max)
            return []

        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, blit=False, interval=1000 / fps)
        out_path = video_dir / filename_mp4
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="backtest"), bitrate=5000)
        anim.save(str(out_path), writer=writer)
        plt.close(fig)
        print(f"Saved evolving MP4: {out_path}")

    def save_confidence_vs_return_mp4(summary_df, out_dir):
        """2K, 5s: scatter points appear over time."""
        if summary_df is None or len(summary_df) == 0:
            return
        if "Adj_Prob" not in summary_df.columns or "Actual_Return%" not in summary_df.columns:
            return
        x = summary_df["Adj_Prob"].values
        y = summary_df["Actual_Return%"].values
        y = np.where(np.isfinite(y), y, np.nan)
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        x, y = x[valid], y[valid]
        n_pts = len(x)
        n_frames = int(DURATION_5S * FPS_VIDEO)
        fig, ax = plt.subplots(figsize=(W_2K, H_2K), dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_xlim(np.nanmin(x) - 0.02, np.nanmax(x) + 0.02)
        y_min, y_max = np.nanmin(y) - 2, np.nanmax(y) + 2
        ax.set_ylim(min(y_min, -1), max(y_max, 1))
        ax.axhline(0, color="white", linestyle="--", lw=2, alpha=0.8)
        ax.set_xlabel("Adjusted probability at buy", color="white", fontsize=18)
        ax.set_ylabel("Actual return (%)", color="white", fontsize=18)
        ax.set_title("Model confidence vs actual return (each point = one trade)", color="white", fontsize=20)
        ax.tick_params(colors="white", labelsize=14)
        scat = ax.scatter([], [], alpha=0.7, s=150, color="#39FF14", edgecolors="white", linewidths=1.2, zorder=5)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return [scat]

        def update(frame):
            n_show = min(n_pts, max(1, int((frame + 1) / n_frames * n_pts)))
            scat.set_offsets(np.c_[x[:n_show], y[:n_show]])
            return [scat]

        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, blit=True, interval=1000 / FPS_VIDEO)
        out_path = out_dir / "confidence_vs_actual_return.mp4"
        writer = FFMpegWriter(fps=FPS_VIDEO, metadata=dict(artist="backtest"), bitrate=5000)
        anim.save(str(out_path), writer=writer)
        plt.close(fig)
        print(f"Saved confidence vs return MP4 (2K 5s): {out_path}")

    def save_confidence_vs_return(summary_df, out_dir):
        """B-roll: Predicted confidence (Adj_Prob) vs actual return %."""
        if summary_df is None or len(summary_df) == 0:
            return
        if "Adj_Prob" not in summary_df.columns or "Actual_Return%" not in summary_df.columns:
            return
        # YouTube 1080p (1920x1080), 16:9
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        x = summary_df["Adj_Prob"].values
        y = summary_df["Actual_Return%"].values
        y = np.where(np.isfinite(y), y, np.nan)
        valid = np.isfinite(y)
        if not np.any(valid):
            plt.close(fig)
            return
        ax.scatter(x[valid], y[valid], alpha=0.7, s=120, color="#39FF14", edgecolors="white", linewidths=1.2)
        ax.axhline(0, color="white", linestyle="--", lw=2, alpha=0.8)
        ax.set_xlabel("Adjusted probability at buy", color="white", fontsize=16)
        ax.set_ylabel("Actual return (%)", color="white", fontsize=16)
        ax.set_title("Model confidence vs actual return (each point = one trade)", color="white", fontsize=20)
        ax.tick_params(colors="white", labelsize=14)
        out_path = out_dir / "confidence_vs_actual_return.png"
        plt.savefig(out_path, dpi=100, facecolor="black")
        force_png_rgb(out_path)
        plt.close(fig)
        print(f"Saved confidence vs return: {out_path}")

    save_confidence_vs_return(summary_df, output_plots_dir)
    save_confidence_vs_return_mp4(summary_df, output_plots_dir)

    if n_dates > 0:
        save_final_plot(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            show_uncertainty=False, filename="random_vs_prob_strategy_clean.png",
        )
        save_final_plot(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            random_mean=random_mean, random_std=random_std, show_uncertainty=True,
            filename="random_vs_prob_strategy_uncertainty.png",
        )
        # 10-second evolving MP4s; y-axis tracks max of strategy (neon green) line
        save_evolving_mp4(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            show_uncertainty=False, filename_mp4="random_vs_prob_strategy_clean.mp4",
            duration_sec=10,
        )
        save_evolving_mp4(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            random_mean=random_mean, random_std=random_std, show_uncertainty=True,
            filename_mp4="random_vs_prob_strategy_uncertainty.mp4",
            duration_sec=10,
        )
        # 2K (2560x1440), 5 seconds
        save_evolving_mp4(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            show_uncertainty=False, filename_mp4="random_vs_prob_strategy_clean_2k_5s.mp4",
            duration_sec=DURATION_5S, fps=FPS_VIDEO, figsize=(W_2K, H_2K),
        )
        save_evolving_mp4(
            common_dates, random_results, strategy_history_arr, spy_values_arr,
            random_mean=random_mean, random_std=random_std, show_uncertainty=True,
            filename_mp4="random_vs_prob_strategy_uncertainty_2k_5s.mp4",
            duration_sec=DURATION_5S, fps=FPS_VIDEO, figsize=(W_2K, H_2K),
        )
    else:
        print("\nNo shared dates; skipping plots.")
except Exception as e:
    print(f"\nSkipping plots (matplotlib/PIL unavailable): {e}")
