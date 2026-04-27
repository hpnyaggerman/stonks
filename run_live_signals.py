import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "TrainingData"
RAW_STOCKS_DIR = TRAINING_DIR / "indicators_data" / "raw" / "stocksData"
PROCESSED_STOCKS_DIR = TRAINING_DIR / "indicators_data" / "processed" / "stocksData"
STOCK_LIST_PATH = TRAINING_DIR / "stockList.csv"

MODELS_DIR = PROJECT_ROOT / "models"
SIGNALS_DIR = PROJECT_ROOT / "signals"
MODEL_PATH_KERAS = MODELS_DIR / "latest_model.keras"
MODEL_PATH_H5 = MODELS_DIR / "latest_model.h5"
SCALER_PATH = MODELS_DIR / "latest_scalar.pkl"
META_PATH = MODELS_DIR / "latest_meta.json"


def resolve_saved_model_path(models_dir: Path) -> Path:
    """
    Keras 3 loads `.keras` only as a zip archive. Legacy full models are HDF5 and
    must use a `.h5` path. Some exports incorrectly used `.keras` for HDF5; we
    rename those once to `latest_model.h5`.
    """
    keras_path = models_dir / MODEL_PATH_KERAS.name
    h5_path = models_dir / MODEL_PATH_H5.name
    if keras_path.exists() and zipfile.is_zipfile(keras_path):
        return keras_path
    if h5_path.exists():
        return h5_path
    if keras_path.exists():
        with keras_path.open("rb") as handle:
            head = handle.read(8)
        if head.startswith(b"\x89HDF\r\n"):
            if h5_path.exists():
                raise FileExistsError(
                    "Both latest_model.keras and latest_model.h5 exist, and .keras "
                    "is legacy HDF5 (not a zip). Remove or rename one of them."
                )
            keras_path.rename(h5_path)
            print(
                "[INFO] Renamed legacy HDF5 weights latest_model.keras -> "
                "latest_model.h5 (Keras 3 expects a zip for .keras)."
            )
            return h5_path
        raise ValueError(
            f"{keras_path} is not a Keras 3 zip archive and not a legacy HDF5 model."
        )
    raise FileNotFoundError(
        "Missing model weights. Add native latest_model.keras (zip) or "
        "legacy latest_model.h5 under models/."
    )


def model_weights_file_present(models_dir: Path) -> bool:
    return (models_dir / MODEL_PATH_KERAS.name).exists() or (models_dir / MODEL_PATH_H5.name).exists()


def load_tickers():
    df = pd.read_csv(STOCK_LIST_PATH, header=None)
    tickers = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
    return [t for t in tickers if t and t not in {"SYMBOL", "TICKER"}]


def latest_completed_trading_day():
    """
    Last daily bar we expect raw/processed files to include (approximate equity session).

    Calendar "today" is wrong on Mondays: files usually end on the prior Friday, so
    comparing to Monday incorrectly marks fresh Friday data as stale. Use one business
    day back from the run date (not a full NYSE holiday calendar).
    """
    today = pd.Timestamp.today().normalize()
    return (today - pd.offsets.BDay(1)).normalize()


def safe_max_date(csv_path, date_col):
    if not csv_path.exists():
        
        return None
    try:
        df = pd.read_csv(csv_path, usecols=[date_col], parse_dates=[date_col])
        if df.empty:
            return None
        return pd.to_datetime(df[date_col]).max().normalize()
    except Exception:
        return None


def get_file_path_for_ticker(base_dir, ticker, suffix):
    return base_dir / f"{ticker}_{suffix}.csv"


def is_data_stale(tickers, expected_date):
    """
    Treat data as stale if any ticker is missing raw/processed file, if either
    file's max date is before expected_date, or if processed lags raw (common when
    only raw CSVs were refreshed).
    """
    for ticker in tickers:
        raw_path = get_file_path_for_ticker(RAW_STOCKS_DIR, ticker, "daily")
        proc_path = get_file_path_for_ticker(PROCESSED_STOCKS_DIR, ticker, "daily_processed")
        raw_max = safe_max_date(raw_path, "date")
        proc_max = safe_max_date(proc_path, "date")
        if raw_max is None or proc_max is None:
            return True
        if raw_max < expected_date or proc_max < expected_date:
            return True
        if proc_max < raw_max:
            return True
    return False


def run_refresh_pipeline():
    print("[INFO] Data stale/missing. Running downloader + processor...")
    subprocess.run([sys.executable, str(TRAINING_DIR / "downloader.py")], check=True, cwd=PROJECT_ROOT)
    subprocess.run([sys.executable, str(TRAINING_DIR / "processor.py")], check=True, cwd=PROJECT_ROOT)
    print("[INFO] Refresh complete.")


def ensure_optional_zero_defaults(df):
    # Fill optional feeds with zero when missing/stale.
    zero_cols = [
        "insider_shares",
        "insider_amount",
        "insider_buy_flag",
        "sentiment",
        "num_articles",
        "fear_greed",
        "fear_greed_correlation",
    ]
    for col in zero_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df


def mc_dropout_predict(model, x, n_samples=25):
    preds = np.array([model(x, training=True).numpy() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)


def build_single_window(df, feature_cols, scaler, window_size, as_of_date):
    """
    Returns the last (window_size + 1) rows with dates <= as_of_date.
    pred_date is the latest included trading day (last bar fed to the model), not
    a forecast calendar date.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"] <= as_of_date].copy()
    if len(df) < (window_size + 1):
        return None, None, None

    window_df = df.iloc[-(window_size + 1):].copy()
    for col in feature_cols:
        if col not in window_df.columns:
            window_df[col] = 0.0
    feats = window_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    x = scaler.transform(feats)[np.newaxis, :, :].astype(np.float32)
    pred_date = window_df["date"].iloc[-1]
    close_val = float(window_df["close"].iloc[-1]) if "close" in window_df.columns else np.nan
    return x, pred_date, close_val


def main():
    parser = argparse.ArgumentParser(description="Generate live buy/no-buy signals.")
    parser.add_argument("--min-accepted", type=float, default=0.2, help="Minimum adjusted probability.")
    parser.add_argument("--std-factor", type=float, default=0.0, help="Adjusted prob = prob - std_factor * std.")
    parser.add_argument("--mc-samples", type=int, default=25, help="MC dropout samples.")
    parser.add_argument("--force-refresh", action="store_true", help="Always run downloader+processor.")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Latest calendar date to include from processed data (default: previous business day).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    SIGNALS_DIR.mkdir(exist_ok=True)

    if not model_weights_file_present(MODELS_DIR) or not SCALER_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifacts in models/. "
            "Run notebook export cell to create latest_model.keras (native zip) or "
            "latest_model.h5 (legacy), plus latest_scalar.pkl and latest_meta.json."
        )

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    window_size = int(meta["window_size"])
    horizons = meta["horizons"]

    import tensorflow as tf
    from tensorflow.keras.layers import Dropout, LSTM

    class MCDropout(Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    class LSTMCompat(LSTM):
        """Strips args removed in Keras 3 (e.g. time_major) from legacy H5 configs."""

        @classmethod
        def from_config(cls, config):
            cfg = dict(config)
            cfg.pop("time_major", None)
            return cls(**cfg)

    model_path = resolve_saved_model_path(MODELS_DIR)
    custom_objects = {"MCDropout": MCDropout}
    if model_path.suffix.lower() in (".h5", ".hdf5"):
        custom_objects["LSTM"] = LSTMCompat
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects,
    )
    scaler = joblib.load(SCALER_PATH)

    tickers = load_tickers()
    expected_date = latest_completed_trading_day()
    if args.as_of_date:
        expected_date = pd.Timestamp(args.as_of_date).normalize()
    run_date = str(pd.Timestamp.today().normalize().date())
    # Calendar day of this run — what "today's decision" means in the output CSV.
    decision_date_str = run_date
    print(
        f"[INFO] Expected last session in data files: {expected_date.date()} "
        f"(script run / decision calendar date: {run_date}). "
        "Scores use date=decision day; as_of_close_date=last daily bar in the model window."
    )
    if args.force_refresh or is_data_stale(tickers, expected_date):
        run_refresh_pipeline()

    all_rows = []
    candidates = []

    for ticker in tickers:
        proc_path = get_file_path_for_ticker(PROCESSED_STOCKS_DIR, ticker, "daily_processed")
        raw_path = get_file_path_for_ticker(RAW_STOCKS_DIR, ticker, "daily")
        if not proc_path.exists():
            all_rows.append(
                {
                    "run_date": run_date,
                    "date": decision_date_str,
                    "as_of_close_date": "",
                    "ticker": ticker,
                    "status": "missing_processed",
                    "best_horizon": "",
                    "best_pred_prob": np.nan,
                    "best_pred_std": np.nan,
                    "best_adj_prob": np.nan,
                }
            )
            continue

        try:
            df = pd.read_csv(proc_path, parse_dates=["date"])
            if df.empty:
                raise ValueError("empty processed file")

            df = ensure_optional_zero_defaults(df)
            raw_max = safe_max_date(raw_path, "date")
            proc_max = pd.to_datetime(df["date"]).dt.normalize().max()
            if raw_max is not None and proc_max < raw_max:
                print(
                    f"[WARN] {ticker}: processed data ends {proc_max.date()} but raw ends "
                    f"{raw_max.date()}; run TrainingData/processor.py so features match raw."
                )
                all_rows.append(
                    {
                        "run_date": run_date,
                        "date": decision_date_str,
                        "as_of_close_date": str(proc_max.date()),
                        "ticker": ticker,
                        "status": "processed_stale_vs_raw",
                        "best_horizon": "",
                        "best_pred_prob": np.nan,
                        "best_pred_std": np.nan,
                        "best_adj_prob": np.nan,
                    }
                )
                continue

            x, pred_date, close_val = build_single_window(
                df, feature_cols, scaler, window_size, expected_date
            )

            if x is None:
                all_rows.append(
                    {
                        "run_date": run_date,
                        "date": decision_date_str,
                        "as_of_close_date": "",
                        "ticker": ticker,
                        "status": "insufficient_window",
                        "best_horizon": "",
                        "best_pred_prob": np.nan,
                        "best_pred_std": np.nan,
                        "best_adj_prob": np.nan,
                    }
                )
                continue

            mean_pred, std_pred = mc_dropout_predict(model, x, n_samples=args.mc_samples)
            mean_pred = mean_pred[0]
            std_pred = std_pred[0]

            best = None
            for i, horizon in enumerate(horizons):
                pred_prob = float(mean_pred[i])
                pred_std = float(max(std_pred[i], 1e-8))
                adj_prob = pred_prob - args.std_factor * pred_std
                row = {
                    "run_date": run_date,
                    "date": decision_date_str,
                    "as_of_close_date": str(pd.Timestamp(pred_date).date()),
                    "ticker": ticker,
                    "close": close_val,
                    "horizon": horizon,
                    "pred_prob": pred_prob,
                    "pred_std": pred_std,
                    "adj_prob": adj_prob,
                }
                if best is None or row["adj_prob"] > best["adj_prob"]:
                    best = row

            status = "buy_candidate" if best["adj_prob"] > args.min_accepted else "no_buy"
            all_rows.append(
                {
                    "run_date": run_date,
                    "date": best["date"],
                    "as_of_close_date": best["as_of_close_date"],
                    "ticker": ticker,
                    "status": status,
                    "best_horizon": best["horizon"],
                    "best_pred_prob": best["pred_prob"],
                    "best_pred_std": best["pred_std"],
                    "best_adj_prob": best["adj_prob"],
                    "close": best["close"],
                }
            )
            if status == "buy_candidate":
                candidates.append(best)
        except Exception as exc:
            all_rows.append(
                {
                    "run_date": run_date,
                    "date": decision_date_str,
                    "as_of_close_date": "",
                    "ticker": ticker,
                    "status": f"error:{type(exc).__name__}",
                    "best_horizon": "",
                    "best_pred_prob": np.nan,
                    "best_pred_std": np.nan,
                    "best_adj_prob": np.nan,
                }
            )

    scores_df = pd.DataFrame(all_rows).sort_values(
        ["status", "best_adj_prob"], ascending=[True, False], na_position="last"
    )
    _scores_cols = [
        "run_date",
        "date",
        "as_of_close_date",
        "ticker",
        "status",
        "best_horizon",
        "best_pred_prob",
        "best_pred_std",
        "best_adj_prob",
        "close",
    ]
    scores_df = scores_df.reindex(columns=_scores_cols)

    scores_path = SIGNALS_DIR / f"live_scores_{run_date}.csv"
    decision_path = SIGNALS_DIR / f"live_decision_{run_date}.csv"
    scores_df.to_csv(scores_path, index=False)

    if not candidates:
        decision_df = pd.DataFrame(
            [
                {
                    "run_date": run_date,
                    "date": decision_date_str,
                    "as_of_close_date": "",
                    "decision": "NO_BUY",
                    "reason": f"No ticker above min_accepted={args.min_accepted}",
                    "ticker": "",
                    "horizon": "",
                    "adj_prob": np.nan,
                    "pred_prob": np.nan,
                    "pred_std": np.nan,
                }
            ]
        )
    else:
        best_global = max(candidates, key=lambda row: row["adj_prob"])
        decision_df = pd.DataFrame(
            [
                {
                    "run_date": run_date,
                    "date": decision_date_str,
                    "as_of_close_date": best_global.get("as_of_close_date", ""),
                    "decision": "BUY",
                    "reason": f"Best adjusted probability above threshold {args.min_accepted}",
                    "ticker": best_global["ticker"],
                    "horizon": best_global["horizon"],
                    "adj_prob": best_global["adj_prob"],
                    "pred_prob": best_global["pred_prob"],
                    "pred_std": best_global["pred_std"],
                }
            ]
        )

    decision_df.to_csv(decision_path, index=False)
    print(f"[INFO] Wrote scores: {scores_path}")
    print(f"[INFO] Wrote decision: {decision_path}")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()