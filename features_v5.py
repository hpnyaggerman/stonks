"""Single source of truth for the model input matrix, shared by trainer and scorer.

Both training and live scoring build their features here, so the two cannot drift:
the same code computes every channel, the same normalization is applied, and the
output passes through one fp16 round-trip so a value seen at training time is bit
-identical to the value seen at scoring time.

Inputs:

* **OHLCV** — daily candles from the Tiingo parquet shards
  (``TrainingData/ohlcv_parts/*.parquet``, long-format, US exchanges only). The feed
  is split-unadjusted and carries no split-factor column, so a heuristic spike guard
  protects the labels (see :func:`ewma_sigma_hat` and :func:`label_mask`).
* **Fear & Greed** — one market-wide CNN series joined on date.

Every price-derived indicator is recomputed here from OHLCV rather than read from a
preprocessed CSV; :func:`raw_indicators` reproduces the indicator suite the rest of
the pipeline historically wrote to disk, so a value comparison isolates any
reimplementation drift from the change of price vendor.

The 40 channels are all stationary and scale-free — log-returns, ratio gaps, bounded
oscillators, rolling z-scores, periodic calendar phases, and a validity flag — so no
input is monotone in time or proxies the era a row came from.
"""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PARTS_DIR = PROJECT_ROOT / "TrainingData" / "ohlcv_parts"
FEAR_GREED_PATH = PROJECT_ROOT / "TrainingData" / "indicators_data" / "raw" / "fear_greed.csv"

# Tiingo exchange labels that are US listings. "NYSE MKT" is NYSE American (the former
# AMEX), "NYSE NAT" is NYSE National, "NYSE ARCA" is the ARCA venue.
US_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "BATS", "NYSE ARCA", "NYSE MKT", "NYSE NAT")

HORIZON_DAYS = (1, 5, 21, 126)              # trading-day counts for the four heads
SPIKE_LOG_THRESHOLD = float(np.log(1.5))    # |daily close log-return| above this = split print
EWMA_LAMBDA = 0.94                          # RiskMetrics decay for daily variance
EWMA_SEED_WINDOW = 20                       # rows of non-spike returns to seed the EWMA
EWMA_SIGMA_FLOOR = 0.005                    # floor on daily sigma_hat (50 bp)
EWMA_RESEED_GAP_DAYS = 90                   # calendar gap that resets the EWMA seed
# Calendar-day ceiling on the anchor -> target span per horizon: ceil(1.5*d) + 4.
# The +4 floor keeps every normal week; a "d-row" label across a longer hole is not
# a d-day return (halts, recycled symbols) and dies in the mask.
GAP_LIMIT_DAYS = tuple(int(np.ceil(1.5 * d)) + 4 for d in HORIZON_DAYS)
PHANTOM_MIN_NAMES = 3                       # census floor below which a session is phantom
CENSUS_PATH = PROJECT_ROOT / "TrainingData" / "session_census.csv"


@dataclass(frozen=True)
class FeatureChannel:
    """One ordered input channel.

    ``scaled`` marks the pooled-robust-z channels; the rest are already bounded or
    affine when the builder emits them and the scaler passes them through unchanged.
    ``warm_up`` is the number of leading rows the channel cannot be computed for;
    ``lag`` is a causal shift (always zero for the base channels, reserved for feeds).
    """

    name: str
    source_cols: tuple
    transform: str
    warm_up: int
    scaled: bool
    lag: int = 0


# Channel order is the model's input order and is frozen by the name hash in the
# normalization file. New channels are inert until appended here.
FEATURE_SPEC = (
    FeatureChannel("r_open", ("open",), "log_return", 1, True),
    FeatureChannel("r_high", ("high",), "log_return", 1, True),
    FeatureChannel("r_low", ("low",), "log_return", 1, True),
    FeatureChannel("r_close", ("close",), "log_return", 1, True),
    FeatureChannel("r_vol", ("volume",), "log_return_clip1", 1, True),
    FeatureChannel("gap_ma10", ("close",), "log_ratio_ma10", 10, True),
    FeatureChannel("gap_ma20", ("close",), "log_ratio_ma20", 20, True),
    FeatureChannel("gap_ma30", ("close",), "log_ratio_ma30", 30, True),
    FeatureChannel("gap_ema10", ("close",), "log_ratio_ema10", 10, True),
    FeatureChannel("gap_ema30", ("close",), "log_ratio_ema30", 30, True),
    FeatureChannel("boll_pos", ("close",), "bollinger_position", 20, True),
    FeatureChannel("boll_width", ("close",), "bollinger_width", 20, True),
    FeatureChannel("macd_rel", ("close",), "macd_over_close", 26, True),
    FeatureChannel("macd_sig_rel", ("close",), "macd_signal_over_close", 34, True),
    FeatureChannel("rsi", ("close",), "rsi_affine", 14, False),
    FeatureChannel("zscore", ("close",), "rolling_close_z", 20, False),
    FeatureChannel("Volatility_10", ("close",), "pctchange_std_10", 11, True),
    FeatureChannel("Volatility_20", ("close",), "pctchange_std_20", 21, True),
    FeatureChannel("Volatility_30", ("close",), "pctchange_std_30", 31, True),
    FeatureChannel("volatility_5d", ("close",), "pctchange_std_5_ann", 6, True),
    FeatureChannel("volatility_20d", ("close",), "pctchange_std_20_ann", 21, True),
    FeatureChannel("log_sigma_hat", ("close",), "log_ewma_sigma", EWMA_SEED_WINDOW, True),
    FeatureChannel("vol_z", ("volume",), "log_volume_z60", 60, True),
    FeatureChannel("fg", ("fear_greed",), "fear_greed_affine", 0, False),
    FeatureChannel("fg_corr", ("close", "fear_greed"), "fg_corr_126", 126, False),
    FeatureChannel("overnight_gap", ("open", "close"), "overnight_gap", 1, True),
    FeatureChannel("abnormal_vol", ("volume",), "abnormal_vol_z20", 20, True),
    FeatureChannel("momentum_5d", ("close",), "momentum_5", 5, True),
    FeatureChannel("momentum_20d", ("close",), "momentum_20", 20, True),
    FeatureChannel("skew_5d", ("close",), "pctchange_skew_5", 6, True),
    FeatureChannel("intraday_range", ("high", "low", "close"), "intraday_range", 0, True),
    FeatureChannel("sin_week", ("date",), "sin_week", 0, False),
    FeatureChannel("cos_week", ("date",), "cos_week", 0, False),
    FeatureChannel("sin_month", ("date",), "sin_month", 0, False),
    FeatureChannel("cos_month", ("date",), "cos_month", 0, False),
    FeatureChannel("sin_year", ("date",), "sin_year", 0, False),
    FeatureChannel("cos_year", ("date",), "cos_year", 0, False),
    FeatureChannel("sin_year4", ("date",), "sin_year_q", 0, False),
    FeatureChannel("cos_year4", ("date",), "cos_year_q", 0, False),
    FeatureChannel("is_pad", (), "validity", 0, False),
)

FEATURE_NAMES = tuple(c.name for c in FEATURE_SPEC)
N_FEATURES = len(FEATURE_SPEC)
PAD_COL = FEATURE_NAMES.index("is_pad")
SCALED_IDX = tuple(i for i, c in enumerate(FEATURE_SPEC) if c.scaled)
# The hard floor on real rows is the largest warm_up + lag over all channels; the
# 126-row fear/greed correlation dominates. Recomputed here, never hand-set.
MIN_REAL_ROWS = max(c.warm_up + c.lag for c in FEATURE_SPEC)


def feature_spec_records():
    """Ordered list of channel dicts for the model metadata file."""
    return [
        {"name": c.name, "source_cols": list(c.source_cols), "transform": c.transform,
         "lag": c.lag, "warm_up": c.warm_up, "scaled": c.scaled}
        for c in FEATURE_SPEC
    ]


def feature_names_hash():
    """Stable hash of the ordered channel names; asserted equal by metadata and
    normalization files at load so the two cannot fall out of positional sync."""
    joined = "\n".join(FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


# --------------------------------------------------------------------------- IO

def load_fear_greed(path=FEAR_GREED_PATH):
    """Market-wide Fear & Greed series as ``date, fear_greed``, dates normalized.

    The archive mixes ``M/D/YYYY`` and ISO date formats; mixed-format parsing keeps
    both blocks instead of coercing one to NaT.
    """
    df = pd.read_csv(path)
    try:
        parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = df.assign(date=parsed).dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df[["date", "fear_greed"]].reset_index(drop=True)


def _read_parts_filtered(tickers, exchanges, parts_dir):
    import pyarrow.parquet as pq

    filt = [("exchange", "in", list(exchanges))]
    if tickers is not None:
        filt.append(("ticker", "in", list(tickers)))
    frames = []
    for part in sorted(glob.glob(str(Path(parts_dir) / "*.parquet"))):
        tbl = pq.read_table(part, filters=filt)
        if tbl.num_rows:
            frames.append(tbl.to_pandas())
    if not frames:
        return pd.DataFrame(columns=["ticker", "exchange", "date", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


def list_us_tickers(exchanges=US_EXCHANGES, parts_dir=PARTS_DIR):
    """Sorted unique US ticker symbols present in the parquet shards."""
    import pyarrow.parquet as pq

    found = set()
    for part in sorted(glob.glob(str(Path(parts_dir) / "*.parquet"))):
        tbl = pq.read_table(part, columns=["ticker", "exchange"],
                            filters=[("exchange", "in", list(exchanges))])
        found.update(tbl.column("ticker").to_pylist())
    return sorted(found)


_census_cache = {}


def load_session_census(path=CENSUS_PATH):
    """Phantom-session dates from the committed census: dates whose distinct-name
    count is below ``PHANTOM_MIN_NAMES``. Returns ``None`` (with a single warning)
    when the census file is absent, so small dev environments keep working. Any feed
    append must regenerate the census in the same operation, otherwise new dates
    would bypass the filter semantics."""
    key = str(Path(path))
    if key in _census_cache:
        return _census_cache[key]
    p = Path(path)
    if not p.exists():
        print(f"[features] WARNING: session census missing at {p}; "
              "phantom-session filtering disabled")
        _census_cache[key] = None
        return None
    census = pd.read_csv(p, parse_dates=["date"])
    phantom = census.loc[census["n_names"] < PHANTOM_MIN_NAMES, "date"]
    _census_cache[key] = set(phantom.dt.normalize())
    return _census_cache[key]


def load_us_ohlcv(tickers=None, exchanges=US_EXCHANGES, parts_dir=PARTS_DIR):
    """Map ticker -> OHLCV frame (``date, open, high, low, close, volume``), sorted.

    Rows are filtered to US exchanges (and to ``tickers`` if given) during the parquet
    read so memory stays bounded. Duplicate dates are dropped, keeping the last.
    Phantom sessions (census-listed dates with fewer than ``PHANTOM_MIN_NAMES``
    distinct names: pre-1976 holidays, weekend rows, singleton prints) are dropped;
    dates absent from the census (newer than its end) pass unconditionally.
    """
    raw = _read_parts_filtered(tickers, exchanges, parts_dir)
    if raw.empty:
        return {}
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw = raw.dropna(subset=["date"])
    phantom = load_session_census(CENSUS_PATH)
    if phantom:
        raw = raw[~raw["date"].isin(phantom)]
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    out = {}
    for ticker, grp in raw.groupby("ticker", sort=True):
        grp = grp.sort_values("date").drop_duplicates("date", keep="last")
        out[str(ticker)] = grp[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return out


# ------------------------------------------------------------- raw indicators

def _rolling_close_z(close, window=20):
    """Z-score of close against its own trailing ``window`` values, computed
    independently per window (two-pass mean / sum of squared deviations on the raw
    slice).

    Pandas' rolling kernel carries a running accumulator across window steps whose
    drift can report an exactly-zero std on windows with real spread; dividing by
    the historical 1e-10 floor then manufactured |z| ~ 1e9, which the fp16 memmap
    cast overflowed to +-inf. Per-window computation has no cross-window state, so
    that artifact class cannot occur. Because the current close is a member of its
    own window, Samuelson's inequality bounds the true value at
    ``(window - 1) / sqrt(window)`` (~4.25 for window 20), fp16-safe by theorem.
    Bit-constant windows are the structural 0/0 (numerator exactly zero) and emit
    the unique deviation-free value 0. Leading ``window - 1`` rows are NaN, the
    warm-up the channel spec declares.
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    z = np.full(n, np.nan)
    if n < window:
        return z
    win = np.lib.stride_tricks.sliding_window_view(close, window)
    m = win.mean(axis=1)
    s = np.sqrt(((win - m[:, None]) ** 2).sum(axis=1) / (window - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = (close[window - 1:] - m) / s
    zz[win.max(axis=1) == win.min(axis=1)] = 0.0
    z[window - 1:] = zz
    return z


def raw_indicators(df):
    """Recompute the price-derived indicator suite from OHLCV.

    Column names and formulas match the pipeline's historical preprocessing output
    so the two can be compared value-for-value, with one deliberate deviation:
    ``ZScore`` comes from :func:`_rolling_close_z` and differs from the legacy
    rolling kernel exactly where that kernel's std degenerates. ``df`` must have a
    clean integer index and columns ``date, open, high, low, close, volume``.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    out = pd.DataFrame({"date": df["date"], "close": df["close"]})

    out["YesterdayOpenLogR"] = np.log(df["open"] / df["open"].shift(1))
    out["YesterdayHighLogR"] = np.log(df["high"] / df["high"].shift(1))
    out["YesterdayLowLogR"] = np.log(df["low"] / df["low"].shift(1))
    _vol = df["volume"].clip(lower=1.0)
    out["YesterdayVolumeLogR"] = np.log(_vol / _vol.shift(1))
    out["YesterdayCloseLogR"] = np.log(df["close"] / df["close"].shift(1))

    out["MA10"] = df["close"].rolling(10).mean()
    out["MA20"] = df["close"].rolling(20).mean()
    out["MA30"] = df["close"].rolling(30).mean()
    out["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    out["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan).fillna(1e-10)
    out["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    out["BollingerUpper"] = ma20 + 2 * std20
    out["BollingerLower"] = ma20 - 2 * std20

    out["Volatility_10"] = df["close"].pct_change().rolling(10).std()
    out["Volatility_20"] = df["close"].pct_change().rolling(20).std()
    out["Volatility_30"] = df["close"].pct_change().rolling(30).std()

    out["ZScore"] = _rolling_close_z(df["close"].to_numpy(dtype=np.float64))

    out["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    rolling_vol = df["volume"].rolling(20)
    vol_std = rolling_vol.std().replace(0, np.nan).fillna(1e-10)
    out["abnormal_vol"] = (df["volume"] - rolling_vol.mean()) / vol_std
    out["volatility_5d"] = df["close"].pct_change().rolling(5).std() * np.sqrt(252)
    out["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["momentum_5d"] = df["close"] / df["close"].shift(5) - 1
    out["momentum_20d"] = df["close"] / df["close"].shift(20) - 1
    out["skew_5d"] = df["close"].pct_change().rolling(5).skew()
    out["intraday_range"] = (df["high"] - df["low"]) / df["close"]
    return out


# ----------------------------------------------------------------- volatility

def ewma_sigma_hat(close, dates=None, reseed_gap_days=EWMA_RESEED_GAP_DAYS):
    """Causal RiskMetrics daily volatility with split-spike exclusion and gap re-seed.

    Returns ``(sigma_hat, log_return, spike)`` aligned to ``close`` (length N).

    ``sigma_hat`` is seeded at the row where ``EWMA_SEED_WINDOW`` non-spike returns
    have accumulated, using their sample variance, then updated as
    ``sigma2_t = 0.94 sigma2_{t-1} + 0.06 r_t^2`` and floored at ``EWMA_SIGMA_FLOOR``.
    A split-print return (``|r_t| > ln(1.5)``) is excluded from both the seed and the
    update (``sigma2_t = sigma2_{t-1}``), so a single split candle cannot inflate the
    volatility used to normalize weeks of labels. Pre-seed rows are ``NaN``.

    When ``dates`` is given and the calendar gap from the previous row exceeds
    ``reseed_gap_days``, the seed state resets: the next ``EWMA_SEED_WINDOW``
    non-spike returns re-seed the variance, the first post-gap return (which spans
    the halt, not a trading day) is excluded from the new seed, and the pre-re-seed
    rows carry ``NaN`` so their labels die via the mask's finiteness term. Without
    the reset, the first post-halt anchors would normalize labels with pre-halt
    volatility.
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    log_return = np.full(n, np.nan)
    log_return[1:] = np.log(close[1:] / close[:-1])
    spike = np.abs(np.nan_to_num(log_return, nan=0.0)) > SPIKE_LOG_THRESHOLD

    gap_reset = np.zeros(n, dtype=bool)
    if dates is not None:
        day = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[D]").astype(np.int64)
        gap_reset[1:] = (day[1:] - day[:-1]) > reseed_gap_days

    sigma_hat = np.full(n, np.nan)
    floor2 = EWMA_SIGMA_FLOOR ** 2
    seed_vals, seed_done, sigma2 = [], False, np.nan
    for t in range(1, n):
        if gap_reset[t]:
            seed_vals, seed_done, sigma2 = [], False, np.nan
            continue                     # the halt-spanning return never enters the seed
        if not np.isfinite(log_return[t]):
            if seed_done:
                sigma_hat[t] = np.sqrt(max(sigma2, floor2))
            continue
        if not seed_done:
            if not spike[t]:
                seed_vals.append(log_return[t])
                if len(seed_vals) == EWMA_SEED_WINDOW:
                    sigma2 = float(np.var(seed_vals, ddof=1))
                    sigma2 = max(sigma2, floor2)
                    seed_done = True
                    sigma_hat[t] = np.sqrt(sigma2)
            continue
        if not spike[t]:
            sigma2 = EWMA_LAMBDA * sigma2 + (1 - EWMA_LAMBDA) * log_return[t] ** 2
            sigma2 = max(sigma2, floor2)
        sigma_hat[t] = np.sqrt(sigma2)
    return sigma_hat, log_return, spike


# ------------------------------------------------------------------- calendar

def _calendar_channels(dates):
    """Eight periodic phase channels (week, month, year, quarter) in [-1, 1].

    Quarter phase is the 4th annual harmonic, supplying the earnings-cycle period the
    annual pair cannot represent through a linear stem.
    """
    d = pd.DatetimeIndex(dates)
    w = d.weekday.to_numpy(dtype=np.float64)
    phi_week = 2 * np.pi * w / 5.0
    m = d.day.to_numpy(dtype=np.float64)
    days_in_month = d.days_in_month.to_numpy(dtype=np.float64)
    phi_month = 2 * np.pi * (m - 1) / days_in_month
    y = d.dayofyear.to_numpy(dtype=np.float64)
    days_in_year = np.where(d.is_leap_year, 366.0, 365.0)
    phi_year = 2 * np.pi * (y - 1) / days_in_year
    phi_year4 = 2 * np.pi * 4 * (y - 1) / days_in_year
    return {
        "sin_week": np.sin(phi_week), "cos_week": np.cos(phi_week),
        "sin_month": np.sin(phi_month), "cos_month": np.cos(phi_month),
        "sin_year": np.sin(phi_year), "cos_year": np.cos(phi_year),
        "sin_year4": np.sin(phi_year4), "cos_year4": np.cos(phi_year4),
    }


# -------------------------------------------------------------- feature frame

def _fg_corr(close, fear_greed, window=126, min_obs=60):
    """126-day rolling correlation of close log-returns against the Fear & Greed
    series, with the leading window masked. Already bounded to [-1, 1]."""
    close = pd.Series(pd.to_numeric(close, errors="coerce"))
    fg = pd.Series(pd.to_numeric(fear_greed, errors="coerce"))
    log_return = np.log(close).diff()
    corr = fg.rolling(window=window, min_periods=min_obs).corr(log_return)
    warmup = window - 1
    if len(corr) > warmup:
        corr.iloc[:warmup] = np.nan
    return corr.replace([np.inf, -np.inf], np.nan).to_numpy()


@dataclass
class FeatureFrame:
    """Per-row builder output for one ticker, before windowing and normalization.

    ``features`` is ``(N, N_FEATURES)`` unnormalized with ``is_pad`` zeroed; ``z`` is
    ``(N, 4)`` vol-normalized targets (``NaN`` where unavailable); ``spike_free`` and
    ``target_dates`` feed the label mask; ``close`` and ``dates`` index the rows.
    ``volume`` is the raw print; ``tradable`` is the causal candidate-gate flag
    (close >= $5 and 63-session rolling median dollar volume >= $1M); ``vol_med63``
    is the causal 63-session rolling median of raw volume (the split detector's
    corroboration reference, NaN where unavailable). Computed once here so trainer,
    live scorer, and backtest cannot drift.
    """

    ticker: str
    dates: np.ndarray
    close: np.ndarray
    features: np.ndarray
    z: np.ndarray
    spike_free: np.ndarray
    target_dates: np.ndarray
    sigma_hat: np.ndarray
    volume: np.ndarray
    tradable: np.ndarray
    vol_med63: np.ndarray


def build_feature_frame(ticker, ohlcv, fear_greed):
    """Compute every channel and the vol-normalized targets for one ticker."""
    df = ohlcv.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    n = len(df)
    ind = raw_indicators(df)
    close = df["close"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy()
    volume = df["volume"].to_numpy(dtype=np.float64)

    close_s = pd.Series(close)
    dollar_med63 = (close_s * pd.Series(volume)).rolling(63, min_periods=63).median()
    tradable = ((close_s >= 5.0) & (dollar_med63 >= 1e6)).fillna(False).to_numpy(dtype=bool)
    vol_med63 = pd.Series(volume).rolling(63, min_periods=63).median().to_numpy(dtype=np.float64)

    fg_merged = df[["date"]].merge(fear_greed, on="date", how="left")
    fear = fg_merged["fear_greed"].ffill().to_numpy(dtype=np.float64)

    sigma_hat, _, spike = ewma_sigma_hat(close, dates)

    with np.errstate(divide="ignore", invalid="ignore"):
        ch = {
            "r_open": ind["YesterdayOpenLogR"].to_numpy(),
            "r_high": ind["YesterdayHighLogR"].to_numpy(),
            "r_low": ind["YesterdayLowLogR"].to_numpy(),
            "r_close": ind["YesterdayCloseLogR"].to_numpy(),
            "r_vol": ind["YesterdayVolumeLogR"].to_numpy(),
            "gap_ma10": np.log(close / ind["MA10"].to_numpy()),
            "gap_ma20": np.log(close / ind["MA20"].to_numpy()),
            "gap_ma30": np.log(close / ind["MA30"].to_numpy()),
            "gap_ema10": np.log(close / ind["EMA10"].to_numpy()),
            "gap_ema30": np.log(close / ind["EMA30"].to_numpy()),
            "boll_pos": ((close - ind["BollingerLower"].to_numpy())
                         / (ind["BollingerUpper"].to_numpy() - ind["BollingerLower"].to_numpy() + 1e-9)),
            "boll_width": ((ind["BollingerUpper"].to_numpy() - ind["BollingerLower"].to_numpy())
                           / ind["MA20"].to_numpy()),
            "macd_rel": ind["MACD"].to_numpy() / close,
            "macd_sig_rel": ind["MACD_Signal"].to_numpy() / close,
            "rsi": ind["RSI"].to_numpy() / 50.0 - 1.0,
            "zscore": ind["ZScore"].to_numpy(),
            "Volatility_10": ind["Volatility_10"].to_numpy(),
            "Volatility_20": ind["Volatility_20"].to_numpy(),
            "Volatility_30": ind["Volatility_30"].to_numpy(),
            "volatility_5d": ind["volatility_5d"].to_numpy(),
            "volatility_20d": ind["volatility_20d"].to_numpy(),
            "log_sigma_hat": np.log(sigma_hat),
            "vol_z": _log_volume_z(df["volume"].to_numpy(dtype=np.float64)),
            "fg": (fear - 50.0) / 50.0,
            "fg_corr": _fg_corr(close, fear),
            "overnight_gap": ind["overnight_gap"].to_numpy(),
            "abnormal_vol": ind["abnormal_vol"].to_numpy(),
            "momentum_5d": ind["momentum_5d"].to_numpy(),
            "momentum_20d": ind["momentum_20d"].to_numpy(),
            "skew_5d": ind["skew_5d"].to_numpy(),
            "intraday_range": ind["intraday_range"].to_numpy(),
        }
    ch.update(_calendar_channels(dates))
    ch["is_pad"] = np.zeros(n, dtype=np.float64)

    features = np.empty((n, N_FEATURES), dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        features[:, j] = ch[name]

    # zscore is unscaled and skips the robust-z clip; the per-window computation is
    # Samuelson-bounded at (w-1)/sqrt(w) ~ 4.25, so anything larger means the exact
    # builder regressed toward the fp16-overflow class.
    zs = ch["zscore"]
    zs = zs[np.isfinite(zs)]
    if zs.size and float(np.max(np.abs(zs))) > 5.0:
        raise ValueError(f"{ticker}: |zscore| max {np.max(np.abs(zs)):.3g} exceeds "
                         "the Samuelson bound for a same-window z-score")

    z, spike_free, target_dates = _targets(close, dates, sigma_hat, spike)
    return FeatureFrame(ticker, dates, close, features, z, spike_free, target_dates,
                        sigma_hat, volume, tradable, vol_med63)


def _log_volume_z(volume, window=60):
    lnv = pd.Series(np.log(np.clip(volume, 1.0, None)))
    z = (lnv - lnv.rolling(window).mean()) / lnv.rolling(window).std()
    return z.to_numpy()


def _targets(close, dates, sigma_hat, spike):
    """Vol-normalized forward log-returns, the spike-free flag, and target dates.

    Labels are anchored to next-close entry (the earliest interval a signal computed
    from close(t) can actually hold): entry index ``t+1``, target index ``t+1+d_h``,
    ``z[t, h] = ln(C_{t+1+d_h} / C_{t+1}) / (sigma_hat_t * sqrt(d_h))``. The
    denominator stays causal at the anchor ``t``. ``z`` is ``NaN`` when the horizon
    runs off the end of the series or ``sigma_hat_t`` is undefined;
    ``target_dates[t, h] = dates[t+1+d_h]``. ``spike_free[t, h]`` is true when no
    split print falls in ``(t+1, t+1+d_h]``: a print at ``t+1`` must NOT censor (both
    entry and target closes sit on the post-split basis, so it cancels out of the
    ratio), while a print at ``t+1+d_h`` must.
    """
    n = close.size
    h = len(HORIZON_DAYS)
    z = np.full((n, h), np.nan, dtype=np.float64)
    spike_free = np.zeros((n, h), dtype=bool)
    target_dates = np.full((n, h), np.datetime64("NaT"), dtype="datetime64[ns]")
    cum_spike = np.concatenate([[0], np.cumsum(spike.astype(np.int64))])  # length n+1
    logc = np.log(close)
    for hi, d in enumerate(HORIZON_DAYS):
        valid = np.arange(n) + 1 + d < n
        idx = np.where(valid)[0]                  # last valid anchor: n - d - 2
        entry = idx + 1
        tgt = idx + 1 + d
        y = logc[tgt] - logc[entry]
        denom = sigma_hat[idx] * np.sqrt(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[idx, hi] = y / denom
        target_dates[idx, hi] = dates[tgt]
        # spikes strictly after t+1 through t+1+d: cum_spike[t+d+2] - cum_spike[t+2]
        spike_free[idx, hi] = (cum_spike[tgt + 1] - cum_spike[entry + 1]) == 0
    return z.astype(np.float32), spike_free, target_dates


def label_mask(z, spike_free, target_dates, anchor_dates, split_end_dates,
               horizon_days=HORIZON_DAYS):
    """Per-horizon training mask: target exists, z is finite, the label is spike-free,
    the anchor -> target span fits the horizon's calendar-gap ceiling, and the target
    lands on or before the row's split boundary (the embargo against labels that peek
    past the split).

    The finiteness term closes the channel where a NaN label would otherwise train as
    a full-weight z = 0 target downstream (load-bearing once the EWMA re-seed
    introduces NaN sigma_hat mid-series). The gap term breaks mask monotonicity by
    design: a post-anchor halt can kill 1d while 6m survives.

    ``split_end_dates`` is the split-end date applicable to each row. The comparison
    ``target_dates <= split_end`` is the single leakage-critical decision and is unit
    tested directly.
    """
    has_future = ~np.isnat(target_dates)
    finite = np.isfinite(z)
    anchor_days = np.asarray(anchor_dates, dtype="datetime64[ns]").astype("datetime64[D]")
    target_days = target_dates.astype("datetime64[D]")
    gap_limits = np.asarray([int(np.ceil(1.5 * d)) + 4 for d in horizon_days])
    with np.errstate(invalid="ignore"):
        span = (target_days.astype(np.int64)
                - anchor_days.astype(np.int64)[:, None])          # garbage on NaT rows,
    within_gap = span <= gap_limits[None, :]                      # ANDed out by has_future
    within_split = target_dates <= np.asarray(split_end_dates, dtype="datetime64[ns]")[:, None]
    return (has_future & finite & spike_free & within_gap & within_split).astype(np.float32)


# ---------------------------------------------------------------- windowing

def assemble_window(norm_features, anchor, window, min_real_rows):
    """Left-padded ``(window, N_FEATURES)`` window ending at ``anchor`` inclusive.

    Returns ``None`` when fewer than ``min_real_rows`` real rows precede the anchor.
    Pad rows are zeros with ``is_pad = 1``; real rows keep their ``is_pad = 0``.
    """
    start = anchor - window + 1
    real = norm_features[max(0, start):anchor + 1]
    n_real = real.shape[0]
    if n_real < min_real_rows:
        return None
    if n_real == window:
        return real
    pad = np.zeros((window - n_real, norm_features.shape[1]), dtype=norm_features.dtype)
    pad[:, PAD_COL] = 1.0
    return np.concatenate([pad, real], axis=0)


# --------------------------------------------------------------- normalization

def _robust_params(col):
    """Median and positive scale for one channel, with the degenerate-IQR fallback.

    Default scale is ``IQR / 1.349``. When the IQR is zero (a channel that is zero on
    most rows), fall back to ``1.4826 * MAD`` over nonzero entries, then the std over
    nonzero entries, then ``1.0`` for a constant channel. The rule used is returned so
    it can be recorded and audited.
    """
    col = col[np.isfinite(col)]
    if col.size == 0:
        return 0.0, 1.0, "empty"
    med = float(np.median(col))
    q1, q3 = np.percentile(col, [25, 75])
    s = (q3 - q1) / 1.349
    if s > 0:
        return med, float(s), "iqr"
    nz = col[col != 0]
    if nz.size:
        mad = float(np.median(np.abs(nz - np.median(nz))) * 1.4826)
        if mad > 0:
            return med, mad, "mad_nonzero"
        sd = float(np.std(nz))
        if sd > 0:
            return med, sd, "std_nonzero"
    return med, 1.0, "degenerate"


class RobustScaler:
    """Pooled robust-z normalization with a frozen, self-verifying spec.

    Robust-z channels are centered on the pooled median and scaled by a positive
    robust spread, then clipped to ``[-10, 10]``. Channels marked unscaled are emitted
    already bounded by the builder and pass through unchanged. After scaling, residual
    non-finite values become ``0`` (the robust median). Output is round-tripped through
    fp16 so trainer and scorer observe bit-identical inputs.
    """

    def __init__(self, medians, scales, rules, names_hash):
        self.medians = np.asarray(medians, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.rules = list(rules)
        self.names_hash = names_hash
        assert np.all(self.scales > 0), "every channel scale must be positive"

    @classmethod
    def fit(cls, rows):
        """Fit on a 2-D array of real training rows (``(R, N_FEATURES)``)."""
        rows = np.asarray(rows, dtype=np.float64)
        medians = np.zeros(N_FEATURES)
        scales = np.ones(N_FEATURES)
        rules = ["identity"] * N_FEATURES
        for j in SCALED_IDX:
            medians[j], scales[j], rules[j] = _robust_params(rows[:, j])
        return cls(medians, scales, rules, feature_names_hash())

    def transform(self, features):
        out = np.asarray(features, dtype=np.float32).copy()
        out[~np.isfinite(out)] = np.nan
        for j in SCALED_IDX:
            out[:, j] = (out[:, j] - self.medians[j]) / self.scales[j]
            np.clip(out[:, j], -10.0, 10.0, out=out[:, j])
        out = np.nan_to_num(out, nan=0.0)
        out16 = out.astype(np.float16)
        # Inputs are finite here (scrubbed above), so a nonfinite fp16 value is a
        # finite float32 beyond the fp16 range (65504) on an unscaled channel --
        # refuse to emit a poisoned row rather than let training consume it.
        bad = ~np.isfinite(out16)
        if bad.any():
            names = sorted({FEATURE_NAMES[j] for j in np.nonzero(bad)[1]})
            raise ValueError(f"fp16 overflow in channel(s) {names}: "
                             f"{int(bad.sum())} values exceed the fp16 range")
        return out16.astype(np.float32)

    def to_dict(self):
        return {
            "feature_names_hash": self.names_hash,
            "channels": [
                {"name": FEATURE_NAMES[j], "median": float(self.medians[j]),
                 "scale": float(self.scales[j]), "rule": self.rules[j]}
                for j in range(N_FEATURES)
            ],
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        chans = data["channels"]
        names = [c["name"] for c in chans]
        if tuple(names) != FEATURE_NAMES:
            raise ValueError("normalization channel order does not match the feature spec")
        medians = [c["median"] for c in chans]
        scales = [c["scale"] for c in chans]
        rules = [c["rule"] for c in chans]
        scaler = cls(medians, scales, rules, data["feature_names_hash"])
        if scaler.names_hash != feature_names_hash():
            raise ValueError("normalization feature-name hash does not match the builder")
        return scaler

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
