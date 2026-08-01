"""Train the deep-ensemble return-prediction backbone and export all artifacts.

Pipeline:

1. Select a US ticker universe and build per-ticker feature frames (optionally
   truncated to ``--data-start`` / ``--data-end`` BEFORE the feature build, so a
   rolling-origin run computes labels and sigma as if the feed ended then).
2. Resolve splits along the active holdout axes (time, ticker, or both; at least one
   is required) and the per-row split boundaries used for the label embargo.
3. Center labels cross-sectionally (subtract the per-session median of eligible
   pool labels), fit the pooled robust-z scaler on training rows only, normalize
   every ticker, and write one concatenated fp16 memmap plus a per-ticker index.
4. Train ``M`` ensemble members (different seeds and shuffles) with a scheduled
   evaluation cadence and rank-IC-driven early stopping on the stopping blocks,
   then fit per-horizon calibration temperatures on the stopping-side validation
   labels and derive the per-horizon live score floors.
5. Write the model checkpoints, the feature/normalization/config metadata, the run
   manifest, the split description (with per-horizon embargo dates and the eval
   mode), the eval-surface manifest, and one forecast CSV per ticker.

The defaults train a production-sized model; ``--smoke`` shrinks every dimension so
the full pipeline runs on a CPU in seconds for verification.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import features_v5 as fx
from features_v5 import HORIZON_DAYS, MIN_REAL_ROWS, N_FEATURES
from v5 import metrics as mx
from v5_backbone import (V5Backbone, V5Config, ensemble_predict, hl_gauss_targets,
                         optimizer_param_groups, v5_loss, v5_loss_components)
from v5.forecast import HORIZON_LABELS, build_forecast_columns

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
V5_MODEL_DIR = MODELS_DIR / "v5"
RUNS_DIR = V5_MODEL_DIR / "runs"
FORECAST_DIR = PROJECT_ROOT / "forecasts"
CACHE_DIR = PROJECT_ROOT / "cache" / "v5"
CLASS_NAMES = ("down", "neutral", "up")
STALENESS_K = 63
DEFAULT_EVAL_DAYS = 254          # trading sessions reserved for EACH of val and oos (fixed tails)
THIN_STRIDE = {2: 4, 3: 25}      # train-mask thinning stride per horizon index (1m, 6m)
TRAIN_PRICE_FLOOR = 1.0          # sub-$1 anchors never train (tick-quantization noise)
SCORE_FLOOR_PCT = 95.0           # per-horizon live score floor percentile
ERA_EDGES = ("2008-01-01", "2016-01-01")   # half-open era partition for integrity logs

# Manifest keys that pin the deterministic split/scaler recomputation; the
# finalize snapshot assert additionally checks ``members``.
SPLIT_ARG_KEYS = ("eval_mode", "seed", "ticker_holdout_frac", "eval_days",
                  "data_start", "data_end", "window", "val_subsample",
                  "max_windows", "tickers", "max_tickers")


# ----------------------------------------------------------------- universe

def assign_eval_tickers(tickers, seed, frac):
    """Deterministic seeded ticker partition; the held-out set is reproducible from
    the seed alone. A ticker is held out iff ``sha256(seed|ticker) mod 1e4 / 1e4 < frac``."""
    if frac <= 0:
        return set()
    held = set()
    for t in tickers:
        digest = hashlib.sha256(f"{seed}|{t}".encode("utf-8")).hexdigest()
        if (int(digest, 16) % 10_000) / 10_000 < frac:
            held.add(t)
    return held


def _era_of(dates):
    """Era index per date for the integrity logs: [data_start, 2008) / [2008, 2016) /
    [2016, data_end]."""
    e0 = np.datetime64(ERA_EDGES[0])
    e1 = np.datetime64(ERA_EDGES[1])
    d = np.asarray(dates, dtype="datetime64[ns]")
    return np.where(d < e0, 0, np.where(d < e1, 1, 2))


ERA_NAMES = ("pre2008", "2008-2015", "2016+")


def build_frames(tickers, data_start=None, data_end=None):
    """Build feature frames, truncating each ticker's OHLCV BEFORE the feature build
    (truncating after would leak post-cutoff prices into z and sigma), and log the
    per-era integrity counters: sigma-floor share, sub-$1 share, per-horizon spike
    censorship of labels, plus the fear/greed coverage end vs the OHLCV data end."""
    import pandas as pd

    fear_greed = fx.load_fear_greed()
    ohlcv = fx.load_us_ohlcv(tickers=tickers)
    frames = {}
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None:
            continue
        if data_start is not None:
            df = df[df["date"] >= pd.Timestamp(data_start)]
        if data_end is not None:
            df = df[df["date"] <= pd.Timestamp(data_end)]
        if len(df) < MIN_REAL_ROWS + 1:
            continue
        frames[ticker] = fx.build_feature_frame(ticker, df.reset_index(drop=True), fear_greed)

    # Per-era integrity counters, once per run.
    n_rows = np.zeros(3, dtype=np.int64)
    n_floor = np.zeros(3, dtype=np.int64)
    n_sub1 = np.zeros(3, dtype=np.int64)
    lab_all = np.zeros((3, 4), dtype=np.int64)
    lab_cens = np.zeros((3, 4), dtype=np.int64)
    data_end_seen = None
    for f in frames.values():
        era = _era_of(f.dates)
        np.add.at(n_rows, era, 1)
        np.add.at(n_floor, era, (np.abs(f.sigma_hat - fx.EWMA_SIGMA_FLOOR) < 1e-12))
        np.add.at(n_sub1, era, f.close < 1.0)
        candidate = ~np.isnat(f.target_dates) & np.isfinite(f.z)
        for hi in range(4):
            np.add.at(lab_all[:, hi], era, candidate[:, hi])
            np.add.at(lab_cens[:, hi], era, candidate[:, hi] & ~f.spike_free[:, hi])
        last = f.dates.max()
        data_end_seen = last if data_end_seen is None else max(data_end_seen, last)
    for e, name in enumerate(ERA_NAMES):
        if n_rows[e] == 0:
            continue
        cens = "  ".join(
            f"{HORIZON_LABELS[hi]}:{lab_cens[e, hi] / max(1, lab_all[e, hi]):.4f}"
            for hi in range(4))
        print(f"[v5] integrity [{name}]: rows {n_rows[e]} | sigma-floor "
              f"{n_floor[e] / n_rows[e]:.4f} | close<$1 {n_sub1[e] / n_rows[e]:.4f} | "
              f"spike-censorship {cens}")
    if len(fear_greed) and data_end_seen is not None:
        fg_end = fear_greed["date"].max()
        print(f"[v5] fear_greed coverage ends {fg_end.date()} vs OHLCV data end "
              f"{np.datetime_as_string(np.datetime64(data_end_seen), unit='D')}"
              + (" (FROZEN-FG SPAN: refresh fear_greed.csv)" if
                 np.datetime64(fg_end) < np.datetime64(data_end_seen) else ""))
    return frames


# -------------------------------------------------------------------- splits

def era_proxy_audit(rows, date_ordinals, threshold=0.2):
    """Flag any non-calendar channel whose values are rank-correlated with calendar
    time on the training rows (a stationarity / era-proxy check). Calendar channels are
    periodic by construction and are reported but not flagged for de-trending.
    """
    from scipy.stats import spearmanr

    flagged = []
    for j, ch in enumerate(fx.FEATURE_SPEC):
        if ch.name.startswith(("sin_", "cos_")) or ch.name == "is_pad":
            continue
        v = rows[:, j]
        finite = np.isfinite(v)
        if finite.sum() < 50:
            continue
        rho = float(spearmanr(v[finite], date_ordinals[finite]).statistic)
        if np.isfinite(rho) and abs(rho) > threshold:
            flagged.append((ch.name, rho))
    return flagged


def global_date_bounds(frames, eval_days=DEFAULT_EVAL_DAYS):
    """Fixed-size chronological tails: the last ``eval_days`` unique trading sessions are the
    OOS split, the ``eval_days`` before that are validation, and every earlier session is
    training. Train therefore gets all but ``2 * eval_days`` sessions (maximum historical
    exposure), and val/oos stay adjacent in time to train instead of sitting on a far, thin
    calendar slice. Eval windows still draw their look-back from before the tail, so the
    short tails cost no warm-up.

    ``train_cut`` is the first validation session and ``train_val_cut`` the first OOS session,
    so the returned keys keep the meaning the rest of the pipeline already expects. Requires
    at least ``2 * eval_days`` unique sessions.
    """
    all_dates = np.concatenate([f.dates for f in frames.values()])
    unique = np.unique(all_dates)
    if len(unique) < 2 * eval_days:
        raise SystemExit(
            f"need >= {2 * eval_days} unique trading sessions for {eval_days}-session "
            f"val+oos tails; have {len(unique)}. Lower --eval-days or add history.")
    dmin, dmax = unique.min(), unique.max()
    train_cut = unique[-2 * eval_days]          # first validation session
    train_val_cut = unique[-eval_days]          # first OOS session (== oos_start)
    train_end = unique[unique < train_cut].max()
    val_mask = (unique >= train_cut) & (unique < train_val_cut)
    val_end = unique[val_mask].max() if val_mask.any() else train_end
    return {
        "unique_dates": unique, "data_start": dmin, "data_end": dmax,
        "train_cut": train_cut, "train_val_cut": train_val_cut,
        "train_end": train_end, "val_end": val_end,
    }


def row_splits(frame, bounds, eval_mode, is_eval_ticker, val_ticker):
    """Per-row split label and the split-end date used for the embargo.

    Split labels: 0 train, 1 val, 2 eval, -1 unused. Under a time axis the split-end
    is the last date of the row's time bucket; under ticker-only there is no time
    embargo so the split-end is the data end. Under ``both``, an eval ticker's
    val-era rows are validation (tag-1; their labels are embargoed at val_end, so
    training never sees them) and a non-eval ticker's OOS-era rows are eval -- the
    forecast/backtest universe is the full universe, not just the holdout.
    """
    n = len(frame.dates)
    split = np.full(n, -1, dtype=np.int8)
    split_end = np.full(n, bounds["data_end"], dtype="datetime64[ns]")
    is_train_time = frame.dates < bounds["train_cut"]
    is_val_time = (frame.dates >= bounds["train_cut"]) & (frame.dates < bounds["train_val_cut"])
    is_oos_time = frame.dates >= bounds["train_val_cut"]
    if eval_mode in ("time", "both"):
        split_end = np.where(is_train_time, bounds["train_end"],
                             np.where(is_val_time, bounds["val_end"], bounds["data_end"]))
    if eval_mode == "time":
        split[is_train_time], split[is_val_time], split[is_oos_time] = 0, 1, 2
    elif eval_mode == "ticker":
        if is_eval_ticker:
            split[:] = 2
        else:
            split[:] = 1 if val_ticker else 0
    else:  # both
        if is_eval_ticker:
            split[is_val_time] = 1
            split[is_oos_time] = 2
        else:
            split[is_train_time] = 0
            split[is_val_time] = 1
            split[is_oos_time] = 2
    return split, split_end


def class_rate_log(frames, bounds, cfg, eval_mode, eval_tickers, val_tickers,
                   design_freeze=False):
    """Per-split realized class rates (the threshold-drift probe), measured on the
    population training actually sees: real ``row_splits`` under the active eval
    mode, the full label mask, the enumerable-anchor floor, and the $1 train price
    floor on train rows. The thinning predicate is deliberately NOT applied
    (deterministic-rate label selection, approximately class-rate-neutral). The OOS
    row prints only under ``--design-freeze`` -- routine OOS-rate printing erodes
    the one-shot-OOS discipline."""
    half_theta = [k * cfg.bin_width for k in cfg.theta_bins]
    counts = {s: np.zeros((len(cfg.horizons), 3)) for s in ("train", "val", "oos")}
    name = {0: "train", 1: "val", 2: "oos"}
    for ticker, f in frames.items():
        split, split_end = row_splits(f, bounds, eval_mode,
                                      ticker in eval_tickers, ticker in val_tickers)
        mask = fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end)
        n = len(f.dates)
        anchor_ok = np.zeros(n, dtype=bool)
        anchor_ok[MIN_REAL_ROWS - 1:] = True
        train_ok = anchor_ok & (f.close >= TRAIN_PRICE_FLOOR)
        for hi, th in enumerate(half_theta):
            z = f.z[:, hi]
            with np.errstate(invalid="ignore"):
                cls = (z > -th).astype(int) + (z > th).astype(int)
            valid = mask[:, hi] > 0
            for s_id, s_name in name.items():
                sel = (split == s_id) & valid & (train_ok if s_id == 0 else anchor_ok)
                for c in range(3):
                    counts[s_name][hi, c] += int(((cls == c) & sel).sum())
    rates = {}
    for s, c in counts.items():
        tot = c.sum(1, keepdims=True)
        rates[s] = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    for s in ("train", "val") + (("oos",) if design_freeze else ()):
        per_h = "  ".join(f"{lbl}:{rates[s][hi].round(2).tolist()}"
                          for hi, lbl in enumerate(HORIZON_LABELS))
        print(f"[v5] class rates [{s}] (down/neutral/up): {per_h}")
    return rates


# --------------------------------------------------------- label transforms

def center_labels(frames, bounds, eval_mode, eval_tickers, k_min):
    """Subtract the per-(session, horizon) cross-sectional median of eligible
    labels from every label (market-relative target z').

    Pool per (session, horizon): labels of non-eval tickers (under ticker/both eval
    modes), at enumerable anchors only (row >= MIN_REAL_ROWS - 1; listing-era rows
    that can never train do not vote), passing the full label mask with that row's
    split_end (the embargo term prevents a gap ticker's post-boundary price from
    leaking into a median subtracted into trained labels). The $1 train floor is
    deliberately NOT a pool criterion. c = median(pool) when the pool has at least
    ``k_min`` members, else undefined: the subtraction then yields NaN, and the
    mask's finiteness term drops the label (a mixed centered/uncentered target
    would be incoherent). Applied to ALL frames including eval tickers.

    Mutates ``f.z`` in place; returns stats for the log and ``v5_meta.json``.
    """
    unique = bounds["unique_dates"]
    n_sessions = len(unique)
    H = len(HORIZON_DAYS)
    c_table = np.full((n_sessions, H), np.nan, dtype=np.float32)
    exclude_eval = eval_mode in ("ticker", "both")

    per_h_idx = [[] for _ in range(H)]
    per_h_z = [[] for _ in range(H)]
    for ticker, f in frames.items():
        if exclude_eval and ticker in eval_tickers:
            continue
        split, split_end = row_splits(f, bounds, eval_mode, False, False)
        mask = fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end)
        date_idx = np.searchsorted(unique, f.dates)
        eligible_anchor = np.zeros(len(f.dates), dtype=bool)
        eligible_anchor[MIN_REAL_ROWS - 1:] = True
        for hi in range(H):
            sel = eligible_anchor & (mask[:, hi] > 0)
            if sel.any():
                per_h_idx[hi].append(date_idx[sel].astype(np.int64))
                per_h_z[hi].append(f.z[sel, hi].astype(np.float32))

    pool_sessions = np.zeros(H, dtype=np.int64)
    for hi in range(H):
        if not per_h_idx[hi]:
            continue
        di = np.concatenate(per_h_idx[hi])
        zv = np.concatenate(per_h_z[hi])
        order = np.argsort(di, kind="stable")
        di, zv = di[order], zv[order]
        starts = np.flatnonzero(np.concatenate([[True], di[1:] != di[:-1]]))
        bounds_seg = np.concatenate([starts, [di.size]])
        for i in range(len(starts)):
            lo, hi_b = bounds_seg[i], bounds_seg[i + 1]
            if hi_b - lo >= k_min:
                c_table[di[lo], hi] = np.median(zv[lo:hi_b])
                pool_sessions[hi] += 1

    for f in frames.values():
        date_idx = np.searchsorted(unique, f.dates)
        f.z -= c_table[date_idx]        # undefined c (NaN) propagates: label masked

    masked_per_h = (n_sessions - pool_sessions).tolist()
    stats = {"method": "median", "k_min": int(k_min),
             "pool": "mask ^ anchor-floor ^ split-embargo ^ non-eval-tickers",
             "masked_dates_per_h": masked_per_h}
    print(f"[v5] label centering: K={k_min} | sessions with undefined c per horizon "
          f"{masked_per_h} of {n_sessions}")
    return stats


def label_shuffle_within_date(frames, bounds, eval_mode, eval_tickers, val_tickers, seed):
    """Null (c): permute mask-passing TRAIN labels' z' across tickers within every
    (session, horizon). Destroys any cross-sectional input-label alignment while
    preserving each date's label dispersion; the end-to-end falsifier of
    'input-side leakage: none found'."""
    unique = bounds["unique_dates"]
    rng = np.random.default_rng(seed)
    frame_list = list(frames.values())
    masks, splits = [], []
    for ticker, f in frames.items():
        split, split_end = row_splits(f, bounds, eval_mode,
                                      ticker in eval_tickers, ticker in val_tickers)
        masks.append(fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end))
        splits.append(split)
    n_shuffled = 0
    for hi in range(len(HORIZON_DAYS)):
        fidx, ridx, didx, vals = [], [], [], []
        for k, f in enumerate(frame_list):
            sel = (splits[k] == 0) & (masks[k][:, hi] > 0)
            rows = np.flatnonzero(sel)
            if rows.size:
                fidx.append(np.full(rows.size, k, dtype=np.int32))
                ridx.append(rows.astype(np.int32))
                didx.append(np.searchsorted(unique, f.dates[rows]).astype(np.int64))
                vals.append(f.z[rows, hi].copy())
        if not fidx:
            continue
        fidx = np.concatenate(fidx)
        ridx = np.concatenate(ridx)
        didx = np.concatenate(didx)
        vals = np.concatenate(vals)
        order = np.argsort(didx, kind="stable")
        fidx, ridx, didx, vals = fidx[order], ridx[order], didx[order], vals[order]
        starts = np.flatnonzero(np.concatenate([[True], didx[1:] != didx[:-1]]))
        seg = np.concatenate([starts, [didx.size]])
        for i in range(len(starts)):
            lo, hi_b = seg[i], seg[i + 1]
            if hi_b - lo > 1:
                perm = rng.permutation(hi_b - lo)
                vals[lo:hi_b] = vals[lo:hi_b][perm]
        for k in range(len(frame_list)):
            sel = fidx == k
            if sel.any():
                frame_list[k].z[ridx[sel], hi] = vals[sel]
        n_shuffled += int(didx.size)
    print(f"[v5] NULL RUN: shuffled {n_shuffled} train labels within (session, horizon) "
          f"(seed {seed})")


# ------------------------------------------------------------------- dataset

class WindowDataset(Dataset):
    """Windows sliced on demand from the shared fp16 row memmap.

    Each sample stores only the absolute memmap row of its window start and the count
    of real rows; the left-pad is applied per item so the overlapping stride-1 windows
    never have to be materialized.
    """

    def __init__(self, memmap, start_abs, n_real, z, mask, window):
        self.mm = memmap
        self.start_abs = start_abs
        self.n_real = n_real
        self.z = z
        self.mask = mask
        self.window = window

    def __len__(self):
        return len(self.start_abs)

    def __getitem__(self, i):
        lo, nr = int(self.start_abs[i]), int(self.n_real[i])
        real = np.asarray(self.mm[lo:lo + nr], dtype=np.float32)
        if nr < self.window:
            pad = np.zeros((self.window - nr, N_FEATURES), dtype=np.float32)
            pad[:, fx.PAD_COL] = 1.0
            win = np.concatenate([pad, real], axis=0)
        else:
            win = real
        return (torch.from_numpy(win), torch.from_numpy(self.z[i]), torch.from_numpy(self.mask[i]))


def _subsample_bucket(b, cap, seed):
    """Keep a fixed seeded random subset of at most ``cap`` windows; a no-op when
    ``cap`` is None or the bucket is already smaller. Random (not strided) so the fold
    stays representative across tickers and dates. When the bucket carries a ``tag``
    key the cap applies to each tag group separately, so holdout rows cannot
    displace stopping-fold rows inside one shared cap. Key-agnostic: any dict of
    equal-length aligned lists subsamples coherently."""
    n = len(next(iter(b.values()))) if b else 0
    if not cap or n == 0:
        return b
    if "tag" in b:
        tags = np.asarray(b["tag"])
        keep_parts = []
        for t in np.unique(tags):
            grp = np.flatnonzero(tags == t)
            if grp.size > cap:
                grp = grp[np.random.default_rng(seed + int(t)).choice(
                    grp.size, size=cap, replace=False)]
            keep_parts.append(grp)
        keep = np.sort(np.concatenate(keep_parts))
        if keep.size == n:
            return b
    else:
        if n <= cap:
            return b
        keep = np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
    return {k: [b[k][i] for i in keep] for k in b}


_META_KEYS = ("ticker_id", "date", "tdates", "tradable", "close", "sigma_hat", "tag",
              "m_uncens", "comp_momentum_20d", "comp_momentum_5d", "comp_vol_z")


def _bucket_meta(b):
    return {
        "ticker_id": np.asarray(b["ticker_id"], dtype=np.int64),
        "date": np.asarray(b["date"], dtype=np.int64),
        "tdates": np.asarray(b["tdates"], dtype=np.int64),
        "tradable": np.asarray(b["tradable"], dtype=bool),
        "close": np.asarray(b["close"], dtype=np.float64),
        "sigma_hat": np.asarray(b["sigma_hat"], dtype=np.float64),
        "tag": np.asarray(b["tag"], dtype=np.int64),
        "m_uncens": np.asarray(b["m_uncens"], dtype=np.float32),
        "comp_momentum_20d": np.asarray(b["comp_momentum_20d"], dtype=np.float64),
        "comp_momentum_5d": np.asarray(b["comp_momentum_5d"], dtype=np.float64),
        "comp_vol_z": np.asarray(b["comp_vol_z"], dtype=np.float64),
    }


def build_memmap_and_samples(frames, scaler, bounds, eval_mode, eval_tickers, val_tickers,
                             window, max_windows=None, val_subsample=None, seed=0,
                             cache_dir=None, panel_seed=None):
    """Normalize each ticker, write the concatenated fp16 memmap, and enumerate the
    train / val samples, the report-only train panel, and the per-ticker eval
    surface. ``cache_dir`` isolates the memmap so concurrent processes (one per GPU)
    do not write the same file.

    Returns ``(mm, offsets, train_ds, (val_ds, val_meta), (panel_ds, panel_meta),
    eval_surface)``.
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_rows = sum(len(f.dates) for f in frames.values())
    need = int(total_rows * N_FEATURES * 2 * 1.2)
    free = shutil.disk_usage(cache_dir).free
    if free < need:
        raise SystemExit(f"cache preflight: {cache_dir} has {free} bytes free but the "
                         f"memmap needs {need} (1.2x headroom); free disk space first")
    mm_path = cache_dir / "features_f16.dat"
    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(total_rows, N_FEATURES))

    # Report-only train-panel sessions: 4 seeded picks per sub-era.
    panel_dates = set()
    if panel_seed is not None:
        rng = np.random.default_rng(panel_seed)
        u = bounds["unique_dates"]
        train_sessions = u[u < bounds["train_cut"]]
        sub_edges = [np.datetime64("1990-01-01"), np.datetime64("2008-01-01"),
                     np.datetime64("2016-01-01")]
        eras = [train_sessions[train_sessions < sub_edges[0]],
                train_sessions[(train_sessions >= sub_edges[0]) & (train_sessions < sub_edges[1])],
                train_sessions[(train_sessions >= sub_edges[1]) & (train_sessions < sub_edges[2])],
                train_sessions[train_sessions >= sub_edges[2]]]
        for sess in eras:
            if sess.size:
                pick = rng.choice(sess.size, size=min(4, sess.size), replace=False)
                panel_dates.update(sess[pick].tolist())

    ticker_ids = {t: i for i, t in enumerate(sorted(frames))}
    comparator_idx = {"comp_momentum_20d": fx.FEATURE_NAMES.index("momentum_20d"),
                      "comp_momentum_5d": fx.FEATURE_NAMES.index("momentum_5d"),
                      "comp_vol_z": fx.FEATURE_NAMES.index("vol_z")}

    def new_bucket(with_meta):
        b = {"start": [], "nreal": [], "z": [], "m": []}
        if with_meta:
            for k in _META_KEYS:
                b[k] = []
        return b

    offsets, cursor = {}, 0
    train = new_bucket(False)
    val = new_bucket(True)
    panel = new_bucket(True)
    eval_surface = {}     # ticker -> list of local anchor indices
    thin_before = np.zeros(4, dtype=np.int64)
    thin_after = np.zeros(4, dtype=np.int64)
    floor_dropped = np.zeros(3, dtype=np.int64)
    floor_total = np.zeros(3, dtype=np.int64)
    zero_label_train_anchors = 0

    for ticker, f in frames.items():
        norm = scaler.transform(f.features)
        n = len(f.dates)
        mm[cursor:cursor + n] = norm.astype(np.float16)
        offsets[ticker] = (cursor, n)
        split, split_end = row_splits(f, bounds, eval_mode,
                                      ticker in eval_tickers, ticker in val_tickers)
        mask = fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end)
        # Uncensored-outcome alternative mask: the same mask with the spike term
        # neutralized (all-True spike_free), for the censorship sensitivity reports.
        m_uncens = fx.label_mask(f.z, np.ones_like(f.spike_free), f.target_dates,
                                 f.dates, split_end)
        mask_prethin = mask.copy()
        era = _era_of(f.dates)
        tag = 1 if ticker in eval_tickers else 0

        # Train-mask thinning (1m/6m), split-0 labels only. Calendar-day key so
        # a backfill cannot silently re-select every ticker's surviving label subset;
        # stable hash offset exactly as assign_eval_tickers (builtin hash() is
        # per-process randomized and would desynchronize the per-GPU builds).
        date_ord = f.dates.astype("datetime64[D]").astype(np.int64)
        is_train_row = split == 0
        for hi, stride in THIN_STRIDE.items():
            off = int(hashlib.sha256(f"{ticker}|{hi}".encode("utf-8")).hexdigest(),
                      16) % stride
            keep = ((date_ord + off) % stride) == 0
            kill = is_train_row & (mask[:, hi] > 0) & ~keep
            thin_before[hi] += int((is_train_row & (mask[:, hi] > 0)).sum())
            mask[kill, hi] = 0.0
            thin_after[hi] += int((is_train_row & (mask[:, hi] > 0)).sum())
        for hi in (0, 1):
            cnt = int((is_train_row & (mask[:, hi] > 0)).sum())
            thin_before[hi] += cnt
            thin_after[hi] += cnt

        anchors = np.arange(MIN_REAL_ROWS - 1, n)
        for a in anchors:
            s = int(split[a])
            if s == 2:
                # Every eval anchor is forecast regardless of label availability --
                # labels are not needed to forecast, and skipping label-less anchors
                # would let future information select the evaluation universe.
                eval_surface.setdefault(ticker, []).append(int(a))
                continue
            if s == 0:
                floor_total[era[a]] += 1
                if f.close[a] < TRAIN_PRICE_FLOOR:
                    floor_dropped[era[a]] += 1
                    continue
            if s == 0 and mask[a].sum() == 0:
                if mask_prethin[a].sum() > 0:
                    zero_label_train_anchors += 1
                continue
            if s == 1 and mask[a].sum() == 0:
                continue
            bucket = train if s == 0 else (val if s == 1 else None)
            if bucket is None:
                continue
            lo = cursor + max(0, a - window + 1)
            bucket["start"].append(lo)
            bucket["nreal"].append(min(a + 1, window))
            bucket["z"].append(f.z[a])
            bucket["m"].append(mask[a])
            if bucket is val:
                val["ticker_id"].append(ticker_ids[ticker])
                val["date"].append(int(f.dates[a].astype("datetime64[ns]").astype(np.int64)))
                val["tdates"].append(f.target_dates[a].astype("datetime64[ns]").astype(np.int64))
                val["tradable"].append(bool(f.tradable[a]))
                val["close"].append(float(f.close[a]))
                val["sigma_hat"].append(float(f.sigma_hat[a]))
                val["tag"].append(tag)
                val["m_uncens"].append(m_uncens[a])
                for key, j in comparator_idx.items():
                    val[key].append(float(f.features[a, j]))
            # Report-only train panel: unthinned mask, same meta keys.
            if (s == 0 and panel_dates and f.dates[a] in panel_dates
                    and mask_prethin[a].sum() > 0):
                panel["start"].append(lo)
                panel["nreal"].append(min(a + 1, window))
                panel["z"].append(f.z[a])
                panel["m"].append(mask_prethin[a])
                panel["ticker_id"].append(ticker_ids[ticker])
                panel["date"].append(int(f.dates[a].astype("datetime64[ns]").astype(np.int64)))
                panel["tdates"].append(f.target_dates[a].astype("datetime64[ns]").astype(np.int64))
                panel["tradable"].append(bool(f.tradable[a]))
                panel["close"].append(float(f.close[a]))
                panel["sigma_hat"].append(float(f.sigma_hat[a]))
                panel["tag"].append(tag)
                panel["m_uncens"].append(m_uncens[a])
                for key, j in comparator_idx.items():
                    panel[key].append(float(f.features[a, j]))
        cursor += n
    mm.flush()

    print("[v5] thinning retained train labels: "
          + "  ".join(f"{HORIZON_LABELS[hi]}:{thin_after[hi]}/{thin_before[hi]}"
                      for hi in range(4)))
    if zero_label_train_anchors:
        print(f"[v5] thinning: {zero_label_train_anchors} train anchors lost all labels "
              "(gap-straddling or thinned out) and were skipped")
    for e, name in enumerate(ERA_NAMES):
        if floor_total[e]:
            print(f"[v5] train price floor [{name}]: dropped "
                  f"{floor_dropped[e]}/{floor_total[e]} anchors "
                  f"({floor_dropped[e] / floor_total[e]:.4f}) below ${TRAIN_PRICE_FLOOR}")

    def pack(b, cap, with_meta=False):
        b = _subsample_bucket(b, cap, seed)
        if not b["start"]:
            return (None, None) if with_meta else None
        ds = WindowDataset(mm, np.asarray(b["start"], dtype=np.int64),
                           np.asarray(b["nreal"], dtype=np.int64),
                           np.asarray(b["z"], dtype=np.float32),
                           np.asarray(b["m"], dtype=np.float32), window)
        if with_meta:
            return ds, _bucket_meta(b)
        return ds

    # Validation is held to a fixed subsample: a full-universe val set is millions of
    # windows, which would make every eval and the temperature fit ruinously slow.
    val_caps = [c for c in (max_windows, val_subsample) if c]
    val_cap = min(val_caps) if val_caps else None
    # Panel cross-sections are capped at build time (2000 names per session).
    if panel["start"]:
        pdate = np.asarray(panel["date"])
        keep_parts = []
        rng = np.random.default_rng(panel_seed if panel_seed is not None else 0)
        for d in np.unique(pdate):
            grp = np.flatnonzero(pdate == d)
            if grp.size > 2000:
                grp = grp[rng.choice(grp.size, size=2000, replace=False)]
            keep_parts.append(grp)
        keep = np.sort(np.concatenate(keep_parts))
        panel = {k: [panel[k][i] for i in keep] for k in panel}
    return (mm, offsets, pack(train, max_windows), pack(val, val_cap, True),
            pack(panel, None, True), eval_surface)


# --------------------------------------------------------------- run manifest

def _git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def data_fingerprint():
    """Sorted (filename, size, mtime) for every shard plus fear_greed.csv plus the
    census sha256 -- the deterministic-inputs half of the finalize snapshot assert."""
    entries = []
    for p in sorted(glob.glob(str(fx.PARTS_DIR / "*.parquet"))):
        entries.append([Path(p).name, os.path.getsize(p), os.path.getmtime(p)])
    fg = fx.FEAR_GREED_PATH
    if fg.exists():
        entries.append([fg.name, os.path.getsize(fg), os.path.getmtime(fg)])
    census = fx.CENSUS_PATH
    census_sha = (hashlib.sha256(Path(census).read_bytes()).hexdigest()
                  if Path(census).exists() else None)
    return {"files": entries, "census_sha256": census_sha}


def _jsonable_args(args):
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def next_run_dir(root=RUNS_DIR):
    """Claim the next numbered run directory (r1, r2, ...) under ``root``.

    Numbered side-by-side run dirs are the chronology: a run never overwrites a
    previous one, and number gaps mark discarded runs. The mkdir is the claim, so
    two simultaneous launches cannot resolve to the same number; protocol runs
    (nullc, ro1..ro3) and smoke pass an explicit --run-dir instead.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[1:]) for p in root.glob("r[0-9]*") if p.name[1:].isdigit()]
    n = max(taken, default=0) + 1
    while True:
        cand = root / f"r{n}"
        try:
            cand.mkdir()
            return cand
        except FileExistsError:
            n += 1


def write_run_manifest(args, run_dir):
    """Written only by the ``--member-start 0`` process, immediately after argument
    resolution (post --smoke mutation, so recorded values are effective), via
    temp-file + atomic rename: four unsynchronized writers of one contract file
    would leave nondeterministic or torn provenance."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": _jsonable_args(args),
        "git_revision": _git_revision(),
        "data_fingerprint": data_fingerprint(),
        "feature_names_hash": fx.feature_names_hash(),
        "run_nonce": os.environ.get("RUN_NONCE"),
        "scaler_rows": None,
    }
    tmp = run_dir / "run_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run_manifest.json")
    print(f"[v5] wrote {run_dir / 'run_manifest.json'} "
          f"(nonce {manifest['run_nonce']})")


def append_manifest_scaler_rows(run_dir, n_rows):
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["scaler_rows"] = int(n_rows)
    tmp = run_dir / "run_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def check_run_manifest(args, run_dir, timeout_s=60):
    """Reader-side manifest check for concurrent multi-GPU trainers: poll for a
    manifest carrying the current launch nonce (token equality -- a mismatch is a
    stale file from a previous launch), then SystemExit on any split-relevant-args
    mismatch. Skipped entirely when RUN_NONCE is absent (single-process runs)."""
    nonce = os.environ.get("RUN_NONCE")
    if not nonce:
        return
    path = run_dir / "run_manifest.json"
    deadline = time.time() + timeout_s
    manifest = None
    while time.time() < deadline:
        if path.exists():
            try:
                m = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                m = None
            if m and m.get("run_nonce") == nonce:
                manifest = m
                break
        time.sleep(2)
    if manifest is None:
        raise SystemExit(f"run manifest with nonce {nonce} not found at {path} within "
                         f"{timeout_s}s -- writer process missing or stale launch")
    mine = _jsonable_args(args)
    for k in SPLIT_ARG_KEYS + ("members",):
        if manifest["args"].get(k) != mine.get(k):
            raise SystemExit(f"run manifest args mismatch on '{k}': "
                             f"{manifest['args'].get(k)} != {mine.get(k)}")
    print(f"[v5] run manifest check passed (nonce {nonce})")


def assert_finalize_snapshot(args, run_dir):
    """--forecast-only recomputes splits, scaler, and val fold from
    disk + code + args; data fingerprint + git revision + the split-relevant args
    together pin that deterministic recomputation. Any mismatch is a SystemExit."""
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise SystemExit(f"--forecast-only requires {path} (written by this run's "
                         "training processes); train first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fp = json.loads(json.dumps(data_fingerprint()))
    if fp != manifest["data_fingerprint"]:
        raise SystemExit("finalize refused: data fingerprint changed since training "
                         "(shards, fear_greed.csv, or census differ)")
    rev = _git_revision()
    if rev != manifest["git_revision"]:
        raise SystemExit(f"finalize refused: git revision {rev} != training revision "
                         f"{manifest['git_revision']}")
    mine = _jsonable_args(args)
    for k in SPLIT_ARG_KEYS + ("members",):
        if manifest["args"].get(k) != mine.get(k):
            raise SystemExit(f"finalize refused: arg '{k}' = {mine.get(k)} differs "
                             f"from training value {manifest['args'].get(k)}")
    return manifest


# ------------------------------------------------------------------ training

def eval_due(step):
    """Scheduled evaluation cadence: every 250 steps up to 2k (both executed runs
    crossed the marginal baseline inside warmup -- the candidate-optimum region was
    never measured), 1000 up to 30k, 5000 after."""
    if step <= 2000:
        interval = 250
    elif step <= 30_000:
        interval = 1000
    else:
        interval = 5000
    return step % interval == 0


def _spread_indices(n, cap):
    """Evenly spread subset of [0, n): spans the full ticker-ordered fold at every
    fold size (the old floor-stride truncated to the head whenever
    cap <= n < 2*cap, deterministically excluding the alphabetically-last tickers)."""
    if not cap or n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def stopping_selection(val_meta, mask, blocks, roles):
    """Interval-membership stopping-side selection shared by the transient judged
    temperatures, the calibrated-CE tie-breaker, and the persisted temperature fit:
    tag-0 rows whose ANCHOR date lies in a stopping block, with per-label masks kept
    only where the stored TARGET date also lies in a stopping block, horizons
    {1d, 1w} only (1m/6m are excluded from all stopping-side quantities).
    Returns ``(m_stop, rows)``: the (N, 4) stopping-side mask and the row indices
    with at least one qualifying label."""
    role_ok = mx.role_label_mask(val_meta["tdates"], blocks, roles, "stop")
    anchor_block = mx.block_index(val_meta["date"], blocks)
    roles_arr = np.asarray(roles)
    anchor_stop = np.zeros(len(val_meta["date"]), dtype=bool)
    valid = anchor_block >= 0
    anchor_stop[valid] = roles_arr[anchor_block[valid]] == "stop"
    m_stop = np.asarray(mask, dtype=np.float32).copy()
    m_stop[:, 2:] = 0.0
    m_stop *= (role_ok & anchor_stop[:, None]).astype(np.float32)
    m_stop[np.asarray(val_meta["tag"]) != 0] = 0.0
    rows = np.flatnonzero(m_stop[:, :2].sum(1) > 0)
    return m_stop, rows


def _fit_temps_from_logits(logits, z, m, cfg, horizons=(0, 1)):
    """Per-horizon temperature grid search on the given logits/labels; horizons not
    listed keep T = 1. Reuses the 3-class NLL objective."""
    H = len(cfg.horizons)
    half = cfg.n_bins // 2
    grid = torch.linspace(0.5, 5.0, 46)
    temps = [1.0] * H
    for hi in horizons:
        k = cfg.theta_bins[hi]
        mh = m[:, hi].bool()
        if mh.sum() < 1:
            continue
        th = k * cfg.bin_width
        zf = torch.nan_to_num(z[:, hi], nan=0.0)
        true_cls = (zf > -th).long() + (zf > th).long()
        lh = logits[:, hi]
        best_T, best_nll = 1.0, float("inf")
        for T in grid:
            p = torch.softmax(lh / T, dim=-1)
            cls3 = torch.stack([p[:, :half - k].sum(-1), p[:, half - k:half + k].sum(-1),
                                p[:, half + k:].sum(-1)], -1).clamp_min(1e-8)
            nll = torch.nn.functional.nll_loss(cls3.log()[mh], true_cls[mh])
            if nll.item() < best_nll:
                best_nll, best_T = nll.item(), float(T)
        temps[hi] = best_T
    return temps


def _judged_from_logits(logits, z, m, cfg, cal_cap, calibrate=True):
    """One-row-set judged CE on precomputed logits: cap via the full-span
    spread, split the capped subset into deterministic parity halves, fit transient
    temperatures on the fit half, and score raw + calibrated CE on the SAME score
    half. Returns ``(cal_ce, raw_ce, temps, score_rows)`` where ``score_rows``
    indexes into the input rows (for the row-identical baseline)."""
    n = logits.shape[0]
    if n == 0:
        return float("inf"), float("inf"), [1.0] * len(cfg.horizons), np.asarray([], dtype=np.int64)
    capped = _spread_indices(n, cal_cap)
    l, zz, mm_ = logits[capped], z[capped], m[capped]
    parity = torch.arange(l.shape[0]) % 2
    fit, score = parity == 0, parity == 1
    score_rows = capped[score.numpy()]
    if float(mm_[score].sum()) < 1:
        return float("inf"), float("inf"), [1.0] * len(cfg.horizons), score_rows
    raw_ce = v5_loss(l[score], zz[score], mm_[score], cfg).item()
    if not calibrate or float(mm_[fit].sum()) < 1:
        return raw_ce, raw_ce, [1.0] * len(cfg.horizons), score_rows
    temps = _fit_temps_from_logits(l[fit], zz[fit], mm_[fit], cfg)
    T = torch.tensor(temps, dtype=l.dtype).view(1, -1, 1)
    cal_ce = v5_loss(l[score] / T, zz[score], mm_[score], cfg).item()
    return cal_ce, raw_ce, temps, score_rows


@torch.no_grad()
def judged_val_ce(model, ds, cfg, device, batch_size, num_workers=0, cal_cap=100_000):
    """Standalone calibration-fair judged val CE: forwards the
    full-span capped subsample and computes raw, calibrated, and half-split
    quantities on one row set. Retained for ad-hoc probes; the training loop's
    judged numbers come from :func:`eval_pass` on the stopping-side subset."""
    was_training = model.training
    model.eval()
    n = len(ds)
    if n == 0:
        if was_training:
            model.train()
        return float("inf"), float("inf"), [1.0] * len(cfg.horizons)
    idx = _spread_indices(n, cal_cap)
    sub = Subset(ds, idx.tolist())
    pin = device == "cuda"
    loader = DataLoader(sub, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    L, Z, M = [], [], []
    for x, z, m in loader:
        L.append(model(x.to(device, non_blocking=pin)).cpu())
        Z.append(z)
        M.append(m)
    if was_training:
        model.train()
    logits, z, m = torch.cat(L, 0), torch.cat(Z, 0), torch.cat(M, 0)
    cal_ce, raw_ce, temps, _ = _judged_from_logits(
        logits, z, m, cfg, cal_cap=None, calibrate=True)
    return cal_ce, raw_ce, temps


@torch.no_grad()
def val_marginal_baseline_ce(val_ds, cfg, rows=None, horizons=None):
    """Loss of the best input-ignoring predictor on the val fold (or the ``rows``
    subset, restricted to ``horizons`` when given): the per-horizon marginal
    histogram plus the marginal 3-class term, mask-weighted exactly like the
    training loss. The trained model must beat this to be learning anything
    conditional."""
    z_np = np.asarray(val_ds.z, dtype=np.float32)
    m_np = np.asarray(val_ds.mask, dtype=np.float32)
    if rows is not None:
        z_np, m_np = z_np[rows], m_np[rows]
    z = torch.from_numpy(z_np)
    m = torch.from_numpy(m_np)
    soft = hl_gauss_targets(torch.nan_to_num(z), cfg)                  # (N, H, n_bins)
    theta = torch.tensor(cfg.theta_bins, dtype=z.dtype) * cfg.bin_width
    znz = torch.nan_to_num(z)
    cls = (znz > -theta).long() + (znz > theta).long()                # (N, H)
    total, count = 0.0, 0.0
    h_iter = range(len(cfg.horizons)) if horizons is None else horizons
    for h in h_iter:
        mh = m[:, h]
        nh = float(mh.sum())
        if nh < 1:
            continue
        q = (soft[:, h] * mh[:, None]).sum(0) / nh                    # mean soft target
        h_hist = float(-(q * q.clamp_min(1e-12).log()).sum())
        rate = torch.stack([((cls[:, h] == c).float() * mh).sum() / nh for c in range(3)])
        h_cls = float(-(rate * rate.clamp_min(1e-12).log()).sum())
        total += nh * (h_hist + cfg.lam_cls * h_cls)
        count += nh
    return total / count if count else float("inf")


def val_marginal_baseline_ce_subset(val_ds, cfg, indices, mask_override=None,
                                    horizons=None):
    """Baseline CE on an explicit row subset, optionally under an overriding label
    mask -- the row-identical BEAT/MISS counterpart of the judged score half."""
    z_np = np.asarray(val_ds.z, dtype=np.float32)[indices]
    m_np = (np.asarray(mask_override, dtype=np.float32)[indices]
            if mask_override is not None
            else np.asarray(val_ds.mask, dtype=np.float32)[indices])

    class _Tmp:
        z = z_np
        mask = m_np

    return val_marginal_baseline_ce(_Tmp, cfg, horizons=horizons)


@torch.no_grad()
def val_baselines(val_ds, val_meta, cfg, train_z=None, train_m=None, rows=None):
    """Per-horizon and per-date baseline components in both flavors: the val-oracle
    marginal and (when train labels are given) the train-marginal-on-val. The
    per-date components feed the paired per-date CE gate; millinat-margin verdicts
    must not hinge on the baseline flavor."""
    z = np.asarray(val_ds.z, dtype=np.float32)
    m = np.asarray(val_ds.mask, dtype=np.float32)
    dates = np.asarray(val_meta["date"])
    if rows is not None:
        z, m, dates = z[rows], m[rows], dates[rows]
    zt = torch.from_numpy(np.nan_to_num(z))
    mt = torch.from_numpy(m)
    soft = hl_gauss_targets(zt, cfg)
    theta = torch.tensor(cfg.theta_bins, dtype=zt.dtype) * cfg.bin_width
    cls = (zt > -theta).long() + (zt > theta).long()

    def flavor(q_hist_h, q_cls_h):
        per_h, per_date = [], []
        total_num, total_den = 0.0, 0.0
        for h in range(len(cfg.horizons)):
            mh = mt[:, h]
            nh = float(mh.sum())
            if nh < 1:
                per_h.append(float("nan"))
                per_date.append(({}, 0))
                continue
            q = q_hist_h[h].clamp_min(1e-12)
            qc = q_cls_h[h].clamp_min(1e-12)
            ce_rows = (-(soft[:, h] * q.log()).sum(-1)
                       - cfg.lam_cls * qc.log()[cls[:, h]]) * mh
            per_h.append(float(ce_rows.sum() / nh))
            total_num += float(ce_rows.sum())
            total_den += nh
            by_date = {}
            ce_np = ce_rows.numpy()
            mh_np = mh.numpy()
            for d in np.unique(dates):
                sel = dates == d
                nsel = float(mh_np[sel].sum())
                if nsel >= 1:
                    by_date[int(d)] = float(ce_np[sel].sum() / nsel)
            per_date.append((by_date, len(by_date)))
        return {"total": total_num / total_den if total_den else float("inf"),
                "per_h": per_h, "per_date": per_date}

    out = {}
    q_hist, q_cls = [], []
    for h in range(len(cfg.horizons)):
        mh = mt[:, h]
        nh = float(mh.sum())
        q_hist.append((soft[:, h] * mh[:, None]).sum(0) / max(nh, 1.0))
        q_cls.append(torch.stack([((cls[:, h] == c).float() * mh).sum() / max(nh, 1.0)
                                  for c in range(3)]))
    out["oracle"] = flavor(q_hist, q_cls)
    if train_z is not None:
        tz = torch.from_numpy(np.nan_to_num(np.asarray(train_z, dtype=np.float32)))
        tm = torch.from_numpy(np.asarray(train_m, dtype=np.float32))
        tsoft = hl_gauss_targets(tz, cfg)
        tcls = (tz > -theta).long() + (tz > theta).long()
        tq_hist, tq_cls = [], []
        for h in range(len(cfg.horizons)):
            mh = tm[:, h]
            nh = float(mh.sum())
            tq_hist.append((tsoft[:, h] * mh[:, None]).sum(0) / max(nh, 1.0))
            tq_cls.append(torch.stack([((tcls[:, h] == c).float() * mh).sum() / max(nh, 1.0)
                                       for c in range(3)]))
        out["train_marginal"] = flavor(tq_hist, tq_cls)
    return out


@torch.no_grad()
def eval_pass(model, val_ds, val_meta, cfg, device, batch_size, num_workers,
              blocks, roles, ic_floor, m_stop, stop_rows, base_stop=None,
              cal_cap=100_000, calibrate=True, panel=None):
    """One full evaluation of a member checkpoint on the validation fold.

    Forwards the tag-stratified fold once and computes, on RAW logits (the persisted
    temperatures do not exist during training, and half-sample judged temperatures
    would inject selection noise into the checkpoint choice): the full-fold tag-0
    raw CE (report), the stopping-side judged raw/calibrated CE (one row set on
    the {1d, 1w} interval-membership subset), the per-date IC suite for
    both scores with block-role splits, tradability and uncensored-outcome variants,
    top-of-ranking, the stopping score inputs, and the report-only train-panel and
    tag-1 views. Gating-side values are computed but only ever written to the eval
    history file -- never printed as headline, never consulted by stopping.
    """
    was_training = model.training
    model.eval()
    pin = device == "cuda"
    loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=pin)
    L = []
    for x, _, _ in loader:
        L.append(model(x.to(device, non_blocking=pin)).float().cpu())
    logits = torch.cat(L, 0)
    z_np = np.asarray(val_ds.z, dtype=np.float32)
    m_np = np.asarray(val_ds.mask, dtype=np.float32)
    z_t = torch.from_numpy(z_np)
    m_t = torch.from_numpy(m_np)
    tag = np.asarray(val_meta["tag"])
    t0 = tag == 0

    out = {}
    t0_idx = np.flatnonzero(t0)
    out["raw_ce_tag0"] = (v5_loss(logits[t0_idx], z_t[t0_idx], m_t[t0_idx], cfg).item()
                          if t0_idx.size else float("inf"))
    t1_idx = np.flatnonzero(~t0)
    out["raw_ce_tag1"] = (v5_loss(logits[t1_idx], z_t[t1_idx], m_t[t1_idx], cfg).item()
                          if t1_idx.size else None)

    # Stopping-side judged CE: one row set, {1d, 1w} interval membership.
    m_stop_t = torch.from_numpy(m_stop)
    cal_ce, raw_ce_stop, jt, score_rows_local = _judged_from_logits(
        logits[stop_rows], z_t[stop_rows], m_stop_t[stop_rows], cfg,
        cal_cap=cal_cap, calibrate=calibrate)
    out["cal_ce_stop"] = cal_ce
    out["raw_ce_stop"] = raw_ce_stop
    out["judged_T"] = jt
    out["n_stop_score_rows"] = int(len(score_rows_local))
    out["base_stop"] = base_stop

    # Scores from raw logits.
    probs = logits.softmax(-1)
    half = cfg.n_bins // 2
    p_up = np.stack([probs[:, hi, half + k:].sum(-1).numpy()
                     for hi, k in enumerate(cfg.theta_bins)], 1)
    p_down = np.stack([probs[:, hi, :half - k].sum(-1).numpy()
                       for hi, k in enumerate(cfg.theta_bins)], 1)

    meta0 = {k: np.asarray(val_meta[k])[t0] for k in ("date", "ticker_id", "tdates")}
    mcfg = mx.default_cfg(min_names=ic_floor)
    records, series = mx.ic_suite(p_up[t0], p_down[t0], z_np[t0], m_np[t0], meta0,
                                  mcfg, blocks, roles,
                                  tradable=np.asarray(val_meta["tradable"])[t0],
                                  alt_mask=np.asarray(val_meta["m_uncens"])[t0])
    out["ic_records"] = records

    def _rec(hl, score, role, trad="all", maskv="primary"):
        for r in records:
            if (r["horizon"] == hl and r["score"] == score and r["role"] == role
                    and r["trad"] == trad and r["mask"] == maskv):
                return r
        return None

    r1d = _rec("1d", "score", "stop")
    r1w = _rec("1w", "score", "stop")
    out["t_1d_stop"] = r1d["t"] if r1d else float("nan")
    out["t_1w_stop"] = r1w["t"] if r1w else float("nan")
    out["n_1d_stop"] = r1d["n_dates"] if r1d else 0
    out["n_1w_stop"] = r1w["n_dates"] if r1w else 0
    out["stopping_score"] = mx.stopping_score(out["t_1d_stop"], out["t_1w_stop"],
                                              out["n_1d_stop"], out["n_1w_stop"])

    score_arr = p_up - p_down
    top_records, _ = mx.top_of_ranking(
        score_arr[t0], z_np[t0], m_np[t0], meta0,
        sigma=np.asarray(val_meta["sigma_hat"])[t0],
        tradable=np.asarray(val_meta["tradable"])[t0],
        alt_mask=np.asarray(val_meta["m_uncens"])[t0],
        min_names=ic_floor)
    out["top_records"] = top_records

    if t1_idx.size:
        meta1 = {k: np.asarray(val_meta[k])[~t0] for k in ("date", "ticker_id", "tdates")}
        rec1, _ = mx.ic_suite(p_up[~t0], p_down[~t0], z_np[~t0], m_np[~t0], meta1,
                              mcfg)
        out["ic_records_tag1"] = rec1

    if panel is not None and panel[0] is not None:
        panel_ds, panel_meta = panel
        ploader = DataLoader(panel_ds, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin)
        PL = []
        for x, _, _ in ploader:
            PL.append(model(x.to(device, non_blocking=pin)).float().cpu())
        plogits = torch.cat(PL, 0)
        pprobs = plogits.softmax(-1)
        pp_up = np.stack([pprobs[:, hi, half + k:].sum(-1).numpy()
                          for hi, k in enumerate(cfg.theta_bins)], 1)
        pp_down = np.stack([pprobs[:, hi, :half - k].sum(-1).numpy()
                            for hi, k in enumerate(cfg.theta_bins)], 1)
        prec, _ = mx.ic_suite(pp_up, pp_down, np.asarray(panel_ds.z, dtype=np.float32),
                              np.asarray(panel_ds.mask, dtype=np.float32),
                              {k: np.asarray(panel_meta[k])
                               for k in ("date", "ticker_id", "tdates")}, mcfg)
        out["panel_records"] = prec

    if was_training:
        model.train()
    return out


def comparator_report(val_meta, val_ds, ic_floor):
    """Checkpoint-independent trivial-comparator ICs from the anchor feature rows
    (ranks are scaler-invariant): momentum_20d, 5-day reversal (-momentum_5d),
    vol_z. 'Continue' must mean beating a screen of the model's own inputs."""
    z = np.asarray(val_ds.z, dtype=np.float32)
    m = np.asarray(val_ds.mask, dtype=np.float32)
    tag = np.asarray(val_meta["tag"])
    t0 = tag == 0
    dates = np.asarray(val_meta["date"])[t0]
    sessions = np.unique(dates)
    lines, recs = [], []
    for name, key, sgn in (("momentum_20d", "comp_momentum_20d", 1.0),
                           ("reversal_5d", "comp_momentum_5d", -1.0),
                           ("vol_z", "comp_vol_z", 1.0)):
        col = sgn * np.asarray(val_meta[key])[t0]
        parts = []
        for hi, hl in enumerate(HORIZON_LABELS):
            el = (m[t0][:, hi] > 0) & np.isfinite(col)
            keys, ics = mx.spearman_ic(col[el], z[t0][el, hi], dates[el],
                                       min_names=ic_floor)
            mean, se, t, cnt = mx.hac_t(ics, np.searchsorted(sessions, keys),
                                        mx.HAC_LAG[hi])
            recs.append({"comparator": name, "horizon": hl, "mean_ic": mean, "t": t,
                         "n_dates": cnt,
                         "ic_std": float(np.std(ics)) if cnt else float("nan")})
            parts.append(f"{hl}: IC {mean:+.4f} t {t:+.2f} ({cnt}d)"
                         if cnt else f"{hl}: -")
        lines.append(f"[v5] comparator {name}: " + " | ".join(parts))
    return "\n".join(lines), recs


def train_member(member_idx, seed, cfg, train_ds, val_ds, val_meta, device, max_steps,
                 eval_every_steps, patience, lr, batch_size, steps_per_epoch,
                 num_workers, log_every, *, blocks, roles, ic_floor, run_dir,
                 panel=None, base_stop=None, lr_patience=4, lr_factor=0.5,
                 min_lr_scale=0.04, warmup_steps=1000, calibrate=True,
                 cal_cap=100_000):
    torch.manual_seed(seed)
    model = V5Backbone(cfg).to(device)
    opt = torch.optim.AdamW(optimizer_param_groups(model, weight_decay=0.01),
                            lr=lr, betas=(0.9, 0.95))

    # Plateau LR schedule: warm-up, then a constant scale the eval block multiplies
    # down whenever the stopping criterion stalls; decay is driven by the same
    # signal as early-stop and cannot decouple from the run length.
    lr_scale = 1.0
    plateau_since_drop = 0

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return lr_scale

    have_val = val_ds is not None
    if have_val:
        m_stop, stop_rows = stopping_selection(val_meta, val_ds.mask, blocks, roles)
    else:
        m_stop, stop_rows = None, None
    history_path = Path(run_dir) / f"eval_history_member{member_idx}.jsonl"
    history_f = open(history_path, "w", encoding="utf-8")
    train_history_path = Path(run_dir) / f"train_history_member{member_idx}.jsonl"
    train_f = open(train_history_path, "w", encoding="utf-8")

    pin = device == "cuda"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0,
                        pin_memory=pin, generator=torch.Generator().manual_seed(seed))
    # bf16 autocast on GPU only (the selective scan keeps fp32 internally); a no-op on
    # CPU, where bf16 autocast is unsupported and the fallback block runs fp32.
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())
    best_score = float("-inf")
    best_ce = float("inf")
    best_state, since_improve, step = None, 0, 0
    score_hist = []
    bootstrap = True          # CE-driven until the smoothed stopping score is finite
    loss_ema, t0_time, stop = None, time.time(), False
    per_h_ema = [None] * len(cfg.horizons)
    cadence = (f"every {eval_every_steps} steps" if eval_every_steps
               else "scheduled (250<=2k, 1000<=30k, 5000 after)")
    print(f"[member {member_idx}] start | {len(train_ds)} train / "
          f"{0 if val_ds is None else len(val_ds)} val windows | {steps_per_epoch} steps/epoch | "
          f"eval {cadence} | max {max_steps} steps | patience {patience} | "
          f"warmup {warmup_steps} | plateau LRx{lr_factor}@{lr_patience} floor {min_lr_scale} | "
          f"stopping on smoothed rank-IC score (min improvement 0.05, median of last 3)")
    model.train()

    def run_eval():
        nonlocal best_score, best_ce, best_state, since_improve, stop
        nonlocal plateau_since_drop, lr_scale, bootstrap
        res = eval_pass(model, val_ds, val_meta, cfg, device, batch_size, num_workers,
                        blocks, roles, ic_floor, m_stop, stop_rows,
                        base_stop=base_stop, cal_cap=cal_cap, calibrate=calibrate,
                        panel=panel)
        score_hist.append(res["stopping_score"])
        smoothed = float(np.median(score_hist[-3:]))
        res["smoothed_score"] = smoothed
        if bootstrap and np.isfinite(smoothed):
            bootstrap = False
        if bootstrap:
            improved = res["cal_ce_stop"] < best_ce - 1e-5
        else:
            improved = smoothed > best_score + 0.05
        if res["cal_ce_stop"] < best_ce - 1e-5:
            best_ce = res["cal_ce_stop"]
        if improved:
            if not bootstrap:
                best_score = smoothed
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
            plateau_since_drop = 0
        else:
            since_improve += 1
            plateau_since_drop += 1
            if plateau_since_drop >= lr_patience and lr_scale > min_lr_scale:
                lr_scale = max(min_lr_scale, lr_scale * lr_factor)
                plateau_since_drop = 0
                print(f"[member {member_idx}] LR DROP -> {lr * lr_scale:.2e} "
                      f"(scale {lr_scale:.3f}) after {lr_patience} no-improve evals")
        # Crash durability (run-2's finalize died with every member state only in
        # RAM): latest at each eval, best on improvement. The final member_{i}.pt
        # written after training (best state) remains the loader contract.
        torch.save(model.state_dict(), Path(run_dir) / f"member_{member_idx}_latest.pt")
        if improved:
            torch.save(best_state, Path(run_dir) / f"member_{member_idx}_best.pt")
        base_str = ("" if base_stop is None else
                    f" | base {base_stop:.4f} "
                    f"{'BEAT' if res['cal_ce_stop'] < base_stop else 'MISS'}")
        print(f"[member {member_idx}] EVAL step {step} (ep {step / steps_per_epoch:.2f}) | "
              f"stop-score {res['stopping_score']:+.3f} (smoothed {smoothed:+.3f}, "
              f"best {best_score:+.3f}) | t1d {res['t_1d_stop']:+.2f}/{res['n_1d_stop']}d "
              f"t1w {res['t_1w_stop']:+.2f}/{res['n_1w_stop']}d | "
              f"cal_ce {res['cal_ce_stop']:.4f} ({res['n_stop_score_rows']} rows)"
              f"{base_str} | raw_tag0 {res['raw_ce_tag0']:.4f} | "
              f"lr_scale {lr_scale:.3f} | "
              f"{'IMPROVED' if improved else f'no-improve {since_improve}/{patience}'}"
              + (" [bootstrap: CE-driven]" if bootstrap else ""))
        line = {"step": step, "smoothed_score": smoothed, "improved": bool(improved),
                "bootstrap": bool(bootstrap)}
        for k, v in res.items():
            line[k] = v
        history_f.write(json.dumps(line, default=float) + "\n")
        history_f.flush()
        if since_improve >= patience:
            stop = True

    while step < max_steps and not stop:
        for x, z, m in loader:
            x = x.to(device, non_blocking=pin)
            z = z.to(device, non_blocking=pin)
            m = m.to(device, non_blocking=pin)
            cur_lr = lr * lr_at(step)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.zero_grad()
            with amp:
                loss, per_h = v5_loss_components(model(x), z, m, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            ph = per_h.cpu().numpy()
            for hi in range(len(cfg.horizons)):
                if np.isfinite(ph[hi]) and ph[hi] > 0:
                    per_h_ema[hi] = (ph[hi] if per_h_ema[hi] is None
                                     else 0.98 * per_h_ema[hi] + 0.02 * float(ph[hi]))
            if step % log_every == 0:
                sps = step / max(1e-9, time.time() - t0_time)
                per_h_str = " ".join(
                    f"{HORIZON_LABELS[hi]}:{per_h_ema[hi]:.3f}" if per_h_ema[hi] else
                    f"{HORIZON_LABELS[hi]}:-" for hi in range(len(cfg.horizons)))
                print(f"[member {member_idx}] step {step}/{max_steps} (ep {step / steps_per_epoch:.2f}) | "
                      f"loss {loss_ema:.4f} [{per_h_str}] | lr {cur_lr:.2e} | {sps:.1f} it/s | "
                      f"eta {(max_steps - step) / max(1e-9, sps) / 60:.0f}m")
                train_f.write(json.dumps(
                    {"step": step, "lr": cur_lr, "loss_ema": loss_ema,
                     "per_h_ema": {HORIZON_LABELS[hi]: per_h_ema[hi]
                                   for hi in range(len(cfg.horizons))},
                     "it_s": sps, "elapsed_s": time.time() - t0_time},
                    default=float) + "\n")
                train_f.flush()
            eval_now = (step % eval_every_steps == 0) if eval_every_steps else eval_due(step)
            if (eval_now or step >= max_steps) and have_val:
                run_eval()
                if stop or step >= max_steps:
                    stop = True
                    break
            elif step >= max_steps:
                stop = True
                break
    history_f.close()
    train_f.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    crit = f"score {best_score:+.3f}" if np.isfinite(best_score) else f"cal CE {best_ce:.4f}"
    print(f"[member {member_idx}] done | {step} steps | best {crit} | "
          f"{(time.time() - t0_time) / 60:.1f} min")
    return model, best_score if np.isfinite(best_score) else best_ce


@torch.no_grad()
def evaluate_ce(model, ds, cfg, device, batch_size, num_workers=0):
    was_training = model.training
    model.eval()
    pin = device == "cuda"
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    total, count = 0.0, 0
    for x, z, m in loader:
        x = x.to(device, non_blocking=pin)
        z = z.to(device, non_blocking=pin)
        m = m.to(device, non_blocking=pin)
        n = int(m.sum().item())
        if n:
            total += v5_loss(model(x), z, m, cfg).item() * n
            count += n
    if was_training:
        model.train()
    return total / count if count else float("inf")


# -------------------------------------------------------------- calibration

@torch.no_grad()
def fit_temperatures(members, val_ds, val_meta, cfg, device, batch_size,
                     num_workers=0, blocks=None, roles=None, chunk=100_000):
    """Per-horizon persisted temperatures minimizing the masked 3-class NLL of the
    ensemble mean, fit on the tag-0 {1d, 1w} stopping-side labels under interval
    membership (anchor AND target inside stopping blocks) -- identical scoping to
    the training-loop judged temperatures, so gating-block labels never shape the
    consumed score. 1m and 6m ALWAYS persist T = 1.0 (rule, not fallback): T = 1
    keeps their consumed Score identical across stopping, gating, and consumers,
    and the score floors are quantiles of the emitted score, so nothing downstream
    needs a fitted 1m/6m temperature. Member logits are cached as fp16 CPU tensors
    (halves the float32 materialization that killed the run-2 finalize); the grid
    search upcasts 100k-row chunks.

    Returns ``(temps, cache)`` where ``cache = (logits_fp16, z, m)`` is reused by
    the score-floor computation.
    """
    H = len(cfg.horizons)
    if val_ds is None or len(val_ds) == 0:
        return [1.0] * H, None
    print(f"[v5] fitting temperatures on {len(val_ds)} val windows "
          f"(fp16 logit cache) ...")
    loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device == "cuda")
    logits_all, z_all, m_all = [], [], []
    for x, z, m in loader:
        x = x.to(device)
        logits_all.append(torch.stack([mem(x) for mem in members]).to(torch.float16).cpu())
        z_all.append(z)
        m_all.append(m)
    logits = torch.cat(logits_all, dim=1)          # (M, N, H, n_bins) fp16
    z = torch.cat(z_all, 0)
    m = torch.cat(m_all, 0)

    if blocks is not None and roles is not None:
        m_stop, _ = stopping_selection(val_meta, m.numpy(), blocks, roles)
        m_fit = torch.from_numpy(m_stop)
    else:
        m_fit = m.clone()
        m_fit[:, 2:] = 0.0

    half = cfg.n_bins // 2
    grid = torch.linspace(0.5, 5.0, 46)
    temps = [1.0] * H
    for hi in (0, 1):
        k = cfg.theta_bins[hi]
        th = k * cfg.bin_width
        mh = m_fit[:, hi].bool()
        n_fit = int(mh.sum())
        if n_fit == 0:
            print(f"[v5] temperature[{HORIZON_LABELS[hi]}]: empty fit population, T=1.0")
            continue
        rows = torch.nonzero(mh, as_tuple=True)[0]
        zf = torch.nan_to_num(z[rows, hi], nan=0.0)
        true_cls = (zf > -th).long() + (zf > th).long()
        best_T, best_nll = 1.0, float("inf")
        for T in grid:
            nll_sum = 0.0
            for lo in range(0, rows.numel(), chunk):
                r = rows[lo:lo + chunk]
                lh = logits[:, r, hi].float() / T          # upcast chunk only
                p = torch.softmax(lh, dim=-1).mean(0)
                cls3 = torch.stack([p[:, :half - k].sum(-1),
                                    p[:, half - k:half + k].sum(-1),
                                    p[:, half + k:].sum(-1)], -1).clamp_min(1e-8)
                nll_sum += float(torch.nn.functional.nll_loss(
                    cls3.log(), true_cls[lo:lo + chunk], reduction="sum"))
            nll = nll_sum / n_fit
            if nll < best_nll:
                best_nll, best_T = nll, float(T)
        temps[hi] = best_T
        print(f"[v5] temperature[{HORIZON_LABELS[hi]}] = {best_T:.2f} "
              f"({n_fit} stopping-side labels)")
    print("[v5] temperature[1m] = 1.0, temperature[6m] = 1.0 "
          "(rule: 1m/6m are excluded from the stopping-side fit and always persist T=1)")
    return temps, (logits, z, m)


@torch.no_grad()
def compute_score_floors(cache, temps, val_meta, cfg, blocks, roles):
    """Per-horizon live score floors: the 95th percentile of the finalize ensemble's
    TEMPERED Score over tradability-filtered tag-0 val samples whose ANCHOR dates
    lie in stopping blocks (anchor membership: floors are quantiles of the score
    distribution and consult no outcomes; stopping-block anchors keep the gating
    surface naive). The unfiltered percentile is recorded beside for reference.
    Recomputed inside every finalize so a re-finalize cannot silently destroy them.
    """
    if cache is None:
        return None, None
    logits, _, _ = cache
    tag = np.asarray(val_meta["tag"])
    anchor_block = mx.block_index(val_meta["date"], blocks)
    roles_arr = np.asarray(roles)
    anchor_stop = np.zeros(len(tag), dtype=bool)
    valid = anchor_block >= 0
    anchor_stop[valid] = roles_arr[anchor_block[valid]] == "stop"
    base_sel = (tag == 0) & anchor_stop
    trad_sel = base_sel & np.asarray(val_meta["tradable"])
    half = cfg.n_bins // 2
    T = torch.tensor(temps, dtype=torch.float32).view(1, 1, -1, 1)
    floors, floors_unf = {}, {}
    chunk = 100_000
    n = logits.shape[1]
    score_all = np.zeros((n, len(cfg.horizons)), dtype=np.float32)
    for lo in range(0, n, chunk):
        lh = logits[:, lo:lo + chunk].float() / T
        p = torch.softmax(lh, dim=-1).mean(0)          # ensemble-mean histogram
        for hi, k in enumerate(cfg.theta_bins):
            ph = p[:, hi]
            score_all[lo:lo + chunk, hi] = (ph[:, half + k:].sum(-1)
                                            - ph[:, :half - k].sum(-1)).numpy()
    for hi, hl in enumerate(HORIZON_LABELS):
        floors[hl] = (float(np.percentile(score_all[trad_sel, hi], SCORE_FLOOR_PCT))
                      if trad_sel.any() else None)
        floors_unf[hl] = (float(np.percentile(score_all[base_sel, hi], SCORE_FLOOR_PCT))
                          if base_sel.any() else None)
    print(f"[v5] score floors (p{SCORE_FLOOR_PCT:.0f}, tradability-filtered, "
          f"stopping-block anchors): {floors}")
    return floors, floors_unf


# ----------------------------------------------------------------- forecast

@torch.no_grad()
def make_forecast(members, frames, offsets, mm, eval_surface, cfg, temps, device, batch_size,
                  max_rows=None, log_every=250, forecast_dir=None):
    forecast_dir = Path(forecast_dir) if forecast_dir else FORECAST_DIR
    forecast_dir.mkdir(parents=True, exist_ok=True)
    total = sum(1 for a in eval_surface.values() if a)
    written, rows_done, t0 = 0, 0, time.time()
    print(f"[v5] forecasting {total} eval tickers "
          f"({'all OOS rows' if not max_rows else f'<= {max_rows} rows each'}) ...", flush=True)
    for ticker, anchors in eval_surface.items():
        if not anchors:
            continue
        if max_rows and len(anchors) > max_rows:
            anchors = anchors[-max_rows:]
        f = frames[ticker]
        start, _ = offsets[ticker]
        wins = []
        for a in anchors:
            lo = start + max(0, a - cfg.window + 1)
            real = np.asarray(mm[lo:start + a + 1], dtype=np.float32)
            nr = real.shape[0]
            if nr < cfg.window:
                pad = np.zeros((cfg.window - nr, N_FEATURES), dtype=np.float32)
                pad[:, fx.PAD_COL] = 1.0
                real = np.concatenate([pad, real], axis=0)
            wins.append(real)
        X = torch.from_numpy(np.stack(wins)).to(device)
        up, up_std, down, neutral, score, score_std, hist = [], [], [], [], [], [], []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            h, cls3, ustd, sstd = ensemble_predict(members, xb, cfg, temps=temps)
            up.append(cls3[..., 2]); down.append(cls3[..., 0]); neutral.append(cls3[..., 1])
            score.append(cls3[..., 2] - cls3[..., 0]); up_std.append(ustd); score_std.append(sstd)
            hist.append(h)
        cat = lambda parts: torch.cat(parts, 0).cpu().numpy()
        anchors = np.asarray(anchors)
        cols = build_forecast_columns(
            dates=f.dates[anchors], close=f.close[anchors], sigma_hat=f.sigma_hat[anchors],
            horizon_days=cfg.horizons,
            up=cat(up), up_std=cat(up_std), down=cat(down), neutral=cat(neutral),
            score=cat(score), score_std=cat(score_std), hist=cat(hist),
            bin_width=cfg.bin_width, n_bins=cfg.n_bins,
            volume=f.volume[anchors], tradable=f.tradable[anchors],
            vol_med63=f.vol_med63[anchors])
        import pandas as pd
        pd.DataFrame(cols).to_csv(forecast_dir / f"{ticker}_forecast.csv", index=False)
        written += 1
        rows_done += len(anchors)
        if written % log_every == 0:
            el = time.time() - t0
            rate = written / max(1e-9, el)
            eta = (total - written) / max(1e-9, rate)
            print(f"[v5] forecast {written}/{total} tickers | {rows_done} rows | "
                  f"{el / 60:.1f}m elapsed | {rate:.1f} tic/s | eta {eta / 60:.0f}m", flush=True)
    print(f"[v5] forecast complete | {written}/{total} tickers | {rows_done} rows | "
          f"{(time.time() - t0) / 60:.1f}m", flush=True)
    return written


# ------------------------------------------------------------------ artifacts

def _embargo_dates(bounds, horizons):
    """Latest anchor date per horizon whose label stays within the train / val split,
    written for the backtest to audit the embargo. Labels target index anchor+1+d,
    so the last safe anchor sits d+1 sessions before the split end."""
    u = bounds["unique_dates"]
    out = {}
    for split, end in (("train", bounds["train_end"]), ("val", bounds["val_end"])):
        pos = int(np.searchsorted(u, end, side="right")) - 1
        per_h = {}
        for d, label in zip(horizons, HORIZON_LABELS):
            ap = pos - (d + 1)
            per_h[label] = str(np.datetime_as_string(u[ap], unit="D")) if ap >= 0 else None
        out[split] = per_h
    return out


def write_artifacts(cfg, scaler, bounds, eval_mode, seed, ticker_frac, eval_tickers,
                    temps, seeds, run_dir, forecast_dir, extra_meta=None):
    run_dir = Path(run_dir)
    forecast_dir = Path(forecast_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(run_dir / "v5_norm.json")

    d = lambda x: str(np.datetime_as_string(np.datetime64(x), unit="D"))
    meta = {
        "feature_spec": fx.feature_spec_records(),
        "feature_names_hash": fx.feature_names_hash(),
        "window": cfg.window,
        "min_real_rows": cfg.min_real_rows,
        "horizons": list(HORIZON_LABELS),
        "horizon_days": list(cfg.horizons),
        "n_bins": cfg.n_bins,
        "bin_width": cfg.bin_width,
        "z_clip": cfg.z_clip,
        "theta_bins": list(cfg.theta_bins),
        "class_names": list(CLASS_NAMES),
        "n_members": len(seeds),
        "d_id": cfg.d_id, "d_mkt": cfg.d_mkt, "d_feed": cfg.d_feed,
        "p_cond": cfg.p_cond,
        "lam_cls": cfg.lam_cls,
        "staleness_k": STALENESS_K,
        "ewma_sigma_floor": fx.EWMA_SIGMA_FLOOR,
        "spike_log_threshold": fx.SPIKE_LOG_THRESHOLD,
        "temperatures": temps,
        "seeds": seeds,
        "seam_c_feed_spec": {
            "channels": ["insider_shares", "insider_amount", "insider_buy_flag",
                         "sentiment", "sentiment_change", "news_abnormal"],
            "lags_trading_days": {"insider": 4, "sentiment": 1},
            "status": "reserved",
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    (run_dir / "v5_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")

    split_info = {
        "eval_mode": eval_mode,
        "train_start": d(bounds["data_start"]), "train_end": d(bounds["train_end"]),
        "val_start": d(bounds["train_cut"]), "val_end": d(bounds["val_end"]),
        "data_end": d(bounds["data_end"]),
        "embargo": _embargo_dates(bounds, cfg.horizons),
        "ticker_holdout": {"seed": seed, "frac": ticker_frac,
                           "eval_tickers": sorted(eval_tickers)},
    }
    forecast_dir.mkdir(parents=True, exist_ok=True)
    if eval_mode in ("time", "both"):
        split_info["oos_start"] = d(bounds["train_val_cut"])
        (forecast_dir / "oos_start_date.txt").write_text(d(bounds["train_val_cut"]))
    (forecast_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description="Train the v5 return-prediction ensemble.")
    p.add_argument("--eval-mode", choices=["time", "ticker", "both"], default="time")
    p.add_argument("--ticker-holdout-frac", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
                   help="Trading sessions reserved for EACH of validation and OOS (fixed "
                        "chronological tails); all earlier sessions are training.")
    p.add_argument("--data-start", default=None, metavar="YYYY-MM-DD",
                   help="Truncate every ticker's OHLCV before the feature build (the "
                        "survivorship ablation start).")
    p.add_argument("--data-end", default=None, metavar="YYYY-MM-DD",
                   help="Truncate every ticker's OHLCV before the feature build, so "
                        "labels/sigma/features are computed as if the feed ended then "
                        "(the rolling-origin prerequisite).")
    p.add_argument("--run-dir", default=None,
                   help="Run directory; default claims the next numbered "
                        "models/v5/runs/rN. Protocol runs pass explicit dirs "
                        "(models/v5/runs/{nullc,ro1,ro2,ro3}) so finalizes never "
                        "clobber each other's scaler and metadata; consumers read "
                        "the promoted copies in models/v5 (tools/promote_run.py).")
    p.add_argument("--epochs", type=float, default=3.0,
                   help="Training budget in epochs (one pass over the train windows). The "
                        "step budget is derived from this so it auto-scales with the data.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard step-budget override; takes precedence over --epochs.")
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="Fixed eval interval override; default None uses the scheduled "
                        "cadence (250 steps <= 2k, 1000 <= 30k, 5000 after).")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop after this many evals without improvement.")
    p.add_argument("--lr-patience", type=int, default=4,
                   help="Evals on a plateau before a multiplicative LR drop.")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="Multiplicative LR drop applied on each plateau.")
    p.add_argument("--min-lr-scale", type=float, default=0.04,
                   help="Floor on the LR scale (LR floor = min-lr-scale * --lr).")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear LR warm-up length in steps.")
    p.add_argument("--judge-cal-cap", type=int, default=100_000,
                   help="Max stopping-side rows for the in-loop judged CE metric.")
    p.add_argument("--no-calibrate-eval", dest="calibrate_eval", action="store_false",
                   help="Judge the tie-breaker CE on raw logits (disable the transient "
                        "in-loop temperature fit).")
    p.set_defaults(calibrate_eval=True)
    p.add_argument("--val-subsample", type=int, default=150_000,
                   help="Fixed seeded validation fold size (per tag group) for evals "
                        "and the temperature fit.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--log-every", type=int, default=50, help="Steps between progress logs.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tickers", nargs="*", default=None, help="Explicit ticker list.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap train/val windows (smoke only; spread subset).")
    p.add_argument("--max-forecast-rows", type=int, default=None,
                   help="Cap forecast rows per ticker (most recent kept).")
    p.add_argument("--label-shuffle-within-date", action="store_true",
                   help="Null (c): permute mask-passing TRAIN labels within every "
                        "(session, horizon). The end-to-end input-side-leakage falsifier; "
                        "expect IC ~ 0 (a modest centered-CE beat is expected, not a "
                        "failure).")
    p.add_argument("--design-freeze", action="store_true",
                   help="Authorizes the one-shot OOS reads (the OOS class-rate row and "
                        "the scorer's OOS population).")
    # Multi-GPU ensemble split: run one process per GPU over disjoint member ranges,
    # each with its own --cache-dir, then a single --forecast-only pass to finalize.
    p.add_argument("--member-start", type=int, default=0,
                   help="Index of the first ensemble member this process trains.")
    p.add_argument("--member-count", type=int, default=None,
                   help="Members to train from --member-start (default: all remaining).")
    p.add_argument("--cache-dir", default=None,
                   help="Feature-memmap directory; give concurrent processes distinct dirs.")
    p.add_argument("--skip-forecast", action="store_true",
                   help="Train the member range only; skip temperatures/artifacts/forecasts.")
    p.add_argument("--forecast-only", action="store_true",
                   help="Skip training; load all members and write temperatures/artifacts/forecasts.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny model/universe/step budget for a CPU end-to-end check.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.members = args.members if args.members <= 2 else 2
        args.window, args.d_model, args.n_blocks = 130, 48, 2
        args.batch_size, args.max_steps = 32, 24
        args.eval_every_steps, args.patience = 8, 2
        args.lr_patience, args.warmup_steps = 1, 4
        args.eval_days = 40
        args.num_workers, args.log_every = 0, 8
        if args.max_tickers is None:
            args.max_tickers = 6
        if args.max_windows is None:
            args.max_windows = 512
        if args.max_forecast_rows is None:
            args.max_forecast_rows = 90

    if args.run_dir is None:
        args.run_dir = str(next_run_dir().relative_to(PROJECT_ROOT))
        print(f"[v5] claimed run dir {args.run_dir}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Non-default run dirs keep separate forecast dirs so rolling-origin runs do not
    # overwrite the primary run's CSVs.
    forecast_dir = (FORECAST_DIR if run_dir.resolve() == V5_MODEL_DIR.resolve()
                    else FORECAST_DIR / run_dir.name)

    # Run manifest: written by the member-start-0 trainer immediately after argument
    # resolution (so vars(args) records effective post-smoke values); never under
    # --forecast-only (the finalize only reads it for the snapshot assert).
    if not args.forecast_only and args.member_start == 0:
        write_run_manifest(args, run_dir)

    # Log the resolved device so a multi-GPU split can confirm each process pinned its
    # own card: under CUDA_VISIBLE_DEVICES=k the count is 1 and current is cuda:0.
    if args.device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"[v5] device=cuda | visible GPUs={torch.cuda.device_count()} | "
              f"current=cuda:{idx} ({torch.cuda.get_device_name(idx)})")
    elif args.device == "cuda":
        print("[v5] device=cuda requested but CUDA is unavailable -- placement will fail")
    else:
        print(f"[v5] device={args.device}")

    if args.eval_mode in ("ticker", "both") and args.ticker_holdout_frac <= 0:
        raise SystemExit("ticker/both eval modes require --ticker-holdout-frac > 0")

    if args.forecast_only:
        manifest = assert_finalize_snapshot(args, run_dir)
        print("[v5] finalize snapshot assert passed (data fingerprint, git revision, "
              "split-relevant args)")
    else:
        manifest = None

    cfg = V5Config(n_features=N_FEATURES, window=args.window, min_real_rows=MIN_REAL_ROWS,
                   d_model=args.d_model, n_blocks=args.n_blocks, horizons=tuple(HORIZON_DAYS))

    tickers = args.tickers or fx.list_us_tickers()
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print(f"[v5] building frames for {len(tickers)} tickers ...")
    frames = build_frames(tickers, args.data_start, args.data_end)
    print(f"[v5] usable frames: {len(frames)}")
    if not frames:
        raise SystemExit("no usable tickers (need > min_real_rows history)")

    # Reader-side manifest check for concurrent multi-GPU trainers, after the frame
    # build (minutes after launch, so the writer's startup write has landed).
    if not args.forecast_only and args.member_start != 0:
        check_run_manifest(args, run_dir)

    bounds = global_date_bounds(frames, args.eval_days)
    eval_tickers = assign_eval_tickers(sorted(frames), args.seed, args.ticker_holdout_frac)
    # Under ticker-only holdout a small fold of train tickers becomes the validation set.
    val_tickers = set()
    if args.eval_mode == "ticker":
        val_tickers = assign_eval_tickers(sorted(set(frames) - eval_tickers),
                                          args.seed + 1, max(0.1, args.ticker_holdout_frac))
    print(f"[v5] eval_mode={args.eval_mode} eval_tickers={len(eval_tickers)} "
          f"val_tickers={len(val_tickers)}")

    # Fit the scaler on training rows only (real rows, train split, training tickers).
    train_rows, train_dates = [], []
    for ticker, f in frames.items():
        if ticker in eval_tickers:
            continue
        split, _ = row_splits(f, bounds, args.eval_mode, False, ticker in val_tickers)
        sel = (split == 0)
        if sel.any():
            train_rows.append(f.features[sel])
            train_dates.append(f.dates[sel].astype("datetime64[ns]").astype(np.int64))
    if not train_rows:
        raise SystemExit("no training rows for scaler fit")
    train_matrix = np.concatenate(train_rows, axis=0)
    scaler = fx.RobustScaler.fit(train_matrix)
    print(f"[v5] scaler fit on {len(train_matrix)} train rows")
    if not args.forecast_only and args.member_start == 0:
        append_manifest_scaler_rows(run_dir, len(train_matrix))

    flagged = era_proxy_audit(train_matrix, np.concatenate(train_dates))
    print(f"[v5] era-proxy audit (|rho|>0.2): "
          + (", ".join(f"{n}={r:+.2f}" for n, r in flagged) if flagged else "none"))
    del train_matrix, train_rows, train_dates

    # Cross-sectional label centering (before the class-rate probe, which must
    # surface the centering-induced rate shift, and before the memmap build).
    k_min = min(50, math.ceil(len(frames) / 2))
    center_stats = center_labels(frames, bounds, args.eval_mode, eval_tickers, k_min)

    if args.label_shuffle_within_date:
        label_shuffle_within_date(frames, bounds, args.eval_mode, eval_tickers,
                                  val_tickers, args.seed)

    if args.eval_mode in ("time", "both"):
        class_rate_log(frames, bounds, cfg, args.eval_mode, eval_tickers, val_tickers,
                       design_freeze=args.design_freeze)

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    mm, offsets, train_ds, val_pack, panel_pack, eval_surface = build_memmap_and_samples(
        frames, scaler, bounds, args.eval_mode, eval_tickers, val_tickers, cfg.window,
        max_windows=args.max_windows, val_subsample=args.val_subsample, seed=args.seed,
        cache_dir=cache_dir, panel_seed=args.seed)
    val_ds, val_meta = val_pack
    print(f"[v5] train windows: {0 if train_ds is None else len(train_ds)} | "
          f"val windows (subsampled): {0 if val_ds is None else len(val_ds)} | "
          f"panel windows: {0 if panel_pack[0] is None else len(panel_pack[0])} | "
          f"eval tickers: {len(eval_surface)}")

    # Validation blocks on the global val sessions (int64 ns to match meta dates).
    u = bounds["unique_dates"]
    val_sessions = u[(u >= bounds["train_cut"]) & (u < bounds["train_val_cut"])]
    blocks, roles = mx.block_partition(
        val_sessions.astype("datetime64[ns]").astype(np.int64), 4)
    ic_floor = (mx.IC_MIN_NAMES if len(frames) >= 60
                else min(mx.IC_MIN_NAMES, max(3, math.ceil(len(frames) / 2))))
    if ic_floor != mx.IC_MIN_NAMES:
        print(f"[v5] small universe ({len(frames)} tickers): stopping-side IC name "
              f"floor scaled to {ic_floor} (gate reads always use {mx.IC_MIN_NAMES})")

    # Member seeds are global (indexed by member id), so a multi-GPU split that trains
    # disjoint ranges still produces distinct init + shuffle per member.
    seeds = [args.seed + 100 * i for i in range(args.members)]

    if args.forecast_only:
        members = []
        for i in range(args.members):
            ckpt = run_dir / f"member_{i}.pt"
            if not ckpt.exists():
                raise SystemExit(f"--forecast-only needs every member; missing {ckpt}")
            model = V5Backbone(cfg)
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
            members.append(model.to(args.device).eval())
        print(f"[v5] forecast-only: loaded {len(members)} members from {run_dir}")
    else:
        if train_ds is None:
            raise SystemExit("no training windows")
        start = args.member_start
        end = min(start + (args.member_count or args.members), args.members)
        if not 0 <= start < end <= args.members:
            raise SystemExit(f"invalid member range [{start}, {end}) for --members {args.members}")
        steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
        max_steps = args.max_steps if args.max_steps else math.ceil(args.epochs * steps_per_epoch)
        if args.patience <= args.lr_patience:
            print(f"[v5] WARNING: --patience ({args.patience}) <= --lr-patience "
                  f"({args.lr_patience}); early-stop may fire before the LR anneals.")
        cadence = (f"every {args.eval_every_steps} steps" if args.eval_every_steps
                   else "scheduled (250<=2k, 1000<=30k, 5000 after)")
        print(f"[v5] schedule | {steps_per_epoch} steps/epoch | budget {max_steps} steps "
              f"(~{max_steps / steps_per_epoch:.1f} ep) | warmup {args.warmup_steps} | "
              f"plateau LR x{args.lr_factor} after {args.lr_patience} no-improve evals, "
              f"floor {args.min_lr_scale} | eval {cadence} | "
              f"patience {args.patience} | workers {args.num_workers} | "
              f"members [{start}, {end})")

        base_stop = None
        comp_text = None
        if val_ds is not None:
            tag = np.asarray(val_meta["tag"])
            rows_tag0 = np.flatnonzero(tag == 0)
            base_tag0 = val_marginal_baseline_ce(val_ds, cfg, rows=rows_tag0)
            print(f"[v5] val constant-marginal baseline CE (tag-0) = {base_tag0:.4f} "
                  f"({rows_tag0.size} rows; the model must beat this to be learning "
                  "anything conditional)")
            if (tag == 1).any():
                rows_tag1 = np.flatnonzero(tag == 1)
                print(f"[v5] val constant-marginal baseline CE (tag-1 holdout) = "
                      f"{val_marginal_baseline_ce(val_ds, cfg, rows=rows_tag1):.4f} "
                      f"({rows_tag1.size} rows; report-only)")
            # Startup baseline flavors with per-horizon components (both flavors --
            # millinat verdicts must not hinge on the flavor).
            train_z, train_m = None, None
            if train_ds is not None and len(train_ds) > 0:
                zt = np.asarray(train_ds.z, dtype=np.float32)
                mt = np.asarray(train_ds.mask, dtype=np.float32)
                rng = np.random.default_rng(args.seed)
                want = min(len(zt), max(1, int(1_000_000 / max(1, mt.sum() / len(mt)))))
                pick = rng.choice(len(zt), size=want, replace=False) if want < len(zt) else np.arange(len(zt))
                train_z, train_m = zt[pick], mt[pick]
            vb = val_baselines(val_ds, val_meta, cfg, train_z, train_m, rows=rows_tag0)
            for flavor_name, comp in vb.items():
                per_h = " ".join(f"{HORIZON_LABELS[h]}:{comp['per_h'][h]:.4f}"
                                 if np.isfinite(comp["per_h"][h]) else f"{HORIZON_LABELS[h]}:-"
                                 for h in range(4))
                print(f"[v5] baseline[{flavor_name}] total {comp['total']:.4f} | {per_h}")
            # Stopping-side baseline on the judged score half: identical rows.
            m_stop, stop_rows = stopping_selection(val_meta, val_ds.mask, blocks, roles)
            if stop_rows.size:
                capped = _spread_indices(stop_rows.size, args.judge_cal_cap)
                parity = np.arange(capped.size) % 2
                score_half = stop_rows[capped[parity == 1]]
                base_stop = val_marginal_baseline_ce_subset(
                    val_ds, cfg, score_half, mask_override=m_stop, horizons=(0, 1))
                print(f"[v5] stopping-side baseline CE = {base_stop:.4f} "
                      f"({score_half.size} score-half rows, {{1d,1w}} interval labels)")
            comp_text, _ = comparator_report(val_meta, val_ds, ic_floor)

        members = []
        for i in range(start, end):
            if comp_text:
                print(comp_text)
            model, crit = train_member(
                i, seeds[i], cfg, train_ds, val_ds, val_meta, args.device, max_steps,
                args.eval_every_steps, args.patience, args.lr, args.batch_size,
                steps_per_epoch, args.num_workers, args.log_every,
                blocks=blocks, roles=roles, ic_floor=ic_floor, run_dir=run_dir,
                panel=panel_pack, base_stop=base_stop,
                lr_patience=args.lr_patience, lr_factor=args.lr_factor,
                min_lr_scale=args.min_lr_scale, warmup_steps=args.warmup_steps,
                calibrate=args.calibrate_eval, cal_cap=args.judge_cal_cap)
            model.eval()
            torch.save(model.state_dict(), run_dir / f"member_{i}.pt")
            members.append(model)
            print(f"[v5] member {i} (seed {seeds[i]}) selection criterion = {crit:+.4f}")
        if start != 0 or end != args.members or args.skip_forecast:
            print(f"[v5] trained members [{start}, {end}); checkpoints in {run_dir}. "
                  f"Once all {args.members} exist, run --forecast-only (same --seed/--members/"
                  f"universe) to write temperatures, artifacts, and forecasts.")
            print("[v5] done (partial run).")
            return

    temps, temp_cache = fit_temperatures(members, val_ds, val_meta, cfg, args.device,
                                         args.batch_size, args.num_workers, blocks, roles)
    print(f"[v5] temperatures: {[round(t, 3) for t in temps]}")
    floors, floors_unf = compute_score_floors(temp_cache, temps, val_meta, cfg,
                                              blocks, roles)
    del temp_cache

    parity_num = None
    if args.device == "cuda":
        try:
            from tools.score_checkpoints import mamba_parity_check
            batches = []
            ploader = DataLoader(val_ds, batch_size=args.batch_size)
            for bi, (x, _, _) in enumerate(ploader):
                batches.append(x)
                if bi == 2:
                    break
            parity_num = mamba_parity_check(run_dir / "member_0.pt", cfg, batches,
                                            args.device)
        except Exception as exc:
            print(f"[v5] parity check skipped: {type(exc).__name__}: {exc}")
    else:
        print("[v5] parity check skipped: CPU device")

    extra_meta = {"label_centering": center_stats,
                  "score_floors": floors,
                  "score_floors_unfiltered": floors_unf,
                  "mamba_parity_max_abs": parity_num}
    manifest_path = run_dir / "run_manifest.json"
    if manifest is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest is not None:
        extra_meta["run_manifest"] = manifest

    write_artifacts(cfg, scaler, bounds, args.eval_mode, args.seed,
                    args.ticker_holdout_frac, eval_tickers, temps, seeds,
                    run_dir, forecast_dir, extra_meta)
    surface = {t: ("holdout" if t in eval_tickers else "time-only")
               for t in sorted(eval_surface)}
    forecast_dir.mkdir(parents=True, exist_ok=True)
    (forecast_dir / "surface_manifest.json").write_text(
        json.dumps(surface, indent=2), encoding="utf-8")
    n_fc = make_forecast(members, frames, offsets, mm, eval_surface, cfg, temps,
                         args.device, args.batch_size, max_rows=args.max_forecast_rows,
                         forecast_dir=forecast_dir)
    print(f"[v5] wrote {n_fc} forecast CSVs to {forecast_dir}")
    if args.design_freeze:
        print("[v5] DESIGN FREEZE: for the one-shot OOS read run "
              f"tools/score_checkpoints.py --run-dir {run_dir} --population oos "
              "--design-freeze --ensemble (examined once per freeze)")
    print("[v5] done.")


if __name__ == "__main__":
    main()
