"""Train the deep-ensemble return-prediction backbone and export all artifacts.

Pipeline:

1. Select a US ticker universe and build per-ticker feature frames.
2. Resolve splits along the active holdout axes (time, ticker, or both; at least one
   is required) and the per-row split boundaries used for the label embargo.
3. Fit the pooled robust-z scaler on training rows only, normalize every ticker, and
   write one concatenated fp16 memmap plus a per-ticker offset index.
4. Train ``M`` ensemble members (different seeds and shuffles) with step-anchored
   early stopping, then fit per-horizon calibration temperatures on the validation
   cell.
5. Write the model checkpoints, the feature/normalization/config metadata, the split
   description (with per-horizon embargo dates and the eval mode), and one forecast
   CSV per ticker on the reported evaluation surface.

The defaults train a production-sized model; ``--smoke`` shrinks every dimension so
the full pipeline runs on a CPU in seconds for verification.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import features_v5 as fx
from features_v5 import HORIZON_DAYS, MIN_REAL_ROWS, N_FEATURES
from v5_backbone import (V5Backbone, V5Config, ensemble_predict, hl_gauss_targets,
                         optimizer_param_groups, v5_loss)
from v5.forecast import HORIZON_LABELS, build_forecast_columns

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
V5_MODEL_DIR = MODELS_DIR / "v5"
FORECAST_DIR = PROJECT_ROOT / "forecasts"
CACHE_DIR = PROJECT_ROOT / "cache" / "v5"
CLASS_NAMES = ("down", "neutral", "up")
STALENESS_K = 63
DEFAULT_EVAL_DAYS = 254          # trading sessions reserved for EACH of val and oos (fixed tails)


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


def build_frames(tickers):
    fear_greed = fx.load_fear_greed()
    ohlcv = fx.load_us_ohlcv(tickers=tickers)
    frames = {}
    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_REAL_ROWS + 1:
            continue
        frames[ticker] = fx.build_feature_frame(ticker, df, fear_greed)
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


def class_rate_log(frames, bounds, cfg):
    """Per-split realized class rates (the threshold-drift probe). Computed from the
    bucketized targets, ignoring masked entries."""
    half_theta = [k * cfg.bin_width for k in cfg.theta_bins]
    counts = {s: np.zeros((len(cfg.horizons), 3)) for s in ("train", "val", "oos")}
    name = {0: "train", 1: "val", 2: "oos"}
    for f in frames.values():
        split, _ = row_splits(f, bounds, "time", False, False)
        for hi, th in enumerate(half_theta):
            z = f.z[:, hi]
            cls = np.where(np.isnan(z), -1, (z > -th).astype(int) + (z > th).astype(int))
            for s_id, s_name in name.items():
                sel = (split == s_id) & (cls >= 0)
                for c in range(3):
                    counts[s_name][hi, c] += int(((cls == c) & sel).sum())
    rates = {}
    for s, c in counts.items():
        tot = c.sum(1, keepdims=True)
        rates[s] = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    return rates


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
    embargo so the split-end is the data end.
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
            split[is_oos_time] = 2
        else:
            split[is_train_time] = 0
            split[is_val_time] = 1
    return split, split_end


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
    stays representative across tickers and dates."""
    n = len(b["start"])
    if not cap or n <= cap:
        return b
    keep = np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
    return {k: [b[k][i] for i in keep] for k in b}


def build_memmap_and_samples(frames, scaler, bounds, eval_mode, eval_tickers, val_tickers,
                             window, max_windows=None, val_subsample=None, seed=0, cache_dir=None):
    """Normalize each ticker, write the concatenated fp16 memmap, and enumerate the
    train / val samples and the per-ticker eval surface. ``cache_dir`` isolates the
    memmap so concurrent processes (one per GPU) do not write the same file."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_rows = sum(len(f.dates) for f in frames.values())
    mm_path = cache_dir / "features_f16.dat"
    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(total_rows, N_FEATURES))

    offsets, cursor = {}, 0
    train, val = {"start": [], "nreal": [], "z": [], "m": []}, {"start": [], "nreal": [], "z": [], "m": []}
    eval_surface = {}     # ticker -> list of local anchor indices
    for ticker, f in frames.items():
        norm = scaler.transform(f.features)
        n = len(f.dates)
        mm[cursor:cursor + n] = norm.astype(np.float16)
        offsets[ticker] = (cursor, n)
        split, split_end = row_splits(f, bounds, eval_mode,
                                      ticker in eval_tickers, ticker in val_tickers)
        mask = fx.label_mask(f.spike_free, f.target_dates, split_end)
        anchors = np.arange(MIN_REAL_ROWS - 1, n)
        for a in anchors:
            if mask[a].sum() == 0:
                continue
            s = int(split[a])
            if s == 2:
                eval_surface.setdefault(ticker, []).append(int(a))
                continue
            bucket = train if s == 0 else (val if s == 1 else None)
            if bucket is None:
                continue
            lo = cursor + max(0, a - window + 1)
            bucket["start"].append(lo)
            bucket["nreal"].append(min(a + 1, window))
            bucket["z"].append(f.z[a])
            bucket["m"].append(mask[a])
        cursor += n
    mm.flush()

    def pack(b, cap):
        b = _subsample_bucket(b, cap, seed)
        if not b["start"]:
            return None
        return WindowDataset(mm, np.asarray(b["start"], dtype=np.int64),
                             np.asarray(b["nreal"], dtype=np.int64),
                             np.asarray(b["z"], dtype=np.float32),
                             np.asarray(b["m"], dtype=np.float32), window)

    # Validation is held to a fixed subsample: a full-universe val set is millions of
    # windows, which would make every early-stop eval and the temperature fit ruinously
    # slow (and the temperature fit would materialize all val logits and OOM).
    val_caps = [c for c in (max_windows, val_subsample) if c]
    val_cap = min(val_caps) if val_caps else None
    return mm, offsets, pack(train, max_windows), pack(val, val_cap), eval_surface


# ------------------------------------------------------------------ training

def train_member(member_idx, seed, cfg, train_ds, val_ds, device, max_steps,
                 eval_every_steps, patience, lr, t_max, batch_size, steps_per_epoch,
                 num_workers, log_every, val_base=None, *, lr_patience=4, lr_factor=0.5,
                 min_lr_scale=0.04, warmup_steps=1000, calibrate=True, cal_cap=100_000):
    torch.manual_seed(seed)
    model = V5Backbone(cfg).to(device)
    opt = torch.optim.AdamW(optimizer_param_groups(model, weight_decay=0.01),
                            lr=lr, betas=(0.9, 0.95))

    # Plateau LR schedule (Defect-1 fix): warm-up, then a constant scale the eval block
    # multiplies down whenever validation stalls. There is no t_max horizon -> decay is
    # driven by the same val signal as early-stop and cannot decouple from the run length.
    lr_scale = 1.0
    plateau_since_drop = 0

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return lr_scale

    pin = device == "cuda"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0,
                        pin_memory=pin, generator=torch.Generator().manual_seed(seed))
    # bf16 autocast on GPU only (the selective scan keeps fp32 internally); a no-op on
    # CPU, where bf16 autocast is unsupported and the fallback block runs fp32.
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())
    best_val, best_state, since_improve, step = float("inf"), None, 0, 0
    loss_ema, t0, stop = None, time.time(), False
    print(f"[member {member_idx}] start | {len(train_ds)} train / "
          f"{0 if val_ds is None else len(val_ds)} val windows | {steps_per_epoch} steps/epoch | "
          f"eval every {eval_every_steps} steps | max {max_steps} steps | patience {patience} | "
          f"warmup {warmup_steps} | plateau LRx{lr_factor}@{lr_patience} floor {min_lr_scale} | "
          f"{'calibrated' if calibrate else 'raw'} judging")
    model.train()
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
                loss = v5_loss(model(x), z, m, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % log_every == 0:
                sps = step / max(1e-9, time.time() - t0)
                print(f"[member {member_idx}] step {step}/{max_steps} (ep {step / steps_per_epoch:.2f}) | "
                      f"loss {loss_ema:.4f} | lr {cur_lr:.2e} | {sps:.1f} it/s | "
                      f"eta {(max_steps - step) / max(1e-9, sps) / 60:.0f}m")
            if step % eval_every_steps == 0 or step >= max_steps:
                if val_ds is not None and calibrate:
                    vce, raw_ce, jt = judged_val_ce(model, val_ds, cfg, device, batch_size,
                                                    num_workers, cal_cap)
                elif val_ds is not None:
                    vce, raw_ce, jt = evaluate_ce(model, val_ds, cfg, device, batch_size,
                                                  num_workers), None, None
                else:
                    vce, raw_ce, jt = lv, None, None
                improved = vce < best_val - 1e-5
                if improved:
                    best_val = vce
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    since_improve = 0
                    plateau_since_drop = 0
                else:
                    since_improve += 1
                    plateau_since_drop += 1
                    # Defect-1 fix: drop LR on a val plateau; since_improve is NOT reset by a
                    # drop, so early-stop still terminates at exactly `patience` evals.
                    if plateau_since_drop >= lr_patience and lr_scale > min_lr_scale:
                        lr_scale = max(min_lr_scale, lr_scale * lr_factor)
                        plateau_since_drop = 0
                        print(f"[member {member_idx}] LR DROP -> {lr * lr_scale:.2e} "
                              f"(scale {lr_scale:.3f}) after {lr_patience} no-improve evals")
                raw_str = "" if raw_ce is None else f" | raw {raw_ce:.4f}"
                t_str = "" if jt is None else f" | T {[round(t, 2) for t in jt]}"
                base_str = ("" if val_base is None
                            else f" | base {val_base:.4f} {'BEAT' if vce < val_base else 'MISS'}")
                print(f"[member {member_idx}] EVAL step {step} (ep {step / steps_per_epoch:.2f}) | "
                      f"val_ce {vce:.4f} | best {best_val:.4f}{raw_str}{t_str}{base_str} | "
                      f"lr_scale {lr_scale:.3f} | "
                      f"{'IMPROVED' if improved else f'no-improve {since_improve}/{patience}'}")
                if since_improve >= patience or step >= max_steps:
                    stop = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[member {member_idx}] done | {step} steps | best val CE {best_val:.4f} | "
          f"{(time.time() - t0) / 60:.1f} min")
    return model, best_val


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


@torch.no_grad()
def judged_val_ce(model, ds, cfg, device, batch_size, num_workers=0, cal_cap=100_000):
    """Calibration-fair judged val CE (Defect-2 fix). Forwards a bounded, evenly-strided
    subsample of the val fold, fits a per-horizon temperature on one half by grid search
    (reusing :func:`v5_loss` per horizon, so the metric cannot drift from the training
    loss), and scores the calibrated masked loss on the other half -- making the judged
    number commensurable with the penalty-free :func:`val_marginal_baseline_ce`. The
    temperatures are TRANSIENT (judging only) and never persisted; ``fit_temperatures`` and
    the saved checkpoints are untouched. Returns ``(calibrated_ce, raw_ce, temps)``;
    all-ones temps reproduce ``raw_ce`` (T=1 lies on the grid).
    """
    H = len(cfg.horizons)
    was_training = model.training
    model.eval()
    n = len(ds)
    if n == 0:
        if was_training:
            model.train()
        return float("inf"), float("inf"), [1.0] * H
    if cal_cap and n > cal_cap:                          # the fold is ticker-ordered, so an
        ds = Subset(ds, list(range(0, n, max(1, n // cal_cap)))[:cal_cap])  # even stride, not first-N
    pin = device == "cuda"
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    L, Z, M = [], [], []
    for x, z, m in loader:
        L.append(model(x.to(device, non_blocking=pin)))
        Z.append(z.to(device, non_blocking=pin))
        M.append(m.to(device, non_blocking=pin))
    if was_training:
        model.train()
    logits, z, m = torch.cat(L, 0), torch.cat(Z, 0), torch.cat(M, 0)
    raw_ce = v5_loss(logits, z, m, cfg).item()
    parity = torch.arange(logits.shape[0], device=logits.device) % 2   # deterministic half-split
    fit, score = parity == 0, parity == 1
    if float(m[fit].sum()) < 1 or float(m[score].sum()) < 1:           # too few labels to calibrate
        return raw_ce, raw_ce, [1.0] * H
    grid = torch.linspace(0.5, 5.0, 46)                               # same grid as fit_temperatures
    lf, zf, mf = logits[fit], z[fit], m[fit]
    temps, Tvec = [], torch.ones(H, device=logits.device)
    for h in range(H):
        if float(mf[:, h].sum()) < 1:
            temps.append(1.0)
            continue
        m_h = torch.zeros_like(mf)
        m_h[:, h] = mf[:, h]                                          # isolate horizon h via the mask
        best_T, best = 1.0, float("inf")
        for T in grid:
            Tvec[h] = float(T)
            v = v5_loss(lf / Tvec.view(1, -1, 1), zf, m_h, cfg).item()
            if v < best:
                best, best_T = v, float(T)
        Tvec[h] = 1.0
        temps.append(best_T)
    Ts = torch.tensor(temps, device=logits.device)
    cal_ce = v5_loss(logits[score] / Ts.view(1, -1, 1), z[score], m[score], cfg).item()
    return cal_ce, raw_ce, temps


@torch.no_grad()
def val_marginal_baseline_ce(val_ds, cfg, lam_cls=0.1):
    """Loss of the best input-ignoring predictor on the val fold: the per-horizon marginal
    histogram (its mean HL-Gauss soft target) plus the marginal 3-class term, mask-weighted
    exactly like :func:`evaluate_ce`. The trained model must beat this to be learning
    anything conditional; a run that sits at this line (or at ``log(n_bins)``) has collapsed
    to the prior. Logged every eval so that failure mode is visible immediately.
    """
    z = torch.from_numpy(np.asarray(val_ds.z, dtype=np.float32))
    m = torch.from_numpy(np.asarray(val_ds.mask, dtype=np.float32))
    soft = hl_gauss_targets(torch.nan_to_num(z), cfg)                  # (N, H, n_bins)
    theta = torch.tensor(cfg.theta_bins, dtype=z.dtype) * cfg.bin_width
    znz = torch.nan_to_num(z)
    cls = (znz > -theta).long() + (znz > theta).long()                # (N, H)
    total, count = 0.0, 0.0
    for h in range(len(cfg.horizons)):
        mh = m[:, h]
        nh = float(mh.sum())
        if nh < 1:
            continue
        q = (soft[:, h] * mh[:, None]).sum(0) / nh                    # mean soft target
        h_hist = float(-(q * q.clamp_min(1e-12).log()).sum())
        rate = torch.stack([((cls[:, h] == c).float() * mh).sum() / nh for c in range(3)])
        h_cls = float(-(rate * rate.clamp_min(1e-12).log()).sum())
        total += nh * (h_hist + lam_cls * h_cls)
        count += nh
    return total / count if count else float("inf")


# -------------------------------------------------------------- calibration

@torch.no_grad()
def fit_temperatures(members, val_ds, cfg, device, batch_size, num_workers=0):
    """Per-horizon temperatures minimizing the masked 3-class NLL of the ensemble mean.

    Logits are cached once over the (subsampled) validation fold; the 1-D search per
    horizon is then pure arithmetic. The fold is bounded by ``--val-subsample`` so this
    cache cannot grow to the full universe and OOM.
    """
    if val_ds is None or len(val_ds) == 0:
        return [1.0] * len(cfg.horizons)
    print(f"[v5] fitting temperatures on {len(val_ds)} val windows ...")
    loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device == "cuda")
    logits_all, z_all, m_all = [], [], []
    for x, z, m in loader:
        x = x.to(device)
        logits_all.append(torch.stack([mem(x) for mem in members]).cpu())   # (M,B,H,n_bins)
        z_all.append(z)
        m_all.append(m)
    logits = torch.cat(logits_all, dim=1)
    z = torch.cat(z_all, 0)
    m = torch.cat(m_all, 0)
    half = cfg.n_bins // 2
    temps = []
    grid = torch.linspace(0.5, 5.0, 46)
    for hi, k in enumerate(cfg.theta_bins):
        th = k * cfg.bin_width
        zf = torch.nan_to_num(z[:, hi], nan=0.0)
        true_cls = (zf > -th).long() + (zf > th).long()
        mh = m[:, hi].bool()
        if mh.sum() == 0:
            temps.append(1.0)
            continue
        best_T, best_nll = 1.0, float("inf")
        for T in grid:
            p = torch.softmax(logits[:, :, hi] / T, dim=-1).mean(0)        # (B,n_bins)
            cls3 = torch.stack([p[:, :half - k].sum(-1), p[:, half - k:half + k].sum(-1),
                                p[:, half + k:].sum(-1)], -1).clamp_min(1e-8)
            nll = torch.nn.functional.nll_loss(cls3.log()[mh], true_cls[mh])
            if nll.item() < best_nll:
                best_nll, best_T = nll.item(), float(T)
        temps.append(best_T)
    return temps


# ----------------------------------------------------------------- forecast

@torch.no_grad()
def make_forecast(members, frames, offsets, mm, eval_surface, cfg, temps, device, batch_size,
                  max_rows=None, log_every=250):
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
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
            bin_width=cfg.bin_width, n_bins=cfg.n_bins)
        import pandas as pd
        pd.DataFrame(cols).to_csv(FORECAST_DIR / f"{ticker}_forecast.csv", index=False)
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
    written for the backtest to audit the embargo."""
    u = bounds["unique_dates"]
    out = {}
    for split, end in (("train", bounds["train_end"]), ("val", bounds["val_end"])):
        pos = int(np.searchsorted(u, end, side="right")) - 1
        per_h = {}
        for d, label in zip(horizons, HORIZON_LABELS):
            ap = pos - d
            per_h[label] = str(np.datetime_as_string(u[ap], unit="D")) if ap >= 0 else None
        out[split] = per_h
    return out


def write_artifacts(cfg, scaler, bounds, eval_mode, seed, ticker_frac, eval_tickers, temps, seeds):
    MODELS_DIR.mkdir(exist_ok=True)
    V5_MODEL_DIR.mkdir(exist_ok=True)
    scaler.save(MODELS_DIR / "v5_norm.json")

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
    (MODELS_DIR / "v5_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (V5_MODEL_DIR / "config.json").write_text(
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
    if eval_mode in ("time", "both"):
        split_info["oos_start"] = d(bounds["train_val_cut"])
        (FORECAST_DIR / "oos_start_date.txt").write_text(d(bounds["train_val_cut"]))
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    (FORECAST_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")


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
    p.add_argument("--epochs", type=float, default=3.0,
                   help="Training budget in epochs (one pass over the train windows). The "
                        "step budget is derived from this so it auto-scales with the data.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard step-budget override; takes precedence over --epochs.")
    p.add_argument("--t-max", type=int, default=None,
                   help="Logged-only step horizon kept for the --smoke/--t-max paths; the "
                        "plateau LR schedule no longer reads it.")
    p.add_argument("--eval-every-steps", type=int, default=2000,
                   help="Validate every N optimizer steps. Step-based so the cadence does "
                        "not silently coarsen as the train set grows.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop after this many evals without improvement.")
    p.add_argument("--lr-patience", type=int, default=4,
                   help="Evals on a val plateau before a multiplicative LR drop.")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="Multiplicative LR drop applied on each plateau.")
    p.add_argument("--min-lr-scale", type=float, default=0.04,
                   help="Floor on the LR scale (LR floor = min-lr-scale * --lr).")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear LR warm-up length in steps.")
    p.add_argument("--judge-cal-cap", type=int, default=100_000,
                   help="Max val rows for the in-loop calibrated judging metric (bounded "
                        "regardless of --val-subsample).")
    p.add_argument("--no-calibrate-eval", dest="calibrate_eval", action="store_false",
                   help="Judge val CE on raw logits (disable in-loop temperature calibration).")
    p.set_defaults(calibrate_eval=True)
    p.add_argument("--val-subsample", type=int, default=150_000,
                   help="Fixed seeded validation fold size for early-stop and temperature fit.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--log-every", type=int, default=50, help="Steps between progress logs.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tickers", nargs="*", default=None, help="Explicit ticker list.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap train/val windows (smoke only; spread subset).")
    p.add_argument("--max-forecast-rows", type=int, default=None,
                   help="Cap forecast rows per ticker (most recent kept).")
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
        args.batch_size, args.max_steps, args.t_max = 32, 24, 24
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

    MODELS_DIR.mkdir(exist_ok=True)
    V5_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    # Log the resolved device so a multi-GPU split can confirm each process pinned its
    # own card: under CUDA_VISIBLE_DEVICES=k the count is 1 and current is cuda:0.
    if args.device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"[v5] device=cuda | visible GPUs={torch.cuda.device_count()} | "
              f"current=cuda:{idx} ({torch.cuda.get_device_name(idx)})")
    elif args.device == "cuda":
        print("[v5] device=cuda requested but CUDA is unavailable — placement will fail")
    else:
        print(f"[v5] device={args.device}")

    if args.eval_mode == "ticker" and args.ticker_holdout_frac <= 0:
        raise SystemExit("ticker eval mode requires --ticker-holdout-frac > 0")
    if args.eval_mode in ("ticker", "both") and args.ticker_holdout_frac <= 0:
        raise SystemExit("ticker/both eval modes require --ticker-holdout-frac > 0")

    cfg = V5Config(n_features=N_FEATURES, window=args.window, min_real_rows=MIN_REAL_ROWS,
                   d_model=args.d_model, n_blocks=args.n_blocks, horizons=tuple(HORIZON_DAYS))

    tickers = args.tickers or fx.list_us_tickers()
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print(f"[v5] building frames for {len(tickers)} tickers ...")
    frames = build_frames(tickers)
    print(f"[v5] usable frames: {len(frames)}")
    if not frames:
        raise SystemExit("no usable tickers (need > min_real_rows history)")

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

    flagged = era_proxy_audit(train_matrix, np.concatenate(train_dates))
    print(f"[v5] era-proxy audit (|rho|>0.2): "
          + (", ".join(f"{n}={r:+.2f}" for n, r in flagged) if flagged else "none"))
    if args.eval_mode == "time":
        rates = class_rate_log(frames, bounds, cfg)
        for s in ("train", "val", "oos"):
            per_h = "  ".join(f"{lbl}:{rates[s][hi].round(2).tolist()}"
                              for hi, lbl in enumerate(HORIZON_LABELS))
            print(f"[v5] class rates [{s}] (down/neutral/up): {per_h}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    mm, offsets, train_ds, val_ds, eval_surface = build_memmap_and_samples(
        frames, scaler, bounds, args.eval_mode, eval_tickers, val_tickers, cfg.window,
        max_windows=args.max_windows, val_subsample=args.val_subsample, seed=args.seed,
        cache_dir=cache_dir)
    print(f"[v5] train windows: {0 if train_ds is None else len(train_ds)} | "
          f"val windows (subsampled): {0 if val_ds is None else len(val_ds)} | "
          f"eval tickers: {len(eval_surface)}")

    # Member seeds are global (indexed by member id), so a multi-GPU split that trains
    # disjoint ranges still produces distinct init + shuffle per member.
    seeds = [args.seed + 100 * i for i in range(args.members)]

    if args.forecast_only:
        members = []
        for i in range(args.members):
            ckpt = V5_MODEL_DIR / f"member_{i}.pt"
            if not ckpt.exists():
                raise SystemExit(f"--forecast-only needs every member; missing {ckpt}")
            model = V5Backbone(cfg)
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
            members.append(model.to(args.device).eval())
        print(f"[v5] forecast-only: loaded {len(members)} members from {V5_MODEL_DIR}")
    else:
        if train_ds is None:
            raise SystemExit("no training windows")
        start = args.member_start
        end = min(start + (args.member_count or args.members), args.members)
        if not 0 <= start < end <= args.members:
            raise SystemExit(f"invalid member range [{start}, {end}) for --members {args.members}")
        steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
        max_steps = args.max_steps if args.max_steps else math.ceil(args.epochs * steps_per_epoch)
        t_max = args.t_max if args.t_max else max_steps
        eval_every_steps = max(1, args.eval_every_steps)
        val_base = val_marginal_baseline_ce(val_ds, cfg) if val_ds is not None else None
        if args.patience <= args.lr_patience:
            print(f"[v5] WARNING: --patience ({args.patience}) <= --lr-patience "
                  f"({args.lr_patience}); early-stop may fire before the LR anneals.")
        print(f"[v5] schedule | {steps_per_epoch} steps/epoch | budget {max_steps} steps "
              f"(~{max_steps / steps_per_epoch:.1f} ep) | warmup {args.warmup_steps} | "
              f"plateau LR x{args.lr_factor} after {args.lr_patience} no-improve evals, "
              f"floor {args.min_lr_scale} | eval every {eval_every_steps} steps | "
              f"patience {args.patience} | {'calibrated' if args.calibrate_eval else 'raw'} judging "
              f"(cap {args.judge_cal_cap}) | workers {args.num_workers} | members [{start}, {end})")
        if val_base is not None:
            print(f"[v5] val constant-marginal baseline CE = {val_base:.4f} "
                  f"(model must beat this to be learning anything conditional)")
        members = []
        for i in range(start, end):
            model, vce = train_member(i, seeds[i], cfg, train_ds, val_ds, args.device,
                                       max_steps, eval_every_steps, args.patience, args.lr,
                                       t_max, args.batch_size, steps_per_epoch,
                                       args.num_workers, args.log_every, val_base,
                                       lr_patience=args.lr_patience, lr_factor=args.lr_factor,
                                       min_lr_scale=args.min_lr_scale, warmup_steps=args.warmup_steps,
                                       calibrate=args.calibrate_eval, cal_cap=args.judge_cal_cap)
            model.eval()
            torch.save(model.state_dict(), V5_MODEL_DIR / f"member_{i}.pt")
            members.append(model)
            print(f"[v5] member {i} (seed {seeds[i]}) best val CE = {vce:.4f}")
        if start != 0 or end != args.members or args.skip_forecast:
            print(f"[v5] trained members [{start}, {end}); checkpoints in {V5_MODEL_DIR}. "
                  f"Once all {args.members} exist, run --forecast-only (same --seed/--members/"
                  f"universe) to write temperatures, artifacts, and forecasts.")
            print("[v5] done (partial run).")
            return

    temps = fit_temperatures(members, val_ds, cfg, args.device, args.batch_size, args.num_workers)
    print(f"[v5] temperatures: {[round(t, 3) for t in temps]}")
    write_artifacts(cfg, scaler, bounds, args.eval_mode, args.seed, args.ticker_holdout_frac,
                    eval_tickers, temps, seeds)
    n_fc = make_forecast(members, frames, offsets, mm, eval_surface, cfg, temps,
                         args.device, args.batch_size, max_rows=args.max_forecast_rows)
    print(f"[v5] wrote {n_fc} forecast CSVs to {FORECAST_DIR}")
    print("[v5] done.")


if __name__ == "__main__":
    main()
