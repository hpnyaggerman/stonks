"""Score trained v5 checkpoints on a chosen population with the full metric suite.

One driver shared by step zero (the pre-change measurement of the existing run-2
ensemble), the gate reads of the decision protocol, and the one-shot OOS read:

    python3 tools/score_checkpoints.py --run-dir models/v5 --members 0,1,2,3 \
        --ensemble --population val [--data-end D] [--null shuffle|shift] \
        [--print-seams] [--out FILE] [--design-freeze]

Modes:

* Manifest mode (``<run-dir>/run_manifest.json`` exists): the split-relevant args
  (eval mode, seed, holdout fraction, eval days, data start/end, window,
  val subsample, max windows, universe) are reconstructed from the manifest, never
  from trainer defaults, and the rebuilt scaler row count is asserted against the
  manifest's ``scaler_rows``. A CLI ``--data-end`` contradicting the manifest is a
  SystemExit, not an override. Gate and OOS reads divide the logits by the persisted
  per-horizon temperatures from ``<run-dir>/v5_meta.json`` (the gating object is the
  shipped, tempered ensemble).
* Step-zero mode (no manifest): hardcoded run-2 args plus the 23,948,324 scaler-row
  assert; runs untempered (run-2's crashed finalize left no ``v5_meta.json``) and
  writes ``docs/v5_step0_results.md``. The output records the git revision and the
  sha256 of ``features_v5.py`` so a step-zero run that postdates the label-semantics
  change is detectable after the fact.

The tool rebuilds frames, bounds, scaler, and (when the trainer provides it) the
cross-sectional label centering with the CURRENT code path, so it scores exactly
what the current pipeline would train and consume. Three later-phase concepts are
computed tool-locally so step zero is self-sufficient: the tradability rule (raw
close/volume, the same formula as the feature builder), the uncensored-outcome alternative mask (run-2
semantics when the gap mask does not exist yet), and the MambaRef<->CUDA parity
helper (:func:`mamba_parity_check`, also called from the finalize path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

import features_v5 as fx  # noqa: E402
from features_v5 import HORIZON_DAYS, MIN_REAL_ROWS, N_FEATURES  # noqa: E402
import run_train_v5 as rt  # noqa: E402
from v5 import metrics as mx  # noqa: E402
from v5_backbone import V5Backbone, V5Config, hl_gauss_targets  # noqa: E402

CENSUS_PATH = PROJECT_ROOT / "TrainingData" / "session_census.csv"
STEP0_DOC = PROJECT_ROOT / "docs" / "v5_step0_results.md"
STEP0_SCALER_ROWS = 23_948_324          # the run-2 finalize.log fingerprint
HORIZON_LABELS = ("1d", "1w", "1m", "6m")

# Split-relevant manifest keys reconstructed for every gate read (the finalize
# snapshot-assert set minus ``members``, which the scorer's own --members flag
# supplies).
SPLIT_ARG_KEYS = ("eval_mode", "seed", "ticker_holdout_frac", "eval_days",
                  "data_start", "data_end", "window", "val_subsample",
                  "max_windows", "tickers", "max_tickers")

STEP0_ARGS = {
    "eval_mode": "time", "seed": 42, "ticker_holdout_frac": 0.0, "eval_days": 254,
    "data_start": None, "data_end": None, "window": 252, "val_subsample": 150_000,
    "max_windows": None, "tickers": None, "max_tickers": None,
}


# ------------------------------------------------------------------ helpers

def git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def features_file_sha256():
    return hashlib.sha256((PROJECT_ROOT / "features_v5.py").read_bytes()).hexdigest()


def features_builder_dirty():
    """True when features_v5.py carries uncommitted edits (the rebuilt scaler is then
    only approximately the training scaler)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "features_v5.py"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def compute_tradable(close, volume):
    """The section-0 tradability rule, causal at the anchor: close >= $5 and the
    63-session rolling median dollar volume >= $1M (min_periods=63, NaN -> False).
    Tool-local so step zero needs no builder support; once the FeatureFrame carries
    ``tradable``, a test pins the two implementations bit-identical."""
    close_s = pd.Series(np.asarray(close, dtype=np.float64))
    dollar = close_s * pd.Series(np.asarray(volume, dtype=np.float64))
    med = dollar.rolling(63, min_periods=63).median()
    out = (close_s >= 5.0) & (med >= 1e6)
    return out.fillna(False).to_numpy(dtype=bool)


def compute_label_mask(frame, split_end):
    """Current-tree label mask; tolerates both label_mask generations."""
    try:
        return fx.label_mask(frame.z, frame.spike_free, frame.target_dates,
                             frame.dates, split_end)
    except TypeError:
        return fx.label_mask(frame.spike_free, frame.target_dates, split_end)


def compute_m_uncens(frame, split_end):
    """Uncensored-outcome alternative mask: the label mask with the spike term
    dropped -- finite z, target exists, embargo, and the gap term when the current
    tree has one (run-2 semantics carry no gap masking)."""
    has_future = ~np.isnat(frame.target_dates)
    finite = np.isfinite(frame.z)
    within = frame.target_dates <= np.asarray(split_end, dtype="datetime64[ns]")[:, None]
    m = has_future & finite & within
    gap_limits = getattr(fx, "GAP_LIMIT_DAYS", None)
    if gap_limits is not None:
        anchor_days = frame.dates.astype("datetime64[D]").astype(np.int64)
        tgt_days = frame.target_dates.astype("datetime64[D]").astype(np.int64)
        for hi in range(len(HORIZON_DAYS)):
            m[:, hi] &= (tgt_days[:, hi] - anchor_days) <= gap_limits[hi]
    return m.astype(np.float32)


def load_census_sessions():
    if not CENSUS_PATH.exists():
        raise SystemExit(f"census file missing: {CENSUS_PATH} (run tools/build_session_census.py)")
    census = pd.read_csv(CENSUS_PATH, parse_dates=["date"])
    keep = census[census["n_names"] >= 3]["date"]
    return keep.to_numpy(dtype="datetime64[ns]")


def print_seams(eval_days, data_end=None):
    """Rolling-origin data-ends: the phantom-filtered session exactly k*eval_days
    sessions before the primary data-end, k in {1, 2, 3}."""
    sessions = load_census_sessions()
    if data_end is not None:
        sessions = sessions[sessions <= np.datetime64(data_end)]
    primary = sessions[-1]
    print(f"[seams] phantom-filtered sessions: {len(sessions)}, "
          f"primary data-end {np.datetime_as_string(primary, unit='D')}")
    for k in (1, 2, 3):
        off = k * eval_days
        if off >= len(sessions):
            print(f"[seams] k={k}: not enough sessions")
            continue
        d = sessions[-1 - off]
        print(f"[seams] k={k}: --data-end {np.datetime_as_string(d, unit='D')} "
              f"({off} sessions before primary)")


# ------------------------------------------------------- population assembly

def load_frames_and_raw(tickers, data_start, data_end):
    """Build feature frames with the current builder, truncating each ticker's OHLCV
    BEFORE the feature build (post-cutoff prices must not leak into labels or sigma).
    Returns (frames, tradable_by_ticker)."""
    fear_greed = fx.load_fear_greed()
    ohlcv = fx.load_us_ohlcv(tickers=tickers)
    frames, tradable = {}, {}
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
        df = df.reset_index(drop=True)
        f = fx.build_feature_frame(ticker, df, fear_greed)
        frames[ticker] = f
        if hasattr(f, "tradable"):
            tradable[ticker] = np.asarray(f.tradable, dtype=bool)
        else:
            d2 = df.sort_values("date").drop_duplicates("date", keep="last")
            tradable[ticker] = compute_tradable(d2["close"].to_numpy(),
                                                d2["volume"].to_numpy())
    return frames, tradable


def fit_scaler_like_trainer(frames, bounds, eval_mode, eval_tickers, val_tickers):
    train_rows = []
    for ticker, f in frames.items():
        if ticker in eval_tickers:
            continue
        split, _ = rt.row_splits(f, bounds, eval_mode, False, ticker in val_tickers)
        sel = split == 0
        if sel.any():
            train_rows.append(f.features[sel])
    if not train_rows:
        raise SystemExit("no training rows for scaler fit")
    matrix = np.concatenate(train_rows, axis=0)
    scaler = fx.RobustScaler.fit(matrix)
    return scaler, len(matrix)


def enumerate_population(frames, bounds, eval_mode, eval_tickers, val_tickers,
                         tradable_by_ticker, population):
    """Enumerate (ticker, anchor) samples plus label/meta arrays for ``val`` (split-1
    rows, mirroring the trainer's val bucket exactly so the trainer's seeded
    subsample reproduces) or ``oos`` (split-2 rows)."""
    want_split = 1 if population != "oos" else 2
    ticker_ids = {t: i for i, t in enumerate(sorted(frames))}
    bucket = {"ticker": [], "anchor": [], "z": [], "m": [], "date": [], "tdates": [],
              "tradable": [], "close": [], "sigma_hat": [], "tag": [], "m_uncens": []}
    comparator_idx = {name: fx.FEATURE_NAMES.index(name)
                      for name in ("momentum_20d", "momentum_5d", "vol_z")}
    comps = {name: [] for name in comparator_idx}
    for ticker, f in frames.items():
        split, split_end = rt.row_splits(f, bounds, eval_mode,
                                         ticker in eval_tickers, ticker in val_tickers)
        mask = compute_label_mask(f, split_end)
        m_unc = compute_m_uncens(f, split_end)
        trad = tradable_by_ticker[ticker]
        tag = 1 if ticker in eval_tickers else 0
        for a in range(MIN_REAL_ROWS - 1, len(f.dates)):
            if int(split[a]) != want_split:
                continue
            if mask[a].sum() == 0:
                continue
            bucket["ticker"].append(ticker_ids[ticker])
            bucket["anchor"].append(int(a))
            bucket["z"].append(f.z[a])
            bucket["m"].append(mask[a])
            bucket["date"].append(f.dates[a].astype("datetime64[ns]").astype(np.int64))
            bucket["tdates"].append(f.target_dates[a].astype("datetime64[ns]").astype(np.int64))
            bucket["tradable"].append(bool(trad[a]))
            bucket["close"].append(float(f.close[a]))
            bucket["sigma_hat"].append(float(f.sigma_hat[a]))
            bucket["tag"].append(tag)
            bucket["m_uncens"].append(m_unc[a])
            for name, j in comparator_idx.items():
                comps[name].append(float(f.features[a, j]))
    bucket.update(comps)
    return bucket


def bucket_to_meta(bucket):
    meta = {
        "date": np.asarray(bucket["date"], dtype=np.int64),
        "ticker_id": np.asarray(bucket["ticker"], dtype=np.int64),
        "tdates": np.asarray(bucket["tdates"], dtype=np.int64),
        "tradable": np.asarray(bucket["tradable"], dtype=bool),
        "close": np.asarray(bucket["close"], dtype=np.float64),
        "sigma_hat": np.asarray(bucket["sigma_hat"], dtype=np.float64),
        "tag": np.asarray(bucket["tag"], dtype=np.int64),
        "m_uncens": np.asarray(bucket["m_uncens"], dtype=np.float32),
    }
    z = np.asarray(bucket["z"], dtype=np.float32)
    m = np.asarray(bucket["m"], dtype=np.float32)
    comps = {name: np.asarray(bucket[name], dtype=np.float64)
             for name in ("momentum_20d", "momentum_5d", "vol_z")}
    return z, m, meta, comps


# ------------------------------------------------------------- model forward

def load_members(run_dir, cfg, member_ids, device):
    members = []
    for i in member_ids:
        ckpt = run_dir / f"member_{i}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        model = V5Backbone(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        members.append(model.to(device).eval())
    return members


def _ce_terms(probs, z_rows, m_rows, cfg):
    """Per-sample masked CE contribution (distributional + lam * 3-class) of a batch
    of bin-probability histograms; mirrors v5_loss exactly (log probs == log_softmax
    of the logits that produced them). Returns the scalar sum over masked entries."""
    lam = float(getattr(cfg, "lam_cls", 0.1))
    hist = probs.clamp_min(1e-12)
    logp = hist.log()
    ce = -(hl_gauss_targets(z_rows, cfg) * logp).sum(-1)
    theta = torch.tensor(cfg.theta_bins, dtype=z_rows.dtype, device=z_rows.device) * cfg.bin_width
    cls = (z_rows > -theta).long() + (z_rows > theta).long()
    half = cfg.n_bins // 2
    cols = []
    for hi, k in enumerate(cfg.theta_bins):
        ph = hist[:, hi]
        cols.append(torch.stack([ph[:, :half - k].sum(-1),
                                 ph[:, half - k:half + k].sum(-1),
                                 ph[:, half + k:].sum(-1)], -1))
    pm = torch.stack(cols, 1).clamp_min(1e-8)
    ce3 = torch.nn.functional.nll_loss(pm.log().flatten(0, 1), cls.flatten(),
                                       reduction="none").view_as(z_rows)
    per = torch.where(m_rows.bool(), ce + lam * ce3, torch.zeros_like(ce))
    return float(per.sum())


@torch.no_grad()
def forward_population(members, frames, scaler, bucket, cfg, device, temps, z, m,
                       batch_size=512):
    """Forward every member over the population. Returns per-member class marginals
    (from TEMPERED logits when ``temps`` is given -- the consumed scores), the
    ensemble marginals, and raw / tempered CE accumulators for members and ensemble.
    Windows are built ticker-by-ticker in enumeration order so each ticker's
    normalized feature matrix is materialized once and freed."""
    n = len(bucket["anchor"])
    n_members = len(members)
    H = len(cfg.horizons)
    half = cfg.n_bins // 2
    theta = cfg.theta_bins
    p_up = np.zeros((n_members, n, H), dtype=np.float32)
    p_down = np.zeros((n_members, n, H), dtype=np.float32)
    ens_up = np.zeros((n, H), dtype=np.float32)
    ens_down = np.zeros((n, H), dtype=np.float32)
    T = (torch.tensor(temps, dtype=torch.float32, device=device).view(1, -1, 1)
         if temps is not None else None)
    ce_raw = np.zeros(n_members)
    ce_tmp = np.zeros(n_members)
    ce_ens_raw = 0.0
    ce_ens_tmp = 0.0
    mask_total = float(np.asarray(m).sum())
    z_t = torch.from_numpy(np.nan_to_num(np.asarray(z, dtype=np.float32)))
    m_t = torch.from_numpy(np.asarray(m, dtype=np.float32))
    tickers_sorted = sorted(frames)
    ticker_arr = np.asarray(bucket["ticker"], dtype=np.int64)
    anchor_arr = np.asarray(bucket["anchor"], dtype=np.int64)

    buf, buf_rows = [], []

    def flush():
        nonlocal ce_ens_raw, ce_ens_tmp
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        rows = np.asarray(buf_rows)
        zb = z_t[rows].to(device)
        mb = m_t[rows].to(device)
        mean_raw = None
        mean_tmp = None
        for mi, model in enumerate(members):
            logits = model(x)
            probs_raw = logits.softmax(-1)
            probs_tmp = (logits / T).softmax(-1) if T is not None else probs_raw
            ce_raw[mi] += _ce_terms(probs_raw, zb, mb, cfg)
            if T is not None:
                ce_tmp[mi] += _ce_terms(probs_tmp, zb, mb, cfg)
            mean_raw = probs_raw if mean_raw is None else mean_raw + probs_raw
            mean_tmp = probs_tmp if mean_tmp is None else mean_tmp + probs_tmp
            for hi, k in enumerate(theta):
                ph = probs_tmp[:, hi]
                p_down[mi, rows, hi] = ph[:, :half - k].sum(-1).cpu().numpy()
                p_up[mi, rows, hi] = ph[:, half + k:].sum(-1).cpu().numpy()
        mean_raw = mean_raw / n_members
        mean_tmp = mean_tmp / n_members
        ce_ens_raw += _ce_terms(mean_raw, zb, mb, cfg)
        if T is not None:
            ce_ens_tmp += _ce_terms(mean_tmp, zb, mb, cfg)
        for hi, k in enumerate(theta):
            ph = mean_tmp[:, hi]
            ens_down[rows, hi] = ph[:, :half - k].sum(-1).cpu().numpy()
            ens_up[rows, hi] = ph[:, half + k:].sum(-1).cpu().numpy()
        buf.clear()
        buf_rows.clear()

    order = np.argsort(ticker_arr, kind="stable")
    last_ticker = -1
    norm = None
    for row in order:
        tid = int(ticker_arr[row])
        if tid != last_ticker:
            flush()
            f = frames[tickers_sorted[tid]]
            norm = scaler.transform(f.features)
            last_ticker = tid
        win = fx.assemble_window(norm, int(anchor_arr[row]), cfg.window, cfg.min_real_rows)
        if win is None:
            raise SystemExit(f"window below min_real_rows at row {row} -- enumeration bug")
        buf.append(win)
        buf_rows.append(row)
        if len(buf) >= batch_size:
            flush()
    flush()
    denom = max(mask_total, 1.0)
    ce = {"members_raw": (ce_raw / denom).tolist(),
          "ensemble_raw": ce_ens_raw / denom}
    if T is not None:
        ce["members_tempered"] = (ce_tmp / denom).tolist()
        ce["ensemble_tempered"] = ce_ens_tmp / denom
    return p_up, p_down, ens_up, ens_down, ce


# ---------------------------------------------------------------- CE metrics

def marginal_hist_ce(q_hist, q_cls, z, m, cfg):
    """CE of a fixed marginal predictor (histogram q_hist per horizon, class rates
    q_cls per horizon) evaluated on labels (z, m)."""
    lam = float(getattr(cfg, "lam_cls", 0.1))
    zt = torch.from_numpy(np.nan_to_num(np.asarray(z, dtype=np.float32)))
    mt = torch.from_numpy(np.asarray(m, dtype=np.float32))
    soft = hl_gauss_targets(zt, cfg)
    theta = torch.tensor(cfg.theta_bins, dtype=zt.dtype) * cfg.bin_width
    cls = (zt > -theta).long() + (zt > theta).long()
    total, count = 0.0, 0.0
    for hi in range(len(cfg.horizons)):
        mh = mt[:, hi]
        nh = float(mh.sum())
        if nh < 1:
            continue
        qh = torch.from_numpy(np.asarray(q_hist[hi], dtype=np.float64)).clamp_min(1e-12)
        ce_h = float(-((soft[:, hi] * qh.log()).sum(-1) * mh).sum())
        qc = torch.from_numpy(np.asarray(q_cls[hi], dtype=np.float64)).clamp_min(1e-12)
        ce3_h = float(-(qc.log()[cls[:, hi]] * mh).sum())
        total += ce_h + lam * ce3_h
        count += nh
    return total / count if count else float("inf")


def marginal_from_labels(z, m, cfg):
    """Marginal histogram (mean HL-Gauss soft target) and class rates per horizon
    from masked labels."""
    zt = torch.from_numpy(np.nan_to_num(np.asarray(z, dtype=np.float32)))
    mt = torch.from_numpy(np.asarray(m, dtype=np.float32))
    soft = hl_gauss_targets(zt, cfg)
    theta = torch.tensor(cfg.theta_bins, dtype=zt.dtype) * cfg.bin_width
    cls = (zt > -theta).long() + (zt > theta).long()
    q_hist, q_cls = [], []
    for hi in range(len(cfg.horizons)):
        mh = mt[:, hi]
        nh = float(mh.sum())
        if nh < 1:
            q_hist.append(np.full(cfg.n_bins, 1.0 / cfg.n_bins))
            q_cls.append(np.full(3, 1.0 / 3.0))
            continue
        q = (soft[:, hi] * mh[:, None]).sum(0) / nh
        q_hist.append(q.numpy())
        rate = np.asarray([float(((cls[:, hi] == c).float() * mh).sum() / nh)
                           for c in range(3)])
        q_cls.append(rate)
    return q_hist, q_cls


def train_marginal_subsample(frames, bounds, eval_mode, eval_tickers, val_tickers,
                             cfg, seed, cap=1_000_000):
    """Seeded label subsample of the train bucket for the train-marginal-on-val
    baseline flavor."""
    rng = np.random.default_rng(seed)
    zs, ms = [], []
    kept = 0
    for ticker, f in frames.items():
        split, split_end = rt.row_splits(f, bounds, eval_mode,
                                         ticker in eval_tickers, ticker in val_tickers)
        mask = compute_label_mask(f, split_end)
        sel = split == 0
        if not sel.any():
            continue
        z_t, m_t = f.z[sel], mask[sel]
        keep = rng.random(len(z_t)) < min(1.0, cap / 5_000_000)
        if keep.any():
            zs.append(z_t[keep])
            ms.append(m_t[keep])
            kept += int(keep.sum())
        if kept >= cap:
            break
    if not zs:
        return None
    z = np.concatenate(zs)[:cap]
    m = np.concatenate(ms)[:cap]
    return marginal_from_labels(z, m, cfg)


# -------------------------------------------------------------- parity check

@torch.no_grad()
def mamba_parity_check(ckpt_path, cfg, batches, device="cuda"):
    """Max-abs logit difference between the CUDA-kernel backbone and the forced
    MambaRef backbone on real batches, fp32 eval mode. Returns None (with a printed
    skip) when CUDA or mamba_ssm is unavailable."""
    if device != "cuda" or not torch.cuda.is_available():
        print("[parity] skip: CUDA unavailable")
        return None
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        print("[parity] skip: mamba_ssm not installed")
        return None
    from v5.mamba_ref import MambaRef
    state = torch.load(ckpt_path, map_location="cpu")
    cuda_model = V5Backbone(cfg)
    cuda_model.load_state_dict(state)
    cuda_model.to(device).eval().float()
    ref_model = V5Backbone(cfg, mamba_cls=MambaRef)
    ref_model.load_state_dict(state)
    ref_model.to(device).eval().float()
    worst = 0.0
    for x in batches:
        xb = x.to(device=device, dtype=torch.float32)
        diff = (cuda_model(xb) - ref_model(xb)).abs().max().item()
        worst = max(worst, diff)
    if worst > 1e-3:
        print(f"[parity] WARNING max-abs logit diff {worst:.3e} > 1e-3")
    else:
        print(f"[parity] max-abs logit diff {worst:.3e}")
    return worst


# ----------------------------------------------------------------- reporting

def fmt_records(records, keys=("horizon", "score", "mask", "role", "trad")):
    lines = []
    for r in records:
        tagpart = " ".join(f"{k}={r[k]}" for k in keys if k in r)
        rest = {k: v for k, v in r.items() if k not in keys}
        num = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in rest.items())
        lines.append(f"  {tagpart} | {num}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Score v5 checkpoints on a population.")
    p.add_argument("--run-dir", default="models/v5")
    p.add_argument("--members", default=None,
                   help="Comma-separated member ids (default: all in the manifest, "
                        "or 0,1,2,3 at step zero).")
    p.add_argument("--ensemble", action="store_true",
                   help="Additionally report the member-mean ensemble.")
    p.add_argument("--population", choices=["val", "stopping", "gating", "oos"],
                   default="val")
    p.add_argument("--data-end", default=None,
                   help="Serves --print-seams and manifest-less probes only; a value "
                        "contradicting the manifest is an error.")
    p.add_argument("--null", choices=["shuffle", "shift"], default=None)
    p.add_argument("--print-seams", action="store_true")
    p.add_argument("--design-freeze", action="store_true",
                   help="Required for --population oos (the one-shot OOS read).")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cpu-cap", type=int, default=20_000,
                   help="Seeded population cap when running on CPU (the GPU box "
                        "scores the full fold).")
    p.add_argument("--seed", type=int, default=42,
                   help="Only used for the CPU-cap subsample and the null seeds; the "
                        "split seed always comes from the manifest / step-zero args.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    step_zero = manifest is None

    if step_zero:
        run_args = dict(STEP0_ARGS)
        if args.data_end is not None:
            run_args["data_end"] = args.data_end
    else:
        run_args = {k: manifest["args"][k] for k in SPLIT_ARG_KEYS}
        if args.data_end is not None and str(run_args["data_end"]) != str(args.data_end):
            raise SystemExit(
                f"--data-end {args.data_end} contradicts the manifest "
                f"({run_args['data_end']}); the manifest is authoritative")

    if args.print_seams:
        print_seams(int(run_args["eval_days"]), run_args.get("data_end"))
        return

    if args.population == "oos" and not args.design_freeze:
        raise SystemExit("--population oos requires --design-freeze (one-shot OOS read)")
    if args.population == "oos":
        print("[scorer] WARNING: one-shot OOS read -- this consumes the design-freeze "
              "look at the OOS tail; do not repeat before the next freeze.")

    notes = []
    rev = git_revision()
    fhash = features_file_sha256()
    print(f"[scorer] git revision {rev}")
    print(f"[scorer] features_v5.py sha256 {fhash}")
    if step_zero:
        notes.append("step-zero mode: hardcoded run-2 args; untempered scores "
                     "(run-2's crashed finalize left no v5_meta.json)")
        if features_builder_dirty():
            notes.append("features_v5.py has uncommitted edits: the rebuilt scaler is "
                         "approximate for the run-2 checkpoints")
        print(f"[scorer] {notes[0]}")

    # Config: run-dir config.json when present, run-2 architecture otherwise.
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = V5Config(**json.loads(cfg_path.read_text()))
    else:
        cfg = V5Config(n_features=N_FEATURES, window=int(run_args["window"]),
                       min_real_rows=MIN_REAL_ROWS, d_model=256, n_blocks=6,
                       horizons=tuple(HORIZON_DAYS))
        notes.append("config.json absent: run-2 architecture defaults assumed")

    # Temperatures: mandatory for gate/OOS reads (the gating object is tempered).
    temps = None
    meta_path = run_dir / "v5_meta.json"
    if meta_path.exists():
        temps = json.loads(meta_path.read_text()).get("temperatures")
    if args.population in ("stopping", "gating", "oos") and temps is None and not step_zero:
        raise SystemExit(f"{args.population} read needs temperatures in {meta_path}; "
                         "finalize the run first")
    if temps is not None:
        print(f"[scorer] tempered scores, T={[round(t, 3) for t in temps]}")

    # Universe and frames.
    tickers = run_args["tickers"] or fx.list_us_tickers()
    if run_args["max_tickers"]:
        tickers = tickers[:int(run_args["max_tickers"])]
    print(f"[scorer] building frames for {len(tickers)} tickers ...")
    frames, tradable_by_ticker = load_frames_and_raw(
        tickers, run_args["data_start"], run_args["data_end"])
    print(f"[scorer] usable frames: {len(frames)}")
    if not frames:
        raise SystemExit("no usable frames")

    bounds = rt.global_date_bounds(frames, int(run_args["eval_days"]))
    eval_tickers = rt.assign_eval_tickers(sorted(frames), int(run_args["seed"]),
                                          float(run_args["ticker_holdout_frac"]))
    val_tickers = set()
    if run_args["eval_mode"] == "ticker":
        val_tickers = rt.assign_eval_tickers(sorted(set(frames) - eval_tickers),
                                             int(run_args["seed"]) + 1,
                                             max(0.1, float(run_args["ticker_holdout_frac"])))

    scaler, n_rows = fit_scaler_like_trainer(frames, bounds, run_args["eval_mode"],
                                             eval_tickers, val_tickers)
    print(f"[scorer] scaler fit on {n_rows} train rows")
    if step_zero:
        assert n_rows == STEP0_SCALER_ROWS, (
            f"step-zero scaler row count {n_rows} != {STEP0_SCALER_ROWS} "
            "(finalize.log fingerprint): data or universe drift")
    elif manifest.get("scaler_rows") is not None:
        assert n_rows == int(manifest["scaler_rows"]), (
            f"scaler row count {n_rows} != manifest {manifest['scaler_rows']}")

    # Cross-sectional centering when the current trainer performs it.
    center_fn = getattr(rt, "center_labels", None)
    if center_fn is not None:
        k_min = min(50, math.ceil(len(frames) / 2))
        center_fn(frames, bounds, run_args["eval_mode"], eval_tickers, k_min)
        print("[scorer] labels centered with the current trainer path")

    bucket = enumerate_population(frames, bounds, run_args["eval_mode"], eval_tickers,
                                  val_tickers, tradable_by_ticker, args.population)
    n_before = len(bucket["anchor"])
    if args.population != "oos":
        caps = [c for c in (run_args["max_windows"], run_args["val_subsample"]) if c]
        cap = min(int(c) for c in caps) if caps else None
        bucket = rt._subsample_bucket(bucket, cap, int(run_args["seed"]))
        print(f"[scorer] val fold: {n_before} -> {len(bucket['anchor'])} "
              f"(trainer subsample, cap {cap})")
    if args.device == "cpu" and len(bucket["anchor"]) > args.cpu_cap:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(bucket["anchor"]), args.cpu_cap, replace=False))
        bucket = {k: [bucket[k][i] for i in keep] for k in bucket}
        notes.append(f"CPU fallback: population subsampled to {args.cpu_cap} "
                     f"(seed {args.seed}); GPU reads score the full fold")
        print(f"[scorer] {notes[-1]}")

    z, m, meta, comps = bucket_to_meta(bucket)
    n = len(z)
    print(f"[scorer] population {args.population}: {n} samples, "
          f"{len(np.unique(meta['date']))} dates")

    member_ids = ([int(x) for x in args.members.split(",")] if args.members
                  else list(range(int(manifest["args"]["members"]) if manifest else 4)))
    members = load_members(run_dir, cfg, member_ids, args.device)
    print(f"[scorer] loaded members {member_ids}")

    p_up, p_down, ens_up, ens_down, ce_fwd = forward_population(
        members, frames, scaler, bucket, cfg, args.device, temps, z, m,
        batch_size=args.batch_size)
    up_std = p_up.std(0) if len(members) > 1 else None

    if args.null:
        null_fn = (mx.null_shuffle_within_date if args.null == "shuffle"
                   else mx.null_circular_shift)
        stacked = np.concatenate([ens_up, ens_down], axis=1)
        shuffled = null_fn(stacked, meta, args.seed)
        ens_up, ens_down = shuffled[:, :4], shuffled[:, 4:]
        notes.append(f"null={args.null} applied to the ensemble scores (seed {args.seed}); "
                     "per-member suites skipped under a null read")
        print(f"[scorer] {notes[-1]}")

    cfg_m = mx.default_cfg()
    val_sessions = np.unique(meta["date"])
    blocks, roles = (mx.block_partition(val_sessions, 4)
                     if args.population != "oos" else (None, None))

    report = {"git_revision": rev, "features_sha256": fhash, "scaler_rows": n_rows,
              "population": args.population, "n_samples": int(n),
              "members": member_ids, "tempered": temps is not None,
              "run_args": {k: run_args[k] for k in SPLIT_ARG_KEYS}, "notes": notes}

    # CE vs both baseline flavors.
    q_hist, q_cls = marginal_from_labels(z, m, cfg)
    base_oracle = marginal_hist_ce(q_hist, q_cls, z, m, cfg)
    train_marg = train_marginal_subsample(frames, bounds, run_args["eval_mode"],
                                          eval_tickers, val_tickers, cfg,
                                          int(run_args["seed"]))
    base_train = (marginal_hist_ce(train_marg[0], train_marg[1], z, m, cfg)
                  if train_marg else float("nan"))
    report["ce"] = {**ce_fwd, "baseline_val_oracle": base_oracle,
                    "baseline_train_marginal": base_train}
    print(f"[scorer] CE ensemble(raw) {ce_fwd['ensemble_raw']:.4f} | "
          f"members(raw) {[round(c, 4) for c in ce_fwd['members_raw']]} | "
          f"val-oracle baseline {base_oracle:.4f} | "
          f"train-marginal baseline {base_train:.4f}")

    # IC suites: ensemble and per member.
    records, series = mx.ic_suite(ens_up, ens_down, z, m, meta, cfg_m, blocks, roles,
                                  tradable=meta["tradable"], alt_mask=meta["m_uncens"])
    report["ic_ensemble"] = records
    print("[scorer] ensemble IC suite:")
    print(fmt_records([r for r in records if r["role"] in ("all", "gate", "stop")
                       and r["mask"] == "primary"][:32]))
    per_member = {}
    if not args.null:
        for mi, mid in enumerate(member_ids):
            recs_m, _ = mx.ic_suite(p_up[mi], p_down[mi], z, m, meta, cfg_m, blocks,
                                    roles, tradable=meta["tradable"],
                                    alt_mask=meta["m_uncens"])
            per_member[f"member_{mid}"] = recs_m
    report["ic_members"] = per_member

    score_ens = ens_up - ens_down
    top_records, _ = mx.top_of_ranking(score_ens, z, m, meta, sigma=meta["sigma_hat"],
                                       tradable=meta["tradable"],
                                       alt_mask=meta["m_uncens"])
    report["top_of_ranking"] = top_records
    print("[scorer] top-of-ranking (ensemble Score):")
    print(fmt_records(top_records, keys=("horizon", "mask", "trad")))

    # Trivial comparators from anchor feature rows (ranks preserved under the
    # monotone-affine scaler; the +-10 clip ties only extreme tails).
    comp_records = []
    sessions = val_sessions
    for name, sgn in (("momentum_20d", 1.0), ("momentum_5d", -1.0), ("vol_z", 1.0)):
        col = sgn * comps[name]
        for hi, hl in enumerate(HORIZON_LABELS):
            for trad_name, tmask in (("all", None), ("tradable", meta["tradable"])):
                el = m[:, hi] > 0
                if tmask is not None:
                    el &= tmask
                el &= np.isfinite(col)
                keys, ics = mx.spearman_ic(col[el], z[el, hi], meta["date"][el])
                mean, se, t, cnt = mx.hac_t(ics, np.searchsorted(sessions, keys),
                                            cfg_m["hac_lag"][hi])
                comp_records.append({
                    "comparator": name if sgn > 0 else "reversal_5d", "horizon": hl,
                    "trad": trad_name, "mean_ic": mean, "t": t, "n_dates": cnt,
                    "ic_std": float(np.std(ics)) if cnt else float("nan")})
    report["comparators"] = comp_records
    print("[scorer] comparator ICs (checkpoint-independent):")
    print(fmt_records(comp_records, keys=("comparator", "horizon", "trad")))

    # Per-date IC dispersion summary (powers the pre-registration figures).
    disp = {}
    for (hl, sc, var, role, trad), (keys, ics) in series.items():
        if sc == "score" and var == "primary" and role == "all":
            disp[f"{hl}_{trad}"] = float(np.std(ics)) if len(ics) else float("nan")
    report["ic_dispersion"] = disp
    print(f"[scorer] per-date IC dispersion (Score): {disp}")

    if up_std is not None:
        theta = [k * cfg.bin_width for k in cfg.theta_bins]
        uq = mx.uncertainty_quality(up_std, ens_up, z, m, meta,
                                    {"theta": theta, "horizon_labels": HORIZON_LABELS})
        report["uncertainty_quality"] = uq
        print(f"[scorer] uncertainty quality: "
              + " ".join(f"{h}:rho={v['spearman']:.3f}" for h, v in uq.items()))

    # Parity number (GPU only): 3 real batches from the head of the population.
    if args.device == "cuda":
        tickers_sorted = sorted(frames)
        batches, wins = [], []
        last_tid, norm = -1, None
        for row in range(min(3 * args.batch_size, n)):
            tid = int(meta["ticker_id"][row])
            if tid != last_tid:
                norm = scaler.transform(frames[tickers_sorted[tid]].features)
                last_tid = tid
            wins.append(fx.assemble_window(norm, int(bucket["anchor"][row]),
                                           cfg.window, cfg.min_real_rows))
            if len(wins) == args.batch_size:
                batches.append(torch.from_numpy(np.stack(wins)))
                wins = []
        if wins:
            batches.append(torch.from_numpy(np.stack(wins)))
        parity = mamba_parity_check(run_dir / f"member_{member_ids[0]}.pt", cfg,
                                    batches[:3], args.device)
        report["mamba_parity_max_abs"] = parity

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[scorer] wrote {out_path}")

    if step_zero:
        write_step0_doc(report)
        print(f"[scorer] wrote {STEP0_DOC}")


def write_step0_doc(report):
    STEP0_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# v5 step-zero results (auto-generated by tools/score_checkpoints.py)", ""]
    lines.append(f"Git revision: `{report['git_revision']}`; features_v5.py sha256: "
                 f"`{report['features_sha256']}`; scaler rows: {report['scaler_rows']}.")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    ce = report["ce"]
    members_str = ", ".join(f"{c:.4f}" for c in ce["members_raw"])
    lines.append(f"CE (raw): ensemble {ce['ensemble_raw']:.4f}, members [{members_str}] "
                 f"vs val-oracle baseline {ce['baseline_val_oracle']:.4f} and "
                 f"train-marginal baseline {ce['baseline_train_marginal']:.4f}.")
    lines.append("")
    lines.append("## Ensemble IC suite")
    lines.append("")
    lines.append("| horizon | score | mask | role | trad | mean IC | HAC t | dates | IC std |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in report["ic_ensemble"]:
        lines.append(f"| {r['horizon']} | {r['score']} | {r['mask']} | {r['role']} | "
                     f"{r['trad']} | {r['mean_ic']:.4f} | {r['t']:.2f} | {r['n_dates']} | "
                     f"{r['ic_std']:.4f} |")
    lines.append("")
    lines.append("## Top of ranking (ensemble Score)")
    lines.append("")
    lines.append("| horizon | mask | trad | topdec z mean | t | top1 z mean | dates |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["top_of_ranking"]:
        lines.append(f"| {r['horizon']} | {r['mask']} | {r['trad']} | "
                     f"{r['topdec_z_mean']:.4f} | {r['topdec_z_t']:.2f} | "
                     f"{r['top1_z_mean']:.4f} | {r['n_dates']} |")
    lines.append("")
    lines.append("## Comparator ICs")
    lines.append("")
    lines.append("| comparator | horizon | trad | mean IC | t | dates | IC std |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["comparators"]:
        lines.append(f"| {r['comparator']} | {r['horizon']} | {r['trad']} | "
                     f"{r['mean_ic']:.4f} | {r['t']:.2f} | {r['n_dates']} | "
                     f"{r['ic_std']:.4f} |")
    lines.append("")
    lines.append(f"Per-date IC dispersion (Score): {json.dumps(report['ic_dispersion'])}.")
    if "uncertainty_quality" in report:
        lines.append("")
        lines.append(f"Uncertainty quality: {json.dumps(report['uncertainty_quality'])}.")
    STEP0_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
