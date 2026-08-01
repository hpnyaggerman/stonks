"""Trainer tests: centering, thinning, judged rows, meta/tags, eval pass, both-mode
routing, class rates, temperatures, flags, embargo, manifest, train price floor,
LR-drop gates."""
import argparse
import dataclasses
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import features_v5 as fx
import run_train_v5 as rt
from v5 import metrics as mx
from v5_backbone import V5Backbone, V5Config, v5_loss, v5_loss_components


def _mk_ohlcv(n, seed, start="2014-01-06", drift=0.0005, vol=0.015, base=50.0,
              volume=500_000.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = base * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    openp = close * (1 + 0.002 * rng.standard_normal(n))
    high = np.maximum(openp, close) * 1.003
    low = np.minimum(openp, close) * 0.997
    return pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low,
                         "close": close, "volume": float(volume)})


def _mk_frames(n_tickers=6, n=500, seed=0, **kw):
    fear = pd.DataFrame({"date": pd.bdate_range("2010-01-04", periods=6000),
                         "fear_greed": 50.0})
    frames = {}
    for i in range(n_tickers):
        oh = _mk_ohlcv(n, seed + i, **kw)
        t = f"T{i:02d}"
        frames[t] = fx.build_feature_frame(t, oh, fear)
    return frames


# ------------------------------------------------------------ label centering

def test_centering_pool_median_zero_and_rank_invariant():
    frames = _mk_frames(8, 400, seed=1)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    before = {t: f.z.copy() for t, f in frames.items()}
    rt.center_labels(frames, bounds, "time", set(), k_min=4)
    unique = bounds["unique_dates"]

    # Recompute pool membership and check the centered pool median is ~0 per (s, h),
    # and that within-date rank order among pool members is unchanged.
    for hi in range(4):
        by_date_after, by_date_before = {}, {}
        for t, f in frames.items():
            split, split_end = rt.row_splits(f, bounds, "time", False, False)
            m_before = fx.label_mask(before[t], f.spike_free, f.target_dates,
                                     f.dates, split_end)
            for a in range(fx.MIN_REAL_ROWS - 1, len(f.dates)):
                if m_before[a, hi] > 0:
                    d = int(np.searchsorted(unique, f.dates[a]))
                    by_date_after.setdefault(d, []).append(f.z[a, hi])
                    by_date_before.setdefault(d, []).append(before[t][a, hi])
        checked = 0
        for d, vals in by_date_after.items():
            if len(vals) >= 4 and np.isfinite(vals).all():
                assert abs(np.median(vals)) < 1e-5, (hi, d, np.median(vals))
                order_a = np.argsort(vals, kind="stable")
                order_b = np.argsort(by_date_before[d], kind="stable")
                assert np.array_equal(order_a, order_b)
                checked += 1
        assert checked > 50


def test_centering_k_masking_small_universe():
    frames = _mk_frames(2, 400, seed=2)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    rt.center_labels(frames, bounds, "time", set(), k_min=3)   # pool of 2 < K=3
    for f in frames.values():
        # every label became NaN (undefined c), so the smoke-scaled K path matters
        assert not np.isfinite(f.z[fx.MIN_REAL_ROWS:]).any()
    frames2 = _mk_frames(2, 400, seed=2)
    rt.center_labels(frames2, bounds, "time", set(), k_min=1)  # scaled-down K keeps them
    assert np.isfinite(list(frames2.values())[0].z[200]).any()


def test_centering_eval_ticker_excluded_from_pool_but_centered():
    frames = _mk_frames(6, 400, seed=3)
    tickers = sorted(frames)
    eval_t = {tickers[0]}
    bounds = rt.global_date_bounds(frames, eval_days=40)
    # Plant a huge outlier label in the eval ticker; if it voted, medians would move.
    f_eval = frames[tickers[0]]
    before_others = {t: frames[t].z.copy() for t in tickers[1:]}
    f_eval.z[:, :] = np.where(np.isfinite(f_eval.z), 500.0, f_eval.z)
    frames_ref = _mk_frames(6, 400, seed=3)
    rt.center_labels(frames_ref, bounds, "both", eval_t, k_min=2)
    rt.center_labels(frames, bounds, "both", eval_t, k_min=2)
    for t in tickers[1:]:
        a, b = frames[t].z, frames_ref[t].z
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both]), "eval-ticker labels leaked into the pool"
    # and the eval ticker itself was centered (shifted from its raw values)
    fin = np.isfinite(frames[tickers[0]].z)
    assert fin.any()
    assert not np.allclose(frames[tickers[0]].z[fin], 500.0)


def test_centering_boundary_coherence_gap_ticker():
    """A 6m label whose stored target crosses split_end is mask-dead and must not
    vote in c; verified by planting an outlier in exactly those labels."""
    frames = _mk_frames(6, 400, seed=4)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    t0 = sorted(frames)[0]
    f = frames[t0]
    split, split_end = rt.row_splits(f, bounds, "time", False, False)
    mask = fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end)
    crossing = (~np.isnat(f.target_dates[:, 3])
                & (f.target_dates[:, 3] > np.asarray(split_end, dtype="datetime64[ns]"))
                & np.isfinite(f.z[:, 3]))
    assert crossing.any()
    assert (mask[crossing, 3] == 0).all()
    frames_ref = _mk_frames(6, 400, seed=4)
    f.z[crossing, 3] = 1e6                     # would wreck the medians if pooled
    rt.center_labels(frames, bounds, "time", set(), k_min=2)
    rt.center_labels(frames_ref, bounds, "time", set(), k_min=2)
    for t in sorted(frames)[1:]:
        a, b = frames[t].z[:, 3], frames_ref[t].z[:, 3]
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both])


# ------------------------------------------------------------- label thinning

def _build(frames, bounds, eval_mode="time", eval_tickers=None, val_tickers=None,
           cache=None, seed=42, val_subsample=None, max_windows=None, window=140,
           panel_seed=None):
    eval_tickers = eval_tickers or set()
    val_tickers = val_tickers or set()
    scaler = fx.RobustScaler.fit(
        np.concatenate([f.features[fx.MIN_REAL_ROWS:] for f in frames.values()]))
    cache = cache or tempfile.mkdtemp()
    return rt.build_memmap_and_samples(
        frames, scaler, bounds, eval_mode, eval_tickers, val_tickers, window,
        max_windows=max_windows, val_subsample=val_subsample, seed=seed,
        cache_dir=cache, panel_seed=panel_seed)


def test_thinning_rates_and_val_untouched():
    frames = _mk_frames(8, 600, seed=5)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    mm, offsets, train_ds, (val_ds, val_meta), panel, surface = _build(frames, bounds)
    tm = np.asarray(train_ds.mask)
    vm = np.asarray(val_ds.mask)
    # Unthinned per-anchor label availability for 1m/6m on train anchors is ~1.0;
    # after thinning the retained fraction is ~1/stride.
    frac_1m = tm[:, 2].mean()
    frac_6m = tm[:, 3].mean()
    assert 0.15 < frac_1m < 0.35, frac_1m          # stride 4 -> ~0.25
    assert 0.01 < frac_6m < 0.09, frac_6m          # stride 25 -> ~0.04
    # Val bucket unthinned: 1m availability there is bounded by the embargo alone
    # (~(40 - 22)/40 of the tail), well above the thinned train rate.
    assert vm[:, 2].mean() > 0.33, vm[:, 2].mean()
    assert vm[:, 2].mean() > 1.3 * frac_1m
    assert tm[:, 0].mean() > 0.9                   # 1d/1w untouched


def test_build_determinism():
    frames_a = _mk_frames(4, 400, seed=6)
    frames_b = _mk_frames(4, 400, seed=6)
    bounds = rt.global_date_bounds(frames_a, eval_days=40)
    _, _, tr_a, (va_a, meta_a), _, _ = _build(frames_a, bounds, cache=tempfile.mkdtemp())
    _, _, tr_b, (va_b, meta_b), _, _ = _build(frames_b, bounds, cache=tempfile.mkdtemp())
    assert np.array_equal(np.asarray(tr_a.mask), np.asarray(tr_b.mask))
    assert np.array_equal(np.asarray(tr_a.start_abs), np.asarray(tr_b.start_abs))
    assert np.array_equal(np.asarray(va_a.z), np.asarray(va_b.z), equal_nan=True)
    assert np.array_equal(meta_a["date"], meta_b["date"])


# --------------------------------------------------------------------- lam_cls

def test_lam_cls_hoisted():
    cfg = V5Config(n_features=8, window=16, lam_cls=0.25)
    round_trip = V5Config(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    assert round_trip.lam_cls == 0.25
    logits = torch.randn(3, 4, cfg.n_bins)
    z = torch.randn(3, 4)
    m = torch.ones(3, 4)
    try:
        v5_loss(logits, z, m, cfg, lam_cls=0.5)
    except TypeError:
        pass
    else:
        raise AssertionError("v5_loss must no longer accept a lam_cls kwarg")
    # cfg.lam_cls actually feeds the loss
    a = v5_loss(logits, z, m, V5Config(n_features=8, window=16, lam_cls=0.0))
    b = v5_loss(logits, z, m, V5Config(n_features=8, window=16, lam_cls=1.0))
    assert abs(a.item() - b.item()) > 1e-6


# ------------------------------------------------------------- loss components

def test_loss_components_sum():
    cfg = V5Config(n_features=8, window=16)
    logits = torch.randn(64, 4, cfg.n_bins)
    z = torch.randn(64, 4)
    m = (torch.rand(64, 4) > 0.4).float()
    loss, per_h = v5_loss_components(logits, z, m, cfg)
    n_h = m.sum(0)
    weighted = float((per_h * n_h).sum() / m.sum())
    assert abs(weighted - loss.item()) < 1e-5


# --------------------------------------------------------------- judged rows

def test_spread_indices_span_all_fold_sizes():
    cap = 100
    for n in (99, 100, 150, 370):
        idx = rt._spread_indices(n, cap)
        assert idx[0] == 0 and idx[-1] == n - 1, (n, idx[:3], idx[-3:])
        assert len(idx) <= cap
        assert len(np.unique(idx)) == len(idx)


def test_judged_and_baseline_share_rows():
    """The stopping-side baseline is computed on exactly the judged score half."""
    rng = np.random.default_rng(7)
    N = 400
    dates = np.repeat(np.arange(40, dtype=np.int64), 10)
    blocks, roles = mx.block_partition(np.unique(dates), 4)
    tdates = np.repeat(dates[:, None], 4, axis=1)
    meta = {"date": dates, "tdates": tdates, "tag": np.zeros(N, dtype=np.int64)}
    mask = np.ones((N, 4), dtype=np.float32)
    m_stop, rows = rt.stopping_selection(meta, mask, blocks, roles)
    assert rows.size
    assert (m_stop[:, 2:] == 0).all()              # 1m/6m never stopping-side
    cap = 50
    capped = rt._spread_indices(rows.size, cap)
    parity = np.arange(capped.size) % 2
    score_half_global = rows[capped[parity == 1]]
    cfg = V5Config(n_features=8, window=16)
    logits = torch.randn(rows.size, 4, cfg.n_bins)
    z = torch.from_numpy(rng.standard_normal((rows.size, 4)).astype(np.float32))
    mt = torch.from_numpy(m_stop[rows])
    cal, raw, temps, score_rows_local = rt._judged_from_logits(
        logits, z, mt, cfg, cal_cap=cap)
    assert np.array_equal(rows[score_rows_local], score_half_global)
    assert temps[2] == 1.0 and temps[3] == 1.0


def test_stopping_selection_excludes_seam_and_gate():
    dates = np.arange(100, dtype=np.int64)
    blocks, roles = mx.block_partition(dates, 4)
    stop_sessions = np.concatenate([blocks[0], blocks[2]])
    gate_sessions = np.concatenate([blocks[1], blocks[3]])
    N = 3
    meta = {
        "date": np.asarray([stop_sessions[0], stop_sessions[1], gate_sessions[0]]),
        "tdates": np.asarray([
            [stop_sessions[2]] * 4,        # anchor stop, target stop -> in
            [gate_sessions[1]] * 4,        # anchor stop, target gate -> OUT (seam)
            [stop_sessions[3]] * 4,        # anchor gate, target stop -> OUT
        ], dtype=np.int64),
        "tag": np.zeros(N, dtype=np.int64),
    }
    m_stop, rows = rt.stopping_selection(meta, np.ones((N, 4), np.float32), blocks, roles)
    assert list(rows) == [0]
    assert m_stop[1].sum() == 0 and m_stop[2].sum() == 0


# ------------------------------------------------------ meta alignment / tags

def test_val_meta_alignment_and_tag_caps():
    frames = _mk_frames(8, 500, seed=8)
    tickers = sorted(frames)
    eval_t = set(tickers[:2])
    bounds = rt.global_date_bounds(frames, eval_days=40)
    mm, offsets, train_ds, (val_ds, val_meta), panel, surface = _build(
        frames, bounds, eval_mode="both", eval_tickers=eval_t, val_subsample=30)
    n = len(val_ds)
    assert len(val_meta["date"]) == n
    assert val_meta["tdates"].shape == (n, 4)
    assert val_meta["m_uncens"].shape == (n, 4)
    tags = val_meta["tag"]
    assert set(np.unique(tags)) <= {0, 1}
    assert (tags == 0).sum() <= 30 and (tags == 1).sum() <= 30   # per-tag caps
    assert (tags == 1).any()                   # eval tickers' val rows present, tag 1
    # meta rows aligned with dataset rows: close matches the anchor row's close
    tid = val_meta["ticker_id"][0]
    t = tickers[tid]
    f = frames[t]
    i = np.flatnonzero(f.dates.astype("datetime64[ns]").astype(np.int64)
                       == val_meta["date"][0])
    assert i.size == 1 and np.isclose(f.close[i[0]], val_meta["close"][0])
    # m_uncens is a superset of the stored mask (spike term dropped)
    assert np.all(val_meta["m_uncens"] >= np.asarray(val_ds.mask))


# ------------------------------------------------------------ both-mode routing

def test_both_mode_routing_and_surface():
    frames = _mk_frames(8, 500, seed=9)
    tickers = sorted(frames)
    eval_t = {tickers[0]}
    bounds = rt.global_date_bounds(frames, eval_days=40)
    f = frames[tickers[0]]
    split, _ = rt.row_splits(f, bounds, "both", True, False)
    in_val = (f.dates >= bounds["train_cut"]) & (f.dates < bounds["train_val_cut"])
    in_oos = f.dates >= bounds["train_val_cut"]
    assert (split[in_val] == 1).all()           # eval ticker val-era -> split 1
    assert (split[in_oos] == 2).all()
    assert (split[~in_val & ~in_oos] == -1).all()   # train era unused for eval tickers
    f2 = frames[tickers[1]]
    split2, _ = rt.row_splits(f2, bounds, "both", False, False)
    in_oos2 = f2.dates >= bounds["train_val_cut"]
    assert (split2[in_oos2] == 2).all()         # non-eval OOS rows -> eval surface
    mm, offsets, train_ds, (val_ds, val_meta), panel, surface = _build(
        frames, bounds, eval_mode="both", eval_tickers=eval_t)
    assert tickers[1] in surface                # full-universe forecast surface


def test_eval_surface_includes_labelless_anchors():
    """The most recent anchors (no labels computable) are still forecast."""
    frames = _mk_frames(4, 400, seed=10)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    mm, offsets, train_ds, valp, panel, surface = _build(frames, bounds)
    t = sorted(frames)[0]
    f = frames[t]
    assert surface[t][-1] == len(f.dates) - 1   # the final anchor has no 1d label


# ------------------------------------------------------------------ class rates

def test_class_rate_probe_excludes_embargo_crossers():
    # Uncentered on purpose (centering would NaN the crossing labels via the empty
    # pool and hide the plant), and eval_days > 127 so a val-era 6m target can
    # exist in-series while still crossing val_end.
    frames = _mk_frames(6, 600, seed=11, vol=0.0001, drift=0.0)  # driftless: all neutral
    bounds = rt.global_date_bounds(frames, eval_days=140)
    t0 = sorted(frames)[0]
    f = frames[t0]
    split, split_end = rt.row_splits(f, bounds, "time", False, False)
    crossing = (~np.isnat(f.target_dates[:, 3])
                & (f.target_dates[:, 3] > np.asarray(split_end, dtype="datetime64[ns]")))
    val_rows = crossing & (split == 1) & np.isfinite(f.z[:, 3])
    assert val_rows.any()
    f.z[val_rows, 3] = 100.0                    # would flip the val 6m up-rate
    cfg = V5Config(n_features=fx.N_FEATURES, window=140, min_real_rows=fx.MIN_REAL_ROWS)
    rates = rt.class_rate_log(frames, bounds, cfg, "time", set(), set())
    assert rates["val"][3, 2] < 0.5, rates["val"][3]   # crossers never counted


# ------------------------------------------------------------------ temperatures

def test_fit_temperatures_chunked_scoped_and_1m6m_unit():
    torch.manual_seed(0)
    cfg = V5Config(n_features=12, window=20, min_real_rows=8, d_model=16, n_blocks=1,
                   d_state=4)
    members = [V5Backbone(cfg).eval() for _ in range(2)]
    n_rows, window = 400, 20
    rows = np.random.default_rng(0).standard_normal((n_rows, 12)).astype(np.float32)
    starts = np.arange(0, 300, 2, dtype=np.int64)
    N = len(starts)
    ds = rt.WindowDataset(rows, starts, np.full(N, window, dtype=np.int64),
                          np.random.default_rng(1).standard_normal((N, 4)).astype(np.float32),
                          np.ones((N, 4), dtype=np.float32), window)
    dates = np.repeat(np.arange(30, dtype=np.int64), 5)[:N]
    blocks, roles = mx.block_partition(np.unique(dates), 4)
    meta = {"date": dates, "tdates": np.repeat(dates[:, None], 4, 1),
            "tag": np.zeros(N, dtype=np.int64)}
    t_big, cache = rt.fit_temperatures(members, ds, meta, cfg, "cpu", 64,
                                       blocks=blocks, roles=roles, chunk=100_000)
    t_small, _ = rt.fit_temperatures(members, ds, meta, cfg, "cpu", 64,
                                     blocks=blocks, roles=roles, chunk=7)
    assert t_big == t_small                     # chunked == unchunked
    assert t_big[2] == 1.0 and t_big[3] == 1.0  # persisted 1m/6m exactly 1.0
    # All-gate targets -> empty fit population -> T stays 1 on 1d/1w too.
    gate_dates = blocks[1]
    meta_gate = {"date": np.repeat(gate_dates[:1], N),
                 "tdates": np.full((N, 4), gate_dates[0], dtype=np.int64),
                 "tag": np.zeros(N, dtype=np.int64)}
    t_gate, _ = rt.fit_temperatures(members, ds, meta_gate, cfg, "cpu", 64,
                                    blocks=blocks, roles=roles)
    assert t_gate == [1.0, 1.0, 1.0, 1.0]


# ------------------------------------------------------------------- eval pass

def test_eval_pass_recovers_planted_horizon():
    torch.manual_seed(3)
    cfg = V5Config(n_features=12, window=20, min_real_rows=8, d_model=16, n_blocks=1,
                   d_state=4)
    model = V5Backbone(cfg).eval()
    n_dates, n_names = 36, 40
    N = n_dates * n_names
    rows = np.random.default_rng(3).standard_normal((N * 2, 12)).astype(np.float32)
    starts = np.arange(N, dtype=np.int64)
    z = np.zeros((N, 4), dtype=np.float32)
    ds = rt.WindowDataset(rows, starts, np.full(N, 20, dtype=np.int64), z,
                          np.ones((N, 4), dtype=np.float32), 20)
    with torch.no_grad():
        L = []
        for i in range(0, N, 256):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 256, N))])
            L.append(model(x))
        logits = torch.cat(L, 0)
    probs = logits.softmax(-1)
    half = cfg.n_bins // 2
    k = cfg.theta_bins[0]
    score = (probs[:, 0, half + k:].sum(-1) - probs[:, 0, :half - k].sum(-1)).numpy()
    rng = np.random.default_rng(4)
    z[:, 0] = 3.0 * (score - score.mean()) / (score.std() + 1e-9) + rng.standard_normal(N)
    z[:, 1] = rng.standard_normal(N)
    z[:, 2] = rng.standard_normal(N)
    z[:, 3] = rng.standard_normal(N)
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    blocks, roles = mx.block_partition(np.unique(dates), 4)
    meta = {"date": dates, "ticker_id": np.tile(np.arange(n_names), n_dates),
            "tdates": np.repeat(dates[:, None], 4, 1),
            "tradable": np.ones(N, dtype=bool), "close": np.full(N, 10.0),
            "sigma_hat": np.full(N, 0.02), "tag": np.zeros(N, dtype=np.int64),
            "m_uncens": np.ones((N, 4), dtype=np.float32)}
    m_stop, stop_rows = rt.stopping_selection(meta, ds.mask, blocks, roles)
    out = rt.eval_pass(model, ds, meta, cfg, "cpu", 256, 0, blocks, roles,
                       ic_floor=10, m_stop=m_stop, stop_rows=stop_rows,
                       cal_cap=1000)
    assert out["t_1d_stop"] > 3, out["t_1d_stop"]
    assert abs(out["t_1w_stop"]) < 3
    assert np.isfinite(out["stopping_score"])
    recs = out["ic_records"]
    r6 = next(r for r in recs if r["horizon"] == "6m" and r["score"] == "score"
              and r["role"] == "all" and r["trad"] == "all" and r["mask"] == "primary")
    assert abs(r6["t"]) < 3
    assert np.isfinite(out["cal_ce_stop"])


def test_train_member_bootstrap_selects_checkpoint():
    """With the IC name floor above the universe size, the stopping score stays -inf
    for the whole run and the calibrated-CE bootstrap must still select a
    checkpoint."""
    torch.manual_seed(5)
    cfg = V5Config(n_features=12, window=20, min_real_rows=8, d_model=16, n_blocks=1,
                   d_state=4)
    rows = np.random.default_rng(5).standard_normal((600, 12)).astype(np.float32)
    N = 200
    starts = np.arange(N, dtype=np.int64)
    mk = lambda seed: rt.WindowDataset(
        rows, starts, np.full(N, 20, dtype=np.int64),
        np.random.default_rng(seed).standard_normal((N, 4)).astype(np.float32),
        np.ones((N, 4), dtype=np.float32), 20)
    train_ds, val_ds = mk(6), mk(7)
    dates = np.repeat(np.arange(20, dtype=np.int64), 10)
    blocks, roles = mx.block_partition(np.unique(dates), 4)
    val_meta = {"date": dates, "ticker_id": np.tile(np.arange(10), 20),
                "tdates": np.repeat(dates[:, None], 4, 1),
                "tradable": np.ones(N, dtype=bool), "close": np.full(N, 10.0),
                "sigma_hat": np.full(N, 0.02), "tag": np.zeros(N, dtype=np.int64),
                "m_uncens": np.ones((N, 4), dtype=np.float32)}
    run_dir = tempfile.mkdtemp()
    model, crit = rt.train_member(
        0, 0, cfg, train_ds, val_ds, val_meta, "cpu", max_steps=4,
        eval_every_steps=2, patience=3, lr=1e-3, batch_size=64, steps_per_epoch=4,
        num_workers=0, log_every=2, blocks=blocks, roles=roles,
        ic_floor=999, run_dir=run_dir, warmup_steps=1, cal_cap=100)
    assert np.isfinite(crit)                    # CE-driven selection happened
    hist = [json.loads(l) for l in open(os.path.join(run_dir,
                                                     "eval_history_member0.jsonl"))]
    assert all(line["bootstrap"] for line in hist)
    assert all(line["stopping_score"] == float("-inf") or
               not np.isfinite(line["stopping_score"]) for line in hist)
    thist = [json.loads(l) for l in open(os.path.join(run_dir,
                                                      "train_history_member0.jsonl"))]
    assert thist and all({"step", "lr", "loss_ema", "per_h_ema", "it_s"} <= set(l)
                         for l in thist)
    assert os.path.exists(os.path.join(run_dir, "member_0_latest.pt"))
    assert os.path.exists(os.path.join(run_dir, "member_0_best.pt"))
    gate_keys = {"lr_scale", "train_ema", "lr_gate_train_declining",
                 "lr_gate_ce_regressing", "ce_excess"}
    assert all(gate_keys <= set(line) for line in hist)
    # patience (3) < lr_patience (4): early stop structurally precedes any drop.
    assert hist[-1]["lr_scale"] == 1.0


def test_lr_drop_gates_truth_table():
    """Declining needs a real EMA fall over the window; regressing needs the last
    window CEs all above the best by the margin, which a record eval can never
    satisfy, so improvement of any speed holds the LR."""
    g = rt.lr_drop_gates
    down = [3.40, 3.39, 3.38, 3.37, 3.36]
    assert g(down, [3.40, 3.41, 3.41, 3.41], 3.36, 4, 0.002, 3e-3) == (True, True)
    # record at the current eval: the window min equals the best
    assert g(down, [3.40, 3.401, 3.402, 3.399], 3.399, 4, 0.002, 3e-3)[1] is False
    # rapid improvement: regressing stays False at every prefix
    head = [3.427238, 3.414616, 3.413708, 3.411541, 3.408673]
    best = float("inf")
    for i, c in enumerate(head):
        best = min(best, c)
        assert g(down, head[:i + 1], best, 4, 0.002, 3e-3)[1] is False
    # sustained rise above a stale best fires; sub-margin flat noise does not
    assert g(down, [3.3941, 3.3938, 3.3945, 3.3940], 3.3900, 4, 0.002, 3e-3)[1] is True
    flat = [3.3915, 3.3908, 3.3912, 3.3902]
    assert g(down, flat, min(flat) - 1e-4, 4, 0.002, 3e-3)[1] is False
    # short histories hold: EMA needs window + 1 entries, CE needs window
    assert g(down[:4], [3.41] * 4, 3.36, 4, 0.002, 3e-3)[0] is False
    assert g(down, [3.41] * 3, 3.36, 4, 0.002, 3e-3)[1] is False
    # non-finite endpoints or window members hold; flat EMA holds
    assert g([float("nan")] + down[1:], [3.41] * 4, 3.36, 4, 0.002, 3e-3)[0] is False
    assert g(down, [3.41, float("nan"), 3.41, 3.41], 3.36, 4, 0.002, 3e-3)[1] is False
    assert g([3.36] * 5, [3.41] * 4, 3.36, 4, 0.002, 3e-3)[0] is False


def test_next_run_dir_sequential_and_claiming():
    """Numbered run dirs advance monotonically past any existing number, and the
    claim is the mkdir itself."""
    root = tempfile.mkdtemp()
    a = rt.next_run_dir(root=root)
    b = rt.next_run_dir(root=root)
    assert (a.name, b.name) == ("r1", "r2") and a.is_dir() and b.is_dir()
    os.makedirs(os.path.join(root, "r7"))
    c = rt.next_run_dir(root=root)
    assert c.name == "r8"


# ------------------------------- flags, embargo, manifest, floors, truncation

def test_data_end_truncates_before_feature_build(monkeypatch=None):
    oh = _mk_ohlcv(400, seed=12)
    fear = pd.DataFrame({"date": oh["date"], "fear_greed": 50.0})
    cut = oh["date"].iloc[299]
    full = fx.build_feature_frame("SYN", oh, fear)
    trunc = fx.build_feature_frame("SYN", oh[oh["date"] <= cut].reset_index(drop=True), fear)
    assert trunc.dates.max() == np.datetime64(cut)
    # No z computed from post-cutoff prices: the last 1d label dies at n-d-2.
    n = len(trunc.dates)
    assert np.isnan(trunc.z[n - 2, 0]) and np.isnan(trunc.z[n - 1, 0])
    # And where both are defined, pre-cutoff z agree (truncation leaks nothing in).
    both = np.isfinite(full.z[:250, 0]) & np.isfinite(trunc.z[:250, 0])
    assert np.allclose(full.z[:250, 0][both], trunc.z[:250, 0][both])


def test_embargo_dates_reanchored():
    frames = _mk_frames(2, 400, seed=13)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    out = rt._embargo_dates(bounds, (1, 5, 21, 126))
    u = bounds["unique_dates"]
    pos = int(np.searchsorted(u, bounds["train_end"], side="right")) - 1
    assert out["train"]["1d"] == str(np.datetime_as_string(u[pos - 2], unit="D"))
    assert out["train"]["1w"] == str(np.datetime_as_string(u[pos - 6], unit="D"))


def test_train_price_floor():
    frames = _mk_frames(4, 400, seed=14)
    penny = _mk_ohlcv(400, seed=15, base=0.5, drift=0.0)     # close ~ $0.5
    fear = pd.DataFrame({"date": penny["date"], "fear_greed": 50.0})
    frames["PENNY"] = fx.build_feature_frame("PENNY", penny, fear)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    mm, offsets, train_ds, (val_ds, val_meta), panel, surface = _build(frames, bounds)
    lo, n = offsets["PENNY"]
    starts = np.asarray(train_ds.start_abs)
    # No train window may end inside PENNY's row span.
    anchors_abs = starts + np.asarray(train_ds.n_real) - 1
    assert not ((anchors_abs >= lo) & (anchors_abs < lo + n)).any()
    # but PENNY's val rows survive (evaluation stays honest on the full universe)
    tickers = sorted(frames)
    pen_id = tickers.index("PENNY")
    assert (val_meta["ticker_id"] == pen_id).any()


def test_manifest_write_check_and_finalize_assert():
    run_dir = tempfile.mkdtemp()
    ns = argparse.Namespace(
        eval_mode="time", seed=42, ticker_holdout_frac=0.0, eval_days=40,
        data_start=None, data_end=None, window=130, val_subsample=1000,
        max_windows=512, tickers=None, max_tickers=6, members=2,
        member_start=0, member_count=None, cache_dir=None, run_dir=run_dir,
        smoke=True, forecast_only=False)
    rt.write_run_manifest(ns, __import__("pathlib").Path(run_dir))
    rt.append_manifest_scaler_rows(__import__("pathlib").Path(run_dir), 12345)
    path = os.path.join(run_dir, "run_manifest.json")
    before = open(path, "rb").read()
    manifest = rt.assert_finalize_snapshot(ns, __import__("pathlib").Path(run_dir))
    assert manifest["scaler_rows"] == 12345
    assert open(path, "rb").read() == before      # finalize leaves it byte-identical

    ns2 = argparse.Namespace(**{**vars(ns), "eval_days": 254})
    try:
        rt.assert_finalize_snapshot(ns2, __import__("pathlib").Path(run_dir))
    except SystemExit:
        pass
    else:
        raise AssertionError("changed --eval-days must refuse to finalize")

    tampered = json.loads(before)
    tampered["data_fingerprint"]["files"][0][1] += 1
    open(path, "w").write(json.dumps(tampered))
    try:
        rt.assert_finalize_snapshot(ns, __import__("pathlib").Path(run_dir))
    except SystemExit:
        pass
    else:
        raise AssertionError("tampered fingerprint must refuse to finalize")


def test_label_shuffle_within_date_preserves_sets():
    frames = _mk_frames(6, 400, seed=16)
    bounds = rt.global_date_bounds(frames, eval_days=40)
    rt.center_labels(frames, bounds, "time", set(), k_min=2)
    unique = bounds["unique_dates"]
    def date_sets(hi):
        out = {}
        for t, f in frames.items():
            split, split_end = rt.row_splits(f, bounds, "time", False, False)
            mask = fx.label_mask(f.z, f.spike_free, f.target_dates, f.dates, split_end)
            sel = (split == 0) & (mask[:, hi] > 0)
            for a in np.flatnonzero(sel):
                d = int(np.searchsorted(unique, f.dates[a]))
                out.setdefault(d, []).append(round(float(f.z[a, hi]), 6))
        return {d: sorted(v) for d, v in out.items()}
    before = date_sets(0)
    rt.label_shuffle_within_date(frames, bounds, "time", set(), set(), seed=1)
    after = date_sets(0)
    assert before == after                      # per-date multisets preserved
    # and the assignment actually moved: at least one ticker's series changed
    frames2 = _mk_frames(6, 400, seed=16)
    rt.center_labels(frames2, bounds, "time", set(), k_min=2)
    a = list(frames.values())[0].z[:, 0]
    b = list(frames2.values())[0].z[:, 0]
    both = np.isfinite(a) & np.isfinite(b)
    assert not np.allclose(a[both], b[both])
