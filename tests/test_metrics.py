"""Metrics-module tests: IC recovery, HAC behavior, nulls, blocks, uncertainty."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v5 import metrics as mx


def _planted_panel(n_dates=60, n_names=80, ic=0.3, seed=0):
    """Synthetic panel: scores rank-correlated with outcomes at roughly ``ic``."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    tickers = np.tile(np.arange(n_names, dtype=np.int64), n_dates)
    z = rng.standard_normal(n_dates * n_names)
    noise = rng.standard_normal(n_dates * n_names)
    w = ic / np.sqrt(1 - ic * ic)
    scores = w * z + noise
    return dates, tickers, scores, z


def test_spearman_ic_recovers_planted_signal():
    dates, _, scores, z = _planted_panel(ic=0.3)
    keys, ics = mx.spearman_ic(scores, z, dates, min_names=30)
    assert len(keys) == 60
    assert 0.15 < ics.mean() < 0.45, ics.mean()
    keys0, ics0 = mx.spearman_ic(-scores, z, dates, min_names=30)
    assert ics0.mean() < -0.15


def test_spearman_ic_floors_and_degenerates():
    dates = np.repeat([0, 1], [10, 40])
    z = np.random.default_rng(0).standard_normal(50)
    scores = z + 0.1
    keys, ics = mx.spearman_ic(scores, z, dates, min_names=30)
    assert list(keys) == [1]                       # 10-name date dropped by the floor
    const = np.ones(50)
    keys, ics = mx.spearman_ic(const, z, dates, min_names=30)
    assert len(keys) == 0 and np.isfinite(ics).all()   # zero-variance dropped, no NaN


def test_hac_se_exceeds_naive_on_ar1():
    rng = np.random.default_rng(1)
    n = 300
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):                          # AR(1), rho=0.7
        e[t] = 0.7 * e[t - 1] + rng.standard_normal()
    sess = np.arange(n)
    mean, se, t_stat, cnt = mx.hac_t(e, sess, lag=10)
    naive = e.std(ddof=1) / np.sqrt(n)
    assert cnt == n and np.isfinite(t_stat)
    assert se > 1.5 * naive, (se, naive)


def test_hac_zero_dates_never_raises():
    mean, se, t, n = mx.hac_t(np.asarray([]), np.asarray([], dtype=np.int64), lag=5)
    assert n == 0 and np.isnan(mean) and np.isnan(se) and np.isnan(t)


def test_hac_blocks_are_independent_clusters():
    """A lag-1 pair straddling a block boundary contributes nothing."""
    e = np.asarray([1.0, -1.0, 1.0, -1.0])
    sess = np.asarray([0, 1, 2, 3])
    one = mx.hac_t(e, sess, lag=1, block_ids=np.zeros(4))
    split = mx.hac_t(e, sess, lag=1, block_ids=np.asarray([0, 0, 1, 1]))
    assert one[1] != split[1]                      # boundary product excluded
    assert one[3] == split[3] == 4


def test_block_partition_disjoint_cover():
    sessions = np.arange(254, dtype=np.int64)
    blocks, roles = mx.block_partition(sessions, n_blocks=4)
    assert roles == ["stop", "gate", "stop", "gate"]
    joined = np.concatenate(blocks)
    assert np.array_equal(np.sort(joined), sessions)
    assert sum(b.size for b in blocks) == 254      # disjoint cover, no duplicates


def test_role_membership_no_label_on_both_sides():
    sessions = np.arange(100, dtype=np.int64)
    blocks, roles = mx.block_partition(sessions, n_blocks=4)
    tdates = np.random.default_rng(0).integers(0, 100, size=(500, 4)).astype(np.int64)
    stop = mx.role_label_mask(tdates, blocks, roles, "stop")
    gate = mx.role_label_mask(tdates, blocks, roles, "gate")
    assert not np.any(stop & gate)
    assert np.all(stop | gate)                     # every in-range target has one role


def test_stopping_score_guards():
    assert mx.stopping_score(2.0, 2.0) == float(4.0 / np.sqrt(2.0))
    assert mx.stopping_score(2.0, 2.0, n_1d=5, n_1w=50) == float("-inf")
    assert mx.stopping_score(float("nan"), 2.0) == float("-inf")
    assert not np.isnan(mx.stopping_score(float("nan"), float("nan")))


def test_nulls_kill_planted_signal():
    dates, tickers, scores, z = _planted_panel(n_dates=80, n_names=60, ic=0.4, seed=2)
    meta = {"date": dates, "ticker_id": tickers}
    keys, ics = mx.spearman_ic(scores, z, dates)
    assert mx.hac_t(ics, np.searchsorted(np.unique(dates), keys), 5)[2] > 5

    for null_fn, seed in ((mx.null_shuffle_within_date, 3), (mx.null_circular_shift, 4)):
        s2 = null_fn(scores.reshape(-1, 1), meta, seed)[:, 0]
        k2, i2 = mx.spearman_ic(s2, z, dates)
        _, _, t2, _ = mx.hac_t(i2, np.searchsorted(np.unique(dates), k2), 5)
        assert abs(t2) < 3, (null_fn.__name__, t2)


def test_ic_suite_populations_and_alt_mask_superset():
    rng = np.random.default_rng(5)
    n_dates, n_names = 40, 50
    N = n_dates * n_names
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    z = rng.standard_normal((N, 4))
    up = rng.random((N, 4))
    down = rng.random((N, 4)) * (1 - up)
    mask = np.ones((N, 4), dtype=np.float32)
    censored = rng.random((N, 4)) < 0.1
    mask[censored] = 0.0
    alt = np.ones((N, 4), dtype=np.float32)        # spike term dropped: superset
    tdates = np.repeat(dates[:, None], 4, axis=1)
    meta = {"date": dates, "tdates": tdates, "ticker_id": np.tile(np.arange(n_names), n_dates)}
    cfg = mx.default_cfg(min_names=30)
    blocks, roles = mx.block_partition(np.unique(dates), 4)
    trad = rng.random(N) < 0.7
    records, series = mx.ic_suite(up, down, z, mask, meta, cfg, blocks, roles,
                                  tradable=trad, alt_mask=alt)
    combos = {(r["score"], r["mask"], r["role"], r["trad"]) for r in records}
    assert ("score", "primary", "all", "all") in combos
    assert ("p_up", "uncensored", "stop", "tradable") in combos
    for r in records:
        if r["mask"] != "primary" or r["role"] != "all" or r["trad"] != "all":
            continue
        alt_r = next(a for a in records
                     if a["horizon"] == r["horizon"] and a["score"] == r["score"]
                     and a["mask"] == "uncensored" and a["role"] == "all" and a["trad"] == "all")
        assert alt_r["n_dates"] >= r["n_dates"]    # superset population


def test_top_of_ranking_return_units():
    rng = np.random.default_rng(6)
    n_dates, n_names = 30, 60
    N = n_dates * n_names
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    z = np.tile(np.linspace(-1, 1, n_names), n_dates)[:, None].repeat(4, 1)
    score = z + 0.01 * rng.standard_normal((N, 4))     # near-perfect ranking
    mask = np.ones((N, 4))
    sigma = np.full(N, 0.02)
    meta = {"date": dates, "ticker_id": np.tile(np.arange(n_names), n_dates)}
    records, series = mx.top_of_ranking(score, z, mask, meta, sigma=sigma)
    rec = next(r for r in records if r["horizon"] == "1d" and r["trad"] == "all"
               and r["mask"] == "primary")
    assert rec["topdec_z_mean"] > 0.8
    dec = series[("1d", "primary", "all")]
    assert np.allclose(dec["topdec_ret"], dec["topdec_z"] * 0.02 * 1.0)  # sqrt(1)=1
    rec6 = next(r for r in records if r["horizon"] == "6m" and r["trad"] == "all"
                and r["mask"] == "primary")
    assert rec6["topdec_ret_mean"] > 0             # scaled by sqrt(126), sign kept


def test_uncertainty_quality_recovers_planted_correlation():
    rng = np.random.default_rng(7)
    N = 4000
    std = rng.random((N, 4)) * 0.2
    z = rng.standard_normal((N, 4))
    theta = [0.674] * 4
    truth = (z > theta[0]).astype(float)
    # Error magnitude driven by std: prediction misses truth by ~2*std toward the
    # interior of [0, 1], plus small noise, so |prob - truth| ranks like std.
    err = np.clip(std * 2.0 + 0.02 * rng.random((N, 4)), 0, 1)
    prob_up = np.where(truth > 0, 1.0 - err, err)
    mask = np.ones((N, 4))
    cfg = {"theta": theta, "horizon_labels": mx.HORIZON_LABELS}
    out = mx.uncertainty_quality(std, prob_up, z, mask, {"date": np.zeros(N)}, cfg)
    for hl in mx.HORIZON_LABELS:
        assert out[hl]["spearman"] > 0.3, out[hl]
        assert len(out[hl]["deciles"]) == 10
        assert out[hl]["deciles"][0] < out[hl]["deciles"][-1]
