"""Feature-builder tests: formula parity, label/volatility guards, normalization."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TrainingData"))

import features_v5 as fx


def _synthetic_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(0.001 + 0.02 * rng.standard_normal(n)))
    openp = close * (1 + 0.003 * rng.standard_normal(n))
    high = np.maximum(openp, close) * (1 + 0.004 * np.abs(rng.standard_normal(n)))
    low = np.minimum(openp, close) * (1 - 0.004 * np.abs(rng.standard_normal(n)))
    volume = rng.integers(1_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low,
                         "close": close, "volume": volume})


def test_min_real_rows_is_derived():
    assert fx.MIN_REAL_ROWS == max(c.warm_up + c.lag for c in fx.FEATURE_SPEC)
    assert fx.MIN_REAL_ROWS == 126
    assert fx.N_FEATURES == 40


def test_formula_parity_with_processor():
    """Identical OHLCV through the legacy preprocessor and the in-builder recompute
    agree on every shared indicator column to floating-point noise."""
    import processor as P

    oh = _synthetic_ohlcv()
    tmp = tempfile.mkdtemp()
    raw_csv = os.path.join(tmp, "SYN_daily.csv")
    out_csv = os.path.join(tmp, "SYN_out.csv")
    oh.assign(date=oh["date"].dt.strftime("%Y-%m-%d")).to_csv(raw_csv, index=False)
    P.process_file(raw_csv, out_csv, df_fear_greed=None)
    proc = pd.read_csv(out_csv, parse_dates=["date"])
    mine = fx.raw_indicators(oh)
    shared = [c for c in proc.columns if c in mine.columns and c not in ("date", "close")]
    assert len(shared) >= 25
    merged = proc.merge(mine, on="date", suffixes=("_p", "_m"))
    worst = max(np.nanmax(np.abs(merged[f"{c}_p"].to_numpy(float) - merged[f"{c}_m"].to_numpy(float)))
                for c in shared)
    assert worst < 1e-6, f"max per-column diff {worst}"


def test_ewma_spike_excluded_from_update():
    """A split-print return does not update the EWMA: sigma_hat is unchanged that day."""
    r = np.full(40, 0.01)
    r[25] = np.log(2.0)            # the only spike, > ln(1.5)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    sigma_hat, _, spike = fx.ewma_sigma_hat(close)
    spike_days = np.where(spike)[0]
    assert len(spike_days) == 1
    s = int(spike_days[0])
    assert np.isclose(sigma_hat[s], sigma_hat[s - 1])


def test_ewma_seed_excludes_spike():
    """A spike inside the seed window is excluded from the seed sample variance."""
    r = 0.01 * np.sin(np.arange(1, 31) / 3.0)
    r[5] = np.log(2.0)             # spike inside the first rows
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    sigma_hat, log_ret, spike = fx.ewma_sigma_hat(close)
    seed_idx = np.argmax(np.isfinite(sigma_hat))
    nonspike = log_ret[1:seed_idx + 1][~spike[1:seed_idx + 1]]
    assert len(nonspike) == fx.EWMA_SEED_WINDOW
    expected = max(float(np.var(nonspike, ddof=1)), fx.EWMA_SIGMA_FLOOR ** 2)
    assert np.isclose(sigma_hat[seed_idx] ** 2, expected)


def test_split_spike_zeros_label_mask():
    """A planted spike in (t, t+d_h] zeros that horizon's label; horizons clearing it
    stay valid."""
    n = 300
    r = np.full(n, 0.005)
    s = 200
    r[s] = np.log(2.0)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))[:n]
    dates = pd.bdate_range("2015-01-02", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    data_end = np.full(n, dates.max(), dtype="datetime64[ns]")
    m = fx.label_mask(spike_free, target_dates, data_end)
    t = 190                       # horizons: 1,5,21,126 days
    assert m[t, 0] == 1.0         # 1d clears the spike at +10
    assert m[t, 1] == 1.0         # 1w (5d) clears it
    assert m[t, 2] == 0.0         # 1m (21d) spans index 200
    assert m[t, 3] == 0.0         # 6m (126d) spans index 200


def test_embargo_leakage_line():
    """The target-date <= split-end comparison is the only thing standing between a
    label and the next split."""
    n = 200
    close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    dates = pd.bdate_range("2016-01-04", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    t = 150
    one_day_target = target_dates[t, 0]
    # Split-end one day before the 1d target -> embargoed; on the target -> allowed.
    before = np.full(n, one_day_target - np.timedelta64(1, "D"), dtype="datetime64[ns]")
    on = np.full(n, one_day_target, dtype="datetime64[ns]")
    assert fx.label_mask(spike_free, target_dates, before)[t, 0] == 0.0
    assert fx.label_mask(spike_free, target_dates, on)[t, 0] == 1.0


def test_degenerate_iqr_fallback():
    mostly_zero = np.array([0.0] * 90 + list(range(1, 11)), dtype=float)
    med, scale, rule = fx._robust_params(mostly_zero)
    assert rule == "mad_nonzero" and scale > 0
    constant = np.full(50, 5.0)
    med, scale, rule = fx._robust_params(constant)
    assert rule == "degenerate" and scale == 1.0
    all_zero = np.zeros(50)
    med, scale, rule = fx._robust_params(all_zero)
    assert rule == "degenerate" and scale == 1.0


def test_norm_hash_guard():
    rows = np.random.default_rng(0).standard_normal((500, fx.N_FEATURES))
    sc = fx.RobustScaler.fit(rows)
    data = sc.to_dict()
    data["feature_names_hash"] = "deadbeef"
    try:
        fx.RobustScaler.from_dict(data)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong feature-name hash must be rejected")
    bad_order = sc.to_dict()
    bad_order["channels"][0], bad_order["channels"][1] = bad_order["channels"][1], bad_order["channels"][0]
    try:
        fx.RobustScaler.from_dict(bad_order)
    except ValueError:
        pass
    else:
        raise AssertionError("reordered channels must be rejected")


def test_fp16_roundtrip_idempotent():
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((300, fx.N_FEATURES)).astype(np.float32)
    sc = fx.RobustScaler.fit(rows)
    out = sc.transform(rows)
    assert out.dtype == np.float32
    assert np.array_equal(out, sc.transform(rows))               # deterministic
    assert np.array_equal(out, out.astype(np.float16).astype(np.float32))  # fp16-exact
    assert np.isfinite(out).all()


def test_assemble_window_left_pad():
    norm = np.random.default_rng(0).standard_normal((300, fx.N_FEATURES)).astype(np.float32)
    norm[:, fx.PAD_COL] = 0.0
    full = fx.assemble_window(norm, anchor=299, window=252, min_real_rows=126)
    assert full.shape == (252, fx.N_FEATURES) and full[:, fx.PAD_COL].sum() == 0
    short = fx.assemble_window(norm, anchor=150, window=252, min_real_rows=126)
    assert short.shape == (252, fx.N_FEATURES) and int(short[:, fx.PAD_COL].sum()) == 252 - 151
    assert short[:101, fx.PAD_COL].min() == 1.0     # pads are on the left
    assert fx.assemble_window(norm, anchor=100, window=252, min_real_rows=126) is None


def test_calendar_bounds():
    dates = pd.bdate_range("2020-01-06", periods=20).to_numpy()   # starts on a Monday
    ch = fx._calendar_channels(dates)
    for name, arr in ch.items():
        assert np.all(arr >= -1.0001) and np.all(arr <= 1.0001), name
    assert np.isclose(ch["sin_week"][0], 0.0) and np.isclose(ch["cos_week"][0], 1.0)
