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


def test_targets_reanchored_exact():
    """Exact z on a geometric series under the next-close-entry anchoring:
    z[t, h] = (logC[t+1+d] - logC[t+1]) / (sigma_hat[t] * sqrt(d))."""
    n = 300
    mu = 0.001
    close = 100 * np.exp(mu * np.arange(n))
    dates = pd.bdate_range("2015-01-02", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close, dates)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    t = 150
    for hi, d in enumerate(fx.HORIZON_DAYS):
        expected = mu * d / (sigma_hat[t] * np.sqrt(d))
        assert np.isclose(z[t, hi], expected, rtol=1e-5), (hi, z[t, hi], expected)
        assert target_dates[t, hi] == dates[t + 1 + d]
    for hi, d in enumerate(fx.HORIZON_DAYS):
        assert np.isfinite(z[n - d - 2, hi])          # last valid anchor
        assert np.isnan(z[n - d - 1, hi])
        assert np.isnat(target_dates[n - d - 1, hi])


def test_split_spike_zeros_label_mask():
    """A planted spike in (t+1, t+1+d_h] zeros that horizon's label; horizons
    clearing it stay valid."""
    n = 300
    r = np.full(n, 0.005)
    s = 200
    r[s] = np.log(2.0)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))[:n]
    dates = pd.bdate_range("2015-01-02", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close, dates)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    data_end = np.full(n, dates.max(), dtype="datetime64[ns]")
    m = fx.label_mask(z, spike_free, target_dates, dates, data_end)
    t = 190                       # windows now (t+1, t+1+d]: rows 192 .. 191+d
    assert m[t, 0] == 1.0         # 1d window [192, 192] clears the spike at 200
    assert m[t, 1] == 1.0         # 1w window [192, 196] clears it
    assert m[t, 2] == 0.0         # 1m window [192, 212] spans row 200
    assert m[t, 3] == 0.0         # 6m target (317) runs off the series end


def test_split_at_entry_not_censored():
    """A split print exactly at t+1 (the entry close) must NOT censor -- entry and
    target closes sit on the same post-split basis; a print at t+1+d must censor."""
    n = 300
    r = np.full(n, 0.005)
    s = 195
    r[s - 1] = np.log(2.0)        # r index k is the return into close index k+1
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))[:n]
    dates = pd.bdate_range("2015-01-02", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close, dates)
    assert spike[s]               # the spike row is close index 195
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    assert spike_free[s - 1, 0]           # anchor 194: entry row 195 is the spike -> fine
    assert not spike_free[s - 2, 0]       # anchor 193: target row 195 is the spike -> censored


def test_embargo_leakage_line():
    """The target-date <= split-end comparison is the only thing standing between a
    label and the next split."""
    n = 200
    close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    dates = pd.bdate_range("2016-01-04", periods=n).to_numpy()
    sigma_hat, _, spike = fx.ewma_sigma_hat(close, dates)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    t = 150
    one_day_target = target_dates[t, 0]
    # Split-end one day before the 1d target -> embargoed; on the target -> allowed.
    before = np.full(n, one_day_target - np.timedelta64(1, "D"), dtype="datetime64[ns]")
    on = np.full(n, one_day_target, dtype="datetime64[ns]")
    assert fx.label_mask(z, spike_free, target_dates, dates, before)[t, 0] == 0.0
    assert fx.label_mask(z, spike_free, target_dates, dates, on)[t, 0] == 1.0


def test_gap_mask_and_finite_term():
    """The calendar-gap ceiling kills labels spanning a feed hole per horizon (a
    small hole kills 1d but not 6m -- mask monotonicity broken by design), and
    NaN-z rows (pre-seed sigma) are masked by the finiteness term."""
    a = pd.bdate_range("2015-01-05", periods=150)
    b = pd.bdate_range(a[-1] + pd.Timedelta(days=8), periods=160)   # 8-day hole
    dates = np.concatenate([a.to_numpy(), b.to_numpy()])
    n = len(dates)
    close = 100 * np.exp(0.001 * np.arange(n))
    sigma_hat, _, spike = fx.ewma_sigma_hat(close, dates)
    z, spike_free, target_dates = fx._targets(close, dates, sigma_hat, spike)
    data_end = np.full(n, dates.max(), dtype="datetime64[ns]")
    m = fx.label_mask(z, spike_free, target_dates, dates, data_end)

    # Property check: wherever every other criterion holds, the mask equals the
    # per-horizon calendar-span rule computed straight from the dates.
    anchor_days = dates.astype("datetime64[D]").astype(np.int64)
    for hi, d in enumerate(fx.HORIZON_DAYS):
        for t in range(30, n - d - 2):
            if not (np.isfinite(z[t, hi]) and spike_free[t, hi]):
                continue
            span = int(target_dates[t, hi].astype("datetime64[D]").astype(np.int64)
                       - anchor_days[t])
            assert m[t, hi] == float(span <= fx.GAP_LIMIT_DAYS[hi]), (t, hi, span)

    seam = 148                    # 1d label spans the hole; 6m label also spans it
    assert m[seam, 0] == 0.0      # 8-day span > 6-day 1d limit
    assert m[seam, 3] == 1.0      # ~186-day span <= 193-day 6m limit: alive
    assert np.isnan(z[5, 0]) and m[5, 0] == 0.0   # pre-seed NaN z masked


def test_ewma_reseed_after_long_gap():
    """A >90-calendar-day hole resets the EWMA: NaN sigma for the 20-row re-seed
    span (the halt-spanning return excluded), then re-seeded near the post-gap
    regime's volatility."""
    a = pd.bdate_range("2014-01-06", periods=120)
    b = pd.bdate_range(a[-1] + pd.Timedelta(days=365), periods=120)
    dates = np.concatenate([a.to_numpy(), b.to_numpy()])
    n = len(dates)
    r = np.empty(n - 1)
    r[:119] = 0.01 * np.where(np.arange(119) % 2 == 0, 1, -1)       # pre-gap vol ~1%
    r[119:] = 0.03 * np.where(np.arange(n - 1 - 119) % 2 == 0, 1, -1)  # post-gap ~3%
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    sigma_hat, _, _ = fx.ewma_sigma_hat(close, dates)
    g = 120                                        # first post-gap row
    assert np.isfinite(sigma_hat[g - 1])           # pre-gap EWMA alive
    assert np.all(np.isnan(sigma_hat[g:g + 20]))   # reset + 20-row re-seed span
    reseeded = sigma_hat[g + 20]
    assert np.isfinite(reseeded)
    assert 0.02 < reseeded < 0.04, reseeded        # post-gap regime, not the 1% past
    no_dates, _, _ = fx.ewma_sigma_hat(close)      # without dates: legacy freeze-over
    assert np.isfinite(no_dates[g:g + 20]).all()


def test_tradable_and_vol_med63():
    """Causal candidate-gate flag: $5 close floor and $1M 63-session median dollar
    volume, min_periods=63, NaN -> False."""
    n = 200
    oh = _synthetic_ohlcv(n)
    oh["close"] = 10.0
    oh["volume"] = 200_000.0                       # $2M/day
    oh.loc[100:119, "close"] = 4.0                 # under the $5 floor
    fear = pd.DataFrame({"date": oh["date"], "fear_greed": 50.0})
    f = fx.build_feature_frame("SYN", oh, fear)
    assert not f.tradable[:62].any()               # warm-up rows are never tradable
    assert f.tradable[70]
    assert not f.tradable[110]                     # price floor bites
    assert f.tradable[150]
    assert np.isnan(f.vol_med63[:62]).all()
    assert f.vol_med63[70] == 200_000.0
    low = oh.copy()
    low["volume"] = 50_000.0                       # $0.5M/day: fails dollar volume
    f2 = fx.build_feature_frame("SYN2", low, fear)
    assert not f2.tradable.any()


def test_tool_tradable_pins_builder():
    """The scorer's tool-local tradability must be bit-identical to the builder's
    FeatureFrame field (the two implementations must agree bit-for-bit)."""
    from tools.score_checkpoints import compute_tradable

    oh = _synthetic_ohlcv(300, seed=3)
    fear = pd.DataFrame({"date": oh["date"], "fear_greed": 50.0})
    f = fx.build_feature_frame("SYN", oh, fear)
    tool = compute_tradable(oh["close"].to_numpy(), oh["volume"].to_numpy())
    assert np.array_equal(tool, f.tradable)


def test_census_phantom_filter():
    """Census-listed sub-3-name dates are dropped by load_us_ohlcv; dates newer than
    the census end pass unconditionally; a missing census disables filtering."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = tempfile.mkdtemp()
    census_path = os.path.join(tmp, "census.csv")
    pd.DataFrame({
        "date": ["2020-01-06", "2020-01-07", "2020-01-08"],
        "n_names": [500, 2, 1],
    }).to_csv(census_path, index=False)
    phantom = fx.load_session_census(census_path)
    assert phantom == {pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-08")}

    parts = os.path.join(tmp, "parts")
    os.makedirs(parts)
    rows = pd.DataFrame({
        "ticker": ["SYN"] * 4, "exchange": ["NYSE"] * 4,
        "date": ["2020-01-06", "2020-01-07", "2020-01-08", "2020-02-03"],
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    pq.write_table(pa.Table.from_pandas(rows), os.path.join(parts, "p0.parquet"))

    old_path = fx.CENSUS_PATH
    try:
        fx._census_cache.clear()
        fx.CENSUS_PATH = census_path
        out = fx.load_us_ohlcv(tickers=["SYN"], parts_dir=parts)
        got = [str(d.date()) for d in pd.to_datetime(out["SYN"]["date"])]
        # Phantom sessions dropped; the post-census date passes unconditionally.
        assert got == ["2020-01-06", "2020-02-03"], got

        fx._census_cache.clear()
        fx.CENSUS_PATH = os.path.join(tmp, "absent.csv")
        out2 = fx.load_us_ohlcv(tickers=["SYN"], parts_dir=parts)
        assert len(out2["SYN"]) == 4               # missing census: no filtering
    finally:
        fx.CENSUS_PATH = old_path
        fx._census_cache.clear()


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


def test_rolling_close_z_matches_direct_formula():
    """Per-window zscore equals the direct two-pass formula on every valid row and
    keeps the declared 19-row NaN warm-up."""
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(300)))
    z = fx._rolling_close_z(close)
    assert np.isnan(z[:19]).all() and np.isfinite(z[19:]).all()
    for i in (19, 57, 150, 299):
        win = close[i - 19: i + 1]
        want = (close[i] - win.mean()) / win.std(ddof=1)
        assert abs(z[i] - want) < 1e-10, (i, z[i], want)


def test_rolling_close_z_constant_window_zero():
    """Bit-constant windows are the structural 0/0 and emit exactly 0; windows
    straddling the level change stay finite and Samuelson-bounded."""
    close = np.concatenate([np.full(60, 4.938), [5.375, 5.438], np.full(41, 5.438)])
    z = fx._rolling_close_z(close)
    assert (z[19:60] == 0.0).all()
    assert (z[81:] == 0.0).all()
    fin = z[np.isfinite(z)]
    assert np.abs(fin).max() <= 19 / np.sqrt(20) + 1e-9


def test_rolling_close_z_bounded_fp16_safe_adversarial():
    """The Samuelson bound (w-1)/sqrt(w) holds on adversarial series -- huge price
    levels with ulp-scale jitter, near-flat feed jitter, violent walks, level
    steps -- so the fp16 cast can never overflow."""
    rng = np.random.default_rng(4)
    cases = [
        np.full(120, 631_000.10) + rng.choice([0.0, 1e-6], size=120),
        5.75 + rng.choice([0.0, 0.01, -0.01], size=200),
        100 * np.exp(np.cumsum(0.05 * rng.standard_normal(400))),
        np.concatenate([np.full(30, 2.0), np.full(30, 900_000.0)]),
    ]
    for k, close in enumerate(cases):
        z = fx._rolling_close_z(np.asarray(close, dtype=np.float64))
        fin = z[np.isfinite(z)]
        assert fin.size, k
        assert np.abs(fin).max() <= 19 / np.sqrt(20) + 1e-9, (k, np.abs(fin).max())
        assert np.isfinite(fin.astype(np.float16)).all(), k


def test_zscore_channel_fp16_safe_through_builder():
    """A flat-stretch ticker (the incident class) builds a zscore channel that
    survives the fp16 round-trip finite, with fully-inside-stretch windows at 0."""
    oh = _synthetic_ohlcv(n=300, seed=7)
    oh.loc[100:160, "close"] = 5.83
    fear = pd.DataFrame({"date": oh["date"], "fear_greed": 50.0})
    frame = fx.build_feature_frame("SYN", oh, fear)
    j = fx.FEATURE_NAMES.index("zscore")
    zs = frame.features[:, j]
    fin = zs[np.isfinite(zs)]
    assert np.isfinite(fin.astype(np.float16)).all()
    assert np.abs(fin).max() <= 19 / np.sqrt(20) + 1e-6
    assert (zs[119:161] == 0.0).all()


def test_transform_rejects_fp16_overflow_on_unscaled():
    """The fp16 finiteness guard refuses finite float32 magnitudes beyond the fp16
    range in unscaled channels; scaled channels are clip-protected and pass."""
    rows = np.zeros((8, fx.N_FEATURES), dtype=np.float32)
    scaler = fx.RobustScaler.fit(np.random.default_rng(0).standard_normal((64, fx.N_FEATURES)))
    scaler.transform(rows)
    bad = rows.copy()
    bad[3, fx.FEATURE_NAMES.index("zscore")] = 1e8
    try:
        scaler.transform(bad)
        assert False, "expected ValueError for fp16 overflow"
    except ValueError as e:
        assert "zscore" in str(e)
    clipped = rows.copy()
    clipped[2, fx.SCALED_IDX[0]] = 1e8
    scaler.transform(clipped)
