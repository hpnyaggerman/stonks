"""v5 backtest tests: split detector, booking, forced exits, costs, nulls."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_backtest_v5 as bt


def _dates(n, start="2025-06-02"):
    return pd.bdate_range(start, periods=n)


def _series(n, base=100.0):
    return np.full(n, base, dtype=np.float64)


def test_forward_split_detected_and_adjusted():
    """2:1 forward split with heavy volume: detected, back-adjusted, no fake -69%
    loss (the v4 failure) in the adjusted series."""
    n = 20
    close = _series(n)
    close[10:] = 50.0                    # 2:1 forward split at index 10
    volume = np.full(n, 100_000.0)
    volume[10] = 400_000.0               # split day trades heavy: >= (rho/2)*med
    med = np.full(n, 100_000.0)
    adjust, events = bt.detect_splits(_dates(n), close, volume, med)
    assert len(events) == 1 and events[0]["kind"] == "forward" and events[0]["rho"] == 2.0
    adj = close * adjust
    r = np.diff(np.log(adj))
    assert np.abs(r).max() < 1e-9        # the fake -69% log-return is gone


def test_reverse_split_adjusted_not_deleted():
    n = 20
    close = _series(n, 2.0)
    close[12:] = 10.0                    # 1:5 reverse split
    volume = np.full(n, 500_000.0)
    volume[12] = 100_000.0               # consolidations shrink volume: <= (2/rho)*med
    med = np.full(n, 500_000.0)
    adjust, events = bt.detect_splits(_dates(n), close, volume, med)
    assert len(events) == 1 and events[0]["kind"] == "reverse" and events[0]["rho"] == 5.0
    adj = close * adjust
    assert np.abs(np.diff(np.log(adj))).max() < 1e-9


def test_genuine_crash_on_flat_volume_not_adjusted():
    """A -50% crash that snaps to rho=2 but fails the volume corroboration (flat
    volume, well under (rho/2)*median) stays in P&L unadjusted."""
    n = 20
    close = _series(n)
    close[10:] = 50.0
    volume = np.full(n, 90_000.0)        # flat; 90k < (2/2)*100k = rho/2 * med
    med = np.full(n, 100_000.0)
    adjust, events = bt.detect_splits(_dates(n), close, volume, med)
    assert not events
    assert np.allclose(adjust, 1.0)


def test_multiweek_gap_move_never_a_candidate():
    n = 12
    dates = list(pd.bdate_range("2025-06-02", periods=6))
    dates += list(pd.bdate_range(dates[-1] + pd.Timedelta(days=30), periods=6))
    close = _series(n)
    close[6:] = 50.0                     # -50% across a month-long gap
    volume = np.full(n, 1e6)
    med = np.full(n, 100_000.0)
    adjust, events = bt.detect_splits(np.asarray(dates, dtype="datetime64[ns]"),
                                      close, volume, med)
    assert not events                    # spans > 7 calendar days are legitimate


def _write_forecast(dirpath, ticker, dates, close, score_1d=0.5, tradable=1,
                    volume=1e6, med=1e6, scores=None):
    n = len(dates)
    cols = {"Date": dates, "Close": close,
            "Volume": np.full(n, volume), "Tradable": np.full(n, tradable, dtype=int),
            "VolMed63": np.full(n, med)}
    scores = scores or {}
    for h in ("1d", "1w", "1m", "6m"):
        s = scores.get(h, score_1d if h == "1d" else -1.0)
        cols[f"Score_{h}"] = np.full(n, s)
    pd.DataFrame(cols).to_csv(os.path.join(dirpath, f"{ticker}_forecast.csv"), index=False)


def _mk_dir(tmp, tickers_spec, oos_start="2025-06-02"):
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "split_info.json"), "w") as f:
        f.write('{"oos_start": "%s"}' % oos_start)
    for spec in tickers_spec:
        _write_forecast(tmp, **spec)


def test_entry_is_next_close_and_booking():
    import tempfile
    tmp = tempfile.mkdtemp()
    n = 10
    dates = _dates(n)
    close = np.linspace(100, 109, n)     # +1/day
    _mk_dir(tmp, [dict(ticker="AAA", dates=dates, close=close)])
    frames = bt.load_forecasts(tmp, "2025-06-02")
    floors = {h: 0.0 for h in bt.HORIZON_LABELS}
    by_date, dindex = bt.build_candidates(frames, floors, ["1d"])
    union = sorted({pd.Timestamp(d) for d in dates})
    curve, counters, trades = bt.run_strategy(frames, by_date, dindex, union, 0, 0.0,
                                              pick="argmax", collect_trades=True)
    assert counters["trades"] >= 2
    t0 = trades[0]
    sig = pd.Timestamp(t0["SignalDate"])
    ent = pd.Timestamp(t0["EntryDate"])
    ext = pd.Timestamp(t0["ExitDate"])
    assert ent > sig                     # t+1 fill, never the signal close
    assert (dindex["AAA"][ent] - dindex["AAA"][sig]) == 1
    assert (dindex["AAA"][ext] - dindex["AAA"][ent]) == 1     # 1d horizon hold
    assert np.isclose(t0["LogR"], np.log(close[2] / close[1]))
    # exit day admits no new candidate: the next trade's signal is at least one
    # union session after the previous exit
    t1 = trades[1]
    assert pd.Timestamp(t1["SignalDate"]) > ext


def test_midhold_delisting_forced_exit_and_haircut():
    import tempfile
    tmp = tempfile.mkdtemp()
    long_dates = _dates(30)
    dead_dates = long_dates[:6]          # dies 24 sessions before the window end
    _mk_dir(tmp, [
        dict(ticker="DEAD", dates=dead_dates, close=np.full(6, 100.0),
             scores={"1w": 0.9}),
        dict(ticker="LIVE", dates=long_dates, close=np.full(30, 50.0), score_1d=-1.0,
             tradable=0),
    ])
    frames = bt.load_forecasts(tmp, "2025-06-02")
    floors = {h: 0.0 for h in bt.HORIZON_LABELS}
    by_date, dindex = bt.build_candidates(frames, floors, ["1w"])
    union = sorted({pd.Timestamp(d) for o in frames.values() for d in o["df"]["Date"]})
    curve, counters, trades = bt.run_strategy(frames, by_date, dindex, union, 0, 0.3,
                                              pick="argmax", collect_trades=True)
    # 1w hold entered near the head cannot complete inside 6 prints: forced exit.
    assert counters["forced_exits"] >= 1
    assert counters["haircuts_applied"] >= 1
    forced = [t for t in trades if t["ForcedExit"]]
    assert forced and forced[0]["HaircutApplied"] == 1
    assert curve[-1] < 1.0               # the haircut booked a loss on a flat price


def test_cost_grid_arithmetic():
    import tempfile
    tmp = tempfile.mkdtemp()
    n = 10
    dates = _dates(n)
    close = np.full(n, 100.0)            # flat: returns come only from costs
    _mk_dir(tmp, [dict(ticker="AAA", dates=dates, close=close)])
    frames = bt.load_forecasts(tmp, "2025-06-02")
    floors = {h: 0.0 for h in bt.HORIZON_LABELS}
    by_date, dindex = bt.build_candidates(frames, floors, ["1d"])
    union = sorted({pd.Timestamp(d) for d in dates})
    c0, k0, _ = bt.run_strategy(frames, by_date, dindex, union, 0, 0.0)
    c25, k25, _ = bt.run_strategy(frames, by_date, dindex, union, 25, 0.0)
    assert k0["trades"] == k25["trades"] > 0
    assert np.isclose(c0[-1], 1.0)
    expected = (1 - 25e-4) ** k25["trades"]
    assert np.isclose(c25[-1], expected), (c25[-1], expected)


def test_null_respects_gates_and_blackout():
    import tempfile
    tmp = tempfile.mkdtemp()
    n = 14
    dates = _dates(n)
    _mk_dir(tmp, [
        dict(ticker="GOOD", dates=dates, close=np.full(n, 100.0), score_1d=0.5),
        dict(ticker="GATED", dates=dates, close=np.full(n, 100.0), score_1d=0.9,
             tradable=0),                # untradable: must never appear
        dict(ticker="LOWSC", dates=dates, close=np.full(n, 100.0), score_1d=-0.5),
    ])
    frames = bt.load_forecasts(tmp, "2025-06-02")
    floors = {h: 0.0 for h in bt.HORIZON_LABELS}
    by_date, dindex = bt.build_candidates(frames, floors, ["1d"])
    for cands in by_date.values():
        names = {c[0] for c in cands}
        assert "GATED" not in names and "LOWSC" not in names
    union = sorted({pd.Timestamp(d) for d in dates})
    rng = np.random.default_rng(0)
    curve, counters, trades = bt.run_strategy(frames, by_date, dindex, union, 0, 0.0,
                                              pick="uniform", rng=rng,
                                              collect_trades=True)
    assert counters["trades"] >= 2
    assert all(t["Ticker"] == "GOOD" for t in trades)
    # blackout: consecutive trades never share or abut on the exit session
    for a, b in zip(trades, trades[1:]):
        assert pd.Timestamp(b["SignalDate"]) > pd.Timestamp(a["ExitDate"])
