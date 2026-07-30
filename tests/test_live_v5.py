"""Live-scorer tests: floors from meta, staleness hard-fail, Score-vs-P(up)
ranking, untradable exclusion from the buy decision."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_live_signals_v5 as live


def _args(**overrides):
    ns = argparse.Namespace(min_score_1d=None, min_score_1w=None, min_score_1m=None,
                            min_score_6m=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _cols(scores, p_up=None, tradable=1):
    """Synthetic one-row column dict in build_forecast_columns shape."""
    cols = {"Date": pd.Timestamp("2026-06-05"), "Close": 100.0, "Tradable": tradable,
            "Volume": 1e6, "VolMed63": 1e6}
    p_up = p_up or {}
    for h in ("1d", "1w", "1m", "6m"):
        s = scores.get(h, -1.0)
        up = p_up.get(h, 0.3)
        cols[f"Score_{h}"] = s
        cols[f"Score_Std_{h}"] = 0.01
        cols[f"Pred_Prob_{h}"] = up
        cols[f"Pred_Prob_Std_{h}"] = 0.01
        cols[f"Pred_Prob_Down_{h}"] = up - s
        cols[f"Pred_Prob_Neutral_{h}"] = 1.0 - up - (up - s)
        cols[f"Pred_Q10_{h}"] = -0.05
        cols[f"Pred_Q50_{h}"] = 0.0
        cols[f"Pred_Q90_{h}"] = 0.05
    return cols


def test_floors_read_from_meta_with_cli_override():
    meta = {"score_floors": {"1d": 0.02, "1w": 0.01, "1m": -0.005, "6m": 0.003}}
    floors = live.resolve_score_floors(meta, _args())
    assert floors == {"1d": 0.02, "1w": 0.01, "1m": -0.005, "6m": 0.003}
    floors = live.resolve_score_floors(meta, _args(min_score_1w=0.5))
    assert floors["1w"] == 0.5 and floors["1d"] == 0.02
    assert live.resolve_score_floors({}, _args()) == {h: 0.0 for h in ("1d", "1w", "1m", "6m")}


def test_staleness_hard_fail():
    ok = live.check_feed_staleness(pd.Timestamp("2026-06-05"),
                                   pd.Timestamp("2026-06-10"), staleness_k=63)
    assert ok >= 0
    try:
        live.check_feed_staleness(pd.Timestamp("2026-01-05"),
                                  pd.Timestamp("2026-06-10"), staleness_k=63)
    except SystemExit:
        pass
    else:
        raise AssertionError("a >63-session-stale feed must hard-fail")
    try:
        live.check_feed_staleness(None, pd.Timestamp("2026-06-10"), staleness_k=63)
    except SystemExit:
        pass
    else:
        raise AssertionError("an empty feed must hard-fail")


def test_score_ranking_differs_from_p_up_ranking():
    """1w has the higher P(up) but 1d has the higher Score (P(up) - P(down));
    the decision must follow Score."""
    cols = _cols(scores={"1d": 0.30, "1w": 0.05}, p_up={"1d": 0.35, "1w": 0.60})
    floors = {h: 0.0 for h in ("1d", "1w", "1m", "6m")}
    is_buy, chosen, tradable = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m"], floors)
    assert is_buy and chosen["horizon"] == "1d"
    assert cols["Pred_Prob_1w"] > cols["Pred_Prob_1d"]     # the old rule would pick 1w


def test_untradable_reported_never_buys():
    cols = _cols(scores={"1d": 0.9}, tradable=0)
    floors = {h: 0.0 for h in ("1d", "1w", "1m", "6m")}
    is_buy, chosen, tradable = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m"], floors)
    assert not is_buy and tradable == 0
    assert chosen is not None and chosen["horizon"] == "1d"   # still reported
    cols_t = _cols(scores={"1d": 0.9}, tradable=1)
    is_buy2, _, _ = live.decide_ticker(cols_t, 0.0, ["1d", "1w", "1m"], floors)
    assert is_buy2                                            # tradability was the gate


def test_floor_gates_candidacy_per_horizon():
    cols = _cols(scores={"1d": 0.10, "1w": 0.30})
    floors = {"1d": 0.05, "1w": 0.50, "1m": 0.0, "6m": 0.0}
    is_buy, chosen, _ = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m"], floors)
    # 1w has the higher score but fails its floor; 1d clears its own.
    assert is_buy and chosen["horizon"] == "1d"
    floors_all_high = {h: 0.95 for h in ("1d", "1w", "1m", "6m")}
    is_buy2, chosen2, _ = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m"], floors_all_high)
    assert not is_buy2 and chosen2["horizon"] == "1w"         # argmax still reported


def test_6m_excluded_from_argmax_by_default():
    cols = _cols(scores={"6m": 0.9, "1d": 0.1})
    floors = {h: 0.0 for h in ("1d", "1w", "1m", "6m")}
    is_buy, chosen, _ = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m"], floors)
    assert chosen["horizon"] == "1d"                          # 6m cannot win
    is_buy2, chosen2, _ = live.decide_ticker(cols, 0.0, ["1d", "1w", "1m", "6m"], floors)
    assert chosen2["horizon"] == "6m"                         # --include-6m opt-in
