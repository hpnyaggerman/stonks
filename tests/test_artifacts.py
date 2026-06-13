"""Contract checks on the trained artifacts.

These validate the files the trainer writes and the columns the backtest and live
scorer depend on. They no-op when no model has been trained yet, so the suite still
runs on a fresh checkout.
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import features_v5 as fx
from v5_backbone import V5Backbone, V5Config, assert_theta_on_bin_edges

MODELS = os.path.join(ROOT, "models")
V5DIR = os.path.join(MODELS, "v5")
FORECASTS = os.path.join(ROOT, "forecasts")

# What run_backtest_v4.py reads from each forecast CSV.
REQUIRED_FORECAST_COLS = (["Date", "Close"]
                          + [f"Pred_Prob_{h}" for h in ("1d", "1w", "1m", "6m")]
                          + [f"Pred_Prob_Std_{h}" for h in ("1d", "1w", "1m", "6m")])


def _trained():
    return os.path.exists(os.path.join(V5DIR, "config.json"))


def test_meta_contract():
    if not _trained():
        return
    meta = json.load(open(os.path.join(MODELS, "v5_meta.json")))
    for key in ("feature_spec", "feature_names_hash", "window", "min_real_rows", "horizons",
                "n_bins", "bin_width", "theta_bins", "temperatures", "n_members",
                "spike_log_threshold", "ewma_sigma_floor", "seam_c_feed_spec"):
        assert key in meta, f"missing meta key {key}"
    assert meta["feature_names_hash"] == fx.feature_names_hash()
    assert len(meta["feature_spec"]) == fx.N_FEATURES
    assert meta["min_real_rows"] == fx.MIN_REAL_ROWS
    assert len(meta["temperatures"]) == len(meta["horizon_days"])


def test_norm_contract():
    if not _trained():
        return
    scaler = fx.RobustScaler.load(os.path.join(MODELS, "v5_norm.json"))
    assert len(scaler.scales) == fx.N_FEATURES
    assert (scaler.scales > 0).all()


def test_config_and_members_load():
    if not _trained():
        return
    cfg = V5Config(**json.load(open(os.path.join(V5DIR, "config.json"))))
    assert_theta_on_bin_edges(cfg)
    meta = json.load(open(os.path.join(MODELS, "v5_meta.json")))
    model = V5Backbone(cfg)
    model.load_state_dict(torch.load(os.path.join(V5DIR, "member_0.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, cfg.window, cfg.n_features))
    assert out.shape == (1, len(cfg.horizons), cfg.n_bins)
    assert int(meta["n_members"]) >= 1


def test_split_info_contract():
    if not _trained():
        return
    info = json.load(open(os.path.join(FORECASTS, "split_info.json")))
    assert info["eval_mode"] in ("time", "ticker", "both")
    assert "embargo" in info
    if info["eval_mode"] in ("time", "both"):
        assert "oos_start" in info


def test_forecast_columns():
    if not _trained():
        return
    files = [f for f in os.listdir(FORECASTS) if f.endswith("_forecast.csv")]
    if not files:
        return
    import pandas as pd
    header = pd.read_csv(os.path.join(FORECASTS, files[0]), nrows=0).columns.tolist()
    for col in REQUIRED_FORECAST_COLS:
        assert col in header, f"forecast CSV missing backtest-required column {col}"
    # The v5 additive columns the strategy can later opt into.
    for h in ("1d", "1w", "1m", "6m"):
        for prefix in ("Pred_Prob_Down_", "Pred_Prob_Neutral_", "Score_", "Score_Std_",
                       "Pred_Q10_", "Pred_Q50_", "Pred_Q90_"):
            assert prefix + h in header, f"missing {prefix + h}"
