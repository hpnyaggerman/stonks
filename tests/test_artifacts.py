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
V5DIR = os.path.join(MODELS, "v5")           # the default --run-dir
FORECASTS = os.path.join(ROOT, "forecasts")

# What run_backtest_v4.py reads from each forecast CSV.
REQUIRED_FORECAST_COLS = (["Date", "Close"]
                          + [f"Pred_Prob_{h}" for h in ("1d", "1w", "1m", "6m")]
                          + [f"Pred_Prob_Std_{h}" for h in ("1d", "1w", "1m", "6m")])
# Cross-stage transport columns the v5 backtest's split detector and candidate gate
# consume (not computable from the OOS-only CSVs).
V5_TRANSPORT_COLS = ("Volume", "Tradable", "VolMed63")


def _meta_path():
    # v5 artifacts live inside the run dir; the pre-run-dir layout kept them in
    # models/. Accept both so the suite runs against either generation.
    for p in (os.path.join(V5DIR, "v5_meta.json"), os.path.join(MODELS, "v5_meta.json")):
        if os.path.exists(p):
            return p
    return None


def _norm_path():
    for p in (os.path.join(V5DIR, "v5_norm.json"), os.path.join(MODELS, "v5_norm.json")):
        if os.path.exists(p):
            return p
    return None


def _trained():
    return os.path.exists(os.path.join(V5DIR, "config.json")) and _meta_path()


def test_meta_contract():
    if not _trained():
        return
    meta = json.load(open(_meta_path()))
    for key in ("feature_spec", "feature_names_hash", "window", "min_real_rows", "horizons",
                "n_bins", "bin_width", "theta_bins", "temperatures", "n_members",
                "spike_log_threshold", "ewma_sigma_floor", "seam_c_feed_spec"):
        assert key in meta, f"missing meta key {key}"
    assert meta["feature_names_hash"] == fx.feature_names_hash()
    assert len(meta["feature_spec"]) == fx.N_FEATURES
    assert meta["min_real_rows"] == fx.MIN_REAL_ROWS
    assert len(meta["temperatures"]) == len(meta["horizon_days"])
    # Keys only the run-dir generation of the trainer writes.
    if os.path.dirname(_meta_path()) == V5DIR:
        for key in ("score_floors", "score_floors_unfiltered", "label_centering",
                    "lam_cls", "staleness_k"):
            assert key in meta, f"missing v5 meta key {key}"
        assert meta["temperatures"][2] == 1.0 and meta["temperatures"][3] == 1.0


def test_run_manifest_contract():
    path = os.path.join(V5DIR, "run_manifest.json")
    if not os.path.exists(path):
        return
    manifest = json.load(open(path))
    for key in ("args", "git_revision", "data_fingerprint", "feature_names_hash",
                "scaler_rows"):
        assert key in manifest, f"missing manifest key {key}"
    assert "files" in manifest["data_fingerprint"]
    for k in ("eval_mode", "seed", "eval_days", "window", "val_subsample", "members"):
        assert k in manifest["args"], f"missing manifest arg {k}"


def test_norm_contract():
    if not _trained():
        return
    scaler = fx.RobustScaler.load(_norm_path())
    assert len(scaler.scales) == fx.N_FEATURES
    assert (scaler.scales > 0).all()


def test_config_and_members_load():
    if not _trained():
        return
    cfg = V5Config(**json.load(open(os.path.join(V5DIR, "config.json"))))
    assert_theta_on_bin_edges(cfg)
    meta = json.load(open(_meta_path()))
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
    if not os.path.isdir(FORECASTS):
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
    for col in V5_TRANSPORT_COLS:
        assert col in header, f"forecast CSV missing v5 transport column {col}"


def test_forecast_column_builder_transport():
    """build_forecast_columns emits the transport columns when given the arrays and
    omits them when not (v4 compatibility)."""
    import numpy as np

    from v5.forecast import build_forecast_columns

    rows, H, B = 3, 4, 60
    kw = dict(dates=np.arange(rows), close=np.ones(rows), sigma_hat=np.full(rows, 0.02),
              horizon_days=(1, 5, 21, 126), up=np.random.rand(rows, H),
              up_std=np.random.rand(rows, H), down=np.random.rand(rows, H),
              neutral=np.random.rand(rows, H), score=np.random.rand(rows, H),
              score_std=np.random.rand(rows, H), hist=np.random.rand(rows, H, B),
              bin_width=0.1348, n_bins=B)
    plain = build_forecast_columns(**kw)
    assert not any(c in plain for c in V5_TRANSPORT_COLS)
    full = build_forecast_columns(**kw, volume=np.ones(rows),
                                  tradable=np.asarray([True, False, True]),
                                  vol_med63=np.full(rows, 5.0))
    assert list(full["Tradable"]) == [1, 0, 1]
    assert "Volume" in full and "VolMed63" in full


def test_promote_run_copies_stamps_and_refuses():
    """Promotion copies the artifact set, member checkpoints, and forecast files
    into the served dirs, stamps promoted_from.json, and refuses unfinished runs."""
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import promote_run as pr

    base = tempfile.mkdtemp()
    run_dir = os.path.join(base, "runs", "r1")
    os.makedirs(run_dir)
    for name, payload in (("v5_meta.json", {"members": 1}),
                          ("v5_norm.json", {"channels": []}),
                          ("config.json", {"d_model": 8}),
                          ("run_manifest.json", {"args": {"members": 1},
                                                 "git_revision": "abc",
                                                 "run_nonce": "n-1"})):
        with open(os.path.join(run_dir, name), "w") as f:
            json.dump(payload, f)
    torch.save({"w": torch.zeros(2)}, os.path.join(run_dir, "member_0.pt"))
    fc_root = os.path.join(base, "forecasts")
    os.makedirs(os.path.join(fc_root, "r1"))
    for name in ("AAA_forecast.csv", "split_info.json"):
        with open(os.path.join(fc_root, "r1", name), "w") as f:
            f.write("x")
    model_dir = os.path.join(base, "served")

    stamp = pr.promote(run_dir, model_dir=model_dir, forecast_dir=fc_root)
    for name in ("v5_meta.json", "v5_norm.json", "config.json", "run_manifest.json",
                 "member_0.pt", "promoted_from.json"):
        assert os.path.exists(os.path.join(model_dir, name)), name
    assert os.path.exists(os.path.join(fc_root, "AAA_forecast.csv"))
    assert os.path.exists(os.path.join(fc_root, "split_info.json"))
    assert stamp["run_dir"].endswith("r1") and stamp["run_nonce"] == "n-1"
    assert stamp["members"] == 1 and stamp["forecast_files"] == 2

    unfinished = os.path.join(base, "runs", "r2")
    os.makedirs(unfinished)
    try:
        pr.promote(unfinished, model_dir=model_dir, forecast_dir=fc_root)
        assert False, "expected SystemExit for unfinished run"
    except SystemExit:
        pass
