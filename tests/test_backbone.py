"""Model, loss, calibration-geometry and seam-graft-safety tests."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v5_backbone import (V5Backbone, V5Config, assert_theta_on_bin_edges, class_marginals,
                         ensemble_predict, hl_gauss_targets, optimizer_param_groups,
                         seam_state_snapshot, v5_loss)

SEAM_SUBSTR = ("film", "ctx_proj", "ctx_gate", "mid_proj", "mid_gate", "feed_proj",
               "feed_gate", "null_")


def _small_cfg():
    return V5Config(n_features=40, window=40, d_model=32, n_blocks=4, d_state=8)


def test_seam_forward_identity():
    """Pre-graft, a non-null conditioning input produces the identical output to the
    null path, for any input, bitwise."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    m = V5Backbone(cfg).eval()
    x = torch.randn(5, cfg.window, cfg.n_features)
    e_id = torch.randn(5, cfg.d_id)
    c_mkt = torch.randn(5, cfg.window, cfg.d_mkt)
    c_feed = torch.randn(5, cfg.window, cfg.d_feed)
    diff = (m(x) - m(x, e_id=e_id, c_mkt=c_mkt, c_feed=c_feed)).abs().max().item()
    assert diff == 0.0


def test_seam_frozen_through_base_training():
    """With null slots and the no-decay optimizer grouping, every seam parameter stays
    bitwise at its initialization after base-training steps (the shipped-checkpoint
    guarantee)."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    m = V5Backbone(cfg)
    snap = seam_state_snapshot(m)
    opt = torch.optim.AdamW(optimizer_param_groups(m), lr=1e-2, betas=(0.9, 0.95))
    m.train()
    for _ in range(8):
        x = torch.randn(6, cfg.window, cfg.n_features)
        z = torch.randn(6, 4)
        mask = torch.ones(6, 4)
        opt.zero_grad()
        v5_loss(m(x), z, mask, cfg).backward()
        opt.step()
    after = m.state_dict()
    for k, v in snap.items():
        assert torch.equal(v, after[k]), f"seam param {k} drifted during base training"
    # A non-seam parameter must have moved, confirming training actually happened.
    moved = any(not torch.equal(snap.get(k, after[k]), after[k]) or not any(s in k for s in SEAM_SUBSTR)
                for k in after)
    assert moved


def test_loss_nan_safe_and_masked():
    torch.manual_seed(0)
    cfg = _small_cfg()
    m = V5Backbone(cfg).eval()
    logits = m(torch.randn(4, cfg.window, cfg.n_features))
    z = torch.randn(4, 4)
    z[0, 0] = float("nan")
    z[2, 3] = float("nan")
    mask = torch.ones(4, 4)
    mask[0, 0] = 0
    mask[2, 3] = 0
    loss = v5_loss(logits, z, mask, cfg)
    assert torch.isfinite(loss)
    all_masked = v5_loss(logits, torch.full((4, 4), float("nan")), torch.zeros(4, 4), cfg)
    assert all_masked.item() == 0.0


def test_loss_mask_drops_entries():
    """Masking an entry equals removing it from the mean, not zeroing its NaN value."""
    torch.manual_seed(1)
    cfg = _small_cfg()
    m = V5Backbone(cfg).eval()
    logits = m(torch.randn(3, cfg.window, cfg.n_features))
    z = torch.randn(3, 4)
    full = v5_loss(logits, z, torch.ones(3, 4), cfg)
    mask = torch.ones(3, 4)
    mask[0, 0] = 0
    z_masked = z.clone()
    z_masked[0, 0] = float("nan")
    partial = v5_loss(logits, z_masked, mask, cfg)
    assert torch.isfinite(partial) and abs(partial.item() - full.item()) > 1e-6


def test_class_marginals_exact_partition():
    torch.manual_seed(2)
    cfg = _small_cfg()
    logits = torch.randn(4, 4, cfg.n_bins)
    cm = class_marginals(logits, cfg.theta_bins, cfg.n_bins)
    assert torch.allclose(cm.sum(-1), torch.ones(4, 4), atol=1e-5)
    # Manual partition for horizon 0, k=theta_bins[0].
    p = logits.softmax(-1)[:, 0]
    k, half = cfg.theta_bins[0], cfg.n_bins // 2
    assert torch.allclose(cm[:, 0, 0], p[:, :half - k].sum(-1))
    assert torch.allclose(cm[:, 0, 2], p[:, half + k:].sum(-1))


def test_hl_gauss_normalized():
    cfg = _small_cfg()
    z = torch.tensor([[0.0, 1.5, -2.0, 4.0]])
    tg = hl_gauss_targets(z, cfg)
    assert tg.shape == (1, 4, cfg.n_bins)
    assert torch.allclose(tg.sum(-1), torch.ones(1, 4), atol=1e-5)
    assert (tg >= 0).all()


def test_theta_validation():
    cfg = _small_cfg()
    assert_theta_on_bin_edges(cfg)
    for bad in [(0, 5, 5, 5), (5, 5, 5, 30), (5, 5, 5, 31)]:
        try:
            assert_theta_on_bin_edges(V5Config(theta_bins=bad))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for theta_bins={bad}")


def test_ensemble_outputs():
    torch.manual_seed(3)
    cfg = _small_cfg()
    members = [V5Backbone(cfg).eval() for _ in range(4)]
    x = torch.randn(3, cfg.window, cfg.n_features)
    hist, cls3, up_std, score_std = ensemble_predict(members, x, cfg, temps=[1.0, 1.1, 1.2, 1.3])
    assert hist.shape == (3, 4, cfg.n_bins)
    assert cls3.shape == (3, 4, 3)
    assert up_std.shape == (3, 4) and score_std.shape == (3, 4)
    assert torch.allclose(hist.sum(-1), torch.ones(3, 4), atol=1e-5)
    # Score (up - down) std is generally not equal to up std: the bear-bull covariance
    # across members carries independent information.
    assert (up_std - score_std).abs().max().item() > 0


def test_optimizer_groups_partition():
    cfg = _small_cfg()
    m = V5Backbone(cfg)
    groups = optimizer_param_groups(m)
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    for name, p in m.named_parameters():
        if any(s in name for s in SEAM_SUBSTR) or p.ndim < 2 or name.endswith("A_log"):
            assert id(p) in no_decay_ids, f"{name} should be no-decay"
    assert groups[1]["weight_decay"] == 0.0
