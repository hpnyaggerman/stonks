"""Selective state-space backbone with a per-horizon distributional head.

The network maps a window of ``window`` daily feature rows to, for each of four
forecast horizons, a 60-bin histogram over the volatility-normalized forward
log-return. Three-class {down, neutral, up} probabilities and return quantiles are
exact functionals of that histogram, so a single trained head serves any decision
rule without retraining.

Temporal mixing is performed exclusively by Mamba blocks; every other operator
(the gated MLP, the conditioning seams, the read-out, the heads) acts per timestep,
so the model is causal end to end by construction.

Three conditioning seams are present from the first training run but contribute
exactly zero until they are deliberately trained ("grafted"):

* **A — static identity**: a per-ticker vector modulating each block via
  AdaLN-Zero feature-wise modulation.
* **B — cross-market context**: a per-day vector added through zero-gated taps at
  the stem and mid-stack.
* **C — per-ticker feed**: per-day external-feed channels added through a zero-gated
  stem tap, mechanically identical to seam B.

Each seam's contribution is the zero tensor at initialization (zero-initialized
FiLM weights; zero-initialized additive-tap gates), and every seam parameter
receives zero gradient while its input is the learned null vector, so base training
leaves them bitwise at their initial values and the pre-graft forward function is
identical to a seam-free network. :func:`optimizer_param_groups` keeps them out of
weight decay so that identity is not eroded by the optimizer. These properties are
asserted by the seam-identity test.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class V5Config:
    """Architecture and head geometry.

    ``min_real_rows`` is the count of genuine (non-padded) rows a window must contain
    to be trainable or scoreable; it equals ``max(warm_up + lag)`` over the feature
    channels and must be recomputed when any channel's warm-up or lag changes. Windows
    with fewer than ``window`` real rows are left-padded; windows below
    ``min_real_rows`` are rejected.
    """

    n_features: int = 40            # 31 market/feed + 8 calendar + is_pad
    window: int = 252               # rows ending at the prediction date inclusive
    min_real_rows: int = 126
    d_model: int = 256
    n_blocks: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mlp_mult: int = 2
    dropout: float = 0.1            # exactly one application per residual branch
    d_id: int = 64                  # seam A dim (null until graft)
    d_mkt: int = 64                 # seam B dim (null until graft)
    d_feed: int = 6                 # seam C dim (null until graft)
    p_cond: float = 0.25            # conditioning-dropout prob during graft training
    lam_cls: float = 0.1            # 3-class auxiliary loss weight (single source)
    horizons: tuple = (1, 5, 21, 126)
    n_bins: int = 60
    bin_width: float = 0.1348       # histogram bin width on the z-score axis
    z_clip: float = 4.044           # (n_bins / 2) * bin_width
    theta_bins: tuple = (5, 5, 5, 5)  # per-horizon class threshold in bin units


def assert_theta_on_bin_edges(cfg: "V5Config"):
    """Verify every class threshold lands exactly on a bin edge inside the range.

    Class marginals are exact partial bin sums only when each ``theta_bins[h]`` is an
    integer bin count strictly inside ``(0, n_bins/2)``; loaders call this so a stale
    or hand-edited threshold cannot silently produce interpolation error.
    """
    half = cfg.n_bins // 2
    for h, k in enumerate(cfg.theta_bins):
        if not (isinstance(k, int) and 0 < k < half):
            raise ValueError(
                f"theta_bins[{h}]={k} must be an integer in (0, {half}); "
                "class boundaries must fall on bin edges within the range")


class FiLM(nn.Module):
    """Static-identity conditioning (seam A) in AdaLN-Zero form.

    One bias-free, zero-initialized projection per block maps the identity vector to
    a per-branch ``(gamma, beta, alpha)`` triple for both residual branches: an input
    modulation ``(1 + gamma) * h_norm + beta`` and an output gain ``(1 + alpha)``.

    Zero-init plus no bias makes ``gamma = beta = alpha = 0`` for any input until the
    seam is grafted, so both branches pass through unchanged. The single projection is
    not a zero-times-zero stationary point: at graft time its input is a non-zero
    identity vector, so its weight gradient ``upstream ⊗ e_id`` is non-zero and the
    seam becomes trainable. During base training the input is the zero null vector, so
    the gradient is zero and the weights stay at their initialized values.
    """

    def __init__(self, d_id, d_model):
        super().__init__()
        self.proj = nn.Linear(d_id, 6 * d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, e_id):                       # (B, d_id) -> six (B, 1, d_model)
        return [m.unsqueeze(1) for m in self.proj(e_id).chunk(6, -1)]


class SwiGLU(nn.Module):
    """Gated MLP. Dropout is applied by the enclosing block, not here, so each
    residual branch sees exactly one dropout application."""

    def __init__(self, d, hidden):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * hidden)
        self.w3 = nn.Linear(hidden, d)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class V5Block(nn.Module):
    """Pre-norm residual block: a Mamba temporal mixer followed by a per-step gated
    MLP, both AdaLN-Zero-conditioned on the identity vector. One dropout application
    per branch yields an effective rate of ``cfg.dropout``."""

    def __init__(self, cfg: "V5Config", mamba_cls):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.mamba = mamba_cls(d_model=cfg.d_model, d_state=cfg.d_state,
                               d_conv=cfg.d_conv, expand=cfg.expand)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_mult * cfg.d_model)
        self.film = FiLM(cfg.d_id, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, e_id):                    # (B, L, d_model) -> (B, L, d_model)
        g_m, b_m, a_m, g_s, b_s, a_s = self.film(e_id)
        h = h + (1 + a_m) * self.drop(self.mamba(self.n1(h) * (1 + g_m) + b_m))
        h = h + (1 + a_s) * self.drop(self.mlp(self.n2(h) * (1 + g_s) + b_s))
        return h


class V5Backbone(nn.Module):
    """Stem -> conditioning taps -> Mamba blocks -> multi-scale read-out -> heads.

    The forward signature accepts whole-batch conditioning slots; ``None`` selects a
    seam's learned null input. Per-row degradation (some windows have an identity
    vector, others do not) is the data loader's responsibility — it substitutes the
    ``null_*`` parameters into the rows that lack a real input.
    """

    def __init__(self, cfg: V5Config, mamba_cls=None):
        super().__init__()
        if mamba_cls is None:
            try:
                from mamba_ssm import Mamba as mamba_cls
            except ImportError:
                from v5.mamba_ref import MambaRef as mamba_cls
        self.cfg = cfg
        self.stem = nn.Linear(cfg.n_features, cfg.d_model)
        # Seam B: per-day cross-market context. Bias-free projections behind zero gates.
        self.ctx_proj = nn.Linear(cfg.d_mkt, cfg.d_model, bias=False)
        self.ctx_gate = nn.Parameter(torch.zeros(cfg.d_model))
        self.mid_proj = nn.Linear(cfg.d_mkt, cfg.d_model, bias=False)
        self.mid_gate = nn.Parameter(torch.zeros(cfg.d_model))
        # Seam C: per-ticker per-day feed. Same zero-gated additive tap as seam B.
        self.feed_proj = nn.Linear(cfg.d_feed, cfg.d_model, bias=False)
        self.feed_gate = nn.Parameter(torch.zeros(cfg.d_model))
        # Learned null inputs: the degraded / cold-start path equals the trained path.
        self.null_id = nn.Parameter(torch.zeros(cfg.d_id))
        self.null_ctx = nn.Parameter(torch.zeros(cfg.d_mkt))
        self.null_feed = nn.Parameter(torch.zeros(cfg.d_feed))
        self.blocks = nn.ModuleList(V5Block(cfg, mamba_cls) for _ in range(cfg.n_blocks))
        self.out_norm = nn.RMSNorm(cfg.d_model)
        self.readout = nn.Linear(3 * cfg.d_model, cfg.d_model)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                          nn.Linear(cfg.d_model, cfg.n_bins))
            for _ in cfg.horizons)

    def _resolve_slots(self, x, e_id, c_mkt, c_feed):
        """Fill absent slots with the learned null, and apply conditioning dropout.

        ``None`` for a slot means whole-batch absence and is replaced by the null
        broadcast over the batch. When a real slot is supplied during training, each
        sample's slot is independently replaced by the null with probability
        ``cfg.p_cond`` so the model learns the slot-absent path. Dropout is keyed on
        ``self.training``; production code must gate the conditioning-dropout phase on
        a separate flag and must not enable it during a Monte-Carlo-dropout inference
        pass, which would null live slots and corrupt the prediction.
        """
        B, L, _ = x.shape
        nid = self.null_id.expand(B, -1)
        nctx = self.null_ctx.expand(B, L, -1)
        nfeed = self.null_feed.expand(B, L, -1)
        if e_id is None:
            e_id = nid
        elif self.training and self.cfg.p_cond > 0:
            keep = (torch.rand(B, 1, device=x.device) > self.cfg.p_cond).to(e_id.dtype)
            e_id = keep * e_id + (1 - keep) * nid
        if c_mkt is None:
            c_mkt = nctx
        elif self.training and self.cfg.p_cond > 0:
            keep = (torch.rand(B, 1, 1, device=x.device) > self.cfg.p_cond).to(c_mkt.dtype)
            c_mkt = keep * c_mkt + (1 - keep) * nctx
        if c_feed is None:
            c_feed = nfeed
        elif self.training and self.cfg.p_cond > 0:
            keep = (torch.rand(B, 1, 1, device=x.device) > self.cfg.p_cond).to(c_feed.dtype)
            c_feed = keep * c_feed + (1 - keep) * nfeed
        return e_id, c_mkt, c_feed

    def forward(self, x, e_id=None, c_mkt=None, c_feed=None):
        # x: (B, window, n_features); e_id: (B, d_id)|None;
        # c_mkt: (B, window, d_mkt)|None; c_feed: (B, window, d_feed)|None.
        e_id, c_mkt, c_feed = self._resolve_slots(x, e_id, c_mkt, c_feed)
        h = self.stem(x) + self.ctx_gate * self.ctx_proj(c_mkt) \
                         + self.feed_gate * self.feed_proj(c_feed)
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) // 2:
                h = h + self.mid_gate * self.mid_proj(c_mkt)
            h = blk(h, e_id)
        h = self.out_norm(h)
        # Pads are on the left; with at least min_real_rows real rows the last-day,
        # last-month and last-six-month pools are pad-free. is_pad still lets the model
        # discount any pad positions that reach the pools on short histories.
        pooled = torch.cat([h[:, -1], h[:, -21:].mean(1), h[:, -126:].mean(1)], -1)
        z = F.gelu(self.readout(pooled))
        return torch.stack([head(z) for head in self.heads], 1)   # (B, n_horizons, n_bins)


_SEAM_TAGS = ("film", "ctx_proj", "ctx_gate", "mid_proj", "mid_gate",
              "feed_proj", "feed_gate", "null_")


def optimizer_param_groups(model, weight_decay=0.01):
    """Split parameters into decay and no-decay groups for AdamW.

    Decoupled weight decay shrinks any parameter with a populated gradient, including
    the zero gradients the dormant seam projections receive during base training, so
    leaving the seams in a decay group would erode them away from their identity init.
    The no-decay group therefore holds: all seam parameters; every 1-D parameter
    (norm weights, gates, biases); and any parameter explicitly flagged
    ``_no_weight_decay`` (the Mamba ``A_log`` and ``D`` state parameters).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = (
            any(tag in name for tag in _SEAM_TAGS)
            or p.ndim < 2
            or getattr(p, "_no_weight_decay", False)
        )
        (no_decay if is_no_decay else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def seam_state_snapshot(model):
    """Clone every seam parameter for exact graft rollback.

    Restoring the snapshot recovers the pre-graft model exactly while the backbone is
    still frozen (graft stage 1); once the backbone is unfrozen (stage 2), rollback
    must instead restore the full pre-stage-2 checkpoint.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if any(tag in k for tag in _SEAM_TAGS)}


def hl_gauss_targets(z, cfg: V5Config):
    """Histogram-loss-Gaussian soft targets: ``z`` ``(B, H)`` -> ``(B, H, n_bins)``.

    Each scalar target is spread over the bins as the per-bin integral of a Gaussian
    centered at ``z`` with standard deviation ``0.75 * bin_width``, then renormalized.
    The spread makes the cross-entropy sensitive to how far a prediction misses,
    unlike a one-hot bin target. Bin geometry is derived from ``cfg`` so it cannot
    drift from a stale constant. The caller guarantees ``z`` is finite.
    """
    s = 0.75 * cfg.bin_width
    half = cfg.n_bins // 2
    edges = torch.arange(-half, half + 1, device=z.device, dtype=z.dtype) * cfg.bin_width
    zc = z.clamp(-cfg.z_clip + 2 * s, cfg.z_clip - 2 * s)
    cdf = torch.special.ndtr((edges - zc.unsqueeze(-1)) / s)
    p = cdf[..., 1:] - cdf[..., :-1]
    return p / p.sum(-1, keepdim=True)


def class_marginals(logits, theta_bins, n_bins):
    """Three-class marginals from histogram logits: ``(B, H, n_bins) -> (B, H, 3)``.

    Each per-horizon threshold ``theta_bins[h]`` is an integer bin count, so the class
    boundaries fall exactly on bin edges and ``P(down)``, ``P(neutral)``, ``P(up)`` are
    exact partial sums of bin probabilities (no interpolation). ``theta_bins`` and
    ``n_bins`` are threaded from the config so an edit to either cannot silently
    diverge from a default.
    """
    half = n_bins // 2
    p = logits.softmax(-1)
    cols = []
    for h, k in enumerate(theta_bins):
        ph = p[:, h]
        cols.append(torch.stack([ph[..., :half - k].sum(-1),
                                 ph[..., half - k:half + k].sum(-1),
                                 ph[..., half + k:].sum(-1)], -1))
    return torch.stack(cols, 1)


def v5_loss_components(logits, z, mask, cfg: V5Config):
    """Masked loss plus detached per-horizon components.

    ``logits`` ``(B, H, n_bins)``; ``z`` ``(B, H)`` carrying ``NaN`` where the label is
    masked; ``mask`` ``(B, H)`` in {0, 1}. Masked targets are replaced by a finite
    placeholder and then excluded with ``torch.where`` rather than multiplied by the
    mask, because ``NaN * 0`` is ``NaN`` and would poison the batch sum.

    Returns ``(loss, per_h)``: the masked scalar loss and a detached ``(H,)`` tensor
    of per-horizon masked means (NaN-free: horizons with no labels report 0). The
    label-count-weighted components sum back to the scalar, so the per-horizon view
    stays mix-unconfounded when train-side thinning changes the horizon composition.
    """
    lam_cls = cfg.lam_cls
    z = torch.nan_to_num(z, nan=0.0)
    logp = F.log_softmax(logits, -1)
    ce = -(hl_gauss_targets(z, cfg) * logp).sum(-1)
    theta = torch.tensor(cfg.theta_bins, device=z.device, dtype=z.dtype) * cfg.bin_width
    cls = (z > -theta).long() + (z > theta).long()
    pm = class_marginals(logits, cfg.theta_bins, cfg.n_bins).clamp_min(1e-8)
    ce3 = F.nll_loss(pm.log().flatten(0, 1), cls.flatten(), reduction="none").view_as(z)
    per = torch.where(mask.bool(), ce + lam_cls * ce3, torch.zeros_like(ce))
    loss = per.sum() / mask.sum().clamp_min(1.0)
    per_h = (per.sum(0) / mask.sum(0).clamp_min(1.0)).detach()
    return loss, per_h


def v5_loss(logits, z, mask, cfg: V5Config):
    """Masked distributional cross-entropy plus the ``cfg.lam_cls``-weighted 3-class
    term (see :func:`v5_loss_components`)."""
    loss, _ = v5_loss_components(logits, z, mask, cfg)
    return loss


@torch.no_grad()
def ensemble_predict(members, x, cfg: V5Config, e_id=None, c_mkt=None, c_feed=None,
                     temps=None):
    """Deep-ensemble predictive distribution and its uncertainty decomposition.

    ``members`` is a list of backbones in eval mode. Returns:

    * ``hist`` ``(B, H, n_bins)`` — the ensemble-mean predictive histogram (aleatoric
      shape; quantiles are read off its CDF downstream).
    * ``cls3`` ``(B, H, 3)`` — three-class marginals of the mean histogram.
    * ``up_std`` ``(B, H)`` — std across members of ``P(up)`` (epistemic disagreement).
    * ``score_std`` ``(B, H)`` — std across members of the signed score
      ``P(up) - P(down)``, computed per member before reduction. It encodes the
      bull-bear covariance across members and is not recoverable from the mean
      marginals and ``up_std`` alone.

    ``temps`` are per-horizon temperatures dividing the logits, fit on validation by
    minimizing the masked 3-class NLL of the ensemble-mean prediction. A
    Monte-Carlo-dropout fallback must enable only the dropout modules, never full
    training mode, which would re-enable conditioning dropout.
    """
    horizons = cfg.horizons
    temp_values = temps if temps is not None else [1.0] * len(horizons)
    T = torch.as_tensor(temp_values, dtype=x.dtype, device=x.device).view(1, -1, 1)
    half = cfg.n_bins // 2
    probs, up, score = [], [], []
    for m in members:
        logits = m(x, e_id, c_mkt, c_feed) / T
        probs.append(logits.softmax(-1))
        cm = class_marginals(logits, cfg.theta_bins, cfg.n_bins)
        up.append(cm[..., 2])
        score.append(cm[..., 2] - cm[..., 0])
    hist = torch.stack(probs).mean(0)
    up, score = torch.stack(up), torch.stack(score)
    cls3 = torch.stack([
        torch.stack([hist[:, h, :half - k].sum(-1),
                     hist[:, h, half - k:half + k].sum(-1),
                     hist[:, h, half + k:].sum(-1)], -1)
        for h, k in enumerate(cfg.theta_bins)], 1)
    return hist, cls3, up.std(0), score.std(0)
