"""Loss terms over one training step's embeddings.

A step's embeddings are flattened into row-aligned arrays: row e holds the
embedding of `ticker[e]` produced at window slot `slot[e]` (the same window
drawn twice occupies two slots), at scale `scale[e]` (= group count), inside a
group of size `gsize[e]`. The three views share the row layout, so all index
structures are built once per step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config

VIEWS = ("full", "self", "peer")


@dataclass
class StepStore:
    """Accumulates one step's embeddings and bookkeeping."""
    ticker: list = field(default_factory=list)   # global ticker ids
    slot: list = field(default_factory=list)     # window-slot index 0..M*W-1
    t_w: list = field(default_factory=list)      # window ordinal in the tiling (for kappa)
    scale: list = field(default_factory=list)    # group count n_s
    gsize: list = field(default_factory=list)    # observer's group size
    z: dict = field(default_factory=lambda: {v: [] for v in VIEWS})
    slices: list = field(default_factory=list)   # per (slot, scale): (start, end, group_id array)

    def add(self, tickers, slot, t_w, scale, gsizes, group_ids, z_full, z_self, z_peer):
        start = len(self.ticker)
        n = len(tickers)
        self.ticker.extend(int(t) for t in tickers)
        self.slot.extend([slot] * n)
        self.t_w.extend([t_w] * n)
        self.scale.extend([scale] * n)
        self.gsize.extend(int(g) for g in gsizes)
        self.z["full"].append(z_full)
        self.z["self"].append(z_self)
        self.z["peer"].append(z_peer)
        self.slices.append((start, start + n, np.asarray(group_ids)))

    def finalize(self, device) -> "StepBatch":
        return StepBatch(
            ticker=torch.tensor(self.ticker, dtype=torch.long, device=device),
            slot=np.asarray(self.slot),
            t_w=np.asarray(self.t_w, dtype=np.float64),
            scale=np.asarray(self.scale),
            gsize=np.asarray(self.gsize, dtype=np.float64),
            z={v: torch.cat(self.z[v], dim=0) for v in VIEWS},
            slices=self.slices,
        )


@dataclass
class StepBatch:
    ticker: torch.Tensor
    slot: np.ndarray
    t_w: np.ndarray
    scale: np.ndarray
    gsize: np.ndarray
    z: dict
    slices: list


def _safe_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance with a sqrt floor (cdist's gradient is NaN at 0)."""
    d2 = (a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2.0 * (a @ b.T)
    return torch.sqrt(torch.clamp(d2, min=1e-12))


def _segment_mean(values: torch.Tensor, seg_id: torch.Tensor, n_seg: int) -> torch.Tensor:
    out = torch.zeros(n_seg, values.shape[1], dtype=values.dtype, device=values.device)
    out.index_add_(0, seg_id, values)
    cnt = torch.zeros(n_seg, dtype=values.dtype, device=values.device)
    cnt.index_add_(0, seg_id, torch.ones_like(seg_id, dtype=values.dtype))
    return out / cnt[:, None]


class StepIndex:
    """View-independent index structures for one step (built once, numpy)."""

    def __init__(self, B: StepBatch, cfg: Config, tau_prox: float):
        tick = B.ticker.cpu().numpy()
        # --- (ticker, scale) segments for mu and temporal pairs ---
        key = tick.astype(np.int64) * 1000 + np.log2(B.scale).astype(np.int64)
        uniq, seg_id = np.unique(key, return_inverse=True)
        self.n_seg = len(uniq)
        self.seg_id = torch.from_numpy(seg_id)
        self.seg_ticker = np.empty(self.n_seg, dtype=np.int64)
        self.seg_scale = np.empty(self.n_seg, dtype=np.int64)
        self.seg_ticker[seg_id] = tick
        self.seg_scale[seg_id] = B.scale

        # --- temporal-consistency pairs within each (ticker, scale) segment ---
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        groups = np.split(order, bounds)
        pa, pb, pseg = [], [], []
        for g in groups:
            if len(g) < 2:
                continue
            ii, jj = np.triu_indices(len(g), k=1)
            pa.append(g[ii]); pb.append(g[jj]); pseg.append(np.full(len(ii), seg_id[g[0]]))
        if pa:
            pa, pb, pseg = np.concatenate(pa), np.concatenate(pb), np.concatenate(pseg)
            delta = np.abs(B.t_w[pa] - B.t_w[pb])
            kappa = 1.0 + cfg.alpha_prox * np.exp(-delta / tau_prox)
            gmin = np.minimum(B.gsize[pa], B.gsize[pb])
            # omega per pair via min group size: trusts a pair no more than its thinner
            # context, and reduces to a per-scale omega(g_s) when sizes match across windows.
            omega = (gmin - 1.0) / (gmin - 1.0 + cfg.c_g)
            self.pair_a = torch.from_numpy(pa)
            self.pair_b = torch.from_numpy(pb)
            self.pair_seg = torch.from_numpy(pseg)
            self.pair_kw = torch.from_numpy((kappa * omega).astype(np.float32))
            self.pair_k = torch.from_numpy(kappa.astype(np.float32))
        else:
            self.pair_a = None

        # --- per-ticker grouping of segments (for L_sc) ---
        tuniq, tseg = np.unique(self.seg_ticker, return_inverse=True)
        self.sc_tickers = torch.from_numpy(tuniq)
        self.sc_seg_of_mu = torch.from_numpy(tseg)
        self.sc_count = torch.from_numpy(np.bincount(tseg).astype(np.float32))


def view_terms(cfg: Config, B: StepBatch, idx: StepIndex, view: str, T_global: int):
    """L_sc, L_xsep, L_tc, L_psep for one view, plus per-ticker consistency summands."""
    z = B.z[view]
    dev = z.device
    m_sep = cfg.resolved_m_sep()
    out = {}

    seg_id = idx.seg_id.to(dev)
    mu = _segment_mean(z, seg_id, idx.n_seg)  # [n_seg, D]

    # --- scale consistency: per ticker, Var over scales of mu (population, summed over D)
    sc_seg = idx.sc_seg_of_mu.to(dev)
    n_t = len(idx.sc_tickers)
    mu_bar = _segment_mean(mu, sc_seg, n_t)
    sq = ((mu - mu_bar[sc_seg]) ** 2).sum(-1)
    var_t = torch.zeros(n_t, dtype=z.dtype, device=dev).index_add_(0, sc_seg, sq) / idx.sc_count.to(dev)
    sc_eligible = idx.sc_count.to(dev) >= 2
    sc_per_ticker = torch.zeros(T_global, dtype=z.dtype, device=dev)
    sc_per_ticker[idx.sc_tickers.to(dev)[sc_eligible]] = var_t[sc_eligible]
    out["sc"] = var_t[sc_eligible].sum() if sc_eligible.any() else None

    # --- cross-scale separation: (i, s) vs (j != i, s' != s), hinge^2 on mu distances
    dist = _safe_dist(mu, mu)
    st = torch.from_numpy(idx.seg_ticker).to(dev)
    ss = torch.from_numpy(idx.seg_scale).to(dev)
    valid = (st[:, None] != st[None, :]) & (ss[:, None] != ss[None, :])
    if valid.any():
        h = torch.clamp(m_sep - dist, min=0.0)
        out["xsep"] = (h[valid] ** 2).mean()
    else:
        out["xsep"] = None

    # --- temporal consistency: same ticker across windows, kappa/omega-weighted pairs
    tc_per_ticker = torch.zeros(T_global, dtype=z.dtype, device=dev)
    if idx.pair_a is not None:
        a, b = idx.pair_a.to(dev), idx.pair_b.to(dev)
        d2 = ((z[a] - z[b]) ** 2).sum(-1)
        pseg = idx.pair_seg.to(dev)
        num = torch.zeros(idx.n_seg, dtype=z.dtype, device=dev).index_add_(0, pseg, idx.pair_kw.to(dev) * d2)
        den = torch.zeros(idx.n_seg, dtype=z.dtype, device=dev).index_add_(0, pseg, idx.pair_k.to(dev))
        has = den > 0
        tc_seg = torch.zeros_like(num)
        tc_seg[has] = num[has] / den[has]
        tc_per_ticker.index_add_(0, torch.from_numpy(idx.seg_ticker).to(dev), tc_seg)
        out["tc"] = tc_seg[has].sum()
    else:
        out["tc"] = None

    # --- peer separation: within each group at each (slot, scale), ordered pairs
    hinge_sum = z.new_zeros(())
    cnt = 0
    for start, end, gid in B.slices:
        zs = z[start:end]
        gid_t = torch.from_numpy(gid).to(dev)
        same = gid_t[:, None] == gid_t[None, :]
        same.fill_diagonal_(False)
        if not same.any():
            continue
        d = _safe_dist(zs, zs)
        h = torch.clamp(m_sep - d, min=0.0)
        hinge_sum = hinge_sum + (h[same] ** 2).sum()
        cnt += int(same.sum().item())
    out["psep"] = hinge_sum / cnt if cnt else None

    return out, sc_per_ticker, tc_per_ticker


def step_losses(cfg: Config, B: StepBatch, idx: StepIndex, T_global: int, anchors: "AnchorState"):
    """All raw loss terms + per-ticker inconsistency I_i + the step's z-bar (full view)."""
    terms: dict[str, torch.Tensor | None] = {}
    I = {}
    per_ticker_full = None
    for v in VIEWS:
        out, sc_pt, tc_pt = view_terms(cfg, B, idx, v, T_global)
        for k, val in out.items():
            terms[f"{k}_{v}"] = val
        if out["sc"] is not None and out["tc"] is not None:
            I[v] = out["sc"] + out["tc"]
        if v == "full":
            per_ticker_full = sc_pt + tc_pt

    # --- synergy: full-view inconsistency vs the harmonic mean of the masked views'.
    # No stop-gradient on the masked views — deliberate: freezing the denominator would
    # reduce this term to extra weight on full-view consistency and lose the coupling
    # that forces the full view to outperform both masked routes.
    if len(I) == 3:
        P_self = 1.0 / (I["self"] + cfg.eps)
        P_peer = 1.0 / (I["peer"] + cfg.eps)
        P_full = 1.0 / (I["full"] + cfg.eps)
        terms["syn"] = (P_self + P_peer) / (P_full + cfg.eps)
    else:
        terms["syn"] = None

    # --- z-bar: per-ticker mean over (window, scale), full view; feeds anchor + utilization
    z_full = B.z["full"]
    present, inv = torch.unique(B.ticker, return_inverse=True)
    zbar = _segment_mean(z_full, inv, len(present))

    # --- anchor: distance to the PRE-update anchor; the EMA update runs post-backward in train.py
    elig = anchors.initialized[present]
    if elig.any():
        a = anchors.a[present[elig]]
        terms["anc"] = (zbar[elig] - a.detach()).abs().mean(dim=1).mean()
    else:
        terms["anc"] = None

    # --- utilization over the step's population {z-bar_i} — the anti-collapse backstop.
    # One point per ticker: pooling over (w, s) keeps a ticker's own scatter out of the
    # variance floor, so the floor measures pure between-ticker spread.
    if len(present) >= 2:
        v_d = zbar.var(dim=0, unbiased=False)
        L_var = torch.clamp(cfg.v0 ** 0.5 - torch.sqrt(v_d + cfg.eps), min=0.0).pow(2).mean()
        std = torch.sqrt(v_d + cfg.eps)
        xt = (zbar - zbar.mean(dim=0)) / std
        c = (xt.T @ xt) / len(present)
        D = c.shape[0]
        off = ~torch.eye(D, dtype=torch.bool, device=c.device)
        L_cov = (c[off] ** 2).mean()
        terms["util"] = L_var + cfg.lambda_cov * L_cov
        spectrum = v_d.detach()
    else:
        terms["util"] = None
        spectrum = None

    return terms, I, per_ticker_full, present, zbar, spectrum


class AnchorState:
    """Per-ticker anchor buffers (slow-moving EMAs of step embeddings): training-only
    state, never optimizer parameters, unused at inference."""

    def __init__(self, T: int, D: int, cfg: Config, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.a = torch.zeros(T, D, device=device)
        self.initialized = torch.zeros(T, dtype=torch.bool, device=device)
        self.tau_gain: float | None = None  # running mean of I_i (same beta as the loss normalizers)

    @torch.no_grad()
    def update(self, present: torch.Tensor, zbar: torch.Tensor, I_per_ticker: torch.Tensor) -> dict:
        """EMA update, run AFTER the step's loss has used the pre-update anchors.

        Gain: tickers the model is currently consistent about (low I_i) move their
        anchor faster — the fresh estimate is trustworthy; inconsistent ones barely
        move it.
        """
        cfg = self.cfg
        zb = zbar.detach()
        I_i = I_per_ticker[present].detach()
        stats = {}
        elig = self.initialized[present]
        if elig.any():
            tau = self.tau_gain if self.tau_gain is not None else float(I_i.mean())
            eta = cfg.eta0 * torch.exp(-I_i[elig] / (tau + cfg.eps))
            rows = present[elig]
            drift = (zb[elig] - self.a[rows]).norm(dim=1)
            stats = {
                "anchor_drift_mean": float(drift.mean()),
                "anchor_drift_p90": float(drift.quantile(0.9)),
                "eta_mean": float(eta.mean()),
            }
            self.a[rows] = (1.0 - eta[:, None]) * self.a[rows] + eta[:, None] * zb[elig]
        new = present[~elig]
        if len(new):
            self.a[new] = zb[~elig]
            self.initialized[new] = True
        # tau_gain EMA: use-then-update, like the loss normalizers
        m = float(I_i.mean()) if len(I_i) else None
        if m is not None:
            self.tau_gain = m if self.tau_gain is None else cfg.beta * self.tau_gain + (1 - cfg.beta) * m
        return stats

    def state_dict(self) -> dict:
        return {"a": self.a, "initialized": self.initialized, "tau_gain": self.tau_gain}

    def load_state_dict(self, st: dict) -> None:
        self.a = st["a"].to(self.a.device)
        self.initialized = st["initialized"].to(self.initialized.device)
        self.tau_gain = st["tau_gain"]


class EmaNormalizer:
    """Each loss term divided by a frozen running average of its own magnitude, so the
    lambda weights stay pure relative priorities as raw magnitudes shrink over training.

    Init to the term's first-step value (step 1 normalizes to exactly 1); the EMA
    updates after the normalized loss is computed, using the previous step's EMA.
    """

    def __init__(self, beta: float, eps: float):
        self.beta, self.eps = beta, eps
        self.ema: dict[str, float] = {}

    def normalize(self, name: str, raw: torch.Tensor) -> torch.Tensor:
        prev = self.ema.get(name, float(raw.detach()))
        normalized = raw / (prev + self.eps)
        self.ema[name] = self.beta * prev + (1 - self.beta) * float(raw.detach()) if name in self.ema else float(raw.detach())
        return normalized

    def state_dict(self) -> dict:
        return dict(self.ema)

    def load_state_dict(self, st: dict) -> None:
        self.ema = dict(st)


def combine(cfg: Config, terms: dict, norm: EmaNormalizer):
    """Normalized weighted sum of all loss terms. Absent terms are skipped."""
    lam_view = {"full": cfg.lambda_full, "self": cfg.lambda_self, "peer": cfg.lambda_peer}
    lam_term = {"sc": cfg.lambda_sc, "tc": cfg.lambda_tc, "xsep": cfg.lambda_xsep, "psep": cfg.lambda_psep}
    total = None
    normalized_log = {}
    for v in VIEWS:
        for t in ("sc", "tc", "xsep", "psep"):
            raw = terms.get(f"{t}_{v}")
            if raw is None:
                continue
            n = norm.normalize(f"{t}_{v}", raw)
            normalized_log[f"{t}_{v}"] = float(n.detach())
            contrib = lam_view[v] * lam_term[t] * n
            total = contrib if total is None else total + contrib
    for name, lam in (("syn", cfg.lambda_syn), ("anc", cfg.lambda_anc), ("util", cfg.lambda_util)):
        raw = terms.get(name)
        if raw is None:
            continue
        n = norm.normalize(name, raw)
        normalized_log[name] = float(n.detach())
        total = lam * n if total is None else total + lam * n
    return total, normalized_log
