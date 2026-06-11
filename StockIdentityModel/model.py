"""Temporal encoder and three-view context module.

Conventions:
  - Visibility masks are boolean with True = "query row may attend key column".
    With the relational bias active (REL-3) the context stack carries the same
    visibility as an additive float mask instead: finite bias where visible,
    -inf where not — one bias tensor per (window, scale), shared across blocks
    and across the full/peer views, masked per view out-of-place.
  - Pre-norm blocks throughout; a final LayerNorm precedes each stack's output
    head, as is standard for pre-norm stacks.
  - Feed-forward dropout masks are drawn once per (window, scale) per token and
    reused across the three views — like the observer-dropout masks — so view
    differences are attributable to the withheld evidence, not to noise.
  - Everything on the bias path (planes, PairScore, tanh cap) is a
    deterministic function of its tensor inputs: no RNG, so it may run inside
    the no-RNG checkpointed context segment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config

EDGE_STATS = ("corr0", "beta_prod", "dlogvol")


def rv_from_feats(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(returns, validity) for the relational bias from a normalized candle
    tensor [..., N, 5]: R = the close log-return channel; V = day carries any
    signal. Halted/padded days are exact-zero 5-vectors by construction
    (data.py halt convention; incomplete cells zero-filled), so V is False
    exactly on carried/padded days without consulting the panels."""
    return feats[..., 3], feats.abs().amax(dim=-1) > 0


def compute_edge_planes(
    Rg: torch.Tensor,            # [B, L, N] close log-returns
    Vg: torch.Tensor,            # [B, L, N] bool day-validity
    stats: tuple[str, ...],
    n0: float,
    min_overlap: int,
) -> torch.Tensor:               # [B, L, L, len(stats)] in `stats` order
    """Day-level pairwise co-movement planes (pure function: no params, no RNG).

    Masked estimators: each row is demeaned over ITS OWN valid days and zeroed
    elsewhere, so every cross moment only accumulates over joint-valid days.
    Pad rows (all-False validity) produce 0 for corr0/beta and a large negative
    v for dlogvol — finite everywhere (the pad diagonal stays visible as the
    softmax NaN guard, so bias(pad, pad) must be finite; the tanh cap keeps it
    so), and pad rows/columns are -inf-masked or discarded downstream.
    """
    Vf = Vg.to(Rg.dtype)
    Rv = Rg * Vf
    cnt = Vf.sum(-1)
    cnt1 = cnt.clamp(min=1.0)
    mean_i = Rv.sum(-1) / cnt1
    Rt = (Rg - mean_i[..., None]) * Vf            # demeaned, zero off own valid days
    planes = []
    for name in stats:
        if name == "corr0":
            A = Rt @ Rt.transpose(1, 2)
            B2 = (Rt * Rt) @ Vf.transpose(1, 2)   # sum_t R~_i^2 V_j
            n = Vf @ Vf.transpose(1, 2)           # joint-valid day count
            rho = A / torch.sqrt(B2 * B2.transpose(1, 2) + 1e-12)
            rho = rho * (n / (n + n0))            # shrink toward 0 at thin overlap
            planes.append(rho * (n >= min_overlap))
        elif name == "beta_prod":
            # group-mean factor over valid entries; beta_i = masked corr(r_i, m)
            m = Rv.sum(1) / Vf.sum(1).clamp(min=1.0)              # [B, N]
            mean_m = (m[:, None, :] * Vf).sum(-1) / cnt1          # [B, L]
            Mt = (m[:, None, :] - mean_m[..., None]) * Vf
            num = (Rt * Mt).sum(-1)
            den = torch.sqrt((Rt * Rt).sum(-1) * (Mt * Mt).sum(-1) + 1e-12)
            beta = num / den
            planes.append(beta[:, :, None] * beta[:, None, :])
        elif name == "dlogvol":
            v = 0.5 * torch.log((Rt * Rt).sum(-1) / cnt1 + 1e-8)  # [B, L]
            planes.append(v[:, :, None] - v[:, None, :])
        else:
            raise ValueError(f"unknown edge stat {name!r}; supported: {EDGE_STATS}")
    return torch.stack(planes, dim=-1)


class PairScore(nn.Module):
    """Zero-init additive-ridge pair scorer (the non-bilinear channel):
    raw[i, j] = W2(gelu(P_q ln(H_i) + P_k ln(H_j))) — GELU ridges over query+key
    projections escape the rank-<=head_dim bilinear family that q·k spans.
    W2 (weight AND bias) starts at zero, so the scorer contributes exactly 0
    until trained, and zeroing W2 at inference is an exact ablation."""

    def __init__(self, d_model: int, rank: int, groups: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.P_q = nn.Linear(d_model, rank, bias=True)
        self.P_k = nn.Linear(d_model, rank, bias=False)
        self.W2 = nn.Linear(rank, groups, bias=True)
        with torch.no_grad():
            self.W2.weight.zero_()
            self.W2.bias.zero_()

    def qk(self, Hg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln(Hg)
        return self.P_q(x), self.P_k(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_ff: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.p_ff = p_ff

    def forward(self, x: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.lin1(x))
        if drop_mask is not None:
            h = h * drop_mask
        elif self.training and self.p_ff > 0:
            h = F.dropout(h, self.p_ff, training=True)
        return self.lin2(h)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_temp: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.hd = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        # per-head log-temperature on q (zero-init = x1). Logit-level only:
        # softmax over a single visible key is scale-invariant, so the self
        # view (diagonal attention) never sees it. zeros consume no init RNG.
        self.log_tau = nn.Parameter(torch.zeros(n_heads)) if attn_temp else None

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        vis: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`bias` (float [B, G, Lq, Lk], -inf where invisible, G head-groups)
        replaces the bool `vis` mask when given — visibility is already folded
        in, so callers pass one or the other, never both."""
        B, Lq, _ = q_x.shape
        Lk = kv_x.shape[1]
        q = self.wq(q_x).view(B, Lq, self.h, self.hd).transpose(1, 2)
        k = self.wk(kv_x).view(B, Lk, self.h, self.hd).transpose(1, 2)
        v = self.wv(kv_x).view(B, Lk, self.h, self.hd).transpose(1, 2)
        if self.log_tau is not None:
            q = q * torch.exp(self.log_tau)[None, :, None, None]
        if bias is None:
            mask = vis.unsqueeze(1) if vis is not None else None  # broadcast over heads
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # head-group merge: one SDPA call with the [B*G, 1, Lq, Lk] float
            # mask broadcasting over the h/G heads of its group — numerically
            # equal to a per-head [B, h, Lq, Lk] broadcast, without
            # materializing the bias per head
            G = bias.shape[1]
            hpg = self.h // G
            out = F.scaled_dot_product_attention(
                q.reshape(B * G, hpg, Lq, self.hd),
                k.reshape(B * G, hpg, Lk, self.hd),
                v.reshape(B * G, hpg, Lk, self.hd),
                attn_mask=bias.reshape(B * G, 1, Lq, Lk),
            ).reshape(B, self.h, Lq, self.hd)
        return self.wo(out.transpose(1, 2).reshape(B, Lq, -1))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FF(LN(x)).

    `kv` (when given) supplies keys/values from a different stream through the
    same LN — used by the shared-context peer approximation.
    `attn_res_gate` multiplies the identity path of the attention residual;
    zeros sever an observer's entry residual so the peer view cannot leak the
    observer's own content through the skip connection.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_ff: float, attn_temp: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, attn_temp)
        self.ff = FeedForward(d_model, d_ff, p_ff)

    def forward(
        self,
        x: torch.Tensor,
        vis: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        attn_res_gate: torch.Tensor | None = None,
        ff_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qn = self.ln1(x)
        kn = self.ln1(kv) if kv is not None else qn
        a = self.attn(qn, kn, vis, bias=bias)
        x = a + (x if attn_res_gate is None else x * attn_res_gate)
        return x + self.ff(self.ln2(x), ff_mask)


class TemporalEncoder(nn.Module):
    """One ticker's normalized window -> one summary vector.

    One token per day; learned positional embedding over day slots within the
    window (calendar position never enters the model); the learned summary
    token carries no position.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Linear(5, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.N, cfg.d_model) * 0.02)
        self.summary = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.p_ff) for _ in range(cfg.temporal_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [B, N, 5] -> [B, d_model]
        B = feats.shape[0]
        x = self.proj(feats) + self.pos
        x = torch.cat([self.summary.expand(B, 1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x[:, 0])


class ContextModule(nn.Module):
    """L_ctx attention blocks over a group's summary vectors, producing three
    views per observer: full (self + peers), self-only, and peer-only.

    REL-3: when cfg.edge_stats / cfg.psn_rank are set, one additive logit-bias
    tensor per (window, scale) — cap*tanh((EdgeMLP(planes) + PairScore(H))/cap)
    — modulates the full and peer views' attention at every block. The bias is
    logit-level only (no value-path term): the self view is structurally blind
    to it, and the peer view's content closure is untouched (column o stays
    dead; o's data enters only the weight side, the same constitutive class as
    the H_o -> wq query channel).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        heads = cfg.resolved_n_heads_ctx()
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, heads, cfg.d_ff, cfg.p_ff, attn_temp=cfg.attn_temp_ctx)
            for _ in range(cfg.L_ctx)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.D)
        # conditional construction: defaults build nothing, so the state_dict
        # keys and the init-RNG stream of a default run are byte-identical to
        # the legacy module (old checkpoints/artifacts strict-load)
        self._stats = tuple(cfg.edge_stats or ())
        self.edge = None
        self.psn = None
        if self._stats:
            unknown = [s for s in self._stats if s not in EDGE_STATS]
            if unknown:
                raise ValueError(f"unknown edge_stats {unknown}; supported: {EDGE_STATS}")
            self.edge = nn.Sequential(
                nn.Linear(len(self._stats), cfg.edge_hidden),
                nn.GELU(),
                nn.Linear(cfg.edge_hidden, cfg.edge_head_groups),
            )
            with torch.no_grad():  # bias exactly 0 at init; exact ablation forever
                self.edge[2].weight.zero_()
                self.edge[2].bias.zero_()
        if cfg.psn_rank:
            self.psn = PairScore(cfg.d_model, cfg.psn_rank, cfg.edge_head_groups)
        if self.relational and heads % cfg.edge_head_groups != 0:
            raise ValueError(
                f"edge_head_groups={cfg.edge_head_groups} must divide the context head count {heads}"
            )

    @property
    def relational(self) -> bool:
        return self.edge is not None or self.psn is not None

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.ln_f(x))

    # ------------------------------------------------------------ bias build

    def _bias_chunk(self, planes_c, a_c, b) -> torch.Tensor:
        """Bias for one row-chunk: [B, |c|, L, G], tanh-capped. Deterministic
        (no RNG) — safe under nested checkpointing inside the context segment."""
        raw = None
        if planes_c is not None:
            raw = self.edge(planes_c)
        if a_c is not None:
            p = self.psn.W2(F.gelu(a_c[:, :, None, :] + b[:, None, :, :]))
            raw = p if raw is None else raw + p
        cap = self.cfg.edge_cap
        return cap * torch.tanh(raw / cap)

    def build_bias(self, Hg: torch.Tensor, Rg: torch.Tensor, Vg: torch.Tensor) -> torch.Tensor:
        """One bias backing tensor per (window, scale): [B, G, L, ceil(L/8)*8],
        -inf-prefilled, rows [..., :L] written with capped bias. The padded
        backing keeps the last-dim stride 8-aligned for SDPA bias-grad kernels;
        consumers slice [..., :L]. Shared across blocks and the full/peer views.

        The pair/hidden tensors (the O(L^2 * hidden) memory) live only inside
        per-chunk checkpoints during training, so nothing of that order is
        saved across the step; planes are an intermediate freed with this
        function's scope (and recomputed by the outer context checkpoint)."""
        cfg = self.cfg
        B, L, _ = Hg.shape
        G = cfg.edge_head_groups
        Lpad = -(-L // 8) * 8
        back = Hg.new_full((B, G, L, Lpad), float("-inf"))
        planes = None
        if self.edge is not None:
            planes = compute_edge_planes(Rg, Vg, self._stats, cfg.edge_shrink_n0, cfg.edge_min_overlap)
        a = b = None
        if self.psn is not None:
            a, b = self.psn.qk(Hg)
        chunk = max(1, cfg.edge_chunk)
        use_ckpt = torch.is_grad_enabled() and self.training
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            pc = planes[:, c0:c1] if planes is not None else None
            ac = a[:, c0:c1] if a is not None else None
            if use_ckpt:
                seg = checkpoint(self._bias_chunk, pc, ac, b, use_reentrant=False)
            else:
                seg = self._bias_chunk(pc, ac, b)
            back[:, :, c0:c1, :L] = seg.permute(0, 3, 1, 2)
        return back

    @staticmethod
    def _masked_bias(back: torch.Tensor, L: int, vis: torch.Tensor) -> torch.Tensor:
        """Per-view float mask: clone the padded backing (out-of-place w.r.t.
        the shared bias), -inf where the view's visibility says no. Returns the
        [..., :L] slice — aligned stride preserved."""
        m = back.clone()
        m[..., :L].masked_fill_(~vis[:, None], float("-inf"))
        return m[..., :L]

    def _run(self, x, vis, gates=None, ff_masks=None, kvs=None, bias=None):
        for li, blk in enumerate(self.blocks):
            x = blk(
                x,
                vis=vis,
                kv=None if kvs is None else kvs[li],
                attn_res_gate=None if gates is None else gates[li],
                ff_mask=None if ff_masks is None else ff_masks[li],
                bias=bias,
            )
        return x

    def full_view(self, H: torch.Tensor, real: torch.Tensor, rv: tuple | None = None) -> torch.Tensor:
        """Deterministic full view (inference/eval): everyone attends self + all real peers.

        rv = (returns [B, L, N], validity [B, L, N]) — required (fail-loud)
        whenever the relational pieces are configured, so an artifact can never
        silently run with its bias channel off."""
        L = H.shape[1]
        eye = torch.eye(L, dtype=torch.bool, device=H.device)
        vis = (real[:, None, :] & real[:, :, None]) | eye
        if self.relational:
            if rv is None:
                raise RuntimeError(
                    "relational context module (edge_stats/psn_rank) needs rv=(returns, validity); "
                    "got None — caller must thread the window's candle returns"
                )
            back = self.build_bias(H, rv[0], rv[1])
            return self.project(self._run(H, None, bias=self._masked_bias(back, L, vis)))
        return self.project(self._run(H, vis))

    def views(
        self,
        H: torch.Tensor,            # [B, L, d_model] padded groups
        vis_peers: torch.Tensor,    # [B, L, L] bool, observer dropout already applied; no diagonal; pad rows/cols False
        real: torch.Tensor,         # [B, L] bool
        ff_masks: list[torch.Tensor] | None,  # per block [B, L, d_ff] scaled keep-masks, or None
        rv: tuple | None = None,    # (returns [B, L, N], validity [B, L, N]) — required iff relational
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_full, z_self, z_peer), each [B, L, D]. Pad rows are garbage; caller unpacks by `real`."""
        B, L, _ = H.shape
        dev = H.device
        eye = torch.eye(L, dtype=torch.bool, device=dev)
        pad_diag = eye[None] & ~real[:, :, None]          # pad rows attend themselves (NaN guard)
        vis_full = vis_peers | eye[None]
        vis_self = (eye[None]).expand(B, L, L)
        vis_po = vis_peers | pad_diag                     # observer rows: peers only (no self)

        # one shared bias per (window, scale); per-view masks fold visibility
        # (incl. the observer-dropout pattern) into the float mask. The self
        # view stays on the bool path: single-key softmax is bias-invariant,
        # so skipping the bias there is value- and grad-exact.
        back = mask_full = mask_peer = None
        if self.relational:
            if rv is None:
                raise RuntimeError(
                    "relational context module (edge_stats/psn_rank) needs rv=(returns, validity); got None"
                )
            back = self.build_bias(H, rv[0], rv[1])
            mask_full = self._masked_bias(back, L, vis_full)
            mask_peer = self._masked_bias(back, L, vis_po)

        # FULL — one pass for the group; keep block inputs for the peer approximation.
        inputs = [H]
        x = H
        for li, blk in enumerate(self.blocks):
            x = blk(
                x,
                vis=None if mask_full is not None else vis_full,
                ff_mask=None if ff_masks is None else ff_masks[li],
                bias=mask_full,
            )
            if li < len(self.blocks) - 1:
                inputs.append(x)
        z_full = self.project(x)

        # SELF — attention restricted to each ticker itself; identical weights, so
        # differences from the full view are attributable to the withheld peers.
        z_self = self.project(self._run(H, vis_self, ff_masks=ff_masks))

        # PEER
        g_real = int(real.sum(dim=1).max().item())
        if g_real <= self.cfg.g_exact:
            bias_sl = None if mask_full is None else back[..., :L]  # unmasked shared bias; per-copy masking inside
            z_peer = self._peer_exact(H, vis_full, vis_po, real, ff_masks, bias=bias_sl)
        else:
            z_peer = self._peer_approx(H, inputs, vis_po, ff_masks, mask_peer=mask_peer)
        return z_full, z_self, z_peer

    def _peer_approx(self, H, shared_inputs, vis_po, ff_masks, mask_peer=None):
        """Shared-context peer approximation for groups above g_exact: one extra pass;
        the observer stream's block-l keys/values are the shared (full) pass's block-l
        inputs; self masked everywhere; block-1 residual cut. Peers' shared vectors
        have already absorbed the observer — a leak that shrinks as 1/g, negligible in
        exactly the large-group regime where this path runs."""
        x = H
        for li, blk in enumerate(self.blocks):
            gate = torch.zeros_like(H[..., :1]) if li == 0 else None
            x = blk(
                x,
                vis=None if mask_peer is not None else vis_po,
                kv=None if li == 0 else shared_inputs[li],  # block 1 kv = H = own input
                attn_res_gate=gate,
                ff_mask=None if ff_masks is None else ff_masks[li],
                bias=mask_peer,
            )
        return self.project(x)

    def _peer_exact(self, H, vis_full, vis_po, real, ff_masks, bias=None):
        """Exact peer view: one pass per observer, batched as L copies of the group.
        In copy o: nobody reads o (column o dead in every row — so no peer vector ever
        absorbs o's content for a later block to reflect back), row o reads only its
        visible peers, and row o's block-1 residual is severed.

        `bias` is the shared [B, G, L, L] slice; the expand->reshape materializes a
        fresh per-copy tensor (alignment padding is lost here — harmless at the
        g <= g_exact scales this path serves), masked in place with the same per-copy
        visibility, so the column-o kill applies to the bias path identically."""
        B, L, d = H.shape
        ar = torch.arange(L, device=H.device)
        eye = torch.eye(L, dtype=torch.bool, device=H.device)
        pad_diag = eye[None] & ~real[:, :, None]

        vis_c = vis_full[:, None].expand(B, L, L, L).clone()  # [B, copy o, row i, col j]
        vis_c[:, ar, :, ar] = False                            # kill column o in copy o
        vis_c[:, ar, ar, :] = vis_po                           # row o in copy o: peers only
        vis_c = vis_c | pad_diag[:, None]                      # NaN guard for pad rows
        vis_c = vis_c.reshape(B * L, L, L)

        bias_c = None
        if bias is not None:
            G = bias.shape[1]
            # clone() like vis_c above: reshape of an expand can return an
            # aliased stride-0 view (B=1 single-group case), and the in-place
            # fill needs exclusively-owned memory
            bias_c = bias[:, None].expand(B, L, G, L, L).clone().view(B * L, G, L, L)
            bias_c.masked_fill_(~vis_c[:, None], float("-inf"))

        gate1 = torch.ones(B, L, L, 1, device=H.device)
        gate1[:, ar, ar] = 0.0                                 # sever row o's entry residual, block 1 only
        gate1 = gate1.reshape(B * L, L, 1)

        x = H[:, None].expand(B, L, L, d).reshape(B * L, L, d)
        masks = None
        if ff_masks is not None:  # token-keyed FF masks, shared across copies
            masks = [m[:, None].expand(B, L, L, m.shape[-1]).reshape(B * L, L, -1) for m in ff_masks]
        for li, blk in enumerate(self.blocks):
            x = blk(
                x,
                vis=None if bias_c is not None else vis_c,
                attn_res_gate=gate1 if li == 0 else None,
                ff_mask=None if masks is None else masks[li],
                bias=bias_c,
            )
        z = self.project(x).view(B, L, L, -1)
        return z[:, ar, ar]                                    # row o of copy o


class IdentityEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg)
        self.context = ContextModule(cfg)

    def encode_window(self, feats: torch.Tensor, grad_checkpoint: bool = False) -> torch.Tensor:
        """[U, N, 5] -> [U, d_model] summary vectors."""
        if grad_checkpoint and self.training:
            return checkpoint(self.temporal, feats, use_reentrant=False)
        return self.temporal(feats)

    @torch.no_grad()
    def embed_rows(self, H: torch.Tensor, rv: tuple | None = None) -> torch.Tensor:
        """Deterministic single-group full view over one universe: [U, d_model] -> [U, D] (the inference path).

        rv = (returns [U, N], validity [U, N]) — mandatory (fail-loud in
        full_view) whenever the relational pieces are configured."""
        real = torch.ones(1, H.shape[0], dtype=torch.bool, device=H.device)
        rv2 = None if rv is None else (rv[0][None], rv[1][None])
        return self.context.full_view(H[None], real, rv=rv2)[0]
