"""Temporal encoder and three-view context module.

Conventions:
  - Visibility masks are boolean with True = "query row may attend key column".
  - Pre-norm blocks throughout; a final LayerNorm precedes each stack's output
    head, as is standard for pre-norm stacks.
  - Feed-forward dropout masks are drawn once per (window, scale) per token and
    reused across the three views — like the observer-dropout masks — so view
    differences are attributable to the withheld evidence, not to noise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config


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


class ResidualMLP(nn.Module):
    """Pre-norm residual MLP: x + lin2(gelu(lin1(LN(x)))), applied per token.

    The output layer is zero-initialized, so the block is the identity at init:
    inserted depth leaves step-0 behavior (init calibration, the opening
    contraction-vs-fence race) unchanged and grows in as training recruits it.
    Deliberately no dropout: these blocks run once per view, and fresh draws per
    view would make view differences partly noise rather than withheld evidence.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(F.gelu(self.lin1(self.ln(x))))


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.hd = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, vis: torch.Tensor | None) -> torch.Tensor:
        B, Lq, _ = q_x.shape
        Lk = kv_x.shape[1]
        q = self.wq(q_x).view(B, Lq, self.h, self.hd).transpose(1, 2)
        k = self.wk(kv_x).view(B, Lk, self.h, self.hd).transpose(1, 2)
        v = self.wv(kv_x).view(B, Lk, self.h, self.hd).transpose(1, 2)
        mask = vis.unsqueeze(1) if vis is not None else None  # broadcast over heads
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(B, Lq, -1))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FF(LN(x)).

    `kv` (when given) supplies keys/values from a different stream through the
    same LN — used by the shared-context peer approximation.
    `attn_res_gate` multiplies the identity path of the attention residual;
    zeros sever an observer's entry residual so the peer view cannot leak the
    observer's own content through the skip connection.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_ff: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, p_ff)

    def forward(
        self,
        x: torch.Tensor,
        vis: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        attn_res_gate: torch.Tensor | None = None,
        ff_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qn = self.ln1(x)
        kn = self.ln1(kv) if kv is not None else qn
        a = self.attn(qn, kn, vis)
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

    Optional seam blocks (per-ticker ResidualMLPs, traversed identically by all
    three views): `adapter` re-encodes the temporal summaries before attention —
    a nonlinear matching kernel, and deeper self/peer routes; `head` reshapes
    the metric before the final projection. Both are row-local, so they add no
    set-size dependence and cannot leak an observer's content into peers'
    values; the leak-closure masking below operates downstream unchanged.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.p_ff) for _ in range(cfg.L_ctx)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.D)
        # seam blocks (both default 0 = the r7 architecture). Declared last so the
        # pre-existing parameters above consume the same init-RNG draws as a
        # blocks-free run at the same seed (paired-run comparability).
        self.adapter = nn.ModuleList(  # seam A: re-encode summaries before attention
            ResidualMLP(cfg.d_model, cfg.d_ff) for _ in range(cfg.adapter_blocks)
        )
        self.head = nn.ModuleList(  # seam B: nonlinear trunk before the final projection
            ResidualMLP(cfg.d_model, cfg.d_ff) for _ in range(cfg.head_blocks)
        )

    def _adapt(self, H: torch.Tensor) -> torch.Tensor:
        for blk in self.adapter:
            H = blk(H)
        return H

    def project(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.head:
            x = blk(x)
        return self.out(self.ln_f(x))

    def _run(self, x, vis, gates=None, ff_masks=None, kvs=None):
        for li, blk in enumerate(self.blocks):
            x = blk(
                x,
                vis=vis,
                kv=None if kvs is None else kvs[li],
                attn_res_gate=None if gates is None else gates[li],
                ff_mask=None if ff_masks is None else ff_masks[li],
            )
        return x

    def full_view(self, H: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Deterministic full view (inference/eval): everyone attends self + all real peers."""
        H = self._adapt(H)
        L = H.shape[1]
        eye = torch.eye(L, dtype=torch.bool, device=H.device)
        vis = (real[:, None, :] & real[:, :, None]) | eye
        return self.project(self._run(H, vis))

    def views(
        self,
        H: torch.Tensor,            # [B, L, d_model] padded groups
        vis_peers: torch.Tensor,    # [B, L, L] bool, observer dropout already applied; no diagonal; pad rows/cols False
        real: torch.Tensor,         # [B, L] bool
        ff_masks: list[torch.Tensor] | None,  # per block [B, L, d_ff] scaled keep-masks, or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_full, z_self, z_peer), each [B, L, D]. Pad rows are garbage; caller unpacks by `real`."""
        H = self._adapt(H)  # pad rows (zeros) map to a shared constant; the masks keep them isolated
        B, L, _ = H.shape
        dev = H.device
        eye = torch.eye(L, dtype=torch.bool, device=dev)
        pad_diag = eye[None] & ~real[:, :, None]          # pad rows attend themselves (NaN guard)
        vis_full = vis_peers | eye[None]
        vis_self = (eye[None]).expand(B, L, L)
        vis_po = vis_peers | pad_diag                     # observer rows: peers only (no self)

        # FULL — one pass for the group; keep block inputs for the peer approximation.
        inputs = [H]
        x = H
        for li, blk in enumerate(self.blocks):
            x = blk(x, vis=vis_full, ff_mask=None if ff_masks is None else ff_masks[li])
            if li < len(self.blocks) - 1:
                inputs.append(x)
        z_full = self.project(x)

        # SELF — attention restricted to each ticker itself; identical weights, so
        # differences from the full view are attributable to the withheld peers.
        z_self = self.project(self._run(H, vis_self, ff_masks=ff_masks))

        # PEER
        g_real = int(real.sum(dim=1).max().item())
        if g_real <= self.cfg.g_exact:
            z_peer = self._peer_exact(H, vis_full, vis_po, real, ff_masks)
        else:
            z_peer = self._peer_approx(H, inputs, vis_po, ff_masks)
        return z_full, z_self, z_peer

    def _peer_approx(self, H, shared_inputs, vis_po, ff_masks):
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
                vis=vis_po,
                kv=None if li == 0 else shared_inputs[li],  # block 1 kv = H = own input
                attn_res_gate=gate,
                ff_mask=None if ff_masks is None else ff_masks[li],
            )
        return self.project(x)

    def _peer_exact(self, H, vis_full, vis_po, real, ff_masks):
        """Exact peer view: one pass per observer, batched as L copies of the group.
        In copy o: nobody reads o (column o dead in every row — so no peer vector ever
        absorbs o's content for a later block to reflect back), row o reads only its
        visible peers, and row o's block-1 residual is severed."""
        B, L, d = H.shape
        ar = torch.arange(L, device=H.device)
        eye = torch.eye(L, dtype=torch.bool, device=H.device)
        pad_diag = eye[None] & ~real[:, :, None]

        vis_c = vis_full[:, None].expand(B, L, L, L).clone()  # [B, copy o, row i, col j]
        vis_c[:, ar, :, ar] = False                            # kill column o in copy o
        vis_c[:, ar, ar, :] = vis_po                           # row o in copy o: peers only
        vis_c = vis_c | pad_diag[:, None]                      # NaN guard for pad rows
        vis_c = vis_c.reshape(B * L, L, L)

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
                vis=vis_c,
                attn_res_gate=gate1 if li == 0 else None,
                ff_mask=None if masks is None else masks[li],
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
    def embed_rows(self, H: torch.Tensor) -> torch.Tensor:
        """Deterministic single-group full view over one universe: [U, d_model] -> [U, D] (the inference path)."""
        real = torch.ones(1, H.shape[0], dtype=torch.bool, device=H.device)
        return self.context.full_view(H[None], real)[0]
