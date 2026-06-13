"""CPU-executable fallback for the selective state-space (Mamba) operator.

The training backbone uses the fused CUDA kernel from ``mamba-ssm`` when an NVIDIA
GPU is present. That kernel does not build or run on CPU-only hosts, so live
scoring and continuous-integration tests would be impossible without a substitute.

``MambaRef`` is that substitute: a pure-PyTorch module whose parameters carry the
exact names and shapes of ``mamba_ssm.Mamba`` (``in_proj``, ``conv1d``, ``x_proj``,
``dt_proj``, ``A_log``, ``D``, ``out_proj``). A checkpoint trained with the CUDA
kernel therefore loads into ``MambaRef`` with no key remapping or shape surgery.

Two scan implementations live here and must agree numerically:

* ``selective_scan_ref`` — a faithful port of the reference selective scan shipped
  inside ``mamba-ssm`` (``mamba_ssm.ops.selective_scan_interface.selective_scan_ref``).
  It is the trusted ground truth the CUDA kernel is validated against upstream.
* ``selective_scan_seq`` — the scan ``MambaRef`` actually runs. It is an independent
  batched sequential recurrence, vectorized over the batch/feature/state dimensions
  so a whole batch of windows advances one timestep at a time (no per-window Python
  loop, which would be overhead-bound on large universes).

Keeping the two implementations distinct lets CPU CI assert ``selective_scan_seq``
matches ``selective_scan_ref`` to within a tight tolerance; a periodic GPU job
separately checks the reference against the CUDA kernel, closing the chain
CUDA ≈ reference ≈ MambaRef without ever needing a GPU in CI.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    """Reference selective scan, ported from ``mamba-ssm``.

    Shapes: ``u``/``delta`` ``(b, d, l)``; ``A`` ``(d, n)``; ``B``/``C`` ``(b, n, l)``
    (input-dependent selection); ``D`` ``(d,)``; ``z`` ``(b, d, l)`` (gating branch);
    ``delta_bias`` ``(d,)``. Computation runs in fp32 and casts back to ``u``'s dtype.

    The recurrence is ``x_t = exp(Δ_t·A)·x_{t-1} + (Δ_t·B_t)·u_t`` with emission
    ``y_t = C_t·x_t``, the skip term ``D·u`` and the ``SiLU(z)`` output gate.
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    A, B, C = A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    dstate = A.shape[1]
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = u.new_zeros((batch, dim, dstate))
    ys = []
    for i in range(length):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.float().unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


def selective_scan_seq(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    """Independent batched sequential selective scan used by ``MambaRef``.

    Identical signature and math to :func:`selective_scan_ref`, written separately so
    the two can be cross-checked: per timestep it updates the ``(b, d, n)`` state for
    the entire batch at once, then contracts against ``C_t``.
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    A, B, C = A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    dstate = A.shape[1]
    # Decay and input-injection precomputed for all timesteps: (b, d, l, n).
    decay = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    inject = torch.einsum("bdl,bnl->bdln", delta * u, B)
    state = u.new_zeros((batch, dim, dstate))
    out = u.new_empty((batch, dim, length))
    for t in range(length):
        state = decay[:, :, t] * state + inject[:, :, t]
        out[:, :, t] = (state * C[:, :, t].unsqueeze(1)).sum(-1)
    if D is not None:
        out = out + u * D.float().unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype_in)


class MambaRef(nn.Module):
    """Pure-PyTorch Mamba block with ``mamba_ssm.Mamba`` parameter parity.

    Constructor signature mirrors the subset of ``mamba_ssm.Mamba`` arguments the
    backbone uses, and parameter initialization (``A_log`` log-spaced, ``dt_proj``
    bias seeded through the inverse softplus of a log-uniform timestep) matches the
    upstream module so a model built from this class trains sensibly when no GPU is
    available and so checkpoints are interchangeable between the two.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt(dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Mamba excludes A_log and D from weight decay; flag them so the optimizer
        # grouping in the backbone routes them to the no-decay group.
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _init_dt(self, dt_min, dt_max, dt_init, dt_scale, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"unknown dt_init: {dt_init}")
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Invert softplus so the forward softplus(dt_proj.bias) reproduces ``dt``.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, hidden_states, scan_fn=None):
        """``(B, L, d_model) -> (B, L, d_model)``.

        ``scan_fn`` selects the scan implementation; it defaults to
        :func:`selective_scan_seq` and is overridable so tests can substitute the
        reference scan and assert agreement.
        """
        scan = scan_fn if scan_fn is not None else selective_scan_seq
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)        # (B, 2*d_inner, L)
        x, z = xz.chunk(2, dim=1)                               # each (B, d_inner, L)
        x = self.act(self.conv1d(x)[..., :seqlen])              # causal depthwise conv
        x_dbl = self.x_proj(x.transpose(1, 2))                  # (B, L, dt_rank+2N)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(1, 2)           # (B, d_inner, L)
        B = B.transpose(1, 2)                                   # (B, N, L)
        C = C.transpose(1, 2)                                   # (B, N, L)
        A = -torch.exp(self.A_log.float())                      # (d_inner, N)
        y = scan(x, dt, A, B, C, D=self.D, z=z,
                 delta_bias=self.dt_proj.bias, delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))                 # (B, L, d_model)
