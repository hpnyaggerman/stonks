"""Parity and checkpoint-interchange tests for the CPU Mamba fallback."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v5.mamba_ref import MambaRef, selective_scan_ref, selective_scan_seq

# The parameter key set a checkpoint must carry to load interchangeably with the
# fused CUDA module.
EXPECTED_KEYS = {"in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight",
                 "dt_proj.weight", "dt_proj.bias", "A_log", "D", "out_proj.weight"}


def test_scan_function_parity():
    """MambaRef's own scan matches the reference scan to floating-point noise."""
    torch.manual_seed(1)
    for b, d, length, n in [(2, 8, 13, 4), (3, 16, 30, 16), (1, 32, 7, 8)]:
        u = torch.randn(b, d, length)
        delta = torch.randn(b, d, length)
        A = -torch.rand(d, n)
        B = torch.randn(b, n, length)
        C = torch.randn(b, n, length)
        D = torch.rand(d)
        z = torch.randn(b, d, length)
        bias = torch.randn(d)
        y_ref = selective_scan_ref(u, delta, A, B, C, D=D, z=z, delta_bias=bias, delta_softplus=True)
        y_seq = selective_scan_seq(u, delta, A, B, C, D=D, z=z, delta_bias=bias, delta_softplus=True)
        assert (y_ref - y_seq).abs().max().item() < 1e-3


def test_block_parity():
    """The full block output matches when the reference scan is substituted."""
    torch.manual_seed(2)
    m = MambaRef(d_model=24, d_state=8, d_conv=4, expand=2).eval()
    x = torch.randn(4, 20, 24)
    out_default = m(x)
    out_ref = m(x, scan_fn=selective_scan_ref)
    assert (out_default - out_ref).abs().max().item() < 1e-3


def test_state_dict_keys():
    m = MambaRef(d_model=16, d_state=8)
    assert set(m.state_dict().keys()) == EXPECTED_KEYS


def test_checkpoint_roundtrip():
    """A checkpoint saved from one instance loads cleanly into another and reproduces
    the same output."""
    torch.manual_seed(3)
    a = MambaRef(d_model=24, d_state=8).eval()
    b = MambaRef(d_model=24, d_state=8).eval()
    x = torch.randn(2, 15, 24)
    assert (a(x) - b(x)).abs().max().item() > 1e-4   # distinct before loading
    missing, unexpected = b.load_state_dict(a.state_dict())
    assert not missing and not unexpected
    assert (a(x) - b(x)).abs().max().item() == 0.0
