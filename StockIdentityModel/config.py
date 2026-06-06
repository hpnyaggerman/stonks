"""Configuration for the Stock Identity Encoder."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # --- paths ---
    raw_dir: str = "TrainingData/indicators_data/raw"   # raw OHLCV (the processed CSVs drop these columns)
    run_dir: str = "StockIdentityModel/runs/default"

    # --- data layer ---
    N: int = 64                  # window length, trading days
    M: int = 4                   # number of strata
    W: int = 2                   # windows drawn per stratum per step
    Y: int = 8                   # minimum group size
    calendar: str = "benchmark"  # "benchmark" = SPY trading days; "union" = union of ticker dates
    q_clip: float = 0.999        # log-return clipping quantile (global, symmetric, frozen)
    window_offset: bool = True   # training only: shift the whole tiling by one random offset
                                 # delta in [0, N) per sampler epoch — same 104 fixed inputs
                                 # repeated ~1500x fed the r3 memorization; eval/export keep
                                 # the fixed (delta=0) tiling

    # --- model ---
    D: int = 32                  # embedding dimension
    d_model: int = 128
    n_heads: int = 4
    temporal_layers: int = 2
    L_ctx: int = 2               # context module depth
    d_ff: int = 512              # feed-forward width (4*d_model)
    p_attn: float = 0.15         # observer dropout rate
    p_ff: float = 0.10           # feed-forward dropout rate
    g_exact: int = 64            # exact peer view up to this group size; beyond it, the shared-context approximation

    # --- loss ---
    alpha_prox: float = 1.0      # proximity boost
    tau_prox: float | None = None  # None -> (#usable windows)/10
    c_g: float = 16.0            # context half-trust group size
    m_sep: float | None = None   # None -> sqrt(D)/2
    v0: float = 1.0              # per-dimension variance floor
    lambda_cov: float = 1.0      # decorrelation weight inside L_util
    eta0: float = 0.05           # anchor base gain
    beta: float = 0.99           # EMA decay (loss normalizers and tau_gain)
    eps: float = 1e-6
    kappa_floor: float = 0.01    # EMA-normalizer denominator floor, as a fraction of the term's
                                 # first-step value: caps a shrinking term's gradient
                                 # self-amplification at 1/kappa_floor (collapse guard, run r1)
    lambda_full: float = 1.0
    lambda_self: float = 0.5
    lambda_peer: float = 0.5
    lambda_sc: float = 1.0
    lambda_tc: float = 1.0
    lambda_xsep: float = 1.0
    lambda_psep: float = 1.0
    lambda_anc: float = 1.0
    lambda_util: float = 3.0     # raised 1 -> 3 after r3: dimensional concentration (median dim
                                 # variance ~1e-4 vs v0=1, identity packed into a handful of dims)
                                 # while util's force was the one fighting and losing at 1.0
    lambda_syn: float = 0.3      # principal tuning knob: sets the masked-vs-full equilibrium

    # --- optimizer & schedule ---
    lr_peak: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 1000     # deliberately overlaps the loss-normalizer burn-in
    max_steps: int = 20000       # cosine horizon; the operator usually picks an earlier checkpoint off the eval curves
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0

    # --- holdout ---
    holdout_frac: float = 0.05
    holdout_floor: int = 10      # used when pool < 100
    holdout_seed: int = 7

    # --- training / eval ---
    train_seed: int = 0
    eval_every: int = 250        # eval cadence, in steps; checkpoints written at the same cadence
    eval_windows: int = 16       # windows used by the held-out consistency/retrieval protocol
    eval_redraws: int = 5        # partition-redraw agreement samples
    best_metric: str = "retrieval"  # best.pt selection: "retrieval" (max holdout retrieval_acc) or
                                 # "consistency" (min stratified holdout consistency median;
                                 # scale-dependent — cross-check the retrieval column for collapse)
    cons_eval_windows: int = 32  # windows for the stratified consistency metric, spread over the
                                 # M strata the way training sampling spreads its draws
    K_inf: int = 4               # windows averaged at inference
    calibrate_init: bool = True  # fresh runs: rescale the final projection so per-dim var(z-bar)
                                 # starts at v0 — hinges begin satisfied (fences, not springs)
    grad_diag_every: int = 250   # cadence (steps) of per-term gradient-force diagnostics
                                 # in train_log; 0 disables
    grad_checkpoint: bool = True # recompute temporal encoder in backward (CPU RAM)
    device: str = "auto"         # "auto" -> cuda if available, else cpu; or "cpu" / "cuda" / "cuda:N"
    num_threads: int = 4

    def resolved_m_sep(self) -> float:
        return self.m_sep if self.m_sep is not None else math.sqrt(self.D) / 2.0

    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
