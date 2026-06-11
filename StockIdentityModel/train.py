"""Training loop.

Run from the repo root:
    python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import MISSING, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import KNOB_FIELDS, Config, REPO_ROOT
from .data import StockData, ladder
from .evaluate import run_eval
from .losses import AnchorState, EmaNormalizer, StepIndex, StepStore, combine, grad_force_diag, step_losses
from .model import IdentityEncoder, compute_edge_planes, rv_from_feats
from .sampling import StratumSampler, draw_partitions


def build_optimizer(model: IdentityEncoder, cfg: Config) -> torch.optim.AdamW:
    """AdamW with weight decay excluding biases, LayerNorm params, and the final
    d_model -> D projection (decay there pushes against the variance floor v0
    for no benefit)."""
    no_decay_ids = set()
    for name, p in model.named_parameters():
        if p.ndim < 2 or name.startswith("context.out."):
            no_decay_ids.add(id(p))
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    no_decay = [p for p in model.parameters() if id(p) in no_decay_ids]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr_peak,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def _best_score(cfg: Config, metrics: dict) -> float | None:
    """Checkpoint-selection score, higher = better. None = metric unavailable this eval.

    "retrieval" (default): max holdout retrieval_acc — the acceptance metric.
    "consistency": min stratified holdout consistency median (negated here) —
    smoother, but scale-dependent and collapse-blind.
    "margin": max stratified median margin ratio d(nearest impostor)/d(own key)
    — the continuous, scale-free form of retrieval (>1 iff top-1 hit).
    In all modes the retrieval column stays the acceptance read."""
    if cfg.best_metric == "consistency":
        v = metrics.get("consistency_holdout_stratified")
        return -v if v is not None and not math.isnan(v) else None
    key = "margin_ratio_stratified" if cfg.best_metric == "margin" else "retrieval_acc"
    v = metrics.get(key)
    return v if v is not None and not math.isnan(v) else None


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup to lr_peak, cosine decay to lr_min. 1-based step."""
    if step <= cfg.warmup_steps:
        return cfg.lr_peak * step / cfg.warmup_steps
    t = min(1.0, max(0.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)))
    return cfg.lr_min + (cfg.lr_peak - cfg.lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def calibrate_output_scale(model: IdentityEncoder, ds: StockData, cfg: Config) -> dict:
    """Rescale the final d_model -> D projection, per dimension, so the population
    variance of z-bar — measured through the inference geometry (full view, single
    group, last K_inf windows) — starts at the v0 floor.

    Starts the geometry feasible: the separation/utilization hinges begin satisfied
    and act as fences. Without this, the init spread sits ~2 orders of magnitude
    inside the violation region (r1: var_dim ~0.005 vs v0 = 1) and the consistency
    terms' contraction wins the opening race."""
    model.eval()
    dev = next(model.parameters()).device
    jobs = [(ds.us, w) for w in ds.us.usable_windows[-cfg.K_inf:]]
    if cfg.cross_market:  # calibrate on the union population: the v0 floor must hold over all markets
        for mkt in ds.secondary:
            jobs += [(mkt, w) for w in mkt.usable_windows[-cfg.K_inf:]]
    acc: dict[int, list[torch.Tensor]] = {}
    for mkt, w in jobs:
        uni = mkt.train_universe(w)
        x = torch.from_numpy(mkt.feats_at(uni, w)).to(dev)
        H = model.temporal(x)
        rv = rv_from_feats(x) if model.context.relational else None
        Z = model.embed_rows(H, rv=rv)
        for k, t in enumerate(uni):
            acc.setdefault(int(t), []).append(Z[k])
    zbar = torch.stack([torch.stack(v).mean(0) for v in acc.values()])
    v_d = zbar.var(dim=0, unbiased=False)
    s = torch.sqrt(cfg.v0 / torch.clamp(v_d, min=1e-12)).clamp(max=1e3)
    model.context.out.weight.data.mul_(s[:, None])
    model.context.out.bias.data.mul_(s)
    v, sc = v_d.cpu().numpy(), s.cpu().numpy()
    return {
        "tickers": len(acc),
        "windows": [f"{m.name}:{w}" for m, w in jobs] if cfg.cross_market else [int(w) for _, w in jobs],
        "var_before": {"min": float(v.min()), "med": float(np.median(v)), "max": float(v.max())},
        "scale_applied": {"min": float(sc.min()), "med": float(np.median(sc)), "max": float(sc.max())},
    }


def _probe_sdpa_backend(dev: torch.device, L: int, heads: int, hd: int) -> dict:
    """One-time startup probe: which SDPA backends accept the context stack's
    grad-bearing float-mask call (aligned padded-slice mask, exactly what
    build_bias produces) at the largest group size L and at L+1. Efficient vs
    math decides ~GBs of saved softmax per scale-1 call — risk register #1;
    the result is printed and written to train_log as the step-0 record."""
    if dev.type != "cuda":
        return {"device": dev.type}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return {"device": "cuda", "probe": "unavailable (torch.nn.attention missing)"}
    out: dict = {"device": "cuda"}
    for Lp in (L, L + 1):
        Lpad = -(-Lp // 8) * 8
        q = torch.randn(1, heads, Lp, hd, device=dev)
        k = torch.randn(1, heads, Lp, hd, device=dev)
        v = torch.randn(1, heads, Lp, hd, device=dev)
        backing = torch.zeros(1, 1, Lp, Lpad, device=dev, requires_grad=True)
        m = backing[..., :Lp]
        res = {}
        for name, be in (
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
        ):
            try:
                with sdpa_kernel([be]):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=m).sum().backward()
                res[name] = True
            except RuntimeError:
                res[name] = False
            backing.grad = None
        out[f"L={Lp}"] = res
        del q, k, v, backing, m
    return out


@torch.no_grad()
def relational_diag(model: IdentityEncoder, ds: StockData, cfg: Config, a: int) -> dict:
    """REL-3 engagement meters on the first US job's scale-1 group (whole
    universe at span start `a`), through the deterministic eval path:

      edge_bias_abs — mean |capped bias| over pairs (read against the holdout
                      margin curves: the r10 detector);
      edge_sat_frac — fraction of pairs with |bias| > 0.9 * edge_cap;
      psn_share / edge_share — RMS of each raw channel vs the RMS of block-0's
                      post-temperature bilinear logits (channel attribution).

    no_grad, outside the checkpointed segments, at grad_diag_every cadence."""
    was_training = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        ctx = model.context
        cap = cfg.edge_cap
        uni = ds.us.universe_at(a)
        x = torch.from_numpy(ds.us.window_feats_at(uni, a)).to(dev)
        H = model.temporal(x)
        Hg = H[None]
        R, V = rv_from_feats(x)
        L = Hg.shape[1]
        planes = None
        if ctx.edge is not None:
            planes = compute_edge_planes(R[None], V[None], ctx._stats, cfg.edge_shrink_n0, cfg.edge_min_overlap)
        ab = ctx.psn.qk(Hg) if ctx.psn is not None else None
        chunk = max(1, cfg.edge_chunk)
        s_abs = s_sat = s_e2 = s_p2 = 0.0
        n_pairs = 0
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            e = ctx.edge(planes[:, c0:c1]) if planes is not None else None
            p = (
                ctx.psn.W2(F.gelu(ab[0][:, c0:c1, None, :] + ab[1][:, None, :, :]))
                if ab is not None
                else None
            )
            raw = (e if e is not None else 0) + (p if p is not None else 0)
            bias = cap * torch.tanh(raw / cap)
            s_abs += float(bias.abs().sum())
            s_sat += float((bias.abs() > 0.9 * cap).sum())
            if e is not None:
                s_e2 += float((e ** 2).sum())
            if p is not None:
                s_p2 += float((p ** 2).sum())
            n_pairs += bias.numel()
        # block-0 post-temperature bilinear logits (the denominator channel)
        blk = ctx.blocks[0]
        att = blk.attn
        qn = blk.ln1(Hg)
        q = att.wq(qn).view(1, L, att.h, att.hd).transpose(1, 2)
        k = att.wk(qn).view(1, L, att.h, att.hd).transpose(1, 2)
        if att.log_tau is not None:
            q = q * torch.exp(att.log_tau)[None, :, None, None]
        scale = att.hd ** -0.5
        s_l2 = 0.0
        n_l = 0
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            lg = (q[:, :, c0:c1] @ k.transpose(-2, -1)) * scale
            s_l2 += float((lg ** 2).sum())
            n_l += lg.numel()
        rms_logit = (s_l2 / max(n_l, 1)) ** 0.5
        out = {
            "L": int(L),
            "edge_bias_abs": s_abs / max(n_pairs, 1),
            "edge_sat_frac": s_sat / max(n_pairs, 1),
        }
        if ctx.edge is not None:
            out["edge_share"] = (s_e2 / max(n_pairs, 1)) ** 0.5 / (rms_logit + 1e-12)
        if ctx.psn is not None:
            out["psn_share"] = (s_p2 / max(n_pairs, 1)) ** 0.5 / (rms_logit + 1e-12)
        return out
    finally:
        model.train(was_training)


def draw_observer_dropout(real: torch.Tensor, p_attn: float) -> torch.Tensor:
    """Per-observer random peer-visibility masks. [B, L] -> [B, L, L] bool, no
    diagonal; rows left with zero visible peers are redrawn."""
    B, L = real.shape
    eye = torch.eye(L, dtype=torch.bool, device=real.device)
    cand = (real[:, :, None] & real[:, None, :]) & ~eye[None]
    keep = (torch.rand(B, L, L, device=real.device) >= p_attn) & cand
    bad = cand.any(-1) & ~keep.any(-1)
    while bool(bad.any()):
        redraw = (torch.rand(B, L, L, device=real.device) >= p_attn) & cand
        keep = torch.where(bad[:, :, None], redraw, keep)
        bad = cand.any(-1) & ~keep.any(-1)
    return keep


def _scatter_rows(src, order_t, gid_t, pos_t, B, L):
    """Rows -> padded groups: [U, ...] -> [B, L, ...] (zeros at pad slots)."""
    out = src.new_zeros(B, L, *src.shape[1:])
    out[gid_t, pos_t] = src[order_t]
    return out


def _context_views_segment(ctx, H, R, V, order_t, gid_t, pos_t, real, vis_peers, *ff_masks):
    """Checkpoint segment for one (window, scale): group scatter + three views +
    row gather. Contains NO RNG — the dropout masks enter as tensor arguments,
    and the relational bias (planes/PairScore/tanh) is a deterministic function
    of (H, R, V) — so backward recompute is deterministic and gradient-exact.
    Only the [rows, D] outputs (and the argument tensors — R/V are per-window
    and shared across this window's segments) survive the step; do not add
    stochastic ops in here."""
    B, L = real.shape
    Hg = _scatter_rows(H, order_t, gid_t, pos_t, B, L)
    rv = None
    if ctx.relational:
        rv = (
            _scatter_rows(R, order_t, gid_t, pos_t, B, L),
            _scatter_rows(V, order_t, gid_t, pos_t, B, L),
        )
    z_full, z_self, z_peer = ctx.views(Hg, vis_peers, real, list(ff_masks) or None, rv=rv)
    return z_full[gid_t, pos_t], z_self[gid_t, pos_t], z_peer[gid_t, pos_t]


def _sync_mirrors(model: IdentityEncoder, mirrors: list) -> None:
    """Copy master weights into each device mirror (the model has no buffers,
    so the parameters are the complete state; ~4 MB per mirror)."""
    with torch.no_grad():
        for mir in mirrors:
            for pm, pr in zip(model.parameters(), mir.parameters()):
                pr.copy_(pm, non_blocking=True)


def _merge_mirror_grads(model: IdentityEncoder, mirrors: list, dev: torch.device) -> None:
    """Sum each mirror's gradients into the master's. Must run after backward and
    BEFORE clip_grad_norm_: the global clip has to see the full cross-device
    gradient (by the weight-sharing identity, master+mirror grads at equal
    weights sum to the single-device gradient exactly)."""
    for mir in mirrors:
        for pm, pr in zip(model.parameters(), mir.parameters()):
            if pr.grad is not None:
                g = pr.grad.to(dev)
                pm.grad = g if pm.grad is None else pm.grad.add_(g)
                pr.grad = None


def run_step(replicas: list, devs: list, ds: StockData, draws: list[tuple[int, int]], cfg: Config, offset: int = 0) -> StepStore:
    """One step's forward pass: encode each drawn window once, then group and
    run the three views per scale. `offset` shifts the whole tiling (one value
    per sampler epoch); 0 = the fixed tiling. `replicas[i]` is the model copy on
    `devs[i]`; replicas[0] is the master (single-device mode: just [model]).

    Cross-market mode: each drawn US window also derives one window per
    secondary market via the per-day dominance rule (no secondary day k may
    fall after US day k — see MarketData.dominated_start). Derived windows
    co-reside in the step, so the population terms (xsep, util, the syn/EMA
    pooling) span the union, while attention groups stay single-market. The
    derived window reuses its US partner's ordinal as t_w: kappa pairs are
    within-ticker (hence within-market), and the derived span trails its
    partner by far less than one window unit, so era distances stay correct.

    Multi-device placement is a fixed slot rule — US window slot s on device
    (s % D), its derived market-mi partner on ((s + mi) % D), anti-pairing the
    two largest covarying jobs — and every job's embedding rows are gathered to
    devs[0] mid-graph (`.to` is a no-op on devs[0] jobs), where the one global
    loss is computed. Gradients route back through the copy nodes."""
    store = StepStore()
    D = len(replicas)
    jobs: list[tuple[int, float, object, int, "np.ndarray", tuple]] = []
    n_sec = {m.name: 0 for m in ds.secondary}
    for slot, (w, visit) in enumerate(draws):
        a = ds.us.shifted_start(w, offset)
        uni = ds.us.universe_at(a)
        if len(uni) < cfg.Y:  # shifted span dipped below minimum group size: fall back to the base span
            a = ds.us.base_start(w)
            uni = ds.us.train_universe(w)
        jobs.append((slot, float(w), ds.us, a, uni, (w, visit)))
        if cfg.cross_market:
            us_dates = ds.us.grid.values[a : a + cfg.N]
            for mi, mkt in enumerate(ds.secondary, start=1):
                ca = mkt.dominated_start(us_dates)
                if ca is None:  # the US window predates this market's history
                    continue
                cuni = mkt.universe_at(ca)
                if len(cuni) < cfg.Y:
                    continue
                # partition seed gets the market index so secondary draws never
                # mirror the US stream; US keeps (w, visit) for replay compat
                jobs.append((mi * len(draws) + slot, float(w), mkt, ca, cuni, (w, visit, mi)))
                n_sec[mkt.name] += 1
    store.market_windows = n_sec
    # partitions drawn up front: each draw_partitions call seeds its own rng, so
    # hoisting them out of the compute loop is draw-identical — and it keeps
    # host-side numpy from starving the device queues in multi-device mode
    parts_by_job = [draw_partitions(len(uni), ladder(len(uni), cfg.Y), seed=seed) for *_, uni, seed in jobs]
    for (slot, t_w, mkt, a, uni, _), parts in zip(jobs, parts_by_job):
        # device offset cycles through 1..D-1 for derived partners: a US window
        # and any of its derived secondary jobs never share a device (markets may
        # share one with each other once #markets >= D — unavoidable by pigeonhole)
        mi = slot // len(draws)
        off = 0 if mi == 0 else 1 + (mi - 1) % max(D - 1, 1)
        di = (slot % len(draws) + off) % D
        mdl, dev = replicas[di], devs[di]
        x = torch.from_numpy(mkt.window_feats_at(uni, a)).to(dev)
        H = mdl.encode_window(x, grad_checkpoint=cfg.grad_checkpoint)
        R, V = rv_from_feats(x)  # per-window; every scale's segment shares these tensors
        for n_s, groups in parts.items():
            lengths = np.array([len(g) for g in groups])
            g_max = int(lengths.max())
            order = np.concatenate(groups)
            gid = np.repeat(np.arange(len(groups)), lengths)
            pos = np.concatenate([np.arange(l) for l in lengths])
            order_t = torch.from_numpy(order).to(dev)
            gid_t = torch.from_numpy(gid).to(dev)
            pos_t = torch.from_numpy(pos).to(dev)

            real = torch.zeros(len(groups), g_max, dtype=torch.bool, device=dev)
            real[gid_t, pos_t] = True

            if mdl.training and cfg.p_attn > 0:
                vis_peers = draw_observer_dropout(real, cfg.p_attn)
            else:
                eye = torch.eye(g_max, dtype=torch.bool, device=dev)
                vis_peers = (real[:, :, None] & real[:, None, :]) & ~eye[None]
            ff_masks = None
            if mdl.training and cfg.p_ff > 0:
                ff_masks = [
                    (torch.rand(len(groups), g_max, cfg.d_ff, device=dev) >= cfg.p_ff).float() / (1.0 - cfg.p_ff)
                    for _ in range(cfg.L_ctx)
                ]

            if cfg.context_checkpoint and mdl.training:
                zf, zs, zp = checkpoint(
                    _context_views_segment, mdl.context, H, R, V, order_t, gid_t, pos_t,
                    real, vis_peers, *(ff_masks or ()), use_reentrant=False,
                )
            else:
                B_g, L_g = real.shape
                Hg = _scatter_rows(H, order_t, gid_t, pos_t, B_g, L_g)
                rv = None
                if mdl.context.relational:
                    rv = (
                        _scatter_rows(R, order_t, gid_t, pos_t, B_g, L_g),
                        _scatter_rows(V, order_t, gid_t, pos_t, B_g, L_g),
                    )
                z_full, z_self, z_peer = mdl.context.views(Hg, vis_peers, real, ff_masks, rv=rv)
                zf, zs, zp = z_full[gid_t, pos_t], z_self[gid_t, pos_t], z_peer[gid_t, pos_t]
            store.add(
                tickers=uni[order],
                slot=slot,
                t_w=t_w,
                scale=n_s,
                gsizes=lengths[gid],
                group_ids=gid,
                z_full=zf.to(devs[0]),
                z_self=zs.to(devs[0]),
                z_peer=zp.to(devs[0]),
            )
    return store


def _resume_config_diffs(cfg: Config, ck_config: dict) -> list[str]:
    """Run-permanent fields that differ between the resuming config and the
    checkpoint's (KNOB_FIELDS are exempt). Values compare in canonical JSON
    form — the checkpoint already stores its config that way. Fields an older
    checkpoint lacks compare against today's defaults, the best available
    proxy for what the old code did."""
    ck_config = dict(ck_config)
    if "cons_eval_windows" in ck_config and "strat_eval_windows" not in ck_config:  # pre-rename configs
        ck_config["strat_eval_windows"] = ck_config.pop("cons_eval_windows")
    now = json.loads(json.dumps(asdict(cfg)))
    diffs = []
    for name, fld in Config.__dataclass_fields__.items():
        if name in KNOB_FIELDS:
            continue
        if name in ck_config:
            old = ck_config[name]
        else:
            old = fld.default if fld.default is not MISSING else fld.default_factory()
            old = json.loads(json.dumps(old))
        if old != now[name]:
            diffs.append(f"  {name}: checkpoint={old!r} run={now[name]!r}")
    return diffs


def save_checkpoint(path: Path, step: int, model, opt, anchors, norm, sampler, cfg: Config, ds_meta: dict, eval_metrics: dict | None, best_score: float):
    """Full training state: weights, optimizer, EMA normalizers, anchors, stratum
    queues, visit counters, RNG state, and the running best selection score (so
    a resumed run continues the best.pt comparison instead of restarting it at
    -inf). Visit counters are load-bearing — partition draws are seeded by
    (window id, visit counter)."""
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "anchors": anchors.state_dict(),
        "ema_normalizers": norm.state_dict(),
        "sampler": sampler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "config": json.loads(json.dumps(cfg.__dict__)),
        "data_meta": ds_meta,
        "eval_metrics": eval_metrics,
        "best_score": float(best_score),
    }
    if next(model.parameters()).is_cuda:
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(ck: dict, model, opt, anchors, norm, sampler) -> int:
    """Restore full training state from an already-loaded checkpoint dict
    (torch.load with map_location="cpu" — portable across machines)."""
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])  # optimizer state is recast to the params' device
    anchors.load_state_dict(ck["anchors"])
    norm.load_state_dict(ck["ema_normalizers"])
    sampler.load_state_dict(ck["sampler"])
    torch.set_rng_state(ck["torch_rng"])
    if "cuda_rng" in ck and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        except RuntimeError as e:  # e.g. checkpoint from a box with a different GPU count
            print(f"[resume] cuda RNG not restored ({e}); dropout streams will differ")
    return ck["step"]


def train(cfg: Config, resume: str | None = None) -> Path:
    if cfg.best_metric not in ("retrieval", "consistency", "margin"):
        raise ValueError(f"unknown best_metric {cfg.best_metric!r}")
    if cfg.loss_chunk < 0:
        raise ValueError(f"loss_chunk must be >= 0 (0 = monolithic), got {cfg.loss_chunk}")
    ck = None
    if resume:
        ck = torch.load(Path(resume), weights_only=False, map_location="cpu")  # portable across machines
        diffs = _resume_config_diffs(cfg, ck["config"])
        if diffs:
            raise ValueError(
                "refusing to resume: run-permanent config differs from the checkpoint's:\n"
                + "\n".join(diffs)
                + "\nOnly knobs ("
                + ", ".join(sorted(KNOB_FIELDS))
                + ") may change on resume; start from the run's own config "
                "(--config <run_dir>/config.json) or begin a fresh run."
            )
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.train_seed)
    devs = [torch.device(d) for d in cfg.resolved_devices()]
    dev = devs[0]  # primary: master model, loss, anchors, optimizer, eval
    if any(d.type == "cuda" for d in devs):
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    run_dir = REPO_ROOT / cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    ds = StockData(cfg)
    meta = ds.summary()
    if ck is not None:
        # data identity: the rebuilt dataset must match the checkpoint's snapshot
        # (catches changed raw files even when every config field agrees)
        old_meta = ck["data_meta"]
        bad = sorted(k for k in set(old_meta) | set(meta) if old_meta.get(k) != meta.get(k))
        if bad:
            detail = "\n".join(f"  {k}: checkpoint={old_meta.get(k)!r} run={meta.get(k)!r}" for k in bad)
            raise ValueError(
                "refusing to resume: the dataset no longer matches the checkpoint's data_meta "
                "(raw data changed underneath the run):\n" + detail
            )
    print(f"[device] {', '.join(str(d) for d in devs)}")
    print(f"[data] {json.dumps({k: v for k, v in meta.items() if k != 'holdout'})}")
    print(f"[data] holdout ({meta['holdout_size']}): {' '.join(meta['holdout'])}")
    (run_dir / "data_meta.json").write_text(json.dumps(meta, indent=2))

    model = IdentityEncoder(cfg).to(dev)
    # mirrors under a forked RNG: building them must not perturb the master's
    # init stream, so single- and multi-device fresh runs start from identical
    # weights; mirror weights are overwritten by _sync_mirrors every step anyway
    with torch.random.fork_rng(devices=[]):
        mirrors = [IdentityEncoder(cfg).to(d) for d in devs[1:]]
    replicas = [model] + mirrors
    if cfg.calibrate_init and not resume:
        calib = calibrate_output_scale(model, ds, cfg)
        (run_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
        print(f"[calibrate] {json.dumps(calib)}")
    opt = build_optimizer(model, cfg)
    anchors = AnchorState(ds.T, cfg.D, cfg, device=dev)
    norm = EmaNormalizer(cfg.beta, cfg.eps, cfg.kappa_floor)
    sampler = StratumSampler(
        ds.usable_windows, cfg.M, cfg.W, np.random.default_rng(cfg.train_seed),
        offset_range=cfg.N if cfg.window_offset else 0,
    )

    start_step = 0
    best_score = -math.inf
    if ck is not None:
        start_step = load_checkpoint(ck, model, opt, anchors, norm, sampler)
        best_score = float(ck.get("best_score", -math.inf))  # legacy checkpoints: no score recorded
        print(f"[resume] from {resume} at step {start_step} (best_score={best_score:.6g})")

    sdpa_probe = None
    if model.context.relational:
        Lmax = max(len(m.train_universe(w)) for m in ds.markets.values() for w in m.usable_windows)
        hpg = cfg.resolved_n_heads_ctx() // cfg.edge_head_groups
        sdpa_probe = _probe_sdpa_backend(dev, Lmax, hpg, cfg.d_model // cfg.resolved_n_heads_ctx())
        print(f"[sdpa_ctx_backend] {json.dumps(sdpa_probe)}")

    log_f = open(run_dir / "train_log.jsonl", "a")
    eval_f = open(run_dir / "eval_log.jsonl", "a")
    if sdpa_probe is not None:
        log_f.write(json.dumps({"step": start_step, "sdpa_ctx_backend": sdpa_probe}) + "\n")
        log_f.flush()

    cuda_devs = [d for d in devs if d.type == "cuda"]

    for step in range(start_step + 1, cfg.max_steps + 1):
        t0 = time.time()
        for d in cuda_devs:
            torch.cuda.reset_peak_memory_stats(d)
        model.train()
        if mirrors:
            for mir in mirrors:
                mir.train()
            _sync_mirrors(model, mirrors)  # post-opt.step() weights from the previous step
        draws = sampler.draw()  # may roll the epoch: offset is fixed after this call until the next refill
        store = run_step(replicas, devs, ds, draws, cfg, offset=sampler.offset)
        batch = store.finalize(device=dev)
        idx = StepIndex(batch, cfg, ds.tau_prox)
        terms, I, I_per_ticker, present, zbar, spectrum = step_losses(cfg, batch, idx, ds.T, anchors)
        total, normalized = combine(cfg, terms, norm)
        if total is None:
            raise RuntimeError(f"no loss term computable at step {step} (windows {draws})")
        force = None
        if cfg.grad_diag_every and step % cfg.grad_diag_every == 0:
            force = grad_force_diag(cfg, terms, norm, batch, total)  # before backward: needs the graph

        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total.backward()
        _merge_mirror_grads(model, mirrors, dev)  # no-op single-device; must precede the global clip
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
        anchor_stats = anchors.update(present, zbar, I_per_ticker)

        rec = {
            "step": step,
            "lr": lr,
            "offset": sampler.offset,
            "total": float(total.detach()),
            "grad_norm": float(gn),
            "secs": round(time.time() - t0, 2),
            "raw": {k: (float(v.detach()) if v is not None else None) for k, v in terms.items()},
            "normalized": normalized,
            "I": {k: float(v.detach()) for k, v in I.items()},
            **anchor_stats,
        }
        if spectrum is not None:
            s = spectrum.cpu().numpy()
            rec["var_dim"] = {"min": float(s.min()), "med": float(np.median(s)), "max": float(s.max())}
            if step % cfg.eval_every == 0:  # full collapse-watch spectrum, at eval cadence
                rec["var_spectrum"] = [round(float(x), 6) for x in np.sort(s)]
        with torch.no_grad():  # collapse watch: pairwise distances of the z-bar population
            zd = torch.pdist(zbar)
        if zd.numel():
            rec["zbar_dist"] = {"mean": float(zd.mean()), "min": float(zd.min())}
        if cfg.cross_market:
            # cross-market watch: how many secondary windows landed this step, the
            # US/secondary centroid gap (market-as-axis detector), per-market spread
            rec["windows_mkt"] = store.market_windows
            with torch.no_grad():
                mid = torch.from_numpy(ds.market_id[present.cpu().numpy()].astype(np.int64)).to(zbar.device)
                z_us, z_sec = zbar[mid == 0], zbar[mid != 0]
                if len(z_us) and len(z_sec):
                    rec["market_centroid_dist"] = float((z_us.mean(0) - z_sec.mean(0)).norm())
                    rec["var_med_us"] = float(z_us.var(dim=0, unbiased=False).median())
                    rec["var_med_sec"] = float(z_sec.var(dim=0, unbiased=False).median())
        if force is not None:
            rec["force"] = force
            rec["denom"] = {k: float(v) for k, v in norm.last_denom.items()}
            if model.context.relational:
                # engagement meters on the first US draw's scale-1 group, at the
                # same span run_step used (offset + below-Y fallback replicated)
                w0 = draws[0][0]
                a0 = ds.us.shifted_start(w0, sampler.offset)
                if len(ds.us.universe_at(a0)) < cfg.Y:
                    a0 = ds.us.base_start(w0)
                rec["rel"] = relational_diag(model, ds, cfg, a0)
        if cuda_devs:
            rec["mem_gb"] = {str(d): round(torch.cuda.max_memory_allocated(d) / 1e9, 3) for d in cuda_devs}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % 10 == 0 or step == start_step + 1:
            print(
                f"step {step:5d} total={rec['total']:.3f} "
                f"I_full={rec['I'].get('full', float('nan')):.4g} I_self={rec['I'].get('self', float('nan')):.4g} "
                f"I_peer={rec['I'].get('peer', float('nan')):.4g} vmin={rec.get('var_dim', {}).get('min', float('nan')):.3g} "
                f"dz={rec.get('zbar_dist', {}).get('mean', float('nan')):.3g} "
                f"({rec['secs']}s)"
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            if cfg.eval_parallel and mirrors:
                _sync_mirrors(model, mirrors)  # mirrors are one opt.step stale at this point
            metrics = run_eval(model, ds, cfg, replicas=replicas)
            metrics["step"] = step
            eval_f.write(json.dumps(metrics) + "\n")
            eval_f.flush()
            print(f"[eval @ {step}] {json.dumps(metrics)}")
            score = _best_score(cfg, metrics)
            is_best = False
            if score is not None and score >= best_score:
                best_score = score  # fold in BEFORE latest.pt so resuming from it can't demote best.pt
                is_best = True
            save_checkpoint(run_dir / "latest.pt", step, model, opt, anchors, norm, sampler, cfg, meta, metrics, best_score)
            if is_best:
                save_checkpoint(run_dir / "best.pt", step, model, opt, anchors, norm, sampler, cfg, meta, metrics, best_score)

    if not (run_dir / "latest.pt").exists():
        save_checkpoint(run_dir / "latest.pt", cfg.max_steps, model, opt, anchors, norm, sampler, cfg, meta, None, best_score)
    log_f.close()
    eval_f.close()
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Train the Stock Identity Encoder")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--config", default=None, help="JSON config to start from")
    ap.add_argument("--device", default=None, help='"auto" (default), "cpu", "cuda", or "cuda:N"')
    ap.add_argument(
        "--best-metric",
        default=None,
        choices=["retrieval", "consistency", "margin"],
        help='best.pt selection: "retrieval" (default; max holdout retrieval_acc), '
        '"consistency" (min holdout consistency median over windows stratified like training), or '
        '"margin" (max stratified median of d(nearest impostor)/d(own key) — continuous retrieval)',
    )
    ap.add_argument(
        "--strat-eval-windows", type=int, default=None,
        help='size of the stratified window set used by the "consistency" and "margin" selection modes',
    )
    ap.add_argument("--data-format", default=None, choices=["csv", "parquet"], help="input backend (default csv)")
    ap.add_argument("--parquet-dir", default=None, help="dir of long-format OHLCV parquet shards (data-format=parquet)")
    ap.add_argument("--exchanges", default=None, help="parquet: comma-separated exchanges to keep, or 'all' for no filter")
    ap.add_argument("--min-history-days", type=int, default=None, help="drop tickers with fewer usable grid-days (0 disables; default 252 for parquet, 0 for csv)")
    ap.add_argument(
        "--cross-market", action="store_true",
        help="include secondary markets (e.g. China) on their own calendars (parquet only); "
        "secondary windows are derived from US windows by the per-day dominance rule and "
        "co-reside in the training step, with attention groups staying single-market",
    )
    ap.add_argument(
        "--calendar-validity", default=None, choices=["rows", "traded"],
        help='quorum-grid candidate counting: "rows" (historical: raw rows, admits vendor placeholder '
        'days) or "traded" (valid traded bars only — the feed\'s own encoding of "market open")',
    )
    ap.add_argument(
        "--calendar-frac", type=float, default=None,
        help="drop candidate grid dates below this fraction of the rolling-max market size "
        "(closure/partial-delivery guard; 0 = off, recommended 0.5 with --calendar-validity traded)",
    )
    ap.add_argument("--calendar-frac-window", type=int, default=None,
                    help="rolling-max window for --calendar-frac, in candidate dates (default 121)")
    ap.add_argument(
        "--halt-markets", default=None,
        help='comma-separated markets ingested halt-tolerantly (e.g. "CN"): carried bars admitted as '
        'halted days, missing-row halts synthesized within --max-ffill-days of the last traded bar; "" disables',
    )
    ap.add_argument("--halt-minfrac", type=float, default=None,
                    help="minimum traded fraction per complete window in halt markets (default 0.9)")
    ap.add_argument("--max-ffill-days", type=int, default=None,
                    help="gap-synthesis cap in calendar days from the last traded bar (default 14)")
    ap.add_argument(
        "--holdout-pool-windows", type=int, default=None,
        help="secondary markets draw their holdout from tickers complete in the newest K windows "
        "(>= 2; the primary market always keeps the all-windows rule; default: all windows)",
    )
    ap.add_argument(
        "--context-checkpoint", action="store_true",
        help="recompute each (window, scale) context pass during backward (exact: identical "
        "gradients and draws; cuts step activation memory ~15x at parquet scale for ~+30%% time)",
    )
    ap.add_argument(
        "--loss-chunk", type=int, default=None,
        help="compute the O(n^2) separation terms (xsep/psep) in checkpointed chunks of this many "
        "rows (1024 recommended at parquet scale; same sums, reduction order aside); 0 = monolithic",
    )
    ap.add_argument(
        "--devices", default=None,
        help='comma-separated devices for single-RUN job parallelism, e.g. "cuda:0,cuda:1": jobs '
        "are placed by a fixed slot rule and the ONE global loss is computed on the first device "
        "(never sharded); composable with, but independent of, --context-checkpoint",
    )
    ap.add_argument(
        "--eval-parallel", action="store_true",
        help="split eval windows across --devices (eval is deterministic and per-window "
        "independent: byte-identical metrics on identical device types, ~2x faster evals)",
    )
    ap.add_argument("--no-window-offset", action="store_true", help="train on the fixed tiling only (disable per-epoch offsets)")
    ap.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="keep temporal-encoder activations in memory (faster; fine on a GPU with headroom)",
    )
    # --- REL-3 relational context (architecture fields: run-permanent, fresh run required) ---
    ap.add_argument("--L-ctx", dest="L_ctx", type=int, default=None,
                    help="context-module depth (default 2; REL-3 runs use 3)")
    ap.add_argument("--n-heads-ctx", type=int, default=None,
                    help="context-only head count (default: n_heads; REL-3 runs use 8; zero params)")
    ap.add_argument("--attn-temp-ctx", action="store_true",
                    help="per-block per-head learned log-temperature on context attention (REL-3; zero-init = x1)")
    ap.add_argument("--edge-stats", default=None,
                    help='comma-separated relational bias planes, e.g. "corr0,dlogvol" (the REL-3 run '
                    'set; "beta_prod" supported but opt-in — see config.py); "" disables the edge module')
    ap.add_argument("--psn-rank", type=int, default=None,
                    help="PairScore non-bilinear pair-scorer ridge width (default 0 = off; REL-3 runs use 16)")
    ap.add_argument("--edge-chunk", type=int, default=None,
                    help="row-chunk for the checkpointed bias MLPs (operational knob, value-exact; default 512)")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.run_dir:
        cfg.run_dir = args.run_dir
    for k in ("max_steps", "eval_every", "warmup_steps", "device", "best_metric",
              "strat_eval_windows", "data_format", "parquet_dir", "min_history_days",
              "loss_chunk", "devices", "calendar_validity", "calendar_frac",
              "calendar_frac_window", "halt_minfrac", "max_ffill_days", "holdout_pool_windows",
              "L_ctx", "n_heads_ctx", "psn_rank", "edge_chunk"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.halt_markets is not None:
        cfg.halt_markets = tuple(s.strip() for s in args.halt_markets.split(",") if s.strip())
    if args.edge_stats is not None:
        cfg.edge_stats = tuple(s.strip() for s in args.edge_stats.split(",") if s.strip())
    if args.attn_temp_ctx:
        cfg.attn_temp_ctx = True
    if args.context_checkpoint:
        cfg.context_checkpoint = True
    if args.eval_parallel:
        cfg.eval_parallel = True
    if args.exchanges is not None:
        cfg.exchanges = None if args.exchanges.strip().lower() == "all" else tuple(
            s.strip() for s in args.exchanges.split(",") if s.strip()
        )
    if args.cross_market:
        cfg.cross_market = True
    if args.no_grad_checkpoint:
        cfg.grad_checkpoint = False
    if args.no_window_offset:
        cfg.window_offset = False
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
