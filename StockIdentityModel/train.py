"""Training loop.

Run from the repo root:
    python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, REPO_ROOT
from .data import StockData, ladder
from .evaluate import run_eval
from .losses import AnchorState, EmaNormalizer, StepIndex, StepStore, combine, grad_force_diag, step_losses
from .model import IdentityEncoder
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
    wins = ds.usable_windows[-cfg.K_inf:]
    acc: dict[int, list[torch.Tensor]] = {}
    for w in wins:
        uni = ds.train_universe(w)
        H = model.temporal(torch.from_numpy(ds.feats[uni, w]).to(dev))
        Z = model.embed_rows(H)
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
        "windows": [int(w) for w in wins],
        "var_before": {"min": float(v.min()), "med": float(np.median(v)), "max": float(v.max())},
        "scale_applied": {"min": float(sc.min()), "med": float(np.median(sc)), "max": float(sc.max())},
    }


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


def run_step(model: IdentityEncoder, ds: StockData, draws: list[tuple[int, int]], cfg: Config, offset: int = 0) -> StepStore:
    """One step's forward pass: encode each drawn window once, then group and
    run the three views per scale. `offset` shifts the whole tiling (one value
    per sampler epoch); 0 = the fixed tiling."""
    store = StepStore()
    dev = next(model.parameters()).device
    for slot, (w, visit) in enumerate(draws):
        a = ds.shifted_start(w, offset)
        uni = ds.universe_at(a)
        if len(uni) < cfg.Y:  # shifted span dipped below minimum group size: fall back to the base span
            a = ds.base_start(w)
            uni = ds.train_universe(w)
        U = len(uni)
        x = torch.from_numpy(ds.window_feats_at(uni, a)).to(dev)
        H = model.encode_window(x, grad_checkpoint=cfg.grad_checkpoint)
        parts = draw_partitions(U, ladder(U, cfg.Y), seed=(w, visit))
        for n_s, groups in parts.items():
            lengths = np.array([len(g) for g in groups])
            g_max = int(lengths.max())
            order = np.concatenate(groups)
            gid = np.repeat(np.arange(len(groups)), lengths)
            pos = np.concatenate([np.arange(l) for l in lengths])
            order_t = torch.from_numpy(order).to(dev)
            gid_t = torch.from_numpy(gid).to(dev)
            pos_t = torch.from_numpy(pos).to(dev)

            Hg = H.new_zeros(len(groups), g_max, H.shape[1])
            Hg[gid_t, pos_t] = H[order_t]
            real = torch.zeros(len(groups), g_max, dtype=torch.bool, device=dev)
            real[gid_t, pos_t] = True

            if model.training and cfg.p_attn > 0:
                vis_peers = draw_observer_dropout(real, cfg.p_attn)
            else:
                eye = torch.eye(g_max, dtype=torch.bool, device=dev)
                vis_peers = (real[:, :, None] & real[:, None, :]) & ~eye[None]
            ff_masks = None
            if model.training and cfg.p_ff > 0:
                ff_masks = [
                    (torch.rand(len(groups), g_max, cfg.d_ff, device=dev) >= cfg.p_ff).float() / (1.0 - cfg.p_ff)
                    for _ in range(cfg.L_ctx)
                ]

            z_full, z_self, z_peer = model.context.views(Hg, vis_peers, real, ff_masks)
            store.add(
                tickers=uni[order],
                slot=slot,
                t_w=float(w),
                scale=n_s,
                gsizes=lengths[gid],
                group_ids=gid,
                z_full=z_full[gid_t, pos_t],
                z_self=z_self[gid_t, pos_t],
                z_peer=z_peer[gid_t, pos_t],
            )
    return store


def save_checkpoint(path: Path, step: int, model, opt, anchors, norm, sampler, cfg: Config, ds_meta: dict, eval_metrics: dict | None):
    """Full training state: weights, optimizer, EMA normalizers, anchors, stratum
    queues, visit counters, RNG state. Visit counters are load-bearing — partition
    draws are seeded by (window id, visit counter)."""
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
    }
    if next(model.parameters()).is_cuda:
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path: Path, model, opt, anchors, norm, sampler) -> int:
    ck = torch.load(path, weights_only=False, map_location="cpu")  # portable across machines
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
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.train_seed)
    dev = torch.device(cfg.resolved_device())
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    run_dir = REPO_ROOT / cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    ds = StockData(cfg)
    meta = ds.summary()
    print(f"[device] {dev}")
    print(f"[data] {json.dumps({k: v for k, v in meta.items() if k != 'holdout'})}")
    print(f"[data] holdout ({meta['holdout_size']}): {' '.join(meta['holdout'])}")
    (run_dir / "data_meta.json").write_text(json.dumps(meta, indent=2))

    model = IdentityEncoder(cfg).to(dev)
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
    if resume:
        start_step = load_checkpoint(Path(resume), model, opt, anchors, norm, sampler)
        print(f"[resume] from {resume} at step {start_step}")

    log_f = open(run_dir / "train_log.jsonl", "a")
    eval_f = open(run_dir / "eval_log.jsonl", "a")
    best_score = -math.inf

    for step in range(start_step + 1, cfg.max_steps + 1):
        t0 = time.time()
        model.train()
        draws = sampler.draw()  # may roll the epoch: offset is fixed after this call until the next refill
        store = run_step(model, ds, draws, cfg, offset=sampler.offset)
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
        if force is not None:
            rec["force"] = force
            rec["denom"] = {k: float(v) for k, v in norm.last_denom.items()}
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
            metrics = run_eval(model, ds, cfg)
            metrics["step"] = step
            eval_f.write(json.dumps(metrics) + "\n")
            eval_f.flush()
            print(f"[eval @ {step}] {json.dumps(metrics)}")
            save_checkpoint(run_dir / "latest.pt", step, model, opt, anchors, norm, sampler, cfg, meta, metrics)
            score = _best_score(cfg, metrics)
            if score is not None and score >= best_score:
                best_score = score
                save_checkpoint(run_dir / "best.pt", step, model, opt, anchors, norm, sampler, cfg, meta, metrics)

    if not (run_dir / "latest.pt").exists():
        save_checkpoint(run_dir / "latest.pt", cfg.max_steps, model, opt, anchors, norm, sampler, cfg, meta, None)
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
    ap.add_argument(
        "--adapter-blocks", type=int, default=None,
        help="seam-A ResidualMLP blocks at the context module entry (default 0 = the r7 architecture)",
    )
    ap.add_argument(
        "--head-blocks", type=int, default=None,
        help="seam-B ResidualMLP blocks before the final projection (default 0 = the r7 architecture)",
    )
    ap.add_argument("--no-window-offset", action="store_true", help="train on the fixed tiling only (disable per-epoch offsets)")
    ap.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="keep temporal-encoder activations in memory (faster; fine on a GPU with headroom)",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.run_dir:
        cfg.run_dir = args.run_dir
    for k in ("max_steps", "eval_every", "warmup_steps", "device", "best_metric",
              "strat_eval_windows", "adapter_blocks", "head_blocks"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_grad_checkpoint:
        cfg.grad_checkpoint = False
    if args.no_window_offset:
        cfg.window_offset = False
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
