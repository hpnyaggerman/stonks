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
from .losses import AnchorState, EmaNormalizer, StepIndex, StepStore, combine, step_losses
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


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup to lr_peak, cosine decay to lr_min. 1-based step."""
    if step <= cfg.warmup_steps:
        return cfg.lr_peak * step / cfg.warmup_steps
    t = min(1.0, max(0.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)))
    return cfg.lr_min + (cfg.lr_peak - cfg.lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


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


def run_step(model: IdentityEncoder, ds: StockData, draws: list[tuple[int, int]], cfg: Config) -> StepStore:
    """One step's forward pass: encode each drawn window once, then group and
    run the three views per scale."""
    store = StepStore()
    dev = next(model.parameters()).device
    feats_t = ds.feats  # numpy [T, F, N, 5]
    for slot, (w, visit) in enumerate(draws):
        uni = ds.train_universe(w)
        U = len(uni)
        x = torch.from_numpy(feats_t[uni, w]).to(dev)
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
    opt = build_optimizer(model, cfg)
    anchors = AnchorState(ds.T, cfg.D, cfg, device=dev)
    norm = EmaNormalizer(cfg.beta, cfg.eps)
    sampler = StratumSampler(ds.usable_windows, cfg.M, cfg.W, np.random.default_rng(cfg.train_seed))

    start_step = 0
    if resume:
        start_step = load_checkpoint(Path(resume), model, opt, anchors, norm, sampler)
        print(f"[resume] from {resume} at step {start_step}")

    log_f = open(run_dir / "train_log.jsonl", "a")
    eval_f = open(run_dir / "eval_log.jsonl", "a")
    best_retrieval = -1.0

    for step in range(start_step + 1, cfg.max_steps + 1):
        t0 = time.time()
        model.train()
        draws = sampler.draw()
        store = run_step(model, ds, draws, cfg)
        batch = store.finalize(device=dev)
        idx = StepIndex(batch, cfg, ds.tau_prox)
        terms, I, I_per_ticker, present, zbar, spectrum = step_losses(cfg, batch, idx, ds.T, anchors)
        total, normalized = combine(cfg, terms, norm)
        if total is None:
            raise RuntimeError(f"no loss term computable at step {step} (windows {draws})")

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
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % 10 == 0 or step == start_step + 1:
            print(
                f"step {step:5d} total={rec['total']:.3f} "
                f"I_full={rec['I'].get('full', float('nan')):.4g} I_self={rec['I'].get('self', float('nan')):.4g} "
                f"I_peer={rec['I'].get('peer', float('nan')):.4g} vmin={rec.get('var_dim', {}).get('min', float('nan')):.3g} "
                f"({rec['secs']}s)"
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            metrics = run_eval(model, ds, cfg)
            metrics["step"] = step
            eval_f.write(json.dumps(metrics) + "\n")
            eval_f.flush()
            print(f"[eval @ {step}] {json.dumps(metrics)}")
            save_checkpoint(run_dir / "latest.pt", step, model, opt, anchors, norm, sampler, cfg, meta, metrics)
            if metrics.get("retrieval_acc", -1) >= best_retrieval:
                best_retrieval = metrics["retrieval_acc"]
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
        "--no-grad-checkpoint",
        action="store_true",
        help="keep temporal-encoder activations in memory (faster; fine on a GPU with headroom)",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.run_dir:
        cfg.run_dir = args.run_dir
    for k in ("max_steps", "eval_every", "warmup_steps", "device"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_grad_checkpoint:
        cfg.grad_checkpoint = False
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
