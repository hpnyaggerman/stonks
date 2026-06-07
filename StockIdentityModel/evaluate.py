"""Acceptance metrics: held-out consistency and retrieval, partition-redraw agreement.

All evaluation runs through the deterministic inference path: no dropout,
single-group scale, full view, context = training universe present in the
window. Held-out tickers are embedded exactly as an unseen ticker would be:
appended one at a time to the training context.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .data import StockData, ladder
from .model import IdentityEncoder
from .sampling import draw_partitions, split_strata


@torch.no_grad()
def window_embeddings(model: IdentityEncoder, ds: StockData, w: int) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Per-window inference embeddings.

    Returns (train_tickers, Z_train [U, D], {holdout_ticker: z [D]}).
    One universe pass embeds every training ticker; each held-out ticker gets
    its own pass with itself appended to the context (only the target's row
    is kept).
    """
    model.eval()
    dev = next(model.parameters()).device
    uni = ds.train_universe(w)
    H_uni = model.temporal(torch.from_numpy(ds.feats[uni, w]).to(dev))
    Z_train = model.embed_rows(H_uni).cpu().numpy()
    hold = {}
    for h in ds.holdout_idx:
        if not ds.complete[h, w]:
            continue
        H_h = model.temporal(torch.from_numpy(ds.feats[h : h + 1, w]).to(dev))
        H = torch.cat([H_h, H_uni], dim=0)  # target row 0, context after
        hold[int(h)] = model.embed_rows(H)[0].cpu().numpy()
    return uni, Z_train, hold


def _mean_pairwise(zs: np.ndarray) -> float:
    n = len(zs)
    if n < 2:
        return float("nan")
    d = np.sqrt(np.maximum(((zs[:, None] - zs[None, :]) ** 2).sum(-1), 0.0))
    return float(d[np.triu_indices(n, 1)].mean())


def _series(per_win: dict[int, tuple], windows: list[int]) -> tuple[dict, dict]:
    """Per-ticker embedding series over the given windows: (train, holdout)."""
    train_series: dict[int, list[np.ndarray]] = {}
    hold_series: dict[int, list[np.ndarray]] = {}
    for w in windows:
        uni, Z, hold = per_win[w]
        for k, t in enumerate(uni):
            train_series.setdefault(int(t), []).append(Z[k])
        for h, z in hold.items():
            hold_series.setdefault(h, []).append(z)
    return train_series, hold_series


def stratified_eval_windows(ds: StockData, cfg: Config) -> list[int]:
    """strat_eval_windows windows spread the way training spreads its draws: the
    sampler's M contiguous strata (identical block boundaries via split_strata),
    equal counts per stratum, evenly spaced within each. Deterministic. Feeds
    both stratified selection modes ("consistency" and "margin")."""
    per = max(1, cfg.strat_eval_windows // cfg.M)
    out: list[int] = []
    for s in split_strata(ds.usable_windows, cfg.M):
        if not s:
            continue
        idx = np.unique(np.linspace(0, len(s) - 1, num=min(per, len(s))).round().astype(int))
        out.extend(s[i] for i in idx)
    return sorted(set(out))


def _embed_partitioned(model: IdentityEncoder, H: torch.Tensor, U: int, D: int, parts: list, dev) -> np.ndarray:
    """Deterministic full-view embeddings of one universe under a given partition."""
    Z = np.zeros((U, D), dtype=np.float32)
    for grp in parts:
        Hg = H[torch.from_numpy(grp).to(dev)]
        real = torch.ones(1, len(grp), dtype=torch.bool, device=dev)
        Z[grp] = model.context.full_view(Hg[None], real)[0].cpu().numpy()
    return Z


def _retrieval(per_win: dict[int, tuple], windows: list[int], eps: float = 1e-9) -> dict:
    """A/B retrieval and margin ratios over the given windows.

    A = even-indexed, B = odd-indexed (interleaved, so both halves span the same
    eras and slow drift cancels). Query = holdout ticker's mean over A; key = its
    mean over B; gallery = holdout keys + trained tickers' B-means. Hit = own key
    nearer than every impostor.

    Margin ratio rho_h = d(nearest impostor) / d(own key) — the continuous form
    of the same test: rho > 1 iff hit, and the magnitude keeps what the binary
    hit discards (rho 4 vs 1.05, rho 0.95 vs 0.1). Scale-free: collapse drives
    both distances to 0 and reads as rho ~ 1, never as a win.
    """
    A, B = windows[0::2], windows[1::2]
    _, hold_series = _series(per_win, windows)
    hold_ids = sorted(hold_series.keys())

    def mean_over(wins: list[int], h: int) -> np.ndarray | None:
        zs = [per_win[w][2][h] for w in wins if h in per_win[w][2]]
        return np.stack(zs).mean(0) if zs else None

    tk_acc: dict[int, list[np.ndarray]] = {}
    for w in B:
        uni, Z, _ = per_win[w]
        for k, t in enumerate(uni):
            tk_acc.setdefault(int(t), []).append(Z[k])
    train_keys = {t: np.stack(v).mean(0) for t, v in tk_acc.items()}

    hold_keys = {h: mean_over(B, h) for h in hold_ids}
    gallery_ids = [(h, True) for h in hold_ids if hold_keys[h] is not None] + [
        (t, False) for t in sorted(train_keys.keys())
    ]
    correct, tested, ratios = 0, 0, []
    if gallery_ids:
        gallery = np.stack([hold_keys[i] if is_h else train_keys[i] for i, is_h in gallery_ids])
        for h in hold_ids:
            q = mean_over(A, h)
            if q is None or hold_keys[h] is None:
                continue
            d = np.sqrt(((gallery - q) ** 2).sum(-1))
            own = gallery_ids.index((h, True))
            d_imp = float(np.delete(d, own).min())
            ratios.append((h, float(d_imp / (d[own] + eps))))
            tested += 1
            if d[own] < d_imp:
                correct += 1
    return {
        "acc": correct / tested if tested else float("nan"),
        "tested": tested,
        "gallery_size": len(gallery_ids),
        "margin_ratio": float(np.median([r for _, r in ratios])) if ratios else float("nan"),
        "ratios": ratios,
    }


@torch.no_grad()
def run_eval(model: IdentityEncoder, ds: StockData, cfg: Config) -> dict:
    was_training = model.training
    model.eval()
    eval_windows = ds.usable_windows[-cfg.eval_windows :]
    strat_windows = stratified_eval_windows(ds, cfg) if cfg.best_metric in ("consistency", "margin") else []
    per_win: dict[int, tuple] = {
        w: window_embeddings(model, ds, w) for w in sorted(set(eval_windows) | set(strat_windows))
    }

    # --- across-window consistency: holdout vs trained ---
    train_series, hold_series = _series(per_win, eval_windows)
    train_cons = [_mean_pairwise(np.stack(v)) for v in train_series.values() if len(v) >= 2]
    hold_cons = [_mean_pairwise(np.stack(v)) for v in hold_series.values() if len(v) >= 2]
    consistency_ratio = (
        float(np.median(hold_cons) / np.median(train_cons)) if hold_cons and train_cons else float("nan")
    )

    # --- retrieval + margin ratio: query = mean over window-set A, key = mean over set B;
    # gallery = holdout keys + trained tickers as distractors (holdout-only would be trivially easy) ---
    ret = _retrieval(per_win, eval_windows)

    # --- partition-redraw agreement: fixed window, finest scale, full view ---
    dev = next(model.parameters()).device
    w = ds.usable_windows[-1]
    uni = ds.train_universe(w)
    U = len(uni)
    H = model.temporal(torch.from_numpy(ds.feats[uni, w]).to(dev))
    scales = ladder(U, cfg.Y)
    n_fine = scales[-1]
    reps = np.stack([
        _embed_partitioned(model, H, U, cfg.D, draw_partitions(U, [n_fine], seed=(w, 10_000_019 + r))[n_fine], dev)
        for r in range(cfg.eval_redraws)
    ])  # [R, U, D]
    within = np.sqrt(((reps[:, None] - reps[None, :]) ** 2).sum(-1))  # [R, R, U]
    iu = np.triu_indices(cfg.eval_redraws, 1)
    within_ticker = within[iu].mean()
    inter = _mean_pairwise(reps[0])
    partition_agreement = float(within_ticker / inter) if inter else float("nan")

    # --- scale agreement: same window, same tickers, every ladder scale (deterministic
    # full view, one fixed partition draw serving the whole ladder); per-ticker spread
    # across scales over inter-ticker spread at scale 1 (the inference geometry). The
    # inference-grade form of L_sc — certifies set-size invariance on the clean,
    # dropout-free path the artifact ships. partition_agreement isolates composition at
    # a fixed size; this isolates size (composition variation folded in, as in training).
    if len(scales) >= 2:
        parts_all = draw_partitions(U, scales, seed=(w, 20_000_003))
        zs = np.stack([_embed_partitioned(model, H, U, cfg.D, parts_all[n], dev) for n in scales])  # [S, U, D]
        spread = np.sqrt(((zs[:, None] - zs[None, :]) ** 2).sum(-1))  # [S, S, U]
        ius = np.triu_indices(len(scales), 1)
        inter_s = _mean_pairwise(zs[0])
        scale_agreement = float(spread[ius].mean() / inter_s) if inter_s else float("nan")
    else:
        scale_agreement = float("nan")

    # --- stratified block (best_metric "consistency" or "margin"): the same metrics,
    # but over windows spread across the full timeline the way training samples, and
    # more of them. Raw consistency is scale-dependent and collapse-blind; the margin
    # ratio is the scale-free repair. ---
    strat: dict[str, float] = {}
    if strat_windows:
        tr_s, ho_s = _series(per_win, strat_windows)
        tc = [_mean_pairwise(np.stack(v)) for v in tr_s.values() if len(v) >= 2]
        hc = [_mean_pairwise(np.stack(v)) for v in ho_s.values() if len(v) >= 2]
        ret_s = _retrieval(per_win, strat_windows)
        strat = {
            "consistency_train_stratified": float(np.median(tc)) if tc else float("nan"),
            "consistency_holdout_stratified": float(np.median(hc)) if hc else float("nan"),
            "consistency_ratio_stratified": float(np.median(hc) / np.median(tc)) if tc and hc else float("nan"),
            "retrieval_acc_stratified": ret_s["acc"],
            "margin_ratio_stratified": ret_s["margin_ratio"],
            "strat_windows": len(strat_windows),
        }

    if was_training:
        model.train()
    return {
        "consistency_train_median": float(np.median(train_cons)) if train_cons else float("nan"),
        "consistency_holdout_median": float(np.median(hold_cons)) if hold_cons else float("nan"),
        "consistency_ratio": consistency_ratio,
        "retrieval_acc": ret["acc"],
        "retrieval_tested": ret["tested"],
        "gallery_size": ret["gallery_size"],
        "margin_ratio": ret["margin_ratio"],
        "partition_agreement": partition_agreement,
        "scale_agreement": scale_agreement,
        **strat,
    }
