"""Acceptance metrics: held-out consistency and retrieval, partition-redraw agreement.

All evaluation runs through the deterministic inference path: no dropout,
single-group scale, full view, context = training universe present in the
window. Held-out tickers are embedded exactly as an unseen ticker would be:
appended one at a time to the training context.

Cross-market mode adds, per secondary market: the same newest-block metrics
(prefixed, e.g. cn_*) with the union gallery as distractors, and replaces the
stratified selection metrics with union-protocol versions (queries from every
market's holdout, gallery spanning all markets) — the US-protocol stratified
values are kept under *_us keys for comparability with single-market runs.
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from .config import Config
from .data import StockData, ladder
from .model import IdentityEncoder
from .sampling import draw_partitions, split_strata


def _per_window_embeddings(replicas: list, ds: StockData, windows, mkt=None) -> dict[int, tuple]:
    """per_win dict over `windows`, optionally split across model replicas on
    different devices (one host thread per replica; CUDA ops release the GIL).
    Byte-identical to the single-replica path on identical device types: eval is
    deterministic, no_grad, and per-window independent — the split only changes
    who computes what. Mixed device types differ at float level."""
    windows = sorted(windows)
    if len(replicas) <= 1 or len(windows) <= 1:
        return {w: window_embeddings(replicas[0], ds, w, mkt) for w in windows}
    out: dict[int, tuple] = {}
    errs: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def work(i, rep, wins):
        try:
            for w in wins:
                r = window_embeddings(rep, ds, w, mkt)
                with lock:
                    out[w] = r
        except BaseException as e:  # surfaced after join — a swallowed thread error
            with lock:              # would otherwise resurface as a bare KeyError
                errs.append((i, e)) # in a downstream per_win lookup
    threads = [
        threading.Thread(target=work, args=(i, rep, windows[i :: len(replicas)]))
        for i, rep in enumerate(replicas)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        i, e = errs[0]
        dev = next(replicas[i].parameters()).device
        raise RuntimeError(f"eval replica {i} ({dev}) failed mid-eval") from e
    return out


@torch.no_grad()
def window_embeddings(model: IdentityEncoder, ds: StockData, w: int, mkt=None) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Per-window inference embeddings for one market (default: primary/US).

    Returns (train_tickers, Z_train [U, D], {holdout_ticker: z [D]}), all in
    global ticker ids. One universe pass embeds every training ticker; each
    held-out ticker gets its own pass with itself appended to the context
    (only the target's row is kept).
    """
    mkt = mkt if mkt is not None else ds.us
    model.eval()
    dev = next(model.parameters()).device
    uni = mkt.train_universe(w)
    H_uni = model.temporal(torch.from_numpy(mkt.feats_at(uni, w)).to(dev))
    Z_train = model.embed_rows(H_uni).cpu().numpy()
    hold = {}
    for h in mkt.holdout_tids:
        if not mkt.complete[mkt.row[h], w]:
            continue
        H_h = model.temporal(torch.from_numpy(mkt.feats_at(np.array([h]), w)).to(dev))
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


def stratified_eval_windows(ds: StockData, cfg: Config, mkt=None) -> list[int]:
    """strat_eval_windows windows spread the way training spreads its draws: the
    sampler's M contiguous strata (identical block boundaries via split_strata),
    equal counts per stratum, evenly spaced within each. Deterministic. Feeds
    both stratified selection modes ("consistency" and "margin")."""
    usable = (mkt if mkt is not None else ds.us).usable_windows
    per = max(1, cfg.strat_eval_windows // cfg.M)
    out: list[int] = []
    for s in split_strata(usable, cfg.M):
        if not s:
            continue
        idx = np.unique(np.linspace(0, len(s) - 1, num=min(per, len(s))).round().astype(int))
        out.extend(s[i] for i in idx)
    return sorted(set(out))


def _ab_keys(per_win: dict[int, tuple], windows: list[int]) -> tuple[dict, dict, dict]:
    """A/B key construction for retrieval over the given windows.

    A = even-indexed, B = odd-indexed (interleaved, so both halves span the same
    eras and slow drift cancels). Returns (queries, hold_keys, train_keys):
    query = holdout ticker's mean over A; key = its mean over B; train key =
    trained ticker's mean over B.
    """
    A, B = windows[0::2], windows[1::2]
    hold_ids = sorted({h for w in windows for h in per_win[w][2]})

    def mean_over(wins: list[int], h: int) -> np.ndarray | None:
        zs = [per_win[w][2][h] for w in wins if h in per_win[w][2]]
        return np.stack(zs).mean(0) if zs else None

    tk_acc: dict[int, list[np.ndarray]] = {}
    for w in B:
        uni, Z, _ = per_win[w]
        for k, t in enumerate(uni):
            tk_acc.setdefault(int(t), []).append(Z[k])
    train_keys = {t: np.stack(v).mean(0) for t, v in tk_acc.items()}
    queries = {h: mean_over(A, h) for h in hold_ids}
    hold_keys = {h: mean_over(B, h) for h in hold_ids}
    return queries, hold_keys, train_keys


def _score_retrieval(queries: dict, hold_keys: dict, train_keys: dict, eps: float = 1e-9) -> dict:
    """Retrieval + margin ratios. Hit = own key nearer than every impostor in
    the gallery. The gallery carries EVERY non-None holdout key passed in —
    queried or not, which is how the per-market cross-market scoring gets the
    other market's holdout as impostors — plus all trained tickers' keys as
    distractors; only ids present in `queries` are scored.

    Margin ratio rho_h = d(nearest impostor) / d(own key) — the continuous form
    of the same test: rho > 1 iff hit, and the magnitude keeps what the binary
    hit discards (rho 4 vs 1.05, rho 0.95 vs 0.1). Scale-free: collapse drives
    both distances to 0 and reads as rho ~ 1, never as a win.
    """
    hold_ids = sorted(queries.keys())
    gallery_ids = [(h, True) for h in sorted(hold_keys) if hold_keys[h] is not None] + [
        (t, False) for t in sorted(train_keys.keys())
    ]
    correct, tested, ratios = 0, 0, []
    if gallery_ids:
        gallery = np.stack([hold_keys[i] if is_h else train_keys[i] for i, is_h in gallery_ids])
        for h in hold_ids:
            q = queries[h]
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


def _retrieval(per_win: dict[int, tuple], windows: list[int], eps: float = 1e-9) -> dict:
    """A/B retrieval and margin ratios over the given windows (single gallery)."""
    queries, hold_keys, train_keys = _ab_keys(per_win, windows)
    return _score_retrieval(queries, hold_keys, train_keys, eps)


@torch.no_grad()
def _partition_agreement(model: IdentityEncoder, cfg: Config, mkt) -> float:
    """Re-drawn partitions at the finest scale, fixed (newest usable) window."""
    dev = next(model.parameters()).device
    w = mkt.usable_windows[-1]
    uni = mkt.train_universe(w)
    U = len(uni)
    H = model.temporal(torch.from_numpy(mkt.feats_at(uni, w)).to(dev))
    n_fine = ladder(U, cfg.Y)[-1]
    reps = []
    for r in range(cfg.eval_redraws):
        parts = draw_partitions(U, [n_fine], seed=(w, 10_000_019 + r))[n_fine]
        Z = np.zeros((U, cfg.D), dtype=np.float32)
        for grp in parts:
            Hg = H[torch.from_numpy(grp).to(dev)]
            real = torch.ones(1, len(grp), dtype=torch.bool, device=dev)
            Z[grp] = model.context.full_view(Hg[None], real)[0].cpu().numpy()
        reps.append(Z)
    reps = np.stack(reps)  # [R, U, D]
    within = np.sqrt(((reps[:, None] - reps[None, :]) ** 2).sum(-1))  # [R, R, U]
    iu = np.triu_indices(cfg.eval_redraws, 1)
    within_ticker = within[iu].mean()
    inter = _mean_pairwise(reps[0])
    return float(within_ticker / inter) if inter else float("nan")


def _cross_market_eval(model, ds, cfg, per_win_us, us_eval, us_strat, strat_us_vals, replicas=None) -> dict:
    """Secondary-market metrics + union-protocol stratified selection metrics.

    Per secondary market m (key prefix = lowercased market name):
      {m}_consistency_*  — across-window holdout/train consistency on m's windows
      {m}_retrieval_acc / {m}_margin_ratio — m's holdout queries against the
        UNION gallery (own + US keys), so cross-market distinctiveness is tested
      {m}_partition_agreement
    Union stratified block replaces the canonical *_stratified keys (queries
    from every market's holdout, union gallery); the US-protocol values move to
    *_stratified_us. market_centroid_acc/_dist read whether "market" is a
    separating axis (nearest-centroid accuracy over newest train keys; high =
    markets form separable clouds).
    """
    out: dict = {}
    us_n = _ab_keys(per_win_us, us_eval)
    us_s = _ab_keys(per_win_us, us_strat) if us_strat else None
    sec_blobs = []
    for mkt in ds.secondary:
        pre = mkt.name.lower() + "_"
        m_eval = mkt.usable_windows[-cfg.eval_windows:]
        m_strat = stratified_eval_windows(ds, cfg, mkt) if us_strat else []
        per_win_m = _per_window_embeddings(replicas or [model], ds, set(m_eval) | set(m_strat), mkt)
        tr, ho = _series(per_win_m, m_eval)
        m_tc = [_mean_pairwise(np.stack(v)) for v in tr.values() if len(v) >= 2]
        m_hc = [_mean_pairwise(np.stack(v)) for v in ho.values() if len(v) >= 2]
        out[pre + "consistency_train_median"] = float(np.median(m_tc)) if m_tc else float("nan")
        out[pre + "consistency_holdout_median"] = float(np.median(m_hc)) if m_hc else float("nan")
        m_n = _ab_keys(per_win_m, m_eval)
        r = _score_retrieval(m_n[0], {**us_n[1], **m_n[1]}, {**us_n[2], **m_n[2]})
        out[pre + "retrieval_acc"] = r["acc"]
        out[pre + "retrieval_tested"] = r["tested"]
        out[pre + "margin_ratio"] = r["margin_ratio"]
        out[pre + "partition_agreement"] = _partition_agreement(model, cfg, mkt)
        sec_blobs.append((mkt, m_strat, per_win_m, m_n))

    # union stratified: queries from every market's holdout, gallery spanning all markets
    if us_s is not None:
        uq, uhk, utk = dict(us_s[0]), dict(us_s[1]), dict(us_s[2])
        tr_us, ho_us = _series(per_win_us, us_strat)
        tc_all = [_mean_pairwise(np.stack(v)) for v in tr_us.values() if len(v) >= 2]
        hc_all = [_mean_pairwise(np.stack(v)) for v in ho_us.values() if len(v) >= 2]
        n_strat = len(us_strat)
        for mkt, m_strat, per_win_m, _ in sec_blobs:
            if not m_strat:
                continue
            m_s = _ab_keys(per_win_m, m_strat)
            uq.update(m_s[0])
            uhk.update(m_s[1])
            utk.update(m_s[2])
            tr_m, ho_m = _series(per_win_m, m_strat)
            tc_all += [_mean_pairwise(np.stack(v)) for v in tr_m.values() if len(v) >= 2]
            hc_all += [_mean_pairwise(np.stack(v)) for v in ho_m.values() if len(v) >= 2]
            n_strat += len(m_strat)
        ru = _score_retrieval(uq, uhk, utk)
        for k in ("consistency_train_stratified", "consistency_holdout_stratified",
                  "consistency_ratio_stratified", "retrieval_acc_stratified", "margin_ratio_stratified"):
            if k in strat_us_vals:
                out[k + "_us"] = strat_us_vals[k]
        out["consistency_train_stratified"] = float(np.median(tc_all)) if tc_all else float("nan")
        out["consistency_holdout_stratified"] = float(np.median(hc_all)) if hc_all else float("nan")
        out["consistency_ratio_stratified"] = (
            float(np.median(hc_all) / np.median(tc_all)) if tc_all and hc_all else float("nan")
        )
        out["retrieval_acc_stratified"] = ru["acc"]
        out["margin_ratio_stratified"] = ru["margin_ratio"]
        out["strat_windows"] = n_strat

    # market-as-axis probe: nearest-centroid accuracy over newest train keys
    us_keys = np.stack(list(us_n[2].values())) if us_n[2] else None
    sec_keys_list = [np.stack(list(b[3][2].values())) for b in sec_blobs if b[3][2]]
    if us_keys is not None and sec_keys_list:
        sec_keys = np.concatenate(sec_keys_list)
        c_us, c_sec = us_keys.mean(0), sec_keys.mean(0)
        d_us = lambda z: np.sqrt(((z - c_us) ** 2).sum(-1))
        d_sec = lambda z: np.sqrt(((z - c_sec) ** 2).sum(-1))
        hits = int((d_us(us_keys) < d_sec(us_keys)).sum()) + int((d_sec(sec_keys) < d_us(sec_keys)).sum())
        out["market_centroid_acc"] = hits / (len(us_keys) + len(sec_keys))
        out["market_centroid_dist"] = float(np.sqrt(((c_us - c_sec) ** 2).sum()))
    return out


@torch.no_grad()
def run_eval(model: IdentityEncoder, ds: StockData, cfg: Config, replicas: list | None = None) -> dict:
    was_training = model.training
    model.eval()
    # replicas (model + per-device mirrors, weights already synced by the caller)
    # only split the per-window work when eval_parallel is on; metrics are
    # byte-identical either way
    reps = replicas if (cfg.eval_parallel and replicas and len(replicas) > 1) else [model]
    cross = bool(cfg.cross_market and ds.secondary)
    eval_windows = ds.us.usable_windows[-cfg.eval_windows:]
    strat_windows = (
        stratified_eval_windows(ds, cfg, ds.us)
        if (cfg.best_metric in ("consistency", "margin") or cross)
        else []
    )
    per_win: dict[int, tuple] = _per_window_embeddings(reps, ds, set(eval_windows) | set(strat_windows), ds.us)

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

    partition_agreement = _partition_agreement(model, cfg, ds.us)

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

    out = {
        "consistency_train_median": float(np.median(train_cons)) if train_cons else float("nan"),
        "consistency_holdout_median": float(np.median(hold_cons)) if hold_cons else float("nan"),
        "consistency_ratio": consistency_ratio,
        "retrieval_acc": ret["acc"],
        "retrieval_tested": ret["tested"],
        "gallery_size": ret["gallery_size"],
        "margin_ratio": ret["margin_ratio"],
        "partition_agreement": partition_agreement,
        **strat,
    }
    if cross:
        out.update(_cross_market_eval(model, ds, cfg, per_win, eval_windows, strat_windows, strat, replicas=reps))
    if was_training:
        model.train()
    return out
