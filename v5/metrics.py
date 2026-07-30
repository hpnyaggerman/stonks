"""Cross-sectional ranking metrics for the v5 pipeline (numpy/scipy only, no torch).

Every per-date metric here operates on flat per-sample arrays plus a ``meta`` dict of
row-aligned arrays (keys: ``date`` anchor date int64 ns, ``ticker_id``, ``tdates``
(N, H) stored per-label target dates int64 ns, ``tradable`` bool at anchor,
``sigma_hat`` anchor sigma, plus optional ``tag`` / ``m_uncens`` / ``close``).
Labels are assigned to validation blocks by their stored TARGET dates, never by
anchor + d sessions (wrong for gap tickers); the HAC estimator treats blocks as
independent clusters with within-block adjacency by session index, so metrics can
never pair observations across a role boundary as if they were serially adjacent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

HORIZON_LABELS = ("1d", "1w", "1m", "6m")
HORIZON_DAYS = (1, 5, 21, 126)
# Bartlett truncation lag per horizon: max(5, d_h - 1) covers the label-overlap
# serial correlation while keeping a floor for weekly effects.
HAC_LAG = tuple(max(5, d - 1) for d in HORIZON_DAYS)
IC_MIN_NAMES = 30
STOPPING_MIN_DATES = 10


def default_cfg(min_names=IC_MIN_NAMES):
    """Metric configuration consumed by the suite functions."""
    return {"min_names": int(min_names), "hac_lag": list(HAC_LAG),
            "horizon_labels": list(HORIZON_LABELS), "horizon_days": list(HORIZON_DAYS)}


# ------------------------------------------------------------------ primitives

def spearman_ic(scores, z, group_ids, min_names=IC_MIN_NAMES):
    """Per-group Spearman correlation of ``scores`` against ``z``.

    Groups below ``min_names`` are dropped; a group whose score (or outcome) has zero
    variance is dropped rather than emitted as NaN, so the returned ``ic_values`` are
    always finite. Returns ``(group_keys, ic_values)`` sorted by group key.
    """
    scores = np.asarray(scores, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    order = np.argsort(group_ids, kind="stable")
    g = group_ids[order]
    if g.size == 0:
        return np.asarray([], dtype=group_ids.dtype), np.asarray([], dtype=np.float64)
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    bounds = np.concatenate([starts, [g.size]])
    keys, vals = [], []
    for i in range(len(starts)):
        idx = order[bounds[i]:bounds[i + 1]]
        if idx.size < min_names:
            continue
        s, y = scores[idx], z[idx]
        if not (np.isfinite(s).all() and np.isfinite(y).all()):
            keep = np.isfinite(s) & np.isfinite(y)
            s, y = s[keep], y[keep]
            if s.size < min_names:
                continue
        rs, ry = rankdata(s), rankdata(y)
        ss, sy = rs.std(), ry.std()
        if ss == 0 or sy == 0:
            continue
        ic = float(np.mean((rs - rs.mean()) * (ry - ry.mean())) / (ss * sy))
        if not np.isfinite(ic):
            continue
        keys.append(g[bounds[i]])
        vals.append(ic)
    return np.asarray(keys), np.asarray(vals, dtype=np.float64)


def hac_t(ic, session_idx, lag, block_ids=None):
    """Newey-West (Bartlett) mean / SE / t of a per-session series.

    ``SE^2 = (sum_b S_b) / N^2`` with ``S_b = sum e_t^2 + 2 * sum_{j=1..L}
    (1 - j/(L+1)) * sum_t e_t e_{t+j}``, computed within each block b (blocks are
    independent clusters), ``e_t = ic_t - mean(ic)`` pooled. Lag-j products pair only
    observations whose session indices differ by exactly j, so calendar holes inside a
    block contribute no spurious adjacency. Returns ``(mean, se, t, n)``; on zero
    qualifying observations returns ``(nan, nan, nan, 0)`` and never raises.
    """
    ic = np.asarray(ic, dtype=np.float64)
    session_idx = np.asarray(session_idx, dtype=np.int64)
    finite = np.isfinite(ic)
    ic, session_idx = ic[finite], session_idx[finite]
    if block_ids is None:
        block_ids = np.zeros(ic.size, dtype=np.int64)
    else:
        block_ids = np.asarray(block_ids)[finite]
    n = ic.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(ic.mean())
    e = ic - mean
    lag = int(lag)
    s_total = 0.0
    for b in np.unique(block_ids):
        sel = block_ids == b
        s_idx, e_b = session_idx[sel], e[sel]
        order = np.argsort(s_idx, kind="stable")
        s_idx, e_b = s_idx[order], e_b[order]
        s_b = float(np.sum(e_b * e_b))
        for j in range(1, lag + 1):
            pos = np.searchsorted(s_idx, s_idx + j)
            ok = (pos < s_idx.size)
            ok[ok] = s_idx[pos[ok]] == s_idx[ok] + j
            if ok.any():
                w = 1.0 - j / (lag + 1.0)
                s_b += 2.0 * w * float(np.sum(e_b[ok] * e_b[pos[ok]]))
        s_total += s_b
    se = float(np.sqrt(max(s_total, 0.0)) / n)
    t = mean / se if se > 0 else float("nan")
    return mean, se, t, int(n)


def block_partition(val_sessions, n_blocks=4, phase=0):
    """Partition the sorted unique validation sessions into contiguous blocks.

    Returns ``(blocks, roles)``: ``blocks`` is a list of session-date arrays
    (``np.array_split`` of the sorted unique sessions, so the blocks tile the year
    disjointly), ``roles`` alternates ``stop`` / ``gate`` starting with ``stop`` at
    ``phase=0`` (block 1 on the stopping side).
    """
    sessions = np.unique(np.asarray(val_sessions))
    blocks = [b for b in np.array_split(sessions, n_blocks) if b.size]
    roles = ["stop" if (i + phase) % 2 == 0 else "gate" for i in range(len(blocks))]
    return blocks, roles


def block_index(dates, blocks):
    """Block index for each date (by the block-start boundaries; dates before the
    first block map to -1, dates after the last block's start map into the last
    block). Membership is by date value, so target dates on a ticker's own calendar
    that fall between global sessions still land in the enclosing block."""
    starts = np.asarray([np.asarray(b).min() for b in blocks])
    d = np.asarray(dates)
    return np.searchsorted(starts, d, side="right").astype(np.int64) - 1


def role_label_mask(tdates, blocks, roles, role):
    """(N, H) bool: label belongs to ``role`` iff its stored target date falls in a
    block carrying that role. NaT targets are never assigned to any role."""
    td = np.asarray(tdates)
    flat = block_index(td.reshape(-1), blocks).reshape(td.shape)
    role_arr = np.asarray(roles)
    out = np.zeros(td.shape, dtype=bool)
    valid = flat >= 0
    if np.issubdtype(td.dtype, np.datetime64):
        valid &= ~np.isnat(td)
    out[valid] = role_arr[flat[valid]] == role
    return out


def stopping_score(t_1d, t_1w, n_1d=None, n_1w=None, min_dates=STOPPING_MIN_DATES):
    """Combined early-stopping criterion ``(t_1d + t_1w) / sqrt(2)``.

    Ordinal only: the two t statistics share dates and the 1d label path nests inside
    1w, so the combination is not standard-normal and success gates never read it.
    Degenerate guard: fewer than ``min_dates`` usable stopping dates at either horizon,
    or a non-finite input t, yields ``-inf`` (never NaN), which callers count as
    no-improvement.
    """
    if n_1d is not None and n_1d < min_dates:
        return float("-inf")
    if n_1w is not None and n_1w < min_dates:
        return float("-inf")
    if not (np.isfinite(t_1d) and np.isfinite(t_1w)):
        return float("-inf")
    return float((t_1d + t_1w) / np.sqrt(2.0))


# ----------------------------------------------------------------- IC suite

def _session_axis(meta):
    return np.unique(np.asarray(meta["date"]))


def _iter_scores(prob_up, prob_down):
    yield "p_up", np.asarray(prob_up, dtype=np.float64)
    yield "score", np.asarray(prob_up, dtype=np.float64) - np.asarray(prob_down, dtype=np.float64)


def ic_suite(prob_up, prob_down, z, mask, meta, cfg, blocks=None, roles=None,
             tradable=None, alt_mask=None):
    """Per-horizon, per-score rank-IC report over every requested population.

    Populations are the cross product of mask variant (``primary``; ``uncensored``
    when ``alt_mask`` is given), block role (``all``; each role when ``blocks`` are
    given, with labels assigned by stored target date), and tradability (``all``;
    ``tradable`` when the flag array is given). Returns ``(records, series)``:
    ``records`` is a list of scalar dicts (mean, HAC se/t with the per-horizon lag,
    per-date count, per-date IC dispersion); ``series`` maps the same record key to
    its ``(dates, ic_values)`` pair for persistence checks and plotting.
    """
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask)
    dates = np.asarray(meta["date"])
    sessions = _session_axis(meta)
    labels = cfg["horizon_labels"]
    lags = cfg["hac_lag"]
    min_names = cfg["min_names"]

    variants = [("primary", mask.astype(bool))]
    if alt_mask is not None:
        variants.append(("uncensored", np.asarray(alt_mask).astype(bool)))
    role_masks = [("all", None)]
    if blocks is not None and roles is not None:
        for role in sorted(set(roles)):
            role_masks.append((role, role_label_mask(meta["tdates"], blocks, roles, role)))
    trad_flags = [("all", None)]
    if tradable is not None:
        trad_flags.append(("tradable", np.asarray(tradable).astype(bool)))

    records, series = [], {}
    for score_name, score in _iter_scores(prob_up, prob_down):
        for hi, hl in enumerate(labels):
            for var_name, var_mask in variants:
                for role_name, rmask in role_masks:
                    for trad_name, tmask in trad_flags:
                        el = var_mask[:, hi].copy()
                        if rmask is not None:
                            el &= rmask[:, hi]
                        if tmask is not None:
                            el &= tmask
                        keys, ics = spearman_ic(score[el, hi], z[el, hi], dates[el],
                                                min_names=min_names)
                        sess_idx = np.searchsorted(sessions, keys) if keys.size else keys
                        bid = (block_index(keys, blocks) if blocks is not None and keys.size
                               else None)
                        m, se, t, n = hac_t(ics, sess_idx, lags[hi], bid)
                        rec = {"horizon": hl, "score": score_name, "mask": var_name,
                               "role": role_name, "trad": trad_name,
                               "mean_ic": m, "se": se, "t": t, "n_dates": n,
                               "ic_std": float(np.std(ics)) if n else float("nan")}
                        records.append(rec)
                        series[(hl, score_name, var_name, role_name, trad_name)] = (keys, ics)
    return records, series


def _contiguous_runs(session_idx):
    """Cluster ids splitting wherever consecutive (sorted) sessions are not adjacent;
    used where no explicit block structure is supplied."""
    s = np.asarray(session_idx, dtype=np.int64)
    if s.size == 0:
        return s.copy()
    return np.concatenate([[0], np.cumsum(np.diff(s) > 1)])


def top_of_ranking(score, z, mask, meta, sigma=None, tradable=None, alt_mask=None,
                   min_names=IC_MIN_NAMES, horizon_days=HORIZON_DAYS,
                   horizon_labels=HORIZON_LABELS, hac_lag=HAC_LAG):
    """Per-date top-1 and top-decile realized outcome of the ranking.

    For each date passing the ``min_names`` floor: the mean realized ``z`` of the
    top-scoring name and of the top decile, plus -- when ``sigma`` (anchor sigma_hat)
    is given -- the return-units top-decile mean ``z * sigma * sqrt(d)`` (the label's
    own denormalization; the 1d cost gate consumes this variant, since a dimensionless
    z cannot be compared to a basis-point threshold). Populations: unfiltered and
    (when ``tradable`` is given) tradability-filtered; ``alt_mask`` repeats both on
    the alternative label mask. HAC clusters are contiguous session runs.
    Returns ``(records, series)`` like :func:`ic_suite`; series values are dicts with
    ``top1_z``, ``topdec_z`` and optional ``topdec_ret`` per-date arrays.
    """
    score = np.asarray(score, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask)
    dates = np.asarray(meta["date"])
    sessions = _session_axis(meta)
    sigma = None if sigma is None else np.asarray(sigma, dtype=np.float64)

    variants = [("primary", mask.astype(bool))]
    if alt_mask is not None:
        variants.append(("uncensored", np.asarray(alt_mask).astype(bool)))
    trad_flags = [("all", None)]
    if tradable is not None:
        trad_flags.append(("tradable", np.asarray(tradable).astype(bool)))

    records, series = [], {}
    for hi, hl in enumerate(horizon_labels):
        scale = np.sqrt(float(horizon_days[hi]))
        for var_name, var_mask in variants:
            for trad_name, tmask in trad_flags:
                el = var_mask[:, hi].copy()
                if tmask is not None:
                    el &= tmask
                el &= np.isfinite(score[:, hi]) & np.isfinite(z[:, hi])
                d_el, s_el, z_el = dates[el], score[el, hi], z[el, hi]
                sig_el = sigma[el] if sigma is not None else None
                order = np.argsort(d_el, kind="stable")
                d_el, s_el, z_el = d_el[order], s_el[order], z_el[order]
                if sig_el is not None:
                    sig_el = sig_el[order]
                starts = (np.flatnonzero(np.concatenate([[True], d_el[1:] != d_el[:-1]]))
                          if d_el.size else np.asarray([], dtype=np.int64))
                bounds = np.concatenate([starts, [d_el.size]])
                keys, top1, topdec, topdec_ret = [], [], [], []
                for i in range(len(starts)):
                    lo, hi_b = bounds[i], bounds[i + 1]
                    n_d = hi_b - lo
                    if n_d < min_names:
                        continue
                    s_d, z_d = s_el[lo:hi_b], z_el[lo:hi_b]
                    k = max(1, n_d // 10)
                    top_idx = np.argsort(s_d, kind="stable")[::-1][:k]
                    keys.append(d_el[lo])
                    top1.append(float(z_d[np.argmax(s_d)]))
                    topdec.append(float(z_d[top_idx].mean()))
                    if sig_el is not None:
                        ret = z_d[top_idx] * sig_el[lo:hi_b][top_idx] * scale
                        topdec_ret.append(float(ret.mean()))
                keys = np.asarray(keys)
                sess_idx = np.searchsorted(sessions, keys) if keys.size else keys
                runs = _contiguous_runs(sess_idx)
                rec = {"horizon": hl, "mask": var_name, "trad": trad_name,
                       "n_dates": int(keys.size)}
                for name, arr in (("top1_z", top1), ("topdec_z", topdec),
                                  ("topdec_ret", topdec_ret)):
                    if name == "topdec_ret" and sigma is None:
                        continue
                    m, se, t, n = hac_t(np.asarray(arr), sess_idx, hac_lag[hi], runs)
                    rec[f"{name}_mean"] = m
                    rec[f"{name}_t"] = t
                    rec[f"{name}_se"] = se
                records.append(rec)
                series[(hl, var_name, trad_name)] = {
                    "dates": keys, "top1_z": np.asarray(top1),
                    "topdec_z": np.asarray(topdec),
                    **({"topdec_ret": np.asarray(topdec_ret)} if sigma is not None else {}),
                }
    return records, series


# ------------------------------------------------------- uncertainty quality

def uncertainty_quality(up_std, prob_up, z, mask, meta, cfg):
    """Does ensemble disagreement predict error? Per horizon: the Spearman correlation
    of the member-std of P(up) with the realized absolute error
    ``|mean P(up) - 1{z > theta_h}|`` over mask-passing labels, plus a risk-coverage
    curve (mean absolute error by ascending-std decile). Needs >= 2 members upstream
    (the std is degenerate otherwise); ``cfg['theta']`` supplies the per-horizon class
    threshold on the z axis."""
    up_std = np.asarray(up_std, dtype=np.float64)
    prob_up = np.asarray(prob_up, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    theta = cfg["theta"]
    out = {}
    for hi, hl in enumerate(cfg.get("horizon_labels", HORIZON_LABELS)):
        sel = mask[:, hi] & np.isfinite(z[:, hi]) & np.isfinite(up_std[:, hi])
        n = int(sel.sum())
        if n < 20:
            out[hl] = {"spearman": float("nan"), "deciles": [], "n": n}
            continue
        err = np.abs(prob_up[sel, hi] - (z[sel, hi] > theta[hi]).astype(np.float64))
        std = up_std[sel, hi]
        rs, re = rankdata(std), rankdata(err)
        ss, se_ = rs.std(), re.std()
        corr = (float(np.mean((rs - rs.mean()) * (re - re.mean())) / (ss * se_))
                if ss > 0 and se_ > 0 else float("nan"))
        order = np.argsort(std, kind="stable")
        dec = [float(chunk.mean()) for chunk in np.array_split(err[order], 10) if chunk.size]
        out[hl] = {"spearman": corr, "deciles": dec, "n": n}
    return out


# ------------------------------------------------------------------- nulls

def null_shuffle_within_date(scores, meta, seed):
    """Null (a): permute score rows across tickers within every anchor date. Destroys
    all cross-sectional signal while keeping each date's score distribution, so any
    surviving IC indicates broken mechanics."""
    scores = np.array(scores, dtype=np.float64, copy=True)
    dates = np.asarray(meta["date"])
    rng = np.random.default_rng(seed)
    for d in np.unique(dates):
        idx = np.flatnonzero(dates == d)
        if idx.size > 1:
            scores[idx] = scores[idx[rng.permutation(idx.size)]]
    return scores


def null_circular_shift(scores, meta, seed):
    """Null (b): circularly shift each ticker's date-sorted score series by a random
    offset. Destroys score-outcome alignment while preserving the score series'
    serial correlation, which the within-date permutation cannot test."""
    scores = np.array(scores, dtype=np.float64, copy=True)
    dates = np.asarray(meta["date"])
    tickers = np.asarray(meta["ticker_id"])
    rng = np.random.default_rng(seed)
    for t in np.unique(tickers):
        idx = np.flatnonzero(tickers == t)
        if idx.size < 2:
            continue
        idx = idx[np.argsort(dates[idx], kind="stable")]
        shift = int(rng.integers(1, idx.size))
        scores[idx] = scores[np.roll(idx, shift)]
    return scores
