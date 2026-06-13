"""Turn ensemble predictions into forecast-CSV columns.

Shared by the trainer's forecast export and the live scorer so both emit identical
column semantics. Operates on numpy arrays detached from the model output.
"""
from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = (0.10, 0.50, 0.90)
HORIZON_LABELS = ("1d", "1w", "1m", "6m")


def histogram_quantiles(hist, bin_width, n_bins, levels=QUANTILE_LEVELS):
    """Quantiles of the predictive histogram on the z-score axis.

    ``hist`` is ``(..., n_bins)`` of bin probabilities. The cumulative distribution is
    linearly interpolated across bin edges to invert it at each level. Returns
    ``(..., len(levels))``.
    """
    hist = np.asarray(hist, dtype=np.float64)
    half = n_bins // 2
    edges = (np.arange(n_bins + 1) - half) * bin_width            # (n_bins+1,)
    flat = hist.reshape(-1, n_bins)
    cdf = np.concatenate([np.zeros((flat.shape[0], 1)), np.cumsum(flat, axis=1)], axis=1)
    out = np.empty((flat.shape[0], len(levels)))
    for i in range(flat.shape[0]):
        c = cdf[i]
        total = c[-1]
        # Guard against a degenerate (all-zero) row before normalizing the CDF.
        c = c / total if total > 0 else np.linspace(0.0, 1.0, c.size)
        out[i] = np.interp(levels, c, edges)
    return out.reshape(hist.shape[:-1] + (len(levels),))


def build_forecast_columns(dates, close, sigma_hat, horizon_days,
                           up, up_std, down, neutral, score, score_std,
                           hist, bin_width, n_bins):
    """Assemble the per-row forecast columns for one ticker.

    Probabilities come straight from the ensemble three-class marginals; quantiles are
    histogram z-quantiles rescaled to raw-return units by ``sigma_hat * sqrt(d_h)``.
    All array inputs are indexed ``[row, horizon]`` (except ``hist`` ``[row, horizon, bin]``).
    """
    cols = {"Date": dates, "Close": close}
    qz = histogram_quantiles(hist, bin_width, n_bins)             # (rows, H, 3)
    for hi, (label, d) in enumerate(zip(HORIZON_LABELS, horizon_days)):
        scale = sigma_hat * np.sqrt(d)
        cols[f"Pred_Prob_{label}"] = up[:, hi]
        cols[f"Pred_Prob_Std_{label}"] = up_std[:, hi]
        cols[f"Pred_Prob_Down_{label}"] = down[:, hi]
        cols[f"Pred_Prob_Neutral_{label}"] = neutral[:, hi]
        cols[f"Pred_Q10_{label}"] = qz[:, hi, 0] * scale
        cols[f"Pred_Q50_{label}"] = qz[:, hi, 1] * scale
        cols[f"Pred_Q90_{label}"] = qz[:, hi, 2] * scale
        cols[f"Score_{label}"] = score[:, hi]
        cols[f"Score_Std_{label}"] = score_std[:, hi]
    return cols
