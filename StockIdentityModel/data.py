"""Data layer.

Builds, from raw OHLCV CSVs:
  - the global trading-day grid (benchmark = SPY calendar),
  - non-overlapping N-day windows anchored at the newest day, tiling backward,
  - the per-(ticker, window) completeness mask (bar on every grid day, V > 0),
  - the normalized 5-feature candle tensor,
  - frozen global symmetric clip thresholds from training tickers only,
  - the holdout split.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, REPO_ROOT

OHLCV = ["open", "high", "low", "close", "volume"]
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    return df[OHLCV].astype("float64")


def ladder(universe_size: int, Y: int) -> list[int]:
    """Geometric group counts {1, 2, 4, ...}, capped so the smallest group keeps >= Y tickers."""
    out = [1]
    while universe_size // (out[-1] * 2) >= Y:
        out.append(out[-1] * 2)
    return out


class StockData:
    def __init__(self, cfg: Config, repo_root: Path = REPO_ROOT, use_cache: bool = True):
        self.cfg = cfg
        raw = Path(repo_root) / cfg.raw_dir
        stock_dir = raw / "stocksData"
        files = sorted(stock_dir.glob("*_daily.csv"))
        if not files:
            raise FileNotFoundError(f"no raw candle CSVs under {stock_dir}")
        self.tickers = [f.name[: -len("_daily.csv")] for f in files]
        self.tindex = {t: i for i, t in enumerate(self.tickers)}

        key = self._cache_key(files, raw)
        cache_file = CACHE_DIR / f"data_{key}.npz"
        if use_cache and cache_file.exists():
            z = np.load(cache_file, allow_pickle=False)
            self.grid = pd.DatetimeIndex(z["grid"])
            self.win_start = int(z["win_start"])
            self.complete = z["complete"]
            self.feats = z["feats"]
            self.panels = z["panels"]
            self.ok = z["ok"]
        else:
            self._build(files, raw)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_file,
                    grid=self.grid.to_numpy().astype("datetime64[D]").astype(str),
                    win_start=self.win_start,
                    complete=self.complete,
                    feats=self.feats,
                    panels=self.panels,
                    ok=self.ok,
                )

        self.T = len(self.tickers)
        self.F = self.complete.shape[1]
        self._post_build()

    # ------------------------------------------------------------------ build

    def _cache_key(self, files: list[Path], raw: Path) -> str:
        h = hashlib.sha256()
        h.update(f"v2;N={self.cfg.N};cal={self.cfg.calendar}".encode())  # v2: cache carries raw panels
        for f in files:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        if self.cfg.calendar == "benchmark":
            spy = raw / "SPY-VIX" / "SPY_daily.csv"
            st = spy.stat()
            h.update(f"SPY:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]

    def _build(self, files: list[Path], raw: Path) -> None:
        cfg = self.cfg
        if cfg.calendar == "benchmark":
            spy = _load_csv(raw / "SPY-VIX" / "SPY_daily.csv")
            grid = spy.index
        elif cfg.calendar == "union":
            dates: set = set()
            for f in files:
                dates.update(pd.to_datetime(pd.read_csv(f, usecols=["date"])["date"]))
            grid = pd.DatetimeIndex(sorted(dates))
        else:
            raise ValueError(f"unknown calendar mode {cfg.calendar!r}")

        T, G, N = len(files), len(grid), cfg.N
        F = G // N
        self.win_start = G - F * N  # tile backward from the newest day; partial leftover at the oldest end discarded
        self.grid = grid

        panels = np.full((5, T, G), np.nan, dtype=np.float32)
        for ti, f in enumerate(files):
            df = _load_csv(f).reindex(grid)
            panels[:, ti, :] = df[OHLCV].to_numpy(dtype=np.float32).T
        O, H, L, C, V = panels

        # complete = bar on every grid day, prices > 0, volume > 0 (log V/median is undefined at V = 0)
        ok = np.isfinite(panels).all(axis=0) & (panels[:4] > 0).all(axis=0) & (V > 0)
        complete = np.empty((T, F), dtype=bool)
        feats = np.zeros((T, F, N, 5), dtype=np.float32)
        for k in range(F):
            a = self.win_start + k * N
            b = a + N
            complete[:, k] = ok[:, a:b].all(axis=1)
            base = np.concatenate([O[:, a : a + 1], C[:, a : b - 1]], axis=1)  # day 0 has no prior close; it uses its own open
            with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices of incomplete tickers
                price = np.stack(
                    [np.log(X[:, a:b] / base) for X in (O, H, L, C)], axis=-1
                )  # [T, N, 4]
                med = np.nanmedian(V[:, a:b], axis=1, keepdims=True)
                vol = np.log(V[:, a:b] / med)[..., None]  # [T, N, 1]
            w = np.concatenate([price, vol], axis=-1)
            w[~complete[:, k]] = 0.0
            feats[:, k] = w
        assert np.isfinite(feats).all(), "non-finite features in complete cells"
        self.complete = complete
        self.feats = feats
        self.panels = panels  # raw [5, T, G] grids — offset-window features are cut from these
        self.ok = ok          # [T, G] per-day completeness (finite, prices > 0, V > 0)

    # ------------------------------------------------------ config-dependent

    def _post_build(self) -> None:
        cfg = self.cfg
        # holdout pool = tickers complete in every window (survivors spanning the whole grid)
        pool = np.flatnonzero(self.complete.all(axis=1))
        P = len(pool)
        k = int(round(P * cfg.holdout_frac))
        if P < 100:
            k = max(k, cfg.holdout_floor)  # small pool: floor the holdout so the retrieval statistic isn't anemic
        k = min(k, max(1, P // 2))
        rng = np.random.default_rng(cfg.holdout_seed)
        self.holdout_idx = np.sort(rng.choice(pool, size=k, replace=False)) if P else np.array([], int)
        self.pool_idx = pool
        hold = np.zeros(self.T, dtype=bool)
        hold[self.holdout_idx] = True
        self.train_idx = np.flatnonzero(~hold)
        self._train_mask = ~hold
        # per-day completeness prefix sums: span [a, a+N) complete iff the count equals N
        self._ok_cum = np.zeros((self.T, len(self.grid) + 1), dtype=np.int32)
        self._ok_cum[:, 1:] = np.cumsum(self.ok, axis=1, dtype=np.int32)

        # per-window universes: holdout exclusion is total — held-out tickers appear in no
        # universe, neither as observers nor as attention context
        train_complete = self.complete.copy()
        train_complete[self.holdout_idx, :] = False
        self._train_universe = [np.flatnonzero(train_complete[:, w]) for w in range(self.F)]
        self.usable_windows = [w for w in range(self.F) if len(self._train_universe[w]) >= cfg.Y]
        self.tau_prox = cfg.tau_prox if cfg.tau_prox is not None else len(self.usable_windows) / 10.0

        # clip thresholds: global symmetric quantiles of training tickers' log-returns,
        # computed once here and frozen into the artifact
        cells = self.feats[self.train_idx][self.complete[self.train_idx]]  # [n_cells, N, 5]
        rets = cells[..., :4].ravel()
        self.clip_lo = float(np.quantile(rets, 1.0 - cfg.q_clip))
        self.clip_hi = float(np.quantile(rets, cfg.q_clip))
        self.feats[..., :4] = np.clip(self.feats[..., :4], self.clip_lo, self.clip_hi)

    # ---------------------------------------------------------------- access

    def train_universe(self, w: int) -> np.ndarray:
        return self._train_universe[w]

    # ------------------------------------------------- offset windows (training)

    def base_start(self, w: int) -> int:
        return self.win_start + w * self.cfg.N

    def shifted_start(self, w: int, offset: int) -> int:
        """Window w's start under tiling offset delta (shift back in time);
        falls back to the base tiling when the shift runs off the grid's old end."""
        a = self.base_start(w) - offset
        return a if a >= 0 else self.base_start(w)

    def universe_at(self, a: int) -> np.ndarray:
        """Training tickers complete over the arbitrary span [a, a+N)."""
        N = self.cfg.N
        complete = (self._ok_cum[:, a + N] - self._ok_cum[:, a]) == N
        return np.flatnonzero(complete & self._train_mask)

    def window_feats_at(self, uni: np.ndarray, a: int) -> np.ndarray:
        """Normalized [len(uni), N, 5] features for the span [a, a+N) — the same
        operations the fixed tiling applies (day-0 base = own open, volume vs the
        span's median, frozen clip thresholds), at an arbitrary start day."""
        b = a + self.cfg.N
        O, H, L, C, V = (p[uni, a:b] for p in self.panels)
        base = np.concatenate([O[:, :1], C[:, :-1]], axis=1)
        price = np.stack([np.log(X / base) for X in (O, H, L, C)], axis=-1)
        price = np.clip(price, self.clip_lo, self.clip_hi)
        vol = np.log(V / np.median(V, axis=1, keepdims=True))[..., None]
        return np.concatenate([price, vol], axis=-1).astype(np.float32)

    def window_dates(self, w: int) -> tuple[str, str]:
        a = self.win_start + w * self.cfg.N
        b = a + self.cfg.N
        return str(self.grid[a].date()), str(self.grid[b - 1].date())

    def window_date_list(self, w: int) -> list[str]:
        a = self.win_start + w * self.cfg.N
        return [str(d.date()) for d in self.grid[a : a + self.cfg.N]]

    def summary(self) -> dict:
        return {
            "tickers": self.T,
            "grid_days": len(self.grid),
            "grid_first": str(self.grid[0].date()),
            "grid_last": str(self.grid[-1].date()),
            "windows": self.F,
            "usable_windows": len(self.usable_windows),
            "pool_size": int(len(self.pool_idx)),
            "holdout_size": int(len(self.holdout_idx)),
            "holdout": [self.tickers[i] for i in self.holdout_idx],
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "tau_prox": self.tau_prox,
        }


def normalize_window(df: pd.DataFrame, dates: list[str], clip_lo: float, clip_hi: float) -> np.ndarray | None:
    """Normalized 5-feature candles for one external ticker over one window's grid dates.

    Returns [N, 5] float32, or None when the ticker is incomplete in the window.
    Used by the inference path for tickers never seen during training.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    sub = df.reindex(idx)
    arr = sub[OHLCV].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all() or (arr[:, :4] <= 0).any() or (arr[:, 4] <= 0).any():
        return None
    O, C, V = arr[:, 0], arr[:, 3], arr[:, 4]
    base = np.concatenate([O[:1], C[:-1]])
    price = np.log(arr[:, :4] / base[:, None])
    price = np.clip(price, clip_lo, clip_hi)
    vol = np.log(V / np.median(V))[:, None]
    return np.concatenate([price, vol], axis=1).astype(np.float32)
