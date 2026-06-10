"""Data layer.

Builds, from raw OHLCV sources (per-ticker CSVs or long-format parquet shards):
  - one trading-day grid per market (US = SPY benchmark calendar by default;
    secondary markets = quorum-filtered union of their own tickers' dates),
  - non-overlapping N-day windows per market, anchored at the newest day,
    tiling backward,
  - the per-(ticker, window) completeness mask (bar on every grid day, V > 0;
    halt markets relax this to "traded or halted-carried" per day plus a
    minimum traded fraction — see config.halt_markets),
  - the normalized 5-feature candle tensor,
  - frozen global symmetric clip thresholds from training tickers only
    (pooled across markets in cross-market mode; traded days only for halt
    markets),
  - per-market holdout splits (the US draw is seed-identical to the historical
    single-market draw).

Single-market CSV mode at defaults is byte-identical to the historical
single-grid pipeline (min_history_days resolves to 0 on the csv backend).
Cross-market mode adds secondary markets on their own calendars; their
training windows are derived from US windows at step time via the per-day
dominance rule (MarketData.dominated_start).
"""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, REPO_ROOT

OHLCV = ["open", "high", "low", "close", "volume"]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
PRIMARY_MARKET = "US"


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


class MarketData:
    """One market's calendar, panels, windows, and features.

    Arrays are market-local (row r = self.tickers[r]); universes, holdouts, and
    feature lookups speak GLOBAL ticker ids via tids/row, so training and eval
    code never touches local rows."""

    def __init__(self, name: str, cfg: Config, tickers: list[str], grid: pd.DatetimeIndex, panels: np.ndarray):
        self.name = name
        self.cfg = cfg
        self.tickers = tickers
        self.grid = grid
        self.panels = panels  # [5, T_m, G_m] raw grids — offset-window features are cut from these
        self.halt = name in tuple(cfg.halt_markets or ())
        self._min_traded = math.ceil(cfg.halt_minfrac * cfg.N)
        self.synthesized_days = 0
        if self.halt:
            self._synthesize_carried_bars()  # idempotent: cache reloads re-run it as a no-op
        self._compute_masks()
        self.win_start = 0
        self.F = 0
        self.complete: np.ndarray | None = None  # [T_m, F_m]
        self.feats: np.ndarray | None = None     # [T_m, F_m, N, 5]
        self.tids: np.ndarray | None = None      # local row -> global ticker id
        self.row: np.ndarray | None = None       # global ticker id -> local row (-1 elsewhere)
        self.clip_lo: float | None = None
        self.clip_hi: float | None = None

    # ------------------------------------------------------------- build

    def _compute_masks(self) -> None:
        """traded = real bar on the grid day (finite, prices > 0, V > 0 — log V/median
        is undefined at V = 0): the historical `ok` predicate. Halt markets get a
        second mask, admit (V >= 0: traded OR a carried halt bar); elsewhere admit
        is the same array. `ok` stays as the legacy per-day-completeness alias."""
        p = self.panels
        base = np.isfinite(p).all(axis=0) & (p[:4] > 0).all(axis=0)
        self.traded = base & (p[4] > 0)
        self.admit = (base & (p[4] >= 0)) if self.halt else self.traded
        self.ok = self.traded

    def _synthesize_carried_bars(self) -> None:
        """Materialize the feed's own pre-2024 halt convention across all eras:
        a missing row strictly inside a ticker's listing span becomes a carried
        bar (O=H=L=C = the preceding bar's close, V=0) iff its date lies within
        max_ffill_days CALENDAR days of the ticker's last TRADED (V > 0) bar.
        Anchoring the cap at the last traded bar makes the rule a pure function
        of the vendor panels — synthesized bars are never traded, so re-running
        on cached (already-filled) panels fills nothing new. Long holes (vendor
        coverage gaps, delisting reviews) stay NaN past the cap."""
        p = self.panels
        finite = np.isfinite(p).all(axis=0)
        pos = (p[:4] > 0).all(axis=0)
        traded = finite & pos & (p[4] > 0)
        presence = np.isfinite(p[3])  # close present = a row exists at this cell
        gdays = self.grid.values.astype("datetime64[D]").astype(np.int64)
        cap = self.cfg.max_ffill_days
        ar = np.arange(p.shape[2])
        n_fill = 0
        for t in range(p.shape[1]):
            pres = presence[t]
            hit = np.flatnonzero(pres)
            if len(hit) < 2:
                continue
            first, last = hit[0], hit[-1]
            holes = ~pres & (ar > first) & (ar < last)
            if not holes.any():
                continue
            prev_pres = np.maximum.accumulate(np.where(pres, ar, -1))
            prev_trd = np.maximum.accumulate(np.where(traded[t], ar, -1))
            js = np.flatnonzero(holes & (prev_trd >= 0))
            js = js[gdays[js] - gdays[prev_trd[js]] <= cap]
            if not len(js):
                continue
            p[:4, t, js] = p[3, t, prev_pres[js]]
            p[4, t, js] = 0.0
            n_fill += len(js)
        self.synthesized_days = n_fill

    def apply_age_filter(self) -> None:
        """Drop tickers with fewer than the resolved min_history_days TRADED days on
        THIS market's grid (halted/carried days add no information and don't extend
        a ticker's apparent history; identical to the historical rule elsewhere)."""
        min_days = self.cfg.resolved_min_history_days()
        if min_days <= 0:
            return
        keep = self.traded.sum(axis=1) >= min_days
        if not keep.all():
            self.tickers = [t for t, k in zip(self.tickers, keep) if k]
            self.panels = self.panels[:, keep, :]
            self._compute_masks()
        if not self.tickers:
            raise RuntimeError(
                f"market {self.name}: no tickers left after min_history_days={min_days}"
            )

    def build_windows(self) -> None:
        cfg = self.cfg
        T, G, N = len(self.tickers), len(self.grid), cfg.N
        F = G // N
        self.F = F
        self.win_start = G - F * N  # tile backward from the newest day; partial leftover at the oldest end discarded
        O, H, L, C, V = self.panels
        complete = np.empty((T, F), dtype=bool)
        feats = np.zeros((T, F, N, 5), dtype=np.float32)
        for k in range(F):
            a = self.win_start + k * N
            b = a + N
            if self.halt:
                # every day admitted (traded or halted-carried) + enough traded days
                complete[:, k] = self.admit[:, a:b].all(axis=1) & (
                    self.traded[:, a:b].sum(axis=1) >= self._min_traded
                )
            else:
                complete[:, k] = self.ok[:, a:b].all(axis=1)
            base = np.concatenate([O[:, a : a + 1], C[:, a : b - 1]], axis=1)  # day 0 has no prior close; it uses its own open
            with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices of incomplete tickers
                price = np.stack(
                    [np.log(X[:, a:b] / base) for X in (O, H, L, C)], axis=-1
                )  # [T, N, 4]
                if self.halt:
                    # carried bars produce exact-zero price returns organically; the
                    # volume channel gets the traded-only median and sentinel 0.0
                    trd = self.traded[:, a:b]
                    med = np.nanmedian(np.where(trd, V[:, a:b], np.nan), axis=1, keepdims=True)
                    vol = np.where(trd, np.log(V[:, a:b] / med), 0.0)[..., None]  # [T, N, 1]
                else:
                    med = np.nanmedian(V[:, a:b], axis=1, keepdims=True)
                    vol = np.log(V[:, a:b] / med)[..., None]  # [T, N, 1]
            w = np.concatenate([price, vol], axis=-1)
            w[~complete[:, k]] = 0.0
            feats[:, k] = w
        assert np.isfinite(feats).all(), f"non-finite features in complete cells (market {self.name})"
        self.complete = complete
        self.feats = feats

    def attach_global(self, tids: np.ndarray, T_total: int) -> None:
        self.tids = tids
        self.row = np.full(T_total, -1, dtype=np.int64)
        self.row[tids] = np.arange(len(tids))

    def set_train_mask(self, global_train: np.ndarray) -> None:
        cfg = self.cfg
        self._train_local = global_train[self.tids]
        train_complete = self.complete & self._train_local[:, None]
        self._train_universe = [self.tids[np.flatnonzero(train_complete[:, w])] for w in range(self.F)]
        self.usable_windows = [w for w in range(self.F) if len(self._train_universe[w]) >= cfg.Y]
        # per-day prefix sums: span [a, a+N) complete iff the admitted count equals N
        # (admit == ok outside halt markets) — and, for halt markets, the traded
        # count clears ceil(halt_minfrac * N)
        self._ok_cum = np.zeros((len(self.tickers), len(self.grid) + 1), dtype=np.int32)
        self._ok_cum[:, 1:] = np.cumsum(self.admit, axis=1, dtype=np.int32)
        if self.halt:
            self._trd_cum = np.zeros_like(self._ok_cum)
            self._trd_cum[:, 1:] = np.cumsum(self.traded, axis=1, dtype=np.int32)

    # ------------------------------------------------------------ access

    def train_universe(self, w: int) -> np.ndarray:
        return self._train_universe[w]

    def base_start(self, w: int) -> int:
        return self.win_start + w * self.cfg.N

    def shifted_start(self, w: int, offset: int) -> int:
        """Window w's start under tiling offset delta (shift back in time);
        falls back to the base tiling when the shift runs off the grid's old end."""
        a = self.base_start(w) - offset
        return a if a >= 0 else self.base_start(w)

    def universe_at(self, a: int) -> np.ndarray:
        """Training tickers (global ids) complete over the arbitrary span [a, a+N)."""
        N = self.cfg.N
        complete = (self._ok_cum[:, a + N] - self._ok_cum[:, a]) == N
        if self.halt:
            complete &= (self._trd_cum[:, a + N] - self._trd_cum[:, a]) >= self._min_traded
        return self.tids[np.flatnonzero(complete & self._train_local)]

    def feats_at(self, uni: np.ndarray, w: int) -> np.ndarray:
        """Cached base-tiling features for global ids `uni` at window w: [len(uni), N, 5]."""
        return self.feats[self.row[uni], w]

    def window_feats_at(self, uni: np.ndarray, a: int) -> np.ndarray:
        """Normalized [len(uni), N, 5] features for the span [a, a+N) — the same
        operations the fixed tiling applies (day-0 base = own open, volume vs the
        span's median, frozen clip thresholds), at an arbitrary start day."""
        b = a + self.cfg.N
        rows = self.row[uni]
        O, H, L, C, V = (p[rows, a:b] for p in self.panels)
        base = np.concatenate([O[:, :1], C[:, :-1]], axis=1)
        price = np.stack([np.log(X / base) for X in (O, H, L, C)], axis=-1)
        price = np.clip(price, self.clip_lo, self.clip_hi)
        if self.halt:
            trd = V > 0  # spans come from universe_at: every day admitted, so finite and V >= 0
            with np.errstate(divide="ignore", invalid="ignore"):
                med = np.nanmedian(np.where(trd, V, np.nan), axis=1, keepdims=True)
                vol = np.where(trd, np.log(V / med), 0.0)[..., None]
        else:
            vol = np.log(V / np.median(V, axis=1, keepdims=True))[..., None]
        return np.concatenate([price, vol], axis=-1).astype(np.float32)

    def window_dates(self, w: int) -> tuple[str, str]:
        a = self.base_start(w)
        b = a + self.cfg.N
        return str(self.grid[a].date()), str(self.grid[b - 1].date())

    def window_date_list(self, w: int) -> list[str]:
        a = self.base_start(w)
        return [str(d.date()) for d in self.grid[a : a + self.cfg.N]]

    def dominated_start(self, foreign_dates: np.ndarray) -> int | None:
        """Largest own-grid window start satisfying per-day dominance against a
        foreign window: own day k may never fall after foreign day k, for every
        k. Endpoint alignment is not enough — the two calendars' day-counting
        drifts within a span, so a window can match on both ends and still
        violate dominance mid-window. Returns None when the own grid cannot
        dominate (the foreign window predates this market's history)."""
        g = self.grid.values
        idx = np.searchsorted(g, foreign_dates, side="right") - 1
        a = int((idx - np.arange(len(foreign_dates))).min())
        return a if a >= 0 else None


class StockData:
    """Multi-market container. `markets` maps name -> MarketData (primary "US"
    first); the legacy single-market surface (grid/feats/complete/universe
    methods, used by all pre-cross-market callers) delegates to the primary
    market, whose global ticker ids coincide with its local rows because the
    global ticker order is US-first."""

    def __init__(self, cfg: Config, repo_root: Path = REPO_ROOT, use_cache: bool = True):
        self.cfg = cfg
        repo_root = Path(repo_root)
        raw = repo_root / cfg.raw_dir
        if cfg.cross_market and cfg.data_format != "parquet":
            raise ValueError("cross_market=True requires data_format='parquet'")
        if cfg.calendar_validity not in ("rows", "traded"):
            raise ValueError(f"unknown calendar_validity {cfg.calendar_validity!r}")
        if not 0.0 < cfg.halt_minfrac <= 1.0:
            raise ValueError(f"halt_minfrac must be in (0, 1], got {cfg.halt_minfrac}")
        if cfg.holdout_pool_windows is not None and cfg.holdout_pool_windows < 2:
            raise ValueError(
                f"holdout_pool_windows must be >= 2 (consistency needs two appearances; "
                f"None = all windows), got {cfg.holdout_pool_windows}"
            )
        if cfg.data_format == "parquet":
            src_dir = repo_root / cfg.parquet_dir
            files = sorted(src_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet shards under {src_dir}")
        elif cfg.data_format == "csv":
            src_dir = raw / "stocksData"
            files = sorted(src_dir.glob("*_daily.csv"))
            if not files:
                raise FileNotFoundError(f"no raw candle CSVs under {src_dir}")
        else:
            raise ValueError(f"unknown data_format {cfg.data_format!r}")

        key = self._cache_key(files, raw)
        cache_file = CACHE_DIR / f"data_{key}.npz"
        if use_cache and cache_file.exists():
            self._load_cache(cache_file)
        else:
            self._build_markets(files, raw)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._save_cache(cache_file)
        self._assemble_global()
        self._post_build()

    # ------------------------------------------------------------------ build

    def _cache_key(self, files: list[Path], raw: Path) -> str:
        cfg = self.cfg
        h = hashlib.sha256()
        ex = "all" if cfg.exchanges is None else ",".join(sorted(cfg.exchanges))
        # v5: per-market arrays; key carries format, exchange filter, age filter,
        # calendar semantics, halt-ingestion constants, market map
        h.update(
            f"v5;fmt={cfg.data_format};N={cfg.N};cal={cfg.calendar};ex={ex};"
            f"minh={cfg.resolved_min_history_days()};q={cfg.min_calendar_quorum};"
            f"calv={cfg.calendar_validity};cfrac={cfg.calendar_frac};cfw={cfg.calendar_frac_window};"
            f"halt={','.join(sorted(cfg.halt_markets or ()))};hmf={cfg.halt_minfrac};"
            f"ffill={cfg.max_ffill_days}".encode()
        )
        if cfg.cross_market:
            mm = ";".join(f"{k}={v}" for k, v in sorted(cfg.exchange_market_map.items()))
            h.update(f";xm=1;map={mm}".encode())
        for f in files:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        if cfg.calendar == "benchmark":
            spy = raw / "SPY-VIX" / "SPY_daily.csv"
            st = spy.stat()
            h.update(f"SPY:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]

    def _grid_benchmark(self, raw: Path) -> pd.DatetimeIndex:
        return _load_csv(raw / "SPY-VIX" / "SPY_daily.csv").index

    def _build_markets(self, files: list[Path], raw: Path) -> None:
        cfg = self.cfg
        if cfg.data_format == "csv":
            tickers = [f.name[: -len("_daily.csv")] for f in files]
            if cfg.calendar == "benchmark":
                grid = self._grid_benchmark(raw)
            elif cfg.calendar == "union":
                dates: set = set()
                for f in files:
                    dates.update(pd.to_datetime(pd.read_csv(f, usecols=["date"])["date"]))
                grid = pd.DatetimeIndex(sorted(dates))
            else:
                raise ValueError(f"unknown calendar mode {cfg.calendar!r}")
            panels = np.full((5, len(files), len(grid)), np.nan, dtype=np.float32)
            for ti, f in enumerate(files):
                df = _load_csv(f).reindex(grid)
                panels[:, ti, :] = df[OHLCV].to_numpy(dtype=np.float32).T
            per_market = {PRIMARY_MARKET: (tickers, grid, panels)}
        else:
            per_market = self._read_parquet_markets(files, raw)

        self.markets: dict[str, MarketData] = {}
        for name, (tickers, grid, panels) in per_market.items():
            m = MarketData(name, cfg, tickers, grid, panels)
            m.apply_age_filter()
            m.build_windows()
            self.markets[name] = m

    def _read_parquet_markets(self, files: list[Path], raw: Path) -> dict:
        """Long-format shards (ticker, exchange, date, OHLCV) -> {market: (tickers, grid, panels)}.

        Shard-streamed passes (no shard is ever concatenated): tickers per
        market, then grids, then candle scatter. Rows off a market's grid drop
        out; duplicate (ticker, date) rows resolve last-write-wins, matching
        the CSV loader's drop_duplicates(keep="last")."""
        import pyarrow.parquet as pq

        cfg = self.cfg
        if cfg.cross_market:
            mmap: dict[str, str] | None = dict(cfg.exchange_market_map)
            allowed = sorted(mmap)
        else:
            mmap = None
            allowed = None if cfg.exchanges is None else sorted(cfg.exchanges)
        flt = [("exchange", "in", list(allowed))] if allowed is not None else None

        # pass 1: ticker -> market
        tk_market: dict[str, str] = {}
        for f in files:
            t = pq.read_table(f, columns=["ticker", "exchange"], filters=flt)
            te = t.group_by(["ticker", "exchange"]).aggregate([])
            for k, e in zip(te.column("ticker").to_pylist(), te.column("exchange").to_pylist()):
                tk_market.setdefault(k, mmap[e] if mmap else PRIMARY_MARKET)
        names = sorted(set(tk_market.values()))
        if cfg.cross_market and PRIMARY_MARKET not in names:
            raise RuntimeError("cross_market: no primary-market (US) tickers in the parquet data")
        per_tickers = {n: sorted(k for k, m in tk_market.items() if m == n) for n in names}

        # pass 2: grids (benchmark for US; quorum-filtered union of own dates otherwise)
        grids: dict[str, pd.DatetimeIndex] = {}
        for n in names:
            if n == PRIMARY_MARKET and cfg.calendar == "benchmark":
                grids[n] = self._grid_benchmark(raw)
            else:
                if mmap:
                    exs = sorted(e for e, m in mmap.items() if m == n)
                    mflt = [("exchange", "in", exs)]
                else:
                    mflt = flt
                counts: dict = {}
                read_cols = ["date"] if cfg.calendar_validity == "rows" else ["date", *OHLCV]
                for f in files:
                    t = pq.read_table(f, columns=read_cols, filters=mflt)
                    if t.num_rows == 0:
                        continue
                    d_all = t.column("date").to_numpy(zero_copy_only=False)
                    if cfg.calendar_validity == "traded":
                        # "market open" from the feed's own encoding: count only valid
                        # traded bars — placeholder days (V=0 forward fills) don't vote
                        o = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in OHLCV])
                        d_all = d_all[np.isfinite(o).all(axis=0) & (o[:4] > 0).all(axis=0) & (o[4] > 0)]
                        if not len(d_all):
                            continue
                    d, c = np.unique(d_all, return_counts=True)
                    for dd, cc in zip(d, c):
                        counts[dd] = counts.get(dd, 0) + int(cc)
                q = cfg.min_calendar_quorum if cfg.cross_market else 1
                cand = sorted(d for d, c in counts.items() if c >= q)
                if cfg.calendar_frac > 0 and cand:
                    # era-robust closure/partial guard: a candidate survives only if its
                    # count clears calendar_frac of the local (rolling-max) market size
                    vals = np.array([counts[d] for d in cand], dtype=np.float64)
                    roll = pd.Series(vals).rolling(cfg.calendar_frac_window, center=True, min_periods=1).max().to_numpy()
                    cand = [d for d, v, r in zip(cand, vals, roll) if v >= cfg.calendar_frac * r]
                grids[n] = pd.DatetimeIndex(pd.to_datetime(cand))

        # pass 3: scatter candles onto each market's (ticker, grid-day) cells
        tcode = {n: pd.Index(per_tickers[n]) for n in names}
        panels = {n: np.full((5, len(per_tickers[n]), len(grids[n])), np.nan, dtype=np.float32) for n in names}
        for f in files:
            t = pq.read_table(f, columns=["ticker", "exchange", "date", *OHLCV], filters=flt)
            if t.num_rows == 0:
                continue
            tk = t.column("ticker").to_numpy(zero_copy_only=False)
            dt = pd.to_datetime(t.column("date").to_numpy(zero_copy_only=False))
            ohlcv = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in OHLCV], axis=0)
            mk = pd.Series(t.column("exchange").to_numpy(zero_copy_only=False)).map(mmap).to_numpy() if mmap else None
            for n in names:
                sel = slice(None) if mk is None else (mk == n)
                if mk is not None and not sel.any():
                    continue
                codes = tcode[n].get_indexer(tk[sel])
                col = grids[n].get_indexer(dt[sel])
                vals = ohlcv[:, sel]
                valid = (codes >= 0) & (col >= 0)
                panels[n][:, codes[valid], col[valid]] = vals[:, valid]
            del t, ohlcv
        return {n: (per_tickers[n], grids[n], panels[n]) for n in names}

    # ------------------------------------------------------------------ cache

    def _save_cache(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {"market_names": np.array(list(self.markets))}
        for n, m in self.markets.items():
            p = n + "__"
            arrays[p + "tickers"] = np.array(m.tickers)
            arrays[p + "grid"] = m.grid.to_numpy().astype("datetime64[D]").astype(str)
            arrays[p + "complete"] = m.complete
            arrays[p + "feats"] = m.feats
            arrays[p + "panels"] = m.panels
        np.savez_compressed(path, **arrays)

    def _load_cache(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.markets = {}
        for n in [str(x) for x in z["market_names"]]:
            p = n + "__"
            m = MarketData(  # ctor re-runs the (idempotent) gap synthesis and recomputes
                # the masks from the cached panels; the age filter is already baked in
                n, self.cfg, [str(t) for t in z[p + "tickers"]], pd.DatetimeIndex(z[p + "grid"]), z[p + "panels"]
            )
            m.complete = z[p + "complete"]
            m.feats = z[p + "feats"]
            m.F = m.complete.shape[1]
            m.win_start = len(m.grid) - m.F * self.cfg.N
            self.markets[n] = m

    # ------------------------------------------------------ config-dependent

    def _assemble_global(self) -> None:
        if PRIMARY_MARKET not in self.markets:
            raise RuntimeError(f"primary market {PRIMARY_MARKET!r} missing from data")
        order = [PRIMARY_MARKET] + sorted(n for n in self.markets if n != PRIMARY_MARKET)
        self.markets = {n: self.markets[n] for n in order}
        self.tickers: list[str] = []
        mid: list[int] = []
        for i, m in enumerate(self.markets.values()):
            self.tickers.extend(m.tickers)
            mid.extend([i] * len(m.tickers))
        self.T = len(self.tickers)
        self.market_id = np.array(mid, dtype=np.int8)
        self.market_names = list(self.markets)
        self.tindex = {t: i for i, t in enumerate(self.tickers)}
        if len(self.tindex) != self.T:
            raise RuntimeError("duplicate ticker symbols across markets")
        pos = 0
        for m in self.markets.values():
            m.attach_global(np.arange(pos, pos + len(m.tickers)), self.T)
            pos += len(m.tickers)
        self.us = self.markets[PRIMARY_MARKET]
        self.secondary = [m for n, m in self.markets.items() if n != PRIMARY_MARKET]
        self.F = self.us.F

    def _post_build(self) -> None:
        cfg = self.cfg
        # holdout: per-market pools — survivors complete in every own window, except
        # that secondary markets may draw from the newest holdout_pool_windows windows
        # instead (no CN ticker spans 19 unbroken years; the newest block is what the
        # acceptance metrics read). The primary market ALWAYS keeps the all-windows
        # rule and draws with rng(holdout_seed) exactly as the single-market pipeline
        # did, so the US holdout is identical in every mode
        K = cfg.holdout_pool_windows
        hold_all = []
        for i, m in enumerate(self.markets.values()):
            if i == 0 or K is None:
                pool_mask = m.complete.all(axis=1)
            else:
                pool_mask = m.complete[:, -min(K, m.F):].all(axis=1)
            pool_local = np.flatnonzero(pool_mask)
            P = len(pool_local)
            k = int(round(P * cfg.holdout_frac))
            if P < 100:
                k = max(k, cfg.holdout_floor)  # small pool: floor the holdout so the retrieval statistic isn't anemic
            k = min(k, max(1, P // 2))
            rng = np.random.default_rng(cfg.holdout_seed if i == 0 else (cfg.holdout_seed, i))
            hold_local = np.sort(rng.choice(pool_local, size=k, replace=False)) if P else np.array([], int)
            m.pool_tids = m.tids[pool_local]
            m.holdout_tids = m.tids[hold_local] if P else np.array([], dtype=np.int64)
            hold_all.append(m.holdout_tids)
        self.holdout_idx = np.sort(np.concatenate(hold_all)).astype(np.int64)
        self.pool_idx = np.sort(np.concatenate([m.pool_tids for m in self.markets.values()]))
        hold = np.zeros(self.T, dtype=bool)
        hold[self.holdout_idx] = True
        self.train_idx = np.flatnonzero(~hold)
        self._train_mask = ~hold
        for m in self.markets.values():
            m.set_train_mask(self._train_mask)
        self.usable_windows = self.us.usable_windows
        self.tau_prox = cfg.tau_prox if cfg.tau_prox is not None else len(self.usable_windows) / 10.0

        # clip thresholds: global symmetric quantiles of training tickers' log-returns,
        # pooled over all markets' training-complete cells, computed once and frozen.
        # Halt markets contribute TRADED-day returns only: halted days are synthetic
        # zeros, and point mass at 0 with fraction f maps the q-quantile to
        # (q - f) / (1 - f) — both tails would tighten as a pure artifact of halt
        # frequency rather than any property of traded moves
        rets = []
        for m in self.markets.values():
            tr = np.flatnonzero(m._train_local)
            cells = m.feats[tr][m.complete[tr]]  # [n_cells, N, 5]
            if not len(cells):
                continue
            if m.halt:
                tw = np.stack(
                    [m.traded[:, m.win_start + k * cfg.N : m.win_start + (k + 1) * cfg.N] for k in range(m.F)],
                    axis=1,
                )  # [T, F, N]
                sel = tw[tr][m.complete[tr]]  # [n_cells, N]
                rets.append(cells[..., :4][sel].ravel())
            else:
                rets.append(cells[..., :4].ravel())
        rets = np.concatenate(rets)
        self.clip_lo = float(np.quantile(rets, 1.0 - cfg.q_clip))
        self.clip_hi = float(np.quantile(rets, cfg.q_clip))
        for m in self.markets.values():
            m.feats[..., :4] = np.clip(m.feats[..., :4], self.clip_lo, self.clip_hi)
            m.clip_lo, m.clip_hi = self.clip_lo, self.clip_hi

    # ----------------------------------- legacy single-market surface (primary)

    @property
    def grid(self) -> pd.DatetimeIndex:
        return self.us.grid

    @property
    def win_start(self) -> int:
        return self.us.win_start

    @property
    def complete(self) -> np.ndarray:
        return self.us.complete

    @property
    def feats(self) -> np.ndarray:
        return self.us.feats

    @property
    def panels(self) -> np.ndarray:
        return self.us.panels

    @property
    def ok(self) -> np.ndarray:
        return self.us.ok

    def train_universe(self, w: int) -> np.ndarray:
        return self.us.train_universe(w)

    def base_start(self, w: int) -> int:
        return self.us.base_start(w)

    def shifted_start(self, w: int, offset: int) -> int:
        return self.us.shifted_start(w, offset)

    def universe_at(self, a: int) -> np.ndarray:
        return self.us.universe_at(a)

    def window_feats_at(self, uni: np.ndarray, a: int) -> np.ndarray:
        return self.us.window_feats_at(uni, a)

    def window_dates(self, w: int) -> tuple[str, str]:
        return self.us.window_dates(w)

    def window_date_list(self, w: int) -> list[str]:
        return self.us.window_date_list(w)

    def summary(self) -> dict:
        s = {
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
        if self.cfg.cross_market:
            K = self.cfg.holdout_pool_windows
            s["markets"] = {
                n: {
                    "tickers": len(m.tickers),
                    "grid_first": str(m.grid[0].date()),
                    "grid_last": str(m.grid[-1].date()),
                    "windows": m.F,
                    "usable_windows": len(m.usable_windows),
                    "pool": int(len(m.pool_tids)),
                    "pool_windows": "all" if (n == PRIMARY_MARKET or K is None) else int(min(K, m.F)),
                    # halt diagnostics are pure functions of (cached panels, config) so
                    # they stay stable across build/load — safe inside data_meta
                    **(
                        {
                            "halt_tolerant": True,
                            "carried_days": int((m.admit & ~m.traded).sum()),
                            "min_traded_days": int(m._min_traded),
                        }
                        if m.halt
                        else {}
                    ),
                    "holdout": [self.tickers[i] for i in m.holdout_tids],
                }
                for n, m in self.markets.items()
            }
        return s


def normalize_window(
    df: pd.DataFrame,
    dates: list[str],
    clip_lo: float,
    clip_hi: float,
    halt_aware: bool = False,
    halt_minfrac: float = 0.9,
    max_ffill_days: int = 14,
) -> np.ndarray | None:
    """Normalized 5-feature candles for one external ticker over one window's grid dates.

    Returns [N, 5] float32, or None when the ticker is incomplete in the window.
    Used by the inference path for tickers never seen during training.

    halt_aware mirrors the halt-tolerant training ingestion (config.halt_markets)
    for external tickers: carried bars (V=0) are admitted as halted days, missing
    dates are synthesized to carried bars within max_ffill_days CALENDAR days of
    the ticker's last traded bar (the same grid-free rule training uses), the
    window needs >= ceil(halt_minfrac * N) traded days, and halted days carry
    exact-zero price returns plus the volume sentinel 0.0 with the median over
    traded days only.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if not halt_aware:
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

    sub = df.reindex(idx)
    arr = sub[OHLCV].to_numpy(dtype=np.float32)
    missing = ~np.isfinite(arr[:, 3])  # no close -> no row at this window date
    if missing.any():
        frame = df[OHLCV].to_numpy(dtype=np.float64)
        fin_close = np.isfinite(frame[:, 3])
        bar_dates = df.index.values[fin_close]
        bar_close = frame[fin_close, 3]
        trd = np.isfinite(frame).all(axis=1) & (frame[:, :4] > 0).all(axis=1) & (frame[:, 4] > 0)
        traded_dates = df.index.values[trd]
        for j in np.flatnonzero(missing):
            d = idx.values[j]
            pb = np.searchsorted(bar_dates, d) - 1  # last bar strictly before d
            if pb < 0 or not (bar_dates > d).any():  # no prior bar, or no later bar (no trailing fill)
                continue
            pt = np.searchsorted(traded_dates, d) - 1  # last TRADED bar strictly before d (the cap anchor)
            if pt < 0:
                continue
            gap = (d - traded_dates[pt]).astype("timedelta64[D]").astype(int)
            if gap > max_ffill_days:
                continue
            arr[j, :4] = bar_close[pb]
            arr[j, 4] = 0.0
    if not np.isfinite(arr).all() or (arr[:, :4] <= 0).any() or (arr[:, 4] < 0).any():
        return None
    traded = arr[:, 4] > 0
    if int(traded.sum()) < math.ceil(halt_minfrac * len(idx)):
        return None
    O, C, V = arr[:, 0], arr[:, 3], arr[:, 4]
    base = np.concatenate([O[:1], C[:-1]])
    price = np.log(arr[:, :4] / base[:, None])
    price = np.clip(price, clip_lo, clip_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        med = np.median(V[traded])
        vol = np.where(traded, np.log(V / med), 0.0)[:, None]
    return np.concatenate([price, vol], axis=1).astype(np.float32)


def load_parquet_frames(parquet_dir: str | Path, tickers: list[str], min_date: str) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames (date-indexed, same shape _load_csv returns) from
    long-format shards, restricted to rows at/after min_date (ISO date string;
    string comparison is chronological for ISO dates). Used by the inference
    path when the artifact was trained from parquet."""
    import pyarrow.parquet as pq

    pdir = Path(parquet_dir)
    if not pdir.is_absolute():
        pdir = REPO_ROOT / pdir
    files = sorted(pdir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {pdir}")
    flt = [("ticker", "in", list(tickers)), ("date", ">=", min_date)]
    parts = []
    for f in files:
        t = pq.read_table(f, columns=["ticker", "date", *OHLCV], filters=flt)
        if t.num_rows:
            parts.append(t.to_pandas())
    if not parts:
        return {}
    df = pd.concat(parts, ignore_index=True)
    out = {}
    for tk, g in df.groupby("ticker"):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
        out[str(tk)] = g[OHLCV].astype("float64")
    return out
