"""The embed() recipe: deterministic, no dropout, full view, single-group
scale, context = training universe present in the window, averaged over the
last K_inf windows. Cross-market artifacts carry one recipe per market; a
target is embedded against its own market's context windows (its own calendar).

    python -m StockIdentityModel.inference --artifact StockIdentityModel/artifacts/v1 --ticker AAPL
    python -m StockIdentityModel.inference --artifact ... --ticker 600519 --market CN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, REPO_ROOT
from .data import _load_csv, load_parquet_frames, normalize_window
from .model import IdentityEncoder, rv_from_feats


class Embedder:
    def __init__(self, artifact_dir: str | Path, raw_dir: str | Path | None = None, device: str | None = None):
        art = Path(artifact_dir)
        if not art.is_absolute():
            art = REPO_ROOT / art
        doc = json.loads((art / "config.json").read_text())
        self.cfg = Config(**{k: v for k, v in doc["config"].items() if k in Config.__dataclass_fields__})
        if device is not None:
            self.cfg.device = device
        self.dev = torch.device(self.cfg.resolved_device())
        self.clip_lo = doc["normalization"]["clip_lo"]
        self.clip_hi = doc["normalization"]["clip_hi"]
        recipe = json.loads((art / "context_recipe.json").read_text())
        torch.set_num_threads(self.cfg.num_threads)
        self.model = IdentityEncoder(self.cfg)
        self.model.load_state_dict(torch.load(art / "weights.pt", weights_only=True, map_location="cpu"))
        self.model.to(self.dev)
        self.model.eval()

        # one window list per market ("US" = the legacy top-level recipe)
        self._mkts: dict[str, list[dict]] = {"US": recipe["windows"]}
        for name, blob in recipe.get("markets", {}).items():
            self._mkts[name] = blob["windows"]
        self._raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / self.cfg.raw_dir

        # candle source: per-ticker CSVs, or bulk-loaded parquet frames (one
        # filtered pass over the shards covering every context ticker)
        self._frames: dict[str, pd.DataFrame] | None = None
        if self.cfg.data_format == "parquet":
            needed = sorted({t for wins in self._mkts.values() for w in wins for t in w["context_tickers"]})
            min_date = min(w["dates"][0] for wins in self._mkts.values() for w in wins)
            self._frames = load_parquet_frames(self.cfg.parquet_dir, needed, min_date)

        # halt-tolerant markets mirror the training ingestion at inference; old
        # artifacts lack the config field -> () -> the historical code path
        self._halt_markets = tuple(self.cfg.halt_markets or ())

        # context summary vectors per (market, recipe window), computed once;
        # relational artifacts also retain each window's (returns, validity)
        # — the bias channel's day-level inputs (~MBs) — alongside H_ctx
        self._relational = self.model.context.relational
        self._ctx: dict[str, list[torch.Tensor]] = {}
        self._ctx_rv: dict[str, list[tuple | None]] = {}
        self._ticker_market: dict[str, str] = {}
        with torch.no_grad():
            for name, wins in self._mkts.items():
                Hs = []
                rvs = []
                for win in wins:
                    feats = []
                    for t in win["context_tickers"]:
                        self._ticker_market.setdefault(t, name)
                        f = self._normalize(self._frame(t), win["dates"], name)
                        if f is None:
                            raise RuntimeError(
                                f"context ticker {t} incomplete in window {win['start']}..{win['end']}: "
                                "raw data no longer matches the artifact's context recipe"
                            )
                        feats.append(f)
                    x = torch.from_numpy(np.stack(feats)).to(self.dev)
                    Hs.append(self.model.temporal(x))
                    rvs.append(rv_from_feats(x) if self._relational else None)
                self._ctx[name] = Hs
                self._ctx_rv[name] = rvs

    def _normalize(self, df: pd.DataFrame, dates: list[str], market: str) -> np.ndarray | None:
        return normalize_window(
            df, dates, self.clip_lo, self.clip_hi,
            halt_aware=market in self._halt_markets,
            halt_minfrac=self.cfg.halt_minfrac,
            max_ffill_days=self.cfg.max_ffill_days,
        )

    def _frame(self, ticker: str) -> pd.DataFrame:
        if self._frames is not None:
            if ticker not in self._frames:  # target ticker outside the preloaded context set
                extra = load_parquet_frames(self.cfg.parquet_dir, [ticker], "1900-01-01")
                if ticker not in extra:
                    raise FileNotFoundError(f"ticker {ticker} not found in parquet shards")
                self._frames[ticker] = extra[ticker]
            return self._frames[ticker]
        return _load_csv(self._raw_dir / "stocksData" / f"{ticker}_daily.csv")

    @torch.no_grad()
    def embed_candles(self, df: pd.DataFrame, ticker: str | None = None, market: str | None = None) -> np.ndarray:
        """Embed a ticker from a raw candle DataFrame with columns
        date, open, high, low, close, volume. Returns [D] float32.

        A ticker already in a window's context is embedded as its own universe
        row (the canonical-table path); an unseen ticker is appended to the
        context and only its row is kept. Market resolution: explicit `market`
        arg > membership in a market's context lists > "US". An unseen ticker
        from another calendar must pass `market` (its candles must be complete
        on that market's recipe windows).
        """
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        name = market or (self._ticker_market.get(ticker) if ticker else None) or "US"
        if name not in self._mkts:
            raise ValueError(f"unknown market {name!r}; artifact carries {sorted(self._mkts)}")
        zs = []
        for win, H_ctx, rv_ctx in zip(self._mkts[name], self._ctx[name], self._ctx_rv[name]):
            if ticker is not None and ticker in win["context_tickers"]:
                k = win["context_tickers"].index(ticker)
                zs.append(self.model.embed_rows(H_ctx, rv=rv_ctx)[k].cpu().numpy())
                continue
            f = self._normalize(df, win["dates"], name)
            if f is None:
                continue  # target incomplete in this window
            x_t = torch.from_numpy(f[None]).to(self.dev)
            H = torch.cat([self.model.temporal(x_t), H_ctx], dim=0)  # target row 0
            rv = None
            if self._relational:
                r_t, v_t = rv_from_feats(x_t)
                rv = (torch.cat([r_t, rv_ctx[0]], dim=0), torch.cat([v_t, rv_ctx[1]], dim=0))
            zs.append(self.model.embed_rows(H, rv=rv)[0].cpu().numpy())
        if not zs:
            raise ValueError("target ticker has no complete window among the inference windows")
        return np.stack(zs).mean(0)

    def embed_ticker(self, ticker: str, raw_dir: str | Path | None = None, market: str | None = None) -> np.ndarray:
        if raw_dir is not None and self.cfg.data_format != "parquet":
            df = _load_csv(Path(raw_dir) / "stocksData" / f"{ticker}_daily.csv")
        else:
            df = self._frame(ticker)
        return self.embed_candles(df, ticker=ticker, market=market)


def main():
    ap = argparse.ArgumentParser(description="Embed a ticker with a Stock Identity artifact")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--market", default=None, help="market whose recipe windows to embed against (default: auto-detect, else US)")
    ap.add_argument("--device", default=None, help='"auto", "cpu", "cuda", or "cuda:N"')
    args = ap.parse_args()
    emb = Embedder(args.artifact, device=args.device)
    z = emb.embed_ticker(args.ticker, market=args.market)
    print(json.dumps({"ticker": args.ticker, "embedding": [round(float(x), 6) for x in z]}))


if __name__ == "__main__":
    main()
