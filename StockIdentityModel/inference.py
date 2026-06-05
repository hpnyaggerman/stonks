"""The embed() recipe: deterministic, no dropout, full view, single-group
scale, context = training universe present in the window, averaged over the
last K_inf windows.

    python -m StockIdentityModel.inference --artifact StockIdentityModel/artifacts/v1 --ticker AAPL
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, REPO_ROOT
from .data import _load_csv, normalize_window
from .model import IdentityEncoder


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
        self.recipe = json.loads((art / "context_recipe.json").read_text())
        torch.set_num_threads(self.cfg.num_threads)
        self.model = IdentityEncoder(self.cfg)
        self.model.load_state_dict(torch.load(art / "weights.pt", weights_only=True, map_location="cpu"))
        self.model.to(self.dev)
        self.model.eval()

        raw = Path(raw_dir) if raw_dir else REPO_ROOT / self.cfg.raw_dir
        stock_dir = raw / "stocksData"
        # context summary vectors per recipe window, computed once
        self._H_ctx: list[torch.Tensor] = []
        self._windows = self.recipe["windows"]
        with torch.no_grad():
            for win in self._windows:
                feats = []
                for t in win["context_tickers"]:
                    df = _load_csv(stock_dir / f"{t}_daily.csv")
                    f = normalize_window(df, win["dates"], self.clip_lo, self.clip_hi)
                    if f is None:
                        raise RuntimeError(
                            f"context ticker {t} incomplete in window {win['start']}..{win['end']}: "
                            "raw data no longer matches the artifact's context recipe"
                        )
                    feats.append(f)
                x = torch.from_numpy(np.stack(feats)).to(self.dev)
                self._H_ctx.append(self.model.temporal(x))

    @torch.no_grad()
    def embed_candles(self, df: pd.DataFrame, ticker: str | None = None) -> np.ndarray:
        """Embed a ticker from a raw candle DataFrame with columns
        date, open, high, low, close, volume. Returns [D] float32.

        A ticker already in a window's context is embedded as its own universe
        row (the canonical-table path); an unseen ticker is appended to the
        context and only its row is kept.
        """
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        zs = []
        for win, H_ctx in zip(self._windows, self._H_ctx):
            if ticker is not None and ticker in win["context_tickers"]:
                k = win["context_tickers"].index(ticker)
                zs.append(self.model.embed_rows(H_ctx)[k].cpu().numpy())
                continue
            f = normalize_window(df, win["dates"], self.clip_lo, self.clip_hi)
            if f is None:
                continue  # target incomplete in this window
            H_t = self.model.temporal(torch.from_numpy(f[None]).to(self.dev))
            H = torch.cat([H_t, H_ctx], dim=0)  # target row 0
            zs.append(self.model.embed_rows(H)[0].cpu().numpy())
        if not zs:
            raise ValueError("target ticker has no complete window among the inference windows")
        return np.stack(zs).mean(0)

    def embed_ticker(self, ticker: str, raw_dir: str | Path | None = None) -> np.ndarray:
        raw = Path(raw_dir) if raw_dir else REPO_ROOT / self.cfg.raw_dir
        return self.embed_candles(_load_csv(raw / "stocksData" / f"{ticker}_daily.csv"), ticker=ticker)


def main():
    ap = argparse.ArgumentParser(description="Embed a ticker with a Stock Identity artifact")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--device", default=None, help='"auto", "cpu", "cuda", or "cuda:N"')
    args = ap.parse_args()
    emb = Embedder(args.artifact, device=args.device)
    z = emb.embed_ticker(args.ticker)
    print(json.dumps({"ticker": args.ticker, "embedding": [round(float(x), 6) for x in z]}))


if __name__ == "__main__":
    main()
