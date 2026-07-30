"""Build the session census: one row per feed date with its distinct US-ticker count.

Scans every parquet shard under ``TrainingData/ohlcv_parts`` filtered to
``features_v5.US_EXCHANGES`` and writes ``TrainingData/session_census.csv``
(``date,n_names``, date-sorted, committed). The census feeds the phantom-session
filter (``features_v5.load_us_ohlcv``), the pre-registered rolling-origin seams,
and the survivorship logging; its sha256 goes into the run manifest and the
pre-registration, so the output must be deterministic for a given feed.

Also prints per-year universe death counts (tickers whose maximum feed date falls
in that year) for survivorship visibility, and the census file sha256.
"""
from __future__ import annotations

import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features_v5 import PARTS_DIR, US_EXCHANGES

CENSUS_PATH = PROJECT_ROOT / "TrainingData" / "session_census.csv"
# Bias keeps pre-1970 (negative days-since-epoch) dates packable into unsigned
# low-32 key bits and exactly recoverable on decode.
DAY_BIAS = np.int64(2 ** 31)


def build_census(parts_dir=PARTS_DIR, exchanges=US_EXCHANGES):
    """Return (census_df, deaths_by_year). ``census_df`` has ``date,n_names``."""
    import pyarrow.parquet as pq

    ticker_codes = {}
    pair_keys = []          # per shard: unique (ticker_code << 32) | (day + DAY_BIAS)
    parts = sorted(glob.glob(str(Path(parts_dir) / "*.parquet")))
    if not parts:
        raise SystemExit(f"no parquet shards under {parts_dir}")
    for part in parts:
        tbl = pq.read_table(part, columns=["ticker", "date"],
                            filters=[("exchange", "in", list(exchanges))])
        if not tbl.num_rows:
            continue
        df = tbl.to_pandas()
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        ok = dates.notna().to_numpy()
        days = dates.to_numpy()[ok].astype("datetime64[D]").astype(np.int64)
        tickers = pd.Series(df["ticker"].to_numpy()[ok])
        for t in pd.unique(tickers):
            if t not in ticker_codes:
                ticker_codes[t] = len(ticker_codes)
        codes = tickers.map(ticker_codes).to_numpy(dtype=np.int64)
        keys = np.unique((codes << 32) | (days + DAY_BIAS))
        pair_keys.append(keys)
        print(f"[census] {Path(part).name}: {tbl.num_rows} US rows, "
              f"{len(keys)} unique (ticker, date) pairs")
    all_keys = np.unique(np.concatenate(pair_keys))
    days = (all_keys & np.int64(0xFFFFFFFF)) - DAY_BIAS
    codes = all_keys >> 32
    unique_days, counts = np.unique(days, return_counts=True)
    census = pd.DataFrame({
        "date": pd.to_datetime(unique_days.astype("datetime64[D]")).strftime("%Y-%m-%d"),
        "n_names": counts.astype(np.int64),
    })
    max_day = np.full(len(ticker_codes), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_day, codes, days)
    death_years = pd.to_datetime(max_day.astype("datetime64[D]")).year
    deaths = pd.Series(death_years).value_counts().sort_index()
    return census, deaths


def main():
    census, deaths = build_census()
    CENSUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    census.to_csv(CENSUS_PATH, index=False)
    digest = hashlib.sha256(CENSUS_PATH.read_bytes()).hexdigest()
    print(f"[census] wrote {CENSUS_PATH}: {len(census)} sessions "
          f"{census['date'].iloc[0]} .. {census['date'].iloc[-1]}")
    print(f"[census] sha256 {digest}")
    print("[census] universe deaths by year (tickers whose last feed date falls in that year):")
    for year, n in deaths.items():
        print(f"[census]   {year}: {n}")


if __name__ == "__main__":
    main()
