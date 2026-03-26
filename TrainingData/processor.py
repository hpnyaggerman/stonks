"""
The purpose of this script is to process the raw data found in the indicators_data/raw folder
and place them in the indicators_data/processed folder.
"""
import json
import os
import re
import pandas as pd
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
import sys
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
RAW_DIR = os.path.join(_script_dir, "indicators_data", "raw")
PROCESSED_DIR = os.path.join(_script_dir, "indicators_data", "processed")
STOCK_LIST_PATH = os.path.join(_script_dir, "stockList.csv")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_allowed_tickers():
    """Load the set of ticker symbols from stockList.csv (only these will be processed)."""
    path = STOCK_LIST_PATH
    if not os.path.exists(path):
        path = os.path.join(_script_dir, "..", "stockList.csv")
        path = os.path.normpath(path)
    if not os.path.exists(path):
        print(f"[WARNING] stockList.csv not found; processing all stocks in raw.")
        return None
    try:
        df = pd.read_csv(path, header=None)
        tickers = set(df.iloc[:, 0].astype(str).str.strip().str.upper())
        tickers.discard("")
        # Skip if first row looks like a header
        for header in ("SYMBOL", "TICKER", "Symbol", "Ticker"):
            tickers.discard(header)
        print(f"[INFO] Loaded {len(tickers)} tickers from {path}")
        return tickers
    except Exception as e:
        print(f"[WARNING] Failed to load {path}: {e}; processing all stocks in raw.")
        return None

def process_file(csv_path, output_path, df_fear_greed=None):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df["YesterdayClose"] = df["close"].shift(1)
    df["YesterdayOpenLogR"]  = np.log(df["open"] / df["open"].shift(1))
    df["YesterdayHighLogR"]  = np.log(df["high"] / df["high"].shift(1))
    df["YesterdayLowLogR"]   = np.log(df["low"]  / df["low"].shift(1))
    df["YesterdayVolumeLogR"] = np.log(df["volume"] / df["volume"].shift(1))
    df["YesterdayCloseLogR"] = np.log(df["close"] / df["YesterdayClose"])

    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA30"] = df["close"].rolling(window=30).mean()

    df["DayOfWeek"] = df["date"].dt.weekday         # 0 = Monday, 6 = Sunday
    df["DayOfMonth"] = df["date"].dt.day            # 1 to 31
    df["MonthNumber"] = df["date"].dt.month         # 1 = January, 12 = December

    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

    # Relative strength index (RSI) calculation (avoid div-by-zero when avg_loss is 0)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan).fillna(1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    #Moving average convergence divergence (MACD) calculation
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    #Bollinger Bands - Volitility indicator
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BollingerUpper"] = ma20 + 2 * std20
    df["BollingerLower"] = ma20 - 2 * std20

    #Rolling Volatility
    df["Volatility_10"] = df["close"].pct_change().rolling(window=10).std()
    df["Volatility_20"] = df["close"].pct_change().rolling(window=20).std()
    df["Volatility_30"] = df["close"].pct_change().rolling(window=30).std()

    #On-Balance Volume (OBV) - Volume indicator
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    #Z-score of close
    mean = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["ZScore"] = (df["close"] - mean) / std
    
    # --- Insider Buying Merge ---
    # Only for stock files (not SPY-VIX)
    ticker = os.path.basename(csv_path).split("_")[0]
    insider_dir = os.path.join(RAW_DIR, "insiderBuying")
    insider_path = os.path.join(insider_dir, f"{ticker}_insider_trades_daily.csv")
    if os.path.exists(insider_path):
        df_insider = safe_read_insider(insider_path)
        df = df.merge(
            df_insider[["date", "insider_shares", "insider_amount", "insider_buy_flag"]],
            on="date", how="left"
        )
        df["insider_shares"] = df["insider_shares"].fillna(0)
        df["insider_amount"] = df["insider_amount"].fillna(0)
        df["insider_buy_flag"] = df["insider_buy_flag"].fillna(-1).astype(int)
    else:
        df["insider_shares"] = 0
        df["insider_amount"] = 0
        df["insider_buy_flag"] = -1

    # --- Sentiment Data Merge ---
    sentiment_dir = os.path.join(RAW_DIR, "sentiment")
    sentiment_path = os.path.join(sentiment_dir, f"{ticker}_sentiment_daily.csv")

    if os.path.exists(sentiment_path):
        try:
            df_sentiment = pd.read_csv(sentiment_path, parse_dates=["date"])
            df = df.merge(
                df_sentiment[["date", "sentiment", "num_articles"]],
                on="date", how="left"
            )
            df["sentiment"] = df["sentiment"].fillna(0)
            df["num_articles"] = df["num_articles"].fillna(0)
        except Exception as e:
            print(f"[WARNING] Failed to merge sentiment for {ticker}: {e}")
            df["sentiment"] = 0
            df["num_articles"] = 0
    else:
        df["sentiment"] = 0
        df["num_articles"] = 0

    # --- Political trades (per ticker): daily aggregates merged on transaction_date -> date
    df = merge_political_daily_features(df, ticker)

    # --- Fear & Greed index (market-wide): merge on date; keep only rows with real values.
    # Rows with no fear_greed (e.g. before 2011) are left as NA and dropped by dropna() below.
    if df_fear_greed is not None:
        df = df.merge(df_fear_greed, on="date", how="left")
        # do not fill: leave NaN so dropna() later removes those rows
        from featuresPy.fear_greed_correlation import add_fear_greed_correlation
        df = add_fear_greed_correlation(df, window_trading_days=126, min_obs=60)
    else:
        df["fear_greed"] = 50.0

    # --- If you add VIX/SPY market data: merge on same date only (no shift).
    #     Using VIX at date T to predict return T->T+1 is OK. Using VIX at T+1 would be leakage.

    #Overnight gap
    # Overnight gap % (predicts t+1 move)
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    # Abnormal volume z-score (avoid div-by-zero when rolling std is 0)
    rolling_vol = df['volume'].rolling(20)
    vol_std = rolling_vol.std().replace(0, np.nan).fillna(1e-10)
    df['abnormal_vol'] = (df['volume'] - rolling_vol.mean()) / vol_std
    #Short term realized volatility
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    #Momentum 
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    #Skewness
    df['skew_5d'] = df['close'].pct_change().rolling(5).skew()
    #Intraday change
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    #Sentiment change
    df['sentiment_change'] = df['sentiment'] - df['sentiment'].shift(1)

    df.dropna(inplace=True)
    df = df.drop(['open', 'high', 'low', 'volume'], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed: {output_path}")

POLITICAL_TRADES_JSON = os.path.join(RAW_DIR, "political_trades", "all_ticker_transactions.json")
POLITICAL_TRADES_OUT = os.path.join(PROCESSED_DIR, "political_trades")

POLITICAL_TRADE_COLUMNS = [
    "transaction_date",
    "json_block_ticker",
    "ticker_raw",
    "ticker_symbol",
    "owner",
    "asset_description",
    "asset_type",
    "transaction_type",
    "amount",
    "comment",
    "senator",
    "ptr_link",
]


def _safe_filename_ticker(ticker):
    """Tickers like BRK/B contain '/' which Windows treats as a path separator."""
    s = str(ticker).strip()
    return re.sub(r'[/\\:*?"<>|]+', "-", s)


# Senate disclosure amount ranges -> (ordinal 1..n, midpoint USD) for model-friendly numeric features
_POLITICAL_AMOUNT_META = {
    "$1,001 - $15,000": (1, 8_000.0),
    "$15,001 - $50,000": (2, 32_500.0),
    "$50,001 - $100,000": (3, 75_000.0),
    "$100,001 - $250,000": (4, 175_000.0),
    "$250,001 - $500,000": (5, 375_000.0),
    "$500,001 - $1,000,000": (6, 750_000.0),
    "$1,000,001 - $5,000,000": (7, 3_000_000.0),
    "$5,000,001 - $25,000,000": (8, 15_000_000.0),
    "$25,000,001 - $50,000,000": (9, 37_500_000.0),
}

POLITICAL_MERGED_FEATURE_COLS = [
    "polit_trade_count",
    "polit_purchase_count",
    "polit_sale_count",
    "polit_exchange_count",
    "polit_option_count",
    "polit_stock_count",
    "polit_other_asset_count",
    "polit_distinct_senators",
    "polit_amount_max_ord",
    "polit_amount_sum_logmid",
]


def _political_amount_ord_logmid(amount_val):
    """Map disclosure amount string to (ordinal, log10(midpoint)). Unknown -> (0, nan)."""
    if amount_val is None or (isinstance(amount_val, float) and np.isnan(amount_val)):
        return 0, np.nan
    key = str(amount_val).strip()
    meta = _POLITICAL_AMOUNT_META.get(key)
    if meta is None:
        return 0, np.nan
    ord_, mid = meta
    return ord_, np.log10(mid)


def _political_txn_bucket(type_val):
    s = str(type_val or "").strip().lower()
    if "purchase" in s:
        return "purchase"
    if "sale" in s:
        return "sale"
    if "exchange" in s:
        return "exchange"
    return "other"


def _resolve_political_trades_csv_path(ticker):
    """Pick political export CSV; try raw basename variant with '_' -> '-' for class shares."""
    t = str(ticker).strip()
    candidates = [_safe_filename_ticker(t)]
    if "_" in t:
        candidates.append(_safe_filename_ticker(t.replace("_", "-")))
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    for name in ordered:
        p = os.path.join(POLITICAL_TRADES_OUT, f"{name}_political_trades.csv")
        if os.path.isfile(p):
            return p
    return os.path.join(POLITICAL_TRADES_OUT, f"{ordered[0]}_political_trades.csv")


def build_political_daily_aggregates(ticker):
    """
    One row per calendar date with numeric features derived from transaction_type, amount,
    asset_type, and senator (distinct count per day).
    """
    path = _resolve_political_trades_csv_path(ticker)
    if not os.path.isfile(path):
        return None
    try:
        pol = pd.read_csv(path, parse_dates=["transaction_date"])
    except Exception as e:
        print(f"[WARNING] Could not read political trades for {ticker}: {e}")
        return None
    if pol.empty:
        return None
    pol = pol.dropna(subset=["transaction_date"])
    pol["date"] = pd.to_datetime(pol["transaction_date"], errors="coerce").dt.normalize()
    pol = pol.dropna(subset=["date"])

    amt_meta = pol["amount"].map(_political_amount_ord_logmid)
    pol["_amt_ord"] = amt_meta.map(lambda x: x[0])
    pol["_logmid"] = amt_meta.map(lambda x: x[1])
    pol["_txn"] = pol["transaction_type"].map(_political_txn_bucket)
    pol["_atype"] = pol["asset_type"].fillna("").astype(str)

    def _is_option_row(a):
        return "option" in a.lower()

    pol["_is_opt"] = pol["_atype"].map(_is_option_row)
    pol["_is_stock"] = pol["_atype"].str.strip().str.lower().eq("stock")

    rows = []
    for d, g in pol.groupby("date", sort=False):
        logmid = g["_logmid"].sum(min_count=1)
        if pd.isna(logmid):
            logmid = 0.0
        max_ord = int(g["_amt_ord"].max()) if len(g) else 0
        rows.append(
            {
                "date": d,
                "polit_trade_count": len(g),
                "polit_purchase_count": int((g["_txn"] == "purchase").sum()),
                "polit_sale_count": int((g["_txn"] == "sale").sum()),
                "polit_exchange_count": int((g["_txn"] == "exchange").sum()),
                "polit_option_count": int(g["_is_opt"].sum()),
                "polit_stock_count": int(g["_is_stock"].sum()),
                "polit_other_asset_count": int(
                    (~g["_is_opt"] & ~g["_is_stock"] & (g["_atype"].str.strip() != "")).sum()
                ),
                "polit_distinct_senators": g["senator"].nunique(dropna=True),
                "polit_amount_max_ord": max_ord,
                "polit_amount_sum_logmid": float(logmid),
            }
        )
    return pd.DataFrame(rows)


def merge_political_daily_features(df, ticker):
    pol_daily = build_political_daily_aggregates(ticker)
    if pol_daily is None or pol_daily.empty:
        for c in POLITICAL_MERGED_FEATURE_COLS:
            df[c] = 0
        return df
    df = df.merge(pol_daily, on="date", how="left")
    for c in POLITICAL_MERGED_FEATURE_COLS:
        if c == "polit_distinct_senators":
            df[c] = df[c].fillna(0).astype(int)
        elif c == "polit_amount_max_ord":
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = df[c].fillna(0.0)
            if c != "polit_amount_sum_logmid":
                df[c] = df[c].astype(int)
    return df


def _clean_political_ticker_field(raw):
    """Strip Yahoo link HTML from ticker cell, e.g. <a ...>AAPL</a> -> AAPL."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    m = re.search(r">([A-Z][A-Z0-9.\-]*)</a>", s, re.I)
    if m:
        return m.group(1).upper().replace("-", ".")
    plain = re.sub(r"<[^>]+>", "", s).strip().upper().replace("-", ".")
    return plain if plain and plain != "--" else ""


def process_political_trades_export(allowed_tickers):
    """
    Read all_ticker_transactions.json and write one CSV per ticker in stockList under
    processed/political_trades/{TICKER}_political_trades.csv. process_file() also merges
    daily political aggregates into each stock's *_processed.csv (see POLITICAL_MERGED_FEATURE_COLS).
    Slashes in tickers (e.g. BRK/B) are replaced with '-' in filenames so paths stay valid.
    """
    if not os.path.isfile(POLITICAL_TRADES_JSON):
        print(f"[INFO] Political trades file not found ({POLITICAL_TRADES_JSON}); skipping export.")
        return
    os.makedirs(POLITICAL_TRADES_OUT, exist_ok=True)
    try:
        with open(POLITICAL_TRADES_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read political trades JSON: {e}")
        return
    if not isinstance(data, list):
        print("[ERROR] Political trades JSON must be a list of {ticker, transactions} objects.")
        return

    # Rows keyed by stockList ticker (uppercase)
    rows_by_ticker = {t: [] for t in allowed_tickers}
    seen_by_ticker = {t: set() for t in allowed_tickers}

    for block in data:
        outer = (block.get("ticker") or "").strip().upper().replace("-", ".")
        txs = block.get("transactions") or []
        if not txs:
            continue
        for tr in txs:
            if not isinstance(tr, dict):
                continue
            raw_t = tr.get("ticker", "")
            inner_sym = _clean_political_ticker_field(raw_t)
            # Attribute to stockList tickers: JSON block ticker match, or parsed transaction ticker
            targets = set()
            if outer and outer not in ("--", "NAN") and outer in allowed_tickers:
                targets.add(outer)
            if inner_sym and inner_sym in allowed_tickers:
                targets.add(inner_sym)
            if not targets:
                continue
            ptr = str(tr.get("ptr_link") or "")
            tdate = str(tr.get("transaction_date") or "")
            sen = str(tr.get("senator") or "")
            row = {
                "transaction_date": tr.get("transaction_date"),
                "json_block_ticker": outer if outer not in ("--", "") else "",
                "ticker_raw": raw_t,
                "ticker_symbol": inner_sym,
                "owner": tr.get("owner"),
                "asset_description": tr.get("asset_description"),
                "asset_type": tr.get("asset_type"),
                "transaction_type": tr.get("type"),
                "amount": tr.get("amount"),
                "comment": tr.get("comment"),
                "senator": tr.get("senator"),
                "ptr_link": tr.get("ptr_link"),
            }
            for t in targets:
                dedupe_key = (ptr, tdate, sen)
                if dedupe_key in seen_by_ticker[t]:
                    continue
                seen_by_ticker[t].add(dedupe_key)
                rows_by_ticker[t].append(row.copy())

    n_written = 0
    n_empty = 0
    for ticker in sorted(allowed_tickers):
        rows = rows_by_ticker.get(ticker, [])
        safe_name = _safe_filename_ticker(ticker)
        out_path = os.path.join(POLITICAL_TRADES_OUT, f"{safe_name}_political_trades.csv")
        if not rows:
            df = pd.DataFrame(columns=POLITICAL_TRADE_COLUMNS)
            df.to_csv(out_path, index=False)
            n_empty += 1
            continue
        df = pd.DataFrame(rows)
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], format="%m/%d/%Y", errors="coerce"
        )
        df = df.sort_values("transaction_date", na_position="last")
        df = df[POLITICAL_TRADE_COLUMNS]
        df.to_csv(out_path, index=False)
        n_written += 1
        print(f"[INFO] Political trades: {ticker} -> {len(df)} rows -> {out_path}")

    print(
        f"[INFO] Political trades export: {n_written} tickers with data, "
        f"{n_empty} empty (headers only), dir={POLITICAL_TRADES_OUT}"
    )

def load_fear_greed():
    """Load market-wide Fear & Greed index; dates normalized to match stock date column."""
    path = os.path.join(RAW_DIR, "fear_greed.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if "fear_greed" not in df.columns:
            return None
        return df[["date", "fear_greed"]]
    except Exception as e:
        print(f"[WARNING] Failed to load fear_greed.csv: {e}")
        return None


def safe_read_insider(insider_path):
    try:
        df_insider = pd.read_csv(insider_path)
        # Convert to date only (ignore time, handle both 'YYYY-MM-DD' and 'YYYY-MM-DD-HH:MM' formats)
        df_insider['date'] = pd.to_datetime(
            df_insider['date'].astype(str).str[:10], errors='coerce'
        )
        # Drop rows with invalid dates
        bad_rows = df_insider[df_insider['date'].isna()]
        if not bad_rows.empty:
            print(f"[WARNING] Bad date(s) found in {insider_path}:")
            print(bad_rows)
        df_insider = df_insider.dropna(subset=['date'])
        # Group by date and compute net shares/amount
        grouped = df_insider.groupby('date').agg({
            'shares': 'sum',
            'amount': 'sum'
        }).reset_index()
        # Compute net buy_flag for the day: 1 if net shares > 0, 0 if net shares < 0, -1 if net shares == 0
        grouped['insider_buy_flag'] = grouped['shares'].apply(lambda s: 1 if s > 0 else (0 if s < 0 else -1))
        # Rename columns for merge
        grouped = grouped.rename(columns={
            'shares': 'insider_shares',
            'amount': 'insider_amount'
        })
        return grouped[['date', 'insider_shares', 'insider_amount', 'insider_buy_flag']]
    except Exception as e:
        print(f"[ERROR] Failed to process {insider_path}: {e}")
        return pd.DataFrame(columns=['date', 'insider_shares', 'insider_amount', 'insider_buy_flag'])
from datetime import datetime

def check_missing_today():
    today = pd.Timestamp(datetime.today().date())
    print("\n[INFO] Checking which files are missing today's data...\n")
    missing = []

    for subfolder in ["SPY-VIX", "stocksData"]:
    #for subfolder in ["stocksData"]:
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        if not os.path.exists(processed_subdir):
            continue
        for file in os.listdir(processed_subdir):
            if not file.endswith("_processed.csv"):
                continue
            file_path = os.path.join(processed_subdir, file)
            try:
                df = pd.read_csv(file_path, parse_dates=["date"])
                if df.empty:
                    missing.append((file, "EMPTY"))
                    continue
                last_date = df["date"].max()
                if last_date != today:
                    missing.append((file, last_date.date()))
            except Exception as e:
                print(f"[ERROR] Failed to check {file_path}: {e}")

    if missing:
        print("The following files are missing today's data:")
        for filename, last_date in missing:
            print(f" - {filename}: Last date = {last_date}")
    else:
        print("All files contain today's data.")

def main():
    allowed_tickers = load_allowed_tickers()
    if allowed_tickers is None:
        print(f"[ERROR] {STOCK_LIST_PATH} not found or unreadable. Processor only runs for tickers in that file. Exiting.")
        return
    df_fear_greed = load_fear_greed()
    if df_fear_greed is not None:
        print(f"[INFO] Loaded Fear & Greed index: {len(df_fear_greed)} dates")

    process_political_trades_export(allowed_tickers)

    #for subfolder in ["stocksData"]:
    for subfolder in ["SPY-VIX", "stocksData"]:
        raw_subdir = os.path.join(RAW_DIR, subfolder)
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        os.makedirs(processed_subdir, exist_ok=True)

        for file in os.listdir(raw_subdir):
            if file.startswith("._"):
                print(f"[Skipping] macOS metadata: {file}")
                continue
            if not file.endswith(".csv"):
                continue
            ticker = os.path.splitext(file)[0].split("_")[0]
            if ticker.upper() not in allowed_tickers:
                continue
            raw_file_path = os.path.join(raw_subdir, file)
            processed_file_path = os.path.join(processed_subdir, f"{os.path.splitext(file)[0]}_processed.csv")
            process_file(raw_file_path, processed_file_path, df_fear_greed=df_fear_greed)
        

if __name__ == "__main__":
    main()
    check_missing_today()