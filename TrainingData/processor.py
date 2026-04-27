"""
The purpose of this script is to process the raw data found in the indicators_data/raw folder
and place them in the indicators_data/processed folder.
"""
import os
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

def _parse_csv_dates(series):
    """Parse date column; tolerate M/D/Y vs ISO mixed exports (same issue as fear_greed)."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=False)


def process_file(csv_path, output_path, df_fear_greed=None):
    df = pd.read_csv(csv_path)
    df["date"] = _parse_csv_dates(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.normalize()

    df["YesterdayClose"] = df["close"].shift(1)
    df["YesterdayOpenLogR"]  = np.log(df["open"] / df["open"].shift(1))
    df["YesterdayHighLogR"]  = np.log(df["high"] / df["high"].shift(1))
    df["YesterdayLowLogR"]   = np.log(df["low"]  / df["low"].shift(1))
    # Zero volume prints log(0) → -inf and dropna() removes the tail (common on SPACs).
    _vol = df["volume"].clip(lower=1.0)
    df["YesterdayVolumeLogR"] = np.log(_vol / _vol.shift(1))
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
    std = df["close"].rolling(window=20).std().replace(0, np.nan).fillna(1e-10)
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

    # --- Fear & Greed index (market-wide): merge on date.
    # Raw prices often run past the last CNN Fear & Greed print; without handling, those
    # rows have NaN fear_greed and dropna() deletes the entire tail. Forward-fill uses
    # the last published index (standard for slow macro series). Leading NaNs (pre-2011)
    # remain until the first print and are still dropped at the end.
    if df_fear_greed is not None:
        df = df.merge(df_fear_greed, on="date", how="left")
        df["fear_greed"] = df["fear_greed"].ffill()
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

    # Flat or nearly flat windows yield NaN skew; inf can appear in log-ratios. Both
    # would otherwise drop valid thin-liquidity rows in dropna().
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["skew_5d"] = df["skew_5d"].fillna(0.0)

    df.dropna(inplace=True)
    df = df.drop(['open', 'high', 'low', 'volume'], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed: {output_path}")

def load_fear_greed():
    """Load market-wide Fear & Greed index; dates normalized to match stock date column."""
    path = os.path.join(RAW_DIR, "fear_greed.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # fear_greed.csv often mixes "M/D/YYYY" with "YYYY-MM-DD". Default to_datetime
        # infers one style and coerces the other block to NaT (~1000+ rows), so merge
        # leaves fear_greed NaN for all recent dates and process_file's dropna() chops
        # the last years of every stock. Use mixed-format parsing (pandas 2+).
        date_series = df["date"]
        try:
            parsed = pd.to_datetime(date_series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(date_series, errors="coerce", dayfirst=False)
        df["date"] = parsed
        dropped = int(df["date"].isna().sum())
        if dropped:
            print(
                f"[WARNING] fear_greed.csv: {dropped} rows had unparseable dates and were dropped."
            )
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if "fear_greed" not in df.columns:
            return None
        print(
            f"[INFO] fear_greed.csv usable range: {df['date'].min().date()} .. {df['date'].max().date()} "
            f"({len(df)} rows)"
        )
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
    # Compare to last completed equity session, not calendar "today" (avoids false
    # alarms on Mon morning before daily files include the new week, etc.).
    today = (pd.Timestamp(datetime.today().date()).normalize() - pd.offsets.BDay(1)).normalize()
    print(f"\n[INFO] Checking processed files vs last reference session {today.date()}...\n")
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
                last_date = pd.to_datetime(df["date"]).max().normalize()
                if last_date != today:
                    missing.append((file, last_date.date()))
            except Exception as e:
                print(f"[ERROR] Failed to check {file_path}: {e}")

    if missing:
        print(f"The following files do not reach the reference session ({today.date()}):")
        for filename, last_date in missing:
            print(f" - {filename}: Last date = {last_date}")
    else:
        print(f"All processed files include data through {today.date()}.")

def main():
    allowed_tickers = load_allowed_tickers()
    if allowed_tickers is None:
        print(f"[ERROR] {STOCK_LIST_PATH} not found or unreadable. Processor only runs for tickers in that file. Exiting.")
        return
    df_fear_greed = load_fear_greed()
    if df_fear_greed is not None:
        print(f"[INFO] Loaded Fear & Greed index: {len(df_fear_greed)} dates")

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
            # Benchmarks live outside stockList.csv; do not skip SPY/VIXY or stale
            # *_processed files will linger forever (see VIXY not in list).
            if subfolder == "stocksData" and ticker.upper() not in allowed_tickers:
                continue
            raw_file_path = os.path.join(raw_subdir, file)
            processed_file_path = os.path.join(processed_subdir, f"{os.path.splitext(file)[0]}_processed.csv")
            process_file(raw_file_path, processed_file_path, df_fear_greed=df_fear_greed)
        

if __name__ == "__main__":
    main()
    check_missing_today()