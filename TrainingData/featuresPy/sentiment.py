import requests
import pandas as pd
import os
import time
import math
from datetime import datetime, timedelta

RAW_STOCKS_DIR = 'TrainingData/indicators_data/raw/stocksData'
SENTIMENT_DIR = 'TrainingData/indicators_data/raw/sentiment'
os.makedirs(SENTIMENT_DIR, exist_ok=True)

def load_api_key():
    #Load AlphaVantage API key from config.json
    import json, os

    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            key = cfg.get("ALPHA_VANTAGE_KEY")

            if key is None or key.strip() == "":
                raise ValueError(
                    "\nERROR: Your AlphaVantage API key is missing.\n"
                    "Open config.json and add:\n"
                    '{ "ALPHA_VANTAGE_KEY": "YOUR_KEY_HERE" }\n'
                )
            return key

    except FileNotFoundError:
        raise FileNotFoundError(
            "\nERROR: config.json is missing.\n"
            "Create it (or copy config.example.json) and add your API key.\n"
        )
    
def get_date_range_from_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date')
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    return start_date, end_date

def get_last_sentiment_date(sentiment_path):
    if not os.path.exists(sentiment_path):
        return None
    df = pd.read_csv(sentiment_path, parse_dates=['date'])
    if df.empty:
        return None
    return df['date'].max()

SAVE_EVERY_N = 50          # flush to disk every N new rows instead of every call
NO_NEWS_SKIP_THRESHOLD = 14 # after 14 consecutive no-news days, jump ahead
NO_NEWS_SKIP_DAYS = 90      # how far to jump when streak is hit
ETA_PRINT_INTERVAL = 30     # print an ETA line at most every N seconds
AV_SENTIMENT_EARLIEST = pd.Timestamp('2010-01-01')  # Alpha Vantage NEWS_SENTIMENT hard floor


class ProgressTracker:
    """Tracks API calls and wall time to produce a live ETA across all tickers."""

    def __init__(self, total_weekdays_remaining):
        self.total_weekdays = total_weekdays_remaining
        self.days_done = 0          # weekdays processed (API call or skipped via streak)
        self.api_calls = 0          # actual HTTP requests made
        self.days_skipped = 0       # days filled without an API call (streak jumps)
        self.start_time = time.time()
        self._last_eta_print = 0.0

    def tick_api_call(self, days_covered=1):
        self.api_calls += 1
        self.days_done += days_covered

    def tick_skip(self, days_covered=1):
        self.days_skipped += days_covered
        self.days_done += days_covered

    def tick_already_done(self, days_covered):
        """For days already in CSV — they reduce remaining work but took no time."""
        self.total_weekdays = max(0, self.total_weekdays - days_covered)

    def _fmt_time(self, seconds):
        if seconds < 0 or not math.isfinite(seconds):
            return "???"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def maybe_print_eta(self, force=False):
        now = time.time()
        if not force and (now - self._last_eta_print) < ETA_PRINT_INTERVAL:
            return
        self._last_eta_print = now
        elapsed = now - self.start_time
        if self.days_done <= 0 or elapsed < 1:
            return

        days_remaining = max(0, self.total_weekdays - self.days_done)
        rate = self.days_done / elapsed  # days processed per second
        eta_sec = days_remaining / rate if rate > 0 else float('inf')

        pct = 100.0 * self.days_done / max(1, self.total_weekdays)
        print(
            f"  [ETA] {pct:.1f}% done | "
            f"{self.days_done}/{self.total_weekdays} days | "
            f"{self.api_calls} API calls | "
            f"{self.days_skipped} skipped | "
            f"elapsed {self._fmt_time(elapsed)} | "
            f"remaining ~{self._fmt_time(eta_sec)}"
        )

def _is_empty_feed(data):
    """True when the API says 'no articles' — regardless of response shape."""
    if 'Information' in data:
        return True
    if 'feed' in data and not data['feed']:
        return True
    if str(data.get('items', '')) == '0':
        return True
    return False

def _is_rate_limit(data):
    """True when Alpha Vantage returns a rate-limit / usage message."""
    info = data.get('Information', '') or data.get('Note', '')
    return 'Thank you for using Alpha Vantage' in str(info)

def fetch_sentiment_for_range(ticker, start_date, end_date, sentiment_path, API_KEY,
                              tracker=None):
    if os.path.exists(sentiment_path):
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        sentiment_df.to_csv(sentiment_path, index=False)
        rows = sentiment_df.to_dict('records')
        existing_dates = set(sentiment_df['date'])
    else:
        rows = []
        existing_dates = set()

    # Fill pre-2010 dates locally — the API rejects anything before 2010-01-01
    date = start_date
    pre2010_count = 0
    while date < AV_SENTIMENT_EARLIEST and date <= end_date:
        ds = date.strftime('%Y-%m-%d')
        if ds not in existing_dates:
            rows.append({'date': ds, 'sentiment': None, 'num_articles': 0})
            existing_dates.add(ds)
            pre2010_count += 1
        date += timedelta(days=1)
    if pre2010_count > 0:
        print(f"  Filled {pre2010_count} pre-2010 days locally (API floor is 2010-01-01)")
        if tracker:
            tracker.tick_skip(pre2010_count)
        _flush_rows(rows, sentiment_path)

    no_news_streak = 0
    new_since_save = 0

    while date <= end_date:
        date_str = date.strftime('%Y-%m-%d')

        if date_str in existing_dates:
            date += timedelta(days=1)
            continue

        if date.weekday() >= 5:
            rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
            existing_dates.add(date_str)
            new_since_save += 1
            date += timedelta(days=1)
            continue

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': API_KEY,
            'time_from': date.strftime('%Y%m%dT0000'),
            'time_to': date.strftime('%Y%m%dT2359'),
            'sort': 'LATEST',
            'limit': 100
        }
        try:
            r = requests.get('https://www.alphavantage.co/query', params=params)
            data = r.json()

            if _is_rate_limit(data):
                print(f"  Rate-limited on {date_str} for {ticker} — waiting 60s")
                time.sleep(60)
                continue

            if 'feed' in data and data['feed']:
                no_news_streak = 0
                feed = data['feed']
                num_articles = len(feed)
                scores = [item['overall_sentiment_score'] for item in feed if 'overall_sentiment_score' in item]
                avg_score = sum(scores) / len(scores) if scores else None
                rows.append({'date': date_str, 'sentiment': avg_score, 'num_articles': num_articles})
                if tracker:
                    tracker.tick_api_call(1)
                time.sleep(0.85)
            elif _is_empty_feed(data):
                no_news_streak += 1
                rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
                if tracker:
                    tracker.tick_api_call(1)
                if no_news_streak >= NO_NEWS_SKIP_THRESHOLD:
                    skip_end = min(date + timedelta(days=NO_NEWS_SKIP_DAYS), end_date)
                    print(f"  No news for {no_news_streak} days — skipping {date_str} → {skip_end.strftime('%Y-%m-%d')} for {ticker}")
                    skip_count = 0
                    while date < skip_end:
                        date += timedelta(days=1)
                        ds = date.strftime('%Y-%m-%d')
                        if ds not in existing_dates:
                            rows.append({'date': ds, 'sentiment': None, 'num_articles': 0})
                            existing_dates.add(ds)
                            new_since_save += 1
                            skip_count += 1
                    if tracker and skip_count > 0:
                        tracker.tick_skip(skip_count)
                    no_news_streak = 0
                time.sleep(0.15)
            else:
                print(f"  Unexpected API response on {date_str} for {ticker}: {str(data)[:120]}")
                rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
                if tracker:
                    tracker.tick_api_call(1)
                time.sleep(0.85)
        except Exception as e:
            print(f"  Error on {date_str} for {ticker}: {e}")
            rows.append({'date': date_str, 'sentiment': None, 'num_articles': 0})
            if tracker:
                tracker.tick_api_call(1)
            time.sleep(0.85)

        existing_dates.add(date_str)
        new_since_save += 1
        date += timedelta(days=1)

        if new_since_save >= SAVE_EVERY_N:
            _flush_rows(rows, sentiment_path)
            new_since_save = 0

        if tracker:
            tracker.maybe_print_eta()

    _flush_rows(rows, sentiment_path)
    print(f"Saved sentiment data to {sentiment_path}")


def _flush_rows(rows, path):
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df.to_csv(path, index=False)

def fetch_data_with_retry(api_url, max_retries=10, retry_delay=60):
    retries = 0
    while retries < max_retries:
        response = requests.get(api_url)
        data = response.json()
        if 'Information' in data and 'Thank you for using Alpha Vantage' in data['Information']:
            print(f"API limit reached, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1
            continue
        if 'feed' in data and not data['feed']:
            print("No data returned, retrying in 30 seconds...")
            time.sleep(1)
            retries += 1
            continue
        return data
    print("Max retries reached. Could not fetch valid data.")
    return None

def _count_weekdays(start, end):
    """Count weekdays (Mon-Fri) between two dates, inclusive."""
    if start > end:
        return 0
    count = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


def _prescan_tickers(raw_dir, sentiment_dir):
    """Build a work list: [(ticker, stock_csv, sentiment_csv, fetch_start, stock_end, weekdays_remaining)]."""
    work = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith('_daily.csv'):
            continue
        ticker = filename.split('_')[0]
        stock_csv = os.path.join(raw_dir, filename)
        sent_csv = os.path.join(sentiment_dir, f"{ticker}_sentiment_daily.csv")

        stock_start, stock_end = get_date_range_from_csv(stock_csv)

        last_sent = get_last_sentiment_date(sent_csv)
        if last_sent is not None:
            last_sent = pd.to_datetime(last_sent)
            if last_sent >= stock_end:
                continue  # already up-to-date
            fetch_start = last_sent + timedelta(days=1)
        else:
            fetch_start = stock_start

        wd = _count_weekdays(fetch_start, stock_end)
        if wd > 0:
            work.append((ticker, stock_csv, sent_csv, fetch_start, stock_end, wd))
    return work


def main():
    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    print("Pre-scanning tickers to estimate total work...")
    work = _prescan_tickers(RAW_STOCKS_DIR, SENTIMENT_DIR)

    if not work:
        print("All tickers are already up to date.")
        return

    total_weekdays = sum(w[5] for w in work)
    total_tickers = len(work)
    print(f"  {total_tickers} tickers need data | ~{total_weekdays} weekdays to fetch\n")

    tracker = ProgressTracker(total_weekdays)

    for idx, (ticker, stock_csv, sent_csv, fetch_start, stock_end, wd) in enumerate(work, 1):
        if os.path.exists(sent_csv):
            df_existing = pd.read_csv(sent_csv, parse_dates=['date'])
            df_existing['date'] = pd.to_datetime(df_existing['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df_existing.to_csv(sent_csv, index=False)

        pct = 100.0 * idx / total_tickers
        print(f"\n[{idx}/{total_tickers}] ({pct:.1f}%) {ticker}  |  {fetch_start.date()} → {stock_end.date()}  |  ~{wd} weekdays")
        fetch_sentiment_for_range(ticker, fetch_start, stock_end, sent_csv, API_KEY,
                                  tracker=tracker)
        tracker.maybe_print_eta(force=True)
        print(f"Completed {ticker}")

    tracker.maybe_print_eta(force=True)
    print("\nAll tickers processed.")

if __name__ == "__main__":
    API_KEY = load_api_key() 
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Script ran in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")