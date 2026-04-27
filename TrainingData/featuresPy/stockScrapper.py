from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import time
import json

def download_alpha_vantage_daily(tickers, api_key, save_folder="TrainingData/indicators_data/raw/stocksData", delay_sec=1):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ts = TimeSeries(key=api_key, output_format='pandas')
    counter = 0
    start_time = time.time()

    for ticker in tickers:
        counter += 1
        try:
            filepath = os.path.join(save_folder, f"{ticker}_daily.csv")
            file_exists = os.path.exists(filepath)
            should_call_api = True

            if file_exists:
                existing_data = pd.read_csv(filepath, parse_dates=['date'])
                last_date = existing_data['date'].max()

                # If the latest date is today, no API call needed
                if pd.to_datetime(last_date).date() >= pd.Timestamp.now().date():
                    print(f"{ticker} is already up to date (latest date: {last_date.date()})")
                    should_call_api = False

            if should_call_api:
                data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
                data.reset_index(inplace=True)
                data.rename(columns={
                    'date': 'date',
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                }, inplace=True)
                data.sort_values('date', inplace=True)
                data.reset_index(drop=True, inplace=True)

                if file_exists:
                    new_data = data[data['date'] > last_date]
                    if not new_data.empty:
                        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                        updated_data.sort_values('date', inplace=True)
                        updated_data.to_csv(filepath, index=False)
                        print(f"Appended {len(new_data)} new rows to {filepath}")
                    else:
                        print(f"{ticker} has no new data (latest date: {last_date.date()})")
                else:
                    data.to_csv(filepath, index=False)
                    print(f"Saved {ticker} data to {filepath}")

            # Progress & ETA
            elapsed = time.time() - start_time
            percent = (counter / len(tickers)) * 100
            avg_time = elapsed / counter
            eta_seconds = avg_time * (len(tickers) - counter)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            print(f'Completion: {percent:.2f}% | ETA: {eta_str}')

            # Only wait if we made an API call
            if should_call_api:
                print(f" Waiting {delay_sec} seconds to respect API limits...")
                time.sleep(delay_sec)

        except Exception as e:
            print(f" Error downloading {ticker}: {e}")

def load_api_key():
    #Load AlphaVantage API key from config.json, with error checking.
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            key = cfg.get("ALPHA_VANTAGE_KEY")

            if key is None or key.strip() == "":
                raise ValueError(
                    "\nERROR: Your AlphaVantage API key is missing.\n"
                    "Open config.json and add your key:\n"
                    '{ "ALPHA_VANTAGE_KEY": "YOUR_KEY_HERE" }\n'
                )

            return key

    except FileNotFoundError:
        raise FileNotFoundError(
            "\nERROR: config.json is missing.\n"
            "Create it (or copy config.example.json) and add your API key.\n"
        )
    
if __name__ == "__main__":
    api_key = load_api_key()
    with open(r'TrainingData\stockList.csv', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]

    print(tickers)
    download_alpha_vantage_daily(tickers, api_key)
