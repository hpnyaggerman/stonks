import numpy as np
import pandas as pd

default_window = 126
min_observation = 60

def add_fear_greed_correlation(df: pd.DataFrame, window_trading_days: int=default_window, min_obs: int=min_observation,) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    close = pd.to_numeric(df["close"], errors="coerce")
    fear_greed = pd.to_numeric(df["fear_greed"], errors="coerce")

    log_return = np.log(close).diff()

    correlation = fear_greed.rolling(window=window_trading_days, min_periods=min_obs,).corr(log_return)

    # First window_trading_days - 1 rows have no full window → leave NaN
    correlation.iloc[: window_trading_days - 1] = np.nan
    df["fear_greed_correlation"] = correlation.replace([np.inf, -np.inf], np.nan)
    return df