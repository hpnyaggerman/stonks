"""Stock Identity Encoder.

Learns a 32-dimensional identity embedding per stock from raw daily OHLCV
candles: stable across time windows, peer groups, group sizes, and dropout
draws; far apart for different tickers; derivable from a ticker's own price
behavior alone and from its relations to other tickers alone; and free of
per-ticker parameters, so tickers never seen in training can be embedded at
inference.

Independent of the v4 forecasting pipeline; reads only raw OHLCV candles.
"""
