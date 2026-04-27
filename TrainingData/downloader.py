"""
The purpose of this script is to download stock and market data
for training the model.
"""

import subprocess
import sys

PYTHON = sys.executable

# Run stock data downloader: close/open/high/low/volume.
subprocess.run([PYTHON, "TrainingData/featuresPy/stockScrapper.py"], check=True)

# Run market data downloader for SPY/VIX.
subprocess.run([PYTHON, "TrainingData/featuresPy/markets.py"], check=True)

# Run insider buying downloader.
subprocess.run([PYTHON, "TrainingData/featuresPy/insiderbuying.py"], check=True)

# Run sentiment downloader.
subprocess.run([PYTHON, "TrainingData/featuresPy/sentiment.py"], check=True)