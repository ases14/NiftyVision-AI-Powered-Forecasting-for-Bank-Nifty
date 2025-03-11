import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
    return data

def compute_technical_indicators(data):
    # Use OHLCV data
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Moving averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # RSI calculation
    def compute_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = compute_rsi(data['Close'], window=14)

    # Volatility: 10-day rolling standard deviation
    data['Volatility'] = data['Close'].rolling(window=10).std()

    # Log Returns to capture relative changes
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))

    data.dropna(inplace=True)
    return data

def plot_indicators(data):
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(data.index, data['Close'], label='Close')
    plt.plot(data.index, data['MA10'], label='MA10')
    plt.plot(data.index, data['MA20'], label='MA20')
    plt.plot(data.index, data['MA50'], label='MA50')
    plt.title("Close Price and Moving Averages")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(data.index, data['RSI'], label='RSI', color='purple')
    plt.plot(data.index, data['Volatility'], label='Volatility', color='orange')
    plt.title("RSI and Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()
