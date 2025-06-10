import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, and EMA indicators to the DataFrame.
    Expects OHLCV columns: open, high, low, close, volume.
    """
    df = df.copy()
    
    # RSI
    # Ensure window is an integer and close series is not empty
    if 'close' in df.columns and not df['close'].empty:
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    else:
        df["rsi"] = pd.NA # Assign NaN if 'close' column is missing or empty
        print("Warning: 'close' column missing or empty for RSI calculation.")

    # MACD
    if 'close' in df.columns and not df['close'].empty:
        macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    else:
        df["macd"] = pd.NA
        df["macd_signal"] = pd.NA
        print("Warning: 'close' column missing or empty for MACD calculation.")
    
    # EMA
    if 'close' in df.columns and not df['close'].empty:
        df["ema_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()
    else:
        df["ema_20"] = pd.NA
        df["ema_50"] = pd.NA
        print("Warning: 'close' column missing or empty for EMA calculation.")
    
    return df