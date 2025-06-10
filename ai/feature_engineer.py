# ai/feature_engineer.py

import pandas as pd
import ta # pandas-ta library
import numpy as np

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a more comprehensive set of technical indicators.
    Assumes 'open', 'high', 'low', 'close', 'volume' columns exist.
    """
    df_fe = df.copy()

    # --- Volatility Indicators ---
    # Bollinger Bands
    df_fe['bb_bbm'] = ta.volatility.BollingerBands(df_fe['close'], window=20).bollinger_mavg()
    df_fe['bb_bbh'] = ta.volatility.BollingerBands(df_fe['close'], window=20).bollinger_hband()
    df_fe['bb_bbl'] = ta.volatility.BollingerBands(df_fe['close'], window=20).bollinger_lband()
    df_fe['bb_wband'] = ta.volatility.BollingerBands(df_fe['close'], window=20).bollinger_wband()

    # Average True Range (ATR)
    df_fe['atr'] = ta.volatility.AverageTrueRange(df_fe['high'], df_fe['low'], df_fe['close'], window=14).average_true_range()

    # --- Momentum Indicators ---
    # Existing RSI & MACD (from engine/indicators, but re-calculate here for feature set completeness)
    df_fe["rsi"] = ta.momentum.RSIIndicator(close=df_fe["close"], window=14).rsi()
    macd = ta.trend.MACD(close=df_fe["close"], window_slow=26, window_fast=12, window_sign=9)
    df_fe["macd"] = macd.macd()
    df_fe["macd_signal"] = macd.macd_signal()
    df_fe["macd_diff"] = df_fe["macd"] - df_fe["macd_signal"] # MACD Histogram

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df_fe['high'], df_fe['low'], df_fe['close'], window=14, smooth_window=3)
    df_fe['stoch_k'] = stoch.stoch()
    df_fe['stoch_d'] = stoch.stoch_signal()

    # --- Trend Indicators ---
    # Existing EMAs
    df_fe["ema_20"] = ta.trend.EMAIndicator(close=df_fe["close"], window=20).ema_indicator()
    df_fe["ema_50"] = ta.trend.EMAIndicator(close=df_fe["close"], window=50).ema_indicator()
    df_fe["ema_100"] = ta.trend.EMAIndicator(close=df_fe["close"], window=100).ema_indicator() # Added longer EMA

    # Awesome Oscillator (AO)
    df_fe['ao'] = ta.momentum.awesome_oscillator(df_fe['high'], df_fe['low'])
    
    # --- Volume Indicators ---
    # On-Balance Volume (OBV)
    df_fe['obv'] = ta.volume.OnBalanceVolumeIndicator(df_fe['close'], df_fe['volume']).on_balance_volume()
    
    # --- Price Action Features ---
    df_fe['range'] = df_fe['high'] - df_fe['low']
    df_fe['body_size'] = abs(df_fe['open'] - df_fe['close'])
    df_fe['upper_wick'] = df_fe['high'] - np.maximum(df_fe['open'], df_fe['close'])
    df_fe['lower_wick'] = np.minimum(df_fe['open'], df_fe['close']) - df_fe['low']
    df_fe['close_to_open'] = df_fe['close'] - df_fe['open']
    df_fe['close_to_prev_close'] = df_fe['close'].diff()

    # --- Normalization/Scaling (Simple way: relative to price or range) ---
    df_fe['range_norm'] = df_fe['range'] / df_fe['close']
    df_fe['body_size_norm'] = df_fe['body_size'] / df_fe['close']
    df_fe['upper_wick_norm'] = df_fe['upper_wick'] / df_fe['close']
    df_fe['lower_wick_norm'] = df_fe['lower_wick'] / df_fe['close']

    # --- Price Changes ---
    df_fe['price_change'] = df_fe['close'].pct_change() * 100 # Percentage change

    return df_fe

def add_lagged_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Adds lagged versions of selected features.
    """
    df_lagged = df.copy()
    features_to_lag = [
        'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d',
        'ema_20', 'ema_50', 'ema_100', 'atr', 'bb_bbm', 'bb_wband', 'obv',
        'price_change', 'range_norm', 'body_size_norm'
    ]

    for feature in features_to_lag:
        if feature in df_lagged.columns: # Check if feature exists after previous steps
            for lag in lags:
                df_lagged[f'{feature}_lag_{lag}'] = df_lagged[feature].shift(lag)
        else:
            print(f"Warning: Feature '{feature}' not found for lagging.")

    return df_lagged

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to generate all features from raw OHLCV data.
    """
    df_features = add_advanced_indicators(df)
    df_features = add_lagged_features(df_features)

    # Drop any rows with NaN values that result from indicator calculations or lagging
    # This is important before feeding to ML models
    df_features = df_features.dropna()
    
    # Drop OHLCV and volume if you only want to train on indicators
    # Or keep them if you want the model to see raw price data as well
    # For now, let's keep them and let the model decide importance
    
    return df_features