# utils/data_loader.py

import pandas as pd

def load_ohlcv(filepath):
    """
    Loads OHLCV data from a CSV file into a pandas DataFrame.
    Assumes columns: timestamp, open, high, low, close, volume
    
    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: DataFrame with datetime index and float columns
    """
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]  # Normalize column names

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        raise ValueError("CSV must contain 'timestamp' column")

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.astype(float)
    return df


