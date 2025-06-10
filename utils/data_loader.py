import pandas as pd

def load_ohlcv(filepath: str) -> pd.DataFrame:
    """
    Loads OHLCV data from a CSV file into a pandas DataFrame.
    Assumes columns: timestamp, open, high, low, close, volume
    
    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: DataFrame with datetime index and float columns
    """
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    except KeyError:
        raise ValueError("CSV must contain 'timestamp' column for indexing.")
    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")

    df.columns = [col.lower() for col in df.columns] # Normalize column names

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col} in data file.")

    # Convert OHLCV columns to numeric, coercing errors
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any NaN values in OHLCV columns after conversion
    df.dropna(subset=required_columns, inplace=True)
    
    if df.empty:
        raise ValueError("Dataframe is empty after loading or cleaning. Check CSV content.")

    return df