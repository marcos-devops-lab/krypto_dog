import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
import yaml
import logging

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads OHLCV data from a CSV file.
    Assumes the CSV has columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    The 'timestamp' column should be convertible to datetime.
    """
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index() # Ensure data is sorted by time
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def add_technical_indicators(df):
    """
    Adds a comprehensive set of technical indicators to the DataFrame.
    Uses 'ta' library, which is a wrapper for TA-Lib.
    """
    logging.info("Adding technical indicators...")

    # Volatility Indicators
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['Bollinger_Mid'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['Bollinger_Bandwidth'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
    df['Bollinger_Perc_B'] = ta.volatility.bollinger_pband(df['close'], window=20, window_dev=2)
    df['Donchian_High'] = ta.volatility.donchian_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['Donchian_Low'] = ta.volatility.donchian_channel_lband(df['high'], df['low'], df['close'], window=20)

    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['close'])
    df['MACD_Diff'] = ta.trend.macd_diff(df['close'])
    df['Stoch_RSI'] = ta.momentum.stochrsi(df['close'], window=14)
    df['Stoch_RSI_K'] = ta.momentum.stochrsi_k(df['close'], window=14)
    df['Stoch_RSI_D'] = ta.momentum.stochrsi_d(df['close'], window=14)
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['ADX_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['ADX_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)

    # Trend Indicators
    df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)

    # Relative position of price to EMAs
    df['Price_vs_EMA9'] = df['close'] - df['EMA_9']
    df['Price_vs_EMA20'] = df['close'] - df['EMA_20']
    df['EMA9_vs_EMA20'] = df['EMA_9'] - df['EMA_20'] # Crossover indicator
    df['EMA20_vs_EMA50'] = df['EMA_20'] - df['EMA_50'] # Crossover indicator

    # Volume Indicators
    df['Volume_MA'] = df['volume'].rolling(window=20).mean() # Direct calculation for Volume Moving Average
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['CMF'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)


    logging.info("Technical indicators added.")
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10, 20]):
    """
    Adds lagged versions of key features to capture temporal dependencies.
    """
    logging.info(f"Adding lagged features for lags: {lags}...")
    features_to_lag = [
        'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'ATR', 'Bollinger_Perc_B', 'Price_vs_EMA9', 'EMA9_vs_EMA20'
    ]
    for feature in features_to_lag:
        for lag in lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    logging.info("Lagged features added.")
    return df

def add_rolling_features(df, windows=[5, 10, 20]):
    """
    Adds rolling statistics (mean, std) for key features.
    """
    logging.info(f"Adding rolling features for windows: {windows}...")
    features_to_roll = [
        'close', 'volume', 'RSI', 'MACD_Diff', 'ATR'
    ]
    for feature in features_to_roll:
        for window in windows:
            df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window).std()
    logging.info("Rolling features added.")
    return df

# ... (imports and other functions like add_technical_indicators above) ...

def generate_labels(df, config):
    """
    Generates labels for the ML model based on future price movements.
    A label of 1 indicates a potential profitable trade, 0 otherwise.

    :param df: DataFrame with OHLCV data and technical indicators.
    :param config: The 'model_training' section of the strategy_config.yaml.
                   Expected to contain 'labeling' dictionary with:
                   'look_forward_candles', 'profit_target_pct', 'stop_loss_pct'.
    :return: DataFrame with 'label' column added.
    """
    logging.info("Generating labels...")

    # Access labeling parameters correctly from the 'labeling' sub-dictionary
    # 'config' here is the 'model_training' dictionary passed from model_trainer.py
    look_forward = config.get('labeling', {}).get('look_forward_candles', 100)
    profit_target = config.get('labeling', {}).get('profit_target_pct', 1.0)
    stop_loss = config.get('labeling', {}).get('stop_loss_pct', 5.0)

    # Ensure look_forward is not zero or negative
    if look_forward <= 0:
        logging.error("look_forward_candles must be a positive integer for label generation.")
        df['label'] = np.nan # Assign NaN labels if invalid config
        return df

    # Calculate future high and low within the look_forward window
    # Shift these up so that each row (t) has the future (t+1 to t+look_forward) high/low
    df['future_high'] = df['high'].rolling(window=look_forward, closed='right').max().shift(-look_forward)
    df['future_low'] = df['low'].rolling(window=look_forward, closed='right').min().shift(-look_forward)
    df['future_close'] = df['close'].shift(-look_forward) # Future close for direct prediction comparison if needed

    # Initialize label column
    df['label'] = 0 # Default to 0 (no profitable trade)

    # Condition for a positive label (e.g., price moves up by profit_target before hitting stop_loss)
    # This is a simplified "did it hit profit target within look_forward window?"
    # A more robust labeling would involve barrier methods (triple barrier) to account for time, profit, and stop loss.
    
    # Check if profit target is hit
    # Use the 'profit_target' variable directly
    profit_hit = (df['future_high'] >= df['close'] * (1 + profit_target / 100))
    
    # Check if stop loss is hit
    # Use the 'stop_loss' variable directly
    stop_loss_hit = (df['future_low'] <= df['close'] * (1 - stop_loss / 100))

    # Label as 1 if profit_hit occurs AND it doesn't hit stop loss first
    df.loc[profit_hit & ~stop_loss_hit, 'label'] = 1

    # Drop the temporary future columns
    df = df.drop(columns=['future_high', 'future_low', 'future_close'])

    logging.info(f"Labels generated. Positive labels: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    
    # Drop rows with NaN labels that result from the look_forward window (at the end of the dataframe)
    df = df.dropna(subset=['label'])
    
    # Convert label to integer type after dropping NaNs
    df['label'] = df['label'].astype(int)

    return df

# ... (rest of your feature_engineer.py code) ...

def preprocess_features(df):
    """
    Handles missing values (due to indicator calculation/lagging) and scales/normalizes features.
    """
    logging.info("Preprocessing features...")

    # Drop rows with NaN values created by indicators or lagging.
    # This is crucial as ML models can't handle NaNs directly.
    initial_rows = df.shape[0]
    df = df.dropna()
    logging.info(f"Dropped {initial_rows - df.shape[0]} rows due to NaN values.")

    # Select features for scaling (exclude 'label' and 'open', 'high', 'low', 'close', 'volume' if not used as features directly)
    # The 'label' column is the target, not a feature.
    # OHLCV might be used directly as features, or only through indicators.
    # Let's consider all numerical columns except 'label' as features for now.
    feature_columns = [col for col in df.columns if col not in ['label', 'open', 'high', 'low', 'close', 'volume']] # Assuming OHLCV are not raw features

    if not feature_columns:
        logging.warning("No features selected for scaling after NaN drop. Check feature engineering steps.")
        return df, None # Return df and no scaler if no features

    X = df[feature_columns]

    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame
    df_scaled_features = pd.DataFrame(X_scaled, index=df.index, columns=feature_columns)

    # Merge scaled features back with the original DataFrame (keeping 'label' and OHLCV if needed)
    df = pd.concat([df_scaled_features, df[['label', 'open', 'high', 'low', 'close', 'volume']]], axis=1)

    logging.info("Features preprocessed (NaNs handled, scaled).")
    return df, scaler

def get_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # This block runs when feature_engineer.py is executed directly for testing
    # It will load data, add features, and generate labels based on a config.
    logging.info("Running feature_engineer.py in standalone mode for testing.")

    # Load a dummy config for testing purposes.
    # In main.py, you would load the actual strategy_config.yaml.
    # For testing, you can specify a test config or adapt it.
    config_path = 'config/strategy_config.yaml' # Assuming a default config exists
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Please create one or adjust path.")
        exit(1) # Exit if config not found

    # Adjust data path as needed for testing
    data_file_path = config.get('data_fetching', {}).get('csv_file', 'data/PEPE_1m.csv') # Use a default for testing
    
    # Check if 'data_fetching' and 'model_training' keys exist in config
    if 'data_fetching' not in config or 'model_training' not in config:
        logging.error("Missing 'data_fetching' or 'model_training' section in config. Please ensure your config matches the expected structure.")
        exit(1)

    df = load_data(data_file_path)

    if df is not None:
        df = add_technical_indicators(df)
        df = add_lagged_features(df)
        df = add_rolling_features(df)

        # Generate labels using the model_training config from the loaded config
        df = generate_labels(df, config)

        # Preprocess features (handle NaNs and scale)
        df_processed, scaler = preprocess_features(df.copy()) # Use a copy to avoid modifying original df for subsequent steps

        logging.info(f"Final processed data shape for ML: {df_processed.shape}")
        logging.info(f"Number of positive labels (1s): {df_processed['label'].sum()}")
        logging.info(f"Number of negative labels (0s): {(df_processed.shape[0] - df_processed['label'].sum())}")
        logging.info("Feature engineering and labeling complete. Ready for model training.")

        # You can save this processed DataFrame if needed for later inspection
        # df_processed.to_csv('data/processed_features_and_labels.csv')