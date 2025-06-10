import ccxt
import pandas as pd
import time
import os
import yaml
import logging

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config(config_path='config/strategy_config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. Please ensure it exists.")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return None

def fetch_ohlcv(exchange_id, symbol, timeframe, since, limit=None):
    """
    Fetches OHLCV data from a specified exchange.
    :param exchange_id: Exchange ID (e.g., 'binance')
    :param symbol: Trading pair (e.g., 'ADA/USDT')
    :param timeframe: Candlestick timeframe (e.g., '1m', '5m', '1h')
    :param since: Timestamp in milliseconds from which to fetch data
    :param limit: Number of candles to fetch per request (max 1000 for Binance)
    :return: Pandas DataFrame with OHLCV data
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,  # Ensures adherence to exchange rate limits
        'options': {
            'defaultType': 'future', # Important for Binance Futures
        },
    })

    all_ohlcv = []
    current_since = since
    
    logging.info(f"Starting data fetch for {symbol} on {exchange_id} ({timeframe}) from {pd.to_datetime(since, unit='ms')}...")

    while True:
        try:
            # Fetch 'limit' candles from 'current_since'
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
            
            if not ohlcv:
                logging.info("No more data found. Stopping fetch.")
                break

            all_ohlcv.extend(ohlcv)
            
            # Update current_since to the timestamp of the last fetched candle + 1 minute (or timeframe)
            # This handles potential overlaps or gaps correctly.
            last_timestamp = ohlcv[-1][0]
            current_since = last_timestamp + exchange.parse_timeframe(timeframe) * 1000 # Convert timeframe to milliseconds

            logging.info(f"Fetched {len(ohlcv)} candles up to {pd.to_datetime(last_timestamp, unit='ms')}. Total fetched: {len(all_ohlcv)}")

            # Binance typically limits to 1000 candles per request for 1m timeframe.
            # If we don't get 'limit' candles, it means we've reached the end of available data.
            if len(ohlcv) < limit:
                logging.info(f"Less than {limit} candles fetched, indicating end of available data or current time.")
                break
            
            # Simple rate limiting for very fast loops
            time.sleep(exchange.rateLimit / 1000)

        except ccxt.RateLimitExceeded:
            logging.warning("Rate limit exceeded. Waiting longer before retrying.")
            time.sleep(exchange.rateLimit / 1000 + 5) # Wait rateLimit + 5 seconds
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error: {e}")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred during fetching: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df.sort_index() # Ensure chronological order

    logging.info(f"Finished fetching data. Total candles: {df.shape[0]}")
    return df

def main():
    config = get_config()
    if config is None:
        return

    data_config = config.get('data_fetching', {})
    
    symbol = data_config.get('symbol', 'ADA/USDT') # Default to ADA/USDT
    timeframe = data_config.get('timeframe', '1m') # Default to 1m
    since_str = data_config.get('since', '2017-09-01 00:00:00') # Default to ADA's earliest data on Binance
    limit_per_request = data_config.get('limit_per_request', 1000) # Max per request for Binance
    output_dir = 'data'
    output_filename = data_config.get('csv_file', f"{symbol.replace('/', '_')}_{timeframe}.csv") # Default filename
    output_path = os.path.join(output_dir, output_filename)

    # Convert 'since' string to milliseconds timestamp
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Fetching data for {symbol} ({timeframe}) from {since_str}...")
    df_ohlcv = fetch_ohlcv('binance', symbol, timeframe, since_ms, limit_per_request)

    if df_ohlcv is not None and not df_ohlcv.empty:
        df_ohlcv.to_csv(output_path, index=True)
        logging.info(f"Successfully saved {df_ohlcv.shape[0]} candles to {output_path}")
    else:
        logging.warning("No data fetched or DataFrame is empty. CSV file not created.")

if __name__ == "__main__":
    main()