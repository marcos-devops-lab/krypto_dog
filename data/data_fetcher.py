# data/data_fetcher.py

import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone # Import timezone
import time
import os

def fetch_ohlcv(exchange_id, symbol, timeframe, since_date, limit=1000, output_filename=None):
    """
    Fetches OHLCV data from an exchange, handling pagination, and saves it to a CSV.
    Fetches all available data from since_date up to the current time.
    """
    try:
        exchange = getattr(ccxt, exchange_id)()
        exchange.load_markets()
    except AttributeError:
        print(f"Error: Exchange '{exchange_id}' not found in ccxt.")
        return

    # Convert since_date to milliseconds timestamp (UTC for consistency)
    # Ensure since_date is parsed as UTC to avoid timezone issues
    since_dt_utc = datetime.strptime(since_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    since_ms = exchange.parse8601(since_dt_utc.isoformat())

    all_ohlcv = []
    current_fetch_timestamp = since_ms

    print(f"Fetching {symbol} {timeframe} data from {since_date} onwards...")
    print(f"Starting fetch from {datetime.fromtimestamp(current_fetch_timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Define a target end time, e.g., now minus a small buffer (e.g., 5 minutes)
    # to avoid fetching incomplete current candles
    end_ms = exchange.milliseconds() - exchange.parse_timeframe(timeframe) * 1000 * 5 

    while current_fetch_timestamp < end_ms:
        try:
            # Fetch data with the maximum limit per request
            ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, current_fetch_timestamp, limit)
            
            if not ohlcv_chunk:
                print("No more data to fetch for this period.")
                break # No more data means we've reached the end

            all_ohlcv.extend(ohlcv_chunk)
            
            # Move the timestamp forward for the next fetch
            # The next fetch should start from the timestamp AFTER the last fetched candle
            current_fetch_timestamp = ohlcv_chunk[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            
            # For progress indication
            last_fetched_candle_dt = datetime.fromtimestamp(ohlcv_chunk[-1][0] / 1000, tz=timezone.utc)
            print(f"Fetched up to: {last_fetched_candle_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Implement rate limiting to avoid getting banned or errors
            # Binance's default rate limit is often around 1200 requests/minute (for weight 1 endpoints)
            # A sleep based on exchange.rateLimit is generally safe, but a bit slow.
            # For high volume fetching, you might need to adjust or use async methods.
            time.sleep(exchange.rateLimit / 1000 + 0.1) # Add a small buffer

        except ccxt.RateLimitExceeded as e:
            print(f"Rate limit exceeded: {e}. Sleeping for 10 seconds...")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Check symbol, timeframe, or API limits. Breaking.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Breaking.")
            break
    
    if not all_ohlcv:
        print("No data fetched. Please check parameters or try a different date range.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Ensure UTC conversion
    
    # Sort by timestamp and drop duplicates (in case of overlap due to exchange API behavior)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    if output_filename:
        output_path = os.path.join("data", output_filename) # Use os.path.join for cross-platform paths
        df.to_csv(output_path, index=False)
        print(f"\nSuccessfully fetched {len(df)} unique candles and saved to {output_path}")
    else:
        print(f"\nSuccessfully fetched {len(df)} unique candles (not saved).")
    
    return df

if __name__ == "__main__":
    # --- Configuration for 1000PEPE ---
    exchange_id = 'binance'
    symbol = 'PEPE/USDT' # Make sure this is the correct symbol on Binance
    timeframe = '1m'     # 1-minute candles
    
    # **IMPORTANT**: Set this to the approximate listing date of PEPE on Binance.
    # A quick check shows PEPE was listed around May 2023. Let's use an early date.
    since_date = '2023-05-01' # YYYY-MM-DD - This will attempt to fetch ALL 1m data from this date.
    
    output_file = 'PEPE_1m.csv' # This will overwrite the existing file with more data
    
    # Fetch data
    fetch_ohlcv(exchange_id, symbol, timeframe, since_date, output_filename=output_file)

    print("\nData fetching process complete. The generated CSV should be significantly larger.")