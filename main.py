import argparse
import yaml
import uuid
import pandas as pd
from utils.data_loader import load_ohlcv
from engine.indicators import add_indicators
from engine.strategy import apply_strategy
from utils.logger import Logger # Import the Logger class

def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest from config")
    parser.add_argument("--config", required=True, help="Path to strategy config YAML")
    return parser.parse_args()

def load_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML config file: {exc}")
        exit(1)

def main():
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    run_id = str(uuid.uuid4())[:8]
    strategy_name = config_path.split('/')[-1].replace('_config.yaml', '')

    print(f"\nStarting Backtest Run ID: {run_id} for strategy: {strategy_name}")

    # Initialize logger
    logger = Logger()

    try:
        df = load_ohlcv(config['data']['path'])
    except ValueError as e:
        print(f"Error loading data: {e}")
        exit(1)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config['data']['path']}")
        exit(1)

    df = add_indicators(df)
    
    # Pass the logger instance to apply_strategy
    trades = apply_strategy(df, config, run_id, strategy_name, logger)

    if trades.empty:
        print("⚠️ No trades found. Adjust strategy parameters or data range.")
    else:
        total = len(trades)
        wins = len(trades[trades['outcome'] == 'win'])
        losses = len(trades[trades['outcome'] == 'loss'])

        win_pct = round((wins / total) * 100, 2) if total > 0 else 0
        avg_win = trades[trades['outcome'] == 'win']['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades[trades['outcome'] == 'loss']['pnl_pct'].mean() if losses > 0 else 0
        
        # Ensure avg_loss is a positive value for expectancy calculation
        expectancy = (wins / total * avg_win) - (losses / total * abs(avg_loss)) if total > 0 else 0

        print(f"\n--- Backtest Results ({strategy_name}) ---")
        print(f"Total Trades: {total}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win %: {win_pct}%")
        print(f"Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")
        print(f"Expectancy: {expectancy:.2f}%")
        print(f"Results saved to {logger.RESULTS_FILE}")
        print(f"FINAL_EXPECTANCY_PERCENT: {expectancy:.4f}")


if __name__ == "__main__":
    main()