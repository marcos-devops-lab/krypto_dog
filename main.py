# main.py

import argparse
import yaml
import uuid
from data.data_loader import load_data
from indicators.indicators import add_indicators
from engine.strategy import apply_strategy

def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest from config")
    parser.add_argument("--config", required=True, help="Path to strategy config YAML")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    run_id = str(uuid.uuid4())[:8]
    print(f"\n🚀 Starting Backtest Run ID: {run_id}")

    df = load_data(config['data']['path'])
    df = add_indicators(df)
    trades = apply_strategy(df, config, run_id)

    if trades.empty:
        print("⚠️ No trades found. Adjust strategy parameters.")
    else:
        total = len(trades)
        wins = len(trades[trades['outcome'] == 'win'])
        losses = len(trades[trades['outcome'] == 'loss'])

        win_pct = round((wins / total) * 100, 2)
        avg_win = trades[trades['outcome'] == 'win']['pnl_pct'].mean()
        avg_loss = trades[trades['outcome'] == 'loss']['pnl_pct'].mean()
        expectancy = (wins / total * avg_win) - (losses / total * abs(avg_loss))

        print(f"\n📊 Total Trades: {total}")
        print(f"✅ Wins: {wins} | ❌ Losses: {losses}")
        print(f"🎯 Win %: {win_pct}%")
        print(f"📈 Avg Win: {avg_win:.2f}% | 📉 Avg Loss: {avg_loss:.2f}%")
        print(f"💡 Expectancy: {expectancy:.2f}%")

if __name__ == "__main__":
    main()
