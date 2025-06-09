# engine/strategy.py

import pandas as pd
from utils.logger import log_trade

def apply_strategy(df: pd.DataFrame, config: dict, run_id: str) -> pd.DataFrame:
    """
    Applies rule-based strategy to dataframe using config parameters.
    Logs trades to results CSV.
    """
    df = df.copy()
    trades = []

    in_position = False
    entry_price = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Extract config
        rsi_limit = config['strategy']['rsi']
        macd_cross = config['strategy']['macd_cross']
        pullback_pct = config['strategy']['pullback_pct']
        target_pct = config['strategy']['target_pct']
        stop_pct = config['strategy']['stop_pct']

        # ENTRY CONDITIONS
        rsi_ok = row['rsi'] < rsi_limit
        macd_ok = prev_row['macd'] < prev_row['macd_signal'] and row['macd'] > row['macd_signal']
        pullback_ok = ((row['high'] - row['close']) / row['high']) >= (pullback_pct / 100)

        if not in_position and rsi_ok and macd_ok and pullback_ok:
            in_position = True
            entry_price = row['close']
            entry_time = row.name
            continue

        # EXIT CONDITIONS
        if in_position:
            gain = (row['close'] - entry_price) / entry_price

            if gain >= target_pct / 100:
                outcome = 'win'
                in_position = False
            elif gain <= -stop_pct / 100:
                outcome = 'loss'
                in_position = False
            else:
                continue  # still in trade

            trade = {
                'run_id': run_id,
                'entry_time': entry_time,
                'exit_time': row.name,
                'entry_price': entry_price,
                'exit_price': row['close'],
                'pnl_pct': round(gain * 100, 2),
                'outcome': outcome
            }
            trades.append(trade)
            log_trade(trade)

    return pd.DataFrame(trades)
