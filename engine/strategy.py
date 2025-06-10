import pandas as pd
# No direct import of log_trade; logger instance will be passed

def apply_strategy(df: pd.DataFrame, config: dict, run_id: str, strategy_name: str, logger) -> pd.DataFrame:
    """
    Applies rule-based strategy to dataframe using config parameters.
    Logs trades to results CSV using the provided logger instance.
    """
    df = df.copy()
    trades = []

    in_position = False
    entry_price = 0.0
    entry_time = None
    holding_candles = 0

    # Extract strategy parameters from config
    # Entry Conditions
    rsi_threshold = config['entry_conditions']['rsi']['threshold']
    rsi_condition = config['entry_conditions']['rsi']['condition']
    
    macd_crossover_type = config['entry_conditions']['macd']['crossover']
    
    pullback_pct = config['entry_conditions']['pullback']['percentage']
    price_move_pct = config['entry_conditions']['price_move']['percentage'] # This was in config but not used in old logic

    # Exit Conditions
    holding_period = config['exit_conditions']['holding_period']
    stop_loss_pct = config['exit_conditions']['stop_loss']
    take_profit_pct = config['exit_conditions']['take_profit']

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Ensure indicators have been calculated and are not NaN for current/previous row
        if pd.isna(row['rsi']) or pd.isna(prev_row['macd']) or pd.isna(row['macd_signal']):
            continue

        # ENTRY CONDITIONS
        # RSI Check
        rsi_ok = False
        if rsi_condition == "below":
            rsi_ok = row['rsi'] < rsi_threshold
        elif rsi_condition == "above":
            rsi_ok = row['rsi'] > rsi_threshold
        
        # MACD Crossover Check (assuming bullish for now, as per original logic)
        macd_ok = False
        if macd_crossover_type == "bullish":
            macd_ok = prev_row['macd'] < prev_row['macd_signal'] and row['macd'] > row['macd_signal']
        elif macd_crossover_type == "bearish":
             macd_ok = prev_row['macd'] > prev_row['macd_signal'] and row['macd'] < row['macd_signal']
             # If you plan to trade short, adjust pnl calculation and entry/exit logic accordingly.
             # For now, sticking to long entries based on original logic.
             continue # Skip if bearish crossover and we're only doing long trades


        # Pullback Check (percentage below current candle's high)
        pullback_ok = ((row['high'] - row['close']) / row['high'] * 100) >= pullback_pct

        if not in_position and rsi_ok and macd_ok and pullback_ok:
            # We don't use price_move_pct here for entry. It was "Required move after pullback to count as trade"
            # which usually implies an exit condition or confirmation AFTER entry.
            # For Phase 1, I'll assume entry is purely on these conditions.
            in_position = True
            entry_price = row['close']
            entry_time = row.name
            holding_candles = 0 # Reset holding period counter
            continue # Skip to next candle after entry

        # EXIT CONDITIONS
        if in_position:
            holding_candles += 1
            current_pnl_pct = (row['close'] - entry_price) / entry_price * 100

            outcome = None
            
            # Take Profit
            if current_pnl_pct >= take_profit_pct:
                outcome = 'win'
            # Stop Loss
            elif current_pnl_pct <= -stop_loss_pct:
                outcome = 'loss'
            # Holding Period Exceeded
            elif holding_candles >= holding_period:
                # If holding period is reached, consider it a loss if PnL is negative, else a win.
                # Or, you might treat it as a 'neutral' exit if PnL is small, for simplicity, I'll say loss if PnL <= 0.
                if current_pnl_pct > 0:
                    outcome = 'win'
                else:
                    outcome = 'loss'
                logger.log_message(f"Run ID {run_id}: Trade exited due to holding period limit.")
            
            if outcome is not None:
                in_position = False
                trade = {
                    'run_id': run_id,
                    'strategy_name': strategy_name,
                    'entry_time': entry_time,
                    'exit_time': row.name,
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'pnl_pct': round(current_pnl_pct, 2),
                    'outcome': outcome
                }
                trades.append(trade)
                logger.log_trade(trade) # Log using the passed logger instance

    # Handle case where a trade might still be open at the end of the data
    if in_position:
        # Close the last trade at the last available price
        last_row = df.iloc[-1]
        current_pnl_pct = (last_row['close'] - entry_price) / entry_price * 100
        outcome = 'win' if current_pnl_pct > 0 else 'loss'
        trade = {
            'run_id': run_id,
            'strategy_name': strategy_name,
            'entry_time': entry_time,
            'exit_time': last_row.name,
            'entry_price': entry_price,
            'exit_price': last_row['close'],
            'pnl_pct': round(current_pnl_pct, 2),
            'outcome': outcome
        }
        trades.append(trade)
        logger.log_trade(trade)
        logger.log_message(f"Run ID {run_id}: Last trade closed at end of data.")


    return pd.DataFrame(trades)