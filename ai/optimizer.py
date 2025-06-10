import optuna
import subprocess
import yaml
import os
import numpy as np

# Define the path for the temporary configuration file
TEMP_CONFIG_PATH = "config/optuna_temp_strategy.yaml"

def objective(trial):
    # Suggest parameters for Optuna to optimize
    rsi_threshold = trial.suggest_int('rsi_threshold', 30, 80)
    pullback_pct = trial.suggest_float('pullback_pct', 0.1, 5.0, step=0.1) # Added step for finer control
    holding_period = trial.suggest_int('holding_period', 5, 200)
    stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.5, 10.0, step=0.1)
    take_profit_pct = trial.suggest_float('take_profit_pct', 0.5, 10.0, step=0.1)

    # --- IMPORTANT: Ensure this temporary config matches your main.py's expected YAML structure ---
    temp_config = {
        'strategy': {
            'rsi_threshold': rsi_threshold,
            'pullback_pct': pullback_pct,
            'holding_period': holding_period,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        },
        'data_fetching': {
            'symbol': 'PEPE/USDT', # Placeholder, change if needed
            'timeframe': '1m',     # Placeholder
            'since': '2023-01-01 00:00:00', # Placeholder for your long data range
            'limit': None
        },
        'model_training': {
            'look_forward': 100, # This can also be optimized later
            'profit_target': take_profit_pct, # Link to trial parameter
            'stop_loss': stop_loss_pct, # Link to trial parameter
            'test_size': 0.25 # Assuming 25% for test split
        }
    }

    with open(TEMP_CONFIG_PATH, 'w') as f:
        yaml.dump(temp_config, f)

    try:
        result = subprocess.run(
            ["python", "main.py", "--config", TEMP_CONFIG_PATH],
            capture_output=True,
            text=True,
            check=True # Raise an exception for non-zero exit codes
        )

        output_lines = result.stdout.splitlines()

        # Extract all relevant metrics
        total_trades = 0
        wins = 0
        losses = 0
        win_pct = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0

        for line in output_lines:
            if "Total Trades:" in line:
                total_trades = int(line.split(':')[1].strip())
            elif "Wins:" in line and "Losses:" in line:
                parts = line.split('|')
                wins = int(parts[0].split(':')[1].strip())
                losses = int(parts[1].split(':')[1].strip())
            elif "Win %:" in line:
                win_pct = float(line.split(':')[1].strip().replace('%', ''))
            elif "Avg Win:" in line and "Avg Loss:" in line:
                parts = line.split('|')
                avg_win = float(parts[0].split(':')[1].strip().replace('%', ''))
                avg_loss = float(parts[1].split(':')[1].strip().replace('%', ''))
            elif "FINAL_EXPECTANCY_PERCENT:" in line:
                expectancy = float(line.split(':')[1].strip())

        # Define the objective value
        # This is where we make it smarter.
        # We want to maximize expectancy, but also penalize very low trade counts.
        # And ensure it's not -inf if there were trades but just bad performance.

        if total_trades == 0:
            # Penalize heavily if no trades are found. Adjust this penalty as needed.
            # A negative large number will ensure Optuna avoids these configs.
            objective_value = -100.0 # Return a large negative value for no trades
            print(f"Trial {trial.number} - No trades found. Value: {objective_value}")
        else:
            # Use expectancy as the primary metric.
            # You can make this more complex: e.g., expectancy * (total_trades / max_trades_found_so_far)
            # Or add a risk-adjusted metric like Sharpe Ratio if you calculate it in main.py
            objective_value = expectancy
            print(f"Trial {trial.number} - Expectancy: {expectancy:.4f}%. Total Trades: {total_trades}. Value: {objective_value}")

        return objective_value

    except subprocess.CalledProcessError as e:
        print(f"Error running main.py for trial {trial.number}: {e.cmd} returned non-zero exit status {e.returncode}.")
        print(f"Stdout:\n{e.stdout}") # Print stdout to see what main.py logged before crashing
        print(f"Stderr:\n{e.stderr}") # Print stderr to see the error traceback
        return -1000.0 # Return a very low value for crashes

    except Exception as e:
        print(f"An unexpected error occurred for trial {trial.number}: {e}")
        return -1000.0 # Return a very low value for other unexpected errors

# Study creation and optimization loop (rest of the file)
if __name__ == "__main__":
    # You might want to consider 'sampler' and 'pruner' for more advanced optimization
    # E.g., optuna.samplers.TPESampler(), optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize") # Maximize the objective_value
    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=100) # Run 100 trials

    print("\nOptimization finished.")
    print("Best trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Parameters: {study.best_params}")

    # Optionally save the best parameters to a specific config file
    best_config_path = "config/best_strategy_config.yaml"
    best_params_to_save = {
        'strategy': study.best_params,
        # Add back other static parts of your config that are not being optimized by Optuna
        'data_fetching': {
            'symbol': 'PEPE/USDT',
            'timeframe': '1m',
            'since': '2023-01-01 00:00:00',
            'limit': None
        },
        'model_training': {
            'look_forward': 100,
            'profit_target': study.best_params['take_profit_pct'],
            'stop_loss': study.best_params['stop_loss_pct'],
            'test_size': 0.25
        }
    }
    with open(best_config_path, 'w') as f:
        yaml.dump(best_params_to_save, f, sort_keys=False)
    print(f"Best parameters saved to {best_config_path}")

    # Clean up the temporary config file
    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)