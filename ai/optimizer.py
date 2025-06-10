# ai/optimizer.py

import optuna
import subprocess
import yaml
import re
import os
import shutil # For copying original config
import numpy as np

# Ensure the logs directory exists for main.py to write to
os.makedirs("logs", exist_ok=True)

# Path to your original config file and a temporary one for Optuna to modify
ORIGINAL_CONFIG_PATH = "config/strategy_config.yaml"
TEMP_CONFIG_PATH = "config/optuna_temp_strategy.yaml"

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(path, config):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

def objective(trial: optuna.Trial):
    """
    Objective function for Optuna to optimize.
    Runs main.py with suggested parameters and returns the expectancy.
    """
    # 1. Load the original config to preserve structure
    original_config = load_config(ORIGINAL_CONFIG_PATH)
    
    # 2. Suggest parameters for the current trial
    # Entry Conditions
    rsi_threshold = trial.suggest_int('rsi_threshold', 30, 80)
    # macd_crossover_type can be kept as 'bullish' for now, or made a categorical choice
    # macd_fast = trial.suggest_int('macd_fast', 8, 16) # Example of optimizing indicator periods
    # macd_slow = trial.suggest_int('macd_slow', 20, 30)
    # macd_signal = trial.suggest_int('macd_signal', 7, 12)
    pullback_pct = trial.suggest_float('pullback_pct', 0.1, 2.0, step=0.1)

    # Exit Conditions
    holding_period = trial.suggest_int('holding_period', 5, 50)
    stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.5, 3.0, step=0.1)
    take_profit_pct = trial.suggest_float('take_profit_pct', 0.5, 5.0, step=0.1)

    # Update the config dictionary with trial's parameters
    original_config['entry_conditions']['rsi']['threshold'] = rsi_threshold
    # original_config['entry_conditions']['macd']['fast_period'] = macd_fast # If uncommented
    # original_config['entry_conditions']['macd']['slow_period'] = macd_slow
    # original_config['entry_conditions']['macd']['signal_period'] = macd_signal
    original_config['entry_conditions']['pullback']['percentage'] = pullback_pct
    
    original_config['exit_conditions']['holding_period'] = holding_period
    original_config['exit_conditions']['stop_loss'] = stop_loss_pct
    original_config['exit_conditions']['take_profit'] = take_profit_pct

    # 3. Save the modified config to a temporary file
    save_config(TEMP_CONFIG_PATH, original_config)

    # 4. Run main.py with the temporary config
    try:
        # Capture stdout to parse results
        result = subprocess.run(
            ["python", "main.py", "--config", TEMP_CONFIG_PATH],
            capture_output=True,
            text=True,
            check=True # Raise an exception for non-zero exit codes
        )
        output = result.stdout
        
        # 5. Parse the output for Expectancy (or any other metric you want to optimize)
        expectancy_match = re.search(r"💡 Expectancy: ([-+]?\d*\.\d+|\d+)%", output)
        if expectancy_match:
            expectancy = float(expectancy_match.group(1))
            return expectancy
        else:
            print(f"Warning: Expectancy not found in output for trial {trial.number}. Output:\n{output}")
            return -np.inf # Return a very low value if expectancy isn't found (e.g., no trades)

    except subprocess.CalledProcessError as e:
        print(f"Error running main.py for trial {trial.number}: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return -np.inf # Return a very low value if main.py crashes
    except Exception as e:
        print(f"An unexpected error occurred during trial {trial.number}: {e}")
        return -np.inf

if __name__ == "__main__":
    # Ensure a temporary config file exists to be modified
    if not os.path.exists(ORIGINAL_CONFIG_PATH):
        print(f"Error: Original config file not found at {ORIGINAL_CONFIG_PATH}. Please ensure it exists.")
        exit(1)
    shutil.copyfile(ORIGINAL_CONFIG_PATH, TEMP_CONFIG_PATH)

    # Create an Optuna study
    study = optuna.create_study(direction="maximize") # Maximize expectancy
    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=100) # Run 100 trials

    print("\nOptimization finished.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optionally, save the best parameters to a new config file
    best_config = load_config(ORIGINAL_CONFIG_PATH)
    for key, value in trial.params.items():
        # Update nested parameters based on how they were suggested
        if 'rsi_threshold' in key:
            best_config['entry_conditions']['rsi']['threshold'] = value
        elif 'pullback_pct' in key:
            best_config['entry_conditions']['pullback']['percentage'] = value
        elif 'holding_period' in key:
            best_config['exit_conditions']['holding_period'] = value
        elif 'stop_loss_pct' in key:
            best_config['exit_conditions']['stop_loss'] = value
        elif 'take_profit_pct' in key:
            best_config['exit_conditions']['take_profit'] = value
        # Add more 'elif' for other suggested parameters if you expand them
    
    best_config_path = "config/best_strategy_config.yaml"
    save_config(best_config_path, best_config)
    print(f"\nBest parameters saved to {best_config_path}")

    # Clean up temporary config file
    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)