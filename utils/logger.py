import os
import csv
from datetime import datetime

class Logger:
    # Use a class variable for the results file path for easy access
    RESULTS_FILE = "logs/results.csv"

    def __init__(self):
        os.makedirs(os.path.dirname(self.RESULTS_FILE), exist_ok=True)
        self.headers = [
            "timestamp",
            "run_id",         # Added run_id
            "strategy_name",  # Added strategy_name
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "pnl_pct",        # Changed from pnl to pnl_pct
            "outcome",        # Changed from win to outcome
            "comment"
        ]
        self._init_csv()

    def _init_csv(self):
        # Check if file exists AND is not empty before writing header
        if not os.path.exists(self.RESULTS_FILE) or os.stat(self.RESULTS_FILE).st_size == 0:
            with open(self.RESULTS_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_trade(self, trade_data: dict, comment=""):
        """
        Logs a single trade to the results CSV.
        Expected keys in trade_data: run_id, strategy_name, entry_time, entry_price, exit_time, exit_price, pnl_pct, outcome
        """
        with open(self.RESULTS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            row_to_write = {
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": trade_data.get("run_id"),
                "strategy_name": trade_data.get("strategy_name"),
                "entry_time": trade_data.get("entry_time"),
                "entry_price": trade_data.get("entry_price"),
                "exit_time": trade_data.get("exit_time"),
                "exit_price": trade_data.get("exit_price"),
                "pnl_pct": trade_data.get("pnl_pct"),
                "outcome": trade_data.get("outcome"),
                "comment": comment
            }
            writer.writerow(row_to_write)

    def log_message(self, msg):
        """Logs a general message to console with timestamp."""
        print(f"[{datetime.utcnow().isoformat()}] {msg}")