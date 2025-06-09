# utils/logger.py

import os
import csv
from datetime import datetime

RESULTS_FILE = "logs/results.csv"

class Logger:
    def __init__(self):
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        self.headers = [
            "timestamp",
            "strategy_name",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "pnl",
            "win",
            "comment"
        ]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_trade(self, strategy_name, entry_time, entry_price, exit_time, exit_price, comment=""):
        pnl = exit_price - entry_price
        win = 1 if pnl > 0 else 0
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "strategy_name": strategy_name,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl": pnl,
                "win": win,
                "comment": comment
            })

    def log_message(self, msg):
        print(f"[{datetime.utcnow().isoformat()}] {msg}")
