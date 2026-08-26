from __future__ import annotations

import json
from pathlib import Path

from sleep_mlops.train import train


SUMMARY_PATH = Path("reports/drift_summary.json")


def retrain_if_drift() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Run monitoring first so reports/drift_summary.json exists.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if not summary.get("retraining_recommended", False):
        print("No retraining required: drift is below the configured threshold.")
        return

    print("Drift threshold reached. Starting a new training run.")
    train()


if __name__ == "__main__":
    retrain_if_drift()
