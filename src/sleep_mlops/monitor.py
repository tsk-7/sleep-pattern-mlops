from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import chi2_contingency, ks_2samp


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict:
    results = {}
    drifted = 0
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[column]):
            statistic, p_value = ks_2samp(reference[column].dropna(), current[column].dropna())
            is_drifted = bool(p_value < 0.05)
            results[column] = {
                "test": "kolmogorov_smirnov",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": is_drifted,
            }
        else:
            ref_counts = reference[column].value_counts()
            cur_counts = current[column].value_counts()
            categories = sorted(set(ref_counts.index).union(cur_counts.index))
            table = [
                [int(ref_counts.get(category, 0)), int(cur_counts.get(category, 0))]
                for category in categories
            ]
            _, p_value, _, _ = chi2_contingency(table)
            is_drifted = bool(p_value < 0.05)
            results[column] = {
                "test": "chi_square",
                "p_value": float(p_value),
                "drifted": is_drifted,
            }
        drifted += int(is_drifted)

    share = drifted / max(len(results), 1)
    return {"columns": results, "drifted_columns": drifted, "checked_columns": len(results), "drift_share": share}


def run_monitoring(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)
    monitor_cfg = config["monitoring"]
    reference = pd.read_csv(monitor_cfg["reference_path"])
    current = pd.read_csv(monitor_cfg["current_path"])
    ignored = set(config["data"].get("id_columns", [])) | {config["data"]["target"]}
    columns = [column for column in reference.columns if column not in ignored]

    summary = calculate_drift(reference, current, columns)
    summary["threshold"] = float(monitor_cfg["drift_share_threshold"])
    summary["retraining_recommended"] = summary["drift_share"] >= summary["threshold"]

    report_path = Path(monitor_cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(current_data=current[columns], reference_data=reference[columns]).save_html(str(report_path))
        summary["evidently_report"] = str(report_path)
    except Exception as exc:  # Keep the deterministic summary available if Evidently changes API.
        summary["evidently_report_error"] = str(exc)

    output_path = Path("reports/drift_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_monitoring()
