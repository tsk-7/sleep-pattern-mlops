from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(config: dict, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    model_cfg = config["model"]
    estimator = RandomForestRegressor(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=int(model_cfg["max_depth"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        n_jobs=int(model_cfg["n_jobs"]),
        random_state=int(config["project"]["random_state"]),
    )
    return Pipeline([("preprocessor", preprocessing), ("model", estimator)])


def train(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)
    data_cfg = config["data"]
    train_df = pd.read_csv(data_cfg["train_path"])
    test_df = pd.read_csv(data_cfg["test_path"])
    target = data_cfg["target"]
    drop_columns = set(data_cfg.get("id_columns", [])) | {target}

    X_train = train_df.drop(columns=list(drop_columns))
    y_train = train_df[target]
    X_test = test_df.drop(columns=list(drop_columns))
    y_test = test_df[target]

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    pipeline = build_pipeline(config, numeric_features, categorical_features)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }
        params = {
            "model_type": config["model"]["type"],
            "target": target,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "numeric_features": ",".join(numeric_features),
            "categorical_features": ",".join(categorical_features),
            "random_state": config["project"]["random_state"],
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("data_sha256", sha256_file(data_cfg["raw_path"]))
        mlflow.set_tag("project", config["project"]["name"])
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=config["mlflow"]["registered_model_name"],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        Path("models").mkdir(exist_ok=True)
        model_path = Path("models/sleep_quality_model.joblib")
        metadata_path = Path("models/model_metadata.json")
        joblib.dump(pipeline, model_path)
        metadata = {
            "run_id": run.info.run_id,
            "model_path": str(model_path),
            "target": target,
            "features": X_train.columns.tolist(),
            "metrics": metrics,
            "data_sha256": sha256_file(data_cfg["raw_path"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()
