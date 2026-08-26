# Sleep Pattern Analysis System: End-to-End MLOps Project

This project implements a reproducible MLOps workflow for the supplied `sleep_mobile_stress_dataset_15000.csv` file. The system predicts the continuous `sleep_quality_score` outcome on a 1–10 scale from demographic, mobile-use, lifestyle, stress, and fatigue features. The target is deliberately excluded from the input features to avoid target leakage, and `user_id` is treated only as an identifier.

The project is designed for development in **PyCharm** and uses DVC for dataset versioning, scikit-learn for modeling, MLflow for experiment tracking and model registration, FastAPI for prediction serving, Docker for packaging, Evidently AI for data-drift reporting, Prometheus instrumentation for API metrics, and GitHub Actions for continuous integration. DVC versions data by storing lightweight metadata in Git while keeping the data artifact in its cache or remote storage [1]. MLflow records run parameters, metrics, artifacts, and models and provides a tracking UI [2]. FastAPI supplies typed request validation and interactive API documentation [3]. Evidently supports tabular data quality and drift evaluation [4].

## 1. Dataset assessment and modeling decision

The supplied dataset contains **15,000 rows and 13 columns**, with no missing values in the inspected file. It contains two categorical columns, `gender` and `occupation`, and numeric variables covering screen time, phone use before sleep, sleep duration, stress, caffeine, activity, notifications, and mental fatigue. The default model predicts `sleep_quality_score` with a Random Forest regression pipeline. The project can later be extended with a second classifier, such as a poor-sleep label derived from a documented business threshold, but the first version keeps the target continuous and avoids manufacturing a label that does not exist in the source file.

The training pipeline uses an 80/20 split with `random_state=42`. It fits preprocessing only on the training data and then evaluates on the held-out test data, following the leakage-avoidance principle recommended in the scikit-learn documentation [5]. On the supplied data, the validated run produced an RMSE of approximately **0.765**, an MAE of approximately **0.605**, and an R² of approximately **0.799**. These metrics are a baseline for this dataset, not a clinical claim or a guarantee of real-world performance.

## 2. Project structure

```text
sleep-pattern-mlops/
├── configs/config.yaml                         # model, data, MLflow, monitoring settings
├── data/raw/sleep_mobile_stress_dataset_15000.csv  # DVC-managed source file
├── data/raw/sleep_mobile_stress_dataset_15000.csv.dvc
├── data/processed/                             # generated train/test splits
├── models/                                     # generated Joblib model and metadata
├── monitoring/                                 # Evidently HTML report
├── reports/                                    # JSON drift summary
├── src/sleep_mlops/
│   ├── api.py                                  # FastAPI service
│   ├── monitor.py                              # Evidently + drift threshold report
│   ├── prepare.py                              # validation and reproducible split
│   ├── retrain_if_drift.py                     # conditional retraining entry point
│   └── train.py                                # scikit-learn + MLflow training
├── tests/test_api.py                            # API tests
├── dvc.yaml                                   # reproducible prepare/train stages
├── Dockerfile                                 # API container image
├── requirements.txt                            # pinned-compatible dependency ranges
└── .github/workflows/ci.yml                    # test and Docker build workflow
```

## 3. Initialize it in PyCharm

Open PyCharm and select **Open** to open the `sleep-pattern-mlops` directory. In PyCharm, open **Settings → Project → Python Interpreter**, choose **Add Interpreter → Existing**, and select `.venv/bin/python` on Linux/macOS or `.venv\Scripts\python.exe` on Windows. The commands below can be executed in the PyCharm Terminal at the project root.

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux or macOS, use:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are creating the project from an empty folder instead of using this prepared package, create the folders first and copy the supplied CSV into `data/raw/`:

```bash
mkdir -p sleep-pattern-mlops/data/raw
cd sleep-pattern-mlops
git init -b main
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/sleep_mobile_stress_dataset_15000.csv data/raw/
dvc init --no-scm
dvc add data/raw/sleep_mobile_stress_dataset_15000.csv
git add .
git commit -m "Initialize sleep pattern MLOps project"
```

For PowerShell, replace the final `cp` command with:

```powershell
Copy-Item C:\path\to\sleep_mobile_stress_dataset_15000.csv data\raw\
```

The prepared package already contains the code, configuration, DVC pointer, tests, and supplied dataset. If you recreate the project manually, copy the files from this package before running the pipeline.

## 4. Run the complete local pipeline

Run the stages in this order from the project root. The first command validates the input and creates `data/processed/train.csv` and `data/processed/test.csv`. The second command trains the model, logs the run to MLflow, registers the model, and writes `models/sleep_quality_model.joblib` for FastAPI.

```bash
# Activate the environment first.
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate

PYTHONPATH=src python -m sleep_mlops.prepare
PYTHONPATH=src python -m sleep_mlops.train
```

On Windows PowerShell, use this equivalent form if `PYTHONPATH=src command` is not accepted by the shell:

```powershell
$env:PYTHONPATH = "src"
python -m sleep_mlops.prepare
python -m sleep_mlops.train
```

The DVC equivalent is:

```bash
dvc repro
```

DVC compares the dependencies and outputs declared in `dvc.yaml` and reruns only stages whose inputs changed. After changing the CSV, configuration, or pipeline code, run `dvc repro`, inspect the metrics, and commit the resulting `.dvc` metadata and code changes. For a shared project, configure a remote such as S3, Google Drive, Azure Blob Storage, or a network location and then use `dvc push`; the local DVC documentation describes this remote-storage workflow [1].

## 5. Open MLflow and inspect the experiment

The training script uses SQLite as the local MLflow backend because the current MLflow release requires a database-backed store for the latest tracking and registry features. Start the tracking UI in a second PyCharm Terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The experiment is named `sleep-pattern-analysis`, and the registered model is named `sleep_quality_model`. Review the logged data hash, parameters, MAE, RMSE, R², and model artifact before considering a model eligible for deployment. MLflow's tracking UI is intended for comparing runs and inspecting logged artifacts [2].

## 6. Start and test the FastAPI service

Start the API from the project root after training:

```bash
PYTHONPATH=src uvicorn sleep_mlops.api:app --reload --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
uvicorn sleep_mlops.api:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the automatically generated Swagger interface. The service exposes `/health`, `/metrics`, and `/predict`. A sample request is:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Female",
    "occupation": "Teacher",
    "daily_screen_time_hours": 4.5,
    "phone_usage_before_sleep_minutes": 45,
    "sleep_duration_hours": 7.2,
    "stress_level": 4.0,
    "caffeine_intake_cups": 2,
    "physical_activity_minutes": 45,
    "notifications_received_per_day": 120,
    "mental_fatigue_score": 4.0
  }'
```

The prediction endpoint returns a score clipped to the documented 1–10 range and a simple interpretation. This is a research prototype: it should not be used as medical diagnosis or treatment advice.

## 7. Run monitoring and determine whether to retrain

Run monitoring with:

```bash
PYTHONPATH=src python -m sleep_mlops.monitor
```

The script compares the configured reference data and current data, writes `monitoring/evidently_report.html`, and writes `reports/drift_summary.json`. The current default compares the prepared training split with the prepared test split and ignores `user_id` and the target column. In production, replace `monitoring.current_path` with a file or data extract containing recent prediction inputs. The threshold is configured as `monitoring.drift_share_threshold: 0.30`; change it only after documenting why that threshold is appropriate for your evaluation protocol.

The deterministic summary uses a Kolmogorov–Smirnov test for numeric columns and a chi-square test for categorical columns, while Evidently creates the visual HTML report. The report contains the feature-level evidence needed for an operator to decide whether a new data batch is sufficiently different from the reference data. The current supplied split produced a drift share below the default threshold, so retraining was not recommended for that batch.

## 8. Conditional retraining and model promotion

Run the conditional retraining entry point after monitoring:

```bash
PYTHONPATH=src python -m sleep_mlops.retrain_if_drift
```

The script retrains only when `reports/drift_summary.json` indicates that the drift share has reached the configured threshold. The training process logs a new MLflow run and registers a new model version. Before production promotion, compare the new model's RMSE with the existing model, record an approval decision, and deploy only a validated version. A real deployment should also add a model-performance reference set containing delayed ground-truth outcomes; input drift alone is not the same as predictive-performance degradation.

## 9. Run tests and build Docker

Run the automated tests with:

```bash
PYTHONPATH=src pytest -q
```

Build and run the API container after the model artifact exists:

```bash
docker build -t sleep-pattern-api:1.0 .
docker run --rm -p 8000:8000 sleep-pattern-api:1.0
```

The Docker image copies the trained model into the image and starts Uvicorn on port 8000. When a new approved model is trained, rebuild the image with a new tag rather than overwriting the previous tag. This makes rollback easier.

## 10. GitHub Actions CI/CD

The included workflow runs on pushes and pull requests to `main` or `master`. It installs Python 3.11 dependencies, runs the API tests, and builds the Docker image. Push the repository to GitHub as follows:

```bash
git remote add origin https://github.com/<your-account>/sleep-pattern-mlops.git
git push -u origin main
```

GitHub Actions will then execute `.github/workflows/ci.yml`. For a production-style workflow, add a protected deployment environment, store registry credentials as GitHub Secrets, publish the image to a container registry, and require manual approval before deploying a newly registered model. Do not commit cloud credentials or tokens to the repository.

## 11. PyCharm run configurations

Create four optional PyCharm Run/Debug configurations. Use module `sleep_mlops.prepare` with `PYTHONPATH=src`; use module `sleep_mlops.train` with the same environment variable; use module `sleep_mlops.monitor`; and use a Python server configuration for `uvicorn sleep_mlops.api:app --reload`. Set the working directory to the project root. The simplest first run is to execute the commands in Sections 4–7 in the PyCharm Terminal, then convert the commands into Run/Debug configurations after the workflow succeeds.

## 12. Suggested research deliverables

For an academic project report, document the research question, the definition of the predicted outcome, data-quality checks, train/test protocol, feature-preprocessing design, MLflow screenshots, DVC history, API request/response examples, Docker build output, Evidently drift report, CI workflow result, and retraining decision. Report limitations clearly: the CSV is a static observational dataset, the variables are not a clinical sleep study, and the model estimates a predefined score rather than diagnosing a sleep disorder.

## References

[1]: https://dvc.org/doc/start "DVC: Get Started"

[2]: https://mlflow.org/docs/latest/ml/tracking/ "MLflow Tracking"

[3]: https://fastapi.tiangolo.com/ "FastAPI Documentation"

[4]: https://docs.evidentlyai.com/ "Evidently AI Documentation"

[5]: https://scikit-learn.org/stable/common_pitfalls.html "Scikit-learn: Common Pitfalls and Recommended Practices"

[6]: https://docs.github.com/en/actions/writing-workflows/quickstart "GitHub Actions: Quickstart"

[7]: https://docs.docker.com/get-started/ "Docker Get Started"
