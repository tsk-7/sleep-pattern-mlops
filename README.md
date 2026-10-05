# Sleep Pattern Analysis System: End-to-End MLOps Project

This project implements a reproducible MLOps workflow for the supplied `sleep_mobile_stress_dataset_15000.csv` file. The updated raw dataset contains 15,000 rows and 19 columns: the original 13 columns plus six timing and calendar fields. The system predicts the continuous `sleep_quality_score` outcome on a 1–10 scale from demographic, mobile-use, lifestyle, stress, and fatigue features. The target is deliberately excluded from the input features to avoid target leakage, and `user_id` is treated only as an identifier. The pipeline includes the original nine engineered features and the timing/circadian features.

The project can be opened in **VS Code** or PyCharm and uses DVC for dataset versioning, scikit-learn for modeling, MLflow for experiment tracking and model registration, FastAPI for prediction serving, Docker for packaging, Evidently AI for data-drift reporting, Prometheus instrumentation for API metrics, and GitHub Actions for continuous integration. DVC versions data by storing lightweight metadata in Git while keeping the data artifact in its cache or remote storage [1]. MLflow records run parameters, metrics, artifacts, and models and provides a tracking UI [2]. FastAPI supplies typed request validation and interactive API documentation [3]. Evidently supports tabular data quality and drift evaluation [4].

## 1. Dataset assessment and modeling decision

The original dataset contains 15,000 rows and 13 columns; the updated raw file has 19 columns after adding the six existing timing and calendar fields. It contains two categorical columns, `gender` and `occupation`, and numeric variables covering screen time, phone use before sleep, sleep duration, stress, caffeine, activity, notifications, and mental fatigue. The default model predicts `sleep_quality_score` with a Random Forest regression pipeline. The target remains continuous and unchanged.

The training pipeline uses an 80/20 split with `random_state=42`. It fits preprocessing only on the training data and then evaluates on the held-out test data, following the leakage-avoidance principle recommended in the scikit-learn documentation [5]. On the supplied data, the validated run produced an RMSE of approximately **0.765**, an MAE of approximately **0.605**, and an R² of approximately **0.799**. These metrics are a baseline for this dataset, not a clinical claim or a guarantee of real-world performance.

## 2. Project structure

```text
sleep-pattern-mlops/
├── configs/config.yaml                         # model, data, MLflow, monitoring settings
├── datasets/sleep_mobile_stress_dataset_15000.csv  # updated raw data
├── datasets/sleep_mobile_stress_dataset_15000.csv.dvc  # created by dvc add
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

### Configure the MySQL sleep-record database

Install and start MySQL Server 8.0, then create the application database and a dedicated local account in MySQL Workbench or the MySQL CLI:

```sql
CREATE DATABASE IF NOT EXISTS sleep_pattern_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'sleep_app'@'127.0.0.1'
  IDENTIFIED BY 'replace-with-a-strong-local-password';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX
  ON sleep_pattern_db.* TO 'sleep_app'@'127.0.0.1';
```

Set `SLEEP_DATABASE_URL` in the same shell used to start FastAPI. URL-encode any reserved characters in the password. Keep this connection string out of Git:

```powershell
$env:SLEEP_DATABASE_URL = "mysql+pymysql://sleep_app:<url-encoded-password>@127.0.0.1:3306/sleep_pattern_db?charset=utf8mb4"
$env:PYTHONPATH = "src"
```

For a public deployment, set `SLEEP_DATABASE_URL` to a managed MySQL-compatible database or Supabase PostgreSQL database reachable from the deployment platform. Do not use `127.0.0.1` or `localhost` in a cloud deployment; those addresses refer to the API container itself. For Supabase, use the session pooler connection details and convert the URL to SQLAlchemy format:

```text
postgresql+psycopg2://postgres.<project-ref>:<password>@<pooler-host>:5432/postgres?sslmode=require
```

The API creates the `sleep_records` table in the configured database on its first sleep-record or analysis request.

The Render blueprint also declares `FRONTEND_URLS` as a secret environment setting. Set it to a comma-separated list of approved browser origins, for example `https://your-frontend.example.com`. Local development defaults to `http://127.0.0.1:3000,http://localhost:3000`.

The service creates the indexed `sleep_records` table on the first record/analysis request. Sleep records are unique per `user_id` and `sleep_date`; the calculator currently uses the single-user ID `local-user` because this application does not include authentication.

Sleep-record endpoints:

- `POST /api/sleep-records` — insert one sleep day and return server-calculated metrics.
- `GET /api/sleep-records/{record_id}` and `GET /api/sleep-records/user/{user_id}` — retrieve saved records.
- `PUT /api/sleep-records/{record_id}` and `DELETE /api/sleep-records/{record_id}` — update or remove a record.
- `GET /api/sleep-analysis/daily/{user_id}/{sleep_date}` — daily record details.
- `GET /api/sleep-analysis/weekly/{user_id}`, `/monthly/{user_id}`, `/pattern/{user_id}`, and `/weekend-comparison/{user_id}` — persisted timing analyses.

Time fields use 24-hour `HH:MM`. The sleep date is the date the user woke up. Overnight transitions are ordered by the backend; invalid dates/times, impossible or unreasonable intervals, and duplicate user/date records return HTTP 400.

Start the API from the project root after training:

```bash
PYTHONPATH=src uvicorn sleep_mlops.api:app --reload --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
uvicorn sleep_mlops.api:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the automatically generated Swagger interface. The service exposes `/health`, `/metrics`, the sleep-record endpoints above, and the separate ML `/predict` endpoint. `/health` reports whether the MySQL URL is configured and the database is reachable. A sample prediction request is:

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

## 13. Feature Engineering Details

We implemented the following engineered features to capture complex interactions in the data:

| Feature | Logic | Rationale |
|---|---|---|
| **Digital Intensity** | `(Screen Time * 60 + Phone Usage) / 60` | Captures total daily digital exposure in hours. |
| **Stress-Fatigue Index** | `Stress Level * Mental Fatigue` | Captures the synergistic effect of stress and exhaustion. |
| **Screen-to-Sleep Ratio** | `Screen Time / Sleep Duration` | Measures digital usage relative to recovery time. |
| **Activity-Screen Balance** | `Activity / Screen Time` | Measures the balance between physical movement and sedentary digital use. |

These features improved the model's R² score to **0.8003** and identified **Pre-Sleep Disruption Ratio** and **Digital Intensity** as top-5 predictors of sleep quality.

| Feature | Logic | Rationale |
|---|---|---|
| **Age-Stress Impact** | `Age * Stress Level` | Captures how stress sensitivity might change with age. |
| **Notification Frequency** | `Notifications / (Screen Time + 0.1)` | Measures how "interrupted" a user's screen time is. |
| **Caffeine-Activity Ratio** | `Caffeine / (Activity + 1)` | Relates stimulant intake to physical expenditure. |
| **Pre-Sleep Disruption Ratio** | `Phone Usage / (Sleep Duration + 0.1)` | Impact of late-night phone use relative to total sleep. |
| **Lifestyle Balance Score** | `(Activity / 60) - Caffeine` | A simple balance of healthy activity vs stimulants. |

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


## 14. Updated sleep-timing and circadian dataset

The project now includes a reproducible timing-aware data layer. The original supplied file contained 15,000 rows and 13 columns, with one record per user. The updated raw file contains the same 15,000 records and all original columns, plus `date`, `bedtime`, `wake_time`, `last_caffeine_time`, `day_of_week`, and `is_weekend`. The engineered train and test files contain 41 columns: the original data, nine existing engineered features, and the new timing, circadian, variability, behavior, and schedule features.

The raw dataset is generated by `scripts/generate_updated_dataset.py`. The preparation pipeline in `src/sleep_mlops/prepare.py` then adds all model features. This preserves the existing target, `sleep_quality_score`; the target is never used to create a predictor.

| Feature | Type | Formula or definition | Purpose |
|---|---|---|---|
| `date` | Raw | Stable date derived from `user_id` | Provides a reproducible calendar context. |
| `bedtime`, `wake_time` | Raw | Bedtime is linked to existing behavior; wake time is bedtime plus observed sleep duration modulo 24 hours. | Represents a coherent sleep interval. |
| `last_caffeine_time` | Raw | Deterministic time before bedtime, linked to caffeine quantity. | Enables clearance-window analysis. |
| `day_of_week`, `is_weekend` | Raw | `day_of_week = date.dayofweek`; weekend is 1 for Saturday/Sunday. | Represents weekly schedule context. |
| `*_circular_sin`, `*_circular_cos` | Numeric | `sin/cos(2π × minutes / 1440)` | Treats 23:55 and 00:05 as nearby times. |
| `mid_sleep_time`, `mid_sleep_minutes` | Time/numeric | `(bedtime + ((wake − bedtime) mod 1440)/2) mod 1440` | Represents the sleep midpoint across midnight. |
| `bedtime_variability`, `wake_time_variability` | Numeric | Circular rolling standard deviation over the previous seven same-user records. | Measures schedule regularity without comparing unrelated users. |
| `social_jetlag_score` | Numeric | Circular absolute difference between same-user mean weekday and weekend mid-sleep. | Measures weekday/weekend schedule displacement. |
| `caffeine_clearance_window` | Numeric | `((bedtime − last_caffeine) mod 1440) / 60` | Measures hours between caffeine and bedtime. |
| `work_schedule_mismatch` | Numeric | Circular distance between mid-sleep and an occupation schedule midpoint. | Estimates timing conflict with a transparent occupation heuristic. |
| `late_night_digital_share` | Numeric | `min(phone_usage_before_sleep_minutes, 180) / (daily_screen_time_hours × 60 + 0.00001)`, clipped to [0, 1]. | Estimates the fraction of screen time in the final three hours. |
| `circadian_disruption_index` | Numeric | `pre_sleep_disruption_ratio × bedtime_variability` | Combines late phone use with schedule irregularity. |

The occupation heuristic assigns Doctors and Freelancers a flexible/shift expected midpoint of 04:00, Students 05:00, and all other occupations a daytime midpoint of 03:30. This is a reproducible research assumption rather than a clinical classification. Because event-level screen timestamps are not available, late-night digital share uses the documented pre-sleep-phone-use proxy.

The current source has one observation per user. Consequently, the valid seven-day within-user variability is 0 and social jetlag is 0 when a user has no observations from both a weekday and a weekend. The implementation supports repeated `user_id + date` records in future sleep-tracker exports and will then calculate circular rolling variability and weekday/weekend social jetlag. A future longitudinal dataset should use a user-level or time-based split instead of a random row split.

Validation is generated by `scripts/validate_sleep_data.py` and saved to `reports/updated_dataset_validation.json`. The current run contains zero missing cells and duplicate rows, no impossible sleep durations, consistent weekend flags, valid circular features, non-negative caffeine-clearance windows, valid social-jetlag values, valid late-night share values, no target leakage, and no train/test overlap. Bedtime/wake-time intervals match the observed sleep duration within one minute from clock rounding.

The updated data package is available in `datasets/`, and the reproducible archive is `sleep-pattern-datasets.zip`.

## 15. Regenerating the updated data

From the project root, run:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/generate_updated_dataset.py
PYTHONPATH=src python -m sleep_mlops.prepare
python scripts/validate_sleep_data.py
```

The first command creates the timing-aware raw layer. The second creates `data/processed/train.csv` and `data/processed/test.csv`. The third creates the validation report and prints the data-quality checks.


## 16. Updated backend deployment

The backend now deploys the timing-aware dataset package directly from `datasets/`. The active source configured in `configs/config.yaml` is:

```text
datasets/sleep_mobile_stress_dataset_15000.csv
```

The preparation stage writes the reproducible model inputs to:

```text
data/processed/train.csv
data/processed/test.csv
```

These files contain 12,000 and 3,000 rows respectively, with 41 columns including the original fields, timing fields, nine existing engineered features, and the new circadian and behavioral features. The trained artifact is stored at `models/sleep_quality_model.joblib`.

Run the backend locally after preparing and training:

```bash
PYTHONPATH=src python -m sleep_mlops.prepare
PYTHONPATH=src python -m sleep_mlops.train
PYTHONPATH=src uvicorn sleep_mlops.api:app --host 0.0.0.0 --port 8000
```

The API endpoints are:

- `GET /health` — confirms that the model, updated dataset, and processed splits are available.
- `GET /metadata` — reports the deployed dataset, target, model path, and row counts.
- `POST /predict` — accepts the existing user-profile input and applies the same timing-aware feature pipeline before prediction.
- `GET /docs` — opens the interactive Swagger documentation.

Test the deployment with:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/smoke_test_api.py
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
```

A Docker image can be built and run after the model artifact exists:

```bash
docker build -t sleep-pattern-api:2.0 .
docker run --rm -p 8000:8000 sleep-pattern-api:2.0
```

The Dockerfile copies `datasets/`, the processed data, configuration, source code, and trained model into the image. The API derives the new timing features for prediction requests when a caller provides only the original profile fields, ensuring inference remains compatible with the deployed 41-column training schema.

## 17. VS Code and DagsHub setup

Open the extracted project root in VS Code. Select Python 3.11 at `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux/macOS, then use the integrated terminal from the project root.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The updated raw data is `datasets/sleep_mobile_stress_dataset_15000.csv` (15,000 rows, 19 columns). The processed training and testing datasets each have 41 columns and contain 12,000 and 3,000 rows, respectively. The target is `sleep_quality_score`; it remains unchanged and is not used to construct features.

Regenerate and validate the data from the existing repository source. These commands retain the original columns and do not create a separate dataset:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_updated_dataset.py
python -m sleep_mlops.prepare
python scripts/validate_sleep_data.py
pytest -q
```

On Linux/macOS, prefix each Python command with `PYTHONPATH=src`. Inspect `reports/updated_dataset_validation.json` for row counts, target completeness, and train/test overlap.

### DagsHub Git and DVC

Create the DagsHub repository `sleep-pattern-mlops` under your account before adding remotes. Run `git init` only when `.git` does not exist, and run `dvc init` only when `.dvc` does not exist. Then track the existing updated raw data:

```bash
git init
dvc init
dvc add datasets/sleep_mobile_stress_dataset_15000.csv
git add .
git commit -m "Integrate updated sleep dataset and MLOps backend"
git branch -M main
git remote add origin https://dagshub.com/<your-dagshub-username>/sleep-pattern-mlops.git
git push -u origin main
```

Configure DagsHub DVC storage after replacing the username and token placeholders:

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<your-dagshub-username>/sleep-pattern-mlops.s3
dvc remote modify origin --local access_key_id <your-dagshub-token>
dvc remote modify origin --local secret_access_key <your-dagshub-token>
dvc push -r origin
```

The `--local` credentials belong only in DVC's ignored local configuration. Never commit DagsHub tokens, passwords, or credential-bearing environment files.

### MLflow training

Local tracking defaults to `sqlite:///mlflow.db` from `configs/config.yaml`:

```powershell
$env:PYTHONPATH = "src"
Remove-Item Env:MLFLOW_TRACKING_URI -ErrorAction SilentlyContinue
python -m sleep_mlops.train
```

For remote DagsHub tracking, set credentials in the current shell and train. `MLFLOW_TRACKING_URI` takes precedence; when it is unset, training uses the URI in the YAML configuration.

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/<your-dagshub-username>/sleep-pattern-mlops.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<your-dagshub-username>"
$env:MLFLOW_TRACKING_PASSWORD = "<your-dagshub-token>"
python -m sleep_mlops.train
```

On Linux/macOS, export those same variables and run `PYTHONPATH=src python -m sleep_mlops.train`. MLflow logs model type, target, train/test row counts, numeric/categorical features, MAE, RMSE, R², the raw dataset SHA-256, and the trained model artifact. Training writes `models/sleep_quality_model.joblib` and `models/model_metadata.json`.

### API and Docker

Start the API after training:

```powershell
$env:PYTHONPATH = "src"
uvicorn sleep_mlops.api:app --host 127.0.0.1 --port 8000
```

Linux/macOS: `PYTHONPATH=src uvicorn sleep_mlops.api:app --host 127.0.0.1 --port 8000`. Verify `GET /health`, `GET /metadata`, and `GET /docs` at `http://127.0.0.1:8000`. Metadata reports 15,000 raw rows, 12,000 train rows, 3,000 test rows, target `sleep_quality_score`, and the 41-column engineered schema. Run `python scripts/smoke_test_api.py` to exercise the sample `POST /predict` payload.

Run tests and build/run Docker with:

```powershell
$env:PYTHONPATH = "src"
pytest -q
docker build -t sleep-pattern-api:2.0 .
docker run --rm -p 8000:8000 sleep-pattern-api:2.0
```

Never add DagsHub tokens to Git, tracked DVC configuration, source code, or this README. Git/DVC pushes and remote MLflow training require your repository URL and credentials; those remote actions are not complete until configured with your account.
