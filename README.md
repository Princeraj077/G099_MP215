# G099_MP215 — Warehouse Picking Time Prediction

## 1. Project Overview

**Problem ID:** MP215  
**Group ID:** G099  
**Course:** SCSE3040 — MLOps Mini Project  
**Client:** Kochi Fulfilment Centre

The objective is to predict the time required to pick an order at a fulfilment centre. The target variable is `pick_seconds`, making this a regression problem. Models are evaluated using Mean Absolute Error (MAE), where a lower value indicates better predictions.

The project involves building a complete MLOps pipeline covering data validation, preprocessing, model training, experiment tracking, model registration, API development, containerization, CI, deployment configuration and monitoring.

The project-specific requirements are documented in `PROBLEM.md`, while `STUDENT_HANDBOOK.md` contains the submission requirements, marking criteria and expected repository structure.

## 2. Dataset Information

The project contains three CSV files:

| File | Records | Purpose |
|---|---:|---|
| `train.csv` | ~1000 | January–June 2026 training data |
| `later.csv` | 300 | July–August 2026 data for drift monitoring and retraining |
| `holdout_features.csv` | 250 | September 2026 data for final predictions |

**Target:** `pick_seconds`

**Input features:**
- `items_in_order`
- `aisle_distance_m`
- `picker_exp_months`
- `shelf_height_m`
- `shift_hour`
- `order_weight_kg`
- `zone_type`
- `shift`

`record_id` is used for record identification and final prediction output. `date` is used for temporal analysis and splitting the later dataset.

## 3. Current Project Progress

### Repository and Data Setup

- GitHub repository created and initial project files uploaded.
- Original datasets extracted and placed in `data/raw/`.
- DVC initialized and configured locally.
- No DVC remote storage has been configured.
- CSV files are currently being shared directly through GitHub.

### Data Lead — Completed

All seven required data-quality checks have been performed on the original training data, with `later.csv` and `holdout_features.csv` used for cross-checking.

| Data issue | Finding |
|---|---|
| Leaking column | Not present |
| Unit mix-up | Present in `picker_exp_months`, Evening shift |
| Sentinel values | Not present |
| Duplicate rows | Not present |
| Stuck sensor | Present in `shelf_height_m` |
| Target typos | 9 incorrect values |
| Spelling variants | Present in `zone_type` |

Corrections implemented:

- Converted Evening shift picker experience from days to months using a factor of 30.
- Corrected 86 stuck sensor readings in `shelf_height_m`, recorded at 1.62 between 20 March and 4 April 2026.
- Corrected 9 target values that were entered 10 times too large.
- Standardized `zone_type` into three categories: `ambient`, `bulky` and `chilled`.

The original raw training data is preserved. Cleaning is performed through Python code and produces a separate processed dataset.

The data section of `reports/findings.yaml` has been completed.

## 4. Repository Structure

The project follows the structure specified in the Student Handbook.

```text
G099_MP215/
│
├── README.md
├── PROBLEM.md
├── STUDENT_HANDBOOK.md
├── requirements.txt
├── params.yaml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── later.csv
│   │   └── holdout_features.csv
│   │
│   └── processed/
│       └── train_clean.csv
│
├── src/
│   └── cleaning/
│       ├── __init__.py
│       └── data_cleaning.py
│
├── tests/
│
├── app/
│   └── main.py
│
├── models/
│   └── model.joblib
│
├── reports/
│   ├── findings.yaml
│   └── data_quality_report.json
│
├── predictions/
│   └── holdout_predictions.csv
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .dvc/
└── mlflow.db
```

Some files and folders are expected to be created during the remaining stages.

## 5. Agreed Technical Implementation

### Data Processing

- Use Python files inside `src/cleaning/` for reusable preprocessing and data-quality checks.
- Keep original datasets unchanged.
- Store cleaned datasets separately under `data/processed/`.
- Use `params.yaml` for project configuration and read it through `yaml.safe_load`.
- Preserve the original training data hash for DVC and MLflow verification.

### Model Training and MLflow

The Model Lead will implement the training pipeline using scikit-learn regression models.

Expected implementation:

1. Train at least three model configurations using the cleaned training data.
2. Track parameters, metrics and artifacts using MLflow.
3. Use SQLite tracking with `sqlite:///mlflow.db`.
4. Log the original training data SHA-256 as `data_sha256` in every run.
5. Log MAE for model evaluation.
6. Register the selected model in the MLflow Model Registry.
7. Maintain model versions and assign the `champion` alias to the selected model.
8. Save the final champion model and its preprocessing pipeline as `models/model.joblib`.

The model should be saved as a complete preprocessing-plus-regression pipeline wherever possible, so the API uses the same transformations as training.

### FastAPI Service

The Service Lead will implement the prediction service in `app/main.py`.

Required endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Return predicted picking time |

The request must contain exactly eight fields:

`items_in_order`, `aisle_distance_m`, `picker_exp_months`, `shelf_height_m`, `shift_hour`, `order_weight_kg`, `zone_type`, `shift`.

Required validation rules:

- `items_in_order` must not exceed 65.
- `shelf_height_m` must not be negative.

These restrictions must be implemented using Pydantic `Field` validation.

The service will load the final champion artifact from `models/model.joblib`.

### Testing and Containerization

- Use pytest for automated tests.
- Include at least three tests, with at least one using FastAPI `TestClient`.
- Tests must not depend on the original CSV files.
- Use Docker with `python:3.12-slim`.
- Install pinned dependencies before copying application code.
- Expose port `8000` and run Uvicorn on `0.0.0.0:8000`.

### CI and Kubernetes

GitHub Actions will run pytest on every push.

Kubernetes configuration must follow these exact requirements:

| Setting | Required value |
|---|---|
| Deployment | `mp215-api` |
| Service | `mp215-svc` |
| Docker image | `mp215-api:1.0` |
| Replicas | 3 |
| Container port | 8000 |
| Service port | 80 |

Both readiness and liveness probes must use `/health`, with liveness configured to wait longer.

### Monitoring and Retraining

After v1 is trained and `later.csv` is available for evaluation:

1. Calculate PSI for every numeric feature using 10 bins.
2. Identify the most drifted feature.
3. Calculate v1 MAE separately for Morning, Evening and Night shifts.
4. Sort `later.csv` by date.
5. Train v2 using cleaned training data and the first 70% of `later.csv`.
6. Compare v1 and v2 on the last 30%.
7. Promote v2 only if its MAE is at least 12% lower than v1's MAE on the same evaluation data.
8. Update the MLflow `champion` alias if v2 is promoted.

Finally, use the champion model to predict all 250 records in `holdout_features.csv`.

Save the output as:

`predictions/holdout_predictions.csv`

It must contain exactly two columns:

```text
record_id,prediction
```

## 6. Next Steps

### Model Lead

1. Pull the latest repository changes.
2. Read the data-cleaning implementation and use `data/processed/train_clean.csv`.
3. Configure `params.yaml` and the training modules inside `src/<suitable_name>/`.
4. Train at least three model configurations and compare MAE.
5. Configure MLflow tracking and register the selected v1 model.
6. Save the complete champion pipeline as `models/model.joblib`.
7. Coordinate the model's expected input format with the Service Lead.

### Service Lead

1. Pull the latest repository changes.
2. Create `app/main.py` and implement the FastAPI endpoints.
3. Implement the Pydantic request schema and both validation rules.
4. Write pytest tests using small sample inputs.
5. Initially test the API independently using a temporary prediction function.
6. Integrate the actual `models/model.joblib` once it is available.
7. Create and test the Dockerfile and `.dockerignore`.

### After Both Parts Are Ready

- Integrate the model and API.
- Ensure the API uses the same preprocessing as training.
- Coordinate with the Ops Lead for CI, JSON logging and Kubernetes configuration.
- Complete drift monitoring, retraining and final holdout predictions.
- Update the remaining fields in `reports/findings.yaml`.

## 7. Working with the Repository

After the latest changes are committed and pushed, every member should synchronize their local repository:

```bash
git pull origin main
```

The current CSV files are shared directly through GitHub. If the repository is later switched to DVC-based remote storage, use `dvc pull` to retrieve tracked datasets.

Before committing new changes:

```bash
git status
git add <files>
git commit -m "Describe the changes"
git push origin main
```

Every member should commit their own contributions using their own GitHub account.

---

**Note:** The README documents the team's current implementation and agreed project structure. `PROBLEM.md` and `STUDENT_HANDBOOK.md` remain the authoritative references for exact client requirements and submission rules.