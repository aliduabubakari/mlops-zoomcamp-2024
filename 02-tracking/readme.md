# NYC Taxi Trip Duration Prediction

Predict the duration of New York City taxi trips using state-of-the-art ML models, feature engineering, experiment tracking (MLflow), and flexible CLI tools.

---

## Table of Contents

- [Project Features](#project-features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Install MLflow with `uv`](#install-mlflow-with-uv)
  - [Windows Tips](#windows-tips)
  - [Mac/Linux Tips](#maclinux-tips)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Loading Local vs Remote Data](#loading-local-vs-remote-data)
  - [Model Selection](#model-selection)
  - [Customizing Model Hyperparameters](#customizing-model-hyperparameters)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Feature Engineering Only](#feature-engineering-only)
- [Accessing All Artifacts & Results](#accessing-all-artifacts--results)
- [Working with the MLflow UI](#working-with-the-mlflow-ui)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Project Features

- **Flexible data loading**: Use local or remote taxi trip datasets (NYC TLC).
- **Extensive feature engineering**: Cleans, flags anomalies, and adds insightful time-, financial-, and behavior-based features.
- **Multiple model support**: Linear Regression, Random Forest, XGBoost, LightGBM.
- **MLflow tracking**: Logs metrics, parameters, artifacts, feature importances, predictions, and more.
- **Powerful CLI**: Choose model, data source, custom hyperparameters, and advanced options.
- **Hyperparameter Tuning**: Built-in support using `hyperopt`.
- **Reproducible experiments**: Every run is timestamped and archived with stats and splits.

---

## Requirements

- **Python 3.8+** (Recommended: 3.8–3.11)
- [See `requirements.txt`](./requirements.txt) for Python dependencies (incl. pandas, scikit-learn, mlflow, etc.)

---

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/nyc-taxi-duration-prediction.git
    cd nyc-taxi-duration-prediction
    ```

2. **Create and Activate a Python Virtual Environment (RECOMMENDED):**

    ```bash
    # Linux or MacOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Install MLflow with `uv`

`uv` is a faster replacement for pip. If you want to use it:

**1. Install uv:**

```bash
pip install uv
# OR: see https://github.com/astral-sh/uv#installation for binary installer
```

**2. Install MLflow and all requirements (including possible C extensions for Mac/Windows):**

```bash
uv pip install -r requirements.txt
```

> #### Note
> - On some systems, installing `xgboost` or `lightgbm` may fail. Check their [official docs](https://xgboost.readthedocs.io/en/stable/install.html) if you need trouble-shooting for C++ dependencies.

---

#### Windows Tips

- For `xgboost`/`lightgbm` on Windows, make sure you have a working C++ build toolchain.
- If you encounter errors installing `xgboost` or `lightgbm`, try:
    ```bash
    pip install xgboost
    pip install lightgbm
    # Or use pre-built wheels: https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost
    ```

#### Mac/Linux Tips

- You might need to run `brew install libomp` (macOS) for XGBoost/LightGBM.
- If installed via M1/M2 Apple Silicon, run under Rosetta or use pre-built ARM wheels.

---

## Project Structure

```
.
├── main.py                # Main CLI training entry-point
├── requirements.txt
├── README.md
├── mlruns/                # MLflow tracking directory (auto-created)
├── data/                  # Holds your local raw data (if used)
├── models/                # All models, splits, and experiment artifacts
└── src/
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model_training.py
    └── hyperparameter.py
```

---

## Usage

### Quick Start

Train a Random Forest on local data (must be in `data/green_tripdata.csv`):

```bash
python main.py --model rf
```

Train on two recent months from remote (NYC open data):

```bash
python main.py --model lr --months 01-2024 02-2024
```

### Loading Local vs Remote Data

#### Local Data

- By default, the program looks for `data/green_tripdata.csv`.
- If **no** `--months` argument is provided, local CSV is used.

```bash
python main.py --model lgbm
```

#### Remote NYC TLC Data

- Specify remote month(s) with `--months`. Format: `MM-YYYY`.
- E.g., for Jan & Feb 2024:

```bash
python main.py --model rf --months 01-2024 02-2024
```

**The program will fetch:**
- https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet
- https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet

### Model Selection

Support for 4 model types:

- Linear Regression (`lr`)
- Random Forest (`rf`)
- XGBoost (`xgb`)
- LightGBM (`lgbm`)

Specify via CLI:

```bash
python main.py --model xgb --months 12-2023
```

### Customizing Model Hyperparameters

You can pass a JSON string with custom parameters, e.g.:

```bash
python main.py --model rf --params '{"n_estimators": 200, "max_depth": 15}'
```
OR 

```bash
python main.py --model rf --params '{"n_estimators": 200, "max_depth": 15}' --months 01-2024 02-2024
```

Supported parameters depend on the model. See [src/config.py](src/config.py) and the default param set.

### Hyperparameter Optimization

You can **automatically search for the best hyperparameters** for any supported model by using the `--hyperopt` flag. This is especially useful for advanced models like Random Forest, XGBoost, or LightGBM, which have many tunable parameters.

#### 1. **Install Hyperopt** (if you haven't already)

```bash
pip install hyperopt
```

---

#### 2. **How Does Hyperopt Work Here?**

- When you add `--hyperopt` to your command, the program uses the [hyperopt](http://hyperopt.github.io/hyperopt/) library to find the best combination of model parameters **for you, automatically**.
- It tries different random sets of parameters (e.g., number of trees, learning rate, tree depth, etc.) and evaluates each model's performance.
- After finishing the search, it **selects the set with the best validation RMSE (Root Mean Square Error)** and saves this best model.

You don't need to know which parameters to try—**the system experiments and makes the final selection automatically**.

---

#### 3. **How to Use**

**Example:**  
Suppose you want to tune an XGBoost model using data from January and February 2024:

```bash
python main.py --model xgb --months 01-2024 02-2024 --hyperopt --max-evals 30
```

- `--model xgb`: Use the XGBoost regression model.
- `--months 01-2024 02-2024`: Use data for Jan & Feb 2024 (fetched directly from NYC TLC).
- `--hyperopt`: Enable automated hyperparameter search.
- `--max-evals 30`: Try 30 different parameter combinations (each is called a trial). You can increase this for a bigger search (default is 50).

**You can use this method with any model** (e.g. `rf`, `lgbm`, etc.).

---

#### 4. **Saving the Best Optimized Model**

You can save the best-performing model discovered by hyperopt using a custom filename:

```bash
python main.py --model rf --hyperopt --hyperopt-save-name rf_best_jan_feb
```

- This saves the optimized Random Forest model as `models/rf_best_jan_feb.joblib` in your project folder.

---

#### 5. **How to Interpret the Results**

- The program **automatically picks the best hyperparameters and trains the winning model**.
- It logs all details and performance results in MLflow and saves artifact files under the appropriate `models/` folder.
- All trial results (including parameter combinations and metrics) are saved: you can review them from the MLflow UI.

---

#### 6. **Checking All Runs and Best Parameters**

After tuning, you can view and compare all runs in the MLflow UI:

```bash
mlflow ui
```
Then go to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.  
You'll see:
- All trial runs (with their parameters and scores)
- The best run highlighted
- Downloadable model files and detailed metrics for each trial

---

#### 7. **In Short**

- **Model and parameter search is 100% automated** when you use `--hyperopt`.
- The best model according to validation RMSE is **selected for you**.
- You just need to add the flag—no manual parameter picking!
- **Use MLflow UI** to explore all results, view/compare parameters, and download models.

---

**Try it with any model!**  
For example:

```bash
python main.py --model lgbm --months 12-2023 --hyperopt --max-evals 40 --hyperopt-save-name lgbm_dec_best
```

### Feature Engineering

#### Feature Engineering Only

To run feature engineering and save the processed data (no training):

```bash
python main.py --model rf --months 01-2024 --features-only
```

#### Saving Feature Transformer

You can now save the feature engineering transformer for later use in prediction pipelines:

```bash
python main.py --model rf --save-transformer
```

This will save the feature transformer separately in the run directory.

#### Complete Prediction Pipelines

Save a complete scikit-learn pipeline that includes both feature transformation and model prediction:

```bash
python main.py --model xgb --save-name xgb_complete --save-transformer
```

The pipeline will be saved as `models/xgb_complete_pipeline.joblib` and can be used for end-to-end predictions:

```python
import joblib

# Load the complete pipeline
pipeline = joblib.load("models/xgb_complete_pipeline.joblib")

# Make predictions directly from raw data
predictions = pipeline.predict(new_raw_data)
```

#### Combined with Hyperparameter Optimization

You can combine transformer saving with hyperparameter optimization:

```bash
python main.py --model rf --hyperopt --hyperopt-save-name rf_optimized --save-transformer
```

This creates an optimized model with integrated feature engineering in a single pipeline.

---

## Accessing All Artifacts & Results

- Artifacts (models, splits, logs, metrics, etc.) are saved under:
    ```
    models/run_<TIMESTAMP>/
    ```
- Example:
    ```
    models/run_20250510_121801/
        ├── processed_data/
        |    ├── X_features.csv
        |    ├── y_target.csv
        |    ├── X_train.csv
        |    ├── X_val.csv
        |    ├── y_train.csv
        |    └── y_val.csv
        ├── model_lr_20250510_121805.joblib
        ├── model_metadata.txt
        ├── validation_predictions.csv
        └── ...
    ```
- MLflow tracks and versions every experiment. These are browsed and downloaded via the MLflow UI.

---

## Working with the MLflow UI

You can run the MLflow UI locally **even if you trained with a remote or file-based backend**:

```bash
mlflow ui
```

- Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
- Browse by run, compare metrics, download models/predictions, visualize parameters.

---

## Examples

**Train a Linear Regression on Jan+Feb 2024 TLC data:**

```bash
python main.py --model lr --months 01-2024 02-2024
```

Example output:

```
Using remote MLflow tracking server: http://127.0.0.1:5000
Loading data from remote parquet files for months: ['01-2024', '02-2024']
...
Feature engineering complete. Features: (100260, 37), Target: (100260,)
Train-validation split: Train: (80208, 37), Validation: (20052, 37)
Using default parameters for lr: {}
...
Model saved to: models/run_20250510_121804/model_lr_20250510_121805.joblib
Model Training Results (lr):
  Training RMSE: 277.5452
  Validation RMSE: 293.3884
  Validation R²: 0.7833
...
All artifacts saved in: models/run_20250510_121801
MLflow run ID: 4de88e59613f45f1b3fc2a329a78b4a9
View experiment details in MLflow UI: mlflow ui
```

---

## Troubleshooting

- **XGBoost/LightGBM Not Installed:**  
  If you get "XGBoost not installed. XGBoost models will not be available", run:
  ```bash
  pip install xgboost lightgbm
  ```

- **Parquet Loading Issues:**  
  Make sure you have `pyarrow` installed (`pip install pyarrow`).

- **Remote Tracking Connection Issues:**  
  Double-check your server address, or switch to local file-based tracking via MLflow.

- **Very Large Data:**  
  If you run out of memory, try selecting a single month or sampling the dataset.

---

## Acknowledgments

- NYC Taxi ([Open Data Portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page))
- [MLflow](https://mlflow.org/)
- [`hyperopt`](https://github.com/hyperopt/hyperopt) for Bayesian hyperparameter optimization.

---

For contributions, bugs, or questions—open an issue or pull request!

---

**Happy Modeling!**  
🚕💨📈