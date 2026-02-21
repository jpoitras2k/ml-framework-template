# ML Framework Template

This is a generalized, boilerplate Machine Learning framework designed to kickstart tabular data analysis projects. It abstracts common data preprocessing, analysis, and modeling steps using `pandas` and `scikit-learn` (and includes dependencies for `tensorflow`/`keras` if needed).

## Project Structure

Our framework is modularized into two primary packages, and orchestrated by `main.py`:

```
ml-framework-template/
│
├── ml_framework_template/
│   ├── data_analyzer/            # Data ingestion, preprocessing, and EDA
│   │   ├── analyzer.py           # Exploratory Data Analysis, summaries, correlations
│   │   ├── data_preprocessing.py # Missing values, splitting, shuffling
│   │   ├── data_reader.py        # Loading CSV/Excel
│   │   ├── encoder.py            # Label Encoding & One-Hot Encoding
│   │   └── scaler.py             # Standard Scaling & MinMax Scaling
│   │
│   ├── models/                   # Wrappers around common ML algorithms
│       ├── classifier.py         # Logistic Regression, Random Forest, SVC, KNN, DT
│       ├── clustering.py         # K-Means, DBSCAN
│       └── regressor.py          # Linear/Ridge/Lasso Regression, RF Regressor, SVR
│
├── main.py                       # Example pipeline orchestration script
├── pyproject.toml                # Project dependencies and tool configurations
└── README.md
```

## Getting Started

### 1. Installation

This project is configured using `uv` (a fast Python package installer and resolver). First, ensure you have Python 3.11 or 3.12 installed. Then run:

```bash
# Optional: create a virtual environment
uv venv
# Activate the environment (Windows)
.venv\Scripts\activate

# Install the project and dependencies
uv pip install -e .
```

Alternatively, you could just run scripts using `uv run main.py`.

### 2. Usage

Open the `main.py` base script. It contains a full dummy pipeline demonstrating the usage of all the modules. 

1. **Set your configurations**: Update the `DATA_FILE` path and `TARGET_COLUMN`.
2. **Handle preprocessing**: Call methods on `Encoder`, `Scaler`, and `DataPreprocessor` defining which specific columns needed adjustment.
3. **Select your model**: Initialize a model from `models.classifier`, `models.regressor`, or `models.clustering`.
4. **Train and Evaluate**: Call `train()`, `predict()`, and `evaluate()`.

### 3. Development Tools

This repository uses `taskipy` as a task runner.  The following tasks are defined in `pyproject.toml`:

*   **Format code:** `uv run task format` (runs Black)
*   **Check code formatting:** `uv run task format-check`
*   **Run linter:** `uv run task pylint`
*   **Run tests:** `uv run task test` (runs pytest)
