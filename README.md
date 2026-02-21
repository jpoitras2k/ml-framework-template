# ML Framework Template

A generalized, boilerplate Machine Learning framework designed to kickstart tabular data analysis projects. It abstracts common data preprocessing, analysis, and modeling steps using `pandas` and `scikit-learn`, with optional `tensorflow`/`keras` support for deep learning tasks.

## Project Structure

```
ml-framework-template/
│
├── ml_framework_template/
│   ├── data_analyzer/            # Data ingestion, preprocessing, and EDA
│   │   ├── analyzer.py           # Exploratory Data Analysis, summaries, correlations
│   │   ├── data_preprocessing.py # Missing values, train/test splitting, shuffling
│   │   ├── data_reader.py        # Loading CSV/Excel files
│   │   ├── encoder.py            # Label Encoding & One-Hot Encoding
│   │   ├── scaler.py             # Standard Scaling & MinMax Scaling
│   │   └── visualizer.py        # Correlation matrices, histograms, scatter, boxplots
│   │
│   └── models/                   # Wrappers around common ML algorithms
│       ├── classifier.py         # Logistic Regression, Random Forest, SVC, KNN, Decision Tree
│       ├── clustering.py         # K-Means, DBSCAN
│       └── regressor.py          # Linear/Ridge/Lasso Regression, RF Regressor, SVR
│
├── tests/                        # Pytest test suite
├── main.py                       # Example pipeline orchestration script
├── pyproject.toml                # Project dependencies and tool configurations
└── README.md
```

## Getting Started

### Prerequisites

- Python **3.11** or **3.12** (required — Python 3.13+ is not yet supported by TensorFlow)
- [UV](https://docs.astral.sh/uv/) — fast Python package manager

### 1. Installation

Clone the repo and install all dependencies (including dev tools) with a single command:

```bash
uv sync --all-groups
```

This will automatically:
- Create a `.venv` virtual environment using a compatible Python version
- Install all runtime and development dependencies

### 2. Running the Pipeline

Run the example pipeline using UV (recommended — uses the managed virtual environment automatically):

```bash
uv run python main.py
```

Or activate the virtual environment manually first:

```bash
# Windows
.venv\Scripts\activate
python main.py
```

### 3. Adapting the Template

Open `main.py` and follow the `# TODO` markers to customize the pipeline for your project:

1. **Set your data source** — Update `DATA_FILE` and `TARGET_COLUMN`, then uncomment `DataReader` to load your CSV.
2. **Preprocessing** — Call methods on `Encoder`, `Scaler`, and `DataPreprocessor` to handle encoding, scaling, and missing values.
3. **EDA** — Uncomment `Analyzer` and `Visualizer` calls to explore your data.
4. **Select a model** — Use `Classifier`, `Regressor`, or `Clustering` from the `models/` package.
5. **Train and evaluate** — Call `.train()`, `.predict()`, and `.evaluate()`.

For **regression tasks**, uncomment:
```python
from ml_framework_template.models.regressor import Regressor
```

For **clustering tasks**, uncomment:
```python
from ml_framework_template.models.clustering import Clustering
```

### 4. Development Tools

This repository uses `taskipy` as a task runner. The following tasks are defined in `pyproject.toml`:

| Command                      | Description                          |
|------------------------------|--------------------------------------|
| `uv run task format`         | Format code with Black               |
| `uv run task format-check`   | Check formatting without modifying   |
| `uv run task flake`          | Run Flake8 linter                    |
| `uv run task pylint`         | Run Pylint linter                    |
| `uv run task test`           | Run pytest with coverage report      |

## Dependencies

| Package         | Purpose                          |
|-----------------|----------------------------------|
| `pandas`        | Data manipulation                |
| `scikit-learn`  | ML models and preprocessing      |
| `tensorflow`    | Deep learning (optional)         |
| `keras`         | High-level neural network API    |
| `matplotlib`    | Plotting                         |
| `seaborn`       | Statistical visualizations       |
| `jupyter`       | Notebook support                 |
