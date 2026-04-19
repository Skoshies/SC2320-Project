# SC2320 Project: Truthfulness Classification on the LIAR Dataset

This repository contains a course project notebook for **truthfulness classification** on the **LIAR** dataset.

## Project summary

The project studies how performance changes when we expand the feature space from a simple text-only baseline to richer engineered features.

Main components:
- **Baseline:** TF-IDF + Multinomial Naive Bayes
- **Improvement 1:** metadata features
- **Improvement 2:** speaker-history / credibility features
- **Improvement 3:** similar-claim retrieval features using MinHash + LSH
- **Final models:** improved linear model and XGBoost model

The notebook also includes:
- exploratory checks
- evaluation helpers
- ablation study
- plots
- threshold tuning
- error analysis
- CSV export of key outputs

## Repository structure

```text
SC2320_Project_v2.ipynb
requirements.txt
README.md
data/
  train.tsv
  valid.tsv
  test.tsv
outputs/
  liar_project_outputs/
    figures/
    tables/
```

## Environment setup

Before running the notebook, create a Python virtual environment in the project folder and install the required packages.

### 1. Open the project folder

Clone or download this repository, then open the project folder in VS Code.

### 2. Create and activate a virtual environment

Run the following commands in the terminal from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `jupyter` and `ipykernel` are not already included in `requirements.txt`, install them with:

```bash
pip install jupyter ipykernel
```

### 3. macOS note for XGBoost

If you are using macOS, `xgboost` may fail with a missing `libomp.dylib` error.  
If that happens, install OpenMP with:

```bash
brew install libomp
```

### 4. Open the project in VS Code

If needed, launch VS Code from the project folder with:

```bash
code .
```

### 5. Select the correct Python interpreter / notebook kernel

In VS Code:
- open `SC2320_Project_v2.ipynb`
- click the kernel selector at the top right
- choose the interpreter inside `.venv`

The selected interpreter should point to something like:

```text
.../SC2320-Project/.venv/bin/python
```

### 6. Verify the environment inside the notebook

You can optionally run the following in a notebook cell to confirm the notebook is using the correct Python environment:

```python
import sys
print(sys.executable)
```

## Running the notebook

Run the notebook **from top to bottom** in order:

- `SC2320_Project_v2.ipynb`

Do not skip cells, since later sections depend on variables and helper functions defined earlier.

The notebook expects these files to exist:

- `data/train.tsv`
- `data/valid.tsv`
- `data/test.tsv`

## Generated outputs

The notebook writes outputs to:

```text
outputs/liar_project_outputs/
```

Expected exports include:

### Figures
- `figures/roc_curves_no_graph.png`
- `figures/threshold_tuning_no_graph.png`

### Tables / CSV files
- `tables/model_comparison.csv`
- `tables/ablation_results.csv`
- `tables/improved_cases.csv`
- `tables/still_wrong_cases.csv`

## Notes on scope

The original LIAR dataset does not provide a social propagation or retweet graph.  
Therefore, this project focuses on a reproducible feature-engineering and retrieval-based extension of the task rather than claiming to use unavailable graph data.

## Troubleshooting

### `ModuleNotFoundError`
Make sure the notebook is using the `.venv` kernel and that all packages were installed with:

```bash
pip install -r requirements.txt
```

If needed, also run:

```bash
pip install jupyter ipykernel
```

### `XGBoostError: libomp.dylib not found`
On macOS, install OpenMP:

```bash
brew install libomp
```

### `FileNotFoundError: data/train.tsv`
Make sure the dataset files are located exactly at:

```text
data/train.tsv
data/valid.tsv
data/test.tsv
```

### Notebook kernel issues in VS Code
If the notebook cannot run or the kernel becomes unavailable:
- reload the VS Code window
- reselect the `.venv` kernel
- restart the notebook kernel and run all cells again
