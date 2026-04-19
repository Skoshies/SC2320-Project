# SC2320 Project: Truthfulness Classification on the LIAR Dataset

This repository contains the **v3 notebook** for the SC2320 course project on **truthfulness classification** using the **LIAR** dataset.

## Project summary

The project studies how classification performance changes when the feature space is expanded from a simple text-only baseline to richer engineered features.

The current notebook follows this progression:

- **Baseline:** TF-IDF + Multinomial Naive Bayes
- **Improvement 1:** metadata features
- **Improvement 2:** speaker-history / credibility features
- **Improvement 3:** similar-claim retrieval features using MinHash + LSH
- **Final nonlinear model:** tuned XGBoost on a compressed mixed-feature representation

The notebook also includes:

- exploratory checks
- evaluation helpers
- six-class sanity check for the binary label mapping
- LSH setting selection
- validation-based hyperparameter selection for XGBoost
- threshold tuning
- ablation study
- comparative visualisations
- error analysis
- CSV export of key outputs

## Repository structure

```text
SC2320_Project_v3.ipynb
requirements.txt
README.md
data/
  train.tsv
  valid.tsv
  test.tsv
outputs/
  figures/
  tables/
```

## Environment setup

Create a Python virtual environment in the project folder and install the required packages.

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

### 3. Open the project in VS Code

If needed, launch VS Code from the project folder with:

```bash
code .
```

### 4. Select the correct Python interpreter / notebook kernel

In VS Code:

- open `SC2320_Project_v3.ipynb`
- click the kernel selector at the top right
- choose the interpreter inside `.venv`

The selected interpreter should point to something like:

```text
.../SC2320-Project/.venv/bin/python
```

### 5. Verify the environment inside the notebook

You can optionally run the following in a notebook cell to confirm the notebook is using the correct Python environment:

```python
import sys
print(sys.executable)
```

## Running the notebook

Run the notebook **from top to bottom** in order:

- `SC2320_Project_v3.ipynb`

Do not skip cells, since later sections depend on variables and helper functions defined earlier.

The notebook expects these dataset files to exist:

- `data/train.tsv`
- `data/valid.tsv`
- `data/test.tsv`

## Generated outputs

The notebook writes outputs to:

```text
outputs/
  figures/
  tables/
```

### Figures

The current notebook exports the following figures:

- `figures/class_distribution_across_splits.png`
- `figures/baseline_confusion_matrix_test.png`
- `figures/xgboost_confusion_matrix_test.png`
- `figures/model_comparison_accuracy_macro_f1.png`
- `figures/false_class_recall_comparison.png`
- `figures/roc_curves_all_models_test.png`
- `figures/threshold_tuning_tuned_xgb.png`

### Tables / CSV files

The notebook exports the following CSV files:

- `tables/model_comparison.csv`
- `tables/ablation_results.csv`
- `tables/improved_cases.csv`
- `tables/still_wrong_cases.csv`

## Notes on scope

The original LIAR dataset does not provide an external social graph or propagation graph.  
For that reason, the final plain v3 notebook focuses on:

- text features
- metadata features
- speaker-history features
- retrieval-based features
- nonlinear classification with XGBoost

Graph-derived structural features such as degree and PageRank were explored earlier in development, but were not retained in the final v3 pipeline because they did not provide clear gains and were not grounded in a true propagation or interaction graph.

## GPU note for XGBoost

The notebook currently uses GPU-enabled XGBoost settings during hyperparameter selection:

- `tree_method="hist"`
- `device="cuda"`

If your environment does **not** support CUDA-enabled XGBoost, you have two options:

1. install a working GPU-enabled XGBoost environment, or
2. switch the XGBoost cells to CPU by removing `device="cuda"`.

The notebook will still work on CPU, but the validation-based hyperparameter search may take significantly longer.

### macOS note for XGBoost

If you are using macOS, `xgboost` may fail with a missing `libomp.dylib` error.  
If that happens, install OpenMP with:

```bash
brew install libomp
```

## Troubleshooting

### `ModuleNotFoundError`

Make sure the notebook is using the `.venv` kernel and that all packages were installed with:

```bash
pip install -r requirements.txt
```

### `XGBoostError` or CUDA-related errors

If CUDA is unavailable in your environment:

- remove `device="cuda"` from the XGBoost cells, or
- use a Python environment with a working CUDA-enabled XGBoost installation.

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
