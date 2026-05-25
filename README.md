# COS760 Project: Subword-Aware Neural Language Models

This repository implements the COS760 project proposal: evaluating whether subword embeddings in a CNN-LSTM network improve multilingual stylistic classification compared with traditional TF-IDF, word n-gram, and character n-gram baselines.

## What Is Implemented

- CSV data loading with configurable `text`, `label`, and `language` columns.
- Traditional baselines:
  - TF-IDF word n-grams with logistic regression.
  - TF-IDF character n-grams with logistic regression.
  - TF-IDF character n-grams with linear SVM.
- A lightweight BPE tokenizer trained on the training split.
- CNN-LSTM classifier using learned subword embeddings.
- Evaluation with accuracy, weighted precision, weighted recall, weighted F1, and full classification reports.
- Robustness analysis by language.
- Explanatory analysis:
  - Top TF-IDF features per class for baseline models.
  - Token occlusion explanations for the CNN-LSTM model.
- A small multilingual sample dataset for smoke testing.

## Prerequisites

Install Python 3.10 or newer. Python 3.12/3.13 also works.

Required Python packages are listed in `requirements.txt`:

- `pandas`
- `scikit-learn`
- `torch`
- `numpy`
- `matplotlib`

## First-Time Setup

Open a terminal in the project root directory.

The project root directory is the folder that contains `README.md`, `requirements.txt`, `configs/`, and `src/`. If the project was downloaded as a ZIP, extract it first. If it was cloned with Git, move into the cloned repository folder.

Example:

```bash
cd path/to/COS760-Project
```

Then create a virtual environment and install the requirements.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On WSL/Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, your terminal prompt should show `(.venv)`. Use `python`, not `python3`, while the virtual environment is active.

## Running The Project

Make sure you are still in the project root directory and the virtual environment is active.

Run the full experiment:

```bash
python -m src.run_experiments --config configs/default.json
```

This runs:

- TF-IDF word logistic regression baseline.
- TF-IDF character logistic regression baseline.
- TF-IDF character SVM baseline.
- BPE tokenizer training.
- CNN-LSTM subword model training.
- Overall classification metrics.
- Per-language robustness analysis.
- Explanation/error analysis outputs.

The terminal prints progress while it runs, including baseline names, CNN-LSTM epoch loss/accuracy, early stopping, and output locations.

## Outputs

Results are written to `results/`.

Important files:

- `results/experiment_summary.json`: overall experiment summary.
- `results/baselines/summary.json`: all baseline results.
- `results/baselines/*_metrics.json`: metrics for each baseline.
- `results/baselines/*_top_features.csv`: most influential TF-IDF features.
- `results/cnn_lstm_bpe/metrics.json`: CNN-LSTM metrics and training history.
- `results/cnn_lstm_bpe/predictions.csv`: test predictions.
- `results/cnn_lstm_bpe/model.pt`: saved PyTorch model checkpoint.
- `results/explanations/cnn_lstm_occlusion_explanations.csv`: token-level occlusion explanations.

## Using The Real Dataset

The included file `data/sample/afrisenti_style_sample.csv` is only a small smoke-test dataset.

For the real AfriSenti experiment, export the Hugging Face dataset to a local CSV first:

```bash
python scripts/download_afrisenti.py
```

This downloads all 14 language subsets and writes:

```text
data/raw/afrisenti_all_languages.csv
```

Then run the experiment with the AfriSenti config:

```bash
python -m src.run_experiments --config configs/afrisenti_all.json
```

The export script uses the Hugging Face dataset `shmuhammad/AfriSenti-twitter-sentiment` and its converted Parquet files. It exports these subset codes:

```text
amh, arq, ary, hau, ibo, kin, orm, pcm, por, swa, tir, tso, twi, yor
```

Note that the Hugging Face code for Oromo is `orm`, and the code used for Mozambican Portuguese is `por`.

To download only specific languages, pass them with `--languages`:

```bash
python scripts/download_afrisenti.py --languages amh hau pcm yor
```

To use a different local CSV:

1. Put the CSV file in `data/raw/`.
2. Open a config file such as `configs/default.json` or `configs/afrisenti_all.json`.
3. Update the data path and column names.

Example:

```json
{
  "data": {
    "path": "data/raw/your_dataset.csv",
    "text_col": "text",
    "label_col": "label",
    "language_col": "language",
    "test_size": 0.2,
    "val_size": 0.2,
    "random_state": 42
  }
}
```

For a sentiment-classification version of the proposal, set `label_col` to the sentiment column. For an authorship/stylistic-attribution version, set `label_col` to the author, user, or writer column.

Then rerun:

```bash
python -m src.run_experiments --config configs/default.json
```

## Common Issues

If you see `ModuleNotFoundError: No module named 'pandas'`, the dependencies were not installed in the Python environment currently running the project. Activate the virtual environment and reinstall:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If Ubuntu/WSL shows `externally-managed-environment`, do not use system-wide pip. Create and use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3 -m venv .venv` fails on Ubuntu/WSL, install the venv package:

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv .venv
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Suggested Final Experiments

Run at least these configurations for the report:

1. Full multilingual dataset with labels as authors or sentiment classes.
2. Per-language robustness comparison using the generated `robustness_by_language` metrics.
3. BPE vocabulary sensitivity, for example `vocab_size` of 250, 500, and 1000.
4. Sequence-length sensitivity, for example `max_length` of 48, 96, and 128.
5. Error analysis on misclassified examples using the generated explanation CSVs.

## Repository Layout

- `configs/default.json`: experiment settings.
- `configs/afrisenti_all.json`: experiment settings for the exported AfriSenti dataset.
- `data/sample/`: smoke-test CSV.
- `scripts/download_afrisenti.py`: exports AfriSenti Hugging Face subsets to a local CSV.
- `src/data.py`: dataset loading and train/validation/test splitting.
- `src/bpe.py`: lightweight BPE tokenizer.
- `src/baselines.py`: TF-IDF baselines.
- `src/models.py`: CNN-LSTM model.
- `src/train_cnn_lstm.py`: neural training loop.
- `src/evaluate.py`: metrics and robustness reporting.
- `src/explain.py`: occlusion-based CNN-LSTM explanations.
- `src/run_experiments.py`: one-command experiment runner.
- `reports/final_report_outline.md`: report structure aligned to the brief.

## Notes

The sample dataset is only for verifying that the pipeline runs. Use the real dataset before drawing conclusions in the final report.
