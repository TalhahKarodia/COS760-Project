# COS760 Project: Subword-Aware Neural Language Models

This repository implements a COS760 subword-aware neural language modelling experiment: evaluating whether subword embeddings in a CNN-LSTM network improve multilingual AfriSenti sentiment classification compared with traditional TF-IDF, word n-gram, and character n-gram baselines.

## What Is Implemented

- CSV data loading with configurable text, target-label, and analysis-group columns.
- Traditional baselines:
  - TF-IDF word n-grams with logistic regression.
  - TF-IDF character n-grams with logistic regression.
  - TF-IDF character n-grams with linear SVM.
- A lightweight BPE tokenizer trained on the training split.
- CNN-LSTM classifier using learned subword embeddings.
- Evaluation with accuracy, weighted precision, weighted recall, weighted F1, and full classification reports.
- Robustness analysis by language or another configured grouping column.
- Explanatory analysis:
  - Top TF-IDF features per class for baseline models.
  - Token occlusion explanations for the CNN-LSTM model.
- A real AfriSenti dataset export workflow for running the main experiment.
- A small multilingual sample dataset for optional smoke testing only.

## Prerequisites

Install Python 3.10 or newer. Python 3.12/3.13 also works.

Required Python packages are listed in `requirements.txt`:

- `pandas`
- `scikit-learn`
- `torch`
- `numpy`
- `matplotlib`
- `datasets`
- `huggingface-hub`
- `pyarrow`

The code was prepared for Python 3.10+ and uses the package version ranges specified in `requirements.txt`. The smoke test was last run successfully with Python 3.13.5, pandas 2.3.0, scikit-learn 1.7.0, torch 2.12.0, numpy 2.3.0, matplotlib 3.10.3, datasets 4.8.5, huggingface-hub 1.15.0, and pyarrow 24.0.0.

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

## Running The Project With The Real Dataset

Make sure you are still in the project root directory and the virtual environment is active.

First download and export the real AfriSenti dataset to a local CSV:

```bash
python scripts/download_afrisenti.py
```

This creates:

```text
data/raw/afrisenti_all_languages.csv
```

Then run the full experiment with the real-dataset config:

```bash
python -m src.run_experiments --config configs/afrisenti_all.json
```

This is the main project run. AfriSenti provides `tweet` text and sentiment labels (`positive`, `negative`, and `neutral`), so the supervised task is multilingual sentiment classification. Because the dataset does not include author or user IDs, it should not be presented as true authorship attribution. Robustness is evaluated across the available language column.

It runs:

- TF-IDF word logistic regression baseline.
- TF-IDF character logistic regression baseline.
- TF-IDF character SVM baseline.
- BPE tokenizer training.
- CNN-LSTM subword model training.
- Overall classification metrics.
- Robustness analysis by configured data group.
- Explanation/error analysis outputs.

The terminal prints progress while it runs, including baseline names, CNN-LSTM epoch loss/accuracy, early stopping, and output locations.

To run only the traditional baselines on the real dataset:

```bash
python -m src.run_experiments --config configs/afrisenti_all.json --skip-neural
```

To run only the CNN-LSTM model on the real dataset:

```bash
python -m src.run_experiments --config configs/afrisenti_all.json --skip-baselines
```

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

## Dataset Details

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
2. Open `configs/afrisenti_all.json`.
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

For true authorship or writer-style attribution, use a different dataset with an `author`, `user_id`, or `writer` column and set `label_col` to that column. If no author/user labels exist, do not fabricate them. For AfriSenti, keep `label_col` set to the sentiment `label` column and discuss language-level robustness separately.

Then rerun:

```bash
python -m src.run_experiments --config configs/afrisenti_all.json
```

The included `data/sample/afrisenti_style_sample.csv` file is only for quick smoke testing. Do not use it for the main project results.

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

## Contents Of The Zip File

- `configs/default.json`: optional smoke-test settings using the sample CSV.
- `configs/afrisenti_all.json`: main sentiment-classification settings for the exported AfriSenti dataset.
- `data/sample/`: optional smoke-test CSV.
- `data/raw/`: location for the exported real AfriSenti CSV.
- `requirements.txt`: Python dependency list for recreating the environment.
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

Use `configs/afrisenti_all.json` for the real project run and final report results. It predicts AfriSenti sentiment labels and reports robustness across languages. The sample config exists only to verify that the pipeline starts correctly.
