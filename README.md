# COS760 Project: Subword-Aware Neural Language Models

This repository implements the COS760 project proposal: evaluating whether subword embeddings in a CNN-LSTM network improve multilingual stylistic classification compared with traditional TF-IDF, word n-gram, and character n-gram baselines.

## What Is Implemented

- CSV data loading with configurable `text`, `label`, and `language` columns.
- Traditional baselines:
  - TF-IDF word n-grams with logistic regression.
  - TF-IDF character n-grams with logistic regression.
  - TF-IDF character n-grams with linear SVM.
- A dependency-light BPE tokenizer trained on the training split.
- CNN-LSTM classifier using learned subword embeddings.
- Evaluation with accuracy, weighted precision, weighted recall, weighted F1, and full classification reports.
- Robustness analysis by language.
- Explanatory analysis:
  - Top TF-IDF features per class for baseline models.
  - Token occlusion explanations for the CNN-LSTM model.
- A small multilingual sample dataset for smoke testing.

## Quick Start

From this folder:

```powershell
python -m src.run_experiments --config configs/default.json
```

Outputs are written to `results/`, including:

- `experiment_summary.json`
- `baselines/*_metrics.json`
- `baselines/*_top_features.csv`
- `cnn_lstm_bpe/metrics.json`
- `cnn_lstm_bpe/predictions.csv`
- `explanations/cnn_lstm_occlusion_explanations.csv`


## Suggested Final Experiments

Run at least these configurations for the report:

1. Full multilingual dataset with labels as authors or sentiment classes.
2. Per-language robustness comparison using the generated `robustness_by_language` metrics.
3. BPE vocabulary sensitivity, for example `vocab_size` of 250, 500, and 1000.
4. Sequence-length sensitivity, for example `max_length` of 48, 96, and 128.
5. Error analysis on misclassified examples using the generated explanation CSVs.

## Repository Layout

- `configs/default.json`: experiment settings.
- `data/sample/`: smoke-test CSV.
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

The included sample data is only for verifying that the pipeline runs. Use the real dataset before drawing conclusions.
