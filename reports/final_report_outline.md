# Final Report Outline

## 1. Introduction

State the research question: how well subword embeddings in a CNN-LSTM network improve stylistic or authorship classification compared with traditional extraction methods.

## 2. Background

Discuss word-level TF-IDF, word n-grams, character n-grams, BPE/SentencePiece-style subword tokenisation, and CNN-LSTM sequence models.

## 3. Dataset

Describe the dataset source, languages, label definition, class balance, and preprocessing. Include a table with row counts by language and label.

## 4. Methodology

Describe:

- Train/validation/test split.
- TF-IDF word and character baselines.
- BPE training on the training split only.
- CNN-LSTM architecture: embedding, convolution, LSTM, dropout, classifier.
- Hyperparameters from `configs/default.json`.

## 5. Evaluation Strategy

Report:

- Accuracy.
- Weighted precision.
- Weighted recall.
- Weighted F1-score.
- Per-language robustness metrics.
- Error and explanatory analysis.

## 6. Results

Use the JSON and CSV outputs under `results/` to build tables:

- Overall baseline comparison.
- CNN-LSTM BPE performance.
- Per-language robustness.
- Most influential TF-IDF features.
- Occlusion explanation examples.

## 7. Discussion

Discuss whether subword units improve performance, which languages benefit, where the model fails, and whether errors suggest morphology, spelling variation, class imbalance, or domain effects.

## 8. Conclusion

Answer the research question directly and identify future work, such as pretrained multilingual embeddings, larger BPE vocabularies, SHAP/LIME integration, or stronger transformer baselines.
