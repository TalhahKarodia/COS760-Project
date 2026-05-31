# Final Report Outline

## 1. Introduction

Briefly state the project goal: evaluating whether a subword-aware CNN-LSTM improves multilingual AfriSenti sentiment classification compared with TF-IDF word, character, and SVM baselines.

## 2. Dataset

Describe the AfriSenti Twitter sentiment dataset, the included language subsets, the sentiment labels, and how the CSV was exported using `scripts/download_afrisenti.py`.

## 3. Methodology

Explain the train/validation/test split, TF-IDF baselines, BPE tokenizer, CNN-LSTM architecture, and robustness analysis by language.

## 4. Experimental Setup

Record the Python version, dependency versions from `requirements.txt`, configuration file used, random seed, hardware, and command used to run the experiment.

## 5. Results

Summarise accuracy, weighted precision, weighted recall, weighted F1, classification reports, and language-level robustness from `results/`.

## 6. Explainability And Error Analysis

Discuss top TF-IDF features and CNN-LSTM token occlusion outputs. Include examples of correct and incorrect predictions where useful.

## 7. Discussion

Compare the neural model with traditional baselines, identify strengths and weaknesses, and comment on performance differences across languages.

## 8. Limitations

Mention dataset size/coverage limits, class imbalance, short social-media text, and that AfriSenti supports sentiment classification rather than true authorship attribution.

## 9. Conclusion

State the main findings and possible future improvements.
