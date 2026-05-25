from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    labels = list(range(len(label_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def robustness_by_language(frame: pd.DataFrame, label_names: list[str]) -> dict:
    scores = {}
    for language, group in frame.groupby("language"):
        scores[str(language)] = classification_metrics(group["label_id"], group["prediction_id"], label_names)
    return scores


def prediction_frame(source: pd.DataFrame, predictions, label_encoder) -> pd.DataFrame:
    output = source[["text", "language", "label", "label_id"]].copy()
    output["prediction_id"] = predictions
    output["prediction"] = label_encoder.inverse_transform(predictions)
    output["correct"] = output["label_id"] == output["prediction_id"]
    return output
