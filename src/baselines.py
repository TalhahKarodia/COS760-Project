from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import classification_metrics, prediction_frame, robustness_by_language
from src.utils import ensure_dir, write_json


def build_baselines(config: dict) -> dict[str, Pipeline]:
    max_features = config["baselines"]["max_features"]
    word_range = tuple(config["baselines"]["word_ngram_range"])
    char_range = tuple(config["baselines"]["char_ngram_range"])
    return {
        "tfidf_word_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=word_range, max_features=max_features)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "tfidf_char_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=char_range, max_features=max_features)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "tfidf_char_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=char_range, max_features=max_features)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def run_baselines(splits, config: dict, output_dir: str | Path, logger=None) -> dict:
    output_dir = ensure_dir(Path(output_dir) / "baselines")
    label_names = list(splits.label_encoder.classes_)
    results = {}
    for name, pipeline in build_baselines(config).items():
        if logger:
            logger(f"Training baseline: {name}")
        pipeline.fit(splits.train["text"], splits.train["label_id"])
        if logger:
            logger(f"Evaluating baseline: {name}")
        predictions = pipeline.predict(splits.test["text"])
        pred_frame = prediction_frame(splits.test, predictions, splits.label_encoder)
        metrics = classification_metrics(splits.test["label_id"], predictions, label_names)
        metrics["robustness_by_language"] = robustness_by_language(pred_frame, label_names)
        results[name] = metrics

        pred_frame.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        write_json(metrics, output_dir / f"{name}_metrics.json")
        write_top_features(pipeline, label_names, output_dir / f"{name}_top_features.csv")
        if logger:
            logger(
                f"{name} done: "
                f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_weighted']:.3f}"
            )
    write_json(results, output_dir / "summary.json")
    if logger:
        logger(f"Baseline outputs saved to {output_dir}")
    return results


def write_top_features(pipeline: Pipeline, label_names: list[str], path: Path, top_k: int = 15) -> None:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    coefficients = classifier.coef_
    if coefficients.shape[0] == 1 and len(label_names) == 2:
        class_rows = [(label_names[1], coefficients[0]), (label_names[0], -coefficients[0])]
    else:
        class_rows = list(zip(label_names, coefficients))
    for label, weights in class_rows:
        for index in weights.argsort()[-top_k:][::-1]:
            rows.append({"class": label, "feature": feature_names[index], "weight": float(weights[index])})
    pd.DataFrame(rows).to_csv(path, index=False)
