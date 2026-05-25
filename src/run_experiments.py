from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines import run_baselines
from src.data import load_dataset, split_dataset
from src.explain import explain_saved_cnn_lstm
from src.train_cnn_lstm import run_cnn_lstm
from src.utils import ensure_dir, load_config, write_json


def log(message: str) -> None:
    print(f"[COS760] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COS760 subword-aware stylistic classification experiments.")
    parser.add_argument("--config", default="configs/default.json", help="Path to a JSON experiment config.")
    parser.add_argument("--skip-neural", action="store_true", help="Run only traditional TF-IDF baselines.")
    parser.add_argument("--skip-baselines", action="store_true", help="Run only the CNN-LSTM subword model.")
    args = parser.parse_args()

    log(f"Loading config from {args.config}")
    config = load_config(args.config)
    output_dir = ensure_dir(config["output_dir"])
    log(f"Loading dataset from {config['data']['path']}")
    data = load_dataset(
        config["data"]["path"],
        config["data"]["text_col"],
        config["data"]["label_col"],
        config["data"]["language_col"],
    )
    log(f"Loaded {len(data)} rows")
    log("Creating train/validation/test split")
    splits = split_dataset(
        data,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )

    summary = {
        "dataset_rows": len(data),
        "labels": list(splits.label_encoder.classes_),
        "languages": sorted(data["language"].unique().tolist()),
        "train_rows": len(splits.train),
        "val_rows": len(splits.val),
        "test_rows": len(splits.test),
    }
    log(
        "Split complete: "
        f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}, "
        f"labels={summary['labels']}, languages={summary['languages']}"
    )

    if not args.skip_baselines:
        log("Running TF-IDF baseline models")
        summary["baselines"] = run_baselines(splits, config, output_dir, logger=log)
    if not args.skip_neural:
        log("Running CNN-LSTM with BPE subword embeddings")
        summary["cnn_lstm_bpe"] = run_cnn_lstm(splits, config, output_dir, logger=log)
        model_path = Path(output_dir) / "cnn_lstm_bpe" / "model.pt"
        misclassified = Path(output_dir) / "cnn_lstm_bpe" / "predictions.csv"
        texts = _texts_for_explanation(misclassified, splits.test)
        if texts:
            log(f"Generating occlusion explanations for {len(texts)} examples")
            explain_saved_cnn_lstm(model_path, texts, Path(output_dir) / "explanations")
            log("Explanation output saved")

    log("Writing experiment summary")
    write_json(summary, Path(output_dir) / "experiment_summary.json")
    log(f"Experiment complete. Results written to {output_dir}")


def _texts_for_explanation(prediction_path: Path, fallback_test):
    if prediction_path.exists():
        import pandas as pd

        predictions = pd.read_csv(prediction_path)
        examples = predictions[predictions["correct"] == False]["text"].head(3).tolist()
        return examples or predictions["text"].head(3).tolist()
    return fallback_test["text"].head(3).tolist()


if __name__ == "__main__":
    main()
