from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_weighted": "Weighted precision",
    "recall_weighted": "Weighted recall",
    "f1_weighted": "Weighted F1",
}
MODEL_LABELS = {
    "tfidf_word_logreg": "TF-IDF word LR",
    "tfidf_char_logreg": "TF-IDF char LR",
    "tfidf_char_svm": "TF-IDF char SVM",
    "cnn_lstm_bpe": "CNN-LSTM BPE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polished Markdown, CSV, and figure summaries from results/.")
    parser.add_argument("--results-dir", default="results", help="Directory containing experiment_summary.json.")
    parser.add_argument("--output-dir", default="reports", help="Directory where report assets will be written.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(results_dir / "experiment_summary.json")
    models = collect_models(summary, results_dir)
    if not models:
        raise RuntimeError(f"No model metrics found in {results_dir}. Run the experiment before generating the report.")

    overall = build_overall_table(models)
    robustness = build_robustness_table(models)
    class_report = build_class_report_table(models)

    overall.to_csv(tables_dir / "overall_metrics.csv", index=False)
    if not robustness.empty:
        robustness.to_csv(tables_dir / "language_robustness.csv", index=False)
    if not class_report.empty:
        class_report.to_csv(tables_dir / "class_metrics.csv", index=False)

    plot_overall_metrics(overall, figures_dir / "overall_metrics.png")
    if not robustness.empty:
        plot_language_f1(robustness, figures_dir / "language_weighted_f1.png")
    if "cnn_lstm_bpe" in models and models["cnn_lstm_bpe"].get("history"):
        plot_training_history(models["cnn_lstm_bpe"]["history"], figures_dir / "cnn_lstm_training_history.png")

    report = build_markdown_report(summary, overall, robustness, class_report)
    report_path = output_dir / "results_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] Wrote {report_path}")
    print(f"[report] Wrote tables to {tables_dir}")
    print(f"[report] Wrote figures to {figures_dir}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_models(summary: dict[str, Any], results_dir: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for name, metrics in summary.get("baselines", {}).items():
        models[name] = metrics

    cnn_metrics = summary.get("cnn_lstm_bpe")
    cnn_path = results_dir / "cnn_lstm_bpe" / "metrics.json"
    if cnn_metrics:
        models["cnn_lstm_bpe"] = cnn_metrics
    elif cnn_path.exists():
        models["cnn_lstm_bpe"] = load_json(cnn_path)
    return models


def display_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def build_overall_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, metrics in models.items():
        row = {"model": model, "model_name": display_model_name(model)}
        for metric in METRIC_COLUMNS:
            row[metric] = metrics.get(metric)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("f1_weighted", ascending=False).reset_index(drop=True)


def build_robustness_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, metrics in models.items():
        for language, language_metrics in metrics.get("robustness_by_language", {}).items():
            rows.append(
                {
                    "model": model,
                    "model_name": display_model_name(model),
                    "language": language,
                    "accuracy": language_metrics.get("accuracy"),
                    "precision_weighted": language_metrics.get("precision_weighted"),
                    "recall_weighted": language_metrics.get("recall_weighted"),
                    "f1_weighted": language_metrics.get("f1_weighted"),
                }
            )
    return pd.DataFrame(rows)


def build_class_report_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, metrics in models.items():
        report = metrics.get("classification_report", {})
        for label, label_metrics in report.items():
            if not isinstance(label_metrics, dict) or label in {"macro avg", "weighted avg"}:
                continue
            rows.append(
                {
                    "model": model,
                    "model_name": display_model_name(model),
                    "class": label,
                    "precision": label_metrics.get("precision"),
                    "recall": label_metrics.get("recall"),
                    "f1": label_metrics.get("f1-score"),
                    "support": int(label_metrics.get("support", 0)),
                }
            )
    return pd.DataFrame(rows)


def plot_overall_metrics(overall: pd.DataFrame, path: Path) -> None:
    plot_frame = overall.set_index("model_name")[METRIC_COLUMNS].rename(columns=METRIC_LABELS)
    ax = plot_frame.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_title("Overall Model Performance")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_language_f1(robustness: pd.DataFrame, path: Path) -> None:
    pivot = robustness.pivot_table(index="language", columns="model_name", values="f1_weighted")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    ax = pivot.plot(kind="bar", figsize=(12, 5.5), width=0.82)
    ax.set_title("Weighted F1 By Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_training_history(history: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(frame["epoch"], frame["train_accuracy"], marker="o", label="Train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("CNN-LSTM Training History")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_markdown_report(
    summary: dict[str, Any],
    overall: pd.DataFrame,
    robustness: pd.DataFrame,
    class_report: pd.DataFrame,
) -> str:
    best = overall.iloc[0]
    lines = [
        "# COS760 Results Summary",
        "",
        "## Dataset",
        "",
        f"- Rows: {summary.get('dataset_rows', 'n/a')}",
        f"- Train / validation / test: {summary.get('train_rows', 'n/a')} / {summary.get('val_rows', 'n/a')} / {summary.get('test_rows', 'n/a')}",
        f"- Labels: {', '.join(summary.get('labels', []))}",
        f"- Language groups: {', '.join(summary.get('groups', []))}",
        "",
        "## Overall Performance",
        "",
        f"Best weighted F1: **{best['model_name']}** ({format_score(best['f1_weighted'])}).",
        "",
        markdown_table(
            overall,
            ["model_name", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
            {
                "model_name": "Model",
                "accuracy": "Accuracy",
                "precision_weighted": "Weighted precision",
                "recall_weighted": "Weighted recall",
                "f1_weighted": "Weighted F1",
            },
        ),
        "",
        "![Overall model performance](figures/overall_metrics.png)",
        "",
    ]

    if not robustness.empty:
        top_languages = robustness.sort_values(["model_name", "f1_weighted"], ascending=[True, False])
        lines.extend(
            [
                "## Language Robustness",
                "",
                markdown_table(
                    top_languages,
                    ["model_name", "language", "accuracy", "f1_weighted"],
                    {
                        "model_name": "Model",
                        "language": "Language",
                        "accuracy": "Accuracy",
                        "f1_weighted": "Weighted F1",
                    },
                ),
                "",
                "![Weighted F1 by language](figures/language_weighted_f1.png)",
                "",
            ]
        )

    if not class_report.empty:
        lines.extend(
            [
                "## Per-Class Metrics",
                "",
                markdown_table(
                    class_report.sort_values(["model_name", "class"]),
                    ["model_name", "class", "precision", "recall", "f1", "support"],
                    {
                        "model_name": "Model",
                        "class": "Class",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "support": "Support",
                    },
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Generated Assets",
            "",
            "- `reports/tables/overall_metrics.csv`",
            "- `reports/tables/language_robustness.csv`",
            "- `reports/tables/class_metrics.csv`",
            "- `reports/figures/overall_metrics.png`",
            "- `reports/figures/language_weighted_f1.png`",
            "- `reports/figures/cnn_lstm_training_history.png` when CNN-LSTM history is available",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: dict[str, str]) -> str:
    output = []
    header = [headers[column] for column in columns]
    output.append("| " + " | ".join(header) + " |")
    output.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame[columns].iterrows():
        values = [format_cell(row[column]) for column in columns]
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return format_score(value)
    return str(value)


def format_score(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
