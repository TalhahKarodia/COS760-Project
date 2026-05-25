from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError


DEFAULT_DATASET = "shmuhammad/AfriSenti-twitter-sentiment"
DEFAULT_REVISION = "refs/convert/parquet"
DEFAULT_LANGUAGES = [
    "amh",
    "arq",
    "ary",
    "hau",
    "ibo",
    "kin",
    "orm",
    "pcm",
    "por",
    "swa",
    "tir",
    "tso",
    "twi",
    "yor",
]
DEFAULT_SPLITS = ["train", "validation", "test"]
LABEL_NAMES = {
    0: "positive",
    1: "negative",
    2: "neutral",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AfriSenti Hugging Face Parquet files to one local CSV.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset repository.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset repository revision/branch to use.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Language subset codes to download. Defaults to all 14 AfriSenti subsets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Splits to export. Defaults to train validation test.",
    )
    parser.add_argument("--output", default="data/raw/afrisenti_all_languages.csv", help="Output CSV path.")
    args = parser.parse_args()

    frames = []
    for language in args.languages:
        for split in args.splits:
            print(f"[download] Loading {args.dataset} / {language} / {split}", flush=True)
            frame = load_parquet_split(args.dataset, args.revision, language, split)
            if frame is None:
                print(f"[download] Skipping missing split: {language}/{split}", flush=True)
                continue
            frames.append(normalise_frame(frame, language, split))

    if not frames:
        raise RuntimeError(
            "No AfriSenti files were downloaded. Check your internet connection, Hugging Face access, "
            f"dataset name '{args.dataset}', and language codes: {', '.join(DEFAULT_LANGUAGES)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output, index=False)
    print(f"[download] Wrote {len(combined)} rows to {output}", flush=True)


def load_parquet_split(dataset: str, revision: str, language: str, split: str) -> pd.DataFrame | None:
    filename = f"{language}/{split}/0000.parquet"
    try:
        parquet_path = hf_hub_download(
            repo_id=dataset,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    except EntryNotFoundError:
        return None
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise RuntimeError(
            f"Could not download {dataset}/{filename} from revision {revision}. "
            "If rate-limited, run `huggingface-cli login` and try again."
        ) from exc
    return pd.read_parquet(parquet_path)


def normalise_frame(frame: pd.DataFrame, language: str, split: str) -> pd.DataFrame:
    text_col = "tweet" if "tweet" in frame.columns else "text" if "text" in frame.columns else None
    if text_col is None or "label" not in frame.columns:
        raise ValueError(
            f"Expected text/tweet and label columns for language {language}, split {split}. "
            f"Found columns: {list(frame.columns)}"
        )

    labels = frame["label"]
    if pd.api.types.is_integer_dtype(labels):
        labels = labels.map(lambda value: LABEL_NAMES.get(int(value), str(value)))

    return pd.DataFrame(
        {
            "text": frame[text_col].astype(str),
            "label": labels.astype(str),
            "language": language,
            "hf_split": split,
        }
    )


if __name__ == "__main__":
    main()
