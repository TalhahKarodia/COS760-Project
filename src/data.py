from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_encoder: LabelEncoder


def load_dataset(path: str | Path, text_col: str, label_col: str, language_col: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = {text_col, label_col, language_col} - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data[[text_col, label_col, language_col]].rename(
        columns={text_col: "text", label_col: "label", language_col: "language"}
    )
    data = data.dropna(subset=["text", "label", "language"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""].reset_index(drop=True)
    return data


def _can_stratify(labels: pd.Series, requested_size: float) -> bool:
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return False
    requested_count = int(round(len(labels) * requested_size))
    return requested_count >= labels.nunique()


def split_dataset(
    data: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> DatasetSplits:
    label_encoder = LabelEncoder()
    data = data.copy()
    data["label_id"] = label_encoder.fit_transform(data["label"])

    stratify = data["label_id"] if _can_stratify(data["label_id"], test_size) else None
    train_val, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_fraction_of_train_val = val_size / (1.0 - test_size)
    stratify_tv = (
        train_val["label_id"]
        if _can_stratify(train_val["label_id"], val_fraction_of_train_val)
        else None
    )
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=stratify_tv,
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        label_encoder=label_encoder,
    )
