from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.bpe import BPETokenizer
from src.models import CNNLSTMClassifier
from src.utils import device, ensure_dir


def explain_saved_cnn_lstm(model_path: str | Path, texts: list[str], output_dir: str | Path, top_k: int = 12) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer: BPETokenizer = checkpoint["tokenizer"]
    label_classes = checkpoint["label_classes"]
    active_device = device()
    model = CNNLSTMClassifier(
        vocab_size=tokenizer.vocab_size(),
        num_classes=len(label_classes),
        embedding_dim=config["model"]["embedding_dim"],
        cnn_filters=config["model"]["cnn_filters"],
        kernel_size=config["model"]["kernel_size"],
        lstm_hidden=config["model"]["lstm_hidden"],
        dropout=config["model"]["dropout"],
    ).to(active_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        tokens = tokenizer.encode_tokens(text)
        baseline_prob, predicted = _confidence(model, token_ids, active_device)
        for position, token in enumerate(tokens[: tokenizer.max_length]):
            occluded = token_ids.copy()
            occluded[position] = 0
            occluded_prob, _ = _confidence(model, occluded, active_device, predicted)
            rows.append(
                {
                    "text": text,
                    "predicted_label": label_classes[predicted],
                    "token": token,
                    "position": position,
                    "confidence_drop": float(baseline_prob - occluded_prob),
                }
            )
    frame = pd.DataFrame(rows).sort_values("confidence_drop", ascending=False).head(top_k)
    output_dir = ensure_dir(output_dir)
    frame.to_csv(Path(output_dir) / "cnn_lstm_occlusion_explanations.csv", index=False)
    return frame


def _confidence(model, token_ids: list[int], active_device, class_id: int | None = None) -> tuple[float, int]:
    inputs = torch.tensor([token_ids], dtype=torch.long).to(active_device)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    predicted = int(probabilities.argmax().item()) if class_id is None else class_id
    return float(probabilities[predicted].item()), predicted
