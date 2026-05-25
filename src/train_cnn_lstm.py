from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.bpe import BPETokenizer
from src.evaluate import classification_metrics, prediction_frame, robustness_by_language
from src.models import CNNLSTMClassifier
from src.utils import device, ensure_dir, set_seed, write_json


def _loader(texts, labels, tokenizer: BPETokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    encoded = torch.tensor([tokenizer.encode(text) for text in texts], dtype=torch.long)
    targets = torch.tensor(list(labels), dtype=torch.long)
    return DataLoader(TensorDataset(encoded, targets), batch_size=batch_size, shuffle=shuffle)


def _epoch(model, loader, criterion, optimizer=None, active_device=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(active_device)
        labels = labels.to(active_device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return float(np.mean(losses)), correct / max(total, 1)


def run_cnn_lstm(splits, config: dict, output_dir: str | Path, logger=None) -> dict:
    seed = config["data"]["random_state"]
    set_seed(seed)
    active_device = device()
    output_dir = ensure_dir(Path(output_dir) / "cnn_lstm_bpe")

    if logger:
        logger("Training BPE tokenizer")
    tokenizer = BPETokenizer.train(
        splits.train["text"],
        vocab_size=config["bpe"]["vocab_size"],
        min_pair_frequency=config["bpe"]["min_pair_frequency"],
        max_length=config["bpe"]["max_length"],
    )
    if logger:
        logger(f"BPE vocabulary size: {tokenizer.vocab_size()}")
        logger("Building PyTorch data loaders")
    train_loader = _loader(
        splits.train["text"],
        splits.train["label_id"],
        tokenizer,
        config["model"]["batch_size"],
        shuffle=True,
    )
    val_loader = _loader(
        splits.val["text"],
        splits.val["label_id"],
        tokenizer,
        config["model"]["batch_size"],
        shuffle=False,
    )
    test_loader = _loader(
        splits.test["text"],
        splits.test["label_id"],
        tokenizer,
        config["model"]["batch_size"],
        shuffle=False,
    )

    if logger:
        logger(f"Initialising CNN-LSTM on {active_device}")
    model = CNNLSTMClassifier(
        vocab_size=tokenizer.vocab_size(),
        num_classes=len(splits.label_encoder.classes_),
        embedding_dim=config["model"]["embedding_dim"],
        cnn_filters=config["model"]["cnn_filters"],
        kernel_size=config["model"]["kernel_size"],
        lstm_hidden=config["model"]["lstm_hidden"],
        dropout=config["model"]["dropout"],
    ).to(active_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

    history = []
    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, config["model"]["epochs"] + 1):
        train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, active_device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, active_device)
        if logger:
            logger(
                f"Epoch {epoch}/{config['model']['epochs']}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["model"]["patience"]:
                if logger:
                    logger(f"Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if logger:
        logger("Evaluating CNN-LSTM on test split")
    predictions = predict(model, test_loader, active_device)
    label_names = list(splits.label_encoder.classes_)
    pred_frame = prediction_frame(splits.test, predictions, splits.label_encoder)
    metrics = classification_metrics(splits.test["label_id"], predictions, label_names)
    metrics["robustness_by_language"] = robustness_by_language(pred_frame, label_names)
    metrics["history"] = history
    metrics["vocab_size"] = tokenizer.vocab_size()
    metrics["device"] = str(active_device)

    pred_frame.to_csv(output_dir / "predictions.csv", index=False)
    write_json(metrics, output_dir / "metrics.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "label_classes": list(splits.label_encoder.classes_),
            "config": config,
        },
        output_dir / "model.pt",
    )
    if logger:
        logger(
            "CNN-LSTM done: "
            f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_weighted']:.3f}; "
            f"outputs saved to {output_dir}"
        )
    return metrics


def predict(model, loader, active_device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(active_device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.array(predictions)
