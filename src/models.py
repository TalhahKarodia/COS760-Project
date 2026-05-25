from __future__ import annotations

import torch
from torch import nn


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        cnn_filters: int,
        kernel_size: int,
        lstm_hidden: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(embedding_dim, cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        convolved = self.activation(self.conv(embedded.transpose(1, 2))).transpose(1, 2)
        output, _ = self.lstm(convolved)
        pooled = output.mean(dim=1)
        return self.classifier(self.dropout(pooled))
