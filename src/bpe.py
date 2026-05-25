from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

PAD = "<pad>"
UNK = "<unk>"


@dataclass
class BPETokenizer:
    merges: list[tuple[str, str]]
    token_to_id: dict[str, int]
    max_length: int

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_pair_frequency: int,
        max_length: int,
    ) -> "BPETokenizer":
        word_freq: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freq[tuple(word) + ("</w>",)] += 1

        merges: list[tuple[str, str]] = []
        while len(_symbols(word_freq)) + len(merges) + 2 < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, freq in word_freq.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_frequency:
                break
            merges.append(best_pair)
            word_freq = _merge_pair(word_freq, best_pair)

        vocab = [PAD, UNK]
        for symbols in word_freq:
            for symbol in symbols:
                if symbol != "</w>" and symbol not in vocab:
                    vocab.append(symbol)
        return cls(merges=merges, token_to_id={token: i for i, token in enumerate(vocab)}, max_length=max_length)

    def encode_tokens(self, text: str) -> list[str]:
        output: list[str] = []
        for word in text.lower().split():
            symbols = tuple(word) + ("</w>",)
            for pair in self.merges:
                symbols = _merge_symbols(symbols, pair)
            output.extend(symbol for symbol in symbols if symbol != "</w>")
        return output[: self.max_length]

    def encode(self, text: str) -> list[int]:
        token_ids = [self.token_to_id.get(token, self.token_to_id[UNK]) for token in self.encode_tokens(text)]
        token_ids = token_ids[: self.max_length]
        return token_ids + [self.token_to_id[PAD]] * (self.max_length - len(token_ids))

    def vocab_size(self) -> int:
        return len(self.token_to_id)


def _symbols(word_freq: Counter[tuple[str, ...]]) -> set[str]:
    return {symbol for word in word_freq for symbol in word if symbol != "</w>"}


def _merge_pair(
    word_freq: Counter[tuple[str, ...]],
    pair: tuple[str, str],
) -> Counter[tuple[str, ...]]:
    return Counter({_merge_symbols(symbols, pair): freq for symbols, freq in word_freq.items()})


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)
