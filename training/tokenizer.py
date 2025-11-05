from collections import Counter

import torch


def extract_bigrams(text: str):
    if len(text) < 2:
        return [text]
    return [text[i : i + 2] for i in range(len(text) - 1)]


def build_vocab(samples, min_freq=-1):
    counter = Counter()
    for s, _ in samples:
        counter.update(extract_bigrams(s))
    vocab = [bg for bg, c in counter.items() if c >= min_freq]

    stoi = {bg: i + 1 for i, bg in enumerate(sorted(vocab))}
    itos = {i: bg for bg, i in stoi.items()}

    return stoi, itos


def encode_bigrams(text: str, stoi: dict):
    bigrams = extract_bigrams(text)
    return torch.tensor([stoi.get(bg, 0) for bg in bigrams], dtype=torch.long)
