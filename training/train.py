import json

import torch
from config import META_PATH
from dataset import generate_samples, load_corpus
from tokenizer import build_vocab, encode_bigrams
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, samples, stoi, label2idx):
        self.samples = samples
        self.stoi = stoi
        self.label2idx = label2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        x = encode_bigrams(text, self.stoi)
        y = self.label2idx[label]
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    ys = torch.tensor(ys)
    return xs, ys


def main():
    corpus = load_corpus()
    samples = generate_samples(corpus)

    stoi, itos = build_vocab(samples, min_freq=1)

    labels = sorted(set(label for _, label in samples))
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": itos,
                "label2idx": label2idx,
                "idx2label": idx2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    dataset = TextDataset(samples, stoi, label2idx)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch_x, batch_y = next(iter(dataloader))
    print("batch x shape:", batch_x.shape)
    print("batch y shape:", batch_y.shape)
    print("first sample (tensor):", batch_x[0][:30])
    print("first sample label id:", batch_y[0].item())


if __name__ == "__main__":
    main()
