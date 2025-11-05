import json

import torch
import torch.nn as nn
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


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, 0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


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

    vocab_size = len(stoi)
    num_classes = len(label2idx)
    model = CNN(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes)

    batch_x, batch_y = next(iter(dataloader))
    logits = model(batch_x)
    print("Input shape:", batch_x.shape)
    print("Logits shape:", logits.shape)
    print("Example logits row:", logits[0].detach().numpy()[:5])


if __name__ == "__main__":
    main()
