import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    META_PATH,
    MODEL_PATH,
    TEST_SIZE,
)
from dataset import generate_samples, load_corpus
from tokenizer import build_vocab, encode_bigrams
from torch.utils.data import DataLoader, Dataset

MAX_LEN = 200


class TextDataset(Dataset):
    def __init__(self, samples, stoi, label2idx):
        self.samples = samples
        self.stoi = stoi
        self.label2idx = label2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = encode_bigrams(text[:MAX_LEN], self.stoi)
        x = torch.tensor(tokens[:MAX_LEN], dtype=torch.long)
        x = F.pad(x, (0, MAX_LEN - x.size(0)))
        y = torch.tensor(self.label2idx[label], dtype=torch.long)
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

    split = int(len(samples) * (1.0 - TEST_SIZE))

    train_dataset = TextDataset(samples[:split], stoi, label2idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TextDataset(samples[split:], stoi, label2idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    vocab_size = len(stoi)
    num_classes = len(label2idx)

    model = CNN(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        for idx, (X, y) in enumerate(train_loader):
            start = time.time()
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if idx % 50 == 0 or idx == len(train_loader):
                elapsed = time.time() - start
                print(
                    f"[epoch {epoch}] batch {idx}/{len(train_loader)} - {elapsed:.2f}s"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")

    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    cm_np = cm.numpy()
    label_width = max(len(lbl) for lbl in labels)

    header = " " * (label_width + 2) + " ".join(
        f"{lbl:>{label_width}}" for lbl in labels
    )
    print(header)

    for i, lbl in enumerate(labels):
        row_counts = " ".join(
            f"{cm_np[i, j]:>{label_width}d}" for j in range(len(labels))
        )
        print(f"{lbl:>{label_width}} | {row_counts}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
