import json

from config import META_PATH
from dataset import generate_samples, load_corpus
from tokenizer import build_vocab, encode_bigrams, extract_bigrams


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


if __name__ == "__main__":
    main()
