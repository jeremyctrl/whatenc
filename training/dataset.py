import random
from config import CORPUS_PATH
from encoders import ENCODERS


def load_corpus():
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return lines


def generate_samples(corpus):
    samples = []
    for text in corpus:
        samples.append((text, "plain"))
        for name, encoder in ENCODERS.items():
            try:
                encoded = encoder(text)
                if not encoded or encoded == text:
                    continue
                samples.append((encoded, name))
            except Exception as e:
                print(f"skipped {name}: {e}")
    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    corpus = load_corpus()
    samples = generate_samples(corpus[:10])
    for s, lbl in samples[:10]:
        print(lbl, "->", s[:60])
