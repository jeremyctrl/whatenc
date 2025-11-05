import random

from config import CORPUS_LIMIT, CORPUS_PATH, LANGS
from datasets import load_dataset

def fetch_corpus(lang: str, limit: int) -> list[str]:
    print(f"downloading wikipedia ({lang})...")

    dataset = load_dataset(
        "wikimedia/wikipedia", f"20231101.{lang}", split="train[:1%]"
    )
    dataset = list(dataset)
    random.shuffle(dataset)

    lines = []
    for article in dataset:
        text = article["text"].strip()
        for line in text.split("\n"):
            if len(line.strip()) > 4:
                lines.append(line.strip())
        if len(lines) >= CORPUS_LIMIT:
            break
    print(f"Collected {len(lines)} lines for {lang}")
    return lines

def main():
    lines = []
    for lang in LANGS:
        try:
            lines.extend(fetch_corpus(lang, CORPUS_LIMIT // len(LANGS)))
        except Exception as e:
            print(f"Skipping {lang} due to error: {e}")
    random.shuffle(lines)
    
    CORPUS_PATH.write_text("\n".join(lines[:CORPUS_LIMIT]), encoding="utf-8")
    print(f"saved {len(lines[:CORPUS_LIMIT])} lines across {len(LANGS)} languages to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
