from datasets import load_dataset
from config import CORPUS_PATH, LANG, CORPUS_LIMIT


def main():
    print(f"downloading wikipedia ({LANG})...")
    dataset = load_dataset(
        "wikimedia/wikipedia", f"20231101.{LANG}", split="train[:1%]"
    )
    lines = []
    for article in dataset:
        text = article["text"].strip()
        for line in text.split("\n"):
            if len(line.strip()) > 30:
                lines.append(line.strip())
        if len(lines) >= CORPUS_LIMIT:
            break
    CORPUS_PATH.write_text("\n".join(lines[:CORPUS_LIMIT]), encoding="utf-8")
    print(f"saved {len(lines[:CORPUS_LIMIT])} lines to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
