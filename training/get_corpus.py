import random
import re
from pathlib import Path

import requests
from config import CORPUS_LIMIT, CORPUS_PATH, LANGS, STOPWORDS_PATH, NUMBERS_PATH
from datasets import load_dataset

HTML_TAGS = re.compile(r"<[^>]+>")

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
    print(f"collected {len(lines)} wikipedia lines for {lang}")
    return lines


def main():
    lines = []
    for lang in LANGS:
        try:
            lines.extend(fetch_corpus(lang, CORPUS_LIMIT // len(LANGS)))
        except Exception as e:
            print(f"skipping wikipedia lines for {lang} due to error: {e}")

        template = Path(str(STOPWORDS_PATH).format(lang=lang))
        if template.exists():
            stopwords = template.read_text().splitlines()
            lines.extend(stopwords)
            print(f"collected {len(stopwords)} stopwords for {lang}")
        else:
            try:
                response = requests.get(
                    f"https://raw.githubusercontent.com/stopwords-iso/stopwords-{lang}/refs/heads/master/stopwords-{lang}.txt"
                )
                response.raise_for_status()

                stopwords = response.text.splitlines()
                lines.extend(stopwords)
                print(f"collected {len(stopwords)} stopwords for {lang}")

                template.write_text(response.text)
            except Exception as e:
                print(f"skipping stopwords for {lang} due to error: {e}")
    
    if NUMBERS_PATH.exists():
        numbers = NUMBERS_PATH.read_text().splitlines()
        lines.extend(numbers)
        print(f"collected {len(numbers)} numbers")
    else:
        numbers = []
        for _ in range(40):
            numbers.append(str(random.randint(10, 99)))
        for _ in range(40):
            numbers.append(str(random.randint(100, 999)))
        for _ in range(40):
            numbers.append(str(random.randint(1000, 9999)))
        for _ in range(40):
            numbers.append(str(random.randint(10000, 99999)))
        lines.extend(numbers)
        NUMBERS_PATH.write_text("\n".join(numbers))
        print(f"collected {len(numbers)} numbers")

    lines = [s for s in lines if not HTML_TAGS.search(s)]

    random.shuffle(lines)

    CORPUS_PATH.write_text("\n".join(lines[:CORPUS_LIMIT]), encoding="utf-8")
    print(
        f"saved {len(lines[:CORPUS_LIMIT])} lines across {len(LANGS)} languages to {CORPUS_PATH}"
    )


if __name__ == "__main__":
    main()
