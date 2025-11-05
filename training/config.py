from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

for d in (MODEL_DIR, DATA_DIR):
    d.mkdir(exist_ok=True, parents=True)

CORPUS_PATH = DATA_DIR / "wikipedia_corpus.txt"
META_PATH = MODEL_DIR / "meta.json"
MODEL_PATH = MODEL_DIR / "model.pt"
ONNX_PATH = MODEL_DIR / "model.onnx"

LANGS = ["en", "el", "ru", "he", "ar"]
CORPUS_LIMIT = 10000

TEST_SIZE = 0.2
RANDOM_STATE = 42
