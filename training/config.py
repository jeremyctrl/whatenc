from pathlib import Path

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")

CORPUS_PATH = DATA_DIR / "wikipedia_corpus.txt"

MODEL_PATH = MODEL_DIR / "model.joblib"
ONNX_PATH = MODEL_DIR / "model.onnx"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Corpus
LANG = "en"
CORPUS_LIMIT = 7500

# Model
TEST_SIZE = 0.2
RANDOM_STATE = 42
