import math
import zlib
import numpy as np

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def non_ascii_ratio(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / len(s) if s else 0.0

def word_density(s: str) -> float:
    words = s.split()
    return math.log1p((len(s) / len(words))) if words else 0.0

def extract_features(s: str) -> np.ndarray:
    if not s:
        return np.zeros(10, dtype=float)

    n = len(s)
    encoded = s.encode("utf-8", errors="ignore")

    printable_ratio = sum(c.isprintable() for c in s) / n
    alpha_ratio = sum(c.isalpha() for c in s) / n
    digit_ratio = sum(c.isdigit() for c in s) / n
    padding_ratio = s.count("=") / n
    compressibility = len(zlib.compress(encoded)) / max(1, len(encoded))

    return np.array(
        [
            n,
            n % 4,
            printable_ratio,
            alpha_ratio,
            digit_ratio,
            padding_ratio,
            compressibility,
            shannon_entropy(s),
            non_ascii_ratio(s),
            word_density(s),
        ],
        dtype=float,
    )
