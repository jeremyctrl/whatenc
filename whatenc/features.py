import math
import string
import zlib
import numpy as np


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(s: str) -> np.ndarray:
    if not s:
        return np.zeros(8, dtype=float)

    n = len(s)
    printable = set(string.printable)
    encoded = s.encode("utf-8", errors="ignore")

    printable_ratio = sum(c in printable for c in s) / n
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
        ],
        dtype=float,
    )
