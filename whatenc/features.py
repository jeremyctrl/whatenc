import math
import string
import zlib
import numpy as np

STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
    "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
    "so", "than","too", "very", "can","will", "just", "don", "should", "now",
]

ENGLISH_FREQ = {
    'E': 12.0, 'T': 9.1, 'A': 8.1, 'O': 7.7, 'I': 7.0, 'N': 6.7,
    'S': 6.3, 'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8,
    'U': 2.8, 'M': 2.4, 'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0,
    'P': 1.9, 'B': 1.5, 'V': 1.0, 'K': 0.8, 'J': 0.15, 'X': 0.15,
    'Q': 0.1, 'Z': 0.07,
}

ENGLISH_FREQ_VEC = np.array([ENGLISH_FREQ[c] for c in string.ascii_uppercase])

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def english_letter_corr(s: str) -> float:
    letters = [c for c in s.upper() if c.isalpha()]
    if len(letters) < 20:
        return 0.0
    counts = np.array([letters.count(c) for c in string.ascii_uppercase])
    if np.sum(counts) == 0:
        return 0.0
    corr = np.corrcoef(counts, ENGLISH_FREQ_VEC)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def stopword_ratio(s: str) -> float:
    words = [w.lower() for w in s.split()]
    if not words:
        return 0.0
    found = sum(w in STOPWORDS for w in words)
    return found / len(words)


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
            english_letter_corr(s),
            stopword_ratio(s),
        ],
        dtype=float,
    )
