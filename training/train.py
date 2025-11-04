import sys
from pathlib import Path

import joblib
import numpy as np
from config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from dataset import generate_samples, load_corpus
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from whatenc.features import extract_features


def main():
    corpus = load_corpus()
    samples = generate_samples(corpus)

    X = np.array([extract_features(s) for s, _ in samples])
    y = np.array([label for _, label in samples])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    clf = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
