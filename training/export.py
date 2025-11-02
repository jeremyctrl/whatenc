import sys
from pathlib import Path

import joblib
from config import MODEL_PATH, ONNX_PATH
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.append(str(Path(__file__).resolve().parents[1]))

from whatenc.features import extract_features

MODEL_INPUT_DIM = extract_features("test").shape[0]

def main():
    print("loading trained model")
    clf = joblib.load(MODEL_PATH)

    print("converting to onnx")
    initial_type = [("input", FloatTensorType([None, MODEL_INPUT_DIM]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"saved onnx model to {ONNX_PATH}")


if __name__ == "__main__":
    main()
