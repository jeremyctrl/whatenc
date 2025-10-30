import argparse
import sys
import sysconfig
from pathlib import Path

import numpy as np
import onnxruntime as ort

from whatenc.features import extract_features


def predict(session: ort.InferenceSession, text: str):
    features = extract_features(text).astype(np.float32)
    features = np.expand_dims(features, 0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: features})

    probs = output[1][0]
    labels = list(probs.keys())
    probs = np.array(list(probs.values()), dtype=float)

    top_indices = np.argsort(probs)[::-1][:3]
    top = [(labels[i], probs[i]) for i in top_indices]

    return top


def print_result(line: str, top):
    print(f"[+] input: {line}")
    print(f"   [~] {'top guess':<11} = {top[0][0]}")
    for label, prob in top:
        print(f"      [=] {label:<8} = {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(prog="whatenc")
    parser.add_argument("input", help="string or path to text file")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.3.1")
    args = parser.parse_args()

    print("[*] loading model")

    model_path = Path(sysconfig.get_paths()["data"]) / "models" / "model.onnx"
    if not model_path.exists():
        print("[!] could not find model")
        sys.exit(1)

    session = ort.InferenceSession(model_path)

    path = Path(args.input)
    if path.exists() and path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    top = predict(session, line)
                    print_result(line, top)
        except Exception as e:
            print(f"[!] failed to read file: {e}")
    else:
        top = predict(session, args.input)
        print_result(args.input, top)


if __name__ == "__main__":
    main()
