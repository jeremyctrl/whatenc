import argparse
import json
import sys
import sysconfig
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MAX_LEN = 200


def extract_bigrams(text: str):
    if len(text) < 2:
        return [text]
    return [text[i : i + 2] for i in range(len(text) - 1)]


def encode_bigrams(text: str, stoi: dict):
    bigrams = extract_bigrams(text)
    tokens = [stoi.get(bg, 0) for bg in bigrams[:MAX_LEN]]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    x = np.array(tokens, dtype=np.int64)[None, :]
    l = np.array([min(len(bigrams), MAX_LEN)], dtype=np.float32)  # noqa: E741
    return x, l


def predict(session, text: str, stoi: dict, idx2label: dict):
    x, l = encode_bigrams(text, stoi)  # noqa: E741
    outputs = session.run(None, {"input_text": x, "input_length": l})
    logits = torch.tensor(outputs[0])
    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    top_indices = probs.argsort()[::-1][:3]
    return [(idx2label[str(i)], float(probs[i])) for i in top_indices]


def print_result(line: str, top):
    print(f"[+] input: {line}")
    print(f"   [~] {'top guess':<11} = {top[0][0]}")
    for label, prob in top:
        print(f"      [=] {label:<8} = {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(prog="whatenc")
    parser.add_argument("input", help="string or path to text file")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.6.0")
    args = parser.parse_args()

    data_path = Path(sysconfig.get_paths()["data"]) / "models"
    model_path = data_path / "model.onnx"
    meta_path = data_path / "meta.json"

    if not model_path.exists() or not meta_path.exists():
        print("[!] model or metadata not found")
        sys.exit(1)

    print("[*] loading model")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    idx2label = meta["idx2label"]

    session = ort.InferenceSession(model_path)

    path = Path(args.input)
    if path.exists() and path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    top = predict(session, line, stoi, idx2label)
                    print_result(line, top)
        except Exception as e:
            print(f"[!] failed to read file: {e}")
    else:
        top = predict(session, args.input, stoi, idx2label)
        print_result(args.input, top)


if __name__ == "__main__":
    main()
