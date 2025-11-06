import json
import torch
from config import MODEL_PATH, ONNX_PATH, META_PATH, MAX_LEN
from train import CNN


def main():
    print("loading metadata")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab_size = len(meta["stoi"])
    num_classes = len(meta["label2idx"])

    print("initializing model")
    model = CNN(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("exporting to ONNX")
    dummy_x = torch.zeros(1, MAX_LEN, dtype=torch.long)
    dummy_len = torch.tensor([MAX_LEN], dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x, dummy_len),
        ONNX_PATH,
        input_names=["input_text", "input_length"],
        output_names=["logits"],
        dynamic_axes={
            "input_text": {0: "batch_size"},
            "input_length": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"saved ONNX model to {ONNX_PATH}")


if __name__ == "__main__":
    main()
