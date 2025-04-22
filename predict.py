import os
from typing import List, Union
import torch
from PIL import Image
import base64
import io

from config import DEVICE
from model_utils import load_model, get_transforms

transform = get_transforms()


def predict_batch(image_tensors: List[torch.Tensor]) -> List[str]:
    model, inv_label_map = load_model()
    batch = torch.stack(image_tensors).to(DEVICE)
    with torch.no_grad():
        preds = model(batch).argmax(dim=1).tolist()
    return [inv_label_map[p] for p in preds]


def predict_image(image_path: str) -> str:
    return predict_batch([image_to_tensor(image_path)])[0]


def predict_images(image_paths: List[str]) -> List[str]:
    return predict_batch([image_to_tensor(p) for p in image_paths])


def path_to_image(img_path: Union[str, io.BytesIO]) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def image_to_tensor(img_path: Union[str, io.BytesIO]) -> torch.Tensor:
    return transform(path_to_image(img_path))


def base64_to_tensor(img_b64: str) -> torch.Tensor:
    return image_to_tensor(io.BytesIO(base64.b64decode(img_b64)))


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test(csv_path: str, image_dir: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    total = len(df)
    correct = 0

    print("\nRunning test predictions...\n")
    for _, row in df.iterrows():
        img_id = row['ImgId']
        true_label = row['categories']
        img_path = os.path.join(image_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            print(f"Skipping not existing image in path image: {img_id}")
            total -= 1
            continue

        predicted_label = predict_image(img_path)
        if predicted_label == true_label:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"[{status}] {img_id} | True: {true_label} | Predicted: {predicted_label}")

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")


if __name__ == "__main__":
    test(
        csv_path="retail-products-classification/train.csv",
        image_dir="retail-products-classification/train"
    )

    path = "retail-products-classification/test"
    print("Single prediction:", predict_image(
        f"{path}/097585562X.jpg"
    ))
    print("Batch prediction:", predict_images(
        [
            f"{path}/097585562X.jpg",
            f"{path}/097924837X.jpg"
        ]
    ))
