import requests
from requests import Response

from predict import path_to_image, image_to_base64


def predict_image_batch(images: list[str], api_domain="127.0.0.1:5000") -> Response:
    return requests.post(
        url=f"http://{api_domain}/predict",
        json={
            "images": images
        }
    )

if __name__ == '__main__':
    path_to_test_image_dataset = 'retail-products-classification/train'
    test_image_dataset = {
        'B000GAWSBS': 'Clothing, Shoes & Jewelry',
        'B000YOUIN6': 'Baby Products',
        'B005ARCRXG': 'Appliances'
    }

    api_response_json = predict_image_batch(
        images=[image_to_base64(path_to_image(p)) for p in [
            f'{path_to_test_image_dataset}/{iid}.jpg' for iid in test_image_dataset.keys()
        ]]
    ).json()

    print(f"IMAGE_ID(Actual Category): Predicted category\n{'-'*50}\n" + "\n".join([
        f"{iid}({t}): {api_response_json['predicted_labels'][i]}" for i, (iid, t) in enumerate(test_image_dataset.items())
    ]))

