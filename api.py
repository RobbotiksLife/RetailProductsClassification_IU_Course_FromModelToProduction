from flask import Flask, request, jsonify

from predict import base64_to_tensor, predict_batch


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "images" not in data:
        return jsonify({"error": "Missing 'images' key in JSON"}), 400

    try:
        image_list = [base64_to_tensor(img_b64) for img_b64 in data["images"]]
        labels = predict_batch(image_list)
        return jsonify({"predicted_labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
