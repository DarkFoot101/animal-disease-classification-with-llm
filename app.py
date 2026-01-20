from flask import Flask, jsonify, request, render_template
import os
from flask_cors import CORS
from src.classifier.pipeline.predict import PredictionPipeline
from classifier.utils.common import decodeImage as decode_image
from pathlib import Path

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self, app):
        self.filename = "0001.jpg"
        self.classifier = PredictionPipeline(
            model_path=os.path.join("artifacts", "training", "trained_model.h5")
        )


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    os.system("python main.py")
    return "Training successful!"


@app.route("/predict", methods=["POST"])
def predictRoute():
    """
    Supports:
    multipart/form-data (file upload)  -> request.files["file"]
    application/json (base64 image)    -> request.json["image"]
    """

    try:
        #  CASE 1: New UI (multipart file upload)
        if "file" in request.files:
            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            # Save uploaded file
            file.save(clApp.filename)

            result = clApp.classifier.predict(img_path=clApp.filename)
            return jsonify(result)

        # CASE 2: Old UI (base64 json)
        if request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "Missing 'image' key in JSON"}), 400

            image = data["image"]
            decode_image(imgstring=image, fileName=clApp.filename)

            result = clApp.classifier.predict(img_path=clApp.filename)
            return jsonify(result)

        return jsonify({"error": "Unsupported request format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    clApp = ClientApp(app)

    # Run on Colab / Docker / cloud
    app.run(host="0.0.0.0", port=8080, debug=True)
