from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

MODEL_PATH = "models/model.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = image.convert("RGB")
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada file yang dikirm'}), 400
    
    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read()))
        img_processed = preprocess_image(image)

        preds = model.predict(img_processed)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return jsonify({
            "class_id": pred_class,
            "confidence": confidence,
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


app.run(debug=True)