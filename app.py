from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import io
import os

import os
os.makedirs("models", exist_ok=True)


app = Flask(__name__)

# Path folder SavedModel (BUKAN file)
# model = tf.keras.models.load_model("saved_model_folder")
# model.save("mobilenet_fixed.h5")

# model = tf.keras.models.load_model("models/mobilenetv2_final.h5", compile=False)
model = tf.keras.models.load_model("models/mobilenetv2_safe_(1).keras", compile=False)

LABELS = [
    "normal_skin",
    "tinea_nigra",
    "tinea_ringworm",
    "tinea_versicolor"
]

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
        start = time.time()


        img_processed = preprocess_image(image)

        preds = model.predict(img_processed)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        inference_time = (time.time() - start) * 1000   # ms

        top3_idx = preds[0].argsort()[-3:][::-1]
        top3 = [
            {
                "label": LABELS[i],
                "score": float(preds[0][i])
            }
            for i in top3_idx
        ]

        return jsonify({
            "class_id": pred_class,
            "confidence": confidence,
            "top3": top3,
            "inference_time_ms": round(inference_time, 2)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
