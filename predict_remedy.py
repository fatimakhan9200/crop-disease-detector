# predict_remedy.py

import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os

# -----------------------------
# 1️⃣ Load model
# -----------------------------
model_path = os.path.join("model", "crop_disease_model1.h5")

model = tf.keras.models.load_model(model_path)

# -----------------------------
# 2️⃣ Load class names
# -----------------------------
class_names_path = os.path.join("model", "class_names.json")
with open(class_names_path, "r") as f:
    class_names = json.load(f)

# -----------------------------
# 3️⃣ Load remedies
# -----------------------------
remedies_path = "remedies.json"
with open(remedies_path, "r") as f:
    remedies_dict = json.load(f)

# -----------------------------
# 4️⃣ Prediction function (TOP 3)
# -----------------------------
def predict_and_remedy(img_path):

    # Read image (NO cv2 → Streamlit safe fix)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]

    # Top 3 results
    top_indices = prediction.argsort()[-3:][::-1]

    results = []

    for i in top_indices:
        disease = class_names[i]
        confidence = float(prediction[i]) * 100
        remedy = remedies_dict.get(disease, "No remedy found")

        results.append((disease, round(confidence, 2), remedy))

    return results
