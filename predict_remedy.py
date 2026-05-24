# predict_remedy.py
import gdown
import numpy as np
import cv2
import tensorflow as tf
import json
import os

# -----------------------------
# 1️⃣ Load the trained model
# -----------------------------
model_path = os.path.join("model", "crop_disease_model1.h5")

# Download model if not present
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1cEE2wqwJiJkY_sIl_GOKyl9TqQk9Rp-Q"
    gdown.download(url, model_path, quiet=False)

# Load model
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
# 4️⃣ Function to predict TOP 3 diseases
# -----------------------------
def predict_and_remedy(img_path):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return [("Error reading image", 0, "No remedy")]

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]

    # Get TOP 3 predictions
    top_indices = prediction.argsort()[-3:][::-1]

    results = []

    for i in top_indices:
        disease = class_names[i]
        confidence = float(prediction[i]) * 100
        remedy = remedies_dict.get(disease, "No remedy found")

        results.append((disease, round(confidence, 2), remedy))

    return results