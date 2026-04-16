from flask import Flask, request, jsonify, render_template
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────
ANIMAL_MODEL_PATH = "models/animal_model.pth"
CLASS_NAMES = ["cat", "cow", "dog"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease class labels — alphabetical order (matches Keras flow_from_directory indexing)
DISEASE_CLASSES = {
    "cat": [
        "Dental Disease",
        "Ear Mites",
        "Eye Infection",
        "Feline Leukemia",
        "Feline Panleukopenia",
        "Fungal Infection",
        "Healthy",
        "Ringworm",
        "Scabies",
        "Skin Allergy",
        "Urinary Tract Infection",
        "Worm Infection",
    ],
    "cow": [
        "Foot and Mouth Disease",
        "Healthy",
        "Lumpy Skin Disease",
    ],
    "dog": [
        "Demodicosis",
        "Dermatitis",
        "Fungal Infection",
        "Healthy",
        "Hypersensitivity",
        "Ringworm",
    ],
}

DISEASE_MODEL_PATHS = {
    "cat": "models/best_cat_model.keras",
    "cow": "models/best_cow_model.keras",
    "dog": "models/best_model_Dog.keras",
}

# ── Lazy-load animal classifier ────────────────────────────────
_animal_model = None

def get_animal_model():
    global _animal_model
    if _animal_model is not None:
        return _animal_model
    if not os.path.exists(ANIMAL_MODEL_PATH):
        return None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(ANIMAL_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _animal_model = model
    print("[OK] Loaded animal classifier")
    return _animal_model

# ── Lazy-load disease models ───────────────────────────────────
_disease_models = {}

def get_disease_model(animal):
    if animal in _disease_models:
        return _disease_models[animal]
    path = DISEASE_MODEL_PATHS.get(animal)
    if not path or not os.path.exists(path):
        return None
    try:
        m = tf.keras.models.load_model(path)
        _disease_models[animal] = m
        print(f"[OK] Loaded disease model for {animal}")
        return m
    except Exception as e:
        print(f"[WARN] Could not load disease model for {animal}: {e}")
        return None

# ── Image transforms ───────────────────────────────────────────
animal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_for_disease(image: Image.Image, size=(224, 224)):
    img = image.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    animal_model = get_animal_model()
    if animal_model is None:
        return jsonify({
            "error": f"Animal model not found. Place 'animal_model.pth' in '{os.path.abspath('models')}'"
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── Step 1: Animal classification ──────────────────────
        tensor = animal_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = animal_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probs, 0)

        animal = CLASS_NAMES[top_idx.item()]
        animal_confidence = round(top_prob.item() * 100, 2)
        animal_breakdown = {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # ── Step 2: Disease detection ───────────────────────────
        disease_result = None
        d_model = get_disease_model(animal)
        if d_model is not None:
            d_labels = DISEASE_CLASSES.get(animal, [])
            inp = preprocess_for_disease(image)
            d_probs = d_model.predict(inp)[0]

            top_d_idx = int(np.argmax(d_probs))
            disease_name = d_labels[top_d_idx] if top_d_idx < len(d_labels) else f"Class {top_d_idx}"
            disease_confidence = round(float(d_probs[top_d_idx]) * 100, 2)
            disease_breakdown = {
                (d_labels[i] if i < len(d_labels) else f"Class {i}"): round(float(d_probs[i]) * 100, 2)
                for i in range(len(d_probs))
            }

            disease_result = {
                "disease": disease_name,
                "confidence": disease_confidence,
                "breakdown": disease_breakdown,
                "is_healthy": disease_name.lower() == "healthy",
            }

        # ── Encode image for response ───────────────────────────
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        mime = file.content_type or "image/jpeg"

        return jsonify({
            "animal": animal,
            "animal_confidence": animal_confidence,
            "animal_breakdown": animal_breakdown,
            "disease": disease_result,
            "image": f"data:{mime};base64,{encoded}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
