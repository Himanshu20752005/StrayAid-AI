from flask import Flask, request, jsonify, render_template, g
import os
import sys

# Windows consoles often use cp1252; force UTF-8 so logs/TF output don't crash requests.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import tensorflow as tf
import numpy as np
import json
from datetime import datetime, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

def _load_dotenv(dotenv_path=".env"):
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv(
    "STRAYAID_SECRET",
    os.getenv("SECRET_KEY", "change-this-in-production"),
)

# ── Configuration ──────────────────────────────────────────────
ANIMAL_MODEL_PATH = "models/animal_model.pth"
CLASS_NAMES = ["cat", "cow", "dog"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USERS_DB_PATH = "users.json"
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24


def _ensure_users_file():
    if not os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def _load_users():
    _ensure_users_file()
    try:
        with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_users(users):
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _get_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="stray-aid-auth")


def _create_token(email):
    return _get_serializer().dumps({"email": email})


def _verify_token(token):
    try:
        payload = _get_serializer().loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
        return payload.get("email")
    except (BadSignature, SignatureExpired):
        return None


def _find_user_by_email(email):
    email_l = email.lower().strip()
    for user in _load_users():
        if user.get("email", "").lower() == email_l:
            return user
    return None


def token_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header with Bearer token is required"}), 401
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return jsonify({"error": "Missing token"}), 401

        email = _verify_token(token)
        if not email:
            return jsonify({"error": "Invalid or expired token"}), 401

        user = _find_user_by_email(email)
        if not user:
            return jsonify({"error": "User not found"}), 401
        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper

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

def _safe_log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


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
    _safe_log("[OK] Loaded animal classifier")
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
        _safe_log(f"[OK] Loaded disease model for {animal}")
        return m
    except Exception as e:
        _safe_log(f"[WARN] Could not load disease model for {animal}: {e}")
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
    return render_template("index.html", google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""))


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    if _find_user_by_email(email):
        return jsonify({"error": "email already registered"}), 409

    users = _load_users()
    now_iso = datetime.now(timezone.utc).isoformat()
    user = {
        "id": len(users) + 1,
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": now_iso,
    }
    users.append(user)
    _save_users(users)

    token = _create_token(email)
    return jsonify(
        {
            "message": "Registration successful",
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
        }
    ), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = _find_user_by_email(email)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "invalid email or password"}), 401

    token = _create_token(email)
    return jsonify(
        {
            "message": "Login successful",
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
        }
    )


@app.route("/auth/google", methods=["POST"])
def google_login():
    data = request.get_json(silent=True) or {}
    credential = data.get("credential")
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        return jsonify({"error": "GOOGLE_CLIENT_ID is not configured on server"}), 503
    if not credential:
        return jsonify({"error": "Google credential is required"}), 400

    try:
        info = id_token.verify_oauth2_token(credential, grequests.Request(), client_id)
    except ValueError:
        return jsonify({"error": "Invalid Google token"}), 401

    email = (info.get("email") or "").strip().lower()
    name = (info.get("name") or "").strip() or "Google User"
    if not email:
        return jsonify({"error": "Google account email not available"}), 400

    user = _find_user_by_email(email)
    if not user:
        users = _load_users()
        now_iso = datetime.now(timezone.utc).isoformat()
        user = {
            "id": len(users) + 1,
            "name": name,
            "email": email,
            "password_hash": "",
            "created_at": now_iso,
            "auth_provider": "google",
        }
        users.append(user)
        _save_users(users)

    token = _create_token(email)
    return jsonify(
        {
            "message": "Google login successful",
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
        }
    )


@app.route("/auth/me", methods=["GET"])
@token_required
def me():
    user = g.current_user
    return jsonify({"user": {"id": user["id"], "name": user["name"], "email": user["email"]}})


@app.route("/predict", methods=["POST"])
@token_required
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
            d_probs = d_model.predict(inp, verbose=0)[0]

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

    except UnicodeEncodeError:
        return jsonify({
            "error": "Encoding error on server (Windows console). Restart the app after updating, or set PYTHONIOENCODING=utf-8."
        }), 500
    except Exception as e:
        err = str(e)
        try:
            err.encode("ascii")
        except UnicodeEncodeError:
            err = err.encode("ascii", errors="replace").decode("ascii")
        return jsonify({"error": err}), 500


if __name__ == "__main__":
    app.run(debug=True)
