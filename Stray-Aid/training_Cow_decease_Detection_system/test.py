import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_cow_model.keras"   # use best model
IMG_SIZE = (224, 224)

CLASSES = ["foot_and_mouth", "healthy", "lumpy"]

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD IMAGE
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess (IMPORTANT)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    confidence = np.max(preds)

    return CLASSES[pred_index], confidence

# =========================
# TEST IMAGE
# =========================
if __name__ == "__main__":
    img_path = r"C:\Users\Shailendra\Downloads\Final_Proj\Stray-Aid\training_Cow_decease_Detection_system\test\download.jpg"

    label, conf = predict_image(img_path)

    print("\n🔍 Prediction:")
    print(f"🐄 Disease: {label}")
    print(f"📊 Confidence: {conf*100:.2f}%")