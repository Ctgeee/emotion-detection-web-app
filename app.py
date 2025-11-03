# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import sqlite3
from datetime import datetime
import tensorflow as tf
import cv2

# Config
MODEL_PATH = "emotion_detector_v1.h5"
UPLOAD_DIR = "static/uploads"
DB_PATH = "usage.db"
IMG_SIZE = (128, 128)
EMOTION_LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']  # match training labels order

os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def prepare_image(image: Image.Image):
    # convert to RGB and resize
    img = image.convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_emotion(model, image: Image.Image):
    x = prepare_image(image)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx]), probs

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # columns: id, name (optional), timestamp, image_path, predicted_label, confidence
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image_path TEXT,
            predicted_label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_usage(name, image_path, predicted_label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO usage (name, timestamp, image_path, predicted_label, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, datetime.utcnow().isoformat(), image_path, predicted_label, confidence))
    conn.commit()
    conn.close()

def save_image(image: Image.Image, prefix="img"):
    # generate unique name
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    image.save(path)
    return path

def main():
    st.title("Emotion Detector Web App")
    st.markdown("Upload an image or use webcam. The app will predict the emotion and log usage.")

    init_db()
    model = load_model()

    # Optional: ask for name (offline user)
    name = st.text_input("Your name (optional) — will be saved for usage logging", "")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', use_column_width=True)
            if st.button("Predict (Upload)"):
                img_path = save_image(image, prefix="uploaded")
                label, conf, probs = predict_emotion(model, image)
                st.success(f"Predicted: **{label}** (confidence: {conf:.2f})")
                st.write("Class probabilities:")
                for lbl, p in zip(EMOTION_LABELS, probs):
                    st.write(f"{lbl}: {p:.3f}")
                log_usage(name or None, img_path, label, conf)

    with col2:
        st.header("Webcam capture")
        # Use st.camera_input (works in Streamlit)
        cam_image = st.camera_input("Take a photo with your webcam")
        if cam_image is not None:
            image = Image.open(cam_image)
            st.image(image, caption='Webcam capture', use_column_width=True)
            if st.button("Predict (Webcam)"):
                img_path = save_image(image, prefix="webcam")
                label, conf, probs = predict_emotion(model, image)
                st.success(f"Predicted: **{label}** (confidence: {conf:.2f})")
                st.write("Class probabilities:")
                for lbl, p in zip(EMOTION_LABELS, probs):
                    st.write(f"{lbl}: {p:.3f}")
                log_usage(name or None, img_path, label, conf)

    st.markdown("---")
    if st.checkbox("Show usage logs (last 20)"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, timestamp, image_path, predicted_label, confidence FROM usage ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        st.write(rows)

if __name__ == "__main__":
    main()