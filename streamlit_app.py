import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

MODEL_PATH = "model_jeruk_nipis.h5"
model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['matang', 'mentah', 'setengah_matang']

st.title("🍋 Klasifikasi Kematangan Jeruk Nipis")
st.write("Upload gambar agar diperiksa tingkat kematangannya.")

uploaded_file = st.file_uploader("Upload gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[class_index] * 100

    st.markdown(f"### ✅ Prediksi: `{class_names[class_index]}`")
    st.markdown(f"**Confidence: {confidence:.2f}%**")
