import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Edit these paths if your files are elsewhere
MODEL_PATHS = {
    "resnet50": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\resnet50_best.h5",
    "mobilenetv2": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\mobilenetv2_best.h5",
    "custom_cnn": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\custom_cnn_best.h5",
}

# Replace with your actual class names in order
CLASS_NAMES = ["class_0", "class_1", "class_2"]
IMG_SIZE = (224, 224)

import re
import warnings


@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except ValueError as e:
        msg = str(e)
        names = re.findall(r"Unknown layer: '([^']+)'", msg)
        if not names:
            raise
        warnings.warn(f"Model contains unknown layers {names}; attempting fallback by registering placeholder Lambda layers. Predictions may be incorrect.")
        custom = {n: tf.keras.layers.Lambda(lambda x: x) for n in names}
        with tf.keras.utils.custom_object_scope(custom):
            return tf.keras.models.load_model(path, compile=False)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

st.title("Model Demo — Streamlit")
model_key = st.selectbox("Choose model", list(MODEL_PATHS.keys()))
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)
    if st.button("Predict"):
        try:
            model = load_model(MODEL_PATHS[model_key])
        except Exception as e:
            st.error(f"Failed to load model: {e}")
        else:
            x = preprocess(img)
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
            st.success(f"Predicted: {CLASS_NAMES[idx]} ({preds[idx]:.4f})")
            st.write({CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))})
