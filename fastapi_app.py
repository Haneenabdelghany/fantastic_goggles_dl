from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import io

MODEL_PATHS = {
    "resnet50": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\resnet50_best.h5",
    "mobilenetv2": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\mobilenetv2_best.h5",
    "custom_cnn": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\custom_cnn_best.h5",
}

CLASS_NAMES = ["class_0", "class_1", "class_2"]
IMG_SIZE = (224,224)

app = FastAPI()
models_cache = {}

import re
import warnings


def get_model(key):
    if key not in models_cache:
        path = MODEL_PATHS[key]
        try:
            models_cache[key] = tf.keras.models.load_model(path, compile=False)
        except ValueError as e:
            msg = str(e)
            names = re.findall(r"Unknown layer: '([^']+)'", msg)
            if not names:
                raise
            warnings.warn(f"Model contains unknown layers {names}; attempting fallback by registering placeholder Lambda layers. Predictions may be incorrect.")
            custom = {n: tf.keras.layers.Lambda(lambda x: x) for n in names}
            with tf.keras.utils.custom_object_scope(custom):
                models_cache[key] = tf.keras.models.load_model(path, compile=False)
    return models_cache[key]

def preprocess_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")/255.0
    return np.expand_dims(arr,0)

@app.post("/predict")
async def predict(model: str = Form(...), file: UploadFile = File(...)):
    if model not in MODEL_PATHS:
        return JSONResponse(status_code=400, content={"error":"unknown model"})
    content = await file.read()
    x = preprocess_image_bytes(content)
    m = get_model(model)
    preds = m.predict(x)[0].tolist()
    top_idx = int(np.argmax(preds))
    return {"predicted": CLASS_NAMES[top_idx], "prob": preds[top_idx], "probs": {CLASS_NAMES[i]: preds[i] for i in range(len(preds))}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
