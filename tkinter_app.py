import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import threading

MODEL_PATHS = {
    "resnet50": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\resnet50_best.h5",
    "mobilenetv2": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\mobilenetv2_best.h5",
    "custom_cnn": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\custom_cnn_best.h5",
}

CLASS_NAMES = ["class_0", "class_1", "class_2"]
IMG_SIZE = (224, 224)

models_cache = {}

import re
import warnings


def load_model(name):
    if name not in models_cache:
        path = MODEL_PATHS[name]
        try:
            models_cache[name] = tf.keras.models.load_model(path, compile=False)
        except ValueError as e:
            msg = str(e)
            names = re.findall(r"Unknown layer: '([^']+)'", msg)
            if not names:
                raise
            warnings.warn(f"Model contains unknown layers {names}; attempting fallback by registering placeholder Lambda layers. Predictions may be incorrect.")
            custom = {n: tf.keras.layers.Lambda(lambda x: x) for n in names}
            with tf.keras.utils.custom_object_scope(custom):
                models_cache[name] = tf.keras.models.load_model(path, compile=False)
    return models_cache[name]

def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    a = np.array(img).astype("float32")/255.0
    return np.expand_dims(a,0)

def predict_thread(img_path, model_key, result_label, img_label):
    img = Image.open(img_path)
    img_tk = ImageTk.PhotoImage(img.resize((300,300)))
    img_label.configure(image=img_tk)
    img_label.image = img_tk
    model = load_model(model_key)
    x = preprocess(img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    result_label.config(text=f"Predicted: {CLASS_NAMES[idx]} ({preds[idx]:.3f})")

def browse_and_predict(model_var, result_label, img_label):
    path = filedialog.askopenfilename(filetypes=[("Images","*.jpg;*.jpeg;*.png")])
    if not path:
        return
    threading.Thread(target=predict_thread, args=(path, model_var.get(), result_label, img_label), daemon=True).start()

root = tk.Tk()
root.title("Model Demo - Tkinter")
frm = ttk.Frame(root, padding=10)
frm.grid()
model_var = tk.StringVar(value=list(MODEL_PATHS.keys())[0])
ttk.Label(frm, text="Model:").grid(column=0,row=0,sticky="w")
ttk.Combobox(frm, textvariable=model_var, values=list(MODEL_PATHS.keys())).grid(column=1,row=0)
btn = ttk.Button(frm, text="Load & Predict", command=lambda: browse_and_predict(model_var, result_lbl, image_label))
btn.grid(column=0,row=1,columnspan=2,pady=6)
result_lbl = ttk.Label(frm, text="No prediction yet")
result_lbl.grid(column=0,row=2,columnspan=2)
image_label = ttk.Label(frm)
image_label.grid(column=0,row=3,columnspan=2)
root.mainloop()
