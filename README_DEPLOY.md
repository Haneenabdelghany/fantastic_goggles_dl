# Deploy scripts

Files added to this folder:

- `streamlit_app.py` — Streamlit UI to upload an image and predict using one of the provided .h5 models.
- `tkinter_app.py` — Simple desktop GUI for local prediction.
- `fastapi_app.py` — FastAPI server exposing a `/predict` endpoint.
- `requirements.txt` — Minimal Python packages to install.

Notes
- Edit the `MODEL_PATHS` dict at the top of each script if your model locations differ.
- Replace `CLASS_NAMES` in each script with your actual class labels in the same order used to train the models.

Run examples

Streamlit:
```bash
python -m streamlit run PROJECT_FILES/notebooks/streamlit_app.py
```

FastAPI (dev):
```bash
uvicorn PROJECT_FILES.notebooks.fastapi_app:app --reload --port 8000
```

Tkinter: run the script directly with Python.
