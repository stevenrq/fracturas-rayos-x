"""App Streamlit (local): misma funcionalidad que la de Gradio.

Ejecutar:
    streamlit run app/app_streamlit.py

Variables de entorno opcionales: MODEL_PATH, UMBRAL (ver app_gradio.py).
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import gradcam  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO / "models" / "best_mobilenet.keras"))
MODEL_REPO = os.environ.get("MODEL_REPO", "stevenrq8/fracturas-modelo")
MODEL_FILE = os.environ.get("MODEL_FILE", "best_mobilenet.keras")
IMG_SIZE = (224, 224)
UMBRAL = float(os.environ.get("UMBRAL", "0.310"))  # recall≥0.95 (ver resultados.md)

st.set_page_config(page_title="Detección de fracturas óseas", page_icon="🦴")


def discover_model_path():
    candidates = [Path(MODEL_PATH)] if MODEL_PATH else []
    candidates.extend(
        [
            REPO / "models" / MODEL_FILE,
            REPO / "models" / "best_mobilenet.keras",
            REPO / "models" / "best_mobilenet.h5",
        ]
    )
    candidates.extend(sorted((REPO / "models").glob("*.keras")))
    candidates.extend(sorted((REPO / "models").glob("*.h5")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def cargar_modelo():
    path = discover_model_path()
    if path is not None:
        try:
            return tf.keras.models.load_model(str(path)), None
        except Exception as exc:  # noqa: BLE001
            return None, f"No se pudo cargar el modelo en {path}: {exc}"

    try:
        from huggingface_hub import hf_hub_download

        remote_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        return tf.keras.models.load_model(remote_path), None
    except Exception as exc:  # noqa: BLE001
        return None, (
            "No se encontró un modelo entrenado en 'models/'. "
            f"Define MODEL_PATH o descarga el archivo '.keras' ({exc})."
        )


@st.cache_resource
def cargar_modelo_cacheado():
    return cargar_modelo()


model, model_error = cargar_modelo_cacheado()

st.title("🦴 Detección de fracturas óseas")
st.warning(
    "⚠️ Demostración académica. No apta para uso clínico real. Las predicciones "
    "son orientativas y no sustituyen el diagnóstico de un profesional de la salud."
)

if model is None:
    st.warning(model_error or "No se pudo cargar el modelo.")
    st.info(
        "Para habilitar predicciones, coloca el modelo en models/ o define MODEL_PATH."
    )
    st.stop()

archivo = st.file_uploader("Sube una radiografía", type=["jpg", "jpeg", "png"])
if archivo:
    pil = Image.open(archivo).convert("RGB")
    arr = np.asarray(pil.resize(IMG_SIZE), dtype="float32")[None, ...]
    p = float(model.predict(arr, verbose=0).ravel()[0])
    etiqueta = "FRACTURA" if p >= UMBRAL else "NORMAL"

    col1, col2 = st.columns(2)
    col1.image(pil, caption="Radiografía cargada", use_container_width=True)
    try:
        cam = gradcam.gradcam_on_image(pil, model, IMG_SIZE)
        col2.image(cam, caption="Grad-CAM (zona observada)", use_container_width=True)
    except Exception as e:  # noqa: BLE001
        col2.info(f"Grad-CAM no disponible: {e}")

    st.metric("Predicción", etiqueta, f"{max(p, 1 - p):.1%} de confianza")
    st.progress(
        min(max(p, 0.0), 1.0), text=f"P(fractura) = {p:.1%}  (umbral {UMBRAL:.2f})"
    )
