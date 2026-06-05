"""App Gradio (local): sube una radiografía -> P(fractura) + etiqueta + confianza
+ mapa de calor Grad-CAM. Versión para correr en tu máquina apuntando al modelo
en ``models/best_mobilenet.keras``. La versión del Space está en ``space/app.py``.

Ejecutar:
    python app/app_gradio.py        # abre http://127.0.0.1:7860

Variables de entorno opcionales:
    MODEL_PATH  ruta al modelo .keras (por defecto models/best_mobilenet.keras)
    UMBRAL      umbral de decisión P(fractura) (por defecto 0.31; usa el de resultados.md)
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import gradcam  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO / "models" / "best_mobilenet.keras"))
MODEL_REPO = os.environ.get("MODEL_REPO", "stevenrq8/fracturas-modelo")
MODEL_FILE = os.environ.get("MODEL_FILE", "best_mobilenet.keras")
IMG_SIZE = (224, 224)
UMBRAL = float(
    os.environ.get("UMBRAL", "0.310")
)  # recall≥0.95 (ver reports/resultados.md y notebooks/02_modelado.ipynb)


def discover_model_path():
    """Busca un modelo local en varios nombres/ubicaciones comunes."""
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


def load_model():
    """Carga el modelo si existe; si no, devuelve un mensaje claro."""
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


def model_status_message():
    return model_error or "Modelo listo para inferencia."


model, model_error = load_model()
MODEL_AVAILABLE = model is not None

DISCLAIMER = (
    "⚠️ **Demostración académica.** Las predicciones son orientativas, pueden "
    "contener errores y **NO sustituyen el diagnóstico de un profesional de la salud**. "
    "No la utilice para decisiones clínicas reales."
)
STATUS_MESSAGE = model_error if model_error else "Modelo listo para inferencia."

EJ = REPO / "app" / "ejemplos"
EXAMPLES = (
    [[str(EJ / "normal.jpg")], [str(EJ / "fractura.jpg")]]
    if (EJ / "normal.jpg").exists()
    else None
)


def predecir(img: Image.Image):
    if img is None:
        return None, "Sube una radiografía.", None
    if model is None:
        return ({"fractura": 0.5, "normal": 0.5}, STATUS_MESSAGE, None)

    pil = img.convert("RGB")
    arr = np.asarray(pil.resize(IMG_SIZE), dtype="float32")[
        None, ...
    ]  # [1,224,224,3] en [0,255]
    p = float(model.predict(arr, verbose=0).ravel()[0])  # P(fractura), fractura=1
    etiqueta = "FRACTURA" if p >= UMBRAL else "NORMAL"
    try:
        cam = gradcam.gradcam_on_image(pil, model, IMG_SIZE)
    except Exception as e:  # noqa: BLE001 - el Grad-CAM no debe tumbar la predicción
        print("Grad-CAM no disponible:", e)
        cam = None
    resumen = (
        f"Predicción: {etiqueta}  ·  P(fractura) = {p:.1%}  ·  "
        f"Confianza = {max(p, 1 - p):.1%}  (umbral {UMBRAL:.2f})"
    )
    return {"fractura": p, "normal": 1 - p}, resumen, cam


demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="pil", label="Radiografía"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidades"),
        gr.Textbox(label="Resultado"),
        gr.Image(type="pil", label="Grad-CAM (zona observada por el modelo)"),
    ],
    title="🦴 Detección de fracturas óseas (demo académica)",
    description=(
        "Sube una radiografía y el modelo estimará la probabilidad de fractura.\n\n"
        + DISCLAIMER
        + "\n\n"
        + STATUS_MESSAGE
    ),
    article=DISCLAIMER,
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
