"""Hugging Face Space (Gradio) -- Deteccion de fracturas oseas (demo academica).

Standalone: carga el modelo desde un Model repo de Hugging Face (o un archivo
local junto a app.py si usas Git LFS), expone una interfaz para subir radiografias
y muestra P(fractura), etiqueta, confianza, un mapa de calor Grad-CAM y un
comentario textual generado por LLM via HF Inference API.

Configurable por variables de entorno (pestana Settings -> Variables del Space):
    MODEL_REPO   repo del modelo en HF      (def. "stevenrq8/fracturas-modelo")
    MODEL_FILE   nombre del archivo .keras  (def. "best_mobilenet.keras")
    UMBRAL       umbral P(fractura)         (def. "0.31"; usa el de resultados.md)
    HF_TOKEN     token de HF (secret recomendado)
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import gradio as gr

IMG_SIZE = (224, 224)
UMBRAL = float(os.environ.get("UMBRAL", "0.310"))
MODEL_REPO = os.environ.get("MODEL_REPO", "stevenrq8/fracturas-modelo")
MODEL_FILE = os.environ.get("MODEL_FILE", "best_mobilenet.keras")

DISCLAIMER = (
    "Demostracion academica. Las predicciones son orientativas, pueden "
    "contener errores y NO sustituyen el diagnostico de un profesional de la "
    "salud. No la utilice para decisiones clinicas reales."
)


def _load_model():
    if os.path.exists(MODEL_FILE):
        return keras.models.load_model(MODEL_FILE)
    from huggingface_hub import hf_hub_download
    return keras.models.load_model(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


model = _load_model()


def _build_cam_models(outer_model):
    x = outer_model.inputs[0]
    feat_tensor = None
    for layer in outer_model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x)
        if (isinstance(layer, keras.Model)
                and not isinstance(layer, keras.Sequential)
                and feat_tensor is None):
            feat_tensor = x
    if feat_tensor is None:
        raise ValueError("No se encontro el sub-modelo base para Grad-CAM.")
    feat_model = keras.models.Model(outer_model.inputs[0], feat_tensor)

    head_input = keras.Input(shape=tuple(feat_tensor.shape[1:]))
    hx = head_input
    after_base = False
    for layer in outer_model.layers:
        if (isinstance(layer, keras.Model)
                and not isinstance(layer, keras.Sequential)):
            after_base = True
            continue
        if after_base and not isinstance(layer, keras.layers.InputLayer):
            hx = layer(hx)
    head_model = keras.models.Model(head_input, hx)
    return feat_model, head_model


def _gradcam(pil):
    import cv2
    arr = np.asarray(pil.convert("RGB").resize(IMG_SIZE), dtype="float32")[None, ...]
    feat_model, head_model = _build_cam_models(model)
    features = feat_model(arr, training=False)
    with tf.GradientTape() as tape:
        tape.watch(features)
        preds = head_model(features, training=False)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, features)
    if grads is None:
        raise ValueError("Gradiente None.")
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(features[0] @ pooled[..., tf.newaxis])
    heatmap = (tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    img = np.array(pil.convert("RGB"))
    h, w = img.shape[:2]
    hm = cv2.resize((heatmap * 255).astype("uint8"), (w, h))
    color = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray((color * 0.4 + img * 0.6).astype("uint8"))


def _resumen_medgemma(pil, p_fractura):
    """Genera un comentario orientativo usando LLM via HF Inference API.

    Requiere la variable de entorno / secret HF_TOKEN.
    """
    from huggingface_hub import InferenceClient

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "HF_TOKEN no configurado - comentario no disponible."

    etiqueta = "FRACTURA" if p_fractura >= UMBRAL else "NORMAL"
    texto = (
        "Eres un asistente educativo de radiologia. "
        "Un clasificador CNN analizo una radiografia osea y obtuvo: "
        "Prediccion={}, P(fractura)={:.1%}. "
        "Describe brevemente en 3-4 frases que podria significar este resultado, "
        "que zonas deberia revisar un medico y recuerda que es solo orientativo. "
        "Responde en espanol."
    ).format(etiqueta, p_fractura)

    try:
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": texto}],
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Comentario no disponible: " + str(e)


def predecir(img):
    if img is None:
        return None, "Sube una radiografia.", None, ""
    pil = img.convert("RGB")
    arr = np.asarray(pil.resize(IMG_SIZE), dtype="float32")[None, ...]
    p = float(model.predict(arr, verbose=0).ravel()[0])
    etiqueta = "FRACTURA" if p >= UMBRAL else "NORMAL"
    try:
        cam = _gradcam(pil)
    except Exception as e:
        print("Grad-CAM no disponible:", e)
        cam = None
    resumen = (
        "Prediccion: {}  P(fractura) = {:.1%}  Confianza = {:.1%}  (umbral {:.2f})".format(
            etiqueta, p, max(p, 1 - p), UMBRAL
        )
    )
    comentario = _resumen_medgemma(pil, p)
    return {"fractura": p, "normal": 1 - p}, resumen, cam, comentario


EXAMPLES = (
    [["ejemplos/normal.jpg"], ["ejemplos/fractura.jpg"]]
    if os.path.exists("ejemplos/normal.jpg") else None
)

LLM_NOTE = (
    "\n\n---\nComentario LLM - generado por Meta-Llama-3-8B. Es ilustrativo, "
    "puede contener errores y no constituye un diagnostico clinico."
)

demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="pil", label="Radiografia"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidades"),
        gr.Textbox(label="Resultado"),
        gr.Image(type="pil", label="Grad-CAM (zona observada por el modelo)"),
        gr.Textbox(label="Comentario LLM (ilustrativo, no diagnostico)"),
    ],
    title="Deteccion de fracturas oseas (demo academica)",
    description="Sube una radiografia y el modelo estimara la probabilidad de fractura.\n\n" + DISCLAIMER,
    article=DISCLAIMER + LLM_NOTE,
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()