"""Grad-CAM: mapas de calor que muestran qué región de la radiografía miró el
modelo para decidir. Aumenta la credibilidad de la demo (extensión opcional).

Funciona con los modelos de ``model.py`` (MobileNetV2, EfficientNetB0, CNN), ya
que el preprocesamiento va dentro del modelo y las capas conv quedan accesibles
en el grafo de nivel superior.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import config
except ModuleNotFoundError:  # importado como paquete
    from src import config


def _build_cam_models(outer_model):
    """Devuelve (feat_model, head_model) re-ejecutando capas desde outer_model.inputs[0].

    feat_model : input -> mapa de características 4D (salida del sub-modelo base)
    head_model : mapa de características -> predicción escalar

    Se re-ejecuta cada capa sobre tensores simbólicos frescos para garantizar que
    los tensores de salida estén conectados a outer_model.inputs[0] y no al grafo
    interno del sub-modelo (p.ej. MobileNetV2 tiene su propio InputLayer).
    """
    x = outer_model.inputs[0]
    feat_tensor = None

    for layer in outer_model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x)
        # El sub-modelo base es un Functional Model (no Sequential)
        if (isinstance(layer, keras.Model)
                and not isinstance(layer, keras.Sequential)
                and feat_tensor is None):
            feat_tensor = x

    if feat_tensor is None:
        raise ValueError("No se encontró el sub-modelo base (MobileNetV2/EfficientNet).")

    feat_model = keras.models.Model(outer_model.inputs[0], feat_tensor)

    # Modelo de la cabeza: Input fresco con la forma del mapa de características
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


def find_last_conv_layer(model) -> str:
    """Nombre del sub-modelo base (mapa de características para Grad-CAM)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential):
            return layer.name
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except (AttributeError, ValueError):
            continue
    raise ValueError("No se encontró una capa convolucional para Grad-CAM.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Heatmap 2D normalizado [0,1].

    ``img_array``: tensor (1, H, W, 3) float32 en [0,255] (crudo, como espera el modelo).

    Estrategia de gradiente en dos etapas:
      1. Extraer features fuera del tape (sin tracking).
      2. Llamar a la cabeza dentro del tape con tape.watch(features) para
         obtener d(clase)/d(features) sin depender de variables intermedias.
    """
    feat_model, head_model = _build_cam_models(model)

    features = feat_model(img_array, training=False)

    with tf.GradientTape() as tape:
        tape.watch(features)
        preds = head_model(features, training=False)
        if pred_index is None:
            pred_index = 0                       # salida sigmoide única -> P(fractura)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, features)
    if grads is None:
        raise ValueError("Gradiente None: el mapa de características no está en el tape.")
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = features[0]
    heatmap = tf.squeeze(conv_out @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    """Superpone el heatmap coloreado (JET) sobre la imagen PIL. Devuelve PIL RGB."""
    import cv2
    from PIL import Image

    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    hm = cv2.resize((heatmap * 255).astype("uint8"), (w, h))
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (hm_color * alpha + img * (1 - alpha)).astype("uint8")
    return Image.fromarray(overlay)


def gradcam_on_image(pil_img, model, img_size=config.IMG_SIZE,
                     last_conv_layer_name=None, alpha=0.4):
    """Pipeline completo: imagen PIL -> imagen PIL con el Grad-CAM superpuesto."""
    arr = np.asarray(pil_img.convert("RGB").resize(img_size), dtype="float32")[None, ...]
    heatmap = make_gradcam_heatmap(arr, model, last_conv_layer_name)
    return overlay_heatmap(pil_img, heatmap, alpha)
