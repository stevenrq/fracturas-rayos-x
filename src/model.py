"""Construcción de modelos: transfer learning (MobileNetV2, EfficientNetB0) y
una CNN propia desde cero, más la compilación con métricas clínicas.

El preprocesamiento va **dentro** del modelo, así el despliegue recibe imágenes
crudas ``[0,255]`` y no hay que recordar normalizar igual que en entrenamiento.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnet_preprocess

try:
    import config
    import data
except ModuleNotFoundError:  # importado como paquete
    from src import config
    from src import data


def _classifier_head(feature_map, dropout):
    x = keras.layers.GlobalAveragePooling2D()(feature_map)
    x = keras.layers.Dropout(dropout)(x)
    return keras.layers.Dense(1, activation="sigmoid", name="pred")(x)


def build_mobilenet(img_size=config.IMG_SIZE, dropout=0.3, base_trainable=False):
    """MobileNetV2 (ligero, ~9 MB de pesos) — candidato a desplegar."""
    inputs = keras.Input(shape=img_size + (3,))
    x = data.get_augmentation()(inputs)
    x = mnet_preprocess(x)                       # [0,255] -> [-1,1]
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=img_size + (3,))
    base.trainable = base_trainable
    x = base(x, training=False)
    outputs = _classifier_head(x, dropout)
    return keras.Model(inputs, outputs, name="mobilenetv2"), base


def build_efficientnet(img_size=config.IMG_SIZE, dropout=0.3, base_trainable=False):
    """EfficientNetB0 — comparación (suele dar algo más de exactitud).

    OJO: EfficientNet normaliza internamente, así que se alimenta ``[0,255]``
    directamente, SIN ``preprocess_input``.
    """
    inputs = keras.Input(shape=img_size + (3,))
    x = data.get_augmentation()(inputs)
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=img_size + (3,))
    base.trainable = base_trainable
    x = base(x, training=False)
    outputs = _classifier_head(x, dropout)
    return keras.Model(inputs, outputs, name="efficientnetb0"), base


def build_small_cnn(img_size=config.IMG_SIZE, dropout=0.3, base_trainable=False):
    """CNN pequeña desde cero — 3er baseline para medir el aporte del transfer learning.

    Devuelve ``base = None`` porque no hay base preentrenada que descongelar.
    """
    inputs = keras.Input(shape=img_size + (3,))
    x = data.get_augmentation()(inputs)
    x = keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="pred")(x)
    return keras.Model(inputs, outputs, name="small_cnn"), None


def compile_model(model, lr=1e-3):
    """Compila con pérdida binaria y métricas clínicas (incluye Recall y AUC)."""
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),   # sensibilidad
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# Fábrica usada por train.py / notebooks.
BUILDERS = {
    "mobilenet": build_mobilenet,
    "efficientnet": build_efficientnet,
    "cnn": build_small_cnn,
}
