"""Carga de datos y pipeline ``tf.data`` para la clasificación binaria.

Lee ``data/processed/{train,val,test}/{0_normal,1_fracture}`` y devuelve datasets
listos para entrenar, además de la capa de *data augmentation* (conservadora,
pensada para radiografías) y los pesos de clase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

IMG_EXTS = {".jpg", ".jpeg", ".png"}

try:
    import config
except ModuleNotFoundError:  # importado como paquete: from src import data
    from src import config


def make_datasets(data_dir=None, img_size=config.IMG_SIZE,
                  batch_size=config.BATCH_SIZE, seed=config.SEED):
    """Devuelve (train_ds, val_ds, test_ds, class_names) con cache + prefetch.

    - ``color_mode="rgb"`` unifica las imágenes en escala de grises / RGBA a 3 canales.
    - ``shuffle=False`` en val/test para poder alinear ``y_true`` con ``predict``.
    """
    data_dir = Path(data_dir or config.DATA_DIR)

    def load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            data_dir / split,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="binary",
            color_mode="rgb",
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = load("train", shuffle=True)
    val_ds = load("val", shuffle=False)
    test_ds = load("test", shuffle=False)

    class_names = train_ds.class_names
    assert class_names == config.CLASS_NAMES, (
        f"Orden de clases inesperado: {class_names}. "
        f"Se esperaba {config.CLASS_NAMES} (fractura debe ser la clase 1)."
    )

    autotune = tf.data.AUTOTUNE

    # Pipeline simple: sin ignore_errors porque los archivos corruptos se
    # eliminan previamente con remove_corrupt_images() antes de entrenar.
    train_ds = (train_ds
                .cache()
                .shuffle(buffer_size=2738, seed=seed)
                .prefetch(autotune))
    val_ds  = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def remove_corrupt_images(data_dir=None, dry_run=False):
    """Elimina imágenes que TF no puede decodificar (libjpeg más estricto que PIL).

    Llama esto UNA VEZ antes de entrenar. Los archivos corruptos se borran del
    filesystem para que el pipeline nunca los encuentre.

    Args:
        dry_run: si True, solo reporta sin borrar nada.

    Returns:
        Lista de rutas eliminadas (o que se eliminarían con dry_run=True).
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    removed = []
    all_imgs = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Escaneando {len(all_imgs)} imágenes con el decodificador de TF...")
    for p in all_imgs:
        try:
            raw = tf.io.read_file(str(p))
            tf.io.decode_image(raw, channels=3, expand_animations=False)
        except Exception:  # noqa: BLE001
            removed.append(p)
            if not dry_run:
                p.unlink()
    action = "Encontradas" if dry_run else "Eliminadas"
    print(f"{action} {len(removed)} imágenes corruptas de {len(all_imgs)} totales.")
    for p in removed:
        print(f"  {'[DRY]' if dry_run else '[DEL]'} {p}")
    return removed


def get_augmentation():
    """Aumentación conservadora: válida anatómicamente para radiografías.

    Solo flip horizontal (izq/der de una extremidad), rotaciones y zoom suaves y
    contraste leve. SIN flip vertical ni deformaciones agresivas.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),   # ~±18°
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def compute_class_weights(data_dir=None, **_):
    """Pesos de clase 'balanced' contando archivos en disco (no lee imágenes)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    counts = {
        i: sum(1 for p in (data_dir / "train" / cls).iterdir()
               if p.suffix.lower() in IMG_EXTS)
        for i, cls in enumerate(config.CLASS_NAMES)
    }
    y = np.array([lbl for lbl, n in counts.items() for _ in range(n)])
    classes = np.array(sorted(counts))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}
