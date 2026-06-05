"""Entrenamiento en dos etapas: feature-extraction -> fine-tuning.

Uso (local con venv 3.11/3.12, o en Colab):
    python src/train.py --model mobilenet
    python src/train.py --model efficientnet --epochs-head 12 --epochs-ft 8
    python src/train.py --model cnn --epochs-head 25 --epochs-ft 0
"""
from __future__ import annotations

import argparse

from tensorflow import keras

try:
    import config
    import data
    import model as M
except ModuleNotFoundError:  # importado como paquete
    from src import config
    from src import data
    from src import model as M


def make_callbacks(ckpt_path, monitor="val_auc", mode="max"):
    return [
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=5,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor=monitor, mode=mode,
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]


def train(model_name="mobilenet", data_dir=None, epochs_head=15, epochs_ft=10,
          fine_tune_at=100, dropout=0.3):
    """Entrena un modelo y guarda el mejor en models/best_<model_name>.keras."""
    train_ds, val_ds, test_ds, class_names = data.make_datasets(data_dir)
    print("Orden de clases:", class_names, "(fractura = 1)")

    class_weight = data.compute_class_weights(data_dir)
    print("class_weight:", class_weight)

    net, base = M.BUILDERS[model_name](dropout=dropout, base_trainable=False)
    net = M.compile_model(net, lr=1e-3)

    ckpt = config.MODELS_DIR / f"best_{model_name}.keras"
    callbacks = make_callbacks(ckpt)

    # steps_per_epoch desde filesystem: evita iterar el dataset (lo cual
    # activaría el cache prematuramente y añadiría latencia innecesaria).
    img_exts = {".jpg", ".jpeg", ".png"}
    n_train = sum(
        1 for cls in config.CLASS_NAMES
        for p in (config.DATA_DIR / "train" / cls).iterdir()
        if p.suffix.lower() in img_exts
    )
    steps = n_train // config.BATCH_SIZE

    print(f"\n=== Etapa 1: feature extraction ({model_name}) ===")
    history1 = net.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
                       steps_per_epoch=steps,
                       class_weight=class_weight, callbacks=callbacks)
    history2 = None

    if base is not None and epochs_ft > 0:
        print(f"\n=== Etapa 2: fine-tuning (descongela desde la capa {fine_tune_at}) ===")
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
        net = M.compile_model(net, lr=1e-5)
        history2 = net.fit(train_ds, validation_data=val_ds, epochs=epochs_ft,
                           steps_per_epoch=steps,
                           class_weight=class_weight, callbacks=callbacks)

    print(f"\n✓ Modelo guardado en {ckpt}")
    return net, ckpt, (history1, history2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(M.BUILDERS), default="mobilenet")
    ap.add_argument("--data-dir", default=None, help="Por defecto data/processed")
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--fine-tune-at", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    train(args.model, args.data_dir, args.epochs_head, args.epochs_ft,
          args.fine_tune_at, args.dropout)


if __name__ == "__main__":
    main()
