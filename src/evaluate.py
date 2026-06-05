"""Evaluación con interpretación clínica: métricas, curvas y ajuste de umbral.

Uso:
    python src/evaluate.py --model-path models/best_mobilenet.keras --target-recall 0.95
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sin pantalla (para guardar figuras)
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
                             f1_score)
from tensorflow import keras

try:
    import config
    import data
except ModuleNotFoundError:  # importado como paquete
    from src import config
    from src import data


def get_true_prob(model, test_ds):
    """y_true (0/1) y y_prob = P(fractura). Requiere test_ds SIN shuffle."""
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)
    y_prob = model.predict(test_ds).ravel()
    return y_true, y_prob


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def threshold_for_recall(y_true, y_prob, target=0.95):
    """Mayor umbral que aún garantiza recall >= target (más sensibilidad)."""
    _, rec, thr = precision_recall_curve(y_true, y_prob)
    idx = np.where(rec[:-1] >= target)[0]
    return float(thr[idx[-1]]) if len(idx) else 0.5


def evaluate(model, test_ds, threshold=config.DEFAULT_THRESHOLD, prefix="model", figs_dir=None):
    """Imprime el reporte, guarda matriz de confusión + ROC y devuelve un dict."""
    figs_dir = Path(figs_dir or config.FIGS_DIR)
    figs_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_prob = get_true_prob(model, test_ds)
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_true, y_pred, target_names=config.LABELS_ES, digits=3))
    auc = roc_auc_score(y_true, y_prob)
    spec = specificity_score(y_true, y_pred)
    print(f"AUC-ROC: {auc:.3f} | Especificidad: {spec:.3f} | Umbral: {threshold:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=config.LABELS_ES).plot(cmap="Blues")
    plt.title(f"Matriz de confusión — {prefix}")
    plt.savefig(figs_dir / f"cm_{prefix}.png", dpi=120, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad / Recall (TPR)")
    plt.title(f"Curva ROC — {prefix}")
    plt.legend(loc="lower right")
    plt.savefig(figs_dir / f"roc_{prefix}.png", dpi=120, bbox_inches="tight")
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,   # sensibilidad
        "specificity": float(spec),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "threshold": float(threshold),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def measure_inference_time(model, n=50, img_size=config.IMG_SIZE):
    """Tiempo medio de inferencia por imagen en CPU (segundos)."""
    x = (np.random.rand(1, *img_size, 3) * 255).astype("float32")
    model.predict(x, verbose=0)  # warmup
    start = time.perf_counter()
    for _ in range(n):
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / n


def model_size_mb(path):
    return Path(path).stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Si se omite, se sugiere uno a partir de --target-recall.")
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    _, _, test_ds, _ = data.make_datasets(args.data_dir)
    model = keras.models.load_model(args.model_path)
    y_true, y_prob = get_true_prob(model, test_ds)

    suggested = threshold_for_recall(y_true, y_prob, args.target_recall)
    print(f"Umbral sugerido para recall>={args.target_recall}: {suggested:.3f}")
    threshold = args.threshold if args.threshold is not None else suggested

    prefix = Path(args.model_path).stem
    metrics = evaluate(model, test_ds, threshold, prefix)
    print("\nMétricas:", metrics)
    print(f"Tiempo inferencia/imagen (CPU): {measure_inference_time(model):.3f} s")
    print(f"Tamaño del modelo: {model_size_mb(args.model_path):.1f} MB")


if __name__ == "__main__":
    main()
