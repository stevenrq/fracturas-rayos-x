"""Paso 1 del pipeline multiclase: descarga y particiona el dataset de fracturas por tipo.

Dataset: "Bone Break Classification" (pkdarabi, Kaggle).
Estructura del dataset descargado: clase/{Train,Test}/imágenes  (pre-dividido por clase).

Salida: data/processed_multiclass/{train,val,test}/<clase>/  (partición 70/15/15)

Fuentes soportadas (en orden de prioridad):
  1. data/bone-break-classification/  (descarga manual previa)
  2. kagglehub — descarga automática al caché del sistema.
     Requiere credenciales: https://github.com/Kaggle/kagglehub#authentication

Es idempotente: si data/processed_multiclass/ ya existe y tiene imágenes, no hace nada.

Uso:
    python src/prepare_data_multiclass.py
"""
from __future__ import annotations

import random
import shutil
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_SRC = REPO / "data" / "bone-break-classification"
DST = REPO / "data" / "processed_multiclass"
KAGGLE_ID = "pkdarabi/bone-break-classification-image-dataset"
IMG_EXTS = frozenset({".jpg", ".jpeg", ".png"})
# Nombres de carpetas que indican split, no clase
SPLIT_NAMES = frozenset({"train", "test", "val", "validation", "training", "testing"})


def _has_images(folder: Path) -> bool:
    try:
        return any(f.suffix.lower() in IMG_EXTS for f in folder.iterdir() if f.is_file())
    except PermissionError:
        return False


def _has_content(folder: Path) -> bool:
    """Verdadero si la carpeta contiene imágenes directamente o en un nivel de subcarpetas."""
    if _has_images(folder):
        return True
    try:
        return any(_has_images(sub) for sub in folder.iterdir() if sub.is_dir())
    except PermissionError:
        return False


def _find_class_root(base: Path) -> Path | None:
    """BFS para encontrar el directorio cuyas subcarpetas inmediatas son clases de fractura.

    Ignora carpetas con nombres de splits (Train/Test/Val) para no confundir una clase
    individual con el directorio raíz. Límite: 4 niveles de profundidad.
    """
    queue: deque[tuple[Path, int]] = deque([(base, 0)])
    while queue:
        current, depth = queue.popleft()
        try:
            children_dirs = [p for p in current.iterdir() if p.is_dir()]
        except PermissionError:
            continue
        if not children_dirs:
            continue
        # Solo candidatos cuyo nombre no sea un split conocido
        non_split = [c for c in children_dirs if c.name.lower() not in SPLIT_NAMES]
        if non_split and any(_has_content(c) for c in non_split[:3]):
            return current
        if depth < 4:
            for child in children_dirs:
                queue.append((child, depth + 1))
    return None


def _collect_images(class_root: Path) -> dict[str, list[Path]]:
    """Devuelve {nombre_clase: [rutas de imágenes]} recorriendo subcarpetas si las hay."""
    result: dict[str, list[Path]] = {}
    for class_dir in sorted(class_root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if images:
            result[class_dir.name] = images
    return result


def _split_and_copy(
    class_images: dict[str, list[Path]], dst: Path, seed: int = 42
) -> None:
    """Copia imágenes a dst/{train,val,test}/<clase>/ con partición 70/15/15 estratificada."""
    rng = random.Random(seed)
    for class_name, images in sorted(class_images.items()):
        imgs = list(images)
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train : n_train + n_val],
            "test":  imgs[n_train + n_val :],
        }
        seen: set[str] = set()
        for split_name, split_imgs in splits.items():
            out_dir = dst / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                name = img.name
                if name in seen:
                    name = f"{img.parent.name}_{name}"
                seen.add(name)
                shutil.copy2(img, out_dir / name)


def _resolve_src() -> Path | None:
    """Devuelve la raíz de clases del dataset (carpeta cuyos hijos son carpetas de clase).

    Prioridad:
    1. data/bone-break-classification/  (descarga manual).
    2. Descarga automática vía kagglehub.
    """
    if LOCAL_SRC.exists() and any(LOCAL_SRC.iterdir()):
        return LOCAL_SRC

    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except ImportError:
        print(
            "⚠  kagglehub no instalado. Instálalo con:\n"
            "   pip install kagglehub",
            file=sys.stderr,
        )
        return None

    print(f"Descargando {KAGGLE_ID} vía kagglehub …")
    try:
        kh_path = Path(kagglehub.dataset_download(KAGGLE_ID))
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Error al descargar: {exc}", file=sys.stderr)
        print(
            "  Configura credenciales Kaggle:\n"
            "  https://github.com/Kaggle/kagglehub#authentication",
            file=sys.stderr,
        )
        return None

    print(f"  Descargado en: {kh_path}")
    found = _find_class_root(kh_path)
    if found is None:
        print(
            f"✗ No se encontraron carpetas de clases en:\n  {kh_path}\n"
            "  La versión del dataset puede haber cambiado; revisa la estructura manualmente.",
            file=sys.stderr,
        )
        return None
    print(f"  Raíz de clases detectada: {found}")
    return found


def main() -> int:
    # Idempotencia: si ya está particionado, no hacer nada
    if DST.exists() and any(DST.rglob("*")):
        n = sum(1 for p in DST.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        print(f"✓ data/processed_multiclass/ ya existe ({n} imágenes). Nada que hacer.")
        return 0

    src = _resolve_src()
    if src is None:
        return 1

    class_images = _collect_images(src)
    clases = sorted(class_images)
    total_imgs = sum(len(v) for v in class_images.values())
    print(f"Clases detectadas ({len(clases)}): {clases}")
    print(f"Total de imágenes: {total_imgs}")

    print(f"Particionando\n  -> {DST}  (70/15/15 estratificado) …")
    _split_and_copy(class_images, DST, seed=42)

    splits = {
        s: sum(1 for p in (DST / s).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        for s in ("train", "val", "test")
    }
    for split, n in splits.items():
        print(f"  {split:5s}: {n} imágenes")
    print(f"  TOTAL: {sum(splits.values())}")
    print("✓ data/processed_multiclass/ listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
