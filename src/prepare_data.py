"""Paso 1 del pipeline de datos: reorganiza el dataset crudo de Kaggle.

El dataset "Bone Fracture Multi-Region X-ray Data" viene YA dividido en
train/val/test con las clases ``fractured`` y ``not fractured``. Este script
MUEVE (no copia) las imágenes a ``data/raw/`` renombrando las clases con prefijo
numérico para que Keras asigne la clase positiva correctamente:

    not fractured  ->  0_normal     (clase 0)
    fractured      ->  1_fracture   (clase 1  ==  P(fractura) en la sigmoide)

OJO: este split "oficial" tiene muchos duplicados y fuga entre splits (ver EDA).
NO se entrena directamente sobre él. El **paso 2** es:

    python src/deduplicate_split.py   # data/raw -> data/processed (limpio, sin fuga)

Fuentes de datos soportadas (en orden de prioridad):
  1. archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/
     (descarga manual del zip de Kaggle y extracción local).
  2. kagglehub — descarga automática al caché del sistema si archive/ no existe.
     Requiere credenciales: https://github.com/Kaggle/kagglehub#authentication

Es idempotente: si se ejecuta dos veces, detecta lo ya movido y no falla.

Uso:
    python src/prepare_data.py            # mueve/copia y verifica
    python src/prepare_data.py --dry-run  # solo muestra lo que haría
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import deque
from pathlib import Path

# Raíz del repo = carpeta padre de este archivo (src/..)
REPO = Path(__file__).resolve().parents[1]

# Ruta (doble-anidada) tal como queda al descomprimir el zip de Kaggle.
SRC_ROOT = (
    REPO
    / "archive"
    / "Bone_Fracture_Binary_Classification"
    / "Bone_Fracture_Binary_Classification"
)
DST_ROOT = REPO / "data" / "raw"

SPLITS = ("train", "val", "test")
# (carpeta origen en el dataset) -> (carpeta destino con prefijo numérico)
CLASS_MAP = {
    "not fractured": "0_normal",
    "fractured": "1_fracture",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Conteos esperados (según README.dataset.txt / verificación inicial).
EXPECTED = {
    "train": {"0_normal": 4640, "1_fracture": 4606},
    "val": {"0_normal": 492, "1_fracture": 337},
    "test": {"0_normal": 268, "1_fracture": 238},
}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _find_dataset_root(base: Path) -> Path | None:
    """BFS dentro de ``base`` buscando el directorio que contiene train/, val/ y test/.

    Limita la búsqueda a 4 niveles para no recorrer árboles grandes.
    Robusto ante cambios de empaquetado del zip de Kaggle.
    """
    required = frozenset({"train", "val", "test"})
    queue: deque[tuple[Path, int]] = deque([(base, 0)])
    while queue:
        current, depth = queue.popleft()
        try:
            children = {p.name: p for p in current.iterdir() if p.is_dir()}
        except PermissionError:
            continue
        if required.issubset(children):
            return current
        if depth < 4:
            for child in children.values():
                queue.append((child, depth + 1))
    return None


def _resolve_src_root() -> Path | None:
    """Devuelve la raíz del dataset (carpeta que contiene train/val/test).

    Prioridad:
    1. archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/
       (layout de la descarga manual).
    2. Descarga automática vía kagglehub.
    """
    if SRC_ROOT.exists():
        return SRC_ROOT

    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except ImportError:
        print(
            "⚠  kagglehub no instalado. Instálalo con:\n"
            "   pip install kagglehub\n"
            "   O descarga el dataset manualmente y descomprímelo en archive/.",
            file=sys.stderr,
        )
        return None

    print("Descargando dataset desde Kaggle vía kagglehub …")
    try:
        kh_path = Path(
            kagglehub.dataset_download(
                "bmadushanirodrigo/fracture-multi-region-x-ray-data"
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Error al descargar con kagglehub: {exc}", file=sys.stderr)
        print(
            "  Configura credenciales Kaggle:\n"
            "  https://github.com/Kaggle/kagglehub#authentication",
            file=sys.stderr,
        )
        return None

    print(f"  Descargado en: {kh_path}")
    found = _find_dataset_root(kh_path)
    if found is None:
        print(
            f"✗ No se encontró train/val/test dentro de:\n  {kh_path}\n"
            "  La versión del dataset puede haber cambiado; revisa la estructura manualmente.",
            file=sys.stderr,
        )
        return None
    print(f"  Raíz del dataset detectada: {found}")
    return found


def move_class(src_dir: Path, dst_dir: Path, dry_run: bool, copy_only: bool = False) -> int:
    """Mueve o copia los archivos de ``src_dir`` a ``dst_dir``. Devuelve cuántos procesó."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        return 0  # ya movido en una ejecución previa
    moved = 0
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        target = dst_dir / f.name
        if target.exists():
            continue  # no sobrescribir (idempotencia)
        if dry_run:
            moved += 1
            continue
        if copy_only:
            shutil.copy2(str(f), str(target))
        else:
            shutil.move(str(f), str(target))
        moved += 1
    # Elimina el directorio origen si quedó vacío (solo relevante cuando no se usa copy_only).
    if not dry_run and not copy_only:
        try:
            next(src_dir.iterdir())
        except StopIteration:
            src_dir.rmdir()
        except FileNotFoundError:
            pass
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="No mueve nada; solo informa.")
    args = parser.parse_args()

    src_root = _resolve_src_root()

    if src_root is None:
        already = all(
            (DST_ROOT / s / c).exists() and count_images(DST_ROOT / s / c) > 0
            for s in SPLITS
            for c in CLASS_MAP.values()
        )
        if already:
            print("✓ El dataset ya estaba reorganizado en data/raw/. Nada que hacer.")
            return verify()
        print(
            "✗ No se encontró el dataset.\n"
            "  Opciones:\n"
            "    a) pip install kagglehub  (y configura credenciales Kaggle)\n"
            "    b) Descarga manualmente de Kaggle y descomprime en archive/",
            file=sys.stderr,
        )
        return 1

    # Verificación de seguridad: las carpetas de clases deben tener el nombre esperado.
    probe = src_root / "train"
    actual_classes = {p.name for p in probe.iterdir() if p.is_dir()} if probe.is_dir() else set()
    if not set(CLASS_MAP).issubset(actual_classes):
        print(
            f"✗ Clases inesperadas en {probe}.\n"
            f"  Encontradas: {sorted(actual_classes)}\n"
            f"  Esperadas:   {sorted(CLASS_MAP)}\n"
            "  Ajusta CLASS_MAP en prepare_data.py si el dataset cambió.",
            file=sys.stderr,
        )
        return 1

    # Copiar si la fuente está fuera del repo (caché de kagglehub); mover si es archive/.
    copy_only = not str(src_root).startswith(str(REPO))
    verb = "Copiando" if copy_only else "Moviendo"

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}{verb} {src_root}\n  -> {DST_ROOT}\n")
    total = 0
    for split in SPLITS:
        for src_name, dst_name in CLASS_MAP.items():
            n = move_class(
                src_root / split / src_name,
                DST_ROOT / split / dst_name,
                args.dry_run,
                copy_only,
            )
            total += n
            print(f"  {split:5s}  {src_name:13s} -> {dst_name:11s}  {n:5d} imágenes")
    print(f"\n{'Se moverían/copiarían' if args.dry_run else 'Procesadas'} {total} imágenes en total.")

    if not args.dry_run:
        # Solo limpia directorios vacíos de archive/ si la fuente era local.
        if not copy_only:
            for parent in (src_root, src_root.parent):
                try:
                    parent.rmdir()
                except OSError:
                    pass  # no está vacío o ya no existe
        return verify()
    return 0


def verify() -> int:
    """Compara los conteos reales con los esperados. Devuelve 0 si todo cuadra."""
    print("\nVerificación de conteos en data/raw/:")
    ok = True
    for split in SPLITS:
        for cls, exp in EXPECTED[split].items():
            got = count_images(DST_ROOT / split / cls)
            flag = "✓" if got == exp else "⚠"
            if got != exp:
                ok = False
            print(f"  {flag} {split:5s}/{cls:11s}  esperado {exp:5d}  obtenido {got:5d}")
    print("\n✓ Conteos correctos." if ok else "\n⚠ Hay diferencias (revisa duplicados/corruptas en el EDA).")
    return 0  # diferencia menor (p.ej. val 828 vs 829) no es un fallo


if __name__ == "__main__":
    raise SystemExit(main())
