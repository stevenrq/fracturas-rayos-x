"""Deduplica el dataset y re-divide en train/val/test SIN fuga de datos.

Motivación (hallazgo del EDA): el split "oficial" de Kaggle está en ``data/raw/``
y contiene ~63% de imágenes duplicadas y 873 grupos de duplicados que se filtran
entre train/val/test. Evaluar sobre él da métricas infladas y no creíbles.

Este script construye un dataset limpio en ``data/processed/``:

1. Junta TODAS las imágenes de ``data/raw`` (de los tres splits y dos clases).
2. **Deduplica exactos** por MD5 (se queda con una sola copia de cada archivo).
3. **Agrupa por imagen-fuente**: variantes aumentadas del mismo origen
   (``16.jpg``, ``16-rotated1.jpg``, ``16 (1).jpg`` …) comparten "group key", de
   modo que nunca caen en splits distintos (evita fuga de *casi-duplicados*).
4. **Re-divide estratificado** por clase (70/15/15, semilla fija) asignando
   grupos completos a cada split.
5. Copia el resultado a ``data/processed/{train,val,test}/{0_normal,1_fracture}``.

Es reproducible (semilla) e idempotente (limpia ``data/processed`` antes de poblar).

Uso:
    python src/deduplicate_split.py
    python src/deduplicate_split.py --ratios 0.7 0.15 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
DST = REPO / "data" / "processed"
CLASSES = ("0_normal", "1_fracture")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Sufijos de aumentación a eliminar para agrupar variantes de una misma fuente.
_ROTATED = re.compile(r"(-rotated\d+)+", re.IGNORECASE)
_COPY = re.compile(r"\s*\(\d+\)\s*$")  # " (1)", " (2)" ...


def _unique_path(path: Path) -> Path:
    """Evita sobrescribir si dos imágenes distintas comparten nombre en el destino."""
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.with_name(f"{path.stem}__{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def group_key(filename: str) -> str:
    """Clave de la imagen-fuente: quita extensión, '-rotatedN' y ' (N)'."""
    stem = filename.rsplit(".", 1)[0]
    stem = _COPY.sub("", stem)
    stem = _ROTATED.sub("", stem)
    stem = _COPY.sub("", stem)
    return stem.strip().lower()


def collect_unique() -> dict[str, list[tuple[str, Path]]]:
    """Devuelve {clase: [(group_key, ruta)]} ya deduplicado por MD5."""
    seen_md5: set[str] = set()
    per_class: dict[str, list[tuple[str, Path]]] = {c: [] for c in CLASSES}
    for cls in CLASSES:
        for split_dir in sorted(RAW.iterdir()):
            cdir = split_dir / cls
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                md5 = hashlib.md5(p.read_bytes()).hexdigest()
                if md5 in seen_md5:
                    continue  # duplicado exacto -> descartar
                seen_md5.add(md5)
                per_class[cls].append((group_key(p.name), p))
    return per_class


def split_groups(files: list[tuple[str, Path]], ratios, rng) -> dict[str, list[Path]]:
    """Reparte grupos completos en train/val/test respetando ~ratios por imagen."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for gk, path in files:
        groups[gk].append(path)
    keys = list(groups)
    rng.shuffle(keys)

    total = len(files)
    target_train = ratios[0] * total
    target_val = (ratios[0] + ratios[1]) * total
    out = {"train": [], "val": [], "test": []}
    acc = 0
    for gk in keys:
        paths = groups[gk]
        if acc < target_train:
            dest = "train"
        elif acc < target_val:
            dest = "val"
        else:
            dest = "test"
        out[dest].extend(paths)
        acc += len(paths)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RAW.exists() or not any(RAW.iterdir()):
        raise SystemExit(f"✗ No hay datos en {RAW}. Ejecuta antes src/prepare_data.py.")

    rng = random.Random(args.seed)
    per_class = collect_unique()
    print("Imágenes únicas (tras dedup exacto):")
    for c in CLASSES:
        print(f"   {c:11s}: {len(per_class[c])}")

    # Limpia destino
    for split in ("train", "val", "test"):
        for c in CLASSES:
            d = DST / split / c
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # Reparte cada clase y copia
    placement = {c: split_groups(per_class[c], args.ratios, rng) for c in CLASSES}
    counts = {s: {c: 0 for c in CLASSES} for s in ("train", "val", "test")}
    for c in CLASSES:
        for split, paths in placement[c].items():
            for p in paths:
                dest = _unique_path(DST / split / c / p.name)
                shutil.copy2(p, dest)
                counts[split][c] += 1

    print("\nDataset limpio en data/processed/ (70/15/15 estratificado, sin fuga):")
    for s in ("train", "val", "test"):
        row = "  ".join(f"{c}={counts[s][c]}" for c in CLASSES)
        print(f"   {s:5s}: {row}  (total {sum(counts[s].values())})")
    print(f"   TOTAL: {sum(sum(v.values()) for v in counts.values())}")

    verify()
    return 0


def verify() -> None:
    """Comprueba que ningún MD5 aparece en más de un split (cero fuga)."""
    split_of: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for c in CLASSES:
            for p in (DST / split / c).iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    md5 = hashlib.md5(p.read_bytes()).hexdigest()
                    split_of.setdefault(md5, set()).add(split)
    leaks = sum(1 for s in split_of.values() if len(s) > 1)
    print(f"\nVerificación de fuga: {leaks} hashes en >1 split  "
          + ("✓ (sin fuga)" if leaks == 0 else "⚠ REVISAR"))


if __name__ == "__main__":
    raise SystemExit(main())
