"""Constantes y rutas compartidas por los módulos de ``src/``.

Importable tanto al ejecutar scripts dentro de ``src/`` (``import config``) como
desde la raíz del repo (``from src import config``).
"""
from pathlib import Path

# Raíz del repositorio = carpeta padre de src/
REPO = Path(__file__).resolve().parents[1]

# --- Rutas ---
DATA_DIR = REPO / "data" / "processed"     # dataset LIMPIO (dedup + sin fuga)
RAW_DIR = REPO / "data" / "raw"            # split oficial (referencia/EDA)
MODELS_DIR = REPO / "models"
REPORTS_DIR = REPO / "reports"
FIGS_DIR = REPORTS_DIR / "figures"

# --- Hiperparámetros base de datos/imagen ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# --- Clases ---  (el prefijo numérico garantiza fractura = clase 1 = positiva)
CLASS_NAMES = ["0_normal", "1_fracture"]
LABELS_ES = ["normal", "fractura"]

# Umbral de decisión por defecto. Se reajusta clínicamente en evaluate.py
# (ver reports/resultados.md). Sigmoide -> P(fractura) >= UMBRAL  => "FRACTURA".
DEFAULT_THRESHOLD = 0.5

# Crea carpetas de salida si no existen (no afecta a los datos).
for _d in (MODELS_DIR, FIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
