# 🦴 Detección y clasificación de fracturas óseas mediante rayos X

Proyecto final de **Aprendizaje Computacional** · Ingeniería de Sistemas ·
Universidad de Córdoba. Sistema de **visión por computadora** que clasifica una
radiografía como **fractura / normal** con *transfer learning*, evaluado con
criterio clínico y desplegado como demo interactiva.

> ⚠️ **Aviso médico.** Herramienta **académica y educativa**. No es un dispositivo
> médico ni está validada clínicamente. Sus predicciones son orientativas, pueden
> contener errores y **no sustituyen el diagnóstico de un profesional de la salud**.

- 🔗 **Demo (Hugging Face Space):** [stevenrq8/fracturas-rayos-x](https://huggingface.co/spaces/stevenrq8/fracturas-rayos-x)
- 📊 **Resultados y reflexión crítica:** [`reports/resultados.md`](reports/resultados.md).

---

## 1. Definición del problema

- **Tarea:** clasificación binaria de radiografías → *fractura* (clase 1) / *normal* (clase 0).
- **Contexto de uso:** apoyo a *triage* en urgencias, hospitales rurales y telemedicina.
- **Requisitos (metas medibles):**
  - Sensibilidad (**recall**) de la clase fractura **≥ 0.90** (métrica prioritaria).
  - Tiempo de inferencia **< 2 s** por imagen en CPU.
  - Uso simple: subir imagen → predicción + nivel de confianza.
  - Sin GPU dedicada en despliegue → modelo ligero (MobileNetV2).
- **Justificación clínica:** un **falso negativo** (fractura no detectada) es más
  grave que un falso positivo. Por eso **recall pesa más que accuracy**, y el
  umbral de decisión se ajusta para priorizar la sensibilidad.

## 2. Datos

**Dataset:** *Bone Fracture Multi-Region X-ray Data* (Kaggle, autor
`bmadushanirodrigo`, licencia **ODC-By v1.0**).

El EDA ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)) detectó que el split
"oficial" tenía **63% de imágenes duplicadas** (10 581 → **3 888 únicas**) y
**873 grupos con fuga entre train/val/test**, que inflarían las métricas. Por eso
los datos se **deduplican y se re-dividen sin fuga**:

```
archive/ (zip de Kaggle)
  └─ src/prepare_data.py        →  data/raw/        split oficial (renombrado, con duplicados)
        └─ src/deduplicate_split.py → data/processed/   split LIMPIO: 70/15/15, dedup, SIN fuga
```

Dataset limpio (lo que se entrena): **train 2738 · val 587 · test 563**
(≈59% normal / 41% fractura, tratado con `class_weight`).

> Los datos y los modelos **no** se versionan en git (van en `.gitignore`).

## 3. Estructura del repositorio

```
fracturas-rayos-x/
├── README.md  · LICENSE · requirements.txt · .gitignore
├── data/        raw/ (oficial)  ·  processed/ (limpio)        [ignorados por git]
├── notebooks/   01_eda · 02_modelado · 03_multiclase · 04_medgemma_poc
├── src/         prepare_data · deduplicate_split · config · data · model · train · evaluate · gradcam
├── models/      best_*.keras                                  [ignorados por git]
├── app/         app_gradio.py · app_streamlit.py · ejemplos/
├── space/       app.py · requirements.txt · README.md (para Hugging Face Spaces)
└── reports/     figures/ · resultados.md · metrics_comparacion.csv
```

## 4. Instalación (local)

> **Importante:** TensorFlow **no** tiene wheels para **Python 3.14**. Usa
> **Python 3.11 o 3.12** para el entorno local.

```bash
# Conseguir Python 3.12 en Ubuntu (una opción):
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install python3.12 python3.12-venv
# (alternativas: pyenv, conda o uv)

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Cómo ejecutar

```bash
# 1) Preparar datos (una vez): descomprime el dataset en archive/ y luego
python src/prepare_data.py          # archive/ -> data/raw/
python src/deduplicate_split.py     # data/raw/ -> data/processed/ (limpio, sin fuga)

# 2) EDA
jupyter notebook notebooks/01_eda.ipynb

# 3) Entrenamiento  (recomendado en Google Colab con GPU: notebooks/02_modelado.ipynb)
python src/train.py --model mobilenet
python src/evaluate.py --model-path models/best_mobilenet.keras --target-recall 0.95

# 4) Apps (necesitan models/best_mobilenet.keras entrenado)
python app/app_gradio.py            # http://127.0.0.1:7860
streamlit run app/app_streamlit.py
```

El **entrenamiento** se hace en **Colab** (GPU T4 gratis); descargas el `.keras` y
lo usas en local / en el Space. Las apps leen el umbral de `UMBRAL` (ver
`reports/resultados.md`); puedes sobreescribirlo con la variable de entorno
`UMBRAL`.

## 6. Modelos y resultados

Tres modelos comparados: **CNN desde cero**, **MobileNetV2** (desplegado) y
**EfficientNetB0**. Métrica norte: **recall (fractura)** y **AUC**. Tabla
comparativa, matrices de confusión, curvas ROC y reflexión crítica en
[`reports/resultados.md`](reports/resultados.md).

## 7. Despliegue

App **Gradio** en **Hugging Face Spaces** (CPU gratis). Carpeta lista en
[`space/`](space/); el modelo se carga desde un *Model repo* de HF
(`hf_hub_download`). Ver instrucciones en [`space/README.md`](space/README.md).

## 8. Extensiones incluidas

- **Grad-CAM** — mapa de calor de la zona observada (en las apps y el Space).
- **CNN desde cero** — 3er baseline para medir el aporte del transfer learning.
- **Multiclase** (tipo de fractura) — [`notebooks/03_multiclase.ipynb`](notebooks/03_multiclase.ipynb) (requiere otro dataset).
- **MedGemma** — comentario textual del hallazgo como prueba de concepto en Colab
  ([`notebooks/04_medgemma_poc.ipynb`](notebooks/04_medgemma_poc.ipynb)); **no** se
  despliega en el Space.

## 9. Limitaciones

- Demo académica, **no** validada clínicamente ni apta para uso real.
- Dataset modesto tras deduplicar (3 888 imágenes), de origen heterogéneo y muy
  aumentado; mezcla regiones anatómicas sin etiqueta de región (posible sesgo).
- Una sola vista por caso, sin contexto clínico del paciente, sin validación por radiólogos.

## 10. Licencia y créditos

- Código: **MIT** (ver [`LICENSE`](LICENSE)).
- Datos: *Bone Fracture Multi-Region X-ray Data*, licencia **ODC-By v1.0** (atribución al autor del dataset).
- Autores del proyecto: **Steven Ricardo Quiñones** y **Amaury Díaz Betín** · Prof. Oswaldo Vélez Langs, PhD.
