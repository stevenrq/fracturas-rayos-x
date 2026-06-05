# Resultados del modelado — Detección de fracturas óseas

> Fase 2 (Modelado + Evaluación). Entrenamiento realizado en Google Colab (GPU T4).

## 1 · Datos utilizados

Dataset **limpio** (`data/processed/`), sin duplicados ni fuga entre splits:

| split | normal | fractura | total |
|---|---:|---:|---:|
| train | 1611 | 1127 | 2738 |
| val   |  343 |  244 |  587 |
| test  |  328 |  235 |  563 |
| **total** | **2282** | **1606** | **3888** |

Balance ≈ **59% normal / 41% fractura** → `class_weight: {0: 0.85, 1: 1.21}`.

> El split oficial de Kaggle tenía **63% de duplicados y 873 grupos con fuga** entre
> train/val/test (detectados en el EDA). Usar ese split habría inflado las métricas
> de forma no creíble — por eso se deduplicó y re-dividió.

## 2 · Tabla comparativa (test, umbral 0.5)

> **Métrica norte: Recall (fractura) y AUC.** Accuracy es secundaria.

| Modelo | Accuracy | Recall (fractura) | Especificidad | Precision | F1 | AUC | FN | FP | Inf. CPU | Tamaño |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN desde cero | 0.673 | **0.243** ❌ | **0.982** | 0.905 | 0.383 | 0.820 | 178 | 6 | 0.065 s | 1.2 MB |
| **MobileNetV2** | **0.815** | **0.796** | 0.829 | 0.770 | **0.782** | 0.910 | 48 | 56 | 0.070 s | 9.7 MB |
| EfficientNetB0 | 0.798 | 0.681 | **0.881** | **0.804** | 0.737 | **0.919** | 75 | 39 | 0.076 s | 43.4 MB |

## 3 · Umbral de decisión (ajuste clínico)

A umbral 0.5 ningún modelo alcanza el recall objetivo (≥ 0.90). Bajando el umbral
de MobileNetV2:

| Objetivo recall | Umbral | Recall | Especificidad | Precision | F1 | AUC |
|---|---|---|---|---|---|---|
| ≥ 0.90 | 0.387 | 0.900+ | — | — | — | 0.910 |
| **≥ 0.95** | **0.310** | **0.953** ✅ | 0.649 | 0.661 | 0.781 | 0.910 |
| ≥ 0.98 | 0.223 | 0.980+ | ~0.55 | ~0.61 | ~0.75 | 0.910 |

**Umbral elegido: 0.310** (recall 0.953, especificidad 0.649).

**Justificación clínica:** en triage el coste de un falso negativo (fractura no
detectada = paciente sin atención) supera el de un falso positivo (estudio extra
innecesario). Por eso se prioriza la sensibilidad aunque baje la especificidad.
Con este umbral quedan ~11 FN y ~115 FP sobre 563 casos de test.

## 4 · Modelo elegido para despliegue

**MobileNetV2** con umbral 0.310:
- Cumple el objetivo clínico (recall 0.953 con umbral ajustado ✅)
- Mejor recall a umbral 0.5 (0.796) y mejor accuracy (0.815) de los tres ✅
- Inferencia 0.070 s < 2 s en CPU ✅
- Peso 9.7 MB → cabe en el Space gratuito de HF ✅
- EfficientNetB0 tiene AUC marginalmente mejor (0.919 vs 0.910), pero peor recall
  a umbral 0.5 (0.681 vs 0.796) y pesa 43.4 MB — no es viable para
  despliegue en CPU free.

## 5 · Reflexión crítica

**¿Generaliza o memoriza?**
MobileNetV2: train AUC ~0.96 vs val AUC ~0.84 → brecha de ~0.12 puntos.
Sobreajuste moderado. Las técnicas aplicadas (dropout 0.3, augmentation,
EarlyStopping, ReduceLROnPlateau) lo mitigan pero no eliminan, dado el tamaño
del dataset (solo 2738 imágenes de entrenamiento tras la limpieza).

**¿Hay sobreajuste?**
Sí, moderado en los tres modelos. En EfficientNetB0 la etapa 2 de fine-tuning
lleva el train AUC a ~0.983 mientras val se estanca en ~0.86. La CNN desde cero
mostró val_auc errático (0.63–0.85 en épocas consecutivas) y un recall muy bajo
en test (0.243 a umbral 0.5): predice casi todo como "normal" (especificidad 0.982),
lo que confirma que sin transfer learning el modelo no aprende a detectar la fractura.

**¿Está balanceado el dataset?**
Desbalance leve (59/41) tratado con `class_weight`. No es la causa principal
de los bajos recalls — esos se corrigen bajando el umbral de decisión.

**¿Qué tan confiable clínicamente?**
Como **demo académica**, no para uso real. Limitaciones concretas:
- Solo 3888 imágenes únicas (dataset pequeño tras deduplicar el oficial).
- El dataset mezcla regiones anatómicas (muñeca, hombro, rodilla…) sin etiqueta:
  el modelo podría aprender la región y no la fractura.
- Una sola vista por caso, sin datos del paciente (edad, historial, síntomas).
- Sin validación por radiólogos.
- El val_auc errático de la CNN indica que las métricas de val deben
  interpretarse con cautela.

**Limitaciones del estudio:**
- Dataset de origen heterogéneo y muy pre-aumentado (imágenes con nombres como
  `imagen-rotated1-rotated2.jpg`), lo que reduce la variabilidad real.
- Métricas sobre una sola partición (train/val/test fija). Para resultados más
  robustos convendría validación cruzada.
- No se evaluó el tiempo de carga del modelo (cold start), relevante en el Space.

## 6 · Mejoras aplicadas tras el primer entrenamiento

- `data.py`: `ignore_errors()` solo en train; val/test determinísticos.
- `train.py`: `steps_per_epoch` explícito → elimina warning "ran out of data".
- Umbral ajustado de 0.5 → 0.310 para alcanzar recall clínico (≥ 0.95).

## 7 · Figuras generadas

- `reports/figures/cm_best_cnn.png` — matriz de confusión CNN
- `reports/figures/cm_best_mobilenet.png` — matriz de confusión MobileNetV2
- `reports/figures/cm_best_efficientnet.png` — matriz de confusión EfficientNetB0
- `reports/figures/cm_mobilenet_umbral.png` — MobileNetV2 con umbral 0.310
- `reports/figures/roc_*.png` — curvas ROC
- `reports/figures/curvas_*.png` — pérdida/AUC por época

> ⚠️ Herramienta educativa. No es un dispositivo médico ni sustituye el criterio
> de un profesional de la salud.
